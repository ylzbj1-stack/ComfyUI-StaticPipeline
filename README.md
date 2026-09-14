# ComfyUI-StaticPipeline

English | [中文说明](README_CN.md)

**Static pipeline split for multi-GPU ComfyUI: place DiT blocks across GPUs once, never move weights again.**

A drop-in custom node that splits a large DiT across multiple GPUs **statically** — weights are placed at load time and never move during inference. Includes hybrid residency (resident + CPU-streamed blocks) for long sequences, activation-space LoRA for quantized models, and chunked compute for long-video workloads.

Built and battle-tested on **MiniMax H3 video DiT (34 GB int8_convrot) across 2× RTX 3080 20 GB**, generating up to 15-second single-segment videos with **zero crashes**.

---

## Why: the problem with dynamic layer sharding

The popular dynamic-sharding approach ([ComfyUI-MultiGPU](https://github.com/pollockjj/ComfyUI-MultiGPU) DisTorch) streams weight blocks between devices on demand. On recent ComfyUI forks with DynamicVRAM/aimdo virtual-VRAM internals **combined with packed quantized weights** (int8_convrot etc.), this is fundamentally unstable — we reproduced deterministic `access violation` crashes from three different paths:

1. **VAE eviction crash** — loading the video VAE evicts the DiT → `unpatch_model` touches dangling quantized `qdata` → segfault
2. **Reload crash** — a second prompt re-partially-loads the model → `patch_weight_to_device` touches freed vbar memory → segfault
3. **Block-move crash** — even a *fresh load* crashes when the block-moving dance (dequant→requant through comfy_kitchen) hits a bad page

Our record with dynamic sharding on this stack: **3 successes, 4 hard crashes** — every crash left a faulthandler trace pointing at the same root: dangling pointers into the quantized-weight virtual memory after any move.

The killer detail: `vbar_free_memory` in the aimdo ledger started returning **negative garbage values**. Python-level guards can't fix a corrupted kernel-side ledger.

## The fix: stop moving weights. Ever.

**Static placement**: split the 50 transformer blocks once at load time (e.g. 23 + 27 across two 20 GB cards), then treat the DiT as load-bearing furniture:

- `partially_load` → no-op (patches/counters only)
- `partially_unload` → report freed (lie), so nothing ever evicts it
- `unpatch_model` → forbidden from moving weights
- `free_memory` guard → static models are never eviction candidates
- Activations cross the PCIe bus **once per step** at the block boundary (~140 MB)

### Hybrid residency for long sequences

Weights don't grow with video length — activations do (~linear in tokens). For longer clips the node automatically switches profiles:

| Profile | Frames | Layout | Measured (segment 2+ of a chained run) |
|---|---|---|---|
| `short` | ≤88f | all-resident 23/27 | 73f: **6.80 s/it** (0.093 s/it·frame, zero streamed blocks) |
| `mid` | 110–259f | 22/23 resident + 5 CPU-streamed | 124f: **~27 s/it** · 243f: **~62 s/it** |
| `long` | ≥260f | 19/21 resident + 9 CPU-streamed | 277f: **~78** · 311f: **~93** · 362f: **~107 s/it** |

Measured cost curve on a directed chained run (steps=8 + turbo, 960×544, segment 2+):
124f ≈ 27 s/it · 243f ≈ 62 · 277f ≈ 78 · 311f ≈ 93 · 362f ≈ 107. Segment 1 carries a one-off
~140–240 s cost (Triton compile + first LoRA move + static placement).

Two knobs matter for long clips: **22 hot blocks stay pinned resident per card** (the rest can
stream), and the loaded model survives across prompts (LRU model cache) — the second and later
submissions on the same instance skip reload, Triton recompile and the first LoRA move, saving
**~5.7 min per clip** at the same tier.

(2× RTX 3080 20 GB, MiniMax H3 34 GB int8, 8-step turbo. CPU-streamed blocks are lossless packed-int8 copies; overhead ~1 s/step.)

![benchmark](assets/benchmark.png)

### Four levels of chunking for long sequences

- MLP swiglu: 16k-row chunks (peak transient 2.9 GB → ~250 MB)
- LoRA delta: 1k-row chunks, **in-place `add_` into the layer output** (zero full-size delta allocation)
- Attention qkv/out_proj projections: 4k-row chunks into preallocated buffers
- Attention core: **query chunking** (k,v stay full — that's the math) 

### Activation-space LoRA (bypass mode)

Instead of `weight_function` hooks (which dequantize every layer every step: +5 s/it tax at 208 layers), LoRA is applied in activation space: `y = W·x + α·B(A·x)`. Weights stay packed; the int8 fast path is preserved; the comfy_kitchen requant minefield is never touched. Uses the bypass infrastructure already shipped in ComfyUI's `weight_adapter/bypass.py`, with three fixes: per-card adapter placement, activation-dtype lazy casting (blind fp32 pre-cast = 448 MB transients), and chunked in-place accumulation.

### Bugs found & fixed along the way (highlights)

- **Activation-space LoRA cache thrash**: the adapter cache was single-slot; on long sequences (`x.device` flipping cuda:0 ↔ cuda:1 within one step) it missed on every call and re-copied the whole ~1 GB adapter set — a single step took **>21 minutes** (~8.8 TB moved). Fixed by bucketing the cache per `(dtype, device)` with ≥2 entries per adapter. Tell-tale signature: **PCIe device-1 TX at ~7.8 GB/s** (GPU→host flooding). Short clips never trigger it — the key never flips.
- `adaln_t_table` was `cast_to`'d **every forward call** — 9 calls × 496 MB of duplicate tables alive simultaneously (4.46 GB!) on a 360-frame job. Fixed with an instance-level cache.
- comfy_kitchen's cublas workspace is bound to `torch.cuda.current_device()` — kernels on cuda:1 got cuda:0's workspace → `cudaErrorIllegalAddress`. Fixed by running every block under `torch.cuda.device(its_home)`.
- ComfyUI-MultiGPU's dlpack guard loads `libcudart.so` — **instant crash on Windows**. Patch included (loads `cudart64_13.dll` from torch/lib).
- Streamed blocks originally did an H2D hop **and** a D2H hop per forward. The D2H hop was the access-violation site (it reads GPU pages that aimdo may have remapped). Since weights are read-only during forward, the D2H copy is now eliminated entirely — the saved CPU parameter objects are swapped back by reference (one-way transfer).
- TE encoding leaves a fragmented dead-block pool; sampling then OOMs at 16.5 GB allocated with 3 GB "free" per the ceiling. `empty_cache()` at load completion fixes it.
- WDDM effective VRAM ceiling is ~19.0 GiB on 20 GB cards; `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is mandatory.

Full gory details: `patches/` and the debug methodology (`MGPU_STATIC_MEM_DEBUG=1` gives per-block VRAM curves + top-tensor attribution).

---

## Installation

```
cd ComfyUI/custom_nodes
git clone https://github.com/<YOU>/ComfyUI-StaticPipeline.git
```

Optional but recommended patches (see `patches/`):
- `comfy-core_minimax-model_adaln-cache-embed-free.patch` — kills the 496 MB/call table duplication + frees embed intermediates (benefits single-GPU too)
- `comfy-core_weight-adapter-bypass_inplace-support.patch` — in-place support required by chunked LoRA
- `multigpu_p2p-registry_windows-cudart.patch` — Windows fix for ComfyUI-MultiGPU (only if you also run that pack)

## Usage

Insert the node after your DiT loader:

```
UNETLoader → StaticPipelineSplit(frames=<your frame count>) → ... → sampler
```

- **`frames` must be the length of a single sampling pass** (max segment length for multi-segment Director chains, not the chain total) — it selects the residency profile (<110 all-resident / 110–259 mid / ≥260 long). Passing a multi-segment chain total over-streams the model and multiplies the kitchen `.to()` dispatch count per step, which is the same crash-class this node exists to avoid.
- Launch with:
  ```
  MGPU_CPU_THRESHOLD_PERCENT=999 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  ```
- **Restart the instance when switching frame-count tiers** — placement is one-shot per loaded model
- The video VAE (7.9 GB) won't fit alongside; it runs lowvram-streamed. That's expected.

## Notes & limits

- Requires the ComfyUI fork runtime this was developed against (DynamicVRAM/aimdo builds; tested on torch 2.10.0+cu130). Stock ComfyUI works too — the overrides simply no-op gracefully — but the target use case is packed-quantized DiTs that crash under dynamic sharding.
- **Measured segment tiers** (current dev0 budget + 22 pinned blocks): chained segments passed every tier we tested — **124 / 243 / 277 / 311 / 362 frames**, validated end-to-end. Notes quoting a 226f single-shot / 124f chained ceiling came from the pre-pin dev0 budget and no longer apply. A chained segment still costs ~384 MiB more than the same length as a standalone shot — "the standalone probe passed" does not imply the chain will.
- SageAttention allocates a **513.97 MiB fp32 temporary over the full k** (∝ token count, hence ∝ frames) — the largest single OOM trigger on long segments; intrinsic, unrelated to LoRA/streaming/VAE.
- Three ways to free VRAM that we measured and **killed**: (a) rebalancing weights across cards — streamed blocks hop back onto their home card during their own forward, so the peak is conserved; (b) moving the audio VAE to cuda:1 — aimdo's vbar is only registered on device 0; (c) moving the text encoder to GPU — CUDA-context-level crash.
- Keep the VAEs on the GPU: giving the video VAE ~3 GiB of headroom cut decode from 137 s → 46 s.
- Windows + WDDM: effective ceiling ≈ 19 GiB per 20 GB card. Linux may allow more headroom (untested).

## Support

If this saved your quantized multi-GPU stack from segfaulting, a tip is appreciated:

**USDT (TRC20):** `TXLHM7dayYa7qHzHXWSqfrDhwZfLRT69oT`

<p align="left"><img src="assets/donate_qr.png" width="240" alt="USDT TRC20 donation QR"></p>

## Acknowledgments

- [pollockjj/ComfyUI-MultiGPU](https://github.com/pollockjj/ComfyUI-MultiGPU) — DisTorch; the dynamic approach this project replaces for quantized stacks (and whose `p2p_registry` Windows bug we patch here)
- [Fannovel16/ComfyUI-Frame-Interpolation](https://github.com/Fannovel16/ComfyUI-Frame-Interpolation) — RIFE arch used in our companion interpolation script
- [Comfy-Org](https://huggingface.co/Comfy-Org) — repacked RIFE weights
- MiniMax — H3 video model
- comfy_kitchen / aimdo authors — for the (buggy but fascinating) virtual-VRAM stack that motivated all this

## License

MIT
