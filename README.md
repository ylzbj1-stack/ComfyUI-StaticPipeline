# ComfyUI-StaticPipeline

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

| Profile | Frames | Layout | Measured |
|---|---|---|---|
| `short` | ≤88f | all-resident 23/27 | **10.91 s/it**, 8 steps, end-to-end 9.5 min |
| `mid` | 110–259f | 22/23 resident + 5 CPU-streamed | **17.2 s/it**, end-to-end 10 min |
| `long` | ≥260f | 19/21 resident + 9 CPU-streamed | **91.97 s/it**, 24 min for a 15 s clip |

(2× RTX 3080 20 GB, MiniMax H3 34 GB int8, 8-step turbo. CPU-streamed blocks are lossless packed-int8 copies; overhead ~1 s/step.)

### Four levels of chunking for long sequences

- MLP swiglu: 16k-row chunks (peak transient 2.9 GB → ~250 MB)
- LoRA delta: 1k-row chunks, **in-place `add_` into the layer output** (zero full-size delta allocation)
- Attention qkv/out_proj projections: 4k-row chunks into preallocated buffers
- Attention core: **query chunking** (k,v stay full — that's the math) 

### Activation-space LoRA (bypass mode)

Instead of `weight_function` hooks (which dequantize every layer every step: +5 s/it tax at 208 layers), LoRA is applied in activation space: `y = W·x + α·B(A·x)`. Weights stay packed; the int8 fast path is preserved; the comfy_kitchen requant minefield is never touched. Uses the bypass infrastructure already shipped in ComfyUI's `weight_adapter/bypass.py`, with three fixes: per-card adapter placement, activation-dtype lazy casting (blind fp32 pre-cast = 448 MB transients), and chunked in-place accumulation.

### Bugs found & fixed along the way (highlights)

- `adaln_t_table` was `cast_to`'d **every forward call** — 9 calls × 496 MB of duplicate tables alive simultaneously (4.46 GB!) on a 360-frame job. Found via gc tensor attribution; fixed with an instance-level cache.
- comfy_kitchen's cublas workspace is bound to `torch.cuda.current_device()` — kernels on cuda:1 got cuda:0's workspace → `cudaErrorIllegalAddress`. Fixed by running every block under `torch.cuda.device(its_home)`.
- ComfyUI-MultiGPU's dlpack guard loads `libcudart.so` — **instant crash on Windows**. Patch included (loads `cudart64_13.dll` from torch/lib).
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

- **`frames` must be your actual frame count** — it selects the residency profile (<110 all-resident / 110–259 mid / ≥260 long)
- Launch with:
  ```
  MGPU_CPU_THRESHOLD_PERCENT=999 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
  ```
- **Restart the instance when switching frame-count tiers** — placement is one-shot per loaded model
- The video VAE (7.9 GB) won't fit alongside; it runs lowvram-streamed. That's expected.

## Notes & limits

- Requires the ComfyUI fork runtime this was developed against (DynamicVRAM/aimdo builds; tested on torch 2.10.0+cu130). Stock ComfyUI works too — the overrides simply no-op gracefully — but the target use case is packed-quantized DiTs that crash under dynamic sharding.
- >30 s segments: use segment chaining (Director-style), not longer single shots — attention k,v must stay fully resident, so VRAM grows linearly with duration and compute grows quadratically.
- Windows + WDDM: effective ceiling ≈ 19 GiB per 20 GB card. Linux may allow more headroom (untested).

## Acknowledgments

- [pollockjj/ComfyUI-MultiGPU](https://github.com/pollockjj/ComfyUI-MultiGPU) — DisTorch; the dynamic approach this project replaces for quantized stacks (and whose `p2p_registry` Windows bug we patch here)
- [Fannovel16/ComfyUI-Frame-Interpolation](https://github.com/Fannovel16/ComfyUI-Frame-Interpolation) — RIFE arch used in our companion interpolation script
- [Comfy-Org](https://huggingface.co/Comfy-Org) — repacked RIFE weights
- MiniMax — H3 video model
- comfy_kitchen / aimdo authors — for the (buggy but fascinating) virtual-VRAM stack that motivated all this

## License

MIT
