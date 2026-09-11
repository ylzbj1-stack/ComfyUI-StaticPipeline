# -*- coding: utf-8 -*-
"""Static pipeline split for the MiniMax-H3 DiT across two GPUs.

Core idea: weight tensors are placed on their devices exactly ONCE (blocks
split between cuda:0 / cuda:1, embeddings+token_refiner on cuda:0,
final_layer on cuda:1) and NEVER move afterwards. Only activations cross
the PCIe link at group boundaries (a few hundred MB per step).

This is the exact opposite of DisTorch's runtime block streaming, which is
what corrupts aimdo vbar accounting and produces the dangling-pointer
access violations. No movement -> no dangling pointers -> no crash class.

Integration strategy (all accounting lies are deliberate):
- patcher.partially_load  -> no-op (weights already resident), applies LoRA
  patches as forward-time LowVramPatch hooks (same as core lowvram path,
  avoids the 34G backup copy that baking would create).
- patcher.partially_unload -> lies about having freed memory (model stays).
- patcher.unpatch_model   -> forces device_to=None so nothing is moved.
- patcher.is_dynamic      -> True on the primary patcher (skips the 34G
  host-pin budget request).
- comfy.model_management.free_memory -> wrapped to put every static-split
  LoadedModel into keep_loaded, so no eviction can ever touch it.

If VRAM truly runs out we fail with a recoverable Python OOM instead of a
native access violation - that is the accepted trade.
"""

import logging
import os
import types

import torch

import comfy.lora
import comfy.model_management as mm
import comfy.utils
from comfy.model_patcher import LowVramPatch, ModelPatcher, get_key_weight
from comfy.patcher_extension import CallbacksMP

PRIMARY = torch.device("cuda:0")
SECONDARY = torch.device("cuda:1")

# Weight byte budgets per card (GiB), selected by the node's "frames" input.
# Long clips (>=260 frames, 11s+) have linearly larger activations, so the
# split shifts one block to the primary card and keeps per-card headroom
# ~3.4G; short clips use the sp8-proven 23/27.
PROFILES = {
    "short": (15.8 * (1024 ** 3), 16.8 * (1024 ** 3)),   # 23/27, no streaming
    # 110-259f: activations need ~2G/card, which the all-resident split cannot
    # fit on cuda:1 (16.25 GiB weights + 2G > 19 GiB WDDM ceiling) - measured
    # OOM at 121f. Keep cuda:0 fully loaded, stream ~3 blocks from cuda:1.
    "mid": (15.36 * (1024 ** 3), 14.0 * (1024 ** 3)),
    # 360f+: activations need ~3.4G/card under the ~19GiB WDDM ceiling, so
    # only ~22/24 blocks stay resident; the rest live on CPU and stream
    # card-ward for the duration of their own block call (~0.2s/step total).
    "long": (10.4 * (1024 ** 3), 10.2 * (1024 ** 3)),
}
_BUDGETS = PROFILES["short"]


def _set_budgets(frames):
    global _BUDGETS
    if frames and frames >= 260:
        _BUDGETS = PROFILES["long"]
    elif frames and frames >= 110:
        _BUDGETS = PROFILES["mid"]
    else:
        _BUDGETS = PROFILES["short"]

FLAG = "_static_split"
GUARD_ATTR = "_static_split"
_log = logging.getLogger("StaticPipeline")


# --------------------------------------------------------------------------
# Device-aware dlpack wrap.
#
# The vanilla comfy_kitchen _wrap_for_dlpack exports via tensor.__dlpack__,
# which requires torch's CURRENT device to equal the tensor's device. The
# ComfyUI-MultiGPU guard handles that but adds P2P/CPU-staging logic keyed on
# the fork's *global* current device (cuda:0) - wrong for a static split,
# where every op's tensors are already local to the op's own device.
# We recover the vanilla function from the guard's closure and install a
# minimal device-switching wrapper: no staging, no P2P, just switch context.
# --------------------------------------------------------------------------
def _install_device_aware_dlpack_wrap():
    try:
        import comfy_kitchen.backends.cuda as ck_cuda
        guard = getattr(ck_cuda, "_wrap_for_dlpack", None)
        vanilla = None
        if guard is not None and getattr(guard, "_multigpu_cuda_device_guard", False):
            for cell in getattr(guard, "__closure__", None) or ():
                try:
                    v = cell.cell_contents
                except ValueError:
                    continue
                if callable(v) and getattr(v, "__name__", "") == "_wrap_for_dlpack":
                    vanilla = v
                    break
        if vanilla is None:
            _log.info("dlpack wrap: no MultiGPU guard found, using comfy_kitchen's current wrap as base")
            vanilla = guard
        if vanilla is None:
            _log.warning("dlpack wrap: nothing to wrap - skipping")
            return

        def _device_aware_wrap(tensor, *args, **kwargs):
            dev = getattr(tensor, "device", None)
            if dev is not None and dev.type == "cuda" and dev.index is not None \
                    and torch.cuda.current_device() != dev.index:
                with torch.cuda.device(dev):
                    return vanilla(tensor, *args, **kwargs)
            return vanilla(tensor, *args, **kwargs)

        ck_cuda._wrap_for_dlpack = _device_aware_wrap
        _log.info("dlpack wrap: device-aware wrapper installed (staging/P2P bypassed)")
    except Exception as e:
        _log.warning("dlpack wrap install failed: %s", e)


# --------------------------------------------------------------------------
# Chunked in-place LoRA h(x).
#
# The stock LoRAAdapter.h materializes the full [S, out] delta (600-900MB at
# 88f) and bypass adds another full-size tensor on top -> OOM on both cards.
# This replacement processes the delta in 8192-row chunks and accumulates it
# directly into base_out, so the only large tensors alive are x and base_out
# (+ ~200MB of chunk transients). Requires the bypass.py in-place patch
# (h_out is base_out -> skip the extra add).
# --------------------------------------------------------------------------
def _install_chunked_bypass_h():
    from comfy.weight_adapter import lora as lora_mod
    import torch.nn.functional as F

    orig_h = lora_mod.LoRAAdapter.h
    CHUNK = 1024

    def chunked_h(self, x, base_out):
        v = self.weights
        up, down = v[0], v[1]
        alpha, mid, dora, reshape = v[2], v[3], v[4], v[5]
        if mid is not None or dora is not None or reshape is not None:
            return orig_h(self, x, base_out)
        if getattr(self, "is_conv", False) or not torch.is_tensor(x) or not torch.is_tensor(base_out):
            return orig_h(self, x, base_out)
        if x.device != base_out.device or x.shape[-1] != down.shape[1] or base_out.shape[-1] != up.shape[0]:
            return orig_h(self, x, base_out)
        try:
            rows = x.numel() // x.shape[-1]
            x2 = x.reshape(rows, x.shape[-1])
            base2 = base_out.reshape(rows, base_out.shape[-1])
        except Exception:
            return orig_h(self, x, base_out)

        # Lazily cache dtype/device-matched down/up on the adapter: casting
        # per call allocates fp32 intermediates (the fc1 delta alone is
        # 448MB in fp32); with the cache every matmul runs in x.dtype.
        cache = getattr(self, "_static_h_cache", None)
        if cache is None or cache[0] != (x.dtype, x.device):
            down_c = down.to(device=x.device, dtype=x.dtype)
            up_c = up.to(device=x.device, dtype=x.dtype)
            cache = ((x.dtype, x.device), down_c, up_c)
            self._static_h_cache = cache
        _, down_c, up_c = cache

        rank = down.shape[0]
        scale = (alpha / rank) if alpha is not None else 1.0
        scale = scale * getattr(self, "multiplier", 1.0)
        for i in range(0, rows, CHUNK):
            xc = x2[i:i + CHUNK]
            delta = F.linear(F.linear(xc, down_c), up_c)
            delta.mul_(scale)
            base2[i:i + CHUNK].add_(delta)
        return base_out

    lora_mod.LoRAAdapter.h = chunked_h
    _log.info("chunked in-place LoRA h() installed (chunk=%d rows)", CHUNK)


# --------------------------------------------------------------------------
# free_memory guard: static models are load-bearing furniture.
# --------------------------------------------------------------------------
def _install_free_memory_guard():
    if getattr(mm.free_memory, "_static_guard", False):
        return
    _orig = mm.free_memory

    def guarded(memory_required, device, keep_loaded=None, **kwargs):
        keep = list(keep_loaded) if keep_loaded else []
        try:
            for lm in list(mm.current_loaded_models):
                patcher = getattr(lm, "model", None)
                if patcher is not None and getattr(patcher, GUARD_ATTR, False):
                    if lm not in keep:
                        keep.append(lm)
        except Exception:
            pass
        return _orig(memory_required, device, keep_loaded=keep, **kwargs)

    guarded._static_guard = True
    mm.free_memory = guarded
    _log.info("free_memory guard installed for static-split models.")


# --------------------------------------------------------------------------
# Device bridge: moves activations into the block's home device.
# --------------------------------------------------------------------------
class DeviceBridge(torch.nn.Module):
    def __init__(self, block, device):
        super().__init__()
        self.block = block
        self.device = torch.device(device)

    def extra_repr(self):
        return f"device={self.device}"

    def forward(self, h, t_emb, mod_segments, rope_freqs, transformer_options={}):
        if h.device != self.device:
            h = h.to(self.device, non_blocking=True)
        if t_emb.device != self.device:
            t_emb = t_emb.to(self.device, non_blocking=True)
        if mod_segments is not None and torch.is_tensor(mod_segments) and mod_segments.device != self.device:
            mod_segments = mod_segments.to(self.device, non_blocking=True)
        if rope_freqs is not None and rope_freqs.device != self.device:
            rope_freqs = rope_freqs.to(self.device, non_blocking=True)
        return self.block(h, t_emb, mod_segments, rope_freqs, transformer_options=transformer_options)


# --------------------------------------------------------------------------
# Device bridging via instance-level forward wrapping (no module nesting, so
# patch keys / named_modules paths stay identical to the original tree).
#
# The wrapper ALSO switches torch's current device to the block's home device
# for the duration of the call: comfy_kitchen keys its cublas workspace by
# torch.cuda.current_device(), so an int8 kernel on cuda:1 executed while the
# thread's current device is still cuda:0 would receive a cuda:0 workspace ->
# cudaErrorIllegalAddress. The context switch aligns every such assumption.
# --------------------------------------------------------------------------
def _move_args_to_device(args, device):
    out = []
    for a in args:
        if torch.is_tensor(a):
            out.append(a.to(device, non_blocking=True) if a.device != device else a)
        elif isinstance(a, (tuple, list)):
            out.append(_move_args_to_device(a, device) if any(
                torch.is_tensor(t) and t.device != device for t in a) else a)
        else:
            out.append(a)
    return type(args)(out) if isinstance(args, tuple) else out


def _install_bridge_forward(module, device, tag=""):
    if getattr(module, "_static_bridge_installed", False):
        return
    orig_forward = module.forward
    dev = torch.device(device)
    mem_debug = bool(os.environ.get("MGPU_STATIC_MEM_DEBUG"))

    def bridged(*args, **kw):
        if mem_debug and dev.type == "cuda":
            logging.info("[SP-mem] %s pre %.2f GiB (alloc)", tag,
                         torch.cuda.memory_allocated(dev) / 2**30)
            if tag == "b00":
                import gc
                tops = []
                for o in gc.get_objects():
                    try:
                        if torch.is_tensor(o) and o.device.type == "cuda" and o.device.index == dev.index:
                            tops.append((o.numel() * o.element_size(), tuple(o.shape), str(o.dtype)))
                    except Exception:
                        pass
                tops.sort(reverse=True)
                for sz, shp, dt in tops[:10]:
                    logging.info("[SP-mem-top] %6.0f MB %s %s", sz / 2**20, shp, dt)
        if any(torch.is_tensor(a) and a.device != dev for a in args if a is not None):
            args = _move_args_to_device(args, dev)
        with torch.cuda.device(dev):
            out = orig_forward(*args, **kw)
        if mem_debug and dev.type == "cuda":
            logging.info("[SP-mem] %s post %.2f GiB (alloc)", tag,
                         torch.cuda.memory_allocated(dev) / 2**30)
        return out

    module.forward = bridged
    module._static_bridge_installed = True
    module._static_home_device = dev


def _install_streaming_bridge(block, device):
    """CPU-resident block: hop onto its card for the duration of its own call,
    then return home. int8 packed weights move losslessly (same dtype), so
    quality is untouched; traffic is ~1.2GB per streamed block per step."""
    if getattr(block, "_static_streaming_installed", False):
        return
    orig_forward = block.forward
    dev = torch.device(device)

    def streamed(*args, **kw):
        # 2026-09-11 one-way transfer: weights are read-only during forward
        # (activation-space LoRA never writes them, int8 kernels never write
        # them), so the CPU copy is the permanent source of truth. Copy H2D,
        # then swap the saved CPU parameter objects back by reference - no D2H
        # copy at all. The old return hop (block.to("cpu")) was the access-
        # violation site: its kitchen _handle_to dispatch raced aimdo's page
        # management under WDDM, and crash probability scaled with the number
        # of streamed blocks (4/4 chain runs died there at 20 streamed blocks).
        # GC reclaims the GPU copies via torch's allocator only - no kitchen
        # dispatch, no vbar touch.
        cpu_refs = []
        for mod in block.modules():  # recursive, includes block itself
            for name, p in list(mod._parameters.items()):
                if p is not None:
                    cpu_refs.append((mod, name, p))
            for name, b in list(mod._buffers.items()):
                if b is not None:
                    cpu_refs.append((mod, name, b))

        torch.cuda.synchronize(PRIMARY)
        torch.cuda.synchronize(SECONDARY)
        block.to(dev)
        try:
            if any(torch.is_tensor(a) and a.device != dev for a in args if a is not None):
                args = _move_args_to_device(args, dev)
            with torch.cuda.device(dev):
                return orig_forward(*args, **kw)
        finally:
            for mod, name, cpu_obj in cpu_refs:
                if name in mod._parameters:
                    mod._parameters[name] = cpu_obj
                else:
                    mod._buffers[name] = cpu_obj
            torch.cuda.synchronize(dev)

    block.forward = streamed
    block._static_streaming_installed = True
    block._static_home_device = torch.device("cpu")


# --------------------------------------------------------------------------
# Chunked position-wise layers.
#
# MLP is swiglu: fc1 outputs ffn*2=28672 dims, so at 360f the intermediate is
# ~2.9GB - the single largest activation. fc1/fc2 are strictly position-wise,
# so row-chunking is numerically exact and caps the transient regardless of
# clip length (same technique as the chunked LoRA h).
# --------------------------------------------------------------------------
MLP_CHUNK_ROWS = 16384  # 88f (~13k rows) stays on the unchunked fast path

def _install_chunked_attention(attn, chunk=4096):
    """Chunk the two position-wise projections (qkv_proj / out_proj) of an
    Attention block, writing into preallocated buffers. The rope fusion is
    per-token (in-place on the qkv buffer) and the attention core is flash
    (O(S) memory), so both stay whole; only the big matmul transients shrink
    (full-S x_qdata ~292MB -> ~21MB per chunk). Replicates
    comfy.ldm.minimax.model.Attention.forward verbatim otherwise."""
    if getattr(attn, "_static_chunked", False):
        return
    import comfy.model_management as _mm
    import comfy.quant_ops as _qo
    from comfy.ldm.modules.attention import optimized_attention

    heads, head_dim = attn.heads, attn.head_dim
    inner = heads * head_dim
    qkv_dim = attn.qkv_proj.weight.shape[0]
    hidden = attn.out_proj.weight.shape[0]

    def fwd(x, rope_freqs=None, transformer_options={}):
        s = x.shape[0]
        if s <= chunk or not torch.is_tensor(x):
            return orig_attn_forward(x, rope_freqs, transformer_options)
        d = x.shape[-1]
        try:
            x2 = x.reshape(s, d)
        except Exception:
            return orig_attn_forward(x, rope_freqs, transformer_options)

        # chunked qkv projection -> preallocated [s, qkv_dim] buffer
        probe = attn.qkv_proj(x2[:chunk])
        qkv = torch.empty((s, qkv_dim), dtype=probe.dtype, device=probe.device)
        qkv[:chunk] = probe
        for i in range(chunk, s, chunk):
            qkv[i:i + chunk] = attn.qkv_proj(x2[i:i + chunk])
        del probe

        q, k, v = qkv.split(inner, dim=-1)
        v = v.view(s, heads, head_dim)
        if rope_freqs is not None:
            q = q.view(1, s, heads, head_dim)
            k = k.view(1, s, heads, head_dim)
            qw = _mm.cast_to(attn.q_norm.weight, device=x.device)
            kw = _mm.cast_to(attn.k_norm.weight, device=x.device)
            rot = rope_freqs.shape[-3] * 2
            if _mm.in_training:
                q, k = _qo.ck.rms_rope_split_half(
                    q, k, rope_freqs, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot)
            else:
                _qo.ck.rms_rope_split_half_(
                    q, k, rope_freqs, qw, kw, epsilon=attn.q_norm.eps, rot_dim=rot)
            q = q[0]
            k = k[0]
        else:
            q = attn.q_norm(q.view(s, heads, head_dim))
            k = attn.k_norm(k.view(s, heads, head_dim))
        q = q.transpose(0, 1).unsqueeze(0)
        k = k.transpose(0, 1).unsqueeze(0)
        v = v.transpose(0, 1).unsqueeze(0)
        # query-chunked attention: k,v stay whole (attention needs them all),
        # but q/attention-output/out_proj run per chunk so no [S, inner] or
        # [S, hidden] monolith is ever allocated.
        res = torch.empty((s, hidden), dtype=q.dtype, device=q.device)
        for i in range(0, s, chunk):
            qc = q[:, :, i:i + chunk]
            out_c = optimized_attention(qc, k, v, heads, mask=None, skip_reshape=True,
                                        transformer_options=transformer_options)
            res[i:i + chunk] = attn.out_proj(out_c.squeeze(0))
        return res

    orig_attn_forward = attn.forward
    attn.forward = fwd
    attn._static_chunked = True


def _install_chunked_positionwise(module, chunk=MLP_CHUNK_ROWS):
    if getattr(module, "_static_chunked", False):
        return
    orig_forward = module.forward

    def chunked_forward(x, *args, **kwargs):
        if not torch.is_tensor(x) or x.dim() < 2:
            return orig_forward(x, *args, **kwargs)
        d = x.shape[-1]
        rows = x.numel() // d
        if rows <= chunk:
            return orig_forward(x, *args, **kwargs)
        try:
            x2 = x.reshape(rows, d)
        except Exception:
            return orig_forward(x, *args, **kwargs)
        # Preallocate the full output and write per-chunk: peak memory is one
        # output + one chunk transient, never chunk-list + full copy (a cat
        # would hold both).
        first = orig_forward(x2[:chunk], *args, **kwargs)
        out = torch.empty((rows, first.shape[-1]), dtype=first.dtype, device=first.device)
        out[:chunk] = first
        for i in range(chunk, rows, chunk):
            out[i:i + chunk] = orig_forward(x2[i:i + chunk], *args, **kwargs)
        return out.reshape(x.shape[:-1] + (out.shape[-1],))

    module.forward = chunked_forward
    module._static_chunked = True


# --------------------------------------------------------------------------
# Placement
# --------------------------------------------------------------------------
def _ensure_placed(patcher):
    dm = patcher.model.diffusion_model
    if getattr(dm, "_static_split_done", False):
        return
    if not hasattr(dm, "blocks") or not isinstance(dm.blocks, torch.nn.ModuleList):
        raise RuntimeError("StaticPipelineSplit: diffusion_model has no ModuleList 'blocks'")

    # Unwrap bridges from a previous failed attempt so we never nest them.
    inner_blocks = [b.block if isinstance(b, DeviceBridge) else b for b in dm.blocks]
    pre_modules = [m for n, m in dm.named_children() if n not in ("blocks", "final_layer")]
    pre_bytes = sum(mm.module_size(m) for m in pre_modules)
    final_bytes = mm.module_size(dm.final_layer)
    sizes = [mm.module_size(b) for b in inner_blocks]
    total = sum(sizes)
    _log.info("DiT size: total=%.2f GiB, pre=%.2f GiB, final=%.2f GiB, %d blocks",
              total / 2**30, pre_bytes / 2**30, final_bytes / 2**30, len(sizes))

    split_at = 0
    c0 = pre_bytes
    for i, s in enumerate(sizes):
        if c0 + s <= _BUDGETS[0]:
            c0 += s
            split_at = i + 1
        else:
            break
    # secondary: resident while it fits, the rest streams from CPU
    c1 = 0.0
    streamed_idx = set()
    alt = 0
    for i in range(split_at, len(sizes)):
        s = sizes[i]
        if c1 + s <= _BUDGETS[1]:
            c1 += s
        else:
            streamed_idx.add(i)
    c1 += final_bytes
    _log.info("split: cuda:0=%.2f GiB (%d blocks + pre), cuda:1=%.2f GiB (%d blocks + final), %d CPU-streamed blocks",
              c0 / 2**30, split_at, (c1 - final_bytes) / 2**30,
              len(sizes) - split_at - len(streamed_idx), len(streamed_idx))

    for m in pre_modules:
        if any(p.device.type != "cuda" for p in m.parameters()):
            m.to(PRIMARY)
    stream_devs = (SECONDARY, PRIMARY)
    for i, block in enumerate(inner_blocks):
        if i < split_at:
            dev = PRIMARY
        elif i in streamed_idx:
            dev = stream_devs[alt % 2]
            alt += 1
        else:
            dev = SECONDARY
        needs_move = i not in streamed_idx and dev.type == "cuda" and any(p.device != dev for p in block.parameters())
        if needs_move:
            block.to(dev)
        if i in streamed_idx:
            _install_streaming_bridge(block, dev)
        else:
            _install_bridge_forward(block, dev, tag="b%02d" % i)
        if hasattr(block, "mlp"):
            _install_chunked_positionwise(block.mlp)
        if hasattr(block, "attn"):
            _install_chunked_attention(block.attn)
    if os.environ.get("MGPU_STATIC_MEM_DEBUG"):
        def _embed_probe(mod_name, mod):
            if mod is None:
                return
            orig = mod.forward
            def probed(x, *a, **k):
                logging.info("[SP-mem] embed:%s pre %.2f GiB", mod_name,
                             torch.cuda.memory_allocated(PRIMARY) / 2**30)
                out = orig(x, *a, **k)
                logging.info("[SP-mem] embed:%s post %.2f GiB (out %s)", mod_name,
                             torch.cuda.memory_allocated(PRIMARY) / 2**30,
                             tuple(out.shape) if torch.is_tensor(out) else "?")
                return out
            mod.forward = probed
        _embed_probe("video_patch_proj", getattr(dm, "video_patch_proj", None))
        _embed_probe("audio_patch_proj", getattr(dm, "audio_patch_proj", None))
        _embed_probe("condition_proj", getattr(dm, "condition_proj", None))
        _embed_probe("token_refiner", getattr(dm, "token_refiner", None))
    dm.final_layer.to(SECONDARY)
    _install_bridge_forward(dm.final_layer, SECONDARY, tag="final")
    if os.environ.get("MGPU_STATIC_MEM_DEBUG"):
        torch.cuda.synchronize(PRIMARY)
        torch.cuda.synchronize(SECONDARY)
        logging.info("[SP-mem] placement done: cuda:0 real %.2f GiB / cuda:1 real %.2f GiB (alloc)",
                     torch.cuda.memory_allocated(PRIMARY) / 2**30,
                     torch.cuda.memory_allocated(SECONDARY) / 2**30)

    # Output normalizer: forward returns on the last device; samplers expect
    # the load device (cuda:0).
    orig_forward = dm.forward
    dm._static_primary = PRIMARY

    def _forward(*args, **kwargs):
        out = orig_forward(*args, **kwargs)
        if isinstance(out, (list, tuple)):
            return [t.to(PRIMARY, non_blocking=True) if torch.is_tensor(t) else t for t in out]
        if torch.is_tensor(out):
            return out.to(PRIMARY, non_blocking=True)
        return out

    dm.forward = _forward
    dm._static_split_done = True
    torch.cuda.synchronize(PRIMARY)
    torch.cuda.synchronize(SECONDARY)
    _log.info("static placement done: %.2f GiB on cuda:0, %.2f GiB on cuda:1",
              c0 / 2**30, c1 / 2**30)


# --------------------------------------------------------------------------
# LoRA patch attachment (mirrors core load() lowvram branch, without moving)
# --------------------------------------------------------------------------
def _attach_patches(patcher):
    """Activation-space LoRA (bypass mode).

    The fork ships comfy/weight_adapter/bypass.py: BypassForwardHook wraps each
    patched Linear's forward and adds h(x) = up(down(x)) * (alpha/rank) to the
    output. Weight tensors are never touched, so:
      - weight_function stays empty -> comfy_kitchen keeps the int8 kernel
        fast path (no per-step dequant tax, the whole point), and
      - the kitchen requant crash class (bake attempt) is not on any path.
    Adapter A/B weights are tiny (~400MB total); they are re-placed onto each
    module's home device after injection (bypass.inject() defaults to the
    global torch device, wrong for a two-GPU split).

    Fallback: if no adapter-based bypass hooks could be created, use the
    production-proven LowVramPatch weight hooks instead.
    """
    model = patcher.model
    if model.current_weight_patches_uuid == patcher.patches_uuid:
        return
    patcher.unpatch_model(None, unpatch_weights=True)

    if not hasattr(model, "model_lowvram"):
        model.model_lowvram = False

    # object_patches (e.g. SageAttention override) - mirror patch_model()
    for k in patcher.object_patches:
        old = comfy.utils.set_attr(model, k, patcher.object_patches[k])
        if k not in patcher.object_patches_backup:
            patcher.object_patches_backup[k] = old

    bypassed = 0
    if patcher.patches:
        try:
            from comfy.weight_adapter.bypass import BypassInjectionManager
            from comfy.weight_adapter.base import WeightAdapterBase

            manager = BypassInjectionManager()
            for key, patch_list in patcher.patches.items():
                if not patch_list:
                    continue
                module_key = key.rsplit(".", 1)[0]  # strip trailing .weight/.bias
                for patch in patch_list:
                    try:
                        strength_patch, patch_data, strength_model, offset, function = patch
                    except Exception:
                        continue
                    if isinstance(patch_data, WeightAdapterBase):
                        manager.add_adapter(module_key, patch_data, strength=strength_patch)
            injections = manager.create_injections(model)
            for hook in manager.hooks:
                hook.inject()
                try:
                    dev = hook.module.weight.device  # quantized weights know their home card
                    # Device only; dtype conversion is handled lazily per
                    # x.dtype inside the chunked h() cache.
                    hook._move_adapter_weights_to_device(dev, None)
                except Exception:
                    pass
                bypassed += 1
            setattr(patcher, "_static_bypass_hooks", manager.hooks)
            _log.info("bypass LoRA: injected %d activation-space hooks", bypassed)
        except Exception as e:
            _log.warning("bypass LoRA failed (%s) - falling back to weight hooks", e)
            bypassed = 0

    if bypassed == 0 and patcher.patches:
        _attach_weight_hooks(patcher)

    model.model_lowvram = False
    model.lowvram_patch_counter = 0
    model.current_weight_patches_uuid = patcher.patches_uuid


def _attach_weight_hooks(patcher):
    """Fallback: forward-time LowVramPatch hooks (core lowvram style)."""
    model = patcher.model
    diffusion = model.diffusion_model
    hooked = 0
    module_by_prefix = {}
    for n, m in model.named_modules():
        module_by_prefix[n] = m
    for n, m in diffusion.named_modules():
        module_by_prefix.setdefault(f"diffusion_model.{n}", m)
    for key in list(patcher.patches.keys()):
        prefix, _, leaf = key.rpartition(".")
        if not prefix or leaf not in ("weight", "bias"):
            continue
        m = module_by_prefix.get(prefix)
        if m is None:
            _log.warning("patch key %s has no matching module - skipped", key)
            continue
        _, set_func, convert_func = get_key_weight(model, key)
        try:
            if leaf == "weight" and hasattr(m, "weight_function"):
                m.weight_function = [LowVramPatch(key, patcher.patches, convert_func, set_func)]
                hooked += 1
            elif leaf == "bias" and hasattr(m, "bias_function"):
                m.bias_function = [LowVramPatch(key, patcher.patches, convert_func, set_func)]
                hooked += 1
        except Exception as e:
            _log.warning("failed to attach patch %s: %s", key, e)
    _log.info("attached %d LowVramPatch forward hooks (fallback)", hooked)
    model.model_lowvram = True
    model.lowvram_patch_counter = hooked


def _refresh_counters(patcher):
    model = patcher.model
    model.device = PRIMARY
    model.model_loaded_weight_memory = patcher.model_size()
    model.model_offload_buffer_memory = 0


def _arm(patcher):
    """Node-time state: weights still on CPU. Truthful counters (loaded=0)
    make load_models_gpu's eviction math clear the card BEFORE placement."""
    model = patcher.model
    model.device = PRIMARY
    if not hasattr(model, "model_loaded_weight_memory"):
        model.model_loaded_weight_memory = 0
    model.model_loaded_weight_memory = 0
    model.model_offload_buffer_memory = 0
    if not hasattr(model, "model_lowvram"):
        model.model_lowvram = False
    if not hasattr(model, "current_weight_patches_uuid"):
        model.current_weight_patches_uuid = None


# --------------------------------------------------------------------------
# Patcher overrides
# --------------------------------------------------------------------------
def _override_partially_load(self, device_to, extra_memory=0, force_patch_weights=False):
    _ensure_placed(self)
    _attach_patches(self)
    _refresh_counters(self)
    # TE encoding (minutes of compute) leaves a fragmented dead-block pool;
    # sampling starts right after this call, so return dead segments to CUDA
    # now. Live static weights are untouched by empty_cache().
    if device_to is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()
    return 0


def _override_partially_unload(self, device_to, memory_to_free=0, force_patch_weights=False):
    # Lie: nothing was freed, weights never move. Returning the full amount
    # makes LoadedModel.model_unload report "not fully unloaded" and keeps us.
    _log.debug("partially_unload(%s) ignored for static-split model", memory_to_free)
    return memory_to_free


def _override_unpatch_model(self, device_to=None, unpatch_weights=True):
    # Never allow the follow-up model.to(device) that the legacy path does.
    return ModelPatcher.unpatch_model(self, None, unpatch_weights=unpatch_weights)


def _install_overrides(patcher):
    if getattr(patcher, FLAG, False):
        return
    patcher.partially_load = types.MethodType(_override_partially_load, patcher)
    patcher.partially_unload = types.MethodType(_override_partially_unload, patcher)
    patcher.unpatch_model = types.MethodType(_override_unpatch_model, patcher)
    # NOTE: is_dynamic must stay False (legacy). Returning True would make
    # clone(disable_dynamic=True) take the fork's fresh-model rebuild path
    # (cached_patcher_init) and silently swap in an unsplit model.
    setattr(patcher, FLAG, True)


def _on_clone(src, dst):
    # Propagate overrides through every clone (LoRA nodes etc.).
    try:
        _install_overrides(dst)
    except Exception as e:
        _log.warning("failed to propagate static-split overrides to clone: %s", e)


# --------------------------------------------------------------------------
# Node
# --------------------------------------------------------------------------
class StaticPipelineSplit:
    """Place the DiT statically across both GPUs; weights never move again.

    frames: the clip length this patcher will generate. 0 = unknown (short
    profile). >=260 switches to the long-clip weight split so the larger
    activations still fit under the ~19GiB WDDM ceiling.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {
            "model": ("MODEL",),
            "frames": ("INT", {"default": 0, "min": 0, "max": 4000, "step": 1}),
        }}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "convert"
    CATEGORY = "multigpu/static_pipeline"

    def convert(self, model, frames=0):
        _set_budgets(frames)
        out = model.clone()
        _install_overrides(out)
        _arm(out)
        if not any(getattr(cb, "_static_split_propagator", False)
                   for cbs in out.callbacks.get(CallbacksMP.ON_CLONE, {}).values()
                   for cb in cbs):
            prop = _on_clone
            prop._static_split_propagator = True
            out.add_callback(CallbacksMP.ON_CLONE, prop)
        _log.info("StaticPipelineSplit armed (placement deferred to load time)")
        return (out,)


__all__ = ["StaticPipelineSplit", "DeviceBridge", "_install_free_memory_guard"]
