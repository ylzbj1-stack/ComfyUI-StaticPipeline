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
    # 2026-09-12: local experiment REVERTED to upstream values for a clean control.
    # Findings while it was in place (keep for the record):
    #   (15.8, 16.8) -> activations OOM on device 1 ("Currently allocated 18.82 GiB,
    #                   Requested 263 MB", ceiling ~19.0 GiB/device).
    #   (16.4, 16.8) -> sampling OK, "0 CPU-streamed blocks", 14.8 s/it vs 185 s/it
    #                   upstream, BUT then VRAM is so full that both VAEs fail to
    #                   load on the GPU (video VAE 4.85 GiB + audio 0.58 GiB):
    #                   first "Input type (torch.cuda.FloatTensor) and weight type
    #                   (torch.FloatTensor)" (0 MB loaded), and forcing them to cpu
    #                   hits "float != c10::Half" because vae_dtype() returns fp32
    #                   for cpu while the video VAE file is fp16.
    #   Arithmetic: DiT 31.6 + VAEs 5.43 + activations 3.7 = 40.7 GiB vs ~38 GiB
    #   usable -> "DiT fully resident" and "VAEs on GPU" cannot coexist. The upstream
    #   mid budget is deliberately what leaves the VAEs room.
    # 2026-09-12 CONTROL DONE - upstream mid + fixed env vars, 121f, 8 steps:
    #   SUCCESS, 15.17 s/it with "5 CPU-streamed blocks"; the all-resident (16.4,16.8)
    #   variant was 14.8 s/it, i.e. streaming costs ~2%. So the mid budget buys
    #   nothing, costs the VAEs their headroom, and the local experiment stays
    #   REVERTED. The old "185 s/it at mid" reading came from an instance started
    #   WITHOUT MGPU_CPU_THRESHOLD_PERCENT=999 / PYTORCH_CUDA_ALLOC_CONF - it was an
    #   environment bug, not a host-RAM shortage, so extra RAM is not the fix.
    # 2026-09-12 (b): 243f/960x544 实测 OOM -> 把 ~1.3 GiB 从 cuda:0 挪到 cuda:1。
    # 现场：SageAttention int8 量化缓冲要 514 MiB，device 0 已 allocated 18.68 GiB
    # (= 权重 14.76 + 两个 VAE 常驻 ~0.87 + 激活 ~3.05)，device 1 却还空着 ~6 GiB。
    # 即"装不下"是**卡间不均衡**，不是总量不够：上游 (15.36,14.0) 把 14.76 压在 0 卡、
    # 13.84 压在 1 卡，而两块卡的激活需求几乎对称 -> 0 卡先撞顶(1 GiB 保留区)。
    # (13.8, 14.0) 打包结果（按上游实测反解 pre≈1.56 / final≈0.04 / 每块≈0.60 GiB，
    # 50 块）：dev0 = 20 块 + pre ≈ 13.56，dev1 = 23 块 + final ≈ 13.88（**与上游逐字相同**，
    # 即"已知在 124f 跑通过"的那一侧完全不动），7 块走 CPU 流式（原 5）。
    # 只把 cuda:0 减重 1.2 GiB：两块卡变成 13.56 / 13.88 —— 对称，各留 ~5 GiB 给激活。
    # 只切这一个变量；若下次改报 device 1，再单独降 budget[1]。
    # 速度代价按控制实验的结论（流式块几乎免费，0 块 vs 5 块只差 2%）可忽略。
    # ⚠ 与"上游 mid 预算更快"不冲突：那条比的是 0 块流式 vs 5 块（差 2%），结论是
    #    "别为了常驻去挤 VAE"；这里是"为了装得下把权重挪到空着的卡上"。
    # 2026-09-12 (d): 改后第一次 243f 实跑 -> **dev0 不再爆，改成 device 1 爆**
    # （allocated 18.37 GiB / 请求 513.97 MiB），正是上面那句预案说的情况。
    # 三次 OOM 请求的都是同一个 513.97 MiB = SageAttention 的 int8 量化缓冲；
    # 分块注意力里 k/v 是**整条**的（见 _install_chunked_attention 注释），所以它
    # ∝ 序列 token 数 = 帧数，是 243f 的固有成本，与 LoRA/流式块无关（124f 从不 OOM）。
    # 而 dev1(13.90 权重/23 块) 的激活需求 4.47 比 dev0(3.05) 大 —— per-block 缓冲
    # 随该卡常驻块数走。故只降 budget[1] 14.0 -> 12.8：dev1 = 22 块 + final ≈ 12.61
    # （−1.29 GiB），块数 23→22 顺带把激活需求也压低；dev0 不动（它这轮没爆）。
    # 预期余量：−0.5 -> +1.6 GiB。代价：流式块 7 -> 8。
    # 2026-09-13 (d) 13.8 -> 10.8 on cuda:0 ONLY = 给视频 VAE 让出 3 GiB。**已实测确认，保留。**
    #   问题（probe_dec/probe_dec2，1 段×124f）：decode 139.4s 里视频 VAE 占 135.7s，
    #     py-spy 91% 叶子帧 = `r.copy_(weight, non_blocking=...)`（model_management.py:1531/1535）
    #     —— **权重 H2D 拷贝本身**，不是精度转换（cast_bias_weight 传 dtype=None）；
    #     解码期 PCIe 只有 ~450 MB/s，视频 VAE（fp16 4.85 GiB）只驻留 867 MB
    #     ⇒ 134.6s × 0.45 GB/s ≈ 60 GB ≈ **把整套 VAE 权重搬了约 12 遍**（每处理一个时间块重搬一次）。
    #   修法 = 让出 3 GiB，让 VAE 基本全驻留（实测 4868/4967 MB = 98%）。
    #   实测结果（steps=1 / steps=8 两轮）：
    #     decode      137.2s -> 58.0s（steps=1）｜122.9s -> 45.7s（段 2，steps=8）
    #     sample      162.6s -> 186.8s（steps=1，一次性成本）
    #   ⚠ **真链实测（probe_c11，11 段×124f、steps=8）**：sample 219.1 -> 235.8s（**+17.0s，+7.8%**）
    #     —— 不是 2 段探针给的 "+1%"！2 段探针**系统性低估长链代价**（长链页缓存更紧，
    #     流式块部分要从 D 盘现读：采样期实测系统级磁盘读 68 MB/s、可用内存仅剩 4.33 GB）。
    #     ⇒ **教训：凡是"让资源换性能"的改动，验收必须跑真实段数的链，不能拿 2 段探针下结论。**
    #   真链净收益：段 1 585.6 -> 545.1s；段 2~11 sample +17.0 / decode −65.6 ⇒ **每段净 −48.6s**；
    #     整片 **110.0（无预取）→ 70.2（预取/旧预算）→ 62.4 分钟（本改动）**，
    #     且成片 md5 与改造前**逐字节相同**（a4cf18196573cc7fe41f408d4c073555）⇒ 画面零影响。
    "mid": (10.8 * (1024 ** 3), 12.8 * (1024 ** 3)),
    # 2026-09-13 (e) 192-259f 专用档：**两卡都让**，为长序列的激活腾地方。
    #   来源：probe_len243（2 段×243f、steps=8、dev0=10.8）——
    #     段 1（无 pin）PASS，峰值 cuda:0=18586 / cuda:1=19251 MiB；
    #     段 2（+pin，序列 277f）**OOM on device 1**（allocated 17.97 GiB, 差 634 MB）。
    #   即：dev0 那一侧已经被 (d) 修好了，剩下的缺口在 **dev1 —— 而这台机器上 dev1 从未动过**。
    #   账面：dev1 = 权重 12.64 + 非权重 ~5.3 = 17.97；降到 ~10.6 权重 ⇒ ~16.6 GiB，余 ~2 GiB。
    #   代价：dev1 少放 ~4 个块（流式 14 -> 18）⇒ 按真链 +8.7%/5 块 折算约 +12~19s/段；
    #     在 243f（sample ~533s）上只占 2.5~3.5%，可接受。
    #   ⚠ 之所以**只给 ≥192f**：124f 是主力产线，不该为偶发长段位买单（那是 +7.8% 的税）。
    #     这是"拆档位"的中间形态 —— 等数据够了再换成连续函数 budget=f(frames)。
    "mid_long": (10.8 * (1024 ** 3), 10.8 * (1024 ** 3)),
    # 360f+: activations need ~3.4G/card under the ~19GiB WDDM ceiling, so
    # only ~22/24 blocks stay resident; the rest live on CPU and stream
    # card-ward for the duration of their own block call (~0.2s/step total).
    "long": (10.4 * (1024 ** 3), 10.2 * (1024 ** 3)),
}
_BUDGETS = PROFILES["short"]


# Asymmetric pairs (e.g. 20G + 48G): per-card weight budgets in bytes, set by the
# node's primary_gb / secondary_gb inputs. None = use the profile budgets above.
# Kept separate from _BUDGETS on purpose: _mid_tier() compares identity against
# PROFILES["mid"] to pick the chunked kernels, and that check must keep working.
_BUDGET_OVERRIDE = None


def _set_budget_override(primary_gb=0.0, secondary_gb=0.0):
    global _BUDGET_OVERRIDE
    if primary_gb and primary_gb > 0 and secondary_gb and secondary_gb > 0:
        _BUDGET_OVERRIDE = (primary_gb * (1024 ** 3), secondary_gb * (1024 ** 3))
        _log.info("budget override active: cuda:0=%.2f GiB, cuda:1=%.2f GiB (asymmetric pair)",
                  primary_gb, secondary_gb)
    else:
        if primary_gb or secondary_gb:
            _log.warning("budget override needs BOTH primary_gb and secondary_gb > 0; "
                         "ignoring (got primary=%.2f, secondary=%.2f)", primary_gb, secondary_gb)
        _BUDGET_OVERRIDE = None


def _effective_budgets():
    return _BUDGET_OVERRIDE if _BUDGET_OVERRIDE is not None else _BUDGETS


def _set_budgets(frames):
    global _BUDGETS
    if frames and frames >= 260:
        _BUDGETS = PROFILES["long"]
    elif frames and frames >= 192:
        _BUDGETS = PROFILES["mid_long"]
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
    # ------------------------------------------------------------------
    # 2026-09-12 (c) 诊断 + 修复。
    # 现场：243f 单段第 1 步卡 21 分钟；py-spy 25s 采样 2509 个样本，**99% 叶子帧**
    # 落在这里的 down.to()/up.to() —— 即缓存每次调用都 miss，整套 LoRA（~1GB）
    # 在主机/显卡之间反复重拷；nvidia-smi dmon 显示 PCIe 双向打满
    # (rx 5.9 / tx 7.8 GB/s)，磁盘 0、硬页错误 0。
    # 旧实现是**单槽缓存**（只留最近一次 (dtype,device)），键一翻就丢；换成
    # **每适配器按 (dtype,device) 分桶、最多留 2 份**，并统计 miss 与"键转换"直方图，
    # 这样下次跑完日志能直接告诉我们到底是哪个键在翻、翻得多频繁。
    # ⚠ 显存代价：每适配器最多多留 1 份，整套 LoRA ~1GB → 最坏 +1GB；当前 VRAM 已贴顶，
    #   若 OOM 就把 PROFILES["mid"] 的 budget 再往下压（流式块几乎免费的杠杆）。
    # ------------------------------------------------------------------
    _h_stats = {"calls": 0, "miss": 0}
    _h_trans: dict = {}
    _h_adapters: set = set()

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

        _h_stats["calls"] += 1
        key = (x.dtype, x.device)
        store = self.__dict__.get("_static_h_cache2")
        if store is None:
            store = self.__dict__["_static_h_cache2"] = {}
            _h_adapters.add(id(self))
        hit = store.get(key)
        if hit is None:
            _h_stats["miss"] += 1
            prev = self.__dict__.get("_static_h_key")
            if prev is not None and prev != key:
                t = "%s->%s" % (prev[0], prev[1])
                _h_trans[t] = _h_trans.get(t, 0) + 1
            self.__dict__["_static_h_key"] = key
            if len(store) >= 2:
                store.clear()
            # Lazily cache dtype/device-matched down/up on the adapter: casting
            # per call allocates fp32 intermediates (the fc1 delta alone is
            # 448MB in fp32); with the cache every matmul runs in x.dtype.
            hit = (down.to(device=x.device, dtype=x.dtype),
                   up.to(device=x.device, dtype=x.dtype))
            store[key] = hit
            if _h_stats["miss"] % 200 == 1:
                _log.info(
                    "chunked_h 诊断: calls=%d miss=%.1f%% adapters=%d 键转换TOP4=%s",
                    _h_stats["calls"], 100.0 * _h_stats["miss"] / max(1, _h_stats["calls"]),
                    len(_h_adapters),
                    sorted(_h_trans.items(), key=lambda kv: -kv[1])[:4])
        down_c, up_c = hit

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

# 2026-09-12 local experiment: the mid tier used to buy activation headroom by
# streaming its tail blocks from host RAM. That is far more expensive than it
# looks once host RAM is full (pagefile-speed transfers, measured 185 s/it vs
# 6.8 s/it fully resident), so mid now keeps every weight resident and pays a
# little extra compute instead — smaller chunks shrink the per-block transients.
# Scoped to mid so the short fast path keeps its unchunked GEMMs.
MID_ATTN_CHUNK = 4096       # == upstream default (see the mid-budget note above)
MID_MLP_CHUNK_ROWS = 16384  # == upstream default


def _mid_tier() -> bool:
    return _BUDGETS is PROFILES["mid"]


def _install_chunked_attention(attn, chunk=None):
    """Chunk the two position-wise projections (qkv_proj / out_proj) of an
    Attention block, writing into preallocated buffers. The rope fusion is
    per-token (in-place on the qkv buffer) and the attention core is flash
    (O(S) memory), so both stay whole; only the big matmul transients shrink
    (full-S x_qdata ~292MB -> ~21MB per chunk). Replicates
    comfy.ldm.minimax.model.Attention.forward verbatim otherwise.

    ``chunk=None`` resolves to the mid-tier override when the active profile is
    mid (see MID_ATTN_CHUNK)."""
    if chunk is None:
        chunk = MID_ATTN_CHUNK if _mid_tier() else 4096
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


def _install_chunked_positionwise(module, chunk=None):
    if chunk is None:
        chunk = MID_MLP_CHUNK_ROWS if _mid_tier() else MLP_CHUNK_ROWS
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

    budgets = _effective_budgets()   # profile budgets, or the asymmetric-pair override
    split_at = 0
    c0 = pre_bytes
    for i, s in enumerate(sizes):
        if c0 + s <= budgets[0]:
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
        if c1 + s <= budgets[1]:
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
        }, "optional": {
            "primary_gb": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 160.0, "step": 0.1,
                "tooltip": "Asymmetric GPU pairs: weight budget (GiB) for cuda:0, leaving the "
                           "rest of that card for activations. Set BOTH this and secondary_gb, "
                           "or leave both 0 to use the profile budgets."}),
            "secondary_gb": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 160.0, "step": 0.1,
                "tooltip": "Weight budget (GiB) for cuda:1 on asymmetric pairs."}),
        }}

    RETURN_TYPES = ("MODEL",)
    RETURN_NAMES = ("model",)
    FUNCTION = "convert"
    CATEGORY = "multigpu/static_pipeline"

    def convert(self, model, frames=0, primary_gb=0.0, secondary_gb=0.0):
        _set_budgets(frames)
        _set_budget_override(primary_gb, secondary_gb)
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
