# Patches

Optional patches against upstream repos. All were part of making the static
split work on this stack; apply with `git apply <file>` from the repo root.

## comfy-core_minimax-model_adaln-cache-embed-free.patch

Against `comfyanonymous/ComfyUI` (MiniMax H3 model file):

1. `adaln_t_table` (96768x2688 bf16, 496 MB) was `cast_to`'d on **every
   forward call** - on a 360-frame job 9 simultaneous copies (4.46 GB) stayed
   alive. Now cached on the module instance (the table is read-only).
2. Embed intermediates are deleted right after assembling `h` (~1 GB freed
   before sampling starts).

Both fixes also benefit single-GPU runs.

## comfy-core_weight-adapter-bypass_inplace-support.patch

Against `comfyanonymous/ComfyUI` `comfy/weight_adapter/bypass.py`. The static
pipeline's chunked in-place LoRA `h()` mutates `base_out` and returns it; the
upstream summation `base_out + h_out` would then add the tensor to itself.
This patch short-circuits that case. Required if you use the chunked LoRA.

## multigpu_p2p-registry_windows-cudart.patch

Against `pollockjj/ComfyUI-MultiGPU` `p2p_registry.py`. Upstream loads
`ctypes.CDLL("libcudart.so")` which cannot resolve on Windows. The patch
loads the correct `cudart64_*.dll` from `torch/lib` instead. Only needed if
you run ComfyUI-MultiGPU alongside this node AND trigger its P2P probe.
