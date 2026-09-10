# -*- coding: utf-8 -*-
"""ComfyUI-StaticPipeline: static two-GPU pipeline split for the H3 DiT.

Weights are placed once (blocks split cuda:0/cuda:1) and never move again;
activations cross PCIe at group boundaries. Replaces DisTorch streaming for
single-task dual-GPU acceleration without the aimdo/vbar crash class.
"""
import logging

from .static_pipeline import (StaticPipelineSplit, _install_free_memory_guard,
                              _install_device_aware_dlpack_wrap, _install_chunked_bypass_h)

_install_free_memory_guard()
_install_device_aware_dlpack_wrap()
_install_chunked_bypass_h()

NODE_CLASS_MAPPINGS = {
    "StaticPipelineSplit": StaticPipelineSplit,
}
NODE_DISPLAY_NAME_MAPPINGS = {
    "StaticPipelineSplit": "Static Pipeline Split (2-GPU DiT)",
}

logging.info("[StaticPipeline] custom node loaded, free_memory guard installed")
__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
