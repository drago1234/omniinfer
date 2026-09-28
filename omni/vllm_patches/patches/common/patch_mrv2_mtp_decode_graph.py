# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Independent MRv2 MTP capture sizes for draft positions after the first.

Set additional_config.mtp_decode_cudagraph_capture_sizes to request-count
buckets (one token per request). Target and first-draft sizes stay unchanged.
"""

from copy import copy
from functools import wraps

from vllm.logger import init_logger

from omni_npu.vllm_patches.core import VLLMPatch, register_patch

logger = init_logger(__name__)


def decode_graph_config(vllm_config, decode_query_len):
    sizes = vllm_config.additional_config.get("mtp_decode_cudagraph_capture_sizes")
    spec = vllm_config.speculative_config
    if sizes is None or spec is None or spec.method != "mtp":
        return vllm_config
    if decode_query_len != 1 or spec.uses_dynamic_speculative_decoding():
        raise ValueError("MTP decode graph sizes require fixed, one-token draft steps")
    max_reqs = vllm_config.scheduler_config.max_num_seqs
    if (not isinstance(sizes, list) or not sizes
            or any(type(size) is not int or not 1 <= size <= max_reqs
                   for size in sizes)):
        raise ValueError("mtp_decode_cudagraph_capture_sizes must be a nonempty "
                         "list of integers in [1, max_num_seqs]")
    # Never mutate the shared config: target and draft-prefill managers use it.
    config = copy(vllm_config)
    config.compilation_config = copy(vllm_config.compilation_config)
    config.compilation_config.cudagraph_capture_sizes = sorted(set(sizes))
    config.compilation_config.max_cudagraph_capture_size = max(sizes)
    logger.info("[omni-npu/mrv2] MTP later-draft graph sizes: %s; target sizes: %s",
                config.compilation_config.cudagraph_capture_sizes,
                vllm_config.compilation_config.cudagraph_capture_sizes)
    return config


def build_decode_manager_init(original_init):
    @wraps(original_init)
    def initialize(self, vllm_config, device, cudagraph_mode, decode_query_len,
                   *args, **kwargs):
        config = decode_graph_config(vllm_config, decode_query_len)
        original_init(self, config, device, cudagraph_mode, decode_query_len,
                      *args, **kwargs)
    return initialize


def draft_compile_config(vllm_config):
    config = decode_graph_config(vllm_config, 1)
    if config is vllm_config:
        return config
    compilation = config.compilation_config
    original = vllm_config.compilation_config
    # The NPU compiler specializes the nominal dynamic range to its first
    # shape. Give the draft model exact compile entries for both routines.
    # Keep the shared attention-layer registry: the runner discovers draft
    # layers through the target config's static_forward_context.
    sizes = set(compilation.cudagraph_capture_sizes)
    sizes.update(original.cudagraph_capture_sizes or [])
    for size in original.compile_sizes or []:
        if size != "cudagraph_capture_sizes":
            sizes.add(size)
    compilation.compile_sizes = sorted(sizes)
    compilation.cudagraph_capture_sizes = original.cudagraph_capture_sizes
    compilation.max_cudagraph_capture_size = original.max_cudagraph_capture_size
    logger.info("[omni-npu/mrv2] MTP draft-only compile sizes: %s",
                compilation.compile_sizes)
    return config


def build_mtp_model_loader(original_loader):
    @wraps(original_loader)
    def load_model(target_model, vllm_config):
        return original_loader(target_model, draft_compile_config(vllm_config))
    return load_model


try:
    from vllm.v1.worker.gpu.spec_decode.autoregressive.cudagraph_utils import (
        DecodeSpeculatorCudaGraphManager,
    )
except ImportError:
    logger.warning("[omni-npu/mrv2] DecodeSpeculatorCudaGraphManager unavailable")
else:
    @register_patch("MRv2MTPDecodeGraphPatch", DecodeSpeculatorCudaGraphManager)
    class MRv2MTPDecodeGraphPatch(VLLMPatch):
        _attr_names_to_apply = ["__init__"]

        __init__ = build_decode_manager_init(DecodeSpeculatorCudaGraphManager.__init__)


try:
    import vllm.v1.worker.gpu.spec_decode.mtp.speculator as mtp_module
except ImportError:
    logger.warning("[omni-npu/mrv2] MTPSpeculator unavailable")
else:
    @register_patch("MRv2MTPDraftCompileSizesPatch", mtp_module)
    class MRv2MTPDraftCompileSizesPatch(VLLMPatch):
        _attr_names_to_apply = ["load_eagle_model"]

        load_eagle_model = build_mtp_model_loader(mtp_module.load_eagle_model)
