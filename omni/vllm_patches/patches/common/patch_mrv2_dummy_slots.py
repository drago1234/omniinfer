# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backport the slot-kernel half of vLLM PR #56734 for MRV2 NPU.
Removal criteria: drop the loader patch when upgrading to vLLM >= 0.31.0.
"""
import vllm.v1.worker.gpu.block_table as upstream

from omni_npu.vllm_patches.core import VLLMPatch, register_patch
from omni_npu.worker.npu.ops.dummy_slot_mapping import _compute_slot_mappings_kernel


@register_patch("MRv2DummySlotKernelPatch", upstream)
class MRv2DummySlotKernelPatch(VLLMPatch):
    _attr_names_to_apply = ["_compute_slot_mappings_kernel"]
    _compute_slot_mappings_kernel = _compute_slot_mappings_kernel
