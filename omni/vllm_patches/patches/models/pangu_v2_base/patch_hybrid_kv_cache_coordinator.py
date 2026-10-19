# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Hybrid APC: one common valid cache-hit boundary for every group.

Prefill must have a usable state in every group at the reported computed
boundary. MoME snapshots at longer prefixes cannot be truncated to that
boundary: the shorter snapshot may not exist. Use the coordinator's common
hit fixed point, which rechecks groups until all agree on a valid boundary.
"""

from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


def find_longest_cache_hit_per_group(
    self: HybridKVCacheCoordinator,
    block_hashes,
    max_cache_hit_length: int,
):
    """Return blocks and computed lengths at one valid common boundary."""
    # Pass the original request limit. Capping it at an already reduced
    # per-group hit would apply the EAGLE lookahead drop a second time.
    blocks, hit_length = self.find_longest_cache_hit(
        block_hashes, max_cache_hit_length
    )
    return blocks, (hit_length,) * len(blocks)


@register_patch("HybridAPCConnectorHitPatch", HybridKVCacheCoordinator)
class HybridAPCConnectorHitPatch(VLLMPatch):
    _attr_names_to_apply = ["find_longest_cache_hit_per_group"]

    find_longest_cache_hit_per_group = find_longest_cache_hit_per_group

    @classmethod
    def apply(cls):
        cls.apply_bypass_conflict("find_longest_cache_hit_per_group")
