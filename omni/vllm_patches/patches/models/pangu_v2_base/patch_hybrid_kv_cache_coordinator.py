# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Per-group real hit blocks with a uniform min hit-length scalar.

The scheduler connector path feeds ``max(per_group_hits)`` into
``allocate_slots`` as a single scalar. That scalar must be the common (min)
length so the SWA/Mome groups do not over-skip their window/state blocks.
The blocks, however, must keep each group's true hit (DSA deep, SWA/Mome
window/state) so the deep D-side APC hit is not re-pulled from P.
"""

from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


def find_longest_cache_hit_per_group(
    self: HybridKVCacheCoordinator,
    block_hashes,
    max_cache_hit_length: int,
):
    """Return per-group real blocks with a uniform min length scalar.

    ``find_longest_cache_hit`` truncates the full-attention group's blocks to
    the converged common length, discarding the deep D-side APC hit. Use the
    per-group lookup instead to keep those blocks (they carry a hash, so the
    connector skips them), while still reporting the min length so the
    scheduler allocates a safe scalar (external = total - min fills the
    lagging groups).
    """
    # adapt start
    # Keep each group's true hit blocks; only the scalar is clamped to min.
    blocks, hit_lengths = (
        HybridAPCConnectorHitPatch._upstream_find_longest_cache_hit_per_group(
            self, block_hashes, max_cache_hit_length
        )
    )
    return blocks, (min(hit_lengths),) * len(blocks)
    # adapt end


@register_patch("HybridAPCConnectorHitPatch", HybridKVCacheCoordinator)
class HybridAPCConnectorHitPatch(VLLMPatch):
    _attr_names_to_apply = ["find_longest_cache_hit_per_group"]

    find_longest_cache_hit_per_group = find_longest_cache_hit_per_group

    @classmethod
    def apply(cls):
        cls.apply_bypass_conflict("find_longest_cache_hit_per_group")
