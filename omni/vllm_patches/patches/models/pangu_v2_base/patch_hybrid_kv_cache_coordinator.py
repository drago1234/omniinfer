# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

"""Per-group real hit blocks with a uniform min hit-length scalar.

Full-attention blocks keep their true hit so the deep D-side APC hit is not
re-pulled. The scalar is ``min(hit_lengths)`` so SWA/Mome do not over-skip.
A Mamba/Mome snapshot past that minimum is looked up again at the minimum:
its block list is ``[null] * i + [state]``, and the state for the shorter
prefix is not in that list.
"""

from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
from vllm.v1.core.kv_cache_utils import BlockHashListWithBlockSize
from vllm.v1.kv_cache_interface import MambaSpec

from omni_npu.vllm_patches.core import VLLMPatch, register_patch


def find_longest_cache_hit_per_group(
    self: HybridKVCacheCoordinator,
    block_hashes,
    max_cache_hit_length: int,
):
    """Return per-group real blocks with a uniform min length scalar."""
    # adapt start
    blocks, hit_lengths = (
        HybridAPCConnectorHitPatch._upstream_find_longest_cache_hit_per_group(
            self, block_hashes, max_cache_hit_length
        )
    )
    min_hit = min(hit_lengths)
    aligned = list(blocks)
    for spec, group_ids, manager_cls, _use_eagle in self.attention_groups:
        if not isinstance(spec, MambaSpec):
            continue
        if len(aligned[group_ids[0]]) * spec.block_size <= min_hit:
            continue
        hashes = block_hashes
        if spec.block_size != self.hash_block_size:
            hashes = BlockHashListWithBlockSize(
                block_hashes, self.hash_block_size, spec.block_size
            )
        resolved = manager_cls.find_longest_cache_hit(
            block_hashes=hashes,
            max_length=min_hit,
            kv_cache_group_ids=group_ids,
            block_pool=self.block_pool,
            kv_cache_spec=spec,
            drop_eagle_block=False,
            alignment_tokens=self.scheduler_block_size,
        )
        for gid, blks in zip(group_ids, resolved):
            aligned[gid] = blks
    return tuple(aligned), (min_hit,) * len(aligned)
    # adapt end


@register_patch("HybridAPCConnectorHitPatch", HybridKVCacheCoordinator)
class HybridAPCConnectorHitPatch(VLLMPatch):
    _attr_names_to_apply = ["find_longest_cache_hit_per_group"]

    find_longest_cache_hit_per_group = find_longest_cache_hit_per_group

    @classmethod
    def apply(cls):
        cls.apply_bypass_conflict("find_longest_cache_hit_per_group")
