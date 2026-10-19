# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Exercise Prefill dispatch with real vLLM cache lookup methods, without NPU."""

import importlib.util
import os
import types
import unittest
from collections import namedtuple
from unittest import mock


HAS_VLLM = importlib.util.find_spec("vllm") is not None
if HAS_VLLM:
    from vllm.v1.core.kv_cache_coordinator import HybridKVCacheCoordinator
    from vllm.v1.core.single_type_kv_cache_manager import (
        FullAttentionManager,
        MambaManager,
    )
    from vllm.v1.kv_cache_interface import FullAttentionSpec, MambaSpec

    from omni_npu.vllm_patches.patches.models.pangu_v2_base import (
        patch_hybrid_kv_cache_coordinator as patch,
    )

    class _FASpec(FullAttentionSpec):
        def __init__(self, block_size):
            object.__setattr__(self, "block_size", block_size)

    class _StateSpec(MambaSpec):
        def __init__(self, block_size):
            object.__setattr__(self, "block_size", block_size)


_Group = namedtuple("_Group", "spec group_ids manager_cls use_eagle")


class _Pool:
    def __init__(self, available):
        self.available = available
        self.null_block = object()

    def get_cached_block(self, block_hash, group_ids):
        if all(block_hash in self.available[g] for g in group_ids):
            return [(g, block_hash) for g in group_ids]
        return None


@unittest.skipUnless(HAS_VLLM, "requires the installed vLLM cache algorithms")
class HybridPrefixBoundaryTest(unittest.TestCase):
    def _lookup(self, snapshots, fa_hashes, *, maximum=1024, fa_size=128,
                alignment=256, eagle=False):
        groups = [_Group(_FASpec(fa_size), [0], FullAttentionManager, eagle)]
        groups.extend(
            _Group(_StateSpec(128), [i], MambaManager, False)
            for i in range(1, len(snapshots) + 1)
        )
        pool = _Pool([set(fa_hashes), *map(set, snapshots)])
        coordinator = types.SimpleNamespace(
            attention_groups=groups,
            kv_cache_config=types.SimpleNamespace(kv_cache_groups=groups),
            block_pool=pool,
            hash_block_size=128,
            scheduler_block_size=alignment,
        )
        coordinator.find_longest_cache_hit = types.MethodType(
            HybridKVCacheCoordinator.find_longest_cache_hit, coordinator
        )
        with mock.patch.dict(os.environ, {"OMNI_PD_ROLE": "prefill"}):
            blocks, hits = patch.find_longest_cache_hit_per_group(
                coordinator, list(range(8)), maximum
            )
        self.assertEqual(len(set(hits)), 1)
        for group, row, hit in zip(groups, blocks, hits):
            self.assertEqual(len(row) * group.spec.block_size, hit)
            if hit and isinstance(group.spec, MambaSpec):
                self.assertIsNot(row[-1], pool.null_block)
                self.assertEqual(row[-1][1], hit // 128 - 1)
        return blocks, hits

    def test_missing_shorter_state_recomputes_prefix(self):
        self.assertEqual(self._lookup([{3}], range(2)), (([], []), (0, 0)))

    def test_existing_shorter_state_is_reused(self):
        blocks, hits = self._lookup([{1, 3}], range(2))
        self.assertEqual(hits, (256, 256))
        self.assertEqual(blocks[1][-1], (1, 1))

    def test_later_group_shrink_rechecks_earlier_state_group(self):
        _, hits = self._lookup([{1, 5}, {1, 3}], range(8))
        self.assertEqual(hits, (256, 256, 256))

    def test_disjoint_state_snapshots_have_no_common_prefix(self):
        blocks, hits = self._lookup([{1, 5}, {3, 7}], range(8))
        self.assertEqual(hits, (0, 0, 0))
        self.assertTrue(all(not row for row in blocks))

    def test_different_block_sizes_use_correct_hash_boundaries(self):
        _, hits = self._lookup([{1, 3}], {1, 3}, fa_size=256)
        self.assertEqual(hits, (512, 512))

    def test_eagle_does_not_drop_lookahead_twice(self):
        _, hits = self._lookup([{5}], range(8), eagle=True)
        self.assertEqual(hits, (768, 768))

    def test_request_limit_and_scheduler_alignment_are_respected(self):
        _, hits = self._lookup([{1, 3, 5}], range(8), maximum=511)
        self.assertEqual(hits, (256, 256))


if __name__ == "__main__":
    unittest.main()
