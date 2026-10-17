# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Prefill common-prefix dispatch and Decode independent-hit regressions.

This module also runs without an installed omni_npu or vLLM package.
"""

import importlib.util
import os
import sys
import types
import unittest

from collections import namedtuple
from pathlib import Path
from unittest import mock

_SpecGroup = namedtuple(
    "SpecGroup", ("spec", "group_ids", "manager_cls", "use_eagle")
)


REPO_ROOT = Path(__file__).resolve().parents[6]
BASE_PATCH_PATH = (
    REPO_ROOT
    / "omni/vllm_patches/patches/models/pangu_v2_base/patch_hybrid_kv_cache_coordinator.py"
)

NULL_BLOCK = "NULL"
BLOCK_SIZE = 16


def _exec_patch(path, module_name):
    spec = importlib.util.spec_from_file_location(module_name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _load_base_patch():
    try:
        from omni_npu.vllm_patches.patches.models.pangu_v2_base import (
            patch_hybrid_kv_cache_coordinator as loaded_base,
        )
        from vllm.v1.kv_cache_interface import MambaSpec

        return loaded_base, MambaSpec
    except Exception:  # noqa: BLE001 - use a direct source load on dev machines
        pass

    saved_modules = {}

    def put(name, *, is_package=False, **attrs):
        saved_modules[name] = sys.modules.get(name)
        module = types.ModuleType(name)
        if is_package:
            module.__path__ = []
        for attr, value in attrs.items():
            setattr(module, attr, value)
        sys.modules[name] = module
        return module

    class _StubFullAttentionSpec:
        pass

    class KVCacheSpec:
        pass

    class _StubMambaSpec:
        pass

    class HybridKVCacheCoordinator:
        def find_longest_cache_hit_per_group(self, *args, **kwargs):
            raise AssertionError("producer must use the common-prefix lookup")

    class BlockHashListWithBlockSize:
        def __init__(self, block_hashes, hash_block_size, block_size):
            self.block_hashes = block_hashes
            self.scale = block_size // hash_block_size

        def __len__(self):
            return len(self.block_hashes) // self.scale

        def __getitem__(self, index):
            return self.block_hashes[(index + 1) * self.scale - 1]

    put("vllm", is_package=True)
    put("vllm.v1", is_package=True)
    put("vllm.v1.core", is_package=True)
    put(
        "vllm.v1.core.kv_cache_coordinator",
        HybridKVCacheCoordinator=HybridKVCacheCoordinator,
    )
    put(
        "vllm.v1.core.kv_cache_utils",
        BlockHashListWithBlockSize=BlockHashListWithBlockSize,
    )
    put(
        "vllm.v1.kv_cache_interface",
        FullAttentionSpec=_StubFullAttentionSpec,
        KVCacheSpec=KVCacheSpec,
        MambaSpec=_StubMambaSpec,
    )
    put("omni_npu", is_package=True)
    put("omni_npu.vllm_patches", is_package=True)
    put("omni_npu.vllm_patches.core", VLLMPatch=object, register_patch=lambda *a, **k: (lambda c: c))

    try:
        return (
            _exec_patch(BASE_PATCH_PATH, "patch_hybrid_kv_cache_coordinator_under_test"),
            _StubMambaSpec,
        )
    finally:
        for name, previous in saved_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


base_mod, MambaSpec = _load_base_patch()
FullAttentionSpec = base_mod.FullAttentionSpec


class FakeBlockPool:
    def __init__(self, cached_by_group):
        self.cached_by_group = cached_by_group

    def get_cached_block(self, block_hash, group_id):
        if block_hash in self.cached_by_group[group_id]:
            return f"g{group_id}b{block_hash}"
        return None


class _CacheHitManager:
    _scan_reverse = False

    @classmethod
    def find_longest_cache_hit(
        cls,
        block_hashes,
        max_length,
        kv_cache_group_ids,
        block_pool,
        kv_cache_spec,
        drop_eagle_block,
        alignment_tokens,
    ):
        del alignment_tokens
        group_id = kv_cache_group_ids[0]
        cached = block_pool.cached_by_group[group_id]
        max_blocks = max_length // kv_cache_spec.block_size
        computed = []
        indices = list(range(len(block_hashes)))
        if cls._scan_reverse:
            indices = list(reversed(indices))
        for i in indices:
            if i >= max_blocks:
                continue
            if i in cached:
                if cls._scan_reverse:
                    return ([NULL_BLOCK] * i + [f"g{group_id}b{i}"],)
                computed.append(f"g{group_id}b{i}")
            elif not cls._scan_reverse:
                break
        if drop_eagle_block and computed and not cls._scan_reverse:
            computed = computed[:-1]
        return (computed,)


class FullManager(_CacheHitManager):
    pass


class MomeManager(_CacheHitManager):
    _scan_reverse = True


class _FASpec(FullAttentionSpec):
    """Avoid the frozen dataclass __init__ when the real vLLM spec is loaded."""

    def __init__(self, block_size=BLOCK_SIZE):
        object.__setattr__(self, "block_size", block_size)


class _OtherSpec:
    def __init__(self, block_size=BLOCK_SIZE):
        self.block_size = block_size


class _MomeSpec(MambaSpec):
    """Mome/MambaSpec so its window-tail hits match the real spec class."""

    def __init__(self, block_size=BLOCK_SIZE):
        object.__setattr__(self, "block_size", block_size)


class FakeCoordinator:
    """``groups`` is ``(spec, manager_cls, cached_indices, use_eagle)``."""

    def __init__(self, groups, block_size=BLOCK_SIZE, num_hashes=16):
        self.hash_block_size = block_size
        self.scheduler_block_size = block_size
        self.block_hashes = [f"h{i}" for i in range(num_hashes)]
        self.block_pool = FakeBlockPool(
            [set(cached) for _spec, _manager, cached, _eagle in groups]
        )
        self.kv_cache_config = types.SimpleNamespace(
            kv_cache_groups=[object() for _ in groups]
        )
        self.attention_groups = [
            _SpecGroup(spec, [gid], manager, use_eagle)
            for gid, (spec, manager, _cached, use_eagle) in enumerate(groups)
        ]


class HybridAPCConnectorHitPatchTest(unittest.TestCase):
    @staticmethod
    def _call_per_group_consumer(coordinator, max_cache_hit_length=256):
        with mock.patch.object(base_mod, "_is_kv_consumer", return_value=True):
            return base_mod.find_longest_cache_hit_per_group(
                coordinator, coordinator.block_hashes, max_cache_hit_length
            )

    def test_consumer_reports_per_group_true_hit_lengths(self):
        coordinator = FakeCoordinator(
            [
                (_FASpec(), FullManager, range(8), False),
                (_MomeSpec(), MomeManager, [], False),
            ]
        )

        blocks, hit_lengths = self._call_per_group_consumer(coordinator)

        # per-group true lengths: FA 128, Mome miss 0 (no FA-scalar broadcast)
        self.assertEqual(hit_lengths, (128, 0))
        self.assertEqual(blocks[0], [f"g0b{i}" for i in range(8)])
        self.assertEqual(blocks[1], [])

    def test_consumer_keeps_local_mome_snapshot(self):
        coordinator = FakeCoordinator(
            [
                (_FASpec(), FullManager, range(8), False),
                (_MomeSpec(), MomeManager, [7], False),
            ]
        )

        blocks, hit_lengths = self._call_per_group_consumer(coordinator)

        # Mome window-tail hits stay: [null,..,real] keeps P/D pairing aligned
        self.assertEqual(hit_lengths, (128, 128))
        self.assertEqual(blocks[0], [f"g0b{i}" for i in range(8)])
        self.assertEqual(blocks[1], [NULL_BLOCK] * 7 + ["g1b7"])

    def test_consumer_swa_keeps_independent_hits(self):
        coordinator = FakeCoordinator(
            [
                (_FASpec(), FullManager, range(8), False),
                (_MomeSpec(), MomeManager, [7], False),
                (_OtherSpec(), FullManager, range(4), False),
            ]
        )

        blocks, hit_lengths = self._call_per_group_consumer(coordinator)

        # Every group reports its true prefix boundary, including sparse states.
        self.assertEqual(hit_lengths, (128, 128, 64))
        self.assertEqual(blocks[0], [f"g0b{i}" for i in range(8)])
        self.assertEqual(blocks[1], [NULL_BLOCK] * 7 + ["g1b7"])
        self.assertEqual(blocks[2], [f"g2b{i}" for i in range(4)])

    def test_consumer_zero_hit_is_preserved(self):
        coordinator = FakeCoordinator(
            [
                (_FASpec(), FullManager, [], False),
                (_MomeSpec(), MomeManager, [], False),
            ]
        )

        blocks, hit_lengths = self._call_per_group_consumer(coordinator)

        self.assertEqual(blocks, ([], []))
        self.assertEqual(hit_lengths, (0, 0))

    def test_consumer_registers_fa_group_ids(self):
        base_mod._FA_GROUP_IDS.clear()
        coordinator = FakeCoordinator(
            [
                (_FASpec(), FullManager, range(8), False),
                (_MomeSpec(), MomeManager, [7], False),
                (_OtherSpec(), FullManager, range(4), False),
            ]
        )

        self._call_per_group_consumer(coordinator)

        # only the full-attention group is registered for hash-skip
        self.assertEqual(base_mod.fa_group_ids(), frozenset({0}))

    def test_push_fa_group_ids_sets_connector_module_attribute(self):
        base_mod._FA_GROUP_IDS.clear()
        base_mod._FA_GROUP_IDS.update({0, 3})

        sentinel = types.ModuleType("llmdatadist_connector_v1")
        sentinel.__file__ = "/workspace/omniinfer/omni/connector/llmdatadist_connector_v1.py"
        other = types.ModuleType("other_module")
        other.__file__ = "/workspace/omniinfer/omni/connector/llmdatadist_manager_v1.py"
        with mock.patch.dict(sys.modules, {"_sentinel_conn": sentinel, "_other": other}):
            base_mod._push_fa_group_ids_to_connector()

            self.assertEqual(
                sentinel.FA_GROUP_IDS_SKIP_PULL, frozenset({0, 3})
            )
            self.assertFalse(hasattr(other, "FA_GROUP_IDS_SKIP_PULL"))
        base_mod._FA_GROUP_IDS.clear()

    def test_push_skips_none_modules_in_sys_modules(self):
        base_mod._FA_GROUP_IDS.clear()
        base_mod._FA_GROUP_IDS.update({0})

        sentinel = types.ModuleType("llmdatadist_connector_v1")
        sentinel.__file__ = "/omniinfer/omni/connector/llmdatadist_connector_v1.py"
        # sys.modules may hold None values (stale imports); the scan must skip them
        with mock.patch.dict(sys.modules, {"_sentinel_conn": sentinel, "_none_mod": None}):
            base_mod._push_fa_group_ids_to_connector()
            self.assertEqual(sentinel.FA_GROUP_IDS_SKIP_PULL, frozenset({0}))
        base_mod._FA_GROUP_IDS.clear()

    def test_is_kv_consumer_false_when_omni_pd_role_unset(self):
        env = {
            k: v
            for k, v in os.environ.items()
            if k not in ("OMNI_PD_ROLE", "ROLE")
        }
        with mock.patch.dict(os.environ, env, clear=True):
            self.assertFalse(base_mod._is_kv_consumer())

    def test_role_selection_uses_omni_pd_role(self):
        for role, expected in [("decode", True), (" Decode ", True),
                               ("prefill", False), ("hybrid", False), ("", False)]:
            with self.subTest(role=role), mock.patch.dict(
                os.environ, {"OMNI_PD_ROLE": role, "ROLE": "decode"}
            ):
                self.assertEqual(base_mod._is_kv_consumer(), expected)


class ProducerPathTest(unittest.TestCase):
    def _call(self, common_blocks, hit_length, maximum):
        hashes = ["h0", "h1"]
        lookup = mock.Mock(return_value=(common_blocks, hit_length))
        coordinator = types.SimpleNamespace(find_longest_cache_hit=lookup)
        with mock.patch.object(base_mod, "_is_kv_consumer", return_value=False):
            result = base_mod.find_longest_cache_hit_per_group(
                coordinator, hashes, maximum
            )
        lookup.assert_called_once_with(hashes, maximum)
        return result

    def test_missing_shorter_mome_state_reports_no_computed_prefix(self):
        # Independent hits can be FA=256 and MoME=512 while no MoME state
        # exists at 256. The common lookup must be allowed to reduce to zero.
        blocks = ([], [])
        self.assertEqual(self._call(blocks, 0, 1024), (blocks, (0, 0)))

    def test_preserves_valid_common_state_and_original_eagle_limit(self):
        # The common lookup already accounts for lookahead; using a reduced
        # per-group limit here would discard another EAGLE block.
        blocks = (["fa0", "fa1"], [NULL_BLOCK, "state256"])
        result = self._call(blocks, 256, 1024)
        self.assertIs(result[0], blocks)
        self.assertEqual(result[1], (256, 256))

    def test_broadcasts_only_the_common_boundary_to_all_groups(self):
        blocks = (["fa0"], ["state128"], ["swa0"])
        self.assertEqual(self._call(blocks, 128, 4096), (blocks, (128, 128, 128)))

    def test_apply_calls_bypass_conflict(self):
        called = []
        with mock.patch.object(
            base_mod.HybridAPCConnectorHitPatch,
            "apply_bypass_conflict",
            classmethod(lambda cls, *names: called.append(names)),
            create=True,
        ):
            base_mod.HybridAPCConnectorHitPatch.apply()
        self.assertEqual(called, [("find_longest_cache_hit_per_group",)])
