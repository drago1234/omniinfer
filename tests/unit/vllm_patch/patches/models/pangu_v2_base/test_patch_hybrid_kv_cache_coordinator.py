# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hybrid APC coordinator patch tests: min-scalar producer + Decode FA-skip.

Covers:
* Non-consumer (producer): per-group real blocks with a uniform ``min``
  scalar; MoME/Mamba past the minimum is re-looked up at the minimum
* Decode (``OMNI_PD_ROLE=decode``): FA-skip, empty local Mome/Mamba
* Patch wiring: target attr and ``apply`` calls ``apply_bypass_conflict``

The file can run without installing ``omni_npu``.

Each group reports its own true hit length and keeps its real hit blocks:
the scheduler takes ``max(per_group_hits)`` as the FA length, so no group's
scalar may be padded to the FA scalar (skip-padding / external allocation /
cache_blocks bookkeeping all key off that scalar per group). Mome/Mamba hits
stay so the ``[null,..,real]`` layout keeps the scalar/block-table pairing.
"""

import importlib.util
import os
import sys
import types
import unittest

import pytest
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

        return loaded_base
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

    def _stub_find_longest_cache_hit_per_group(self, block_hashes, max_cache_hit_length):
        return ((), (0,))

    class HybridKVCacheCoordinator:
        find_longest_cache_hit_per_group = _stub_find_longest_cache_hit_per_group

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
        return _exec_patch(BASE_PATCH_PATH, "patch_hybrid_kv_cache_coordinator_under_test")
    finally:
        for name, previous in saved_modules.items():
            if previous is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = previous


base_mod = _load_base_patch()
FullAttentionSpec = base_mod.FullAttentionSpec
MambaSpec = base_mod.MambaSpec


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
                computed.append(f"g{group_id}b{i}")
            elif not cls._scan_reverse:
                break
        if drop_eagle_block and computed and not cls._scan_reverse:
            computed = computed[:-1]
        if cls._scan_reverse:
            computed = list(reversed(computed))
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


class _MomeSpec(base_mod.MambaSpec):
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

    def test_non_consumer_keeps_real_blocks_with_min_scalar(self):
        coordinator = FakeCoordinator(
            [
                (_FASpec(), FullManager, range(8), False),
                (_OtherSpec(), FullManager, range(4), False),
            ]
        )
        sentinel = ((["fa0", "fa1"], ["swa0"]), (128, 64))

        with mock.patch.object(base_mod, "_is_kv_consumer", return_value=False), mock.patch.object(
            base_mod.HybridAPCConnectorHitPatch,
            "_upstream_find_longest_cache_hit_per_group",
            staticmethod(lambda self, block_hashes, max_cache_hit_length: sentinel),
            create=True,
        ):
            result = base_mod.find_longest_cache_hit_per_group(
                coordinator, coordinator.block_hashes, 256
            )

        self.assertIs(result[0][0], sentinel[0][0])
        self.assertEqual(result[1], (min(sentinel[1]),) * 2)

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
        self.assertEqual(hit_lengths, (128, 16))
        self.assertEqual(blocks[0], [f"g0b{i}" for i in range(8)])
        self.assertEqual(blocks[1], ["g1b7"])

    def test_consumer_swa_keeps_independent_hits(self):
        coordinator = FakeCoordinator(
            [
                (_FASpec(), FullManager, range(8), False),
                (_MomeSpec(), MomeManager, [7], False),
                (_OtherSpec(), FullManager, range(4), False),
            ]
        )

        blocks, hit_lengths = self._call_per_group_consumer(coordinator)

        # every group reports its own true length (128, 16, 64)
        self.assertEqual(hit_lengths, (128, 16, 64))
        self.assertEqual(blocks[0], [f"g0b{i}" for i in range(8)])
        self.assertEqual(blocks[1], ["g1b7"])
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


# ---- producer path: upstream !2760 semantics (min scalar, MoME re-lookup) ----
class _NoMamba:
    attention_groups = []


def _stub_upstream(mp, blocks, hit_lengths):
    def upstream(self, block_hashes, max_cache_hit_length):
        return blocks, hit_lengths

    mp.setattr(
        base_mod.HybridAPCConnectorHitPatch,
        "_upstream_find_longest_cache_hit_per_group",
        upstream,
    )


def _stub_upstream_static(mp, blocks, hit_lengths):
    """Bypass the bound-classmethod call the patch makes at runtime."""

    def upstream(obj, block_hashes, max_cache_hit_length):
        return blocks, hit_lengths

    mp.setattr(
        base_mod.HybridAPCConnectorHitPatch,
        "_upstream_find_longest_cache_hit_per_group",
        upstream,
    )


class _Coord:
    def __init__(self, attention_groups):
        self.attention_groups = attention_groups
        self.hash_block_size = 128
        self.block_pool = object()
        self.scheduler_block_size = 128


def _mome_spec_placeholder():
    """A real MambaSpec instance, or a stub when vLLM/torch are unavailable."""
    try:
        import torch
        from vllm.v1.kv_cache_interface import MambaSpec

        return MambaSpec(
            block_size=128,
            shapes=((1,), (1,), (1,)),
            dtypes=(torch.float16, torch.float16, torch.float16),
        )
    except Exception:  # noqa: BLE001
        return _MomeSpec(block_size=128)


def _unused_mamba_spec():
    import torch
    from vllm.v1.kv_cache_interface import MambaSpec

    return MambaSpec(
        block_size=128,
        shapes=((1,), (1,), (1,)),
        dtypes=(torch.float16, torch.float16, torch.float16),
    )


class ProducerPathTest(unittest.TestCase):
    def setUp(self):
        self._mp = mock.patch.object(
            base_mod, "_is_kv_consumer", return_value=False
        )
        self._mp.start()
        self.addCleanup(self._mp.stop)

    def test_clamps_scalar_and_keeps_real_blocks(self):
        blocks = (["dsa0", "dsa1"], ["swa0"])
        with mock.patch.object(
            base_mod.HybridAPCConnectorHitPatch,
            "_upstream_find_longest_cache_hit_per_group",
            lambda self, bh, ml: (blocks, (896, 128)),
            create=True,
        ):
            out_blocks, hit_lengths = base_mod.find_longest_cache_hit_per_group(
                _NoMamba(), ["h0"], 1024
            )
        self.assertEqual(out_blocks, ([block for block in blocks[0]], blocks[1]))
        self.assertEqual(hit_lengths, (128, 128))

    def test_mamba_past_min_is_reresolved(self):
        dsa_blocks = ["dsa"] * 5
        short_state = ["null"] * 4 + ["state_640"]
        captured = {}

        class _Mgr:
            @classmethod
            def find_longest_cache_hit(cls, **kwargs):
                captured.update(kwargs)
                return (short_state,)

        # Mome/Mamba spec: len(blocks)*block_size (768) > min (640) -> re-lookup
        state_list = ["null"] * 5 + ["state_768"]
        coord = _Coord([(_mome_spec_placeholder(), [1], _Mgr, True)])
        with mock.patch.object(
            base_mod.HybridAPCConnectorHitPatch,
            "_upstream_find_longest_cache_hit_per_group",
            lambda self, bh, ml: ((dsa_blocks, state_list), (640, 768)),
            create=True,
        ):
            out_blocks, hit_lengths = base_mod.find_longest_cache_hit_per_group(
                coord, ["h0"], 834
            )
        self.assertEqual(out_blocks[0], dsa_blocks)
        self.assertEqual(out_blocks[1], short_state)
        self.assertEqual(hit_lengths, (640, 640))
        self.assertEqual(captured["max_length"], 640)
        self.assertFalse(captured["drop_eagle_block"])

    def test_mamba_within_min_is_not_requeried(self):
        dsa_blocks = ["dsa"] * 16
        state = ["null"] * 4 + ["state_640"]
        blocks = (dsa_blocks, state)

        class _Mgr:
            @classmethod
            def find_longest_cache_hit(cls, **kwargs):
                raise AssertionError("mamba hit within min must not be re-queried")

        coord = _Coord([(_mome_spec_placeholder(), [1], _Mgr, False)])
        with mock.patch.object(
            base_mod.HybridAPCConnectorHitPatch,
            "_upstream_find_longest_cache_hit_per_group",
            lambda self, bh, ml: (blocks, (2048, 640)),
            create=True,
        ):
            out_blocks, hit_lengths = base_mod.find_longest_cache_hit_per_group(
                coord, ["h0"], 4096
            )
        self.assertEqual(out_blocks[0], dsa_blocks)
        self.assertEqual(out_blocks[1], state)
        self.assertEqual(hit_lengths, (640, 640))

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
