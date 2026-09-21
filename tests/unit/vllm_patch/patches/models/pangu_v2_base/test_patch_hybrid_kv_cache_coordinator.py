# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for the hybrid APC per-group coordinator patch.

``find_longest_cache_hit_per_group`` keeps each group's real hit blocks (so the
deep D-side APC hit is not re-pulled) while clamping the returned scalar to
``min(hit_lengths)`` (so ``external = total - min`` fills the lagging
SWA/Mome groups).
"""

import pytest

from omni_npu.vllm_patches.patches.models.pangu_v2_base import (
    patch_hybrid_kv_cache_coordinator as hybrid_mod,
)


def _stub_upstream(monkeypatch, blocks, hit_lengths):
    """Replace the saved upstream with a deterministic stub."""
    def upstream(self, block_hashes, max_cache_hit_length):
        return blocks, hit_lengths

    monkeypatch.setattr(
        hybrid_mod.HybridAPCConnectorHitPatch,
        "_upstream_find_longest_cache_hit_per_group",
        upstream,
        raising=False,
    )


@pytest.mark.unit
def test_keeps_real_blocks_and_clamps_scalar_to_min(monkeypatch):
    blocks = (["dsa0", "dsa1"], ["swa0"], ["mome0"])
    _stub_upstream(monkeypatch, blocks, hit_lengths=(896, 128, 128))

    out_blocks, hit_lengths = hybrid_mod.find_longest_cache_hit_per_group(
        None, ["h0", "h1"], 1024
    )

    # Blocks are passed through untouched — the deep DSA hit stays.
    assert out_blocks is blocks
    # The scalar is clamped to the common (min) length.
    assert hit_lengths == (128, 128, 128)


@pytest.mark.unit
def test_zero_hit_is_preserved(monkeypatch):
    _stub_upstream(monkeypatch, blocks=([], []), hit_lengths=(0, 0))

    blocks, hit_lengths = hybrid_mod.find_longest_cache_hit_per_group(
        None, [], 256
    )

    assert blocks == ([], [])
    assert hit_lengths == (0, 0)


@pytest.mark.unit
def test_single_group_clamps_to_itself(monkeypatch):
    _stub_upstream(monkeypatch, blocks=(["only"],), hit_lengths=(896,))

    blocks, hit_lengths = hybrid_mod.find_longest_cache_hit_per_group(
        None, ["h0"], 1024
    )

    assert blocks == (["only"],)
    assert hit_lengths == (896,)


@pytest.mark.unit
def test_apply_calls_bypass_conflict(monkeypatch):
    called = []

    def fake_bypass(cls, *names):
        called.append(names)

    monkeypatch.setattr(
        hybrid_mod.HybridAPCConnectorHitPatch,
        "apply_bypass_conflict",
        classmethod(fake_bypass),
    )
    hybrid_mod.HybridAPCConnectorHitPatch.apply()

    assert called == [("find_longest_cache_hit_per_group",)]


@pytest.mark.unit
def test_connector_patch_targets_hybrid_coordinator():
    assert hybrid_mod.HybridAPCConnectorHitPatch._target is (
        hybrid_mod.HybridKVCacheCoordinator
    )
    assert hybrid_mod.HybridAPCConnectorHitPatch._attr_names_to_apply == [
        "find_longest_cache_hit_per_group"
    ]
    assert (
        hybrid_mod.HybridAPCConnectorHitPatch.find_longest_cache_hit_per_group
        is hybrid_mod.find_longest_cache_hit_per_group
    )
