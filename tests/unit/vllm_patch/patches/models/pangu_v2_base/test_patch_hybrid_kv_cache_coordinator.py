# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hybrid APC per-group hit: one common valid boundary for every group."""

import pytest

from omni_npu.vllm_patches.patches.models.pangu_v2_base import (
    patch_hybrid_kv_cache_coordinator as hybrid_mod,
)

NULL_BLOCK = "NULL"


class _Coord:
    """Minimal coordinator recording the common-prefix lookup call."""

    def __init__(self, blocks, hit_length):
        self.blocks = blocks
        self.hit_length = hit_length
        self.calls = []

    def find_longest_cache_hit(self, block_hashes, max_cache_hit_length):
        self.calls.append((block_hashes, max_cache_hit_length))
        return self.blocks, self.hit_length


def _call(coordinator, max_cache_hit_length=1024):
    return hybrid_mod.find_longest_cache_hit_per_group(
        coordinator, ["h0", "h1"], max_cache_hit_length
    )


@pytest.mark.unit
def test_broadcasts_only_the_common_boundary_to_all_groups():
    blocks = (["fa0", "fa1"], [NULL_BLOCK, "state256"], ["swa0"])
    coordinator = _Coord(blocks, 256)

    out_blocks, hit_lengths = _call(coordinator)

    assert out_blocks is blocks
    assert hit_lengths == (256, 256, 256)
    assert coordinator.calls == [(["h0", "h1"], 1024)]


@pytest.mark.unit
def test_missing_shorter_mome_state_reports_no_computed_prefix():
    # Independent hits can be FA=256 and MoME=512 while no MoME state exists
    # at 256. The common lookup must be allowed to reduce to zero.
    blocks = ([], [])
    coordinator = _Coord(blocks, 0)

    out_blocks, hit_lengths = _call(coordinator)

    assert out_blocks is blocks
    assert hit_lengths == (0, 0)


@pytest.mark.unit
def test_preserves_valid_common_state_and_original_eagle_limit():
    # The common lookup already accounts for the EAGLE lookahead; capping it
    # at a reduced per-group hit here would discard another EAGLE block.
    blocks = (["fa0"], [NULL_BLOCK, "state640"])
    coordinator = _Coord(blocks, 640)

    _, hit_lengths = _call(coordinator, max_cache_hit_length=834)

    assert hit_lengths == (640, 640)
    assert coordinator.calls == [(["h0", "h1"], 834)]


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
