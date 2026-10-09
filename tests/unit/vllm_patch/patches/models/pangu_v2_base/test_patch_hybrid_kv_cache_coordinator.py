# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0

"""Hybrid APC per-group hit: min scalar, DSA blocks kept, MoME re-looked up."""

import pytest

from omni_npu.vllm_patches.patches.models.pangu_v2_base import (
    patch_hybrid_kv_cache_coordinator as hybrid_mod,
)


class _NoMamba:
    attention_groups = []


def _stub_upstream(monkeypatch, blocks, hit_lengths):
    def upstream(self, block_hashes, max_cache_hit_length):
        return blocks, hit_lengths

    monkeypatch.setattr(
        hybrid_mod.HybridAPCConnectorHitPatch,
        "_upstream_find_longest_cache_hit_per_group",
        upstream,
        raising=False,
    )


def _mamba_spec():
    import torch
    from vllm.v1.kv_cache_interface import MambaSpec

    return MambaSpec(
        block_size=128,
        shapes=((1,), (1,), (1,)),
        dtypes=(torch.float16, torch.float16, torch.float16),
    )


class _Coord:
    def __init__(self, attention_groups):
        self.attention_groups = attention_groups
        self.hash_block_size = 128
        self.block_pool = object()
        self.scheduler_block_size = 128


@pytest.mark.unit
def test_clamps_scalar_and_keeps_dsa_blocks(monkeypatch):
    blocks = (["dsa0", "dsa1"], ["swa0"])
    _stub_upstream(monkeypatch, blocks, hit_lengths=(896, 128))

    out_blocks, hit_lengths = hybrid_mod.find_longest_cache_hit_per_group(
        _NoMamba(), ["h0"], 1024
    )

    assert out_blocks == blocks
    assert hit_lengths == (128, 128)


@pytest.mark.unit
def test_mamba_past_min_is_reresolved(monkeypatch):
    dsa_blocks = ["dsa"] * 5
    long_state = ["null"] * 5 + ["state_768"]
    short_state = ["null"] * 4 + ["state_640"]
    _stub_upstream(monkeypatch, (dsa_blocks, long_state), hit_lengths=(640, 768))
    captured = {}

    class _Mgr:
        @classmethod
        def find_longest_cache_hit(cls, **kwargs):
            captured.update(kwargs)
            return (short_state,)

    out_blocks, hit_lengths = hybrid_mod.find_longest_cache_hit_per_group(
        _Coord([(_mamba_spec(), [1], _Mgr, True)]),
        ["h0"],
        834,
    )

    assert out_blocks[0] is dsa_blocks
    assert out_blocks[1] == short_state
    assert hit_lengths == (640, 640)
    assert captured["max_length"] == 640
    assert captured["drop_eagle_block"] is False


@pytest.mark.unit
def test_mamba_within_min_is_not_requeried(monkeypatch):
    dsa_blocks = ["dsa"] * 16
    state = ["null"] * 4 + ["state_640"]
    blocks = (dsa_blocks, state)
    _stub_upstream(monkeypatch, blocks, hit_lengths=(2048, 640))

    class _Mgr:
        @classmethod
        def find_longest_cache_hit(cls, **kwargs):
            raise AssertionError("mamba hit within min must not be re-queried")

    out_blocks, hit_lengths = hybrid_mod.find_longest_cache_hit_per_group(
        _Coord([(_mamba_spec(), [1], _Mgr, False)]),
        ["h0"],
        4096,
    )

    assert out_blocks == blocks
    assert hit_lengths == (640, 640)


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
