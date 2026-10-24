# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU regression tests for redundant-expert EPLB routing and accounting."""
from types import SimpleNamespace

import pytest
import torch

pytestmark = pytest.mark.unit


@pytest.fixture
def eplb(load_patch):
    return load_patch("patch_eplb")


@pytest.fixture
def placement():
    return torch.tensor([[0, 4, 5], [1, -1, -1], [2, 6, -1], [3, -1, -1]]), torch.tensor([3, 1, 2, 1])


@pytest.mark.parametrize("rank", [0, 1, 3])
def test_replica_table_balances_valid_replicas(eplb, placement, rank):
    mapping, counts = placement
    table = eplb.build_expert_replica_routing_table(mapping, counts, rank)
    expected = torch.tensor([
        [mapping[expert, (row + rank + expert) % int(counts[expert])]
         for expert in range(4)]
        for row in range(eplb.EPLB_ROUTING_TABLE_ROWS)
    ], dtype=torch.int32)
    torch.testing.assert_close(table, expected)
    assert table.is_contiguous()


@pytest.mark.parametrize("mapping,counts,message", [
    (torch.zeros(2), torch.ones(2), "2D"),
    (torch.zeros(2, 1), torch.ones(2, 1), "1D"),
    (torch.zeros(2, 1), torch.ones(3), "dimensions"),
    (torch.zeros(2, 1), torch.tensor([1, 0]), "at least one"),
])
def test_replica_table_rejects_invalid_metadata(eplb, mapping, counts, message):
    with pytest.raises(ValueError, match=message):
        eplb.build_expert_replica_routing_table(mapping, counts, 0)


@pytest.mark.parametrize("rows", [0, 1, 1023, 1024, 1025, 2051])
@pytest.mark.parametrize("dtype", [torch.int32, torch.int64])
def test_mapping_wraps_table_for_long_batches(eplb, placement, rows, dtype):
    mapping, counts = placement
    table = eplb.build_expert_replica_routing_table(mapping, counts, 1)
    ids = (torch.arange(rows * 2).reshape(rows, 2) % 4).to(dtype)
    actual = eplb.map_to_physical(ids, table)
    expected = torch.empty_like(ids)
    for row in range(rows):
        for col in range(2):
            expert = int(ids[row, col])
            expected[row, col] = mapping[expert, (row % 1024 + 1 + expert) % int(counts[expert])]
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize("enabled,unpadded", [(True, 0), (True, 2), (True, 1025), (False, 1025)])
def test_record_load_excludes_padding_and_accumulates(eplb, placement, enabled, unpadded):
    mapping, counts = placement
    table = eplb.build_expert_replica_routing_table(mapping, counts, 0)
    ids = torch.arange(2050).reshape(1025, 2) % 4
    load = torch.ones(7, dtype=torch.int64)
    physical = eplb.map_to_physical_and_record(ids, table, load, torch.tensor(enabled), torch.tensor(unpadded))
    torch.testing.assert_close(physical, eplb.map_to_physical(ids, table))
    expected = torch.ones_like(load)
    if enabled:
        expected += torch.bincount(physical[:unpadded].flatten(), minlength=7)
    torch.testing.assert_close(load, expected)


@pytest.mark.parametrize("redundant", [False, True])
def test_refresh_preserves_graph_table_and_recording_contract(eplb, monkeypatch, placement, redundant):
    monkeypatch.setattr(eplb, "get_ep_group", lambda: SimpleNamespace(rank_in_group=1, world_size=2))
    mapping, counts = placement
    state = object.__new__(eplb.NPUEplbLayerState)
    state.has_redundant_experts = redundant
    state.logical_to_physical_map = mapping.clone()
    state.logical_replica_count = counts if redundant else torch.ones(4, dtype=torch.int64)
    state.expert_load_view = torch.zeros(8 if redundant else 4, dtype=torch.int32)
    state.local_load = state.expert_load_view[2:4]
    state.should_record_tensor = torch.tensor(True)
    state.num_unpadded_tokens_tensors = [torch.tensor(1), torch.tensor(2)]
    monkeypatch.setattr(eplb, "dbo_current_ubatch_id", lambda: 1)
    state.refresh_replica_table()
    attr = "expert_replica_routing_table" if redundant else "replica_table"
    original = getattr(state, attr)
    state.logical_to_physical_map = (mapping + 1).clamp_max(6)
    state.refresh_replica_table()
    assert getattr(state, attr) is original
    ids = torch.tensor([[0, 1], [2, 3], [0, 2]])
    physical = state.apply_mapping(ids)
    if redundant:
        expected = eplb.map_to_physical(ids, original)
        torch.testing.assert_close(physical, expected)
        torch.testing.assert_close(state.expert_load_view, torch.bincount(expected[:2].flatten(), minlength=8).int())
        before = state.expert_load_view.clone()
        state.record_load(torch.tensor([10, 20]))
        torch.testing.assert_close(state.expert_load_view, before)
    else:
        torch.testing.assert_close(physical, state.logical_to_physical_map[:, 0].int()[ids])
        state.record_load(torch.tensor([10, 20]))
        torch.testing.assert_close(state.expert_load_view, torch.tensor([0, 0, 10, 20], dtype=torch.int32))
