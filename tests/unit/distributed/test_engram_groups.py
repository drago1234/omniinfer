# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Shared-table groups must partition world ranks and respect node boundaries."""

from types import SimpleNamespace
from unittest.mock import Mock

import pytest

from omni_npu.v1.distributed import parallel_state_ext as state

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("shard_heads", [False, True])
def test_shared_group_topology_and_reinitialization(monkeypatch, shard_heads):
    world = SimpleNamespace(world_size=8, ranks=list(range(10, 18)), local_rank=2, rank_in_group=2, cpu_group=object())
    tp = [12, 13]
    memberships = (
        [[10, 12], [11, 13], [10, 12], [11, 13], [14, 16], [15, 17], [14, 16], [15, 17]]
        if shard_heads else [[10, 11, 12, 13]] * 4 + [[14, 15, 16, 17]] * 4
    )

    def gather(output, value, group):
        assert group is world.cpu_group
        output[:] = [[10 + 2 * (i // 2), 11 + 2 * (i // 2)] for i in range(8)] if value == tp else memberships

    monkeypatch.setattr(state, "_ENGRAM_SHARED", None)
    monkeypatch.setattr(state, "_get_world_group", lambda: world)
    monkeypatch.setattr(state, "get_dp_group", lambda: SimpleNamespace(ranks=[10, 12, 14, 16]))
    monkeypatch.setattr(state, "get_tp_group", lambda: SimpleNamespace(ranks=tp))
    monkeypatch.setattr(state.dist, "all_gather_object", gather)
    monkeypatch.setattr(state, "in_the_same_node_as", lambda group, source_rank: [
        i // 4 == source_rank // 4 for i in range(8)
    ])
    coordinator = Mock()
    monkeypatch.setattr(state, "GroupCoordinator", coordinator)
    shared = state.initialize_engram_shared_group(shard_heads)
    assert state.get_engram_shared_group() is shared
    assert state.initialize_engram_shared_group(shard_heads) is shared
    coordinator.assert_called_once_with(
        [list(m) for m in sorted(set(map(tuple, memberships)))], 2, "gloo",
        use_device_communicator=False, group_name="engram_shared",
    )
    monkeypatch.setattr(state, "_ENGRAM_SHARED", None)
    monkeypatch.setattr(state, "in_the_same_node_as", lambda *args, **kwargs: [False] * 8)
    with pytest.raises(RuntimeError, match="Could not determine"):
        state.initialize_engram_shared_group(shard_heads)
    monkeypatch.setattr(
        state.dist, "all_gather_object", lambda out, value, group: out.__setitem__(slice(None), [[10]] * 8)
    )
    with pytest.raises(RuntimeError, match="partition"):
        state._engram_group_from_local_ranks([10])
