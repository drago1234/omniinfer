# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Core split for shared experts overlapping Dispatch/Combine on A5."""

from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

import pytest
import torch


@pytest.fixture
def shared_expert_module(monkeypatch):
    from omni_npu.layers.fused_moe import shared_expert

    limit_core_num = MagicMock(return_value="core_limit_ctx")
    monkeypatch.setattr(
        torch.npu,
        "npugraph_ex",
        SimpleNamespace(scope=SimpleNamespace(limit_core_num=limit_core_num)),
        raising=False,
    )
    return shared_expert, limit_core_num


def _configure(monkeypatch, module, graph_mode, on_a5):
    monkeypatch.setattr(
        module,
        "model_extra_config",
        SimpleNamespace(task_config=SimpleNamespace(graph_mode=graph_mode)),
    )
    monkeypatch.setattr(module, "on_ascend950", lambda: on_a5)


@pytest.mark.unit
def test_core_limit_gives_shared_experts_all_cube_cores_on_a5_acl_graph(
    monkeypatch, shared_expert_module
):
    module, limit_core_num = shared_expert_module
    _configure(monkeypatch, module, "acl_graph", True)

    assert module.shared_expert_core_limit(True) == "core_limit_ctx"
    limit_core_num.assert_called_once_with(32, 8)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("enabled", "graph_mode", "on_a5"),
    [
        (False, "acl_graph", True),
        (True, "eager", True),
        (True, "acl_graph", False),
    ],
    ids=["not_dispatch_combine", "not_acl_graph", "not_a5"],
)
def test_core_limit_is_noop_outside_a5_dispatch_combine_graph(
    monkeypatch, shared_expert_module, enabled, graph_mode, on_a5
):
    module, limit_core_num = shared_expert_module
    _configure(monkeypatch, module, graph_mode, on_a5)

    assert isinstance(module.shared_expert_core_limit(enabled), nullcontext)
    limit_core_num.assert_not_called()
