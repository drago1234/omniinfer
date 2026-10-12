# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""ACL Graph utility control-flow tests for Model Runner V2."""

from contextlib import nullcontext
from importlib import import_module
from types import SimpleNamespace


def _aclgraph_utils():
    return import_module("omni_npu.worker.npu.aclgraph_utils")


def test_capture_window_restores_nested_state():
    cudagraph = _aclgraph_utils()

    assert cudagraph._capturing_default() is False
    with cudagraph.CaptureWindow():
        assert cudagraph._capturing_default() is True
        with cudagraph.CaptureWindow():
            assert cudagraph._capturing_default() is True
        assert cudagraph._capturing_default() is True
    assert cudagraph._capturing_default() is False


def test_capturing_descriptor_reads_thread_local_then_instance():
    """未赋值时回落到线程局部标志，赋过值以实例上的为准。"""
    cudagraph = _aclgraph_utils()
    descriptor = cudagraph._CapturingDescriptor()

    class _Ctx:
        capturing = descriptor

    assert _Ctx.capturing is descriptor  # obj 为 None 时返回描述符自身
    ctx = _Ctx()
    assert ctx.capturing is False
    with cudagraph.CaptureWindow():
        assert ctx.capturing is True
        ctx.capturing = 0
        assert ctx.capturing is False
    assert ctx.capturing is False


def test_refresh_acl_graph_tasks_updates_dynamic_entries(monkeypatch):
    cudagraph = _aclgraph_utils()
    calls = []

    class Event:
        def record(self, stream):
            calls.append(("record", stream))

    class OpDesc:
        workspace_fn = "workspace"

        def compute_dynamic_kwargs(self, context, layer_name, config):
            calls.append(("dynamic", context, layer_name, config))
            return {"input_layout": "BSND", "actual_seq_lengths": [3]}

        def op_out_fn(self, **kwargs):
            calls.append(("op_out", kwargs))

    entry = SimpleNamespace(
        op_desc=OpDesc(), captured_kwargs={"alpha": 1}, event=Event(),
        handle="handle", out_tensors="out",
    )
    manager = SimpleNamespace(
        _omni_acl_task_entries={4: {"layer": entry}},
        _omni_acl_workspaces={4: {"workspace": "ws"}},
    )
    context = SimpleNamespace(
        attn_metadata=object(), batch_descriptor=SimpleNamespace(num_tokens=4)
    )
    npu = cudagraph.torch.npu
    monkeypatch.setattr(npu, "stream", lambda stream: nullcontext())
    monkeypatch.setattr(npu, "graph_task_update_begin",
                        lambda stream, handle: calls.append(("begin", stream, handle)))
    monkeypatch.setattr(npu, "graph_task_update_end",
                        lambda stream: calls.append(("end", stream)))

    cudagraph.refresh_acl_graph_tasks(manager, context, "config", "update")

    assert calls[0][0] == "dynamic"
    assert calls[1] == ("begin", "update", "handle")
    assert calls[2][0] == "op_out"
    assert calls[2][1]["actual_seq_lengths"] is None
    assert calls[2][1]["workspace"] == "ws"
    assert calls[3] == ("end", "update")
    assert calls[4] == ("record", "update")


def test_refresh_acl_graph_tasks_skips_missing_metadata_and_static_entries(monkeypatch):
    cudagraph = _aclgraph_utils()
    called = []
    monkeypatch.setattr(cudagraph.torch.npu, "stream",
                        lambda stream: nullcontext())

    context = SimpleNamespace(attn_metadata=None,
                              batch_descriptor=SimpleNamespace(num_tokens=4))
    cudagraph.refresh_acl_graph_tasks(SimpleNamespace(), context, None, "stream")

    class Event:
        def record(self, stream):
            called.append(stream)

    entry = SimpleNamespace(
        op_desc=SimpleNamespace(compute_dynamic_kwargs=None), event=Event()
    )
    manager = SimpleNamespace(_omni_acl_task_entries={4: {"layer": entry}},
                              _omni_acl_workspaces={4: {}})
    context.attn_metadata = object()
    cudagraph.refresh_acl_graph_tasks(manager, context, None, "stream")
    assert called == ["stream"]


def test_run_fullgraph_wrapper_refreshes_after_upstream_call():
    cudagraph = _aclgraph_utils()
    calls = []

    def upstream(self, desc, marker=None):
        calls.append(("upstream", desc, marker))
        return "result"

    wrapped = cudagraph.build_run_fullgraph_with_acl_refresh(upstream)
    manager = SimpleNamespace(
        _omni_acl_context_provider=lambda desc: "context",
        _omni_acl_update_stream="stream",
        vllm_config="config",
    )
    original = cudagraph.refresh_acl_graph_tasks
    try:
        cudagraph.refresh_acl_graph_tasks = lambda *args: calls.append(("refresh", *args))
        assert wrapped(manager, "desc", marker="marker") == "result"
    finally:
        cudagraph.refresh_acl_graph_tasks = original

    assert calls == [("upstream", "desc", "marker"),
                     ("refresh", manager, "context", "config", "stream")]


def test_run_fullgraph_wrapper_does_not_refresh_without_runtime_hooks():
    cudagraph = _aclgraph_utils()
    wrapped = cudagraph.build_run_fullgraph_with_acl_refresh(lambda self, desc: "ok")
    assert wrapped(SimpleNamespace(vllm_config=None), "desc") == "ok"
