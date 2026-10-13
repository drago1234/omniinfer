# SPDX-License-Identifier: Apache-2.0
"""MRv2 NPU capture lifecycle, manager routing and real NPU replay."""
from contextlib import contextmanager, nullcontext
from types import SimpleNamespace

import pytest
import torch

from vllm.config.compilation import CUDAGraphMode as Mode
from vllm.v1.worker.gpu.cudagraph_utils import BatchExecutionDescriptor, CudaGraphManager
from omni_npu.worker.npu import aclgraph_manager as ag
from omni_npu.worker.npu.aclgraph_utils import _capturing_default


def _desc(mode=Mode.FULL, tokens=8):
    return BatchExecutionDescriptor(cg_mode=mode, num_tokens=tokens,
                                    num_reqs=1, uniform_token_count=tokens)


def _manager(descs):
    obj = object.__new__(ag.AclGraphManager)
    obj.device, obj.pool, obj.graphs = torch.device("npu:0"), None, {}
    obj._graphs_captured = False
    obj._capture_descs = {}
    for desc in descs:
        obj._capture_descs.setdefault(desc.cg_mode, []).append(desc)
    return obj


@pytest.fixture
def capture_env(monkeypatch):
    events = []
    params = SimpleNamespace(task_entries={}, workspaces={})
    monkeypatch.setattr(ag, "get_graph_params", lambda: params)
    monkeypatch.setattr(ag, "graph_capture", lambda **kw: nullcontext())
    monkeypatch.setattr(ag, "is_global_first_rank", lambda: False)
    monkeypatch.setattr(ag, "assert_no_acl_tasks_registered", lambda: events.append("guard"))
    monkeypatch.setattr(ag, "ensure_weak_ref_graph_params", lambda: events.append("weak-ref"))
    monkeypatch.setattr(ag, "get_offloader", lambda: SimpleNamespace(
        sync_prev_onload=lambda: events.append("sync"),
        join_after_forward=lambda: events.append("join")))
    return events, params


def test_full_capture_uses_npu_and_preserves_metadata_and_stream_order(monkeypatch, capture_env):
    events, params = capture_env
    desc = _desc(tokens=16)  # resolved size, not the raw configured size
    manager = _manager([desc])
    states = []

    @contextmanager
    def graph_context(graph, pool):
        events.append("enter")
        yield
        events.append("exit")

    monkeypatch.setattr(torch.npu, "NPUGraph", lambda: "npu-graph")
    monkeypatch.setattr(torch.npu, "graph", graph_context)
    monkeypatch.setattr(torch.cuda, "CUDAGraph", lambda: pytest.fail("CUDA graph called"))
    monkeypatch.setattr(torch.cuda, "graph", lambda *a, **kw: pytest.fail("CUDA context called"))

    def factory(desc, warmup):
        state = object()
        states.append(state)
        def forward(mode):
            assert mode == Mode.FULL
            assert _capturing_default() == (not warmup)
            events.append("warmup" if warmup else "forward")
        return forward, state

    result = manager.capture(factory)
    assert events == ["warmup", "sync", "enter", "forward", "join", "exit", "guard", "weak-ref"]
    assert result[desc].warmup is states[0]
    assert result[desc].captured is states[1]
    assert states[0] is not states[1]
    assert manager.graphs[desc] == "npu-graph"
    assert manager._graphs_captured and not _capturing_default()
    assert 16 in params.task_entries and 16 in params.workspaces


def test_piecewise_keeps_inner_wrapper_ownership(monkeypatch, capture_env):
    desc = _desc(Mode.PIECEWISE)
    manager = _manager([desc])
    monkeypatch.setattr(torch.npu, "NPUGraph", lambda: pytest.fail("outer graph for PIECEWISE"))
    seen = []
    state = object()
    result = manager.capture(lambda desc, warmup: (
        lambda mode: seen.append((mode, _capturing_default())), state))
    assert seen == [(Mode.NONE, False), (Mode.PIECEWISE, False)]
    assert result[desc].warmup is result[desc].captured is state
    assert manager.graphs == {}


def test_failed_capture_restores_flag_and_does_not_publish_graph(monkeypatch, capture_env):
    desc = _desc()
    manager = _manager([desc])
    monkeypatch.setattr(torch.npu, "NPUGraph", object)
    monkeypatch.setattr(torch.npu, "graph", lambda *a, **kw: nullcontext())
    def factory(desc, warmup):
        def forward(mode):
            if not warmup:
                raise RuntimeError("capture failed")
        return forward, None
    with pytest.raises(RuntimeError, match="capture failed"):
        manager.capture(factory)
    assert not manager.graphs and not manager._graphs_captured
    assert not _capturing_default()


def test_unsupported_acl_tasks_fail_before_publishing_graph(monkeypatch, capture_env):
    manager = _manager([_desc()])
    monkeypatch.setattr(torch.npu, "NPUGraph", object)
    monkeypatch.setattr(torch.npu, "graph", lambda *a, **kw: nullcontext())
    def unsupported():
        raise RuntimeError("requires ACL task refresh")
    monkeypatch.setattr(ag, "assert_no_acl_tasks_registered", unsupported)
    with pytest.raises(RuntimeError, match="requires ACL task refresh"):
        manager.capture(lambda desc, warmup: (lambda mode: None, None))
    assert not manager.graphs and not manager._graphs_captured
    assert not _capturing_default()


def test_runner_selects_managers_during_kv_initialization(monkeypatch):
    from vllm.v1.worker.gpu import model_runner
    from omni_npu.worker.npu.model_runner import NPUModelRunnerV2
    original = model_runner.ModelCudaGraphManager
    config = object()
    def initialize(self, cfg):
        assert cfg is config
        assert model_runner.ModelCudaGraphManager is ag.ModelAclGraphManager
        return "initialized"
    monkeypatch.setattr(model_runner.GPUModelRunner, "initialize_kv_cache", initialize)
    runner = object.__new__(NPUModelRunnerV2)
    assert runner.initialize_kv_cache(config) == "initialized"
    assert model_runner.ModelCudaGraphManager is original


def test_scoped_factories_restore_on_error_and_nested_entry():
    from vllm.v1.worker.gpu import model_runner
    from vllm.v1.worker.gpu.spec_decode.autoregressive import speculator
    before = (model_runner.ModelCudaGraphManager,
              speculator.PrefillSpeculatorCudaGraphManager,
              speculator.DecodeSpeculatorCudaGraphManager)
    with pytest.raises(RuntimeError, match="KV init"):
        with ag.graph_manager_wrapper():
            assert model_runner.ModelCudaGraphManager is ag.ModelAclGraphManager
            assert speculator.PrefillSpeculatorCudaGraphManager is ag.PrefillSpeculatorAclGraphManager
            assert speculator.DecodeSpeculatorCudaGraphManager is ag.DecodeSpeculatorAclGraphManager
            with ag.graph_manager_wrapper():
                pass
            assert model_runner.ModelCudaGraphManager is ag.ModelAclGraphManager
            raise RuntimeError("KV init")
    assert before == (model_runner.ModelCudaGraphManager,
                      speculator.PrefillSpeculatorCudaGraphManager,
                      speculator.DecodeSpeculatorCudaGraphManager)


@pytest.mark.parametrize("cls,base", [
    (ag.ModelAclGraphManager, ag.ModelCudaGraphManager),
    (ag.PrefillSpeculatorAclGraphManager, ag.PrefillSpeculatorCudaGraphManager),
    (ag.DecodeSpeculatorAclGraphManager, ag.DecodeSpeculatorCudaGraphManager),
])
def test_all_specialized_managers_route_to_npu_capture(cls, base):
    # The specialized forward/output routines are inherited unchanged; their
    # super().capture resolves to NPU capture rather than the CUDA base method.
    manager = object.__new__(cls)
    assert cls.capture is base.capture
    assert super(base, manager).capture.__func__ is ag.AclGraphManager.capture


def test_graph_param_sizes_extend_without_losing_existing_entries(monkeypatch):
    entry = object()
    params = SimpleNamespace(task_entries={8: {"layer": entry}}, workspaces={8: {}})
    monkeypatch.setattr(ag, "get_graph_params", lambda: params)
    ag._prepare_graph_params({Mode.FULL: [_desc(tokens=16), _desc(tokens=1)]})
    assert set(params.task_entries) == {1, 8, 16}
    assert params.task_entries[8]["layer"] is entry


@pytest.mark.skipif(not torch.npu.is_available(), reason="requires NPU")
def test_real_npu_graph_replay_uses_updated_static_inputs(capture_env, monkeypatch):
    # No distributed model weights needed: exercise the real capture/replay
    # backend with persistent buffers and multiple changing inputs.
    desc = _desc()
    manager = _manager([desc])
    stream = torch.npu.Stream(device=manager.device)
    monkeypatch.setattr(ag, "graph_capture", lambda **kw: torch.npu.stream(stream))
    monkeypatch.setattr(torch.cuda, "CUDAGraph", lambda: pytest.fail("CUDA graph called"))
    x = torch.arange(8, dtype=torch.float32, device=manager.device)
    y = torch.empty_like(x)
    torch.npu.synchronize()
    def factory(desc, warmup):
        return lambda mode: torch.mul(x, 3, out=y), None
    manager.capture(factory)
    monkeypatch.setattr("vllm.v1.worker.gpu.cudagraph_utils.get_offloader", ag.get_offloader)
    for offset in (1, 7, 19):
        x.copy_(torch.arange(8, dtype=torch.float32, device=manager.device) + offset)
        CudaGraphManager.run_fullgraph(manager, desc)
        torch.npu.synchronize()
        torch.testing.assert_close(y.cpu(), (torch.arange(8) + offset).float() * 3,
                                   rtol=0, atol=0)
    assert isinstance(manager.graphs[desc], torch.npu.NPUGraph)
