# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for the NaN/Inf forward-hook detector (CPU only)."""
import json
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from omni_npu.diagnostics.nan_inf import detector as det_mod
from omni_npu.diagnostics.nan_inf.detector import (
    MARKER,
    REPORT_ATTR,
    NanInfDetector,
    module_metadata,
    parse_bool,
    snapshot_metadata,
    tensor_stats,
)

pytestmark = pytest.mark.unit


class Poison(nn.Module):
    """Identity unless armed, then writes ``value`` into row 1 of its output."""

    def __init__(self, value=float("nan")):
        super().__init__()
        self.armed = False
        self.value = value

    def forward(self, x):
        y = x * 1.0
        if self.armed:
            y[1, 0] = self.value
        return y


class Tiny(nn.Module):
    def __init__(self, value=float("nan")):
        super().__init__()
        self.fc1 = nn.Linear(4, 4)
        self.poison = Poison(value)
        self.fc2 = nn.Linear(4, 4)

    def forward(self, x):
        return self.fc2(self.poison(self.fc1(x)))


def _run(detector, model, step=1, ctx=None, forward_context=None):
    checked = detector.should_check(forward_context)
    if checked:
        detector.begin_step(step, forward_context, ctx)
    with torch.no_grad():
        model(torch.randn(3, 4))
    if checked:
        return detector.end_step(forward_context)
    return None


@pytest.fixture
def model():
    torch.manual_seed(0)
    return Tiny()


class TestParse:
    @pytest.mark.parametrize("value,expected", [
        ("true", True), ("1", True), ("ON", True), (True, True),
        ("false", False), ("0", False), ("", False), (0, False),
    ])
    def test_parse_bool(self, value, expected):
        assert parse_bool(value) is expected

    def test_parse_bool_default_and_error(self):
        assert parse_bool(None, True) is True
        with pytest.raises(ValueError):
            parse_bool("maybe")


class TestTensorStats:
    def test_counts_and_finite_stats(self):
        t = torch.tensor([[1.0, float("nan")], [float("inf"), -float("inf")], [2.0, 3.0]])
        stats = tensor_stats(t)
        assert "shape" not in stats and "dtype" not in stats
        assert stats["nan_count"] == 1
        assert stats["posinf_count"] == 1
        assert stats["neginf_count"] == 1
        assert stats["finite_min"] == 1.0 and stats["finite_max"] == 3.0
        assert stats["bad_rows_head"] == [0, 1]

    def test_integer_tensor_has_no_stats(self):
        stats = tensor_stats(torch.arange(4))
        assert stats == {}


class TestModuleMetadata:
    def test_snapshots_attention_metadata_tensors(self):
        metadata = SimpleNamespace(
            num_tokens=2,
            slot_mapping=torch.tensor([3, 4]),
            nested=SimpleNamespace(causal=True),
        )
        forward_context = SimpleNamespace(
            attn_metadata={"poison.attn": metadata, "other.attn": {"skip": True}})
        result = module_metadata("poison", Poison(), forward_context)

        assert result["class"].endswith(".Poison")
        assert result["training"] is True
        attention = result["attention"]
        assert set(attention) == {"poison.attn"}
        assert attention["poison.attn"]["num_tokens"] == 2
        assert torch.equal(attention["poison.attn"]["slot_mapping"], torch.tensor([3, 4]))
        assert attention["poison.attn"]["nested"]["causal"] is True

    def test_metadata_snapshot_handles_cycles(self):
        cyclic = {}
        cyclic["self"] = cyclic
        snapshot = snapshot_metadata(cyclic)
        assert snapshot["self"]["__cycle__"] is True


class TestDetector:
    def test_disabled_never_checks(self, model):
        detector = NanInfDetector()
        model.poison.armed = True
        assert _run(detector, model) is None
        assert detector.num_checked_steps == 0

    def test_clean_step_no_report(self, model, tmp_path):
        detector = NanInfDetector()
        detector.configure(model, enable="true", dump_dir=str(tmp_path))
        assert detector.status()["hooked_modules"] == 4  # root + 3 children
        assert _run(detector, model) is None
        assert detector.num_checked_steps == 1
        assert not (tmp_path / "nan_inf").exists()

    def test_first_offending_module_dumped(self, model, tmp_path):
        detector = NanInfDetector(dp_rank=2)
        detector.configure(model, enable="true", dump_dir=str(tmp_path))
        model.poison.armed = True
        forward_context = SimpleNamespace(attn_metadata={
            "poison.attn": SimpleNamespace(
                num_tokens=3, slot_mapping=torch.tensor([1, 2, 3]))})
        pending = _run(
            detector, model, step=7, ctx={"req_ids": ["a", "b"]},
            forward_context=forward_context)
        assert pending["step"] == 7
        first = pending["detections"][0]
        assert first["module"] == "poison"
        assert first["dp_rank"] == 2

        files = sorted((tmp_path / "nan_inf").iterdir())
        assert [f.suffix for f in files] == [".pt"]
        report = torch.load(files[0])
        for key in ("step", "global_rank", "dp_rank", "tp_rank", "ep_rank",
                    "pp_rank", "cp_rank", "module", "tensors"):
            assert key in report
        assert report["req_ids"] == ["a", "b"]
        assert report["tensors"]["input.0"]["has_nan"] is False
        assert report["tensors"]["output.0"]["nan_count"] == 1
        assert report["tensors"]["output.0"]["bad_rows_head"] == [1]
        assert "shape" not in report["tensors"]["output.0"]
        assert "dtype" not in report["tensors"]["output.0"]
        assert "device" not in report["tensors"]["output.0"]
        assert "numel" not in report["tensors"]["output.0"]
        assert torch.isnan(report["tensors"]["output.0"]["value"]).sum() == 1
        assert report["module_metadata"]["class"].endswith(".Poison")
        assert report["module_metadata"]["training"] is True
        attention = report["module_metadata"]["attention"]["poison.attn"]
        assert attention["num_tokens"] == 3
        assert torch.equal(attention["slot_mapping"], torch.tensor([1, 2, 3]))

    def test_inf_detected(self, tmp_path):
        model = Tiny(value=float("inf"))
        detector = NanInfDetector()
        detector.configure(model, dump_dir=str(tmp_path))
        model.poison.armed = True
        pending = _run(detector, model)
        assert pending["detections"][0]["module"] == "poison"
        assert [f.suffix for f in (tmp_path / "nan_inf").iterdir()] == [".pt"]

    def test_containment_sanitizes_output_before_downstream_module(self, model, tmp_path):
        detector = NanInfDetector()
        detector.configure(model, dump_dir=str(tmp_path))
        observed = {}

        def observe(_module, args):
            observed["input"] = args[0].detach().clone()

        model.fc2.register_forward_pre_hook(observe)
        model.poison.armed = True
        pending = _run(detector, model)

        assert pending["detections"][0]["module"] == "poison"
        assert torch.isfinite(observed["input"]).all()
        report = torch.load(next((tmp_path / "nan_inf").glob("*.pt")))
        assert report["tensors"]["output.0"]["nan_count"] == 1

    def test_containment_sanitizes_injected_module_input(self, model, tmp_path):
        detector = NanInfDetector()
        detector.configure(model, dump_dir=str(tmp_path))

        def inject_before_detector(_module, args):
            return (torch.full_like(args[0], float("nan")),)

        model.fc1.register_forward_pre_hook(inject_before_detector, prepend=True)
        pending = _run(detector, model)

        assert pending["detections"][0]["module"] == "fc1"
        report = torch.load(next((tmp_path / "nan_inf").glob("*.pt")))
        assert report["tensors"]["input.0"]["nan_count"] == 12

    def test_containment_can_be_disabled(self, model, tmp_path):
        detector = NanInfDetector()
        detector.configure(
            model, dump_dir=str(tmp_path), contain_exceptional_values="false")
        observed = {}

        def observe(_module, args):
            observed["input"] = args[0].detach().clone()

        model.fc2.register_forward_pre_hook(observe)
        model.poison.armed = True
        _run(detector, model)

        assert detector.status()["contain_exceptional_values"] is False
        assert torch.isnan(observed["input"]).any()

    def test_max_dumps_limits_files_not_detection(self, model, tmp_path):
        detector = NanInfDetector()
        detector.configure(model, dump_dir=str(tmp_path), max_dumps="1")
        model.poison.armed = True
        _run(detector, model, step=1)
        pending = _run(detector, model, step=2)
        assert pending["detections"][0]["dump_file"] is None
        assert detector.num_detections == 2 and detector.num_dumps == 1

    def test_include_exclude(self, model, tmp_path):
        detector = NanInfDetector()
        detector.configure(model, dump_dir=str(tmp_path), include="fc")
        assert detector.status()["hooked_modules"] == 2
        model.poison.armed = True
        pending = _run(detector, model)
        # poison is not hooked, so fc2 sees the NaN on its input
        report = torch.load(next((tmp_path / "nan_inf").glob("*.pt")))
        assert pending["detections"][0]["module"] == "fc2"
        assert report["tensors"]["input.0"]["has_nan"] is True
        detector.configure(model, include="", exclude="poison|^Tiny$")
        assert detector.status()["hooked_modules"] == 2

    def test_disable_removes_hooks(self, model, tmp_path):
        detector = NanInfDetector()
        detector.configure(model, dump_dir=str(tmp_path))
        status = detector.configure(model, enable="false")
        assert status["enabled"] is False and status["hooked_modules"] == 0
        assert all(not m._forward_hooks for m in model.modules())

    def test_hooks_dormant_outside_step(self, model, tmp_path):
        detector = NanInfDetector()
        detector.configure(model, dump_dir=str(tmp_path))
        model.poison.armed = True
        with torch.no_grad():
            model(torch.randn(3, 4))  # e.g. a dummy run: not bracketed
        assert detector._hit is None and detector.num_dumps == 0

    def test_configure_requires_model(self):
        with pytest.raises(RuntimeError):
            NanInfDetector().configure(None, enable="true")

    def test_bad_option_leaves_state(self, model):
        detector = NanInfDetector()
        with pytest.raises(ValueError):
            detector.configure(model, max_dumps="x")
        assert detector.enabled is False and detector.max_dumps == 1


class TestForceEager:
    def _ctx(self, mode):
        from vllm.config import CUDAGraphMode
        return SimpleNamespace(cudagraph_runtime_mode=getattr(CUDAGraphMode, mode),
                               skip_compiled=False)

    def test_force_eager_overrides_and_restores(self, model, tmp_path):
        from vllm.config import CUDAGraphMode
        detector = NanInfDetector()
        detector.configure(model, dump_dir=str(tmp_path))
        ctx = self._ctx("FULL")
        seen = {}

        def spy(_m, _i, _o):
            seen["mode"] = ctx.cudagraph_runtime_mode
            seen["skip"] = ctx.skip_compiled
        model.fc1.register_forward_hook(spy)
        _run(detector, model, forward_context=ctx)
        assert seen == {"mode": CUDAGraphMode.NONE, "skip": True}
        assert ctx.cudagraph_runtime_mode == CUDAGraphMode.FULL
        assert ctx.skip_compiled is False

    def test_no_force_eager_skips_graph_steps(self, model, tmp_path):
        detector = NanInfDetector()
        detector.configure(model, dump_dir=str(tmp_path), force_eager="false")
        model.poison.armed = True
        assert _run(detector, model, forward_context=self._ctx("FULL")) is None
        assert detector.num_skipped_graph_steps == 1
        assert _run(detector, model, forward_context=self._ctx("NONE")) is not None


class TestAttach:
    def _output(self):
        from vllm.v1.outputs import ModelRunnerOutput
        return ModelRunnerOutput(req_ids=["a"], req_id_to_index={"a": 0})

    def test_attach_stamps_copy_and_clears(self):
        detector = NanInfDetector()
        detector._pending = {"step": 3, "detections": [{"module": "m"}]}
        assert detector.attach(None) is None
        assert detector._pending is not None  # deferred to sample_tokens
        out = self._output()
        stamped = detector.attach(out)
        assert getattr(stamped, REPORT_ATTR)["step"] == 3
        assert not hasattr(out, REPORT_ATTR)
        assert detector._pending is None
        assert detector.attach(self._output()).__dict__.get(REPORT_ATTR) is None

    def test_attach_never_mutates_empty_singleton(self):
        from vllm.v1.outputs import EMPTY_MODEL_RUNNER_OUTPUT
        detector = NanInfDetector()
        detector._pending = {"step": 1, "detections": []}
        stamped = detector.attach(EMPTY_MODEL_RUNNER_OUTPUT)
        assert stamped is not EMPTY_MODEL_RUNNER_OUTPUT
        assert not hasattr(EMPTY_MODEL_RUNNER_OUTPUT, REPORT_ATTR)

    def test_attach_async_output(self):
        from vllm.v1.outputs import AsyncModelRunnerOutput
        inner = self._output()

        class FakeAsync(AsyncModelRunnerOutput):
            def get_output(self):
                return inner

        detector = NanInfDetector()
        detector._pending = {"step": 5, "detections": []}
        async_out = detector.attach(FakeAsync())
        assert getattr(async_out.get_output(), REPORT_ATTR)["step"] == 5

    def test_report_survives_pickle(self):
        import pickle
        detector = NanInfDetector()
        detector._pending = {"step": 9, "detections": [{"module": "m"}]}
        stamped = detector.attach(self._output())
        restored = pickle.loads(pickle.dumps(stamped))
        assert getattr(restored, REPORT_ATTR)["step"] == 9


def test_marker_constant():
    assert MARKER == "ModelDegradation"
    assert det_mod.DUMP_SUBDIR == "nan_inf"


def _gather_worker(rank, port, result_path):
    import os
    import torch.distributed as dist
    from vllm.distributed import parallel_state as ps

    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    dist.init_process_group("gloo", init_method=f"tcp://127.0.0.1:{port}",
                            rank=rank, world_size=2)
    fake_tp = SimpleNamespace(world_size=2, cpu_group=dist.group.WORLD, rank_in_group=rank)
    ps.get_tp_group = lambda: fake_tp
    ps.get_pcp_group = lambda: SimpleNamespace(world_size=1)
    detector = NanInfDetector()
    detector.enabled = True
    results = []
    for step, bad_rank in ((1, None), (2, 1)):
        detector.begin_step(step)
        if rank == bad_rank:
            detector._hit = {"step": step, "module": f"m{rank}", "global_rank": rank}
        results.append(detector.end_step())
        detector._pending = None
    with open(f"{result_path}.{rank}", "w") as f:
        json.dump(results, f)
    dist.destroy_process_group()


def test_verdict_is_shared_across_tp_peers(tmp_path):
    import socket as _socket
    import torch.multiprocessing as mp

    with _socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        port = s.getsockname()[1]
    base = str(tmp_path / "res")
    mp.spawn(_gather_worker, args=(port, base), nprocs=2, join=True)
    for rank in (0, 1):
        clean, tripped = json.loads(open(f"{base}.{rank}").read())
        assert clean is None
        assert tripped["step"] == 2
        assert [d["module"] for d in tripped["detections"]] == ["m1"]