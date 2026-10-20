# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Unit tests for the scheduler/API ModelDegradation plumbing."""
from http import HTTPStatus
from types import SimpleNamespace

import pytest
import vllm.entrypoints.serve.utils.server_utils as server_utils
from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
from vllm.v1.request import RequestStatus

from omni_npu.diagnostics.nan_inf import degradation
from omni_npu.diagnostics.nan_inf.detector import MARKER, REPORT_ATTR
from omni_npu.vllm_patches.patches.common import patch_nan_inf_degradation

pytestmark = pytest.mark.unit

REPORT = {"step": 11, "detections": [
    {"module": "model.layers.3.mlp", "global_rank": 5, "dp_rank": 0, "tp_rank": 1}]}


class FakeRequest:
    def __init__(self, client_index=0):
        self.client_index = client_index
        self.status = RequestStatus.RUNNING

    def is_finished(self):
        return RequestStatus.is_finished(self.status)


class FakeKVManager:
    def __init__(self, blocks):
        self.blocks = blocks
        self.evicted = set()

    def get_block_ids(self, req_id):
        return self.blocks[req_id]

    def evict_blocks(self, block_ids):
        self.evicted |= set(block_ids)


class FakeScheduler:
    def __init__(self):
        self.requests = {"a": FakeRequest(0), "b": FakeRequest(1), "c": FakeRequest(0)}
        self.kv_cache_manager = FakeKVManager(
            {"a": ([1, 2], [7]), "b": ([3],), "c": ([4],)})
        self.finished = []

    def finish_requests(self, req_ids, status):
        for rid in req_ids:
            self.requests[rid].status = status
            self.finished.append((rid, status))


@pytest.fixture(autouse=True)
def _fresh_registry():
    degradation.reset_registry()
    yield
    degradation.reset_registry()


def test_scheduler_patch_chains_existing_update_from_output(monkeypatch):
    calls = []
    sentinel = object()

    def upstream(scheduler, scheduler_output, model_runner_output):
        calls.append((scheduler, scheduler_output, model_runner_output))
        return sentinel

    monkeypatch.setattr(
        patch_nan_inf_degradation.NanInfDegradationSchedulerPatch,
        "_upstream_update_from_output",
        upstream,
        raising=False,
    )
    scheduler = object()
    scheduler_output = object()
    model_runner_output = object()
    result = patch_nan_inf_degradation.NanInfDegradationSchedulerPatch.update_from_output(
        scheduler, scheduler_output, model_runner_output)

    assert result is sentinel
    assert calls == [(scheduler, scheduler_output, model_runner_output)]


def test_get_report_and_reason():
    out = SimpleNamespace()
    assert degradation.get_report(out) is None
    setattr(out, REPORT_ATTR, REPORT)
    assert degradation.get_report(out) is REPORT
    reason = degradation.format_stop_reason(REPORT)
    assert reason.startswith(MARKER)
    assert "model.layers.3.mlp" in reason and "step 11" in reason
    assert degradation.is_degradation_reason(reason)
    assert not degradation.is_degradation_reason(None)
    assert not degradation.is_degradation_reason(3)


def test_fail_degraded_batch_replaces_outputs():
    sched = FakeScheduler()
    sched_out = SimpleNamespace(num_scheduled_tokens={"a": 1, "b": 1, "gone": 1})
    clients, blocks = degradation.snapshot_batch(sched, sched_out)
    assert clients == {"a": 0, "b": 1}
    assert blocks == {1, 2, 3, 7}

    # the original update_from_output stopped "b" and emitted garbage for "a"
    sched.requests["b"].status = RequestStatus.FINISHED_STOPPED
    outputs = {
        0: EngineCoreOutputs(outputs=[
            EngineCoreOutput(request_id="a", new_token_ids=[42]),
            EngineCoreOutput(request_id="c", new_token_ids=[43]),
        ]),
    }
    result = degradation.fail_degraded_batch(sched, clients, blocks, outputs, REPORT)

    assert sched.finished == [("a", RequestStatus.FINISHED_ERROR)]
    assert sched.kv_cache_manager.evicted == {1, 2, 3, 7}
    by_id = {o.request_id: o for eco in result.values() for o in eco.outputs}
    assert set(by_id) == {"a", "b", "c"}
    for rid in ("a", "b"):
        assert by_id[rid].finish_reason == FinishReason.ERROR
        assert by_id[rid].new_token_ids == []
        assert degradation.is_degradation_reason(by_id[rid].stop_reason)
    assert by_id["c"].new_token_ids == [43] and by_id["c"].finish_reason is None
    assert 1 in result  # client 1 had no outputs this step


def test_record_and_lookup_by_external_id():
    reason = degradation.format_stop_reason(REPORT)
    processor = SimpleNamespace(request_states={
        "chatcmpl-1-abcd": SimpleNamespace(external_req_id="chatcmpl-1")})
    degradation.record_engine_outputs(processor, [EngineCoreOutputs(outputs=[
        EngineCoreOutput(request_id="chatcmpl-1-abcd", new_token_ids=[],
                         finish_reason=FinishReason.ERROR, stop_reason=reason),
        EngineCoreOutput(request_id="other", new_token_ids=[1]),
        EngineCoreOutput(request_id="unknown-state", new_token_ids=[],
                         finish_reason=FinishReason.ERROR, stop_reason=reason),
    ])])
    assert degradation.lookup_degraded("chatcmpl-1") == reason
    assert degradation.lookup_degraded("unknown-state") == reason
    assert degradation.lookup_degraded("other") is None


def test_lookup_matches_vllm_internal_request_suffix():
    reason = degradation.format_stop_reason(REPORT)
    processor = SimpleNamespace(request_states={})
    degradation.record_engine_outputs(processor, [EngineCoreOutputs(outputs=[
        EngineCoreOutput(request_id="cmpl-1-0-a1b2", new_token_ids=[],
                         finish_reason=FinishReason.ERROR, stop_reason=reason),
    ])])
    assert degradation.lookup_degraded("cmpl-1") == reason


def test_registry_is_bounded(monkeypatch):
    monkeypatch.setattr(degradation, "_REGISTRY_CAP", 2)
    processor = SimpleNamespace(request_states={})
    for i in range(3):
        degradation.record_engine_outputs(processor, [EngineCoreOutput(
            request_id=f"r{i}", new_token_ids=[], finish_reason=FinishReason.ERROR,
            stop_reason=MARKER)])
    assert degradation.lookup_degraded("r0") is None
    assert degradation.lookup_degraded("r2") == MARKER


class TestServingPatch:
    @pytest.fixture
    def patch_mod(self):
        mod = patch_nan_inf_degradation
        if mod.GenerateBaseServing is None:
            pytest.skip("serving layer unavailable")
        return mod

    def test_raise_if_error_model_degradation(self, patch_mod):
        serving_patch = patch_mod.NanInfDegradationServingPatch
        processor = SimpleNamespace(request_states={})
        degradation.record_engine_outputs(processor, [EngineCoreOutput(
            request_id="req-1", new_token_ids=[], finish_reason=FinishReason.ERROR,
            stop_reason=degradation.format_stop_reason(REPORT))])
        with pytest.raises(patch_mod.ModelDegradationError) as exc:
            serving_patch._raise_if_error(None, "error", "req-1")
        assert exc.value.status_code == HTTPStatus.INTERNAL_SERVER_ERROR
        assert isinstance(exc.value, patch_mod.GenerationError)
        # generic errors keep upstream behaviour
        with pytest.raises(patch_mod.GenerationError) as generic:
            serving_patch._raise_if_error(None, "error", "req-2")
        assert not isinstance(generic.value, patch_mod.ModelDegradationError)
        assert serving_patch._raise_if_error(None, "stop", "req-1") is None

    def test_error_response_type(self, patch_mod):
        err = patch_mod.ModelDegradationError(f"{MARKER}: boom")
        resp = patch_mod.create_error_response(err)
        assert resp.error.type == "ModelDegradation"
        assert resp.error.code == 500
        generic = patch_mod.create_error_response(patch_mod.GenerationError())
        assert generic.error.type == "InternalServerError"

    def test_streaming_error_type(self, patch_mod):
        import json
        serving_patch = patch_mod.NanInfDegradationServingPatch
        fake = SimpleNamespace(create_streaming_error_response=lambda msg, err_type, status_code:
                               json.dumps({"type": err_type, "code": int(status_code)}))
        body = json.loads(serving_patch._convert_generation_error_to_streaming_response(
            fake, patch_mod.ModelDegradationError("x")))
        assert body == {"type": "ModelDegradation", "code": 500}

    def test_rebind_create_error_response(self, patch_mod):
        patch_mod.NanInfDegradationErrorResponsePatch.apply()
        assert server_utils.create_error_response is patch_mod.create_error_response
