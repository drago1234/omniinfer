# SPDX-License-Identifier: Apache-2.0

from types import SimpleNamespace
from unittest.mock import Mock

import pytest


def _request_extra_args(request):
    params = getattr(request, "sampling_params", None)
    return getattr(params, "extra_args", None) or {}


def _load_patch(patch_env):
    copy_worker = Mock()
    position_worker = Mock()
    init_mrope_worker = Mock()
    init_pd_decode_mrope_worker = Mock()

    class ModelRunner:
        def _prepare_inputs(self, scheduler_output, num_scheduled_tokens):
            return "native-inputs"

        def _init_mrope_positions(self, req_state):
            return None

    patch_env.stub(
        "vllm.v1.outputs",
        ModelRunnerOutput=type("ModelRunnerOutput", (SimpleNamespace,), {}),
    )
    patch_env.stub(
        "omni_models.models.openpangu_vl.realtime_video.metadata",
        MROPE_DELTA_KEY="omni_realtime_video_mrope_delta",
        request_extra_args=_request_extra_args,
    )

    patch_env.stub(
        "omni_models.models.openpangu_vl.realtime_video.kv_worker",
        copy_worker=copy_worker,
        position_worker=position_worker,
        init_mrope_worker=init_mrope_worker,
        init_pd_decode_mrope_worker=init_pd_decode_mrope_worker,
    )
    patch_env.stub(
        "omni_npu.worker.npu_model_runner",
        NPUModelRunner=ModelRunner,
    )
    module = patch_env.load(
        "patch_realtime_npu", group="openpangu_vl_realtime_video"
    )
    return (
        module, copy_worker, position_worker,
        init_mrope_worker, init_pd_decode_mrope_worker,
    )


def test_model_runner_hooks_call_realtime_workers_and_native_methods(patch_env):
    (module, copy_worker, position_worker, init_mrope_worker,
     init_pd_decode_mrope_worker) = _load_patch(patch_env)
    patch = module.RealtimeKVModelRunnerPatch
    calls = []
    copy_worker.side_effect = lambda *_: calls.append("copy")
    patch._upstream_prepare_inputs = Mock(
        side_effect=lambda *_: calls.append("prepare") or "input-result"
    )
    position_worker.side_effect = lambda *_: calls.append("position")
    patch._upstream_init_mrope_positions = Mock()
    runner = SimpleNamespace()
    output = object()
    scheduled = object()

    assert patch._prepare_inputs(runner, output, scheduled) == "input-result"
    patch._init_mrope_positions(runner, "request")

    assert "execute_model" not in patch._attr_names_to_apply
    assert "execute_model" not in patch.__dict__
    assert "sample_tokens" in patch._attr_names_to_apply
    assert calls == ["copy", "prepare", "position"]
    copy_worker.assert_called_once_with(runner, output)
    position_worker.assert_called_once_with(runner, scheduled)
    init_mrope_worker.assert_called_once_with(runner, "request")
    init_pd_decode_mrope_worker.assert_called_once_with("request")
    patch._upstream_prepare_inputs.assert_called_once_with(
        runner, output, scheduled
    )
    patch._upstream_init_mrope_positions.assert_called_once_with(
        runner, "request"
    )


def test_decode_mrope_handoff_skips_image_grid_reconstruction(patch_env):
    (module, _, _, init_mrope_worker, init_pd_decode_mrope_worker) = _load_patch(
        patch_env
    )
    patch = module.RealtimeKVModelRunnerPatch
    patch._upstream_init_mrope_positions = Mock(
        side_effect=AssertionError("D must not require image grids")
    )
    request = SimpleNamespace(sampling_params=SimpleNamespace(extra_args={
        "kv_transfer_params": {"omni_realtime_video_mrope_delta": 43}
    }))

    patch._init_mrope_positions(SimpleNamespace(), request)

    patch._upstream_init_mrope_positions.assert_not_called()
    init_mrope_worker.assert_called_once()
    init_pd_decode_mrope_worker.assert_called_once_with(request)


@pytest.mark.parametrize("async_output", [False, True])
def test_sample_attaches_only_completed_request_deltas(patch_env, async_output):
    module, *_ = _load_patch(patch_env)
    patch = module.RealtimeKVModelRunnerPatch
    output = module.ModelRunnerOutput(req_ids=["ready", "no_delta"])
    result = SimpleNamespace(_model_runner_output=output) if async_output else output
    patch._upstream_sample_tokens = Mock(return_value=result)
    runner = SimpleNamespace(_realtime_pd_mrope_deltas={"ready": 43, "pending": 9})

    assert patch.sample_tokens(runner, "schedule", allowed=True) is result

    assert output._realtime_pd_mrope_deltas == {"ready": 43}
    assert runner._realtime_pd_mrope_deltas == {"pending": 9}
    patch._upstream_sample_tokens.assert_called_once_with(
        runner, "schedule", allowed=True
    )


@pytest.mark.parametrize("result", [None, "not a model output"])
def test_sample_without_model_output_keeps_pending_deltas(patch_env, result):
    module, *_ = _load_patch(patch_env)
    patch = module.RealtimeKVModelRunnerPatch
    patch._upstream_sample_tokens = Mock(return_value=result)
    runner = SimpleNamespace(_realtime_pd_mrope_deltas={"pending": 9})

    assert patch.sample_tokens(runner) is result
    assert runner._realtime_pd_mrope_deltas == {"pending": 9}


def test_sample_failure_keeps_pending_deltas(patch_env):
    module, *_ = _load_patch(patch_env)
    patch = module.RealtimeKVModelRunnerPatch
    patch._upstream_sample_tokens = Mock(side_effect=RuntimeError("sample failed"))
    runner = SimpleNamespace(_realtime_pd_mrope_deltas={"pending": 9})

    with pytest.raises(RuntimeError, match="sample failed"):
        patch.sample_tokens(runner)
    assert runner._realtime_pd_mrope_deltas == {"pending": 9}
