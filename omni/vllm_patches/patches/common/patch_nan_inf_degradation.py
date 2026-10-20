# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""OMNI-NAN-INF: fail NaN/Inf-tripped batches with HTTP 500 ``ModelDegradation``.

The worker half (forward hooks toggled through ``/collective_rpc``) lives in
``omni_npu.diagnostics.nan_inf.detector`` and the allowlisted
``NPUWorker.collective_feature`` dispatcher.
It stamps a report on the tripped step's ``ModelRunnerOutput``; this module:

* engine: ``Scheduler.update_from_output`` finishes the step's requests with
  ``FINISHED_ERROR`` + a ``ModelDegradation`` stop_reason (vLLM's retryable
  per-request error, always mapped to 500), instead of emitting the
  garbage tokens.
* api: ``OutputProcessor.process_outputs`` records those external request ids;
  ``_raise_if_error`` then raises ``ModelDegradationError`` and the error body
  carries ``"type": "ModelDegradation"`` (streaming and non-streaming).

Without an enabled detector no report exists and every wrapper is a single
``getattr``/dict lookup on the hot path.
"""
import sys

from vllm.logger import init_logger
from vllm.v1.core.sched.scheduler import Scheduler
from vllm.v1.engine.output_processor import OutputProcessor

from omni_npu.diagnostics.nan_inf import degradation
from omni_npu.vllm_patches.core import VLLMPatch, register_patch

logger = init_logger(__name__)


# --------------------------------------------------------------------------
# engine side
# --------------------------------------------------------------------------

_orig_update_from_output = Scheduler.update_from_output


@register_patch("NanInfDegradationSchedulerPatch", Scheduler)
class NanInfDegradationSchedulerPatch(VLLMPatch):
    _attr_names_to_apply = ["update_from_output"]

    @classmethod
    def apply(cls):
        # SchedulerKvXferFinishedPatch owns this method in production. Preserve
        # that behavior and layer degradation handling over its final version.
        cls.apply_bypass_conflict("update_from_output")

    def update_from_output(self, scheduler_output, model_runner_output):
        report = degradation.get_report(model_runner_output)
        upstream = getattr(
            NanInfDegradationSchedulerPatch,
            "_upstream_update_from_output",
            _orig_update_from_output,
        )
        if report is None:
            return upstream(self, scheduler_output, model_runner_output)
        clients, blocks = degradation.snapshot_batch(self, scheduler_output)
        outputs = upstream(self, scheduler_output, model_runner_output)
        return degradation.fail_degraded_batch(self, clients, blocks, outputs, report)


# --------------------------------------------------------------------------
# api side
# --------------------------------------------------------------------------

_orig_process_outputs = OutputProcessor.process_outputs


@register_patch("NanInfDegradationOutputProcessorPatch", OutputProcessor)
class NanInfDegradationOutputProcessorPatch(VLLMPatch):
    _attr_names_to_apply = ["process_outputs"]

    def process_outputs(self, engine_core_outputs, *args, **kwargs):
        try:
            degradation.record_engine_outputs(self, engine_core_outputs)
        except Exception as record_error:  # noqa: BLE001
            logger.warning("omni-nan-inf: failed to record degraded outputs: %r",
                           record_error)
        return _orig_process_outputs(self, engine_core_outputs, *args, **kwargs)


try:
    from vllm.entrypoints.generate.base.serving import GenerateBaseServing
    from vllm.entrypoints.openai.engine.protocol import GenerationError
    import vllm.entrypoints.serve.utils.error_response as _error_response_mod
except Exception as import_error:  # noqa: BLE001
    GenerateBaseServing = None
    logger.warning("omni-nan-inf: serving layer unavailable, ModelDegradation "
                   "type falls back to InternalServerError: %r", import_error)

if GenerateBaseServing is not None:

    class ModelDegradationError(GenerationError):
        """500 raised when a request was failed by the NaN/Inf detector."""

        error_type = degradation.ERROR_TYPE

    _orig_raise_if_error = GenerateBaseServing._raise_if_error
    _orig_convert_streaming = (
        GenerateBaseServing._convert_generation_error_to_streaming_response)

    @register_patch("NanInfDegradationServingPatch", GenerateBaseServing)
    class NanInfDegradationServingPatch(VLLMPatch):
        _attr_names_to_apply = [
            "_raise_if_error",
            "_convert_generation_error_to_streaming_response",
        ]

        def _raise_if_error(self, finish_reason, request_id):
            if finish_reason == "error":
                reason = degradation.lookup_degraded(request_id)
                if reason is not None:
                    logger.error("Request %s failed: %s", request_id, reason)
                    raise ModelDegradationError(reason)
            return _orig_raise_if_error(self, finish_reason, request_id)

        def _convert_generation_error_to_streaming_response(self, error):
            if isinstance(error, ModelDegradationError):
                return self.create_streaming_error_response(
                    str(error), err_type=ModelDegradationError.error_type,
                    status_code=error.status_code)
            return _orig_convert_streaming(self, error)

    _orig_create_error_response = _error_response_mod.create_error_response

    def create_error_response(message, *args, **kwargs):
        response = _orig_create_error_response(message, *args, **kwargs)
        if isinstance(message, ModelDegradationError):
            response.error.type = ModelDegradationError.error_type
        return response

    @register_patch("NanInfDegradationErrorResponsePatch", _error_response_mod)
    class NanInfDegradationErrorResponsePatch(VLLMPatch):
        """Rebind create_error_response in every module that imported it by name.

        The FastAPI GenerationError handler (server_utils) calls the
        module-level function it imported, so patching the defining module
        alone is not enough.
        """

        @classmethod
        def apply(cls):
            _error_response_mod.create_error_response = create_error_response
            rebound = []
            for name, module in list(sys.modules.items()):
                if not name.startswith("vllm") or module is None:
                    continue
                if getattr(module, "create_error_response", None) is _orig_create_error_response:
                    setattr(module, "create_error_response", create_error_response)
                    rebound.append(name)
            logger.info("patch applied: %s => create_error_response in %s",
                        cls.__name__, rebound)
