# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Runtime hooks for request-stage metrics."""

from __future__ import annotations

import importlib
import logging
import time
from functools import wraps
from typing import Any

from omni_npu.diagnostics.metrics import stage_metrics

logger = logging.getLogger(__name__)

_SCHED_GETTER_HOOK = "_omni_ttft_sched_getter"
_PLATFORM_CONFIG_HOOK = "_omni_ttft_platform_config"
_SERVING_HEADER_HOOK = "_omni_ttft_serving_headers"


def _safe_metric(call, *args, **kwargs) -> None:
    """Keep optional instrumentation out of the serving failure path."""
    try:
        call(*args, **kwargs)
    except Exception:
        logger.exception("request-stage metrics hook failed")


def _waiting_for_remote_kv(request: Any) -> bool:
    status = getattr(request, "status", None)
    return (getattr(status, "name", None) or str(status)) == "WAITING_FOR_REMOTE_KVS"


def record_schedule_result(scheduler: Any, output: Any) -> None:
    """Record the first remote-wait or running admission after schedule()."""
    now = time.monotonic()
    requests = getattr(scheduler, "requests", None) or {}
    candidates = getattr(scheduler, "_inflight_prefills", None)
    if candidates is None:
        candidates = requests.values()
    for request in candidates:
        if _waiting_for_remote_kv(request):
            stage_metrics.record_admit_first(request, now)
            state = getattr(request, stage_metrics._STATE_ATTR, None) or {}
            if not state.get("kv_source"):
                prefill_stats = getattr(request, "prefill_stats", None)
                external_tokens = getattr(
                    prefill_stats, "num_external_cached_tokens", 0
                )
                stage_metrics.record_kv_load_start(
                    request,
                    source="pd",
                    external_tokens=external_tokens,
                    when=now,
                )
    for req_id in getattr(output, "num_scheduled_tokens", None) or {}:
        request = requests.get(req_id)
        if request is None:
            continue
        stage_metrics.record_admit_first(request, now)
        stage_metrics.record_admit(request, now)
        state = getattr(request, stage_metrics._STATE_ATTR, None) or {}
        if "t_kv_ready" not in state:
            stage_metrics.record_kv_ready(request, now)


def wrap_connector_lookup(connector: Any) -> None:
    if connector is None or getattr(connector, "_omni_ttft_lookup_wrapped", False):
        return
    original = connector.get_num_new_matched_tokens

    @wraps(original)
    def timed_lookup(request, num_computed_tokens):
        if not stage_metrics.enabled():
            return original(request, num_computed_tokens)
        started = time.monotonic()
        try:
            return original(request, num_computed_tokens)
        finally:
            _safe_metric(
                stage_metrics.record_lookup, request, started, time.monotonic()
            )

    connector.get_num_new_matched_tokens = timed_lookup
    connector._omni_ttft_lookup_wrapped = True


def record_update_from_output(
    scheduler: Any, scheduler_output: Any, model_runner_output: Any
) -> None:
    if not stage_metrics.enabled():
        return
    output_seen = time.monotonic()
    sampled = model_runner_output.sampled_token_ids
    forward_start = getattr(model_runner_output, "forward_start_monotonic", None)
    forward_end = getattr(model_runner_output, "forward_end_monotonic", None)
    for request_id in scheduler_output.num_scheduled_tokens:
        request = scheduler.requests.get(request_id)
        if request is None or request.is_finished():
            continue
        request_index = model_runner_output.req_id_to_index.get(request_id)
        if request_index is None:
            continue
        generated = sampled[request_index] if sampled else []
        stage_metrics.record_step(
            request,
            forward_start=forward_start,
            forward_end=forward_end,
            produced_token=bool(generated),
            output_seen=output_seen,
        )


def _carry_p_first_token_in_outputs(engine_core_outputs: Any) -> None:
    """Add P's first-token wall clock to existing PD transfer metadata."""
    if stage_metrics.envs.OMNI_PD_ROLE != "prefill":
        return
    for batch in getattr(engine_core_outputs, "values", lambda: ())():
        for output in getattr(batch, "outputs", ()):
            raw = stage_metrics._header_value(
                getattr(output, "trace_headers", None),
                stage_metrics.P_FIRST_TOKEN_PARAM,
            )
            params = getattr(output, "kv_transfer_params", None)
            if raw is not None and isinstance(params, dict):
                params[stage_metrics.P_FIRST_TOKEN_PARAM] = raw


def make_scheduler_cls(base_cls: type) -> type:
    """Subclass the configured scheduler without copying its schedule body."""
    if getattr(base_cls, "_omni_ttft_scheduler", False):
        return base_cls

    class TTFTScheduler(base_cls):
        _omni_ttft_scheduler = True

        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
            if stage_metrics.enabled():
                _safe_metric(wrap_connector_lookup, self.connector)

        def add_request(self, request):
            _safe_metric(stage_metrics.record_arrival, request)
            return super().add_request(request)

        def schedule(self, *args, **kwargs):
            output = super().schedule(*args, **kwargs)
            if stage_metrics.enabled():
                _safe_metric(record_schedule_result, self, output)
            return output

        def _update_waiting_for_remote_kv(self, request):
            if stage_metrics.enabled():
                failed_ids = getattr(self, "failed_recving_kv_req_ids", ())
                _safe_metric(
                    stage_metrics.record_kv_ready,
                    request,
                    failed=request.request_id in failed_ids,
                )
            return super()._update_waiting_for_remote_kv(request)

        def update_from_output(self, scheduler_output, model_runner_output):
            _safe_metric(
                record_update_from_output, self, scheduler_output, model_runner_output
            )
            outputs = super().update_from_output(scheduler_output, model_runner_output)
            _safe_metric(_carry_p_first_token_in_outputs, outputs)
            return outputs

        def _free_request(self, request, delay_free_blocks: bool = False):
            _safe_metric(stage_metrics.finish, request, freed_early=True)
            return super()._free_request(request, delay_free_blocks)

    TTFTScheduler.__name__ = f"TTFT{base_cls.__name__}"
    TTFTScheduler.__qualname__ = TTFTScheduler.__name__
    return TTFTScheduler


def install_scheduler_on_config(vllm_config: Any) -> None:
    """Install the stage scheduler directly on an existing config."""
    if not stage_metrics.enabled():
        return
    config = getattr(vllm_config, "scheduler_config", None)
    if config is None:
        return
    scheduler_cls = config.get_scheduler_cls()
    if not getattr(scheduler_cls, "_omni_ttft_scheduler", False):
        config.scheduler_cls = make_scheduler_cls(scheduler_cls)


def install_scheduler_getter_hook(scheduler_config_cls: type | None = None) -> None:
    """Wrap SchedulerConfig.get_scheduler_cls before EngineCore resolves it."""
    if scheduler_config_cls is None:
        try:
            from vllm.config.scheduler import SchedulerConfig

            scheduler_config_cls = SchedulerConfig
        except ImportError:
            return
    if getattr(scheduler_config_cls, _SCHED_GETTER_HOOK, False):
        return
    original = scheduler_config_cls.get_scheduler_cls

    def get_scheduler_cls(self):
        existing = getattr(self, "scheduler_cls", None)
        existing_is_wrapped = (
            existing is not None
            and not isinstance(existing, str)
            and getattr(existing, "_omni_ttft_scheduler", False)
        )
        if stage_metrics.enabled() and existing_is_wrapped:
            return existing
        cls = original(self)
        if not stage_metrics.enabled() or getattr(cls, "_omni_ttft_scheduler", False):
            return cls
        wrapped = make_scheduler_cls(cls)
        self.scheduler_cls = wrapped
        return wrapped

    scheduler_config_cls.get_scheduler_cls = get_scheduler_cls
    setattr(scheduler_config_cls, _SCHED_GETTER_HOOK, True)


def install_platform_config_hook() -> None:
    """Install a fallback for paths resolving the scheduler during config setup."""
    try:
        from omni_npu.platform import NPUPlatform
    except ImportError:
        return
    if getattr(NPUPlatform, _PLATFORM_CONFIG_HOOK, False):
        return
    original = NPUPlatform.check_and_update_config

    @classmethod
    def check_and_update_config(cls, vllm_config: Any) -> None:
        original(vllm_config)
        _safe_metric(install_scheduler_on_config, vllm_config)

    NPUPlatform.check_and_update_config = check_and_update_config
    setattr(NPUPlatform, _PLATFORM_CONFIG_HOOK, True)


def _wrap_serving_get_trace_headers(cls: type) -> None:
    if getattr(cls, _SERVING_HEADER_HOOK, False):
        return
    original = cls._get_trace_headers

    @wraps(original)
    async def _get_trace_headers(self, headers):
        result = await original(self, headers)
        try:
            extra = stage_metrics.copy_proxy_start_headers(headers)
        except Exception:
            logger.exception("failed to copy Proxy stage headers")
            return result
        if not extra:
            return result
        merged = dict(result or {})
        merged.update(extra)
        return merged

    cls._get_trace_headers = _get_trace_headers
    setattr(cls, _SERVING_HEADER_HOOK, True)


def install_serving_header_hook() -> None:
    """Forward Proxy wall-clock headers even when tracing is disabled."""
    if not stage_metrics.enabled():
        return
    for mod_name, cls_name in (
        ("vllm.entrypoints.generate.base.serving", "GenerateBaseServing"),
        ("vllm.entrypoints.pooling.base.serving", "PoolingBaseServing"),
        (
            "vllm.entrypoints.generate.generative_scoring.serving",
            "ServingGenerativeScoring",
        ),
    ):
        try:
            module = importlib.import_module(mod_name)
            _wrap_serving_get_trace_headers(getattr(module, cls_name))
        except (ImportError, AttributeError):
            continue
