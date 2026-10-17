# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Engine/API plumbing that turns a worker NaN/Inf report into HTTP 500.

engine: ``fail_degraded_batch`` finishes every request of the tripped step
    with ``FINISHED_ERROR`` and a ``stop_reason`` starting with ``MARKER``,
    replaces whatever (garbage) tokens the step produced for them, and evicts
    their KV blocks from the prefix cache so polluted KV is never reused.
api: ``record_engine_outputs`` remembers which external request ids were
    failed that way, so the serving layer can raise ``ModelDegradation``
    instead of the generic ``InternalServerError``.
"""
import logging
from collections import OrderedDict

from vllm.v1.engine import EngineCoreOutput, EngineCoreOutputs, FinishReason
from vllm.v1.request import RequestStatus

from omni_npu.diagnostics.nan_inf.detector import MARKER, REPORT_ATTR

logger = logging.getLogger(__name__)

ERROR_TYPE = MARKER
_REGISTRY_CAP = 4096
_degraded = OrderedDict()


def get_report(model_runner_output):
    return getattr(model_runner_output, REPORT_ATTR, None)


def format_stop_reason(report):
    report = report or {}
    detections = report.get("detections") or [{}]
    first = detections[0] or {}
    return (f"{MARKER}: NaN/Inf detected at step {report.get('step')} "
            f"in module '{first.get('module')}' "
            f"(global_rank={first.get('global_rank')}, "
            f"dp_rank={first.get('dp_rank')}, tp_rank={first.get('tp_rank')})")


def is_degradation_reason(stop_reason):
    return isinstance(stop_reason, str) and stop_reason.startswith(MARKER)


# --------------------------------------------------------------------------
# engine side (scheduler process)
# --------------------------------------------------------------------------

def snapshot_batch(scheduler, scheduler_output):
    """Capture client index and KV blocks of the step's requests.

    Must run before the original ``update_from_output``, which may free
    requests that stopped in this step.
    """
    clients, blocks = {}, set()
    for req_id in scheduler_output.num_scheduled_tokens:
        request = scheduler.requests.get(req_id)
        if request is None:
            continue
        clients[req_id] = request.client_index
        try:
            for group in scheduler.kv_cache_manager.get_block_ids(req_id):
                blocks.update(group)
        except Exception:  # noqa: BLE001
            pass
    return clients, blocks


def fail_degraded_batch(scheduler, clients, blocks, engine_core_outputs, report):
    reason = format_stop_reason(report)
    alive = [
        req_id for req_id in clients
        if (req := scheduler.requests.get(req_id)) is not None and not req.is_finished()
    ]
    if alive:
        scheduler.finish_requests(alive, RequestStatus.FINISHED_ERROR)
    if blocks:
        try:
            scheduler.kv_cache_manager.evict_blocks(blocks)
        except Exception as e:  # noqa: BLE001
            logger.warning("omni-nan-inf: prefix-cache eviction failed: %r", e)

    for req_id, client_index in clients.items():
        client_outputs = engine_core_outputs.get(client_index)
        if client_outputs is None:
            client_outputs = engine_core_outputs[client_index] = EngineCoreOutputs()
        client_outputs.outputs = [
            output for output in client_outputs.outputs
            if output.request_id != req_id]
        client_outputs.outputs.append(EngineCoreOutput(
            request_id=req_id,
            new_token_ids=[],
            finish_reason=FinishReason.ERROR,
            stop_reason=reason,
        ))
    logger.error("omni-nan-inf: failed %d request(s) with %s; report=%s",
                 len(clients), MARKER, report)
    return engine_core_outputs


# --------------------------------------------------------------------------
# api side (AsyncLLM / serving process)
# --------------------------------------------------------------------------

def record_engine_outputs(output_processor, engine_core_outputs):
    batches = (engine_core_outputs.values()
               if isinstance(engine_core_outputs, dict)
               else engine_core_outputs or ())
    for batch in batches:
        outputs = getattr(batch, "outputs", (batch,))
        for eco in outputs:
            stop_reason = getattr(eco, "stop_reason", None)
            if not is_degradation_reason(stop_reason):
                continue
            state = output_processor.request_states.get(eco.request_id)
            external_id = getattr(state, "external_req_id", None) or eco.request_id
            _degraded[external_id] = stop_reason
            _degraded.move_to_end(external_id)
            while len(_degraded) > _REGISTRY_CAP:
                _degraded.popitem(last=False)


def lookup_degraded(request_id):
    reason = _degraded.get(request_id)
    if reason is not None:
        return reason
    if not isinstance(request_id, str):
        return None
    prefix = f"{request_id}-"
    for candidate, candidate_reason in reversed(_degraded.items()):
        if candidate.startswith(prefix) or request_id.startswith(f"{candidate}-"):
            return candidate_reason
    return None


def reset_registry():
    _degraded.clear()
