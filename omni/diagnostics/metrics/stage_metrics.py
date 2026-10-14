# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Request-stage latency collection.

This module owns request state, timestamp calculation, and the completed
record queue. Runtime integration lives in :mod:`stage_hooks`; transport and
Prometheus export live in :mod:`kv_transfer`.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Any

from omni_npu import envs

logger = logging.getLogger(__name__)

_STATE_ATTR = "_omni_ttft_stages"
_FWD_START_ATTR = "_omni_fwd_start_monotonic"
# Proxy arrival is carried in trace headers; P first-token time uses dedicated
# PD transfer metadata so both cross-process stages keep distinct semantics.
_PROXY_START_HEADERS = ("Start_time_ns", "TTFT_Start_time")
P_FIRST_TOKEN_PARAM = "omni_p_first_token_ns"
_NS_THRESHOLD = 1e15

_STAGE_INTERVALS = {
    "arrive_to_first_lookup": ("t_arrive", "t_first_lookup"),
    "first_lookup_to_lookup_done": ("t_first_lookup", "t_lookup_done"),
    "lookup_done_to_first_admit": ("t_lookup_done", "t_admit_first"),
    "admit_first_to_kv_ready": ("t_admit_first", "t_kv_ready"),
    "kv_ready_to_admit": ("t_kv_ready", "t_admit"),
    "fwd_wait_first": ("t_admit", "t_first_step_start"),
    "fwd_total": ("t_first_step_start", "t_prefill_done"),
    "done_to_first_token": ("t_prefill_done", "t_first_token"),
    "worker_to_scheduler": ("t_prefill_done", "t_output_seen"),
    "scheduler_to_token": ("t_output_seen", "t_first_token"),
    "arrive_to_admit": ("t_arrive", "t_admit"),
    "admit_to_first_token": ("t_admit", "t_first_token"),
    "arrive_to_first_token": ("t_arrive", "t_first_token"),
}

_unpublished: list[dict[str, Any]] = []
_unpublished_lock = threading.Lock()


def enabled() -> bool:
    return envs.OMNI_ENABLE_TTFT_STAGE_METRICS


def _state(request: Any, create: bool = True) -> dict[str, Any] | None:
    state = getattr(request, _STATE_ATTR, None)
    if state is None and create:
        state = {}
        setattr(request, _STATE_ATTR, state)
    return state


def _header_value(headers: Any, name: str) -> str | None:
    if not headers:
        return None
    try:
        items = headers.items()
    except (AttributeError, TypeError):
        return None
    target = name.lower()
    for key, value in items:
        if str(key).lower() == target and value:
            return str(value)
    return None


def copy_proxy_start_headers(headers: Any) -> dict[str, str]:
    """Keep Proxy wall-clock headers under their canonical names."""
    return {
        name: value
        for name in _PROXY_START_HEADERS
        if (value := _header_value(headers, name)) is not None
    }


def _parse_wall_seconds(headers: Any, names: tuple[str, ...]) -> float | None:
    for name in names:
        raw = _header_value(headers, name)
        if raw is None:
            continue
        try:
            value = float(raw.strip())
        except (TypeError, ValueError):
            continue
        if value > 0:
            return value / 1e9 if value >= _NS_THRESHOLD else value
    return None


def parse_p_first_token_wall_seconds(kv_transfer_params: Any) -> float | None:
    """Parse P's first-token wall clock from PD transfer metadata."""
    return _parse_wall_seconds(kv_transfer_params, (P_FIRST_TOKEN_PARAM,))


def parse_proxy_start_wall_seconds(headers: Any) -> float | None:
    """Parse the Proxy request-arrival wall clock from trace headers."""
    return _parse_wall_seconds(headers, _PROXY_START_HEADERS)


def record_arrival(request: Any) -> None:
    if not enabled():
        return
    state = _state(request)
    if state is None:
        return
    state.setdefault("t_arrive", time.monotonic())
    state.setdefault("t_arrive_wall", time.time())
    state.setdefault("prompt_tokens", request.num_tokens)
    if envs.OMNI_PD_ROLE == "prefill":
        proxy_start_wall = parse_proxy_start_wall_seconds(
            getattr(request, "trace_headers", None)
        )
        if proxy_start_wall is not None:
            state.setdefault("t_proxy_start_wall", proxy_start_wall)
    elif envs.OMNI_PD_ROLE == "decode":
        p_first_token_wall = parse_p_first_token_wall_seconds(
            getattr(request, "kv_transfer_params", None)
        )
        if p_first_token_wall is not None:
            state.setdefault("t_p_first_token_wall", p_first_token_wall)


def record_lookup(request: Any, started: float, completed: float) -> None:
    state = _state(request, create=False)
    if state is None:
        return
    state.setdefault("t_first_lookup", started)
    state["t_lookup_done"] = completed
    state["lookup_seconds_total"] = (
        state.get("lookup_seconds_total", 0.0) + completed - started
    )
    state["n_lookups"] = state.get("n_lookups", 0) + 1


def _stamp(request: Any, key: str, when: float | None = None) -> None:
    state = _state(request, create=False)
    if state is not None:
        state.setdefault(key, time.monotonic() if when is None else when)


def record_admit_first(request: Any, when: float | None = None) -> None:
    _stamp(request, "t_admit_first", when)


def record_admit(request: Any, when: float | None = None) -> None:
    _stamp(request, "t_admit", when)
    state = _state(request, create=False)
    if state is not None and hasattr(request, "num_computed_tokens"):
        state["num_computed"] = request.num_computed_tokens


def record_kv_load_start(
    request: Any,
    source: str,
    external_tokens: int,
    when: float | None = None,
) -> None:
    state = _state(request, create=False)
    if state is None:
        return
    state.setdefault("t_kv_load_start", time.monotonic() if when is None else when)
    state["kv_source"] = source
    state["external_tokens"] = max(0, int(external_tokens))


def record_kv_ready(
    request: Any,
    when: float | None = None,
    *,
    failed: bool = False,
) -> None:
    _stamp(request, "t_kv_ready", when)
    state = _state(request, create=False)
    if state is not None:
        state["kv_load_status"] = "failed" if failed else "ok"


def _seconds(state: dict[str, Any], start: str, end: str) -> float:
    first, second = state.get(start), state.get(end)
    if first is None or second is None:
        return -1.0
    return max(0.0, float(second) - float(first))


def _milliseconds(state: dict[str, Any], start: str, end: str) -> float:
    seconds = _seconds(state, start, end)
    return -1.0 if seconds < 0 else seconds * 1000.0


def _wall_milliseconds(state: dict[str, Any], start: str, end: str) -> float:
    first, second = state.get(start), state.get(end)
    if first is None or second is None:
        return -1.0
    delta = float(second) - float(first)
    return delta * 1000.0 if delta >= 0 else -1.0


def infer_kv_source(state: dict[str, Any], stamps: dict[str, Any]) -> str:
    """Classify the first-token KV path, preferring connector evidence."""
    if source := state.get("kv_source"):
        return str(source)
    if stamps.get("t_pool_load_start") is not None:
        return "pool"
    prompt = int(state.get("prompt_tokens") or 0)
    computed = int(state.get("num_computed") or 0)
    if prompt > 0 and computed >= prompt:
        return "hbm"
    return "mixed" if computed > 0 else "compute"


def _stage_state(state: dict[str, Any]) -> dict[str, Any]:
    """Fill synchronous admit/load stamps for a consistent stage chain."""
    out = dict(state)
    admit = out.get("t_admit")
    admit_first = (
        out.get("t_admit_first")
        or out.get("t_kv_load_start")
        or out.get("t_pool_load_start")
        or admit
    )
    if admit_first is not None:
        out["t_admit_first"] = admit_first
    if out.get("t_kv_ready") is None:
        kv_ready = admit_first if admit_first is not None else admit
        if kv_ready is not None:
            out["t_kv_ready"] = kv_ready
    return out


def _build_record(
    request: Any, state: dict[str, Any], *, freed_early: bool
) -> dict[str, Any]:
    stamps = _stage_state(state)
    stages = {
        name: _milliseconds(stamps, start, end)
        for name, (start, end) in _STAGE_INTERVALS.items()
    }
    stages["proxy_to_p_arrive"] = _wall_milliseconds(
        stamps, "t_proxy_start_wall", "t_arrive_wall"
    )
    stages["p_first_token_to_d_arrive"] = _wall_milliseconds(
        stamps, "t_p_first_token_wall", "t_arrive_wall"
    )
    lookup_total = float(state.get("lookup_seconds_total", 0.0))
    pool_total = float(state.get("pool_lookup_seconds_total", 0.0))
    has_pool = "pool_lookup_seconds_total" in state or bool(
        state.get("pool_lookup_count")
    )
    stages.update(
        lookup_total_ms=(
            lookup_total * 1000.0 if lookup_total or state.get("n_lookups") else -1.0
        ),
        pool_lookup=pool_total * 1000.0 if has_pool else -1.0,
        pool_load=_milliseconds(stamps, "t_pool_load_start", "t_kv_ready"),
    )
    return {
        "duration_ms": max(0.0, stages["arrive_to_first_token"]),
        "external_tokens": int(state.get("external_tokens") or 0),
        "status": "freed_early" if freed_early else "completed",
        "kv_load_status": str(state.get("kv_load_status") or "not_applicable"),
        "stages_ms": stages,
        "prompt_tokens": int(state.get("prompt_tokens", -1)),
        "num_computed": int(state.get("num_computed", -1)),
        "n_chunks": int(state.get("n_chunks", 0)),
        "n_lookups": int(state.get("n_lookups", 0)),
        "lookup_total_ms": stages["lookup_total_ms"],
        "request_id": request.request_id,
        "arrive_wall_ms": int(state.get("t_arrive_wall", 0.0) * 1000),
        "kv_source": infer_kv_source(state, stamps),
    }


def drain_unpublished() -> list[dict[str, Any]]:
    """Take completed records waiting for ``kv_transfer.collect()``."""
    with _unpublished_lock:
        records = list(_unpublished)
        _unpublished.clear()
        return records


def finish(
    request: Any,
    connector: Any = None,
    *,
    freed_early: bool = False,
) -> dict[str, Any] | None:
    state = _state(request, create=False)
    if state is None:
        return None
    try:
        delattr(request, _STATE_ATTR)
    except AttributeError:
        pass
    record = _build_record(request, state, freed_early=freed_early)
    with _unpublished_lock:
        _unpublished.append(record)
    stages_ms = {
        name: round(value, 1) if value >= 0 else -1
        for name, value in record["stages_ms"].items()
    }
    logger.info(
        "[TTFT] req=%s outcome=%s kv_source=%s prompt_tokens=%d "
        "num_computed=%d n_chunks=%d n_lookups=%d external_tokens=%d "
        "kv_load_status=%s stages_ms=%s arrive_wall_ms=%d",
        request.request_id,
        record["status"],
        record["kv_source"],
        record["prompt_tokens"],
        record["num_computed"],
        record["n_chunks"],
        record["n_lookups"],
        record["external_tokens"],
        record["kv_load_status"],
        stages_ms,
        record["arrive_wall_ms"],
    )
    return record


def stamp_model_runner_output(output: Any, start: float, end: float) -> None:
    """Attach worker monotonic stamps. Extra attrs survive pickle IPC."""
    if output is None:
        return
    output = getattr(output, "_model_runner_output", output)
    try:
        output.forward_start_monotonic = start
        output.forward_end_monotonic = end
    except (AttributeError, TypeError):
        pass


def mark_forward_start(runner: Any) -> float | None:
    if not enabled():
        return None
    started = time.monotonic()
    setattr(runner, _FWD_START_ATTR, started)
    return started


def stamp_sync_forward(runner: Any, output: Any, started: float | None) -> None:
    if started is not None and not getattr(runner, "use_async_scheduling", False):
        stamp_model_runner_output(output, started, time.monotonic())


def stamp_async_forward(runner: Any, output: Any) -> None:
    if not enabled() or not getattr(runner, "use_async_scheduling", False):
        return
    start = getattr(runner, _FWD_START_ATTR, None) or time.monotonic()
    stamp_model_runner_output(output, start, time.monotonic())


def record_step(
    request: Any,
    connector: Any = None,
    *,
    forward_start: float | None,
    forward_end: float | None,
    produced_token: bool,
    output_seen: float | None = None,
) -> None:
    state = _state(request, create=False)
    if state is None:
        return
    if forward_start is not None:
        state.setdefault("t_first_step_start", forward_start)
    if hasattr(request, "num_computed_tokens"):
        state["num_computed"] = request.num_computed_tokens
    state["n_chunks"] = state.get("n_chunks", 0) + 1
    if not produced_token:
        return
    if envs.OMNI_PD_ROLE == "prefill" and "t_first_token" not in state:
        headers = dict(getattr(request, "trace_headers", None) or {})
        headers[P_FIRST_TOKEN_PARAM] = str(time.time_ns())
        request.trace_headers = headers
    now = time.monotonic()
    state["t_prefill_done"] = forward_end if forward_end is not None else now
    if output_seen is not None:
        state["t_output_seen"] = output_seen
    state["t_first_token"] = now
    finish(request)
