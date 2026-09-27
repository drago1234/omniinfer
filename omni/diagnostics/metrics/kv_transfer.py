# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Worker-to-frontend metrics over vLLM KVConnectorStats (no multiprocess).

One channel carries KV transfer failures and worker memory:

    {"fail": {kind: count}, "mem": {rank: {"alloc": b, "reserved": b}}}

Workers ``collect()`` into connector stats each step; the frontend
aggregates and ``observe``s in a single process.
"""

import threading
import time
from dataclasses import dataclass
from typing import Any

from omni_npu import envs
from omni_npu.diagnostics.metrics import stage_metrics, worker_mem
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
)
from vllm.logger import init_logger

logger = init_logger(__name__)

# Inject one synthetic failure at DecodeWorker start when
# OMNI_METRICS_KV_TRANSFER_SELFTEST=1 (smoke test).
_SELFTEST = envs.OMNI_METRICS_KV_TRANSFER_SELFTEST

_lock = threading.Lock()
_counts: dict[str, int] = {}  # kind -> failures since last drain
_pd_pulls: list[dict[str, Any]] = []

_TTFT_KEY = "_omni_ttft"
_PD_KEY = "_omni_pd"
_LIST_FIELDS = (_TTFT_KEY, _PD_KEY)
_TTFT_STAGE_BUCKETS = (1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2500, 5000, 10000)


def record_failure(kind: str = "pull", n: int = 1) -> None:
    # Called from connector transfer/guard threads. Memory-only; never raises.
    # kind: pull / send_done.
    try:
        with _lock:
            _counts[kind] = _counts.get(kind, 0) + n
    except Exception:  # noqa: BLE001, S110
        pass


def record_pd_pull(seconds: float, num_blocks: int = 0) -> None:
    """Record one low-level Decode PD pull operation."""
    try:
        with _lock:
            _pd_pulls.append(
                {
                    "ms": float(seconds) * 1000.0,
                    "num_blocks": int(num_blocks),
                }
            )
    except Exception:  # noqa: BLE001, S110
        pass


def _drain_failures() -> dict[str, int]:
    with _lock:
        if not _counts:
            return {}
        snap = dict(_counts)
        _counts.clear()
        return snap


def _drain_pd_pulls() -> list[dict[str, Any]]:
    with _lock:
        if not _pd_pulls:
            return []
        snap = list(_pd_pulls)
        _pd_pulls.clear()
        return snap


def maybe_selftest() -> None:
    if _SELFTEST:
        record_failure("selftest")
        logger.info(
            "kv_transfer selftest failure injected (kind=selftest); "
            'expect omni_npu:kv_transfer_failures{kind="selftest"} at frontend /metrics'
        )


@dataclass
class OmniConnectorStats(KVConnectorStats):
    def reset(self) -> None:
        self.data = {}

    def is_empty(self) -> bool:
        return not (
            self.data.get("fail")
            or self.data.get("mem")
            or any(self.data.get(key) for key in _LIST_FIELDS)
        )

    def aggregate(self, other: "KVConnectorStats") -> "OmniConnectorStats":
        # Sum failures by kind. Union mem by rank (each rank writes only itself).
        fail = dict(self.data.get("fail") or {})
        for k, v in (other.data.get("fail") or {}).items():
            fail[k] = fail.get(k, 0) + v
        mem = dict(self.data.get("mem") or {})
        mem.update(other.data.get("mem") or {})
        data: dict[str, Any] = {"fail": fail, "mem": mem}
        for key in _LIST_FIELDS:
            records = list(self.data.get(key) or [])
            records.extend(other.data.get(key) or [])
            if records:
                data[key] = records
        return OmniConnectorStats(data=data)

    def reduce(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            f"fail_{k}": v for k, v in (self.data.get("fail") or {}).items()
        }
        out["mem_ranks"] = len(self.data.get("mem") or {})
        return out


class OmniConnectorPromMetrics(KVConnectorPromMetrics):
    def __init__(self, vllm_config, metric_types, labelnames, per_engine_labelvalues):
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)
        ln = list(labelnames)
        kv_config = getattr(vllm_config, "kv_transfer_config", None)
        self._instance_role = {
            "kv_producer": "prefill",
            "kv_consumer": "decode",
        }.get(getattr(kv_config, "kv_role", None), "unknown")
        self._failures = self._counter_cls(
            name="omni_npu:kv_transfer_failures",
            documentation="PD KV transfer failures on decode side (by kind).",
            labelnames=ln + ["kind"],
        )
        self._mem_alloc = self._gauge_cls(
            name="omni_npu:worker_mem_allocated_bytes",
            documentation="Torch allocator allocated bytes per worker rank.",
            labelnames=ln + ["rank"],
        )
        self._mem_reserved = self._gauge_cls(
            name="omni_npu:worker_mem_reserved_bytes",
            documentation="Torch allocator reserved bytes per worker rank.",
            labelnames=ln + ["rank"],
        )
        ttft_labels = ln + ["instance_role", "outcome"]
        self._ttft_stage = self._histogram_cls(
            name="omni_npu:ttft_stage_milliseconds",
            documentation=(
                "Latency of each request stage contributing to TTFT, in "
                "milliseconds. Includes proxy_to_p_arrive and "
                "p_first_token_to_d_arrive."
            ),
            labelnames=ttft_labels + ["stage", "kv_source"],
            buckets=_TTFT_STAGE_BUCKETS,
        )
        self._ttft_requests = self._counter_cls(
            name="omni_npu:ttft_stage_requests_total",
            documentation="Requests carrying an Omni TTFT stage breakdown.",
            labelnames=ttft_labels,
        )
        self._ttft_info = self._histogram_cls(
            name="omni_npu:ttft_request_info",
            documentation=(
                "Per-request TTFT counters: prompt tokens, lookups, chunks, "
                "external tokens and PD blocks."
            ),
            labelnames=ttft_labels + ["field"],
        )
        logger.info(
            "omni connector prom metrics registered (KVConnectorStats channel): "
            "kv_transfer_failures + worker_mem + ttft_stage"
        )

    @staticmethod
    def _operation_records(payload: Any, operation: str):
        if hasattr(payload, "data"):
            payload = payload.data
        if not isinstance(payload, dict):
            return
        for record in payload.get(operation) or []:
            if isinstance(record, dict):
                yield record
        for value in payload.values():
            if isinstance(value, dict) or hasattr(value, "data"):
                yield from OmniConnectorPromMetrics._operation_records(value, operation)

    def _observe_stage_metrics(
        self, transfer_stats_data: dict[str, Any], labelvalues: list[object]
    ) -> None:
        for record in self._operation_records(transfer_stats_data, _TTFT_KEY):
            outcome = str(record.get("status", "unknown"))
            source = str(record.get("kv_source") or "unknown")
            metric_labels = [*labelvalues, self._instance_role, outcome]
            self._ttft_requests.labels(*metric_labels).inc()
            for stage, raw in (record.get("stages_ms") or {}).items():
                value = float(raw)
                if value >= 0:
                    self._ttft_stage.labels(*metric_labels, str(stage), source).observe(
                        value
                    )
            for field in (
                "prompt_tokens",
                "n_lookups",
                "n_chunks",
                "external_tokens",
            ):
                raw = record.get(field)
                if raw is not None and float(raw) >= 0:
                    self._ttft_info.labels(*metric_labels, field).observe(float(raw))

        for pull in transfer_stats_data.get(_PD_KEY) or []:
            duration_ms = float(pull.get("ms", -1))
            if duration_ms < 0:
                seconds = float(pull.get("seconds", -1))
                duration_ms = seconds * 1000.0 if seconds >= 0 else -1.0
            if duration_ms >= 0:
                self._ttft_stage.labels(
                    *labelvalues,
                    self._instance_role,
                    "completed",
                    "pd_load",
                    "pd",
                ).observe(duration_ms)
            num_blocks = pull.get("num_blocks")
            if num_blocks is not None and float(num_blocks) >= 0:
                self._ttft_info.labels(
                    *labelvalues,
                    self._instance_role,
                    "completed",
                    "pd_blocks",
                ).observe(float(num_blocks))

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0) -> None:
        lv = self.per_engine_labelvalues.get(engine_idx)
        if lv is None or not transfer_stats_data:
            return
        for kind, count in (transfer_stats_data.get("fail") or {}).items():
            if count:
                self._failures.labels(*lv, kind).inc(count)
        for rank, m in (transfer_stats_data.get("mem") or {}).items():
            self._mem_alloc.labels(*lv, rank).set(m.get("alloc", 0))
            self._mem_reserved.labels(*lv, rank).set(m.get("reserved", 0))
        try:
            self._observe_stage_metrics(transfer_stats_data, lv)
        except Exception:
            logger.exception("failed to observe Omni stage metrics")


# LLMDataDistConnector entry points.
def collect(rank: int | None = None) -> OmniConnectorStats | None:
    # Called by the worker each step via the connector. Pass rank only on
    # workers that should sample memory. Returns None when everything is empty.
    fail = _drain_failures()
    mem = worker_mem.maybe_sample(rank) if rank is not None else None
    unpublished = stage_metrics.drain_unpublished()
    pd_pulls = _drain_pd_pulls()
    if not any((fail, mem, unpublished, pd_pulls)):
        return None
    data: dict[str, Any] = {"fail": fail, "mem": mem or {}}
    if unpublished:
        data[_TTFT_KEY] = unpublished
    if pd_pulls:
        data[_PD_KEY] = pd_pulls
    return OmniConnectorStats(data=data)


def build_stats(data: dict[str, Any] | None = None) -> OmniConnectorStats:
    return OmniConnectorStats(data=dict(data) if data else {})


def build_prom_metrics(
    vllm_config, metric_types, labelnames, per_engine_labelvalues
) -> OmniConnectorPromMetrics:
    return OmniConnectorPromMetrics(
        vllm_config, metric_types, labelnames, per_engine_labelvalues
    )


_PD_PULL_HOOK = "_omni_pd_pull_wrapped"


def install_pd_pull_hooks(manager_cls: Any = None) -> None:
    """Time CacheManager pulls without changing connector business code."""
    if manager_cls is None:
        try:
            from omni_npu.connector.llmdatadist_manager_v1 import CacheManager
        except (ImportError, AttributeError):
            return
        manager_cls = CacheManager
    original = manager_cls.pull_blocks
    if getattr(original, _PD_PULL_HOOK, False):
        return

    def pull_blocks(self, addr, p_blocks, d_blocks, layer_ids):
        started = time.monotonic()
        try:
            return original(self, addr, p_blocks, d_blocks, layer_ids)
        finally:
            try:
                num_blocks = len(d_blocks or [])
            except Exception:  # noqa: BLE001
                num_blocks = 0
            record_pd_pull(time.monotonic() - started, num_blocks=num_blocks)

    setattr(pull_blocks, _PD_PULL_HOOK, True)
    manager_cls.pull_blocks = pull_blocks
