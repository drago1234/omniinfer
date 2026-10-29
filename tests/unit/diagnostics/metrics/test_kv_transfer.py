# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT
"""Unit tests for omni_npu.diagnostics.metrics.kv_transfer（KVConnectorStats 通道，含 worker_mem 合并）。"""

import sys
import types

import pytest

pytest.importorskip("prometheus_client")
pytest.importorskip("vllm.distributed.kv_transfer.kv_connector.v1.metrics")

from omni_npu.diagnostics.metrics import kv_transfer, worker_mem
from prometheus_client import REGISTRY, Counter, Gauge, Histogram


def _unregister_omni():
    # build_prom_metrics 每次都在全局 REGISTRY 注册 omni_npu:* 指标；多用例复跑会
    # "Duplicated timeseries" 报错，故每例前后摘干净。
    for c in list(REGISTRY._collector_to_names):
        if getattr(c, "_name", "").startswith("omni_npu:"):
            REGISTRY.unregister(c)


@pytest.fixture(autouse=True)
def _clean(monkeypatch):
    with kv_transfer._lock:
        kv_transfer._counts.clear()
        kv_transfer._pd_pulls.clear()
    from omni_npu.diagnostics.metrics import stage_metrics

    stage_metrics.drain_unpublished()
    # 默认关掉显存采样（不注入 torch），逐用例按需打开
    monkeypatch.setattr(worker_mem, "_step", 0)
    _unregister_omni()
    yield
    _unregister_omni()
    with kv_transfer._lock:
        kv_transfer._counts.clear()
        kv_transfer._pd_pulls.clear()
    from omni_npu.diagnostics.metrics import stage_metrics

    stage_metrics.drain_unpublished()


def _fake_torch(monkeypatch, alloc=111, reserved=222):
    m = types.ModuleType("torch")
    m.npu = types.SimpleNamespace(
        memory_allocated=lambda: alloc, memory_reserved=lambda: reserved
    )
    monkeypatch.setitem(sys.modules, "torch", m)


# ── 失败累加 + collect ──────────────────────────────────────────────────────
def test_collect_failures_only():
    kv_transfer.record_failure("pull")
    kv_transfer.record_failure("pull")
    kv_transfer.record_failure("send_done")
    st = kv_transfer.collect(rank=None)  # rank=None → 不采显存
    assert st.data == {"fail": {"pull": 2, "send_done": 1}, "mem": {}}
    # drain 后清空
    assert kv_transfer.collect(rank=None) is None


def test_collect_empty_returns_none():
    assert kv_transfer.collect(rank=None) is None


def test_collect_mem_by_rank(monkeypatch):
    _fake_torch(monkeypatch)
    monkeypatch.setattr(worker_mem, "SAMPLE_EVERY_N_STEPS", 1)
    st = kv_transfer.collect(rank=6)
    assert st.data == {"fail": {}, "mem": {"6": {"alloc": 111.0, "reserved": 222.0}}}


# ── aggregate：fail 求和，mem 并集 ─────────────────────────────────────────
def test_aggregate_fail_sum_mem_union():
    a = kv_transfer.build_stats(
        {"fail": {"pull": 1}, "mem": {"0": {"alloc": 10, "reserved": 11}}}
    )
    b = kv_transfer.build_stats(
        {
            "fail": {"pull": 2, "send_done": 5},
            "mem": {"1": {"alloc": 20, "reserved": 21}},
        }
    )
    m = a.aggregate(b)
    assert m.data["fail"] == {"pull": 3, "send_done": 5}  # 求和
    assert m.data["mem"] == {
        "0": {"alloc": 10, "reserved": 11},  # 并集，per-rank 保留
        "1": {"alloc": 20, "reserved": 21},
    }
    assert not m.is_empty()
    assert kv_transfer.build_stats({}).is_empty()


# ── 前端 observe：失败→Counter，显存→Gauge ────────────────────────────────
def test_observe_emits_failures_and_mem():
    vllm_config = type("C", (), {"kv_transfer_config": None})()
    mt = {Gauge: Gauge, Counter: Counter, Histogram: Histogram}
    prom = kv_transfer.build_prom_metrics(
        vllm_config, mt, ["model_name", "engine"], {0: ["m", "0"]}
    )
    prom.observe(
        {"fail": {"pull": 3}, "mem": {"6": {"alloc": 50.0, "reserved": 51.0}}},
        engine_idx=0,
    )

    assert (
        REGISTRY.get_sample_value(
            "omni_npu:kv_transfer_failures_total",
            {"model_name": "m", "engine": "0", "kind": "pull"},
        )
        == 3
    )
    assert (
        REGISTRY.get_sample_value(
            "omni_npu:worker_mem_allocated_bytes",
            {"model_name": "m", "engine": "0", "rank": "6"},
        )
        == 50.0
    )
    assert (
        REGISTRY.get_sample_value(
            "omni_npu:worker_mem_reserved_bytes",
            {"model_name": "m", "engine": "0", "rank": "6"},
        )
        == 51.0
    )


def test_observe_emits_ttft_stages():
    vllm_config = type("C", (), {"kv_transfer_config": None})()
    mt = {Gauge: Gauge, Counter: Counter, Histogram: Histogram}
    prom = kv_transfer.build_prom_metrics(
        vllm_config, mt, ["model_name", "engine"], {0: ["m", "0"]}
    )
    prom.observe(
        {
            "_omni_ttft": [
                {
                    "status": "completed",
                    "prompt_tokens": 130,
                    "n_lookups": 1,
                    "n_chunks": 1,
                    "external_tokens": 64,
                    "stages_ms": {
                        "proxy_to_p_arrive": 8.0,
                        "p_first_token_to_d_arrive": 12.0,
                        "arrive_to_first_token": 40.0,
                        "fwd_total": -1,
                    },
                }
            ]
        },
        engine_idx=0,
    )
    assert (
        REGISTRY.get_sample_value(
            "omni_npu:ttft_stage_requests_total",
            {
                "model_name": "m",
                "engine": "0",
                "instance_role": "unknown",
                "outcome": "completed",
            },
        )
        == 1
    )
    assert (
        REGISTRY.get_sample_value(
            "omni_npu:ttft_stage_milliseconds_count",
            {
                "model_name": "m",
                "engine": "0",
                "instance_role": "unknown",
                "outcome": "completed",
                "stage": "p_first_token_to_d_arrive",
                "kv_source": "unknown",
            },
        )
        == 1
    )
    assert (
        REGISTRY.get_sample_value(
            "omni_npu:ttft_stage_milliseconds_count",
            {
                "model_name": "m",
                "engine": "0",
                "instance_role": "unknown",
                "outcome": "completed",
                "stage": "proxy_to_p_arrive",
                "kv_source": "unknown",
            },
        )
        == 1
    )
    assert (
        REGISTRY.get_sample_value(
            "omni_npu:ttft_request_info_count",
            {
                "model_name": "m",
                "engine": "0",
                "instance_role": "unknown",
                "outcome": "completed",
                "field": "prompt_tokens",
            },
        )
        == 1
    )


def test_observe_empty_or_unknown_engine_noop():
    vllm_config = type("C", (), {"kv_transfer_config": None})()
    mt = {Gauge: Gauge, Counter: Counter, Histogram: Histogram}
    prom = kv_transfer.build_prom_metrics(
        vllm_config, mt, ["model_name", "engine"], {0: ["m", "0"]}
    )
    prom.observe({}, engine_idx=0)  # 空 data 不抛
    prom.observe({"fail": {"pull": 1}}, engine_idx=9)  # 未知 engine 不抛


# ── selftest 注入 ──────────────────────────────────────────────────────────
def test_selftest_injects(monkeypatch):
    monkeypatch.setattr(kv_transfer, "_SELFTEST", True)
    kv_transfer.maybe_selftest()
    st = kv_transfer.collect(rank=None)
    assert st.data["fail"] == {"selftest": 1}


def test_selftest_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(kv_transfer, "_SELFTEST", False)
    kv_transfer.maybe_selftest()
    assert kv_transfer.collect(rank=None) is None


def test_collect_pd_pulls():
    kv_transfer.record_pd_pull(0.12, num_blocks=4)
    st = kv_transfer.collect(rank=None)
    assert st.data["_omni_pd"] == [{"ms": 120.0, "num_blocks": 4}]
    assert kv_transfer.collect(rank=None) is None


def test_observe_emits_pd_load():
    vllm_config = type("C", (), {"kv_transfer_config": None})()
    mt = {Gauge: Gauge, Counter: Counter, Histogram: Histogram}
    prom = kv_transfer.build_prom_metrics(
        vllm_config, mt, ["model_name", "engine"], {0: ["m", "0"]}
    )
    prom.observe(
        {"_omni_pd": [{"ms": 120.0, "num_blocks": 4}]},
        engine_idx=0,
    )
    assert (
        REGISTRY.get_sample_value(
            "omni_npu:ttft_stage_milliseconds_count",
            {
                "model_name": "m",
                "engine": "0",
                "instance_role": "unknown",
                "outcome": "completed",
                "stage": "pd_load",
                "kv_source": "pd",
            },
        )
        == 1
    )
    assert (
        REGISTRY.get_sample_value(
            "omni_npu:ttft_request_info_sum",
            {
                "model_name": "m",
                "engine": "0",
                "instance_role": "unknown",
                "outcome": "completed",
                "field": "pd_blocks",
            },
        )
        == 4
    )


def test_install_pd_pull_hooks_records_duration():
    class DummyManager:
        def pull_blocks(self, addr, p_blocks, d_blocks, layer_ids):
            return True

    kv_transfer.install_pd_pull_hooks(DummyManager)
    assert DummyManager().pull_blocks("a", [1], [1, 2, 3], [0]) is True
    kv_transfer.install_pd_pull_hooks(DummyManager)
    st = kv_transfer.collect(rank=None)
    record = st.data["_omni_pd"][0]
    assert record["num_blocks"] == 3
    assert record["ms"] >= 0.0


def test_collect_drains_unpublished_ttft(monkeypatch):
    from omni_npu.diagnostics.metrics import stage_metrics

    monkeypatch.setattr(stage_metrics, "enabled", lambda: True)
    req = types.SimpleNamespace(request_id="r1", num_tokens=8)
    stage_metrics.record_arrival(req)
    stage_metrics.finish(req, connector=None)
    st = kv_transfer.collect(rank=None)
    assert st.data["_omni_ttft"][0]["request_id"] == "r1"
    assert kv_transfer.collect(rank=None) is None
