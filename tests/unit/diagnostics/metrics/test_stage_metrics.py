# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT
"""Unit tests for omni_npu.diagnostics.metrics.stage_metrics."""

from __future__ import annotations

import asyncio
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import pytest

_ROOT = next(
    p for p in Path(__file__).resolve().parents if (p / "omni" / "diagnostics").is_dir()
)
_OMNI = _ROOT / "omni"
if "omni_npu" not in sys.modules:
    pkg = types.ModuleType("omni_npu")
    pkg.__path__ = [str(_OMNI)]
    pkg.__file__ = str(_OMNI / "__init__.py")
    sys.modules["omni_npu"] = pkg

from omni_npu.diagnostics.metrics import stage_hooks, stage_metrics


@pytest.fixture(autouse=True)
def _reset(monkeypatch):
    stage_metrics.drain_unpublished()
    monkeypatch.setattr(stage_metrics, "enabled", lambda: True)
    yield
    stage_metrics.drain_unpublished()


def _request(req_id="req-1", num_tokens=16):
    return SimpleNamespace(
        request_id=req_id,
        num_tokens=num_tokens,
        num_computed_tokens=8,
    )


def test_disabled_records_no_state(monkeypatch):
    monkeypatch.setattr(stage_metrics, "enabled", lambda: False)
    req = _request()
    stage_metrics.record_arrival(req)
    assert getattr(req, stage_metrics._STATE_ATTR, None) is None
    assert stage_metrics.finish(req, connector=None) is None
    assert stage_metrics.drain_unpublished() == []


def test_parse_p_first_token_wall_seconds_uses_pd_metadata_only():
    ns = 1_700_000_000_123_000_000
    params = {stage_metrics.P_FIRST_TOKEN_PARAM: str(ns)}
    assert stage_metrics.parse_p_first_token_wall_seconds(params) == pytest.approx(
        1_700_000_000.123
    )
    assert stage_metrics.parse_p_first_token_wall_seconds(
        {stage_metrics.P_FIRST_TOKEN_PARAM: "bad", "Start_time_ns": str(ns)}
    ) is None
    assert stage_metrics.parse_p_first_token_wall_seconds({}) is None
    assert stage_metrics.parse_p_first_token_wall_seconds(None) is None


def test_proxy_to_p_arrive_from_wall_header(monkeypatch):
    monkeypatch.setattr(stage_metrics.envs, "OMNI_PD_ROLE", "prefill")
    monkeypatch.setattr(stage_metrics.time, "time", lambda: 1_700_000_000.250)
    req = _request()
    req.trace_headers = {
        "Start_time_ns": str(1_700_000_000_200_000_000),
        "TTFT_Start_time": str(1_700_000_000_000_000_000),
    }
    stage_metrics.record_arrival(req)
    stage_metrics.record_admit(req, 1.0)
    stage_metrics.record_step(
        req, connector=None, forward_start=1.1, forward_end=1.2, produced_token=True
    )
    stages = stage_metrics.drain_unpublished()[0]["stages_ms"]
    assert stages["proxy_to_p_arrive"] == pytest.approx(50.0)
    assert stages["p_first_token_to_d_arrive"] == -1.0


def test_p_first_token_to_d_arrive_from_wall_header(monkeypatch):
    monkeypatch.setattr(stage_metrics.envs, "OMNI_PD_ROLE", "decode")
    monkeypatch.setattr(stage_metrics.time, "time", lambda: 1_700_000_000.250)
    req = _request()
    req.kv_transfer_params = {
        stage_metrics.P_FIRST_TOKEN_PARAM: str(1_700_000_000_200_000_000)
    }
    stage_metrics.record_arrival(req)
    stage_metrics.record_admit(req, 1.0)
    stage_metrics.record_step(
        req, connector=None, forward_start=1.1, forward_end=1.2, produced_token=True
    )
    stages = stage_metrics.drain_unpublished()[0]["stages_ms"]
    assert stages["proxy_to_p_arrive"] == -1.0
    assert stages["p_first_token_to_d_arrive"] == pytest.approx(50.0)


def test_p_first_token_to_d_arrive_missing_header_is_sentinel(monkeypatch):
    monkeypatch.setattr(stage_metrics.envs, "OMNI_PD_ROLE", "decode")
    req = _request()
    req.trace_headers = {"TTFT_Start_time": "1700000000000000000"}
    stage_metrics.record_arrival(req)
    stage_metrics.record_admit(req, 1.0)
    stage_metrics.record_step(
        req, connector=None, forward_start=1.1, forward_end=1.2, produced_token=True
    )
    assert (
        stage_metrics.drain_unpublished()[0]["stages_ms"][
            "p_first_token_to_d_arrive"
        ]
        == -1.0
    )


def test_prefill_first_token_is_carried_in_pd_metadata(monkeypatch):
    monkeypatch.setattr(stage_metrics.envs, "OMNI_PD_ROLE", "prefill")
    monkeypatch.setattr(
        stage_metrics.time, "time_ns", lambda: 1_700_000_000_200_000_000
    )
    req = _request()
    stage_metrics.record_arrival(req)
    stage_metrics.record_step(
        req, connector=None, forward_start=1.1, forward_end=1.2, produced_token=True
    )
    headers = req.trace_headers
    output = SimpleNamespace(
        trace_headers=headers,
        kv_transfer_params={"remote_cluster_id": "0"},
    )
    outputs = {0: SimpleNamespace(outputs=[output])}
    stage_hooks._carry_p_first_token_in_outputs(outputs)
    assert output.kv_transfer_params[stage_metrics.P_FIRST_TOKEN_PARAM] == str(
        1_700_000_000_200_000_000
    )


def test_p_first_token_does_not_create_incomplete_pd_metadata(monkeypatch):
    monkeypatch.setattr(stage_metrics.envs, "OMNI_PD_ROLE", "prefill")
    output = SimpleNamespace(
        trace_headers={stage_metrics.P_FIRST_TOKEN_PARAM: "1"},
        kv_transfer_params=None,
    )
    outputs = {0: SimpleNamespace(outputs=[output])}
    stage_hooks._carry_p_first_token_in_outputs(outputs)
    assert output.kv_transfer_params is None


def test_serving_header_hook_returns_proxy_time_when_tracing_off():
    class DummyServing:
        async def _get_trace_headers(self, headers):
            return None

    stage_hooks._wrap_serving_get_trace_headers(DummyServing)
    headers = {
        "TTFT_Start_time": "1700000000000000000",
        "Start_time_ns": "1700000000000000000",
        "traceparent": "00-abc-def-01",
    }
    result = asyncio.run(DummyServing()._get_trace_headers(headers))
    assert result["TTFT_Start_time"] == "1700000000000000000"
    assert result["Start_time_ns"] == "1700000000000000000"
    first = DummyServing._get_trace_headers
    stage_hooks._wrap_serving_get_trace_headers(DummyServing)
    assert DummyServing._get_trace_headers is first


def test_serving_header_hook_merges_into_existing_trace_headers():
    class DummyServing:
        async def _get_trace_headers(self, headers):
            return {"traceparent": "00-abc-def-01"}

    stage_hooks._wrap_serving_get_trace_headers(DummyServing)
    result = asyncio.run(DummyServing()._get_trace_headers({"TTFT_Start_time": "1"}))
    assert result["traceparent"] == "00-abc-def-01"
    assert result["TTFT_Start_time"] == "1"


def test_stages_and_first_token_publish():
    req = _request()
    stage_metrics.record_arrival(req)
    stage_metrics.record_lookup(req, 1.0, 1.2)
    stage_metrics.record_admit(req, 10.0)
    stage_metrics.record_step(
        req,
        connector=None,
        forward_start=10.0,
        forward_end=10.5,
        produced_token=True,
    )
    records = stage_metrics.drain_unpublished()
    assert records
    record = records[0]
    assert record["status"] == "completed"
    assert record["n_lookups"] == 1
    assert record["n_chunks"] == 1
    assert record["stages_ms"]["first_lookup_to_lookup_done"] == pytest.approx(200.0)
    assert record["stages_ms"]["lookup_total_ms"] == pytest.approx(200.0)
    assert record["stages_ms"]["admit_first_to_kv_ready"] == pytest.approx(0.0)
    assert record["stages_ms"]["kv_ready_to_admit"] == pytest.approx(0.0)
    assert record["stages_ms"]["fwd_wait_first"] == pytest.approx(0.0)
    assert getattr(req, stage_metrics._STATE_ATTR, None) is None


def test_async_kv_load_window_splits_arrive_to_admit():
    req = _request()
    stage_metrics.record_arrival(req)
    getattr(req, stage_metrics._STATE_ATTR).update(
        {
            "t_arrive": 100.0,
            "t_arrive_wall": 0.0,
            "prompt_tokens": 130,
        }
    )
    stage_metrics.record_lookup(req, 100.3, 102.7)
    stage_metrics.record_admit_first(req, 107.9)
    stage_metrics.record_kv_ready(req, 362.4)
    stage_metrics.record_admit(req, 362.5)
    stage_metrics.record_step(
        req,
        connector=None,
        forward_start=363.0,
        forward_end=428.0,
        produced_token=True,
    )
    stages = stage_metrics.drain_unpublished()[0]["stages_ms"]
    lookup_chain = (
        stages["arrive_to_first_lookup"]
        + stages["first_lookup_to_lookup_done"]
        + stages["lookup_done_to_first_admit"]
    )
    assert stages["arrive_to_first_lookup"] == pytest.approx(300.0)
    assert stages["first_lookup_to_lookup_done"] == pytest.approx(2400.0)
    assert stages["lookup_done_to_first_admit"] == pytest.approx(5200.0)
    assert stages["admit_first_to_kv_ready"] == pytest.approx(254500.0)
    assert stages["kv_ready_to_admit"] == pytest.approx(100.0)
    assert stages["fwd_wait_first"] == pytest.approx(500.0)
    assert stages["arrive_to_admit"] == pytest.approx(262500.0)
    assert stages["arrive_to_admit"] == pytest.approx(
        lookup_chain + stages["admit_first_to_kv_ready"] + stages["kv_ready_to_admit"]
    )
    assert stages["pool_lookup"] == pytest.approx(-1.0)
    assert stages["pool_load"] == pytest.approx(-1.0)


def test_pool_lookup_and_load_stages():
    req = _request()
    stage_metrics.record_arrival(req)
    getattr(req, stage_metrics._STATE_ATTR).update(
        {
            "t_arrive": 1.0,
            "pool_lookup_seconds_total": 0.2,
            "pool_lookup_count": 1,
            "t_pool_load_start": 1.4,
            "t_kv_ready": 1.9,
            "kv_source": "pool",
            "external_tokens": 128,
        }
    )
    stage_metrics.record_admit(req, 1.95)
    stage_metrics.record_step(
        req,
        connector=None,
        forward_start=2.0,
        forward_end=2.1,
        produced_token=True,
    )
    record = stage_metrics.drain_unpublished()[0]
    stages = record["stages_ms"]
    assert stages["pool_lookup"] == pytest.approx(200.0)
    assert stages["pool_load"] == pytest.approx(500.0)
    assert stages["kv_ready_to_admit"] == pytest.approx(50.0)
    assert stages["fwd_wait_first"] == pytest.approx(50.0)
    assert record["kv_source"] == "pool"
    assert record["external_tokens"] == 128


def test_kv_source_hbm_when_fully_cached():
    req = _request(num_tokens=16)
    req.num_computed_tokens = 16
    stage_metrics.record_arrival(req)
    stage_metrics.record_admit(req, 1.0)
    stage_metrics.record_step(
        req, connector=None, forward_start=1.1, forward_end=1.2, produced_token=True
    )
    assert stage_metrics.drain_unpublished()[0]["kv_source"] == "hbm"


def test_kv_source_compute_when_nothing_cached():
    req = _request(num_tokens=16)
    req.num_computed_tokens = 0
    stage_metrics.record_arrival(req)
    stage_metrics.record_admit(req, 1.0)
    stage_metrics.record_step(
        req, connector=None, forward_start=1.1, forward_end=1.2, produced_token=True
    )
    assert stage_metrics.drain_unpublished()[0]["kv_source"] == "compute"


def test_kv_source_pd_when_remote_wait_without_pool():
    req = _request()
    stage_metrics.record_arrival(req)
    getattr(req, stage_metrics._STATE_ATTR).update(
        {
            "t_arrive": 1.0,
            "t_admit_first": 1.1,
            "t_kv_ready": 1.4,
            "t_admit": 1.41,
            "kv_source": "pd",
        }
    )
    stage_metrics.record_step(
        req, connector=None, forward_start=1.5, forward_end=1.6, produced_token=True
    )
    assert stage_metrics.drain_unpublished()[0]["kv_source"] == "pd"


def test_worker_to_scheduler_splits_done_to_first_token():
    req = _request()
    stage_metrics.record_arrival(req)
    stage_metrics.record_admit(req, 1.0)
    req.is_finished = lambda: False
    scheduler = SimpleNamespace(requests={"req-1": req}, connector=None)
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"req-1": 8})
    model_runner_output = SimpleNamespace(
        sampled_token_ids=[[42]],
        req_id_to_index={"req-1": 0},
        forward_start_monotonic=1.2,
        forward_end_monotonic=1.5,
    )
    stage_hooks.record_update_from_output(
        scheduler, scheduler_output, model_runner_output
    )
    stages = stage_metrics.drain_unpublished()[0]["stages_ms"]
    assert stages["fwd_total"] == pytest.approx(300.0)
    assert stages["done_to_first_token"] >= 0
    assert stages["worker_to_scheduler"] >= 0
    assert stages["scheduler_to_token"] >= 0
    assert stages["done_to_first_token"] == pytest.approx(
        stages["worker_to_scheduler"] + stages["scheduler_to_token"]
    )


def test_external_tokens_from_connector_state():
    req = _request()
    stage_metrics.record_arrival(req)
    getattr(req, stage_metrics._STATE_ATTR).update(
        {"external_tokens": 128, "t_admit": 1.0}
    )
    stage_metrics.record_step(
        req, connector=None, forward_start=1.1, forward_end=1.2, produced_token=True
    )
    record = stage_metrics.drain_unpublished()[0]
    assert record["external_tokens"] == 128


def test_schedule_result_stamps_waiting_for_remote_kv():
    req = _request()
    stage_metrics.record_arrival(req)
    req.status = SimpleNamespace(name="WAITING_FOR_REMOTE_KVS")
    stage_hooks.record_schedule_result(
        SimpleNamespace(requests={"req-1": req}),
        SimpleNamespace(num_scheduled_tokens={}),
    )
    assert getattr(req, stage_metrics._STATE_ATTR)["t_admit_first"] > 0


def test_schedule_result_stamps_first_running():
    req = _request()
    stage_metrics.record_arrival(req)
    req.status = SimpleNamespace(name="RUNNING")
    stage_hooks.record_schedule_result(
        SimpleNamespace(requests={"req-1": req}),
        SimpleNamespace(num_scheduled_tokens={"req-1": 8}),
    )
    state = getattr(req, stage_metrics._STATE_ATTR)
    assert "t_admit_first" in state
    assert "t_admit" in state
    assert "t_kv_ready" in state
    assert state["t_admit_first"] == state["t_admit"] == state["t_kv_ready"]


def test_chunked_prefill_does_not_finish_until_token():
    req = _request()
    stage_metrics.record_arrival(req)
    stage_metrics.record_step(
        req, connector=None, forward_start=1.0, forward_end=1.1, produced_token=False
    )
    assert getattr(req, stage_metrics._STATE_ATTR)["n_chunks"] == 1
    assert stage_metrics.drain_unpublished() == []


def test_freed_early_without_record_ttft_stashes():
    req = _request()
    stage_metrics.record_arrival(req)
    record = stage_metrics.finish(req, connector=None, freed_early=True)
    assert record["status"] == "freed_early"
    stashed = stage_metrics.drain_unpublished()
    assert stashed[0]["request_id"] == "req-1"


def test_stamp_model_runner_output():
    output = SimpleNamespace()
    stage_metrics.stamp_model_runner_output(output, 1.5, 2.5)
    assert output.forward_start_monotonic == 1.5
    assert output.forward_end_monotonic == 2.5
    stage_metrics.stamp_model_runner_output(None, 1.0, 2.0)


def test_stamp_async_model_runner_output():
    inner = SimpleNamespace()
    output = SimpleNamespace(_model_runner_output=inner)
    stage_metrics.stamp_model_runner_output(output, 1.5, 2.5)
    assert inner.forward_start_monotonic == 1.5
    assert inner.forward_end_monotonic == 2.5


def test_lookup_wrap_is_noop_when_disabled(monkeypatch):
    monkeypatch.setattr(stage_metrics, "enabled", lambda: False)
    calls = []

    class Connector:
        def get_num_new_matched_tokens(self, request, num_computed_tokens):
            calls.append((request, num_computed_tokens))
            return 0, False

    connector = Connector()
    stage_hooks.wrap_connector_lookup(connector)
    req = SimpleNamespace()
    connector.get_num_new_matched_tokens(req, 4)
    assert calls == [(req, 4)]


def test_lookup_wrap_records_when_enabled():
    req = _request()
    stage_metrics.record_arrival(req)

    class Connector:
        def get_num_new_matched_tokens(self, request, num_computed_tokens):
            return 2, True

    connector = Connector()
    stage_hooks.wrap_connector_lookup(connector)
    assert connector.get_num_new_matched_tokens(req, 0) == (2, True)
    assert getattr(req, stage_metrics._STATE_ATTR)["n_lookups"] == 1


def test_record_update_from_output_disabled_skips_hot_path(monkeypatch):
    monkeypatch.setattr(stage_metrics, "enabled", lambda: False)
    stepped = []
    monkeypatch.setattr(
        stage_metrics, "record_step", lambda *a, **k: stepped.append(True)
    )
    stage_hooks.record_update_from_output(SimpleNamespace(), object(), object())
    assert stepped == []


def test_record_update_from_output_records_first_token():
    req = _request()
    stage_metrics.record_arrival(req)
    req.is_finished = lambda: False
    scheduler = SimpleNamespace(requests={"req-1": req}, connector=object())
    scheduler_output = SimpleNamespace(num_scheduled_tokens={"req-1": 8})
    model_runner_output = SimpleNamespace(
        sampled_token_ids=[[42]],
        req_id_to_index={"req-1": 0},
        forward_start_monotonic=1.0,
        forward_end_monotonic=1.5,
    )
    stage_hooks.record_update_from_output(
        scheduler, scheduler_output, model_runner_output
    )
    records = stage_metrics.drain_unpublished()
    assert records[0]["status"] == "completed"


class _BaseScheduler:
    def __init__(self, connector=None):
        self.connector = connector
        self.requests = {}
        self.added = []
        self.freed = []

    def add_request(self, request):
        self.added.append(request)

    def schedule(self, *args, **kwargs):
        return SimpleNamespace(
            num_scheduled_tokens={req.request_id: 1 for req in self.requests.values()}
        )

    def _update_waiting_for_remote_kv(self, request):
        return None

    def update_from_output(self, scheduler_output, model_runner_output):
        return "upstream"

    def _free_request(self, request, delay_free_blocks=False):
        self.freed.append((request, delay_free_blocks))


def test_make_scheduler_cls_records_arrival_and_schedule():
    cls = stage_hooks.make_scheduler_cls(_BaseScheduler)
    sched = cls()
    req = _request()
    sched.add_request(req)
    assert getattr(req, stage_metrics._STATE_ATTR)["t_arrive"] > 0
    req.status = SimpleNamespace(name="RUNNING")
    sched.requests = {"req-1": req}
    sched.schedule()
    state = getattr(req, stage_metrics._STATE_ATTR)
    assert "t_admit" in state
    assert "t_kv_ready" in state


def test_record_schedule_result_marks_pd_wait():
    req = _request()
    stage_metrics.record_arrival(req)
    req.status = SimpleNamespace(name="WAITING_FOR_REMOTE_KVS")
    req.prefill_stats = SimpleNamespace(num_external_cached_tokens=12)
    scheduler = SimpleNamespace(requests={"req-1": req})
    output = SimpleNamespace(num_scheduled_tokens={})

    stage_hooks.record_schedule_result(scheduler, output)

    state = getattr(req, stage_metrics._STATE_ATTR)
    assert state["kv_source"] == "pd"
    assert state["external_tokens"] == 12
    assert state["t_kv_load_start"] == state["t_admit_first"]


def test_record_schedule_result_preserves_pool_source():
    req = _request()
    stage_metrics.record_arrival(req)
    req.status = SimpleNamespace(name="WAITING_FOR_REMOTE_KVS")
    req.prefill_stats = SimpleNamespace(num_external_cached_tokens=12)
    state = getattr(req, stage_metrics._STATE_ATTR)
    state.update(kv_source="pool", external_tokens=8, t_pool_load_start=1.0)

    stage_hooks.record_schedule_result(
        SimpleNamespace(requests={"req-1": req}),
        SimpleNamespace(num_scheduled_tokens={}),
    )

    assert state["kv_source"] == "pool"
    assert state["external_tokens"] == 8
    assert "t_kv_load_start" not in state


def test_make_scheduler_cls_is_idempotent():
    cls = stage_hooks.make_scheduler_cls(_BaseScheduler)
    assert stage_hooks.make_scheduler_cls(cls) is cls


def test_install_scheduler_on_existing_config(monkeypatch):
    config = SimpleNamespace(
        scheduler_cls=None, get_scheduler_cls=lambda: _BaseScheduler
    )
    vllm_config = SimpleNamespace(scheduler_config=config)
    stage_hooks.install_scheduler_on_config(vllm_config)
    assert issubclass(config.scheduler_cls, _BaseScheduler)

    monkeypatch.setattr(stage_metrics, "enabled", lambda: False)
    disabled = SimpleNamespace(
        scheduler_cls=None, get_scheduler_cls=lambda: _BaseScheduler
    )
    stage_hooks.install_scheduler_on_config(SimpleNamespace(scheduler_config=disabled))
    assert disabled.scheduler_cls is None


def test_install_scheduler_getter_hook_wraps_after_config(monkeypatch):
    class DummySchedulerConfig:
        def __init__(self):
            self.scheduler_cls = None

        def get_scheduler_cls(self):
            if self.scheduler_cls is not None:
                return self.scheduler_cls
            return _BaseScheduler

    monkeypatch.setattr(stage_metrics, "enabled", lambda: True)
    stage_hooks.install_scheduler_getter_hook(DummySchedulerConfig)
    cfg = DummySchedulerConfig()
    wrapped = cfg.get_scheduler_cls()
    assert issubclass(wrapped, _BaseScheduler)
    assert getattr(wrapped, "_omni_ttft_scheduler", False)
    assert cfg.scheduler_cls is wrapped
    stage_hooks.install_scheduler_getter_hook(DummySchedulerConfig)
    assert cfg.get_scheduler_cls() is wrapped

    monkeypatch.setattr(stage_metrics, "enabled", lambda: False)
    off = DummySchedulerConfig()
    assert off.get_scheduler_cls() is _BaseScheduler


def test_install_platform_config_hook(monkeypatch):
    calls = []

    class DummyPlatform:
        @classmethod
        def check_and_update_config(cls, vllm_config):
            calls.append(vllm_config)

    module = types.ModuleType("omni_npu.platform")
    module.NPUPlatform = DummyPlatform
    monkeypatch.setitem(sys.modules, "omni_npu.platform", module)

    config = SimpleNamespace(
        scheduler_cls=None, get_scheduler_cls=lambda: _BaseScheduler
    )
    vllm_config = SimpleNamespace(scheduler_config=config)
    stage_hooks.install_platform_config_hook()
    DummyPlatform.check_and_update_config(vllm_config)
    assert calls == [vllm_config]
    assert issubclass(config.scheduler_cls, _BaseScheduler)
