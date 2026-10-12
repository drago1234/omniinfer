# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU-set transaction and inheritance tests, without NPU initialization."""
import importlib.util
import json
import logging
import os
from pathlib import Path
import subprocess
import sys
from types import SimpleNamespace
from unittest.mock import patch

import pytest

SOURCE = Path(os.environ.get("OMNI_CPU_AFFINITY_TEST_SOURCE", Path(__file__).parents[3] / "omni/worker/cpu_affinity.py"))
spec = importlib.util.spec_from_file_location("affinity_under_test", SOURCE)
affinity = importlib.util.module_from_spec(spec)
spec.loader.exec_module(affinity)


@pytest.fixture(autouse=True)
def _capture_affinity_logs(caplog, monkeypatch):
    """Capture this logger directly, independent of global logging configuration."""
    monkeypatch.setattr(affinity.logger, "disabled", False)
    monkeypatch.setattr(affinity.logger, "propagate", False)
    added = caplog.handler not in affinity.logger.handlers
    if added:
        affinity.logger.addHandler(caplog.handler)
    try:
        with caplog.at_level(logging.WARNING, logger=affinity.logger.name):
            yield
    finally:
        if added:
            affinity.logger.removeHandler(caplog.handler)


def parallel_config(dp=1, tp=16, pp=1, nodes=1):
    return SimpleNamespace(data_parallel_size=dp, tensor_parallel_size=tp,
                           pipeline_parallel_size=pp, nnodes_within_dp=nodes)


def test_missing_configuration_does_not_query_or_change_affinity(monkeypatch):
    monkeypatch.delenv("OMNI_WORKER_CPUSET_MAP", raising=False)
    with patch.object(affinity.os, "sched_getaffinity") as get_affinity, \
         patch.object(affinity.os, "sched_setaffinity") as set_affinity:
        assert affinity.apply_worker_cpu_affinity(0, parallel_config(dp=32)) is None
    get_affinity.assert_not_called()
    set_affinity.assert_not_called()


def test_parse_cpu_list():
    assert affinity.parse_cpu_set("1, 3-5,3") == {1, 3, 4, 5}


@pytest.mark.parametrize("value", ["", "3-1", "-1", "1-2-3", "a", "1;echo", True, [1], "8192"])
def test_reject_invalid_cpu_lists(value):
    with pytest.raises(ValueError):
        affinity.parse_cpu_set(value)


@pytest.mark.parametrize("mapping,rank,dp", [('{"0":"0-1"}', 0, 32), ('[]', 0, 1), ('{"1":"0-1"}', 0, 1), ('bad JSON', 0, 1)])
def test_bad_configuration_does_not_change_affinity(monkeypatch, mapping, rank, dp):
    monkeypatch.setenv("OMNI_WORKER_CPUSET_MAP", mapping)
    with patch.object(affinity.os, "sched_setaffinity") as set_affinity:
        assert affinity.apply_worker_cpu_affinity(rank, parallel_config(dp=dp)) is None
    set_affinity.assert_not_called()


def fake_threads(monkeypatch, masks):
    monkeypatch.setattr(affinity.os, "getpid", lambda: 10)
    monkeypatch.setattr(affinity.Path, "iterdir", lambda _path: [SimpleNamespace(name=str(t)) for t in masks])
    monkeypatch.setattr(affinity.os, "sched_getaffinity", lambda tid: masks[tid or 10].copy())
    monkeypatch.setattr(affinity.os, "sched_setaffinity", lambda tid, cpus: masks.__setitem__(tid or 10, set(cpus)))


def test_bind_existing_threads_without_widening_prebound_masks(monkeypatch):
    monkeypatch.setenv("OMNI_WORKER_CPUSET_MAP", '{"0":"1-2"}')
    masks={10:{0,1,2,3},11:{0,1,2,3},12:{1},13:{3}}
    fake_threads(monkeypatch,masks)
    result=affinity.apply_worker_cpu_affinity(0,parallel_config())
    assert masks=={10:{1,2},11:{1,2},12:{1},13:{3}}
    assert result['preserved_prebound_threads']=={13:[3]}
    assert result['changed_threads']==2


def test_outside_container_cpuset_skipped_before_mutation(monkeypatch):
    monkeypatch.setenv("OMNI_WORKER_CPUSET_MAP", '{"0":"1-3"}')
    masks={10:{0,1,2}}
    fake_threads(monkeypatch,masks)
    assert affinity.apply_worker_cpu_affinity(0,parallel_config()) is None
    assert masks=={10:{0,1,2}}


def test_exited_thread_is_ignored(monkeypatch):
    monkeypatch.setenv("OMNI_WORKER_CPUSET_MAP", '{"0":"1"}')
    masks={10:{0,1},11:{0,1}}
    fake_threads(monkeypatch,masks)
    def getter(tid):
        if tid==11:
            raise ProcessLookupError
        return masks[tid or 10].copy()
    monkeypatch.setattr(affinity.os,"sched_getaffinity",getter)
    result=affinity.apply_worker_cpu_affinity(0,parallel_config())
    assert result['vanished_threads']==[11]
    assert masks[10]=={1}


def test_failed_binding_rolls_back_successful_changes(monkeypatch, caplog):
    monkeypatch.setenv("OMNI_WORKER_CPUSET_MAP", '{"0":"1"}')
    masks={10:{0,1,2},11:{0,1,2},12:{0,1,2}}
    fake_threads(monkeypatch,masks)
    def setter(tid,cpus):
        if tid==12:
            raise PermissionError('controlled failure')
        masks[tid or 10]=set(cpus)
    monkeypatch.setattr(affinity.os,"sched_setaffinity",setter)
    assert affinity.apply_worker_cpu_affinity(0,parallel_config()) is None
    assert all(mask=={0,1,2} for mask in masks.values())
    assert "continuing worker startup" in caplog.text


def test_auto_reproduces_the_320_cpu_tp16_experiment():
    nodes = [set(range(node * 40, (node + 1) * 40)) for node in range(8)]
    groups = affinity._build_auto_cpu_groups(nodes, 16)
    assert len(groups) == 16
    for rank, group in enumerate(groups):
        start = (rank // 2) * 40 + 2 + (rank % 2) * 19
        assert group == set(range(start, start + 19))
    assert sum(map(len, groups)) == len(set().union(*groups)) == 304


def test_auto_uses_actual_topology_and_balances_remainders():
    nodes = [{10, 12, 14, 16, 18, 20, 22}, {100, 102, 104, 106, 108, 110, 112}]
    assert affinity._build_auto_cpu_groups(nodes, 4) == [
        {14, 16, 18}, {20, 22}, {104, 106, 108}, {110, 112},
    ]
    assert affinity._build_auto_cpu_groups(nodes, 2) == [
        {14, 16, 18, 20, 22}, {104, 106, 108, 110, 112},
    ]


@pytest.mark.parametrize("nodes,tp", [([], 16), ([{0, 1, 2}], 4),
    ([{0, 1, 2}, {3, 4, 5}], 1), ([{0, 1, 2}, {3, 4, 5}], 3)])
def test_auto_skips_unsupported_or_exhausted_groups(nodes, tp):
    assert affinity._build_auto_cpu_groups(nodes, tp) is None


def test_read_topology_intersects_cpuset_and_orders_nodes(monkeypatch, tmp_path):
    for name, cpus in [("node10", "40-79"), ("node2", "0-39"),
                        ("node8", "80-99"), ("node9", "")]:
        node = tmp_path / name
        node.mkdir()
        (node / "cpulist").write_text(cpus)
    monkeypatch.setattr(affinity.Path, "glob", lambda *args: list(tmp_path.iterdir()))
    assert affinity._read_numa_cpu_sets({2, 3, 42, 43}) == [{2, 3}, {42, 43}]


@pytest.mark.parametrize("cpu_lists", [[], ["0-1"], ["0-3", "2-3"]])
def test_incomplete_or_overlapping_topology_skipped(monkeypatch, tmp_path, cpu_lists):
    for index, cpus in enumerate(cpu_lists):
        node = tmp_path / f"node{index}"
        node.mkdir()
        (node / "cpulist").write_text(cpus)
    monkeypatch.setattr(affinity.Path, "glob", lambda *args: list(tmp_path.iterdir()))
    assert affinity._read_numa_cpu_sets({0, 1, 2, 3}) is None


def test_auto_applies_group_and_retains_existing_thread_rules(monkeypatch):
    monkeypatch.setenv("OMNI_WORKER_CPUSET_MAP", "auto")
    masks = {10: set(range(8)), 11: set(range(8)), 12: {2}, 13: {0}}
    fake_threads(monkeypatch, masks)
    monkeypatch.setattr(affinity, "_read_numa_cpu_sets", lambda allowed: [allowed])
    result = affinity.apply_worker_cpu_affinity(0, parallel_config(tp=2))
    assert masks == {10: {2, 3, 4}, 11: {2, 3, 4}, 12: {2}, 13: {0}}
    assert result["mode"] == "auto"
    assert result["preserved_prebound_threads"] == {13: [0]}


@pytest.mark.parametrize("rank,tp,pp,nodes", [(0, None, 1, 1), (0, 0, 1, 1),
    (0, True, 1, 1), (-1, 2, 1, 1), (2, 2, 1, 1),
    (0, 2, 2, 1), (0, 2, 1, 2)])
def test_auto_bad_configuration_does_not_bind(monkeypatch, rank, tp, pp, nodes):
    monkeypatch.setenv("OMNI_WORKER_CPUSET_MAP", "auto")
    monkeypatch.delenv("NODE_IP_LIST", raising=False)
    with patch.object(affinity.os, "sched_setaffinity") as set_affinity:
        assert affinity.apply_worker_cpu_affinity(rank, parallel_config(tp=tp, pp=pp, nodes=nodes)) is None
    set_affinity.assert_not_called()


def test_auto_topology_failure_does_not_bind(monkeypatch):
    monkeypatch.setenv("OMNI_WORKER_CPUSET_MAP", "auto")
    masks = {10: {0, 1, 2}}
    fake_threads(monkeypatch, masks)
    monkeypatch.setattr(affinity, "_read_numa_cpu_sets", lambda allowed: [allowed])
    assert affinity.apply_worker_cpu_affinity(0, parallel_config(tp=4)) is None
    assert masks == {10: {0, 1, 2}}


def test_resolve_auto_only_selects_cpus_and_ignores_deployment_environment(monkeypatch):
    monkeypatch.setenv("NODE_IP_LIST", "192.0.2.1,192.0.2.2")
    masks = {10: set(range(8))}
    fake_threads(monkeypatch, masks)
    monkeypatch.setattr(affinity, "_read_numa_cpu_sets", lambda allowed: [allowed])
    with patch.object(affinity.os, "sched_setaffinity", side_effect=AssertionError):
        assert affinity._resolve_worker_cpu_set("auto", 0, parallel_config(tp=2)) == {2, 3, 4}
    assert masks == {10: set(range(8))}


def test_cross_host_auto_skipped_before_reading_topology(monkeypatch):
    monkeypatch.delenv("NODE_IP_LIST", raising=False)
    with patch.object(affinity.os, "sched_getaffinity") as get_affinity, \
         patch.object(affinity, "_read_numa_cpu_sets") as read_topology:
        assert affinity._resolve_worker_cpu_set("auto", 0, parallel_config(nodes=2)) is None
    get_affinity.assert_not_called()
    read_topology.assert_not_called()


def test_explicit_map_keeps_existing_pp_and_cross_host_behavior(monkeypatch):
    monkeypatch.setenv("OMNI_WORKER_CPUSET_MAP", '{"7":"1-2"}')
    masks = {10: {0, 1, 2, 3}}
    fake_threads(monkeypatch, masks)
    result = affinity.apply_worker_cpu_affinity(7, parallel_config(pp=2, nodes=2))
    assert masks[10] == {1, 2}
    assert result["worker_rank"] == 7
    assert result["mode"] == "explicit"


def test_auto_binding_failure_restores_all_changed_threads(monkeypatch):
    monkeypatch.setenv("OMNI_WORKER_CPUSET_MAP", "auto")
    original = set(range(8))
    masks = {10: original.copy(), 11: original.copy(), 12: original.copy()}
    fake_threads(monkeypatch, masks)
    monkeypatch.setattr(affinity, "_read_numa_cpu_sets", lambda allowed: [allowed])
    def setter(tid, cpus):
        if tid == 12:
            raise PermissionError("controlled failure")
        masks[tid or 10] = set(cpus)
    monkeypatch.setattr(affinity.os, "sched_setaffinity", setter)
    assert affinity.apply_worker_cpu_affinity(0, parallel_config(tp=2)) is None
    assert all(mask == original for mask in masks.values())


@pytest.mark.parametrize("cpu_list", ["", "3-1", "8192"])
def test_invalid_explicit_cpu_list_skips_binding(monkeypatch, cpu_list, caplog):
    monkeypatch.setenv("OMNI_WORKER_CPUSET_MAP", json.dumps({"0": cpu_list}))
    with patch.object(affinity.os, "sched_setaffinity") as set_affinity:
        assert affinity.apply_worker_cpu_affinity(0, parallel_config()) is None
    set_affinity.assert_not_called()
    assert "continuing worker startup" in caplog.text


@pytest.mark.parametrize("failure", [OSError("topology unreadable"), AttributeError("affinity unavailable")])
def test_unavailable_topology_or_affinity_skips_binding(monkeypatch, failure, caplog):
    monkeypatch.setenv("OMNI_WORKER_CPUSET_MAP", "auto")
    with patch.object(affinity.os, "sched_getaffinity", side_effect=failure), \
         patch.object(affinity.os, "sched_setaffinity") as set_affinity:
        assert affinity.apply_worker_cpu_affinity(0, parallel_config()) is None
    set_affinity.assert_not_called()
    assert str(failure) in caplog.text


def test_topology_read_failure_skips_binding(monkeypatch):
    monkeypatch.setenv("OMNI_WORKER_CPUSET_MAP", "auto")
    with patch.object(affinity.os, "sched_getaffinity", return_value={0, 1, 2, 3}), \
         patch.object(affinity.Path, "glob", side_effect=OSError("topology unreadable")), \
         patch.object(affinity.os, "sched_setaffinity") as set_affinity:
        assert affinity.apply_worker_cpu_affinity(0, parallel_config()) is None
    set_affinity.assert_not_called()


def test_caller_binding_failure_keeps_original_mask(monkeypatch):
    monkeypatch.setenv("OMNI_WORKER_CPUSET_MAP", '{"0":"1"}')
    masks = {10: {0, 1, 2}, 11: {0, 1, 2}}
    fake_threads(monkeypatch, masks)
    with patch.object(affinity.os, "sched_setaffinity", side_effect=PermissionError("denied")):
        assert affinity.apply_worker_cpu_affinity(0, parallel_config()) is None
    assert all(mask == {0, 1, 2} for mask in masks.values())


def test_rollback_failure_does_not_stop_restoring_other_threads(monkeypatch, caplog):
    monkeypatch.setenv("OMNI_WORKER_CPUSET_MAP", '{"0":"1"}')
    original = {0, 1, 2}
    masks = {tid: original.copy() for tid in (10, 11, 12, 13)}
    fake_threads(monkeypatch, masks)
    def setter(tid, cpus):
        tid = tid or 10
        if tid == 13 or (tid == 10 and set(cpus) == original):
            raise PermissionError("controlled failure")
        masks[tid] = set(cpus)
    monkeypatch.setattr(affinity.os, "sched_setaffinity", setter)
    assert affinity.apply_worker_cpu_affinity(0, parallel_config()) is None
    assert masks[10] == {1}
    assert all(masks[tid] == original for tid in (11, 12, 13))
    assert "Could not restore CPU affinity for thread 10" in caplog.text
    assert "continuing worker startup" in caplog.text


def test_shutdown_signal_is_not_swallowed_and_changes_are_restored(monkeypatch):
    monkeypatch.setenv("OMNI_WORKER_CPUSET_MAP", '{"0":"1"}')
    masks = {10: {0, 1, 2}, 11: {0, 1, 2}}
    fake_threads(monkeypatch, masks)
    def setter(tid, cpus):
        if tid == 11:
            raise KeyboardInterrupt
        masks[tid or 10] = set(cpus)
    monkeypatch.setattr(affinity.os, "sched_setaffinity", setter)
    with pytest.raises(KeyboardInterrupt):
        affinity.apply_worker_cpu_affinity(0, parallel_config())
    assert all(mask == {0, 1, 2} for mask in masks.values())


@pytest.mark.skipif(not hasattr(os,"sched_getaffinity"), reason="Linux affinity required")
def test_real_child_process_inherits_cpu_group(monkeypatch):
    # Isolated child, not the pytest process or any running inference worker.
    allowed=sorted(os.sched_getaffinity(0))
    cpu=allowed[0]
    code='''
import importlib.util,json,os,subprocess,sys
from types import SimpleNamespace
spec=importlib.util.spec_from_file_location('binding',sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
os.environ['OMNI_WORKER_CPUSET_MAP']=json.dumps({'0':sys.argv[2]})
r=m.apply_worker_cpu_affinity(0,SimpleNamespace(data_parallel_size=1))
out=subprocess.check_output([sys.executable,'-c','import os,json;print(json.dumps(sorted(os.sched_getaffinity(0))))'],text=True)
assert json.loads(out)==[int(sys.argv[2])]
assert sorted(os.sched_getaffinity(0))==[int(sys.argv[2])]
print('inheritance verified')
'''
    run=subprocess.run([sys.executable,"-c",code,str(SOURCE),str(cpu)],capture_output=True,text=True,check=True)
    assert 'inheritance verified' in run.stdout
    assert sorted(os.sched_getaffinity(0))==allowed


@pytest.mark.skipif(not hasattr(os,"sched_getaffinity"), reason="Linux affinity required")
def test_real_auto_child_inherits_topology_group(monkeypatch):
    allowed = set(os.sched_getaffinity(0))
    nodes = affinity._read_numa_cpu_sets(allowed)
    if any(len(cpus) < 4 for cpus in nodes):
        pytest.skip("Real topology lacks room for two workers per NUMA node")
    tp = len(nodes) * 2
    expected = sorted(affinity._build_auto_cpu_groups(nodes, tp)[0])
    code = '''
import importlib.util,json,os,subprocess,sys
from types import SimpleNamespace
spec=importlib.util.spec_from_file_location('binding',sys.argv[1]);m=importlib.util.module_from_spec(spec);spec.loader.exec_module(m)
os.environ['OMNI_WORKER_CPUSET_MAP']='auto'
config=SimpleNamespace(data_parallel_size=1,tensor_parallel_size=int(sys.argv[2]),pipeline_parallel_size=1,nnodes_within_dp=1)
r=m.apply_worker_cpu_affinity(0,config)
out=subprocess.check_output([sys.executable,'-c','import os,json;print(json.dumps(sorted(os.sched_getaffinity(0))))'],text=True)
assert json.loads(out)==r['cpus']
print(json.dumps(r['cpus']))
'''
    run = subprocess.run([sys.executable, "-c", code, str(SOURCE), str(tp)],
                         capture_output=True, text=True, check=True)
    assert json.loads(run.stdout) == expected
    assert set(os.sched_getaffinity(0)) == allowed
