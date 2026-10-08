# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Deployment contracts and failure cleanup without running Mooncake or etcd."""

import importlib.util
import json
import os
import signal
import subprocess
import sys
import time
import types
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parents[3] / "tools" / "start_mooncake.py"
SPEC = importlib.util.spec_from_file_location("start_mooncake_test_target", SCRIPT)
launcher = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(launcher)
HOSTS = ["10.20.0.1", "10.20.0.2", "10.20.0.3"]


def argv(action="start", **changes):
    values = {
        "hosts": HOSTS,
        "cluster-id": "training-ha",
    }
    values.update(changes)
    result = [action]
    for name, value in values.items():
        if isinstance(value, list):
            result.extend(["--" + name, *map(str, value)])
        elif value is not None:
            result.extend(["--" + name, str(value)])
    return result


def start_args(tmp_path, **changes):
    options = {"node-rank": 1, "data-dir": tmp_path / "node"}
    options.update(changes)
    return launcher.parse_args(argv(**options))


def check_args(**changes):
    options = {"local-host": "10.20.1.8"}
    options.update(changes)
    return launcher.parse_args(argv("check", **options))


@pytest.mark.parametrize(
    "changes",
    [
        {"hosts": [HOSTS[0], HOSTS[0], HOSTS[2]]},
        {"hosts": ["127.0.0.1", *HOSTS[1:]]},
        {"hosts": ["0.0.0.0", *HOSTS[1:]]},
        {"hosts": ["224.0.0.1", *HOSTS[1:]]},
        {"hosts": ["example.org", *HOSTS[1:]]},
        {"hosts": ["::1", *HOSTS[1:]]},
        {"cluster-id": "../another-cluster"},
        {"cluster-id": "training;touch /tmp/x"},
        {"cluster-id": "x" * 65},
        {"master-port": 2379},
        {"etcd-port": 80},
        {"metrics-port": 65536},
        {"provider-ports": [20000, 19999]},
        {"provider-ports": [20000, 21000]},
        {"provider-ports": [2378, 2381]},
        {"client-ports": [15299, 15399]},
        {"pool-mib": 63},
        {"buffer-mib": 31},
        {"startup-timeout": 0},
        {"max-restarts": 0},
        {"node-rank": None},
        {"data-dir": None},
    ],
)
def test_start_rejects_unsafe_or_ambiguous_configuration(tmp_path, changes):
    with pytest.raises(SystemExit) as raised:
        start_args(tmp_path, **changes)
    assert raised.value.code == 2
    assert not (tmp_path / "node").exists()


@pytest.mark.parametrize("action", ["check", "client-config"])
def test_remote_clients_require_their_own_bind_address(action, tmp_path):
    with pytest.raises(SystemExit):
        launcher.parse_args(
            argv(action, model="served-model", **{"output-dir": tmp_path})
        )


def test_plan_forms_one_three_member_etcd_cluster_and_ha_master(tmp_path):
    plans = [
        launcher.build_plan(start_args(tmp_path, **{"node-rank": i})) for i in range(3)
    ]
    expected_cluster = "--initial-cluster=" + ",".join(
        f"mc-{i}=http://{host}:2380" for i, host in enumerate(HOSTS)
    )
    for rank, plan in enumerate(plans):
        assert expected_cluster in plan["etcd"]
        assert f"--name=mc-{rank}" in plan["etcd"]
        assert f"--advertise-client-urls=http://{HOSTS[rank]}:2379" in plan["etcd"]
        assert "--initial-cluster-token=training-ha" in plan["etcd"]
        assert "--enable_ha=true" in plan["master"]
        assert "--enable_oplog=true" in plan["master"]
        assert "--ha_backend_type=etcd" in plan["master"]
        assert "--cluster_id=training-ha" in plan["master"]
        assert (
            "--ha_backend_connstring=" + ";".join(f"{host}:2379" for host in HOSTS)
            in plan["master"]
        )
        child = launcher.parse_args(plan["provider"][3:])
        assert child.child == "provider"
        assert child.node_rank == rank
        assert child.hosts == HOSTS


def test_client_files_bind_to_rl_host_and_request_two_replicas(tmp_path, monkeypatch):
    output = tmp_path / "client files with spaces"
    args = launcher.parse_args(
        argv(
            "client-config",
            model="served-model",
            **{
                "local-host": "10.20.1.8",
                "output-dir": output,
                "client-ports": [16400, 16500],
            },
        )
    )
    launcher.generate_client(args)
    config = json.loads((output / "feature-store.json").read_text())
    assert config["model"] == "served-model"
    mooncake = config["mooncake"]
    assert mooncake["local_hostname"] == "10.20.1.8"
    assert mooncake["replica_num"] == 2
    assert "global_segment_size" not in mooncake
    assert mooncake["metadata_server"] == mooncake["master_server_addr"]
    assert mooncake["metadata_server"] == "etcd://" + ";".join(
        f"{host}:2379" for host in HOSTS
    )
    text = (output / "feature-store.env").read_text()
    assert "export MC_TCP_BIND_ADDRESS=10.20.1.8" in text
    assert "export MC_STORE_CLUSTER_ID=training-ha" in text
    assert "export MC_METADATA_CLUSTER_ID=training-ha" in text
    assert "export MC_MIN_RPC_PORT=16400" in text
    assert "export MC_MAX_RPC_PORT=16500" in text
    assert "unset MC_STORE_USE_HUGEPAGE" in text
    assert "export OMNI_FEATURE_STORE_CONFIG='" in text
    monkeypatch.setenv("MC_STORE_USE_HUGEPAGE", "1")
    env = launcher.child_environment(args, args.local_host)
    assert "MC_STORE_USE_HUGEPAGE" not in env
    assert env["MC_TCP_BIND_ADDRESS"] == "10.20.1.8"


@pytest.mark.parametrize("action", ["start", "check", "client-config"])
@pytest.mark.parametrize("mode", ["ha", "standalone"])
def test_dry_run_does_not_create_files_or_start_native_processes(
    tmp_path, monkeypatch, capsys, action, mode
):
    def forbidden(*_args, **_kwargs):
        pytest.fail("A dry run must not touch local services or dependencies")

    monkeypatch.setattr(launcher, "check_dependencies", forbidden)
    monkeypatch.setattr(launcher, "check_cluster", forbidden)
    monkeypatch.setattr(launcher.subprocess, "Popen", forbidden)
    if action == "start":
        options = {"node-rank": 0, "data-dir": tmp_path / "node"}
    else:
        options = {"local-host": "10.20.1.8"}
        if action == "client-config":
            options.update({"model": "served-model", "output-dir": tmp_path / "client"})
    options["mode"] = mode
    if mode == "standalone":
        options["hosts"] = [HOSTS[0]]
    assert launcher.main(argv(action, **options) + ["--dry-run"]) == 0
    output = json.loads(capsys.readouterr().out)
    assert output
    if action == "client-config":
        assert output["feature_store"]["mooncake"]["replica_num"] == (
            2 if mode == "ha" else 1
        )
    assert list(tmp_path.iterdir()) == []


@pytest.fixture
def fake_fcntl(monkeypatch):
    module = types.ModuleType("fcntl")
    module.LOCK_EX, module.LOCK_NB = 2, 4
    module.flock = lambda *_: None
    monkeypatch.setitem(sys.modules, "fcntl", module)
    return module


def test_deployment_identity_allows_restart_but_rejects_changed_topology(
    tmp_path, fake_fcntl
):
    args = start_args(tmp_path)
    with launcher.deployment_lock(args):
        marker = args.data_dir / "deployment.json"
        saved = marker.read_bytes()
    with launcher.deployment_lock(args):
        assert marker.read_bytes() == saved
    for field, replacement in [
        ("cluster_id", "unrelated"),
        ("node_rank", 2),
        ("hosts", list(reversed(HOSTS))),
        ("peer_port", 2390),
        ("etcd_port", 2391),
    ]:
        original = getattr(args, field)
        setattr(args, field, replacement)
        with (
            pytest.raises(RuntimeError, match="different cluster/topology"),
            launcher.deployment_lock(args),
        ):
            pytest.fail("Mismatched data must never be adopted")
        assert marker.read_bytes() == saved
        setattr(args, field, original)


def test_deployment_lock_rejects_unmanaged_data_and_concurrent_owner(
    tmp_path, fake_fcntl
):
    args = start_args(tmp_path)
    (args.data_dir / "etcd").mkdir(parents=True)
    payload = args.data_dir / "etcd" / "important.db"
    payload.write_bytes(b"unmanaged data")
    with (
        pytest.raises(RuntimeError, match="Unmanaged etcd data"),
        launcher.deployment_lock(args),
    ):
        pass
    assert payload.read_bytes() == b"unmanaged data"
    assert not (args.data_dir / "deployment.json").exists()

    def locked(*_):
        raise BlockingIOError("held")

    fake_fcntl.flock = locked
    with (
        pytest.raises(RuntimeError, match="already owns"),
        launcher.deployment_lock(args),
    ):
        pass


class Process:
    def __init__(self, pid):
        self.pid = pid
        self.returncode = None

    def poll(self):
        return self.returncode

    def wait(self, timeout=None):
        return self.returncode


class SupervisorHarness:
    def __init__(self, args, monkeypatch):
        self.now = 0
        self.waits = 0
        self.stop_after = 4
        self.tick = lambda: None
        self.spawned = []
        self.cleaned = []
        args.data_dir.mkdir()
        monkeypatch.setattr(launcher, "stop_event", lambda: self)
        monkeypatch.setattr(launcher.time, "monotonic", lambda: self.now)
        monkeypatch.setattr(launcher, "node_healthy", lambda _: False)
        monkeypatch.setattr(
            launcher,
            "build_plan",
            lambda _: {
                "etcd": ["etcd"],
                "master": ["master"],
                "provider": ["provider"],
            },
        )
        monkeypatch.setattr(launcher.subprocess, "Popen", self.spawn)
        monkeypatch.setattr(launcher, "terminate_children", self.cleanup)

    def is_set(self):
        return self.waits >= self.stop_after

    def wait(self, seconds):
        self.now += seconds
        self.waits += 1
        self.tick()
        return self.is_set()

    def spawn(self, command, **kwargs):
        process = Process(100 + len(self.spawned))
        self.spawned.append((command[0], process, kwargs))
        return process

    def cleanup(self, processes, timeout):
        self.cleaned.append((processes, timeout))


def test_partial_launch_failure_cleans_only_owned_children(tmp_path, monkeypatch):
    args = start_args(tmp_path)
    harness = SupervisorHarness(args, monkeypatch)

    def spawn(command, **kwargs):
        if command[0] == "master":
            raise OSError("cannot launch master")
        return harness.spawn(command, **kwargs)

    monkeypatch.setattr(launcher.subprocess, "Popen", spawn)
    with pytest.raises(OSError, match="cannot launch master"):
        launcher.supervise(args)
    assert [name for name, _, _ in harness.spawned] == ["etcd"]
    assert harness.cleaned == [([harness.spawned[0][1]], args.shutdown_timeout)]
    assert harness.spawned[0][2]["stdout"].closed
    assert not (args.data_dir / "node-ready.json").exists()


def test_child_exit_retries_only_that_service_with_a_new_session(tmp_path, monkeypatch):
    args = start_args(tmp_path)
    harness = SupervisorHarness(args, monkeypatch)

    def first_exit():
        if harness.waits == 1:
            harness.spawned[0][1].returncode = 7

    harness.tick = first_exit
    launcher.supervise(args)
    assert [name for name, _, _ in harness.spawned] == [
        "etcd",
        "master",
        "provider",
        "etcd",
    ]
    assert all(options["start_new_session"] for _, _, options in harness.spawned)
    assert all(options["stdout"].closed for _, _, options in harness.spawned)
    cleaned = harness.cleaned[0][0]
    assert harness.spawned[0][1] not in cleaned
    assert set(cleaned) == {p for _, p, _ in harness.spawned[1:]}


def test_repeated_child_exits_exhaust_budget_and_stop_survivors(tmp_path, monkeypatch):
    args = start_args(tmp_path, **{"max-restarts": 1})
    harness = SupervisorHarness(args, monkeypatch)
    harness.stop_after = 10

    def exit_etcd():
        for name, process, _ in harness.spawned:
            if name == "etcd":
                process.returncode = 9

    harness.tick = exit_etcd
    with pytest.raises(RuntimeError, match="restart budget"):
        launcher.supervise(args)
    assert len([name for name, _, _ in harness.spawned if name == "etcd"]) == 2
    assert harness.cleaned
    assert all(options["stdout"].closed for _, _, options in harness.spawned)


def test_startup_timeout_reaps_children_and_clears_ready_markers(tmp_path, monkeypatch):
    args = start_args(tmp_path, **{"startup-timeout": 1})
    harness = SupervisorHarness(args, monkeypatch)
    for name in ("provider-ready.json", "node-ready.json"):
        (args.data_dir / name).write_text("stale")
    with pytest.raises(RuntimeError, match="Startup timed out"):
        launcher.supervise(args)
    assert len(harness.cleaned[0][0]) == 3
    assert not (args.data_dir / "provider-ready.json").exists()
    assert not (args.data_dir / "node-ready.json").exists()


def test_signal_handlers_request_orderly_stop(monkeypatch):
    handlers = {}
    monkeypatch.setattr(
        launcher.signal, "signal", lambda sig, fn: handlers.update({sig: fn})
    )
    event = launcher.stop_event()
    assert not event.is_set()
    assert set(handlers) == {signal.SIGINT, signal.SIGTERM}
    handlers[signal.SIGTERM](signal.SIGTERM, None)
    assert event.is_set()


def test_termination_signals_only_live_owned_groups_and_escalates(monkeypatch):
    gone, slow, normal = Process(501), Process(502), Process(503)
    gone.returncode = 0
    signals = []
    waits = []
    monkeypatch.setattr(launcher.signal, "SIGKILL", 9, raising=False)
    monkeypatch.setattr(
        launcher.os,
        "killpg",
        lambda pid, sig: signals.append((pid, sig)),
        raising=False,
    )

    def slow_wait(timeout=None):
        waits.append(timeout)
        if len(waits) == 1:
            raise subprocess.TimeoutExpired("slow", timeout)
        slow.returncode = -9
        return -9

    slow.wait = slow_wait
    launcher.terminate_children([gone, slow, normal], 3)
    assert signals == [(502, signal.SIGTERM), (503, signal.SIGTERM), (502, 9)]
    assert len(waits) == 2
    assert waits[0] <= 3
    assert waits[1] is None


@pytest.mark.parametrize("failure", ["timeout", "interrupt"])
def test_cluster_probe_timeout_or_interrupt_cleans_child(
    tmp_path, monkeypatch, failure
):
    args = check_args(**{"startup-timeout": 1})
    process = Process(700)
    cleaned = []
    spawned = []
    monkeypatch.setattr(launcher, "read_http", lambda _: '{"health":"true"}')

    def spawn(command, **options):
        spawned.append((command, options))
        return process

    clock = [0]
    monkeypatch.setattr(launcher.time, "monotonic", lambda: clock[0])
    monkeypatch.setattr(
        launcher, "stop_event", lambda: types.SimpleNamespace(is_set=lambda: False)
    )

    def wait(timeout=None):
        clock[0] += 2
        if failure == "timeout":
            raise subprocess.TimeoutExpired("native probe", 1)
        raise KeyboardInterrupt()

    process.wait = wait
    monkeypatch.setattr(launcher.subprocess, "Popen", spawn)
    monkeypatch.setattr(
        launcher,
        "terminate_children",
        lambda ps, timeout: cleaned.append((ps, timeout)),
    )
    expected = RuntimeError if failure == "timeout" else KeyboardInterrupt
    with pytest.raises(expected):
        launcher.check_cluster(args)
    assert cleaned == [([process], args.shutdown_timeout)]
    assert spawned[0][1]["start_new_session"] is True
    assert spawned[0][1]["env"]["MC_TCP_BIND_ADDRESS"] == args.local_host


def test_missing_quorum_never_starts_native_probe(monkeypatch):
    args = check_args()

    def health(url):
        return json.dumps({"health": HOSTS[0] in url})

    def forbidden(*_args, **_kwargs):
        pytest.fail("Native probe must not run without quorum")

    monkeypatch.setattr(launcher, "read_http", health)
    monkeypatch.setattr(launcher.subprocess, "Popen", forbidden)
    with pytest.raises(RuntimeError, match="no HA quorum"):
        launcher.check_cluster(args)


def test_check_only_does_not_claim_a_deployment_directory(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        launcher, "check_dependencies", lambda _: calls.append("dependencies")
    )
    monkeypatch.setattr(launcher, "check_ports", lambda _: calls.append("ports"))
    assert (
        launcher.main(
            argv(
                **{
                    "node-rank": 0,
                    "data-dir": tmp_path / "not-created",
                }
            )
            + ["--check-only"]
        )
        == 0
    )
    assert calls == ["dependencies", "ports"]
    assert list(tmp_path.iterdir()) == []


def test_recovered_service_recreates_readiness_marker(tmp_path, monkeypatch):
    args = start_args(tmp_path)
    harness = SupervisorHarness(args, monkeypatch)
    observations = []
    original_spawn = harness.spawn
    monkeypatch.setattr(launcher, "node_healthy", lambda _: True)

    def spawn(command, **options):
        process = original_spawn(command, **options)
        if command[0] == "provider":
            (args.data_dir / "provider-ready.json").write_text("mounted")
        return process

    def tick():
        observations.append((args.data_dir / "node-ready.json").exists())
        if harness.waits == 1:
            harness.spawned[0][1].returncode = 7

    harness.tick = tick
    monkeypatch.setattr(launcher.subprocess, "Popen", spawn)
    launcher.supervise(args)
    assert observations == [True, False, True, True]
    assert not (args.data_dir / "node-ready.json").exists()


def test_sigterm_during_probe_wait_reaps_probe(monkeypatch):
    args = check_args()
    stopped = [False]
    process = Process(710)
    cleaned = []
    monkeypatch.setattr(
        launcher, "stop_event", lambda: types.SimpleNamespace(is_set=lambda: stopped[0])
    )
    monkeypatch.setattr(launcher, "read_http", lambda _: '{"health":true}')
    monkeypatch.setattr(launcher.subprocess, "Popen", lambda *_args, **_kwargs: process)
    monkeypatch.setattr(
        launcher, "terminate_children", lambda ps, timeout: cleaned.append(ps)
    )

    def wait(timeout=None):
        stopped[0] = True
        raise subprocess.TimeoutExpired("native probe", 1)

    process.wait = wait
    with pytest.raises(RuntimeError, match="interrupted"):
        launcher.check_cluster(args)
    assert cleaned == [[process]]


@pytest.fixture
def native_probe(monkeypatch):
    module = types.ModuleType("mooncake.store")
    module.ReplicateConfig = type("ReplicateConfig", (), {"replica_num": 1})
    module.ReplicaStatus = types.SimpleNamespace(COMPLETE="complete")
    monkeypatch.setitem(sys.modules, "mooncake", types.ModuleType("mooncake"))
    monkeypatch.setitem(sys.modules, "mooncake.store", module)
    shared = {}
    clients = []

    class Client:
        def __init__(self):
            self.closed = 0
            self.removed = []
            self.put_rc = 0
            self.replicas = [(HOSTS[0], "complete", True), (HOSTS[1], "complete", True)]

        def put(self, key, value, config):
            self.requested_replicas = config.replica_num
            self.key = key
            shared[key] = value
            return self.put_rc

        def get(self, key):
            assert clients[0].closed == 1
            return shared[key]

        def get_replica_desc(self, key):
            result = []
            for host, status, memory in self.replicas:
                descriptor = types.SimpleNamespace(
                    buffer_descriptor=types.SimpleNamespace(
                        transport_endpoint=host + ":15200",
                    )
                )
                result.append(
                    types.SimpleNamespace(
                        status=status,
                        is_memory_replica=lambda m=memory: m,
                        get_memory_descriptor=lambda d=descriptor: d,
                    )
                )
            return result

        def remove(self, key, *, force=False):
            self.removed.append((key, force))
            shared.pop(key, None)
            return 0

        def close(self):
            self.closed += 1
            return 0

    def setup(*_args, **_kwargs):
        client = Client()
        clients.append(client)
        return client

    monkeypatch.setattr(launcher, "setup_store", setup)
    return setup, clients, shared


def test_probe_verifies_independent_reader_and_cleans_only_own_key(native_probe):
    _, clients, shared = native_probe
    shared["training-embedding"] = b"untouched"
    launcher.run_probe(check_args())
    writer, reader = clients
    assert writer.requested_replicas == 2
    assert writer.closed == reader.closed == 1
    assert writer.removed == []
    assert reader.removed == [(writer.key, True)]
    assert writer.key.startswith("omni-ha-probe-")
    assert shared == {"training-embedding": b"untouched"}


@pytest.mark.parametrize(
    "replicas",
    [
        [(HOSTS[0], "complete", True), (HOSTS[0], "complete", True)],
        [(HOSTS[0], "complete", True), (HOSTS[1], "pending", True)],
        [(HOSTS[0], "complete", True), (HOSTS[1], "complete", False)],
        [(HOSTS[0], "complete", True), ("10.99.0.1", "complete", True)],
    ],
)
def test_probe_rejects_missing_complete_replicas_on_distinct_storage_hosts(
    monkeypatch, native_probe, replicas
):
    setup, clients, shared = native_probe

    def configure(*args, **kwargs):
        client = setup(*args, **kwargs)
        client.replicas = replicas
        return client

    monkeypatch.setattr(launcher, "setup_store", configure)
    with pytest.raises(RuntimeError, match="fewer than two storage-host replicas"):
        launcher.run_probe(check_args())
    assert len(clients) == 1
    assert clients[0].removed == [(clients[0].key, True)]
    assert clients[0].closed == 1
    assert not shared


def test_probe_cleans_a_write_that_committed_before_error_return(
    monkeypatch, native_probe
):
    setup, clients, shared = native_probe

    def configure(*args, **kwargs):
        client = setup(*args, **kwargs)
        client.put_rc = -1
        return client

    monkeypatch.setattr(launcher, "setup_store", configure)
    with pytest.raises(RuntimeError, match="Probe put returned -1"):
        launcher.run_probe(check_args())
    assert clients[0].removed == [(clients[0].key, True)]
    assert clients[0].closed == 1
    assert not shared


@pytest.mark.skipif(sys.platform != "linux", reason="POSIX process group isolation")
def test_real_process_group_shutdown_preserves_unrelated_process(tmp_path):
    marker = tmp_path / "grandchild.pid"
    child_code = "import time; time.sleep(60)"
    parent_code = (
        "import pathlib,subprocess,sys,time; "
        "child=subprocess.Popen([sys.executable,'-c',sys.argv[2]]); "
        "pathlib.Path(sys.argv[1]).write_text(str(child.pid)); time.sleep(60)"
    )
    unrelated = subprocess.Popen(
        [sys.executable, "-c", child_code], start_new_session=True
    )
    owned = subprocess.Popen(
        [sys.executable, "-c", parent_code, str(marker), child_code],
        start_new_session=True,
    )
    try:
        deadline = time.monotonic() + 3
        while not marker.exists() and time.monotonic() < deadline:
            time.sleep(0.01)
        assert marker.exists(), "Fake provider did not launch its child"
        grandchild = int(marker.read_text())
        launcher.terminate_children([owned], 2)
        assert owned.poll() is not None
        assert unrelated.poll() is None
        status = Path(f"/proc/{grandchild}/status")
        deadline = time.monotonic() + 2
        while status.exists() and time.monotonic() < deadline:
            if "State:\tZ" in status.read_text():
                break
            time.sleep(0.01)
        assert not status.exists() or "State:\tZ" in status.read_text()
    finally:
        for process in (owned, unrelated):
            if process.poll() is None:
                os.killpg(process.pid, signal.SIGKILL)
                process.wait(timeout=3)


def test_probe_mismatched_read_still_cleans_owned_key(monkeypatch, native_probe):
    setup, clients, shared = native_probe

    def configure(*args, **kwargs):
        client = setup(*args, **kwargs)
        if len(clients) == 2:
            client.get = lambda _: b"corrupt"
        return client

    monkeypatch.setattr(launcher, "setup_store", configure)
    with pytest.raises(RuntimeError, match="did not match"):
        launcher.run_probe(check_args())
    writer, reader = clients
    assert writer.closed == reader.closed == 1
    assert reader.removed == [(writer.key, True)]
    assert not shared


def standalone_args(tmp_path=None, action="start", **changes):
    options = {"mode": "standalone", "hosts": [HOSTS[0]]}
    if action == "start":
        options["data-dir"] = tmp_path / "standalone"
    else:
        options["local-host"] = "10.20.1.8"
    if action == "client-config":
        options.update({"model": "served-model", "output-dir": tmp_path / "client"})
    options.update(changes)
    return launcher.parse_args(argv(action, **options))


def test_standalone_defaults_to_rank_zero_and_disables_master_ha(tmp_path):
    args = standalone_args(tmp_path)
    assert args.node_rank == 0
    plan = launcher.build_plan(args)
    assert set(plan) == {"etcd", "master", "provider"}
    assert f"--initial-cluster=mc-0=http://{HOSTS[0]}:2380" in plan["etcd"]
    assert "--name=mc-0" in plan["etcd"]
    assert "--enable_ha=false" in plan["master"]
    assert "--enable_oplog=false" in plan["master"]
    assert "--enable_ha=true" not in plan["master"]
    child = launcher.parse_args(plan["provider"][3:])
    assert child.mode == "standalone"
    assert child.node_rank == 0
    assert child.hosts == [HOSTS[0]]


@pytest.mark.parametrize(
    "changes",
    [
        {"mode": "standalone", "hosts": HOSTS},
        {"mode": "standalone", "hosts": HOSTS[:2]},
        {"mode": "standalone", "hosts": []},
        {"mode": "standalone", "hosts": [HOSTS[0]], "node-rank": 1},
        {"mode": "standalone", "hosts": [HOSTS[0]], "node-rank": 2},
        {"mode": "ha", "hosts": [HOSTS[0]], "node-rank": 0},
        {"mode": "ha", "hosts": HOSTS[:2], "node-rank": 0},
        {"mode": "ha", "hosts": [*HOSTS, "10.20.0.4"], "node-rank": 0},
        {"mode": "unknown", "hosts": [HOSTS[0]], "node-rank": 0},
    ],
)
def test_mode_host_count_and_rank_must_agree(tmp_path, changes):
    with pytest.raises(SystemExit) as raised:
        launcher.parse_args(argv(**{"data-dir": tmp_path / "node", **changes}))
    assert raised.value.code == 2
    assert list(tmp_path.iterdir()) == []


def test_legacy_cli_defaults_to_ha(tmp_path):
    legacy = start_args(tmp_path)
    explicit = start_args(tmp_path, mode="ha")
    assert legacy.mode == explicit.mode == "ha"
    assert launcher.build_plan(legacy) == launcher.build_plan(explicit)


def test_standalone_client_uses_etcd_metadata_and_direct_master(tmp_path):
    args = standalone_args(
        tmp_path,
        "client-config",
        **{
            "etcd-port": 12379,
            "master-port": 50071,
        },
    )
    launcher.generate_client(args)
    config = json.loads((args.output_dir / "feature-store.json").read_text())
    assert config["mooncake"]["metadata_server"] == f"etcd://{HOSTS[0]}:12379"
    assert config["mooncake"]["master_server_addr"] == f"{HOSTS[0]}:50071"
    assert config["mooncake"]["replica_num"] == 1
    assert config["mooncake"]["local_hostname"] == "10.20.1.8"
    assert "global_segment_size" not in config["mooncake"]
    _, env = launcher.render_client(args)
    assert env["MC_TCP_BIND_ADDRESS"] == "10.20.1.8"
    assert env["MC_STORE_CLUSTER_ID"] == args.cluster_id
    assert env["MC_METADATA_CLUSTER_ID"] == args.cluster_id


def test_standalone_probe_command_preserves_mode_and_custom_addresses():
    args = standalone_args(
        action="check",
        **{
            "etcd-port": 12379,
            "master-port": 50071,
        },
    )
    child = launcher.parse_args(launcher.child_command(args, "probe")[3:])
    assert child.mode == "standalone"
    assert child.action == "check"
    assert child.hosts == [HOSTS[0]]
    assert child.etcd_port == 12379
    assert child.master_port == 50071
    assert child.local_host == args.local_host


@pytest.mark.parametrize("provider", [False, True])
def test_standalone_native_setup_matches_exported_endpoints(
    tmp_path, monkeypatch, provider
):
    args = standalone_args(tmp_path)
    setups = []
    client = types.SimpleNamespace(setup=lambda **kwargs: setups.append(kwargs) or 0)
    module = types.ModuleType("mooncake.store")
    module.MooncakeDistributedStore = lambda: client
    monkeypatch.setitem(sys.modules, "mooncake", types.ModuleType("mooncake"))
    monkeypatch.setitem(sys.modules, "mooncake.store", module)
    monkeypatch.setattr(launcher.os, "environ", {})
    assert launcher.setup_store(args, HOSTS[0], provider=provider) is client
    assert setups[0]["metadata_server"] == f"etcd://{HOSTS[0]}:2379"
    assert setups[0]["master_server_addr"] == f"{HOSTS[0]}:50061"
    assert setups[0]["global_segment_size"] == (
        args.pool_mib * launcher.MIB if provider else 0
    )


def test_manifest_records_mode_and_accepts_legacy_ha_manifest(tmp_path, fake_fcntl):
    ha = start_args(tmp_path)
    with launcher.deployment_lock(ha):
        manifest = ha.data_dir / "deployment.json"
        value = json.loads(manifest.read_text())
        assert value.pop("mode") == "ha"
        manifest.write_text(json.dumps(value))
    with launcher.deployment_lock(ha):
        pass

    single = standalone_args(tmp_path)
    with launcher.deployment_lock(single):
        manifest = single.data_dir / "deployment.json"
        value = json.loads(manifest.read_text())
        assert value["mode"] == "standalone"
    with launcher.deployment_lock(single):
        pass
    value.pop("mode")
    manifest.write_text(json.dumps(value))
    with (
        pytest.raises(RuntimeError, match="different cluster/topology"),
        launcher.deployment_lock(single),
    ):
        pytest.fail("A missing mode belongs to HA, not standalone")


def test_standalone_probe_accepts_one_complete_storage_replica(
    monkeypatch, native_probe
):
    setup, clients, shared = native_probe

    def configure(*args, **kwargs):
        client = setup(*args, **kwargs)
        client.replicas = [(HOSTS[0], "complete", True)]
        return client

    monkeypatch.setattr(launcher, "setup_store", configure)
    launcher.run_probe(standalone_args(action="check"))
    writer, reader = clients
    assert writer.requested_replicas == 1
    assert writer.closed == reader.closed == 1
    assert reader.removed == [(writer.key, True)]
    assert not shared


def test_standalone_probe_still_requires_a_complete_local_storage_replica(
    monkeypatch, native_probe
):
    setup, clients, shared = native_probe

    def configure(*args, **kwargs):
        client = setup(*args, **kwargs)
        client.replicas = [(HOSTS[0], "pending", True)]
        return client

    monkeypatch.setattr(launcher, "setup_store", configure)
    with pytest.raises(RuntimeError, match="replica"):
        launcher.run_probe(standalone_args(action="check"))
    assert clients[0].removed == [(clients[0].key, True)]
    assert not shared


@pytest.mark.parametrize("healthy", [True, False])
def test_standalone_check_requires_its_one_etcd_endpoint(monkeypatch, healthy):
    args = standalone_args(action="check")
    spawned, requests, cleaned = [], [], []
    process = Process(850)
    process.returncode = 0

    def health(url):
        requests.append(url)
        return json.dumps({"health": healthy})

    def spawn(command, **options):
        spawned.append(command)
        return process

    monkeypatch.setattr(
        launcher, "stop_event", lambda: types.SimpleNamespace(is_set=lambda: False)
    )
    monkeypatch.setattr(launcher, "read_http", health)
    monkeypatch.setattr(launcher.subprocess, "Popen", spawn)
    monkeypatch.setattr(
        launcher, "terminate_children", lambda ps, timeout: cleaned.append(ps)
    )
    if healthy:
        launcher.check_cluster(args)
        assert len(spawned) == 1
        assert cleaned == [[process]]
    else:
        with pytest.raises(RuntimeError, match="etcd endpoint is unavailable"):
            launcher.check_cluster(args)
        assert not spawned
    assert requests == [f"http://{HOSTS[0]}:2379/health"]


@pytest.mark.parametrize(
    "mode,summary,expected",
    [
        ("standalone", "role=leader, state=serving, service_ready=true", True),
        ("standalone", "role=standby, state=standby", False),
        ("standalone", "role=leader, state=serving, service_ready=false", False),
        ("standalone", "role=leader, state=warmup, service_ready=true", False),
        ("ha", "role=leader, state=serving, service_ready=true", True),
        ("ha", "role=standby, state=standby", True),
    ],
)
def test_readiness_requires_serving_master_in_standalone_mode(
    tmp_path, monkeypatch, mode, summary, expected
):
    args = standalone_args(tmp_path) if mode == "standalone" else start_args(tmp_path)

    def read(url):
        return summary if url.endswith("/metrics/summary") else '{"health":true}'

    monkeypatch.setattr(launcher, "read_http", read)
    assert launcher.node_healthy(args) is expected


@pytest.mark.parametrize("mode", ["ha", "standalone"])
@pytest.mark.parametrize("master_bin", [None, "/opt/master-lease60.sh"])
@pytest.mark.parametrize("lease_ms", [None, 60000])
def test_read_lease_dry_run_targets_only_master(
    tmp_path, monkeypatch, capsys, mode, master_bin, lease_ms
):
    def forbidden(*_args, **_kwargs):
        pytest.fail("Lease dry-run must not invoke native processes")

    monkeypatch.setattr(launcher, "check_dependencies", forbidden)
    monkeypatch.setattr(launcher.subprocess, "Popen", forbidden)
    options = {
        "mode": mode,
        "hosts": HOSTS if mode == "ha" else HOSTS[:1],
        "node-rank": 0,
        "data-dir": tmp_path / "node",
        "master-bin": master_bin,
        "kv-lease-ttl-ms": lease_ms,
    }
    assert launcher.main(argv(**options) + ["--dry-run"]) == 0
    output = json.loads(capsys.readouterr().out)
    commands = output["commands"]
    flags = [
        arg for arg in commands["master"] if arg.startswith("--default_kv_lease_ttl=")
    ]
    assert flags == ([] if lease_ms is None else ["--default_kv_lease_ttl=60000"])
    if master_bin:
        assert commands["master"][0] == master_bin
    for role in ("etcd", "provider"):
        assert not any(
            arg.startswith(("--kv-lease-ttl-ms", "--default_kv_lease_ttl"))
            for arg in commands[role]
        )
    assert not any("lease" in key.lower() for key in output["environment"])
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("value", [0, -1, 2**64, "1.5", "inf", "invalid"])
def test_invalid_read_lease_rejected_before_start(tmp_path, value):
    with pytest.raises(SystemExit) as raised:
        start_args(tmp_path, **{"kv-lease-ttl-ms": value})
    assert raised.value.code == 2
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("action", ["check", "client-config"])
def test_read_lease_cannot_be_silently_set_on_client_action(tmp_path, capsys, action):
    with pytest.raises(SystemExit):
        launcher.parse_args(
            argv(
                action,
                **{
                    "local-host": "10.20.1.8",
                    "model": "served-model",
                    "output-dir": tmp_path,
                    "kv-lease-ttl-ms": 60000,
                },
            )
        )
    assert "--kv-lease-ttl-ms is supported only for start" in capsys.readouterr().err


@pytest.mark.parametrize("lease_ms", [None, 60000])
@pytest.mark.parametrize("supported", [False, True])
def test_preflight_checks_requested_master_read_lease(
    tmp_path, monkeypatch, lease_ms, supported
):
    args = start_args(tmp_path, **{"kv-lease-ttl-ms": lease_ms})
    calls = []
    flags = "enable_ha ha_backend_connstring enable_oplog enable_snapshot_restore"
    if supported:
        flags += " default_kv_lease_ttl"

    def run(command, **_kwargs):
        calls.append(command)
        if "--help" in command:
            return subprocess.CompletedProcess(command, 1, stdout=flags, stderr="")
        return subprocess.CompletedProcess(command, 0)

    monkeypatch.setattr(launcher.sys, "platform", "linux")
    monkeypatch.setattr(launcher.shutil, "which", lambda binary: binary)
    monkeypatch.setattr(launcher.subprocess, "run", run)
    if lease_ms is not None and not supported:
        with pytest.raises(RuntimeError, match="Master lacks --default_kv_lease_ttl"):
            launcher.check_dependencies(args)
        assert len(calls) == 2
    else:
        launcher.check_dependencies(args)
        assert len(calls) == 3


def test_read_lease_can_change_on_restart_without_resetting_etcd(tmp_path, fake_fcntl):
    args = start_args(tmp_path)
    with launcher.deployment_lock(args):
        manifest = (args.data_dir / "deployment.json").read_bytes()
        (args.data_dir / "etcd").mkdir()
        data = args.data_dir / "etcd" / "data"
        data.write_bytes(b"existing cluster data")
    updated = start_args(tmp_path, **{"kv-lease-ttl-ms": 60000})
    with launcher.deployment_lock(updated):
        assert (args.data_dir / "deployment.json").read_bytes() == manifest
        assert data.read_bytes() == b"existing cluster data"
        assert "--default_kv_lease_ttl=60000" in launcher.build_plan(updated)["master"]
