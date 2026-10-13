#!/usr/bin/env python3
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Start a standalone Mooncake pool or one member of a three-host HA pool.

HA is the default; use --mode standalone for a single host. See
docs/mooncake_deployment.md. The foreground supervisor owns only its children.
"""

from __future__ import annotations

import argparse
import importlib.metadata
import ipaddress
import json
import logging
import os
import re
import shlex
import shutil
import signal
import socket
import subprocess
import sys
import threading
import time
import urllib.request
import uuid
from contextlib import contextmanager
from pathlib import Path

LOG = logging.getLogger("mooncake-launcher")
MIB = 1024**2
VERSION = "0.3.13.post1"
CANN_ENV_SCRIPT = Path("/usr/local/Ascend/ascend-toolkit/set_env.sh")
MOONCAKE_ENV_SCRIPT = Path("/opt/mooncake-runtime/activate.sh")
IMAGE_ENV_READY = "_OMNI_MOONCAKE_IMAGE_ENV_READY"


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("start", "check", "client-config"))
    parser.add_argument(
        "--mode",
        choices=("ha", "standalone"),
        default="ha",
        help="ha: three nodes (default); standalone: one node without failover",
    )
    parser.add_argument(
        "--hosts",
        nargs="+",
        required=True,
        metavar="IP",
        help="One storage host for standalone, or three distinct hosts for ha (same order everywhere)",
    )
    parser.add_argument(
        "--cluster-id",
        required=True,
        help="Unique deployment namespace; reuse it on restart",
    )
    parser.add_argument("--node-rank", type=int, choices=range(3))
    parser.add_argument(
        "--data-dir", type=Path, help="Persistent local directory, required for start"
    )
    parser.add_argument("--local-host", help="RL/worker/probe host IPv4 address")
    parser.add_argument(
        "--model", help="Exact served model name, required for client-config"
    )
    parser.add_argument(
        "--output-dir", type=Path, help="Output directory for client-config"
    )
    parser.add_argument("--pool-mib", type=int, default=4096)
    parser.add_argument("--buffer-mib", type=int, default=64)
    parser.add_argument("--max-payload-mib", type=int,
                        help="client-config only: embedding size limit in MiB (default: 32)")
    parser.add_argument("--max-inflight-mib", type=int,
                        help="client-config only: in-flight memory budget in MiB (default: 1024)")
    parser.add_argument("--max-items", type=int,
                        help="client-config only: feature references per request, 1..128 (default: 8)")
    parser.add_argument("--io-workers", type=int,
                        help="client-config only: feature I/O workers, 1..64 (default: 4)")
    parser.add_argument("--max-inflight-requests", type=int,
                        help="client-config only: concurrent preparations, 1..1024 (default: 8)")
    parser.add_argument("--io-timeout-seconds", type=float,
                        help="client-config only: feature I/O timeout, >0..600 (default: 30)")
    parser.add_argument(
        "--kv-lease-ttl-ms",
        type=int,
        help=(
            "Master object read lease in milliseconds (start only); omitted: keep "
            "the Master default (10000) or --master-bin wrapper setting. "
            "Use the same value on all HA nodes"
        ),
    )
    parser.add_argument("--etcd-port", type=int, default=2379)
    parser.add_argument("--peer-port", type=int, default=2380)
    parser.add_argument("--master-port", type=int, default=50061)
    parser.add_argument("--metrics-port", type=int, default=9013)
    parser.add_argument("--provider-ports", nargs=2, type=int, default=[15200, 15299])
    parser.add_argument("--client-ports", nargs=2, type=int, default=[15300, 15399])
    parser.add_argument("--etcd-bin", default="etcd")
    parser.add_argument(
        "--master-bin",
        help="Optional explicit Master executable; defaults to this Python environment",
    )
    parser.add_argument(
        "--dependency-timeout",
        type=int,
        default=90,
        help="Seconds allowed for each Master help/native import preflight command",
    )
    parser.add_argument("--startup-timeout", type=int, default=180)
    parser.add_argument("--shutdown-timeout", type=int, default=45)
    parser.add_argument(
        "--max-restarts",
        type=int,
        default=10,
        help="Per-child exit limit in a rolling five-minute window",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print the plan; do not write or start anything",
    )
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Check local start dependencies and ports, then exit",
    )
    parser.add_argument(
        "--child", choices=("provider", "probe", "dependency"), help=argparse.SUPPRESS
    )
    args = parser.parse_args(argv)
    try:
        validate_args(args)
    except ValueError as exc:
        parser.error(str(exc))
    return args


def validate_args(args):
    for host in [*args.hosts, *([args.local_host] if args.local_host else [])]:
        address = ipaddress.IPv4Address(host)
        if (
            address.is_unspecified
            or address.is_loopback
            or address.is_multicast
            or address.is_reserved
        ):
            raise ValueError(
                "Use reachable, unicast host IPv4 addresses (not loopback)"
            )
    expected_hosts = 3 if args.mode == "ha" else 1
    if len(args.hosts) != expected_hosts or len(set(args.hosts)) != expected_hosts:
        raise ValueError(
            f"{args.mode} requires {expected_hosts} distinct storage host(s)"
        )
    if args.node_rank is not None and not 0 <= args.node_rank < expected_hosts:
        raise ValueError("node-rank must identify a host in --hosts")
    if args.mode == "standalone" and args.action == "start" and args.node_rank is None:
        args.node_rank = 0
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,63}", args.cluster_id):
        raise ValueError(
            "cluster-id must contain 1-64 letters, digits, dots, underscores or hyphens"
        )
    ports = [args.etcd_port, args.peer_port, args.master_port, args.metrics_port]
    if len(set(ports)) != len(ports) or any(not 1024 <= p <= 65535 for p in ports):
        raise ValueError("Service ports must be distinct and within 1024..65535")
    used = set(ports)
    for lower, upper in (args.provider_ports, args.client_ports):
        if not 1024 <= lower <= upper <= 65535 or upper - lower > 999:
            raise ValueError(
                "RPC port ranges must be ordered, within 1024..65535 and at most 1000 ports"
            )
        current = set(range(lower, upper + 1))
        if current & used:
            raise ValueError("Service, provider and client ports must not overlap")
        used |= current
    if args.pool_mib < 64 or args.buffer_mib < 32:
        raise ValueError("pool-mib must be >=64 and buffer-mib must be >=32")
    client_options = (args.max_payload_mib, args.max_inflight_mib,
                      args.max_items, args.io_workers, args.max_inflight_requests,
                      args.io_timeout_seconds)
    if any(value is not None for value in client_options) and args.action != "client-config":
        raise ValueError("FeatureStore limits and I/O options are supported only for client-config")
    for name, value, maximum in (
        ("max-items", args.max_items, 128),
        ("io-workers", args.io_workers, 64),
        ("max-inflight-requests", args.max_inflight_requests, 1024),
    ):
        if value is not None and not 1 <= value <= maximum:
            raise ValueError(f"{name} must be within 1..{maximum}")
    if args.io_timeout_seconds is not None and not 0 < args.io_timeout_seconds <= 600:
        raise ValueError("io-timeout-seconds must be finite and within (0, 600]")
    payload = args.max_payload_mib if args.max_payload_mib is not None else 32
    inflight = args.max_inflight_mib if args.max_inflight_mib is not None else 1024
    if payload <= 0 or inflight < 4 * payload:
        raise ValueError("max-payload-mib must be positive and max-inflight-mib must be >= 4 * max-payload-mib")
    if args.kv_lease_ttl_ms is not None:
        if not 1 <= args.kv_lease_ttl_ms <= 2**64 - 1:
            raise ValueError(
                "kv-lease-ttl-ms must be within 1..18446744073709551615 milliseconds"
            )
        if args.action != "start":
            raise ValueError("--kv-lease-ttl-ms is supported only for start")
    if min(
        args.dependency_timeout,
        args.startup_timeout,
        args.shutdown_timeout,
        args.max_restarts,
    ) <= 0:
        raise ValueError("Timeouts and max-restarts must be positive")
    if args.action == "start" and (args.node_rank is None or args.data_dir is None):
        raise ValueError("start requires --node-rank and --data-dir")
    if args.action in ("check", "client-config") and not args.local_host:
        raise ValueError(f"{args.action} requires the calling host's --local-host")
    if args.action == "client-config" and (not args.model or not args.output_dir):
        raise ValueError("client-config requires --model and --output-dir")
    if args.check_only and args.action != "start":
        raise ValueError("--check-only is supported only for start")
    if args.data_dir is not None:
        args.data_dir = args.data_dir.resolve()


def bootstrap_image_environment(args, argv):
    """Re-exec with the bundled image environment before loading native code."""
    if (
        sys.platform != "linux"
        or (not args.child and (args.dry_run or args.action == "client-config"))
        or os.environ.get(IMAGE_ENV_READY) == "1"
        or not MOONCAKE_ENV_SCRIPT.is_file()
    ):
        return
    if not CANN_ENV_SCRIPT.is_file():
        raise RuntimeError(
            f"Found {MOONCAKE_ENV_SCRIPT}, but required CANN environment is missing: "
            f"{CANN_ENV_SCRIPT}"
        )
    # LD_LIBRARY_PATH and PYTHONPATH must apply when Python starts. Updating
    # os.environ after startup is not sufficient for this process's native imports.
    # Keep user arguments out of shell source and preserve the launcher PID.
    command = (
        'set -e\n'
        'source "$1"\n'
        'source "$2"\n'
        'export LD_LIBRARY_PATH="/usr/local/Ascend/driver/lib64:'
        '/usr/local/Ascend/driver/lib64/common:/usr/local/Ascend/driver/lib64/driver'
        '${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}"\n'
        'export MC_FORCE_TCP=1\n'
        f'export {IMAGE_ENV_READY}=1\n'
        'cd -- "$3"\n'
        'shift 3\n'
        'exec "$@"\n'
    )
    LOG.info("Loading CANN and Mooncake image environment before native preflight")
    os.execv(
        "/bin/bash",
        [
            "/bin/bash",
            "-c",
            command,
            "mooncake-environment",
            str(CANN_ENV_SCRIPT),
            str(MOONCAKE_ENV_SCRIPT),
            os.getcwd(),
            sys.executable,
            "-u",
            str(Path(__file__).resolve()),
            *argv,
        ],
    )


def endpoint(args):
    return "etcd://" + ";".join(f"{host}:{args.etcd_port}" for host in args.hosts)


def master_endpoint(args):
    if args.mode == "ha":
        return endpoint(args)
    return f"{args.hosts[0]}:{args.master_port}"


def environment(args, host, *, provider=False):
    ports = args.provider_ports if provider else args.client_ports
    return {
        "MC_STORE_CLUSTER_ID": args.cluster_id,
        "MC_METADATA_CLUSTER_ID": args.cluster_id,
        "MC_TCP_BIND_ADDRESS": host,
        "MC_MIN_RPC_PORT": str(ports[0]),
        "MC_MAX_RPC_PORT": str(ports[1]),
        "MC_DISABLE_MMAP_ARENA": "1",
        "MC_MS_AUTO_DISC": "0",
        # This launcher and its generated client configuration use CPU/TCP only.
        "MC_FORCE_TCP": "1",
    }


def child_environment(args, host, *, provider=False):
    env = dict(os.environ)
    env.pop("MC_STORE_USE_HUGEPAGE", None)
    env.update(environment(args, host, provider=provider))
    return env


def child_command(args, role):
    # argv is passed directly; no shell expansion, SSH credentials or shell code.
    command = [
        sys.executable,
        "-u",
        str(Path(__file__).resolve()),
        args.action,
        "--mode",
        args.mode,
        "--hosts",
        *args.hosts,
        "--cluster-id",
        args.cluster_id,
        "--etcd-port",
        str(args.etcd_port),
        "--peer-port",
        str(args.peer_port),
        "--master-port",
        str(args.master_port),
        "--metrics-port",
        str(args.metrics_port),
        "--provider-ports",
        *map(str, args.provider_ports),
        "--client-ports",
        *map(str, args.client_ports),
        "--pool-mib",
        str(args.pool_mib),
        "--buffer-mib",
        str(args.buffer_mib),
        "--child",
        role,
    ]
    if args.node_rank is not None:
        command += ["--node-rank", str(args.node_rank)]
    if args.data_dir:
        command += ["--data-dir", str(args.data_dir)]
    if args.local_host:
        command += ["--local-host", args.local_host]
    return command


def master_command(args):
    return (
        [args.master_bin] if args.master_bin else [sys.executable, "-m", "mooncake.cli"]
    )


def build_plan(args):
    host = args.hosts[args.node_rank]
    master_lease = (
        [f"--default_kv_lease_ttl={args.kv_lease_ttl_ms}"]
        if args.kv_lease_ttl_ms is not None
        else []
    )
    cluster = ",".join(
        f"mc-{i}=http://{ip}:{args.peer_port}" for i, ip in enumerate(args.hosts)
    )
    master_mode = ["--enable_ha=false", "--enable_oplog=false"]
    if args.mode == "ha":
        master_mode = [
            "--enable_ha=true",
            "--ha_backend_type=etcd",
            f"--ha_backend_connstring={endpoint(args)[7:]}",
            "--enable_oplog=true",
        ]
    return {
        "etcd": [
            args.etcd_bin,
            f"--name=mc-{args.node_rank}",
            f"--data-dir={args.data_dir / 'etcd'}",
            f"--listen-client-urls=http://{host}:{args.etcd_port}",
            f"--advertise-client-urls=http://{host}:{args.etcd_port}",
            f"--listen-peer-urls=http://{host}:{args.peer_port}",
            f"--initial-advertise-peer-urls=http://{host}:{args.peer_port}",
            f"--initial-cluster={cluster}",
            f"--initial-cluster-token={args.cluster_id}",
            "--initial-cluster-state=new",
        ],
        "master": [
            *master_command(args),
            f"--rpc_address={host}",
            f"--rpc_port={args.master_port}",
            "--rpc_thread_num=2",
            "--metrics_host=127.0.0.1",
            f"--metrics_port={args.metrics_port}",
            *master_mode,
            *master_lease,
            f"--cluster_id={args.cluster_id}",
            "--enable_snapshot=false",
            "--enable_snapshot_restore=false",
            "--enable_http_metadata_server=false",
            "--enable_offload=false",
            "--logtostderr=true",
        ],
        "provider": child_command(args, "provider"),
    }


def render_client(args):
    config = {
        "model": args.model,
        "max_items": args.max_items if args.max_items is not None else 8,
        "io_workers": args.io_workers if args.io_workers is not None else 4,
        "max_inflight_requests": args.max_inflight_requests if args.max_inflight_requests is not None else 8,
        "io_timeout_seconds": args.io_timeout_seconds if args.io_timeout_seconds is not None else 30.0,
        "max_payload_bytes": (args.max_payload_mib if args.max_payload_mib is not None else 32) * MIB,
        "max_inflight_bytes": (args.max_inflight_mib if args.max_inflight_mib is not None else 1024) * MIB,
        "mooncake": {
            "local_hostname": args.local_host,
            "metadata_server": endpoint(args),
            "master_server_addr": master_endpoint(args),
            "protocol": "tcp",
            "rdma_devices": "",
            "local_buffer_size": args.buffer_mib * MIB,
            "replica_num": 2 if args.mode == "ha" else 1,
        },
    }
    env = environment(args, args.local_host)
    env["OMNI_FEATURE_STORE_CONFIG"] = str(
        (args.output_dir / "feature-store.json").resolve()
    )
    return config, env


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


def generate_client(args):
    config, env = render_client(args)
    if args.dry_run:
        print(json.dumps({"feature_store": config, "environment": env}, indent=2))
        return
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_json(args.output_dir / "feature-store.json", config)
    text = "# Source before importing Mooncake or starting Ray/vLLM on THIS host.\n"
    text += "unset MC_STORE_USE_HUGEPAGE\n"
    text += "".join(
        f"export {key}={shlex.quote(value)}\n" for key, value in env.items()
    )
    (args.output_dir / "feature-store.env").write_text(
        text, encoding="utf-8", newline="\n"
    )
    LOG.info(
        "Source %s before launching RL or inference",
        args.output_dir / "feature-store.env",
    )


def check_dependencies(args):
    if sys.platform != "linux":
        raise RuntimeError(
            "Start/check require Linux; dry-run and client-config work without native dependencies"
        )
    for binary in [args.etcd_bin, *([args.master_bin] if args.master_bin else [])]:
        if not shutil.which(binary):
            raise RuntimeError(
                f"Missing {binary}; use an image with etcd and Mooncake {VERSION} (etcd support)"
            )
    subprocess.run(
        [args.etcd_bin, "--version"], check=True, timeout=10, capture_output=True
    )
    help_result = subprocess.run(
        [*master_command(args), "--help"],
        check=False,
        timeout=args.dependency_timeout,
        capture_output=True,
        text=True,
    )
    # gflags can exit with status 1 for --help.
    output = help_result.stdout + help_result.stderr
    required_flags = [
        "enable_ha",
        "ha_backend_connstring",
        "enable_oplog",
        "enable_snapshot_restore",
    ]
    if args.kv_lease_ttl_ms is not None:
        required_flags.append("default_kv_lease_ttl")
    for flag in required_flags:
        if flag not in output:
            raise RuntimeError(f"Master lacks --{flag}; requires Mooncake {VERSION}")
    subprocess.run(
        child_command(args, "dependency"),
        check=True,
        timeout=args.dependency_timeout,
    )


def check_native_dependency():
    installed = []
    for name in (
        "mooncake-transfer-engine-non-cuda",
        "mooncake-transfer-engine",
        "mooncake-transfer-engine-npu",
    ):
        try:
            installed.append((name, importlib.metadata.version(name)))
        except importlib.metadata.PackageNotFoundError:
            pass
    if len(installed) != 1 or installed[0][1] != VERSION:
        raise RuntimeError(
            f"Expected exactly one Mooncake {VERSION} distribution, found {installed}"
        )
    import mooncake
    from mooncake.store import MooncakeDistributedStore, ReplicateConfig

    if not list(Path(mooncake.__file__).parent.rglob("libetcd_wrapper.so*")):
        raise RuntimeError(
            "Mooncake lacks libetcd_wrapper.so; use an etcd-enabled Mooncake build"
        )
    if not callable(MooncakeDistributedStore) or not hasattr(
        ReplicateConfig(), "replica_num"
    ):
        raise RuntimeError("Mooncake Store/ReplicateConfig API is unavailable")
    LOG.info("Native package preflight passed: %s", installed[0])


def check_ports(args):
    host = args.hosts[args.node_rank]
    sockets = []
    try:
        for address, port in [
            (host, args.etcd_port),
            (host, args.peer_port),
            (host, args.master_port),
            ("127.0.0.1", args.metrics_port),
            *(
                (host, p)
                for p in range(args.provider_ports[0], args.provider_ports[1] + 1)
            ),
        ]:
            sock = socket.socket()
            sockets.append(sock)
            sock.bind((address, port))
    finally:
        for sock in sockets:
            sock.close()


@contextmanager
def deployment_lock(args):
    import fcntl

    args.data_dir.mkdir(parents=True, exist_ok=True)
    with (args.data_dir / "launcher.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("A launcher already owns this data-dir") from exc
        identity = {
            "mode": args.mode,
            "hosts": args.hosts,
            "cluster_id": args.cluster_id,
            "node_rank": args.node_rank,
            "etcd_port": args.etcd_port,
            "peer_port": args.peer_port,
        }
        manifest = args.data_dir / "deployment.json"
        if manifest.exists():
            saved = json.loads(manifest.read_text(encoding="utf-8"))
            # Manifests generated before mode selection supported HA only.
            saved.setdefault("mode", "ha")
            if saved != identity:
                raise RuntimeError(
                    "data-dir belongs to a different cluster/topology; do not reuse its etcd data"
                )
        elif (args.data_dir / "etcd").exists():
            raise RuntimeError(
                "Unmanaged etcd data found; refusing to adopt or overwrite it"
            )
        else:
            write_json(manifest, identity)
        yield


def setup_store(args, host, *, provider=False):
    os.environ.pop("MC_STORE_USE_HUGEPAGE", None)
    os.environ.update(environment(args, host, provider=provider))
    from mooncake.store import MooncakeDistributedStore

    store = MooncakeDistributedStore()
    try:
        rc = store.setup(
            local_hostname=host,
            metadata_server=endpoint(args),
            master_server_addr=master_endpoint(args),
            protocol="tcp",
            rdma_devices="",
            global_segment_size=args.pool_mib * MIB if provider else 0,
            local_buffer_size=args.buffer_mib * MIB,
            enable_ssd_offload=False,
            enable_client_http_server=False,
        )
        if rc != 0:
            raise RuntimeError(f"Mooncake setup returned {rc}")
    except BaseException:
        store.close()
        raise
    return store


def stop_event():
    event = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: event.set())
    return event


def run_provider(args):
    stop = stop_event()
    marker = args.data_dir / "provider-ready.json"
    store = setup_store(args, args.hosts[args.node_rank], provider=True)
    try:
        write_json(marker, {"pid": os.getpid(), "pool_mib": args.pool_mib})
        LOG.info("Provider mounted %s MiB", args.pool_mib)
        stop.wait()
    finally:
        marker.unlink(missing_ok=True)
        store.close()


def read_http(url):
    # Cluster health must not go through an inherited HTTP proxy.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    with opener.open(url, timeout=2) as response:
        return response.read(1024 * 1024).decode("utf-8")


def node_healthy(args):
    try:
        host = args.hosts[args.node_rank]
        health = json.loads(read_http(f"http://{host}:{args.etcd_port}/health"))
        summary = read_http(f"http://127.0.0.1:{args.metrics_port}/metrics/summary")
        master_ok = "role=leader, state=serving, service_ready=true" in summary or (
            args.mode == "ha" and "role=standby, state=standby" in summary
        )
        return health.get("health") in (True, "true") and master_ok
    except (OSError, ValueError):
        return False


def terminate_children(children, timeout):
    # Only process groups we created; never pkill/killall or reuse saved PIDs.
    alive = [p for p in children if p.poll() is None]
    for process in alive:
        try:
            os.killpg(process.pid, signal.SIGTERM)
        except ProcessLookupError:
            pass
    deadline = time.monotonic() + timeout
    for process in alive:
        try:
            process.wait(timeout=max(0, deadline - time.monotonic()))
        except subprocess.TimeoutExpired:
            try:
                os.killpg(process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            process.wait()


def supervise(args):
    stop = stop_event()
    plan = build_plan(args)
    env = child_environment(args, args.hosts[args.node_rank], provider=True)
    ready = args.data_dir / "node-ready.json"
    provider_ready = args.data_dir / "provider-ready.json"
    ready.unlink(missing_ok=True)
    provider_ready.unlink(missing_ok=True)
    children, logs, exits, retry_at = {}, {}, {name: [] for name in plan}, {}
    started = time.monotonic()
    was_ready = False
    ever_ready = False
    try:
        for name in plan:
            logs[name] = (args.data_dir / f"{name}.log").open("ab", buffering=0)
        while not stop.is_set():
            now = time.monotonic()
            for name, command in plan.items():
                process = children.get(name)
                if process is not None and process.poll() is not None:
                    LOG.warning(
                        "%s exited with %s; see %s.log", name, process.returncode, name
                    )
                    ready.unlink(missing_ok=True)
                    was_ready = False
                    if name == "provider":
                        provider_ready.unlink(missing_ok=True)
                    exits[name] = [t for t in exits[name] if now - t < 300] + [now]
                    if len(exits[name]) > args.max_restarts:
                        raise RuntimeError(
                            f"{name} exceeded restart budget; inspect {args.data_dir / (name + '.log')}"
                        )
                    retry_at[name] = now + min(2 ** len(exits[name]), 15)
                    children.pop(name)
                if name not in children and now >= retry_at.get(name, 0):
                    children[name] = subprocess.Popen(
                        command,
                        env=env,
                        stdout=logs[name],
                        stderr=subprocess.STDOUT,
                        start_new_session=True,
                    )
                    LOG.info("Started %s pid=%s", name, children[name].pid)
            healthy = (
                len(children) == len(plan)
                and all(p.poll() is None for p in children.values())
                and provider_ready.exists()
                and node_healthy(args)
            )
            if healthy and not was_ready:
                write_json(
                    ready,
                    {
                        "status": "node-ready",
                        "mode": args.mode,
                        "host": args.hosts[args.node_rank],
                        "cluster_id": args.cluster_id,
                        "pid": os.getpid(),
                    },
                )
                LOG.info(
                    "Node ready. Run 'check' from the RL host before starting training."
                )
                ever_ready = True
            elif not healthy:
                ready.unlink(missing_ok=True)
                if was_ready:
                    LOG.warning(
                        "Node degraded; waiting for local services/quorum to recover"
                    )
            was_ready = healthy
            if not ever_ready and now - started > args.startup_timeout:
                raise RuntimeError(
                    "Startup timed out; ensure all configured nodes are running and inspect service logs"
                )
            stop.wait(2)
    finally:
        ready.unlink(missing_ok=True)
        terminate_children(
            list(reversed(list(children.values()))), args.shutdown_timeout
        )
        provider_ready.unlink(missing_ok=True)
        for handle in logs.values():
            handle.close()


def replica_hosts(store, key):
    from mooncake.store import ReplicaStatus

    return {
        replica.get_memory_descriptor().buffer_descriptor.transport_endpoint.rsplit(
            ":", 1
        )[0]
        for replica in store.get_replica_desc(key)
        if replica.status == ReplicaStatus.COMPLETE and replica.is_memory_replica()
    }


def run_probe(args):
    from mooncake.store import ReplicateConfig

    store = setup_store(args, args.local_host)
    key = "omni-ha-probe-" + uuid.uuid4().hex
    payload = os.urandom(MIB)
    try:
        replica = ReplicateConfig()
        replica.replica_num = 2 if args.mode == "ha" else 1
        rc = store.put(key, payload, replica)
        if rc != 0:
            raise RuntimeError(f"Probe put returned {rc}")
        hosts = replica_hosts(store, key)
        if len(hosts & set(args.hosts)) < replica.replica_num:
            count = "two" if args.mode == "ha" else "one"
            raise RuntimeError(
                f"Probe has fewer than {count} storage-host replicas: {sorted(hosts)}"
            )
        # A new pure client must read after the writer closes. No client hosts payload.
        store.close()
        store = None
        store = setup_store(args, args.local_host)
        if store.get(key) != payload:
            raise RuntimeError("Probe read from a new client did not match")
        LOG.info(
            "Cross-client PUT/GET and %s-host replicas passed (not a failover test)",
            replica.replica_num,
        )
    finally:
        if store is not None:
            try:
                # This UUID belongs only to the probe; force bypasses its read lease.
                if store.remove(key, force=True) != 0:
                    LOG.warning("Could not remove probe key %s", key)
            finally:
                store.close()
        else:
            LOG.warning("Probe reconnect failed; probe key may remain: %s", key)


def check_cluster(args):
    stop = stop_event()
    healthy = 0
    for host in args.hosts:
        try:
            value = json.loads(read_http(f"http://{host}:{args.etcd_port}/health"))
            if value.get("health") in (True, "true"):
                healthy += 1
        except (OSError, ValueError):
            LOG.warning("etcd endpoint unavailable: %s", host)
    if args.mode == "ha" and healthy < 2:
        raise RuntimeError("Fewer than two healthy etcd endpoints; no HA quorum")
    if args.mode == "standalone" and healthy < 1:
        raise RuntimeError("Standalone etcd endpoint is unavailable")
    if healthy < len(args.hosts):
        LOG.warning(
            "Cluster is degraded: only %s/%s etcd endpoints healthy",
            healthy,
            len(args.hosts),
        )
    process = subprocess.Popen(
        child_command(args, "probe"),
        start_new_session=True,
        env=child_environment(args, args.local_host),
    )
    try:
        deadline = time.monotonic() + args.startup_timeout
        while not stop.is_set():
            try:
                if process.wait(timeout=1) != 0:
                    raise RuntimeError("Mooncake read/write probe failed")
                return
            except subprocess.TimeoutExpired:
                if time.monotonic() >= deadline:
                    raise RuntimeError("Mooncake read/write probe timed out")
        raise RuntimeError("Mooncake read/write probe interrupted")
    finally:
        terminate_children([process], args.shutdown_timeout)


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s"
    )
    try:
        bootstrap_image_environment(args, argv)
        if args.child == "dependency":
            check_native_dependency()
        elif args.child == "provider":
            run_provider(args)
        elif args.child == "probe":
            run_probe(args)
        elif args.action == "client-config":
            generate_client(args)
        elif args.dry_run:
            plan = (
                build_plan(args)
                if args.action == "start"
                else {"probe": child_command(args, "probe")}
            )
            host = (
                args.hosts[args.node_rank]
                if args.action == "start"
                else args.local_host
            )
            print(
                json.dumps(
                    {
                        "commands": plan,
                        "environment": environment(
                            args, host, provider=args.action == "start"
                        ),
                    },
                    indent=2,
                )
            )
        elif args.action == "check":
            if sys.platform != "linux":
                raise RuntimeError(
                    "Cluster probes require Linux and an etcd-enabled Mooncake installation"
                )
            check_cluster(args)
        else:
            check_dependencies(args)
            if args.check_only:
                check_ports(args)
                LOG.info("Local dependencies and ports passed; no services started")
            else:
                with deployment_lock(args):
                    check_ports(args)
                    supervise(args)
    except (OSError, ValueError, RuntimeError, subprocess.SubprocessError) as exc:
        LOG.error("%s", exc)
        return 1
    except KeyboardInterrupt:
        return 130
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
