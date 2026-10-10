# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Opt-in early CPU groups; no inferred NPU-to-NUMA mapping or page reservation."""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path


logger = logging.getLogger(__name__)
# Leave two logical CPUs per NUMA node outside TP worker groups as scheduling
# headroom for other processes. This does not isolate CPUs or bind other tasks.
_NUMA_HEADROOM_CPUS = 2


def parse_cpu_set(value: str) -> set[int]:
    if not isinstance(value, str) or not value.strip():
        raise ValueError("Worker CPU set must be a nonempty Linux CPU-list string")
    result = set()
    for item in value.split(","):
        parts = item.strip().split("-")
        if len(parts) not in (1, 2) or not all(p.isdecimal() for p in parts):
            raise ValueError(f"Invalid worker CPU list: {value!r}")
        start, end = int(parts[0]), int(parts[-1])
        if end < start or end >= 8192:
            raise ValueError(f"Invalid worker CPU range: {item!r}")
        result.update(range(start, end + 1))
    return result


def _read_numa_cpu_sets(allowed: set[int]) -> list[set[int]] | None:
    """Read CPU NUMA groups without probing or initializing an NPU."""
    nodes = [path for path in Path("/sys/devices/system/node").glob("node[0-9]*")
             if path.name[4:].isdecimal()]
    groups = []
    covered = set()
    for node in sorted(nodes, key=lambda path: int(path.name[4:])):
        value = (node / "cpulist").read_text().strip()
        cpus = parse_cpu_set(value) & allowed if value else set()
        if not cpus:
            continue
        if covered & cpus:
            logger.warning("Skipping auto CPU affinity: NUMA CPU lists overlap")
            return None
        covered.update(cpus)
        groups.append(cpus)
    if not groups or covered != allowed:
        logger.warning("Skipping auto CPU affinity: NUMA topology does not cover allowed CPUs")
        return None
    return groups


def _build_auto_cpu_groups(numa_cpu_sets: list[set[int]], tp_size: int) -> list[set[int]] | None:
    """Split NUMA CPUs for a validated TP size, leaving two CPUs per node."""
    node_count = len(numa_cpu_sets)
    if not node_count or tp_size % node_count:
        logger.warning("Skipping auto CPU affinity: TP=%s cannot be split across %s NUMA nodes",
                       tp_size, node_count)
        return None
    workers_per_node = tp_size // node_count
    groups = []
    for cpus in numa_cpu_sets:
        available = sorted(cpus)[_NUMA_HEADROOM_CPUS:]
        if len(available) < workers_per_node:
            logger.warning("Skipping auto CPU affinity: not enough CPUs after NUMA headroom")
            return None
        size, remainder = divmod(len(available), workers_per_node)
        offset = 0
        for index in range(workers_per_node):
            count = size + (index < remainder)
            groups.append(set(available[offset:offset + count]))
            offset += count
    return groups


def _resolve_worker_cpu_set(setting: str, worker_rank: int, parallel_config) -> set[int] | None:
    """Validate configuration and select CPUs without changing affinity."""
    if parallel_config.data_parallel_size != 1:
        logger.warning("Skipping CPU affinity for worker rank %s: only DP=1 TP workers are supported",
                       worker_rank)
        return None
    if setting.strip() == "auto":
        if parallel_config.pipeline_parallel_size != 1 or parallel_config.nnodes_within_dp != 1:
            logger.warning("Skipping auto CPU affinity for worker rank %s: single-host TP with PP=1 is required",
                           worker_rank)
            return None
        tp_size = parallel_config.tensor_parallel_size
        if (isinstance(tp_size, bool) or not isinstance(tp_size, int)
                or not 0 <= worker_rank < tp_size):
            logger.warning("Skipping auto CPU affinity: invalid worker rank %s or TP size %s",
                           worker_rank, tp_size)
            return None
        allowed = set(os.sched_getaffinity(0))
        nodes = _read_numa_cpu_sets(allowed)
        if nodes is None:
            return None
        groups = _build_auto_cpu_groups(nodes, tp_size)
        return groups[worker_rank] if groups is not None else None

    mapping = json.loads(setting)
    if not isinstance(mapping, dict) or str(worker_rank) not in mapping:
        logger.warning("Skipping CPU affinity: no CPU group configured for worker rank %s",
                       worker_rank)
        return None
    return parse_cpu_set(mapping[str(worker_rank)])


def _bind_worker_threads(requested: set[int]):
    """Bind the caller and existing threads, restoring changes on failure.

    Future threads inherit the caller's CPU group. Existing narrower or
    disjoint thread masks are never widened. No NUMA memory policy is changed.
    """
    original = set(os.sched_getaffinity(0))
    if not requested <= original:
        logger.warning("Skipping CPU affinity: requested CPUs %s exceed allowed CPUs %s",
                       sorted(requested), sorted(original))
        return None
    pid = os.getpid()
    changed = {}
    preserved = {}
    vanished = set()
    try:
        # Set the caller first. Newly created threads inherit this group.
        if original != requested:
            os.sched_setaffinity(0, requested)
            changed[pid] = original
        # Include threads created during the first enumeration. This runs
        # before device/transfer setup, rather than after the service is ready.
        for _ in range(2):
            for path in Path("/proc/self/task").iterdir():
                tid = int(path.name)
                if tid == pid:
                    continue
                try:
                    prior = set(os.sched_getaffinity(tid))
                    target = prior & requested
                    if not target:
                        preserved[tid] = sorted(prior)
                        continue
                    if prior != target:
                        os.sched_setaffinity(tid, target)
                        changed.setdefault(tid, prior)
                except ProcessLookupError:
                    vanished.add(tid)
    except BaseException:
        for tid, prior in changed.items():
            try:
                os.sched_setaffinity(tid, prior)
            except ProcessLookupError:
                pass
            except Exception as exc:
                logger.warning("Could not restore CPU affinity for thread %s: %s", tid, exc)
        raise
    return dict(pid=pid, cpus=sorted(requested),
                changed_threads=len(changed), preserved_prebound_threads=preserved,
                vanished_threads=sorted(vanished),
                scope="CPU groups only; device NUMA locality is not inferred")


def apply_worker_cpu_affinity(worker_rank: int, parallel_config) -> dict | None:
    """Apply opt-in CPU groups before device initialization.

    The caller checks whether CPU affinity is enabled. JSON selects explicit rank groups;
    "auto" splits allowed NUMA CPUs for single-host TP with DP=1 and PP=1.
    CPU headroom is outside worker groups, not reserved against other processes.
    Configuration or binding failures are logged and do not abort worker startup.
    Return the placement on success and None on failure.
    """
    try:
        setting = os.environ["OMNI_WORKER_CPUSET_MAP"]
        cpus = _resolve_worker_cpu_set(setting, worker_rank, parallel_config)
        if cpus is None:
            return None
        placement = _bind_worker_threads(cpus)
        if placement is None:
            return None
        placement.update(worker_rank=worker_rank,
                         mode="auto" if setting.strip() == "auto" else "explicit")
        return placement
    except Exception as exc:
        logger.warning("Skipping CPU affinity for worker rank %s: %s; continuing worker startup",
                       worker_rank, exc)
        return None
