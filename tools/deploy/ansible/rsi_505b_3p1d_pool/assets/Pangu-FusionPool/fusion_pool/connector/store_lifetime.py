# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Torch-free ownership ledger for asynchronous reads of KV allocations."""

from dataclasses import dataclass, field
import threading
import uuid


@dataclass
class PendingStore:
    request_id: str
    blocks: frozenset[int]
    completed_ranks: set[int] = field(default_factory=set)


class StoreLedger:
    def __init__(self, num_workers: int):
        if num_workers < 1:
            raise ValueError("A store job needs at least one participant")
        self.epoch = uuid.uuid4().hex
        self.ranks = frozenset(range(num_workers))
        self.jobs: dict[str, PendingStore] = {}
        self.by_block: dict[int, set[str]] = {}
        self.sequence = 0
        self.retired: set[str] = set()

    def add(self, request_id, blocks):
        self.sequence += 1
        job_id = f"{self.epoch}:{self.sequence}"
        job = PendingStore(request_id, frozenset(blocks))
        self.jobs[job_id] = job
        for block in job.blocks:
            self.by_block.setdefault(block, set()).add(job_id)
        return job_id

    def conflicts(self, blocks):
        return set().union(*(self.by_block.get(b, ()) for b in blocks))

    def for_request(self, request_id):
        return {j for j, state in self.jobs.items() if state.request_id == request_id}

    def report(self, job_id, rank, error=None):
        if not job_id.startswith(self.epoch + ":"):
            raise RuntimeError("Store completion from a different engine epoch")
        if rank not in self.ranks:
            raise RuntimeError(f"Unexpected store participant {rank}")
        if error:
            # Failure is NOT proof that DMA is quiescent. Keep the dependency.
            raise RuntimeError(f"Store job {job_id} failed on rank {rank}: {error}")
        job = self.jobs.get(job_id)
        if job is None:
            # A duplicate after retirement is harmless; unissued IDs are not.
            sequence = int(job_id.rsplit(":", 1)[1])
            if not 0 < sequence <= self.sequence:
                raise RuntimeError(f"Unknown store job {job_id}")
            return
        job.completed_ranks.add(rank)
        if job.completed_ranks != self.ranks:
            return
        for block in job.blocks:
            readers = self.by_block[block]
            readers.remove(job_id)
            if not readers:
                del self.by_block[block]
        del self.jobs[job_id]
        self.retired.add(job_id)


class LocalStoreJobs:
    """Events exist before enqueue; waits never depend on scheduler progress."""

    def __init__(self):
        self.lock = threading.Lock()
        self.events = {}
        self.errors = {}
        self.reports = {}

    def register(self, job_id):
        with self.lock:
            if job_id in self.events:
                raise RuntimeError(f"Duplicate store dispatch {job_id}")
            self.events[job_id] = threading.Event()

    def complete(self, job_id, error=None):
        with self.lock:
            self.errors[job_id] = error
            self.reports[job_id] = error
            self.events[job_id].set()

    def wait(self, job_ids, timeout=120):
        for job_id in job_ids:
            with self.lock:
                event = self.events.get(job_id)
            if event is None:
                raise RuntimeError(f"Fence references undispatched store {job_id}")
            if not event.wait(timeout):
                raise RuntimeError(f"Store source fence timed out: {job_id}")
            with self.lock:
                error = self.errors.get(job_id)
            if error:
                raise RuntimeError(f"Store source is not known safe: {job_id}: {error}")

    def take_reports(self):
        with self.lock:
            reports, self.reports = self.reports, {}
        return reports

    def acknowledge(self, job_ids):
        with self.lock:
            for job_id in job_ids:
                event = self.events.get(job_id)
                if event is None or not event.is_set() or self.errors.get(job_id):
                    raise RuntimeError(f"Premature store retirement: {job_id}")
                del self.events[job_id]
                self.errors.pop(job_id, None)

    def drain(self):
        with self.lock:
            pending = tuple(self.events)
        self.wait(pending)
