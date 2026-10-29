# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Opt-in GET observation. Descriptor locality is not actual replica selection."""

import json
import logging
import time

logger = logging.getLogger(__name__)


def classify_source(descriptors, local_ip):
    # Multiple replicas are deliberately unknown: Python does not see which
    # replica the transfer implementation selects (or retries against).
    if not descriptors or len(descriptors) != 1:
        return "unknown", None
    replica = descriptors[0]
    predicate = getattr(replica, "is_memory_replica", None)
    if not callable(predicate) or not predicate():
        return "unknown", None
    # Mooncake's Python binding differs from the master's admin JSON:
    # ReplicaDescriptor -> MemoryDescriptor -> Descriptor.transport_endpoint.
    memory = replica.get_memory_descriptor()
    endpoint = getattr(memory.buffer_descriptor, "transport_endpoint", None)
    if not isinstance(endpoint, str) or ":" not in endpoint:
        return "unknown", None
    host = endpoint.rsplit(":", 1)[0].strip("[]")
    return ("local" if host == local_ip else "remote"), host


class LocalityStore:
    def __init__(self, store, local_ip, rank, request_id, emit):
        self._store = store
        self._local_ip = local_ip
        self._rank = rank
        self._request_id = request_id
        self._emit = emit
        self._keys = None
        self._descriptors = None
        self._lookup_ms = 0.0
        self._batch = 0

    def __getattr__(self, name):
        return getattr(self._store, name)

    def batch_get_replica_desc(self, keys):
        if tuple(keys) != self._keys:
            start = time.perf_counter()
            try:
                self._descriptors = self._store.batch_get_replica_desc(keys)
                self._keys = tuple(keys)
            finally:
                self._lookup_ms += (time.perf_counter() - start) * 1000
        return self._descriptors

    def batch_get_into_multi_buffers(self, keys, *args, **kwargs):
        sources = [("unknown", None)] * len(keys)
        try:
            descriptors = self.batch_get_replica_desc(keys)
            for i, key in enumerate(keys):
                try:
                    sources[i] = classify_source(descriptors[key], self._local_ip)
                except Exception:
                    logger.exception("Unable to classify Mooncake replica locality")
        except Exception:
            logger.exception("Unable to query Mooncake replica locality")
        start = time.perf_counter()
        result = None
        try:
            result = self._store.batch_get_into_multi_buffers(keys, *args, **kwargs)
            return result
        finally:
            end = time.perf_counter()
            try:
                valid = result is not None and len(result) == len(keys)
                totals = {}
                for i, (source, host) in enumerate(sources):
                    bucket = totals.setdefault((source, host), {
                        "source": source, "source_host": host,
                        "submitted_keys": 0, "success_keys": 0,
                        "failed_keys": 0, "unconfirmed_keys": 0,
                        "success_bytes": 0,
                    })
                    bucket["submitted_keys"] += 1
                    if not valid:
                        bucket["unconfirmed_keys"] += 1
                    elif result[i] < 0:
                        bucket["failed_keys"] += 1
                    else:
                        bucket["success_keys"] += 1
                        bucket["success_bytes"] += int(result[i])
                self._emit("Mooncake locality: " + json.dumps({
                    "schema": 1, "attribution": "single_replica_descriptor",
                    "reader_ip": self._local_ip, "rank": self._rank,
                    "request_id": self._request_id, "batch_id": self._batch,
                    "get_start_monotonic": start, "get_end_monotonic": end,
                    "get_ms": (end - start) * 1000,
                    "descriptor_ms": self._lookup_ms,
                    "result_complete": valid, "sources": list(totals.values()),
                }, separators=(",", ":")))
            except Exception:
                logger.exception("Unable to emit Mooncake locality observation")
            self._batch += 1
            self._keys = self._descriptors = None
            self._lookup_ms = 0.0
