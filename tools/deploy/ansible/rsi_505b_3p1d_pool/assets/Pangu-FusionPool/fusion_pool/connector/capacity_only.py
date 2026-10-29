# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Capacity-only support for the NPUMooncakeStore connector trio.

Mirrors upstream vLLM PR #50498: an instance with ``kv_role=kv_consumer``
and extra_config ``enable_lookup=false`` contributes its mounted segment to
the shared store pool but performs no lookup (get) and produces no saves
(put). v0.25.1 has no such support, so the guard lives entirely in this
package's subclasses: the ``is_capacity_only`` predicate drives every
behavior gate, and ``suppress_lookup_channel`` keeps capacity-only engines
off master's lookup IPC channel inside the ``super().__init__`` windows
(the root fix for the per-host shared-socket unlink race). Master symbols
are reached lazily via ``sys.modules``/``import_module``, so module import
stays light enough for unit tests with duck-typed configs.
"""

from __future__ import annotations

import inspect
import sys
from contextlib import contextmanager
from importlib import import_module


def is_capacity_only(vllm_config) -> bool:
    """True when this instance must run in capacity-only mode."""
    ktc = getattr(vllm_config, "kv_transfer_config", None)
    if ktc is None:
        return False
    if ktc.kv_role != "kv_consumer":
        return False
    extra = getattr(ktc, "kv_connector_extra_config", None) or {}
    return not extra.get("enable_lookup", True)


_WORKER_MOD = "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker"
_SCHED_MOD = "vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.scheduler"


class OfflineLookupKeyClient:
    """Null-object for master ``LookupKeyClient`` on capacity-only instances.

    Surface mirrors every master reference: ``lookup`` (normally gated off
    by ``enable_lookup`` in NPUMooncakeStoreScheduler; 0 = no external
    hit), ``discard`` (master ``build_connector_meta`` calls it per finished
    request each step), ``reset`` (RL admin channel), ``close`` (PFC
    connector closes the master-built scheduler's client when swapping in
    the NPU one).
    """

    def __init__(self, vllm_config, *args, **kwargs):
        pass

    def lookup(self, *args, **kwargs) -> int:
        return 0

    def discard(self, req_id):
        pass

    def reset(self) -> bool:
        return False

    def close(self):
        pass


def _no_lookup_key_server(store_worker, vllm_config):
    """Master assigns ``lookup_server`` but never calls methods on it."""
    return None


@contextmanager
def suppress_temporary_store_mount():
    """Prevent master's temporary worker from mounting a store segment."""
    worker_mod = sys.modules.get(_WORKER_MOD) or import_module(_WORKER_MOD)
    store_config_cls = worker_mod.MooncakeStoreConfig
    original_descriptor = inspect.getattr_static(
        store_config_cls, "load_from_config"
    )
    original_load = store_config_cls.load_from_config

    def no_mount_config():
        config = original_load()
        config.global_segment_size = 0
        config.local_buffer_size = 0
        return config

    store_config_cls.load_from_config = staticmethod(
        no_mount_config
    )
    try:
        yield
    finally:
        store_config_cls.load_from_config = original_descriptor


@contextmanager
def suppress_lookup_channel(vllm_config):
    """Neutralize master's lookup channel construction when capacity-only.

    Non-capacity instances (P side) pass through untouched. Both module
    namespaces are swapped for the whole window: worker-side construction
    resolves ``LookupKeyServer`` from the worker module, scheduler-side
    resolves ``LookupKeyClient`` from the scheduler module (a ``from``-
    import binding, so patching the defining module alone would not take
    effect).
    """
    if not is_capacity_only(vllm_config):
        yield
        return

    # Guaranteed already imported by every caller's import chain. The
    # sys.modules lookup keeps unit-test stub layers (which register only
    # leaf modules) working without the parent-package import machinery.
    worker_mod = sys.modules.get(_WORKER_MOD) or import_module(_WORKER_MOD)
    sched_mod = sys.modules.get(_SCHED_MOD) or import_module(_SCHED_MOD)

    orig_server = worker_mod.LookupKeyServer
    orig_client = sched_mod.LookupKeyClient
    worker_mod.LookupKeyServer = _no_lookup_key_server
    sched_mod.LookupKeyClient = OfflineLookupKeyClient
    try:
        yield
    finally:
        worker_mod.LookupKeyServer = orig_server
        sched_mod.LookupKeyClient = orig_client
