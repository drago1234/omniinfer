# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""NPU MooncakeStoreConnector with tuple/list KV cache registration support.

Same shape as :class:`NPUOffloadingConnector`: master
``MooncakeStoreConnector.register_kv_caches`` assumes each attention layer is
a single full-page tensor and each mamba layer is a list. Pangu's NPU backends
register views as tuple/list of strided sub-tensors sharing one raw allocation
per layer. We pre-process those into the shape the framework expects before
delegating.
"""

from __future__ import annotations

from functools import wraps
import time

from vllm.config import VllmConfig
from vllm.distributed.kv_events import KVCacheEvent, KVEventBatch, NullEventPublisher
from vllm.distributed.kv_transfer.kv_connector.v1 import KVConnectorRole
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.connector import (
    MooncakeStoreConnector,
)
from vllm.logger import init_logger
from vllm.v1.kv_cache_interface import KVCacheConfig

from kv_common.npu_kv_views import (
    prepare_kv_caches_for_offloading_registration,
)
from fusion_pool.connector.npu_mooncake_store_worker import NPUMooncakeStoreWorker
from fusion_pool.connector.capacity_only import (
    suppress_lookup_channel,
    suppress_temporary_store_mount,
)

from fusion_pool.connector.npu_mooncake_store_scheduler import NPUMooncakeStoreScheduler

logger = init_logger(__name__)


class PoolReset(KVCacheEvent):
    """The external pool changed; subscribers must reconcile, not assume empty."""


def _pending_connectors(connector):
    if connector is None:
        return
    if getattr(connector, "_pool_reset_event_pending", False):
        yield connector
    for child in getattr(connector, "_connectors", ()):
        yield from _pending_connectors(child)


def _publish_pending_resets(scheduler):
    pending = list(_pending_connectors(scheduler.connector))
    if not pending:
        return
    try:
        # Reuse this publisher, including its sequence counter and worker thread.
        # Putting the event in the normal inference queue instead can strand it
        # indefinitely when reset is followed by an idle engine.
        if (not scheduler.enable_kv_cache_events or
                isinstance(scheduler.kv_event_publisher, NullEventPublisher)):
            raise RuntimeError("engine KV events are disabled")
        batch = KVEventBatch(ts=time.time(), events=[PoolReset()])
        scheduler.kv_event_publisher.publish(batch)
    except Exception:
        # A failed notification must not turn an acknowledged store reset into
        # a failed reset. Retain pending state for the next reset-path attempt.
        logger.warning(
            "Pool reset completed, but PoolReset publication failed; "
            "subscribers must recover through periodic reconciliation",
            exc_info=True,
        )
    else:
        for connector in pending:
            connector._pool_reset_event_pending = False


def install_pool_reset_event_dispatch():
    """Attach once when a scheduler-role pool connector is constructed.

    This is an explicit compatibility hook for Scheduler.reset_connector_cache.
    It changes no scheduling/inference path and needs no Proxy address. The
    finally clause also reports a pool reset that succeeded before another
    MultiConnector child returned False or raised.
    """
    from vllm.v1.core.sched.scheduler import Scheduler

    original = Scheduler.reset_connector_cache
    if getattr(original, "_fusion_pool_reset_dispatch", False):
        return

    @wraps(original)
    def reset_connector_cache(scheduler, *args, **kwargs):
        try:
            return original(scheduler, *args, **kwargs)
        finally:
            _publish_pending_resets(scheduler)

    reset_connector_cache._fusion_pool_reset_dispatch = True
    Scheduler.reset_connector_cache = reset_connector_cache


class NPUMooncakeStoreConnector(MooncakeStoreConnector):
    """MooncakeStoreConnector that runs on Ascend NPU.

    Subclasses master ``MooncakeStoreConnector`` so that we inherit the
    lookup and prefix-cache behavior. Stores use a separate job ledger and
    source-reuse fences; they do not participate in request send completion.
    Worker construction and cache registration are Ascend-specific.
    """

    @staticmethod
    def _validate_kv_cache_config(vllm_config, kv_cache_config):
        from fusion_pool.connector.capacity_only import is_capacity_only

        if is_capacity_only(vllm_config):
            # Capacity-only transfers no KV (upstream PR #50498 skips these
            # checks too): the master's cache-shape invariants cannot be
            # reached, so don't fail startup on them.
            return None
        MooncakeStoreConnector._validate_kv_cache_config(
            vllm_config, kv_cache_config
        )
        return None

    def __init__(
        self,
        vllm_config: VllmConfig,
        role: KVConnectorRole,
        kv_cache_config: KVCacheConfig | None = None,
    ):
        # The master connector's __init__ builds a throwaway
        # MooncakeStoreWorker whose store mounts an 8 GB segment with the
        # master at an auto-bound host:port. We replace that worker with
        # NPUMooncakeStoreWorker below, but the original is abandoned without
        # unmounting -> the master keeps a phantom segment (most free space)
        # that the allocation strategy later picks for replicas; every put
        # targeting it fails at openSegment on the dead port (surfaced as
        # INTERNAL_ERROR). Zero the sizes the original worker reads so it
        # mounts no segment, then restore before the NPU worker is built.
        with suppress_temporary_store_mount():
            # Capacity-only engines must not build the lookup channel at
            # all: master's throwaway worker server AND the master-built
            # scheduler's client become no-ops inside this window.
            with suppress_lookup_channel(vllm_config):
                super().__init__(vllm_config, role, kv_cache_config)

        if role == KVConnectorRole.SCHEDULER:
            self.connector_scheduler.client.close()
            self.connector_scheduler = NPUMooncakeStoreScheduler(vllm_config, kv_cache_config)
            self._pool_reset_event_pending = False
            install_pool_reset_event_dispatch()

        # Replace the master-built worker (RDMA-only) with our NPU one.
        # The super constructor calls register_cross_layers_kv_caches via
        # register_kv_caches later, so we must swap before that runs.
        if role != KVConnectorRole.SCHEDULER:
            if self.connector_worker is None:
                raise RuntimeError(
                    "MooncakeStoreConnector expected to have built a worker"
                )
            # Close the abandoned master worker's store (no segment was
            # mounted thanks to the config patch above) so its TransferEngine
            # and client connection are released instead of leaking.
            old_worker = self.connector_worker
            try:
                if getattr(old_worker, "store", None) is not None:
                    old_worker.store.close()
            except Exception:  # pragma: no cover — best-effort cleanup
                pass
            vllm_config._local_rank = getattr(vllm_config, "_local_rank", 0)
            self.connector_worker = NPUMooncakeStoreWorker(
                vllm_config, kv_cache_config
            )

    def register_kv_caches(self, kv_caches) -> None:
        if self._kv_cache_config is None:
            raise RuntimeError(
                "register_kv_caches requires _kv_cache_config to be set"
            )
        prepared = prepare_kv_caches_for_offloading_registration(
            kv_caches,
            self._kv_cache_config,
        )
        super().register_kv_caches(prepared)

    def bind_gpu_block_pool(self, block_pool):
        if self.connector_scheduler is not None:
            self.connector_scheduler.bind_gpu_block_pool(block_pool)

    def reset_cache(self):
        if self.connector_scheduler is not None and self.connector_scheduler.has_pending_push_work():
            return False
        result = super().reset_cache()
        if result is True:
            # Only an acknowledged external reset produces PoolReset. HBM reset
            # and worker-role no-ops keep their existing event semantics.
            self._pool_reset_event_pending = True
        return result

    def handle_preemptions(self, kv_connector_metadata):
        self.connector_worker.handle_store_fences(kv_connector_metadata)

    def wait_for_save(self):
        # The upstream store connector leaves this hook empty. vLLM invokes it
        # after MTP drafting, which is the first safe point for a whole-model PUT.
        self.connector_worker.wait_for_save(self._get_connector_metadata())

    def build_connector_worker_meta(self):
        return self.connector_worker.build_store_worker_meta()

    def update_connector_output(self, connector_output):
        super().update_connector_output(connector_output)
        self.connector_scheduler.update_store_output(connector_output)

    def has_pending_push_work(self):
        return self.connector_scheduler is not None and self.connector_scheduler.has_pending_push_work()

    # ------------------------------------------------------------------
    # Lookup diagnostics override.
    #
    # Scheduler end of the external prefix-cache lookup chain. The master
    # ``MooncakeStoreScheduler.get_num_new_matched_tokens`` early-returns
    # ``(0, False)`` when ``request.num_tokens // block_size * block_size``
    # is below ``block_size`` (short prompts never reach the store), then
    # issues ``client.lookup(...)`` over ZMQ to the worker rank-0
    # LookupKeyServer. Logging the inputs and result here tells us whether
    # the lookup is attempted at all and what it returns, without touching
    # the vllm package.
    # ------------------------------------------------------------------
    def get_num_new_matched_tokens(
        self,
        request,
        num_computed_tokens: int,
    ) -> tuple[int | None, bool]:
        sched_probe = getattr(self, "connector_scheduler", None)
        if getattr(sched_probe, "enable_lookup", True) is False:
            # Capacity-only: the scheduler returns (0, False) without
            # consulting the store; skip the per-request INFO logs too.
            return 0, False
        sched = getattr(self, "connector_scheduler", None)
        block_size = getattr(sched, "_block_size", None)
        num_hashes = len(request.block_hashes) if request.block_hashes else 0
        logger.info(
            "[NPUMooncake][scheduler] lookup req=%s num_tokens=%d "
            "computed=%d num_hashes=%d block_size=%s",
            request.request_id,
            request.num_tokens,
            num_computed_tokens,
            num_hashes,
            block_size,
        )
        result = super().get_num_new_matched_tokens(
            request, num_computed_tokens
        )
        logger.info(
            "[NPUMooncake][scheduler] lookup req=%s -> %s",
            request.request_id,
            result,
        )
        return result
