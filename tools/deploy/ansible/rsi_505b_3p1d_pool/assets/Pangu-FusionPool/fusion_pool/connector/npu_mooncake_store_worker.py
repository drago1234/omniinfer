# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
#
# NPU adapter for vLLM master MooncakeStoreWorker.
#
# Goal: keep every byte of master scheduling / transfer-thread logic intact
# (dataclasses, ChunkedTokenDatabase, KVCacheStoreSendingThread, etc.) and
# only replace the two NPU-specific pieces:
#   1) TransferEngine creation (must use Ascend transport "ascend")
#   2) KV cache HBM buffer registration (must use the same TransferEngine
#      instance that store.setup was injected with, otherwise D2H/H2D is
#      rejected by AdxlEngine)
#
# Strategy: subclass master ``MooncakeStoreWorker`` and, immediately after
# ``super().__init__``, swap ``self.store`` for an NPU-built one backed by the
# GlobalTE singleton, then override ``register_kv_caches`` to register HBM
# through that same singleton instead of ``self.store.register_buffer``.
#
# SSD storage remains owned by the embedded Mooncake RealClient.

from __future__ import annotations

import json
import os
import threading

from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker import (
    MooncakeStoreWorker,
)
from vllm.utils.network_utils import get_ip
from vllm.v1.kv_cache_interface import (
    KVCacheConfig,
    MLAAttentionSpec,
    MambaSpec,
    SlidingWindowMLASpec,
    UniformTypeKVCacheSpecs,
)

from fusion_pool.transfer_engine import global_te
from fusion_pool.connector.store_lifetime_worker import (
    PoolStoreSendingThread,
    LoadEventWorkerMixin,
    StoreLifetimeWorkerMixin,
    PoolStoreReceivingThread,
)
from fusion_pool.connector.store_lookup import AscendLookupWorkerMixin
from fusion_pool.connector.ssd_store_options import ssd_store_options
from fusion_pool.connector.capacity_only import (
    suppress_lookup_channel,
    suppress_temporary_store_mount,
)
from fusion_pool.connector.store_events import load_events_requested, store_events_requested

# Lazy logger; vllm.init_logger can fire import-time side effects in some
# container setups, so defer.
_logger = None


def _log():
    global _logger
    if _logger is None:
        from vllm.logger import init_logger

        _logger = init_logger(__name__)
    return _logger


def _invalid_mooncake_tenant_id(tenant_id: object) -> bool:
    if not isinstance(tenant_id, str):
        return True
    if not tenant_id:
        return True
    if tenant_id.startswith("_"):
        return True
    if any(ord(char) < 32 or ord(char) == 127 for char in tenant_id):
        return True
    return False


def _get_local_ip() -> str:
    """Return a routable IPv4 address for the host.

    AscendDirectTransport's CheckIp rejects hostnames; vLLM's
    ``get_ip`` falls back to the enp* interface IP, which is exactly
    what we verified in step 1 of the validation pass.
    """
    return get_ip()


class NPUMooncakeStoreWorker(
    AscendLookupWorkerMixin,
    LoadEventWorkerMixin,
    StoreLifetimeWorkerMixin,
    MooncakeStoreWorker,
):
    """MooncakeStoreWorker that runs on Ascend NPU with injected TE.

    Master creates ``MooncakeDistributedStore()`` and calls ``store.setup(...)``
    with ``protocol='rdma'`` and no ``engine=`` kwarg — neither works on NPU.
    We rebuild the store with ``protocol='ascend'`` + an engine from the
    ``GlobalTE`` singleton, then override ``register_kv_caches`` to register
    HBM through the same singleton (so AdxlEngine's D2H/H2D path is valid).
    """

    def __init__(
        self,
        vllm_config: VllmConfig,
        kv_cache_config: KVCacheConfig,
    ):
        from fusion_pool.connector.capacity_only import is_capacity_only

        # Upstream PR #50498 mirror: decided before any mount/registration
        # work. Flips the existing register_kv_caches guard below.
        self._capacity_only = is_capacity_only(vllm_config)
        self._load_events_requested = load_events_requested(vllm_config)
        self._store_events_requested = store_events_requested(vllm_config)
        self._init_group_lookup()

        # 1) Run the master init. This builds the scheduler-side dataclasses
        #    (ChunkedTokenDatabase, MooncakeStoreCoordinator, etc.), the
        #    LookupKeyServer on rank 0, the transfer threads, etc. The store
        #    it creates on the way is useless on NPU, but the rest of the data
        #    structures are exactly what we want to reuse.
        #
        #    The master-built store mounts a segment with the master at an
        #    auto-bound host:port; close() does not unmount it, leaving a
        #    phantom segment (most free space) that the allocation strategy
        #    later picks for replicas -> openSegment on the dead port -> put
        #    INTERNAL_ERROR. So zero the sizes it reads: setup(size=0) creates
        #    the client but mounts no segment. Restore the real config for the
        #    NPU rebuild below.
        from fusion_pool.connector.store_lookup_channel import hbm_lookup_channel

        with suppress_temporary_store_mount(), hbm_lookup_channel(vllm_config, "worker"):
            # Capacity-only engines must not build the lookup channel at
            # all: master's LookupKeyServer creation inside super().__init__
            # becomes a None assignment inside this window.
            with suppress_lookup_channel(vllm_config):
                super().__init__(vllm_config, kv_cache_config)

        # 2) Replace self.store with an Ascend-backed one driven by GlobalTE.
        self._rebuild_store_with_global_te(vllm_config)

    # ------------------------------------------------------------------
    # NPU store rebuild
    # ------------------------------------------------------------------
    def _rebuild_store_with_global_te(self, vllm_config: VllmConfig) -> None:
        from mooncake.store import MooncakeDistributedStore  # type: ignore

        # Local imports keep vllm-package-dependent symbols out of module
        # import time (helpful when the vllm build lacks the latest master
        # helpers).
        from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker import (
            DEFAULT_MOONCAKE_DISK_STAGING_BUFFER_BYTES,
            MooncakeStoreConfig,
        )

        log = _log()
        store_config = MooncakeStoreConfig.load_from_config()

        # prefer_alloc_in_same_node lives in mooncake.json but master
        # ``MooncakeStoreConfig`` doesn't parse it. Read it here from the
        # raw JSON so we can stamp the runtime ReplicateConfig directly
        # (mirroring vllm-ascend MooncakeBackend.put).
        same_node = True
        raw = {}
        config_path = os.environ.get("MOONCAKE_CONFIG_PATH")
        if config_path:
            with open(config_path) as f:
                raw = json.load(f)
            same_node = bool(raw.get("prefer_alloc_in_same_node", True))
        tenant_id = raw.get("tenant_id", "default")
        if _invalid_mooncake_tenant_id(tenant_id):
            raise ValueError("Invalid Mooncake tenant_id")

        ssd_kwargs, ssd_buffer_budget_in_bytes = ssd_store_options(
            raw,
            dp_rank=vllm_config.parallel_config.data_parallel_rank,
            tp_rank=self.tp_rank,
            buffer_bytes=os.environ.get(
                "MOONCAKE_OFFLOAD_LOCAL_BUFFER_SIZE_BYTES",
                DEFAULT_MOONCAKE_DISK_STAGING_BUFFER_BYTES,
            ),
        )
        if ssd_buffer_budget_in_bytes is not None:
            self.disk_offload_buffer_budget_bytes = ssd_buffer_budget_in_bytes

        # Force protocol='ascend' regardless of what mooncake.json says.
        if store_config.protocol != "ascend":
            log.warning(
                "Overriding mooncake.json protocol=%r -> 'ascend' for NPU.",
                store_config.protocol,
            )
            store_config.protocol = "ascend"

        local_ip = _get_local_ip()
        # Initialise the TransferEngine with device_name=None (empty string),
        # exactly like vllm-ascend MooncakeBackend._setup_store. The Ascend
        # transport enumerates every visible device for an empty device_name,
        # so batch_put/batch_get that span TP ranks' HBM no longer take the old
        # "Dest buffer device_id=N >= endpoint count 1" cross-device TCP
        # fallback that initialised HCCL and crashed with the A2
        # rtGetSocVersion illegal-chipver error. Passing a concrete device
        # string here (e.g. str(current_device())) was the bug that pinned the
        # TE to a single device. NB: on this CANN 9.1.0 host the cross-device
        # data path still returns an error (-800) instead of crashing; the
        # real cross-container P/D load is validated end-to-end.
        device_name = None

        # Initialise the global TransferEngine once. If a sibling component
        # has already done so, we reuse the same instance.
        transfer_engine = global_te.get_transfer_engine(local_ip, device_name)
        engine_obj = transfer_engine.get_engine()
        local_seg = f"{local_ip}:{transfer_engine.get_rpc_port()}"

        new_store = MooncakeDistributedStore()
        ret = new_store.setup(
            local_hostname=local_seg,
            metadata_server=store_config.metadata_server,
            global_segment_size=store_config.global_segment_size,
            local_buffer_size=store_config.local_buffer_size,
            protocol=store_config.protocol,
            rdma_devices=store_config.device_name,
            master_server_addr=store_config.master_server_address,
            engine=engine_obj,
            tenant_id=tenant_id,
            enable_ssd_offload=ssd_kwargs.get("enable_ssd_offload", False),
            ssd_offload_path=ssd_kwargs.get("ssd_offload_path", ""),
        )
        if ret != 0:
            raise RuntimeError(
                "Initialize MooncakeDistributedStore on NPU failed "
                f"(ret={ret}, master={store_config.master_server_address}). "
                "Verify mooncake master is reachable and protocol='ascend'."
            )
        if ssd_kwargs:
            log.info(
                "Mooncake SSD enabled path=%s receive_budget=%.2f GiB",
                ssd_kwargs["ssd_offload_path"],
                ssd_buffer_budget_in_bytes / (1024 ** 3),
            )

        # Tear down the master-built store. It contributed no segment (see
        # __init__), so closing it is cheap and leaves no phantom behind.
        try:
            self.store.close()
        except Exception as e:  # pragma: no cover — best-effort cleanup
            log.debug("Ignoring error while closing master-built store: %s", e)

        self.store = new_store

        # Pin NPU stores to this worker's own segment. With mooncake
        # >= 0.3.11.post1 the Ascend transport enumerates every visible
        # device for device_name="" (endpoint count == visible devices), so
        # cross-device puts no longer hit the old "Dest buffer device_id=N >=
        # endpoint count 1" fallback. preferred_segment is kept as a defensive
        # choice to keep every put on the local device regardless, matching
        # vllm-ascend MooncakeBackend.put when preferred_segment is configured.
        self.store_replicate_config.preferred_segment = local_seg

        # Prefer allocating replicas on the same node as the producer. This
        # mirrors vllm-ascend MooncakeBackend.put (mooncake_backend.py:201).
        # Without this knob the master can pick a remote node's replica,
        # which on A3 trips Ascend TE cross-device TCP fallback -> HCCL
        # rtGetSocVersion reads an A2 chip_ver (Ascend910_9392) and crashes
        # the worker with Hccl::RuntimeApiException.
        if same_node:
            try:
                self.store_replicate_config.prefer_alloc_in_same_node = True
            except AttributeError:
                log.warning(
                    "MooncakeDistributedStore.ReplicateConfig has no "
                    "prefer_alloc_in_same_node attribute on this build; "
                    "load may fall back to cross-device sockets."
                )

        log.info(
            "NPUMooncakeStoreWorker: replaced store with Ascend-backed "
            "instance (segment=%s, master=%s, segment_size=%d, "
            "local_buffer=%d, device=%s).",
            local_seg,
            store_config.master_server_address,
            store_config.global_segment_size,
            store_config.local_buffer_size,
            device_name,
        )

        if getattr(self, "_capacity_only", False):
            # The segment above IS the capacity contribution. Drop the
            # lookup/transfer machinery (upstream #50498): no token
            # databases, no HBM registration, no load threads. Master's
            # rank-0 LookupKeyServer is already suppressed by the
            # lookup-offline window (see fusion_pool.connector.capacity_only);
            # the lookup() override below stays guarded defensively.
            self.token_dbs = []
            log.info(
                "Mooncake store in capacity-only mode: segment mounted "
                "(global_segment_size=%d), KV transfer disabled.",
                store_config.global_segment_size,
            )

    # ------------------------------------------------------------------
    # Buffer registration override: GlobalTE owns HBM registration so
    # that AdxlEngine can legally D2H/H2D the same memory we registered.
    # ------------------------------------------------------------------
    def register_kv_caches(
        self,
        kv_caches,
    ) -> None:
        """Resolve each entry to a representative tensor and register
        through GlobalTE instead of ``self.store.register_buffer``.
        """
        if getattr(self, "_capacity_only", False):
            return
        if not kv_caches:
            _log().warning("No KV caches to offload.")
            return

        # Lazy torch import (containers without torch_npu shouldn't even
        # reach this path, but be defensive).
        import torch

        def _repr_tensor(v):
            if isinstance(v, torch.Tensor):
                return v
            return v[0]

        if self.cache_config.num_gpu_blocks is None:
            raise RuntimeError("num_gpu_blocks must be configured")
        self.num_blocks = self.cache_config.num_gpu_blocks

        # Group by storage and merge overlapping regions. Pangu's hybrid KV
        # cache exposes several tensors (DSA latent, indexer, Mome states,
        # ...) that alias the same raw allocation at different offsets. The
        # Ascend TransferEngine rejects overlapping register_memory, so merge
        # each storage's regions into a single [min_start, max_end) span
        # (same approach as vllm-ascend's registered_regions + _align_kv_ptrs).
        registered_regions: dict[int, list[int]] = {}
        for value in kv_caches.values():
            cache = _repr_tensor(value)
            el = cache.element_size()
            base_addr = cache.data_ptr()
            if cache.ndim == 2:
                region_len = (
                    (cache.shape[0] - 1) * cache.stride(0) * el
                    + cache.shape[1] * el
                )
            else:
                region_len = cache.numel() * el
            storage_key = cache.untyped_storage().data_ptr()
            start, end = base_addr, base_addr + region_len
            if storage_key in registered_regions:
                old_start, old_end = registered_regions[storage_key]
                registered_regions[storage_key] = [
                    min(old_start, start),
                    max(old_end, end),
                ]
            else:
                registered_regions[storage_key] = [start, end]

        ptrs = [r[0] for r in registered_regions.values()]
        lengths = [r[1] - r[0] for r in registered_regions.values()]

        # GlobalTE has an idempotent guard so this is safe to call once per
        # process. If a sibling connector already registered the buffer,
        # this returns silently.
        global_te.register_buffer(ptrs, lengths)

        # Replicate master register_kv_caches' segment-layout + transfer-thread
        # wiring, but skip ``self.store.register_buffer``. master registers each
        # raw storage's whole ``storage.nbytes()`` span, which overlaps across
        # Pangu's hybrid KV cache views and is rejected by the Ascend
        # TransferEngine ("overlapped memory region"). GlobalTE already
        # registered the merged, non-overlapping regions above through the same
        # engine the store was injected with, so the redundant per-storage
        # registration is omitted here.
        seen_ptrs: set[int] = set()
        addrs: list[int] = []
        block_lens: list[int] = []
        for value in kv_caches.values():
            cache = _repr_tensor(value)
            cache_storage = cache.untyped_storage()
            base_addr = cache_storage.data_ptr()
            if base_addr in seen_ptrs:
                continue
            seen_ptrs.add(base_addr)
            region_len = cache_storage.nbytes()
            el = cache.element_size()
            page_size_bytes = region_len // self.num_blocks
            outer_dims = [
                d
                for d in range(cache.ndim)
                if cache.stride(d) * el > page_size_bytes
            ]
            if not outer_dims:
                addrs.append(base_addr)
                block_lens.append(page_size_bytes)
            else:
                seg_stride = cache.stride(outer_dims[0]) * el
                for idx in range(cache.shape[outer_dims[0]]):
                    addrs.append(base_addr + idx * seg_stride)
                    block_lens.append(seg_stride // self.num_blocks)

        _log().info(
            "Registered KV caches: num_groups=%d, num_segments=%d, "
            "num_blocks=%d",
            len(self.token_dbs),
            len(addrs),
            self.num_blocks,
        )

        for db in self.token_dbs:
            db.set_kv_caches_base_addr(addrs)
            db.set_block_len(block_lens)

        if self.kv_role in ["kv_producer", "kv_both"]:
            ready_event_sending = threading.Event()
            self.kv_send_thread = PoolStoreSendingThread(
                self.store,
                self.coord,
                self.token_dbs,
                self.block_size,
                self.tp_rank,
                self.put_step,
                self.kv_role,
                ready_event_sending,
                self.enable_kv_events and self._store_events_requested,
                self.store_replicate_config,
                record_operation=self._record_kv_connector_operation,
                jobs=self.store_jobs,
            )
            self.kv_send_thread.start()
            _log().info(
                "Started Mooncake KV-store send thread, store_events=%s",
                self.enable_kv_events and self._store_events_requested,
            )

        self.kv_recv_threads = []
        ready_events_recving = []
        for i in range(self.num_recv_threads):
            ready_event_recving = threading.Event()
            recv_thread = PoolStoreReceivingThread(
                self.store,
                self.coord,
                self.token_dbs,
                self.block_size,
                self.tp_rank,
                ready_event_recving,
                disk_offload_buffer_budget_bytes=self.disk_offload_buffer_budget_bytes,
                enable_kv_event=(self.enable_kv_events and self._load_events_requested),
                locality_ip=(
                    _get_local_ip()
                    if os.environ.get("OMNI_FUSION_POOL_LOCALITY_LOG", "0") == "1"
                    else None
                ),
                record_operation=self._record_kv_connector_operation,
                request_queue=self.recv_request_queue,
            )
            recv_thread.name = f"KVCacheStoreRecvingThread-{i}"
            recv_thread.start()
            self.kv_recv_threads.append(recv_thread)
            ready_events_recving.append(ready_event_recving)
        for ready_event_recving in ready_events_recving:
            ready_event_recving.wait()
        _log().info(
            "Started %d Mooncake KV-load receive thread(s), load_events=%s",
            self.num_recv_threads,
            self.enable_kv_events and self._load_events_requested,
        )

    # ------------------------------------------------------------------
    # Lookup diagnostics override.
    #
    # This is the worker end of the external prefix-cache lookup chain:
    # scheduler -> LookupKeyClient (ZMQ) -> LookupKeyServer -> store_worker.
    #   lookup(). It is only reached when the scheduler's ZMQ request
    # actually arrives, so logging here (vs. the scheduler-side override in
    # NPUMooncakeStoreConnector) lets us tell "the ZMQ hop never arrived"
    # from "it arrived but batch_is_exist found nothing".
    # ------------------------------------------------------------------
    def lookup(self, token_len: int, block_hashes, *, hbm_hit_tokens=0) -> int:
        if getattr(self, "_capacity_only", False):
            return 0
        log = _log()
        num_hashes = len(block_hashes) if block_hashes else 0
        log.info(
            "[NPUMooncake][worker] lookup token_len=%d num_hashes=%d "
            "block_size=%d hash_block_size=%d",
            token_len,
            num_hashes,
            self.block_size,
            self.hash_block_size,
        )
        expected = getattr(self, "_lookup_expected_per_key", None)
        if (expected is not None
                and all(len(prefixes) == expected for prefixes in self._lookup_key_prefixes)):
            try:
                result = self._lookup_with_coordinator(
                    token_len, block_hashes, list(range(len(self.token_dbs))),
                    hbm_hit_tokens=hbm_hit_tokens,
                )
                if result is None:
                    result = super().lookup(token_len, block_hashes)
            except Exception as exc:
                # Match the installed worker's RPC-error -> miss contract.
                log.error("Remote connection failed in lookup: %s", exc)
                result = 0
        else:
            result = super().lookup(token_len, block_hashes)
        log.info("[NPUMooncake][worker] lookup -> %d", result)
        return result

    def start_load_kv(self, forward_context, **kwargs):
        if getattr(self, "_capacity_only", False):
            return set(), set()
        return super().start_load_kv(forward_context, **kwargs)

    # ------------------------------------------------------------------
    # Pangu spec -> replication-factor mapping override.
    #
    # Why this override exists: master ``MooncakeStoreWorker`` falls through
    # to ``tp_size // num_kv_head`` for anything it does not recognise, which
    # silently drops Pangu's ``DSAAttentionSpec`` (sparse MLA, num_kv_heads=1)
    # and ``ShareKVSlidingWindowSpec`` (shared-KV sliding window, num_kv_heads=1)
    # into a GQA-shaped namespace partition. Both are MLA-shaped in practice
    # and must be replicated on every TP rank (``tp_size``) to keep the
    # prefix-cache namespace consistent across ranks.
    #
    # ``MomeSpec`` is a ``MambaSpec`` subclass and is therefore already covered
    # by the MambaSpec branch (returns 1) -- no extra branch needed.
    # ------------------------------------------------------------------
    def _spec_tp_replication_factor(self, spec):
        if self.dcp_size > 1:
            # DCP shards KV across ranks -> each rank owns its own namespace.
            return 1

        inner_specs = (
            tuple(spec.kv_cache_specs.values())
            if isinstance(spec, UniformTypeKVCacheSpecs)
            else (spec,)
        )

        # Any rank-specific state makes the whole packed value rank-specific.
        if any(isinstance(inner, MambaSpec) for inner in inner_specs):
            return 1

        # Pangu override: DSA and shared-KV sliding window are MLA-shaped
        # (single KV head, absorbed layout) and must be replicated on every
        # TP rank. The Pangu spec classes live in an omni patch module that
        # may not be importable on every build, so guard the import.
        pangu_replicated_types = []
        try:
            from omni_npu.vllm_patches.usefull_patch.patch_kv_cache_interface import (  # type: ignore
                DSAAttentionSpec,
                ShareKVSlidingWindowSpec,
            )

            pangu_replicated_types.extend(
                [DSAAttentionSpec, ShareKVSlidingWindowSpec]
            )
        except Exception:  # pragma: no cover -- Pangu patch not installed.
            pass

        if pangu_replicated_types and all(
            isinstance(inner, tuple(pangu_replicated_types))
            for inner in inner_specs
        ):
            return self.tp_size

        # Master MLA-shaped packed values are replicated on every TP rank.
        if all(
            isinstance(inner, (MLAAttentionSpec, SlidingWindowMLASpec))
            for inner in inner_specs
        ):
            return self.tp_size

        return max(1, self.tp_size // self.num_kv_head)


def _has_torch_npu() -> bool:
    try:
        import torch

        return hasattr(torch, "npu") and torch.npu.is_available()
    except Exception:
        return False


def torch_npu_current_device() -> int:
    import torch

    return torch.npu.current_device()
