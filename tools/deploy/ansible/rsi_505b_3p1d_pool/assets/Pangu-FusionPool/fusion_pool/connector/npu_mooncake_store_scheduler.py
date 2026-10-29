# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Keep store ownership alive independently of vLLM request ownership."""

from dataclasses import replace
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import LoadSpec
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.scheduler import MooncakeStoreScheduler
from fusion_pool.connector.store_lifetime import StoreLedger
from fusion_pool.connector.store_metadata import StoreMetadata, StoreWorkerMetadata
from fusion_pool.connector.capacity_only import suppress_lookup_channel


def block_ids(groups):
    # Hybrid groups share the allocator's numeric ID namespace. Treat equal
    # IDs across groups as aliases: conservative for disjoint layer storage.
    ids = set()
    for group in groups:
        ids.update(block_id for block_id in group if block_id != 0)
    return ids


class NPUMooncakeStoreScheduler(MooncakeStoreScheduler):
    def __init__(self, config, cache_config):
        # Capacity-only: master __init__ builds LookupKeyClient; the PFC
        # connector's window has already restored the real class by now,
        # so suppress again for THIS construction.
        from fusion_pool.connector.store_lookup_channel import hbm_lookup_channel

        with hbm_lookup_channel(config, "scheduler"), suppress_lookup_channel(config):
            super().__init__(config, cache_config)
        # Upstream PR #50498: instances that never load from the store skip
        # the lookup CPU cost entirely. Capacity-only D sides set this false.
        self.enable_lookup = (
            config.kv_transfer_config.kv_connector_extra_config.get(
                "enable_lookup", True
            )
        )
        self._reject_async_lookup()
        parallel = config.parallel_config
        unsupported_parallelism = (
            parallel.pipeline_parallel_size != 1
            or parallel.prefill_context_parallel_size != 1
            or parallel.decode_context_parallel_size != 1
        )
        if unsupported_parallelism:
            raise ValueError("Mooncake store lifetime protocol currently requires PP=PCP=DCP=1")
        self.store_ledger = StoreLedger(parallel.tensor_parallel_size)
        self._allocated = set()
        self._contexts = {}
        self._context_sequence = 0
        self._finished_contexts = set()
        self._lookup_block_pool = None
        self._lookup_cache_config = cache_config
        self._local_lookup_coord = None

    def _reject_async_lookup(self):
        # The HBM proof is read once and the lookup call waits for the result.
        # An in-flight async lookup can outlive that proof.
        if self.enable_lookup and self.lookup_async:
            raise ValueError("HBM prefix lookup requires lookup_async=false")

    def bind_gpu_block_pool(self, block_pool):
        if not self.enable_lookup:
            return
        from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.coordinator import (
            MooncakeStoreCoordinator,
        )
        self._lookup_block_pool = block_pool
        self._local_lookup_coord = MooncakeStoreCoordinator(
            self._lookup_cache_config.kv_cache_groups,
            self._block_size, self._hash_block_size)

    def _common_hbm_tokens(self, request, num_computed_tokens):
        coord = getattr(self, "_local_lookup_coord", None)
        pool = getattr(self, "_lookup_block_pool", None)
        if not num_computed_tokens or coord is None or pool is None:
            return 0
        if num_computed_tokens % coord.lcm_block_size:
            return 0
        # With one standard dense-attention spec, the engine's reported hit
        # already is the common hit of all groups sharing that spec. The
        # max(per_group_hits) ambiguity exists only with differing specs.
        from vllm.v1.core.single_type_kv_cache_manager import FullAttentionManager
        from fusion_pool.connector.store_lookup import _hit_function
        if (len(coord.attention_groups) == 1
                and _hit_function(coord.attention_groups[0][2])
                is _hit_function(FullAttentionManager)):
            return num_computed_tokens
        # v0.25.1 may pass max(per_group_hits), not the hybrid common hit.
        # Verify the boundary without changing references or applying EAGLE
        # pruning a second time. A missing state must still be checked remotely.
        try:
            from fusion_pool.connector.store_lookup import retained_chunk_range
            # The usual hit can be proved without constructing full block
            # tables. SWA needs only its window, Mamba/MoME only the boundary.
            for spec, group_ids, manager in coord.attention_groups:
                retained = retained_chunk_range(
                    spec, manager, num_computed_tokens // spec.block_size)
                if retained is None:
                    break
                hashes = coord.block_hashes_for_spec(request.block_hashes, spec)
                if any(pool.get_cached_block(hashes[i], group_ids) is None
                       for i in retained):
                    break
            else:
                return num_computed_tokens
            # Divergent hybrid hits still use native convergence to recover a
            # smaller proven boundary rather than pretending every group hit.
            return coord.find_longest_cache_hit(
                request.block_hashes, num_computed_tokens, pool,
                apply_eagle=False)[1]
        except (AttributeError, NotImplementedError, AssertionError, ValueError, IndexError):
            return 0

    def _query_external_hit(self, request, num_computed_tokens, hbm_hit_tokens):
        # No hit cache: each admission retry uses the current HBM proof.
        self.load_specs.pop(request.request_id, None)
        token_len = request.num_tokens // self._block_size * self._block_size
        if token_len < self._block_size:
            return 0, False
        hit = self.client.lookup(
            request.request_id, token_len, request.block_hashes,
            non_block=self.lookup_async, hbm_hit_tokens=hbm_hit_tokens)
        if hit is None:
            return None, False
        if hit == request.num_tokens:
            hit = max(0, (request.num_tokens - 1) // self._block_size * self._block_size)
        need = max(0, hit - num_computed_tokens)
        if not need:
            return 0, False
        self.load_specs[request.request_id] = LoadSpec(
            vllm_cached_tokens=num_computed_tokens,
            kvpool_cached_tokens=hit, can_load=False)
        return need, self.load_async

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        self._allocated.update(block_ids(blocks.get_block_ids()))
        return super().update_state_after_alloc(request, blocks, num_external_tokens)

    def _finish_context(self, req_id):
        context = self._contexts.pop(req_id, None)
        if context is not None:
            self._finished_contexts.add(context)

    def request_finished(self, request, blocks):
        self._finish_context(request.request_id)
        # Store jobs already own immutable block snapshots. LLMDataDist still
        # decides its own request retention in the enclosing MultiConnector.
        return False, None

    def build_connector_meta(self, output):
        writes = self._allocated
        self._allocated = set()
        preempted = output.preempted_req_ids or ()
        flush = set()
        for req_id in preempted:
            flush.update(self.store_ledger.for_request(req_id))
            self._finish_context(req_id)
        for req_id in output.finished_req_ids:
            self._finish_context(req_id)
        # Include all currently scheduled blocks, not only fresh allocation:
        # hybrid state and partial trailing pages can be overwritten in place.
        for req_id in output.num_scheduled_tokens:
            tracker = self._request_trackers.get(req_id)
            if tracker:
                writes.update(block_ids(tracker.allocated_block_ids))
        for req in output.scheduled_new_reqs:
            writes.update(block_ids(req.block_ids))
        for groups in output.scheduled_cached_reqs.new_block_ids:
            if groups:
                writes.update(block_ids(groups))
        original = super().build_connector_meta(output)
        for req in original.requests:
            # Includes asynchronous loads on otherwise empty model batches.
            writes.update(block_ids(req.block_ids))
        flush.update(self.store_ledger.conflicts(writes))
        # Bound queued work without deleting any ownership record. The next
        # worker step drains existing work before accepting another batch.
        if len(self.store_ledger.jobs) >= 256:
            flush.update(self.store_ledger.jobs)
        jobs = {}
        for req in original.requests:
            if not req.can_save:
                continue
            context = self._contexts.get(req.req_id)
            if context is None:
                self._context_sequence += 1
                context = f"{self.store_ledger.epoch}/request/{self._context_sequence}"
                self._contexts[req.req_id] = context
            snapshot = replace(req, block_ids=tuple(list(g) for g in req.block_ids),
                               block_hashes=list(req.block_hashes),
                               token_ids=list(req.token_ids) if req.token_ids is not None else None)
            job_id = self.store_ledger.add(req.req_id, block_ids(snapshot.block_ids))
            jobs[job_id] = (snapshot, context)
        retired, self.store_ledger.retired = self.store_ledger.retired, set()
        contexts, self._finished_contexts = self._finished_contexts, set()
        return StoreMetadata(original, jobs, flush, retired, contexts)

    def update_store_output(self, output):
        meta = output.kv_connector_worker_meta
        if meta is None:
            return
        if not isinstance(meta, StoreWorkerMetadata):
            raise TypeError("Expected Mooncake store worker outcomes")
        for job_id, outcomes in meta.outcomes.items():
            for rank, error in outcomes.items():
                self.store_ledger.report(job_id, rank, error)

    def has_pending_push_work(self):
        return bool(self.store_ledger.jobs or self.store_ledger.retired or self._finished_contexts)

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        if not self.enable_lookup:
            # Capacity-only: report no hit without consulting the store, so
            # the enclosing MultiConnector never routes this request's
            # admission through a pool load.
            return 0, False
        hbm = self._common_hbm_tokens(request, num_computed_tokens)
        token_len = request.num_tokens // self._block_size * self._block_size
        if token_len and hbm >= token_len:
            self.client.discard(request.request_id)
            self.load_specs.pop(request.request_id, None)
            return 0, False
        return self._query_external_hit(request, num_computed_tokens, hbm)
