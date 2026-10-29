# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Mooncake send/receive threads, source lifetime and worker integration."""
from dataclasses import replace
import time
import torch
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker import (
    KVCacheStoreRecvingThread,
    KVCacheStoreSendingThread,
)
from vllm.logger import init_logger
from fusion_pool.connector.store_events import _EventDatabase, _LoadEventStore
from fusion_pool.connector.store_lifetime import LocalStoreJobs
from fusion_pool.connector.store_metadata import StoreMetadata, StoreWorkerMetadata
logger = init_logger(__name__)

_MOONCAKE_TRANSFER_FAIL = -800
_MOONCAKE_OBJECT_NOT_FOUND = -704
_MOONCAKE_RPC_FAIL = -900
# Mooncake 0.3.12 reports an unresolved transfer as -800 after 60 seconds.
# Keep this version-specific fast-failure threshold strictly below that timeout.
_FAST_TRANSFER_FAIL_SECONDS = 30.0


class _SourceSafeStore:
    """Observe PUT results without changing the native sender's cache policy.

    NO_AVAILABLE_HANDLE rejects allocation before any source read. Successful
    entries have completed synchronous transfers. Mooncake 0.3.12 also waits
    for a submitted transfer to reach a terminal state before quickly returning
    TRANSFER_FAIL. Its separate unresolved-transfer timeout is 60 seconds, so a
    narrowly bounded fast -800 response is safe to release but not persistent.
    In the current single-memory-replica PUT path, OBJECT_NOT_FOUND is a
    metadata failure before submission or after successful transfer completion.
    It does not imply persistence, but it permits releasing the source.
    For the audited Mooncake 0.3.12 embedded, memory-only synchronous PUT,
    RPC_FAIL either rejects PutStart before submission or fails finalization
    after successful transfers. A transfer failure takes precedence over a
    finalization RPC error. This permits source release, not persistence or
    any conclusion about target-address reuse across Master restarts.
    Other failures retain the conservative source fence.
    """

    def __init__(self, store):
        self._store = store
        self.put_source_safe = False

    def __getattr__(self, name):
        return getattr(self._store, name)

    def batch_put_from_multi_buffers(self, keys, *args, **kwargs):
        self.put_source_safe = False
        start_time = time.monotonic()
        results = self._store.batch_put_from_multi_buffers(keys, *args, **kwargs)
        elapsed = time.monotonic() - start_time
        # Validate the full response: neither truncation nor an unknown result
        # is evidence that every source read is finished.
        if len(results) != len(keys) or any(result > 0 for result in results):
            raise RuntimeError('Invalid Mooncake PUT completion response')
        known_results = all(
            result in (0, -200, _MOONCAKE_OBJECT_NOT_FOUND,
                       _MOONCAKE_RPC_FAIL, _MOONCAKE_TRANSFER_FAIL)
            for result in results
        )
        fast_transfer_failure = (
            elapsed < _FAST_TRANSFER_FAIL_SECONDS
            and _MOONCAKE_TRANSFER_FAIL in results
            and known_results
        )
        self.put_source_safe = (
            all(result in (0, -200, _MOONCAKE_OBJECT_NOT_FOUND,
                           _MOONCAKE_RPC_FAIL) for result in results)
            or fast_transfer_failure
        )
        if fast_transfer_failure:
            logger.warning(
                'Mooncake PUT fast transfer failure; treating the source as '
                'released without publishing failed keys: elapsed_ms=%.3f '
                'failed_keys=%d total_keys=%d',
                elapsed * 1000,
                results.count(_MOONCAKE_TRANSFER_FAIL),
                len(results),
            )
        if self.put_source_safe and _MOONCAKE_OBJECT_NOT_FOUND in results:
            logger.warning(
                'Mooncake PUT object metadata disappeared; treating the source '
                'as released without publishing failed keys: elapsed_ms=%.3f '
                'object_not_found_keys=%d total_keys=%d',
                elapsed * 1000,
                results.count(_MOONCAKE_OBJECT_NOT_FOUND),
                len(results),
            )
        if self.put_source_safe and _MOONCAKE_RPC_FAIL in results:
            logger.warning(
                'Mooncake synchronous PUT RPC failed; treating the source as '
                'released without confirming cache persistence: elapsed_ms=%.3f '
                'rpc_failed_keys=%d total_keys=%d',
                elapsed * 1000,
                results.count(_MOONCAKE_RPC_FAIL),
                len(results),
            )
        return results


class _RetainedSourceDatabase:
    """Restrict a store plan to populated blocks in its immutable snapshot.

    Hybrid cache hits use block 0 for history they do not retain. Reachability
    describes which keys would be useful, not which source pages still exist.
    Dedup cannot validate a source: an evicted key would otherwise be rebuilt
    from the null page. Never change the receiver's separate load plan.
    """

    def __init__(self, database, block_ids):
        self._database = database
        self._block_ids = block_ids
        self.skipped_null_blocks = 0

    def __getattr__(self, name):
        return getattr(self._database, name)

    def process_tokens(self, *args, **kwargs):
        for start, end, block_hash in self._database.process_tokens(*args, **kwargs):
            index = start // self._database.block_size
            if index >= len(self._block_ids):
                raise ValueError('Store plan exceeds source block snapshot')
            if self._block_ids[index] == 0:
                self.skipped_null_blocks += 1
                continue
            yield start, end, block_hash


class PoolStoreSendingThread(KVCacheStoreSendingThread):
    def __init__(self, *args, jobs, **kwargs):
        super().__init__(*args, **kwargs)
        self.jobs = jobs
        self.store = _SourceSafeStore(self.store)
        self._job_error = None
        self._store_error = None

    def submit(self, job_id, request):
        self.jobs.register(job_id)
        self.add_stored_request(request.req_id)
        self.add_request((job_id, request))

    def _record_operation(self, operation, start_time, num_keys, **kwargs):
        if kwargs.get('status', 'ok') != 'ok':
            self._store_error = f"{operation}: {kwargs['status']}"
            source_safe_failure = (
                operation == 'save_put'
                and kwargs['status'] == 'partial_failure'
                and self.store.put_source_safe
            )
            if not source_safe_failure:
                self._job_error = self._store_error
        return super()._record_operation(operation, start_time, num_keys, **kwargs)

    def update_kv_event(self, events):
        if self._store_error is None:
            super().update_kv_event(events)

    def _handle_request(self, payload):
        job_id, request = payload
        self._job_error = None
        self._store_error = None
        self.store.put_source_safe = False
        databases = self.token_databases
        try:
            if len(databases) != len(request.block_ids):
                raise ValueError('Store source group count mismatch')
            self.token_databases = [
                _RetainedSourceDatabase(db, blocks)
                for db, blocks in zip(databases, request.block_ids, strict=True)
            ]
            # A source-safe acknowledgement is separate from cache persistence:
            # capacity rejection starts no read; successful PUTs finish theirs.
            # Native pressure skipping and dedup paths start no source reads.
            super()._handle_request(request)
        except Exception as exc:
            self._job_error = type(exc).__name__
            logger.exception('Mooncake store job failed: %s', job_id)
        finally:
            skipped = sum(
                getattr(db, 'skipped_null_blocks', 0)
                for db in self.token_databases
            )
            self.token_databases = databases
            if skipped:
                logger.info('Mooncake omitted null source blocks: job=%s rank=%d blocks=%d',
                            job_id, self.tp_rank, skipped)
            self.jobs.complete(job_id, self._job_error)


class PoolStoreReceivingThread(KVCacheStoreRecvingThread):
    def __init__(self, *args, enable_kv_event=False, locality_ip=None, **kwargs):
        super().__init__(*args, **kwargs)
        self._emit_load_events = enable_kv_event
        self._locality_ip = locality_ip

    def _handle_request(self, request):
        """Never leave a request waiting on a load that will not arrive.

        A request in WAITING_FOR_REMOTE_KVS only leaves that state when the
        receive thread reports it finished. The GET loop already handles its
        own failures -- it marks the blocks as load errors so vLLM recomputes
        them, then reports finished. Everything before that loop does not:
        load_spec access, the load mask, key and address preparation, batch
        splitting. An exception there propagates to run(), which logs it and
        moves on, and the request waits forever. The prefill then holds its
        slots until proxy-side load and running counts push it past
        max_batch_num_token or prefill_max_num_seqs and it stops being picked.
        This is one possible cause of stranded requests, not a diagnosis of
        any particular benchmark run.

        Marking the blocks is what makes reporting finished safe: without it
        the engine would treat unfilled blocks as valid KV and read whatever
        is in them.
        """
        result = None
        try:
            result = self._handle_request_inner(request)
        except Exception:
            req_id = getattr(request, "req_id", None)
            block_ids = [
                block_id
                for group in getattr(request, "block_ids", ()) or ()
                for block_id in group
            ]
            if block_ids:
                self._add_load_error_block_ids(block_ids)
            if req_id is not None:
                self.set_finished_request(req_id)
            # Upstream acknowledges successful requests itself. An exception
            # escaping its handler leaves this queue item unacknowledged.
            self.request_queue.task_done()
            logger.exception(
                "Mooncake load failed for request %s; %d blocks marked for "
                "recompute and the request released",
                req_id,
                len(block_ids),
            )
        return result

    def _handle_request_inner(self, request):
        locality_ip = getattr(self, "_locality_ip", None)
        if not self._emit_load_events and not locality_ip:
            return super()._handle_request(request)
        store, databases = self.store, self.token_databases
        event_info = {}
        if locality_ip:
            from fusion_pool.connector.load_locality import LocalityStore

            self.store = LocalityStore(
                store, locality_ip, self.tp_rank, request.req_id, logger.info
            )
        if self._emit_load_events:
            self.store = _LoadEventStore(self.store, event_info, self.update_kv_event)
            self.token_databases = [
                _EventDatabase(db, group_idx, request, event_info)
                for group_idx, db in enumerate(databases)
            ]
        try:
            return super()._handle_request(request)
        finally:
            self.store, self.token_databases = store, databases



class StoreLifetimeWorkerMixin:
    def __init__(self, *args, **kwargs):
        self.store_jobs = LocalStoreJobs()
        self._finished_save_contexts = set()
        self._lifetime_closed = False
        super().__init__(*args, **kwargs)

    def handle_store_fences(self, metadata):
        if not isinstance(metadata, StoreMetadata):
            raise TypeError('Mooncake store lifetime metadata is required')
        start = time.monotonic()
        self.store_jobs.wait(metadata.jobs_to_flush)
        if metadata.jobs_to_flush:
            logger.info('Mooncake source fence rank=%d jobs=%d wait_ms=%.3f',
                        self.tp_rank, len(metadata.jobs_to_flush),
                        (time.monotonic() - start) * 1000)
        # Ordered metadata places old batches' fences before these retirements.
        self.store_jobs.acknowledge(metadata.retired_jobs)
        self._finished_save_contexts.update(metadata.finished_contexts)

    def get_finished(self, finished_req_ids, meta):
        if not isinstance(meta, StoreMetadata):
            raise TypeError('Mooncake store lifetime metadata is required')
        for request in meta.requests:
            spec = request.load_spec
            if spec is not None and spec.can_load:
                spec.token_len = spec.kvpool_cached_tokens
                self.recv_request_queue.put(request)
        sender = self.kv_send_thread
        if sender is not None:
            for context in tuple(self._finished_save_contexts):
                with sender.done_task_lock:
                    done = sender.stored_requests.get(context, 0) == 0
                if done:
                    sender.delete_finished_stored_request(context)
                    self._finished_save_contexts.remove(context)
        recving = set()
        for thread in self.kv_recv_threads:
            recving.update(thread.get_and_clear_finished_requests())
        return set(), recving

    def wait_for_save(self, meta):
        """Queue stores only after target AND speculative draft KV are ready.

        vLLM calls get_finished after the target forward even when connector
        finalization is deferred for MTP. Recording the producer event there
        allows the sender to read draft-layer storage before drafting writes it.
        This hook is invoked after drafting, or immediately after the target
        forward when speculation is disabled. Source lifetime remains async.
        """
        if not isinstance(meta, StoreMetadata):
            raise TypeError('Mooncake store lifetime metadata is required')
        if meta.store_jobs:
            event = torch.npu.Event()
            event.record()
            for job_id, (request, context) in meta.store_jobs.items():
                saved_request = replace(request, req_id=context, current_event=event)
                self.kv_send_thread.submit(job_id, saved_request)

    def build_store_worker_meta(self):
        reports = self.store_jobs.take_reports()
        if not reports:
            return None
        return StoreWorkerMetadata({job: {self.tp_rank: error} for job, error in reports.items()})

    def close(self):
        if self._lifetime_closed:
            return
        # On uncertain transport failure do not unregister a still-read source.
        self.store_jobs.drain()
        self._lifetime_closed = True
        super().close()


class LoadEventWorkerMixin:
    def get_kv_events(self):
        if not self.enable_kv_events:
            return []
        events = []
        if self.kv_send_thread is not None:
            events.extend(self.kv_send_thread.get_kv_events())
        for recv_thread in self.kv_recv_threads:
            events.extend(recv_thread.get_kv_events())
        return events
