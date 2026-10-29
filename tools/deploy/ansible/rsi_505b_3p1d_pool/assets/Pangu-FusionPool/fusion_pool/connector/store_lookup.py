# SPDX-License-Identifier: Apache-2.0
"""Group lookup and HBM-prefix pruning for the PFC Mooncake worker.

Group tasks return detached existence sets. The caller merges them and asks
the native coordinator for the hit length, so EAGLE handling stays unchanged.
"""
import os
from concurrent.futures import ThreadPoolExecutor

from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.coordinator import (
    ExternalCachedBlockPool,
)
from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.data import PoolKey

from fusion_pool.connector.store_lookup_adapter import AscendLookupInterfaces


def _hit_function(manager):
    """Return the unbound hit method for either a manager class or instance."""
    method = manager.find_longest_cache_hit
    return getattr(method, "__func__", method)


def retained_chunk_range(spec, manager, num_chunks):
    """Closed form of the standard managers' all-present load masks."""
    from vllm.v1.core.single_type_kv_cache_manager import (
        FullAttentionManager, MambaManager, SlidingWindowManager,
    )
    # Known managers retain the full prefix, boundary state, or sliding window.
    # Unknown implementations fall back to the native load_mask.
    hit_method = _hit_function(manager)
    if hit_method is _hit_function(FullAttentionManager):
        return range(num_chunks)
    if hit_method is _hit_function(MambaManager):
        return range(max(0, num_chunks - 1), num_chunks)
    if hit_method is _hit_function(SlidingWindowManager):
        num_window_chunks = max(1, manager._contiguous_blocks_for_hit(
            spec.sliding_window, spec.block_size, False))
        return range(max(0, num_chunks - num_window_chunks), num_chunks)
    return None


def _local_masks(worker, token_len, block_hashes, hbm_hit_tokens):
    lcm_block_size = getattr(worker.coord, "lcm_block_size", 0)
    # Only prune at a boundary aligned to every group's block size.
    if not isinstance(hbm_hit_tokens, int) or hbm_hit_tokens <= 0 or hbm_hit_tokens > token_len:
        return ()
    if not lcm_block_size or hbm_hit_tokens % lcm_block_size:
        return ()
    try:
        # The scheduler verified the real block pool. Rebuild only retained
        # blocks; historical SWA/Mamba blocks are not necessarily still local.
        local_masks = [None] * len(worker.token_dbs)
        for spec, group_ids, manager in worker.coord.attention_groups:
            num_prefix_chunks = hbm_hit_tokens // spec.block_size
            retained_chunks = retained_chunk_range(spec, manager, num_prefix_chunks)
            if retained_chunks is None:
                return worker.coord.load_mask(block_hashes, hbm_hit_tokens)
            group_local_mask = [False] * retained_chunks.start + [True] * len(retained_chunks)
            for group_id in group_ids:
                local_masks[group_id] = group_local_mask
        return tuple(local_masks)
    except (AttributeError, NotImplementedError, AssertionError, ValueError):
        return ()


def _local_block_keys(group_id, group_block_hashes, lookup_mask, local_mask):
    """Return only HBM blocks covered by the coordinator's lookup mask."""
    # Proven local blocks count as hits even if the remote Store evicted them.
    existing_block_keys = set()
    if not local_mask:
        return existing_block_keys
    for chunk_id, is_in_hbm in enumerate(local_mask):
        if not is_in_hbm or chunk_id >= len(group_block_hashes):
            continue
        if lookup_mask is not None and (
            chunk_id >= len(lookup_mask) or not lookup_mask[chunk_id]
        ):
            continue
        existing_block_keys.add((group_id, bytes(group_block_hashes[chunk_id])))
    return existing_block_keys


class AscendLookupWorkerMixin(AscendLookupInterfaces):
    def _init_group_lookup(self):
        self.lookup_group_parallel = os.getenv(
            "OMNI_LOOKUP_GROUP_PARALLEL", "0") == "1"
        self.lookup_group_parallel_workers = max(
            1, int(os.getenv("OMNI_LOOKUP_GROUP_PARALLEL_WORKERS", "6")))

    def _lookup_keys(self, group_id, chunk_hash):
        # Reuse PUT/GET namespaces to preserve tenant, rank, and group keys.
        hash_hex = chunk_hash.hex()
        return [
            PoolKey.build_key_string(prefix, hash_hex)
            for prefix in self._lookup_key_prefixes[group_id]
        ]

    def _build_group_lookup_keys(
        self, group_id, group_block_hashes, token_len, hbm_hit_tokens,
        lookup_mask, local_mask,
    ):
        """Build rank keys for reachable chunks not already covered by HBM."""
        block_size = self.token_dbs[group_id].block_size
        num_hbm_chunks = hbm_hit_tokens // block_size
        # A fully local prefix can start at the suffix. SWA/Mamba still check
        # local_mask per chunk; older blocks remain remote candidates.
        prefix_is_fully_local = bool(local_mask) and all(local_mask)
        start_chunk = (
            min(num_hbm_chunks, len(local_mask))
            if prefix_is_fully_local else 0)
        stop_chunk = min(
            len(group_block_hashes), (token_len + block_size - 1) // block_size)
        lookup_keys = []
        remote_chunk_hashes = []
        for chunk_id in range(start_chunk, stop_chunk):
            # lookup_mask selects candidates; local_mask skips proven HBM hits.
            if lookup_mask is not None and (
                chunk_id >= len(lookup_mask) or not lookup_mask[chunk_id]
            ):
                continue
            if chunk_id < len(local_mask) and local_mask[chunk_id]:
                continue
            chunk_hash = group_block_hashes[chunk_id]
            # Keep rank keys contiguous per chunk for slicing the RPC results.
            lookup_keys.extend(self._lookup_keys(group_id, chunk_hash))
            remote_chunk_hashes.append(chunk_hash)
        return lookup_keys, remote_chunk_hashes

    def _lookup_coordinator_group(
        self, group_id, token_len, block_hashes, hbm_hit_tokens,
        lookup_mask, local_mask,
    ):
        """Combine proven HBM hits and one group's remote existence results."""
        group_block_hashes = self.coord.block_hashes_for_spec(
            block_hashes, self._kv_cache_groups[group_id].kv_cache_spec
        )
        existing_block_keys = _local_block_keys(
            group_id, group_block_hashes, lookup_mask, local_mask,
        )
        lookup_keys, remote_chunk_hashes = self._build_group_lookup_keys(
            group_id, group_block_hashes, token_len, hbm_hit_tokens,
            lookup_mask, local_mask,
        )
        if not lookup_keys:
            return existing_block_keys

        # At most one batch RPC per group; all registered ranks must exist
        # for a chunk to count as a remote hit.
        exist_flags = self.m_store.exists(lookup_keys)
        num_rank_keys = len(self._lookup_key_prefixes[group_id])
        for index, chunk_hash in enumerate(remote_chunk_hashes):
            rank_exist_flags = exist_flags[
                index * num_rank_keys:(index + 1) * num_rank_keys]
            if rank_exist_flags and all(flag == 1 for flag in rank_exist_flags):
                existing_block_keys.add((group_id, bytes(chunk_hash)))
        return existing_block_keys

    def _lookup_with_coordinator(
        self, token_len, block_hashes, kv_cache_group_ids, hbm_hit_tokens=0,
    ):
        """Query independent groups, then calculate their common hit length."""
        if sorted(kv_cache_group_ids) != list(range(len(self.token_dbs))):
            return None
        if not block_hashes or token_len <= 0:
            return 0

        lookup_masks = self.coord.lookup_mask(token_len)
        local_masks = _local_masks(self, token_len, block_hashes, hbm_hit_tokens)
        if not local_masks:
            # Unverified HBM coverage falls back to querying all candidates.
            hbm_hit_tokens = 0
            local_masks = ((),) * len(self.token_dbs)

        def query_group(group_id):
            return self._lookup_coordinator_group(
                group_id, token_len, block_hashes, hbm_hit_tokens,
                lookup_masks[group_id], local_masks[group_id],
            )

        if self.lookup_group_parallel:
            # Use a per-call pool and merge task-local sets in the caller.
            # Errors propagate so the worker treats the whole lookup as a miss.
            num_parallel_workers = min(
                self.lookup_group_parallel_workers, len(kv_cache_group_ids))
            with ThreadPoolExecutor(max_workers=num_parallel_workers) as executor:
                per_group_hits = list(executor.map(query_group, kv_cache_group_ids))
        else:
            per_group_hits = [query_group(group_id) for group_id in kv_cache_group_ids]
        existing_block_keys = set()
        for group_hits in per_group_hits:
            existing_block_keys.update(group_hits)

        # Group hits cannot be added: the native coordinator determines the
        # common hit length and preserves its default EAGLE pruning.
        _, num_hit_tokens = self.coord.find_longest_cache_hit(
            block_hashes, token_len, ExternalCachedBlockPool(existing_block_keys)
        )
        return num_hit_tokens
