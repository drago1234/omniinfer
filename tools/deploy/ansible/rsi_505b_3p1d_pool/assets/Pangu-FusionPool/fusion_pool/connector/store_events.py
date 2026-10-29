# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Load/store event switches and receive-side event descriptions.

The source medium is an observation, not a destination-HBM residency promise.
Per-thread facades observe the keys actually submitted by upstream, so rotation,
disk-budget splitting, load masks and failure handling stay owned by vLLM.
"""

from vllm.logger import init_logger

logger = init_logger(__name__)


def store_events_requested(config):
    """Enable sender events only when explicitly requested."""
    extra = config.kv_transfer_config.kv_connector_extra_config
    enabled = extra.get("enable_store_events", False)
    if not isinstance(enabled, bool):
        raise ValueError("enable_store_events must be a JSON boolean")
    return enabled


def load_events_requested(config):
    """Opt into receive-side observation independently of GPU/master events."""
    extra = config.kv_transfer_config.kv_connector_extra_config
    enabled = extra.get("enable_load_events", False)
    if not isinstance(enabled, bool):
        raise ValueError("enable_load_events must be a JSON boolean")
    return enabled



class _EventDatabase:
    def __init__(self, database, group_idx, request, events):
        self._database = database
        self._group_idx = group_idx
        self._request = request
        self._events = events

    def __getattr__(self, name):
        return getattr(self._database, name)

    def process_tokens(self, *args, **kwargs):
        for start, end, block_hash in self._database.process_tokens(*args, **kwargs):
            try:
                # The predecessor belongs to the request's original hash chain,
                # not the rotated GET order or the filtered set of loaded keys.
                hash_size = self._database.hash_block_size
                parent = (
                    self._request.block_hashes[start // hash_size - 1]
                    if start > 0
                    else None
                )
                # Carry the tokens when the load path happens to have them,
                # but never require them. The router matches on block hashes
                # alone -- it parses token_ids and then never reads it -- and
                # on this path request.token_ids is None, so demanding them
                # dropped every load event and left the router seeing 7% of
                # the HBM the engines were actually hitting 54% in.
                token_ids = self._request.token_ids
                token_start = getattr(self._request, "token_ids_start", 0)
                tokens = []
                if token_ids is not None and (
                    token_start <= start and end <= token_start + len(token_ids)
                ):
                    local_start = start - token_start
                    local_end = end - token_start
                    tokens = list(token_ids[local_start:local_end])
                self._events[self._database.key_for(block_hash)] = {
                    "block_hash": block_hash,
                    "parent": parent,
                    "token_ids": tokens,
                    "block_size": self._database.block_size,
                    "group_idx": self._group_idx,
                }
            except Exception:
                # Observability must not turn a usable load into a load failure.
                logger.exception("Unable to describe Mooncake load event")
            yield start, end, block_hash



class _LoadEventStore:
    def __init__(self, store, event_info, publish):
        self._store = store
        self._event_info = event_info
        self._publish = publish
        self._descriptor_keys = None
        self._descriptors = None

    def __getattr__(self, name):
        return getattr(self._store, name)

    def batch_get_replica_desc(self, keys):
        # Reuse the descriptor query if upstream tier logging already made it.
        key_tuple = tuple(keys)
        if key_tuple != self._descriptor_keys:
            self._descriptors = self._store.batch_get_replica_desc(keys)
            self._descriptor_keys = key_tuple
        return self._descriptors

    def batch_get_into_multi_buffers(self, keys, *args, **kwargs):
        tiers = {}
        try:
            from vllm.distributed.kv_transfer.kv_connector.v1.mooncake.store.worker import (
                _get_replica_tiers_by_key,
            )

            tiers = _get_replica_tiers_by_key(self, keys)
        except Exception:
            logger.exception("Unable to classify Mooncake load event source")
        try:
            result = self._store.batch_get_into_multi_buffers(keys, *args, **kwargs)
        finally:
            self._descriptor_keys = None
            self._descriptors = None
        try:
            # A truncated/invalid response is not evidence of successful loads.
            # Return it unchanged so upstream's existing validation still runs.
            if len(result) != len(keys):
                return result
            from vllm.distributed.kv_events import BlockStored
            from vllm.v1.core.kv_cache_utils import maybe_convert_block_hash

            # v0.25.1 has BlockStored but lacks the newer medium constants.
            # Match the newer wire values without requiring a framework upgrade.
            media = {"memory": "CPU", "disk": "STORAGE"}
            events = []
            for key, value in zip(keys, result, strict=True):
                info = self._event_info.get(key)
                if value < 0 or info is None:
                    continue
                events.append(
                    BlockStored(
                        block_hashes=[maybe_convert_block_hash(info["block_hash"])],
                        parent_block_hash=(
                            maybe_convert_block_hash(info["parent"])
                            if info["parent"] is not None
                            else None
                        ),
                        token_ids=info["token_ids"],
                        block_size=info["block_size"],
                        lora_id=None,
                        lora_name=None,
                        medium=media.get(tiers.get(key)),
                        group_idx=info["group_idx"],
                    )
                )
            if events:
                self._publish(events)
        except Exception:
            logger.exception("Unable to publish Mooncake load events")
        return result
