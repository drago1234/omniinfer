# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Opt-in event forwarding for one pool alongside non-event PD connectors.

vLLM v0.25.1 MultiConnector does not collect child KV events. This subclass
forwards the pool's native event container; TP aggregation, scheduler dispatch
and publication retain their upstream implementations. It is deliberately not
a generic implementation for multiple independent event-producing connectors.
"""

from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (
    MultiConnector as _MultiConnector,
)


class MultiConnector(_MultiConnector):
    # Keep the standard config name: v0.25.1's factory uses it to check HMA
    # support of all children, rather than just the wrapper's SupportsHMA tag.
    def __init__(self, vllm_config, role, kv_cache_config=None):
        super().__init__(vllm_config, role, kv_cache_config)
        from fusion_pool.connector.npu_mooncake_store_connector import (
            NPUMooncakeStoreConnector,
        )

        pools = [
            c for c in self._connectors if isinstance(c, NPUMooncakeStoreConnector)
        ]
        if len(pools) != 1:
            raise ValueError(
                "PoolEventMultiConnector requires exactly one NPUMooncakeStoreConnector"
            )
        self._pool_event_connector = pools[0]

    def get_kv_connector_kv_cache_events(self):
        return self._pool_event_connector.get_kv_connector_kv_cache_events()
