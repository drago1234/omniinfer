# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""KV pool connector registration for vLLM."""

from vllm.logger import init_logger

logger = init_logger(__name__)


def _registered_connector_names(factory) -> set[str]:
    """Return connector names exposed by supported vLLM registries."""
    registry = getattr(factory, "_registry", None)
    if not isinstance(registry, dict):
        registry = getattr(factory, "_connectors", None)
    return set(registry) if isinstance(registry, dict) else set()


def _safe_register(name: str, module: str, class_name: str) -> None:
    """Register a KV connector into vLLM's KVConnectorFactory defensively."""
    from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory

    if name in _registered_connector_names(KVConnectorFactory):
        logger.debug("connector already registered; skipping: %s", name)
        return

    KVConnectorFactory.register_connector(name, module, class_name)
    logger.info("connector registered: %s -> %s.%s", name, module, class_name)


def register_npumooncake_connector() -> None:
    """Register NPUMooncakeStoreConnector into vLLM."""

    _safe_register(
        "NPUMooncakeStoreConnector",
        "fusion_pool.connector.npu_mooncake_store_connector",
        "NPUMooncakeStoreConnector",
    )


__all__ = ["register_npumooncake_connector"]
