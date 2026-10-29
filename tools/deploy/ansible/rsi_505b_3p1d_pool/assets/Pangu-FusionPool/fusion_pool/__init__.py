# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""fusion_pool: shared NPU KV cache pool (mooncake-based) for Pangu inference.

Deliberately light on import: importing ``fusion_pool`` must NOT pull in ``vllm``
or the connector submodules. Environments register
``NPUMooncakeStoreConnector`` with vLLM's ``KVConnectorFactory`` explicitly
(see the omni_vllm_25 ``omni/connector/register.py`` registration block), and
the lazy loader only imports this package when a deployment actually requests
the connector.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
