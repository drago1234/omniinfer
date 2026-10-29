# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Build the fusion_pool NPU KV-cache-pool package for pangu-fusioncomm.

Pure-Python package: no C extensions to compile. Runtime requires the
``mooncake`` wheel (imported lazily by ``fusion_pool.transfer_engine``) and the
sibling ``kv_offload`` package (``prepare_kv_caches_for_offloading_registration``).
"""

from __future__ import annotations

from setuptools import setup

setup(
    name="pangu-fusioncomm-fusion-pool",
    version="0.1.0",
    description="Shared NPU KV cache pool connector (mooncake) for Pangu inference",
    packages=["fusion_pool", "fusion_pool.connector"],
    package_dir={"fusion_pool": "."},
    python_requires=">=3.9",
    # Self-registration: omni's platform plugin (omni/platform.py import_kernels)
    # invokes every entry point in the "omni_npu.kv_connectors" group at platform
    # init, which is early enough for vLLM's KVConnectorFactory name lookup.
    entry_points={
        "omni_npu.kv_connectors": [
            "NPUMooncakeStoreConnector = fusion_pool.connector:register_npumooncake_connector",
        ],
    },
)
