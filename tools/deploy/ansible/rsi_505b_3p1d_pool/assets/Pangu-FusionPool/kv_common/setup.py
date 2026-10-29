# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Build the kv_common shared package for pangu-fusioncomm.

Pure-Python package: no C extensions, no heavy imports at package level.
Consumed by kv_offload and fusion_pool for the shared NPU KV registration-view
helpers.
"""

from __future__ import annotations

from setuptools import setup

setup(
    name="pangu-fusioncomm-kv-common",
    version="0.1.0",
    description="Shared NPU KV cache helpers for pangu-fusioncomm KV modules",
    packages=["kv_common"],
    package_dir={"kv_common": "."},
    python_requires=">=3.9",
)
