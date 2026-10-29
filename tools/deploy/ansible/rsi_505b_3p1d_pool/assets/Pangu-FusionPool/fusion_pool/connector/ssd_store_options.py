# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Validate embedded SSD storage options independently of device setup."""

import os
from pathlib import Path


def ssd_store_options(config, dp_rank, tp_rank, buffer_bytes):
    """When enabled, validate options and create <root>/dp_<dp_rank>_rank_<tp_rank>.

    Return Store.setup kwargs and the SSD staging budget in bytes, or ({}, None) when disabled.
    """
    enabled = config.get("enable_ssd_offload", False)
    if not isinstance(enabled, bool):
        raise ValueError("enable_ssd_offload must be a boolean")
    if not enabled:
        return {}, None

    root = config.get("ssd_offload_path")
    if not isinstance(root, str) or not root:
        raise ValueError("ssd_offload_path is required when SSD offload is enabled")
    if not os.path.isabs(root) or ".." in Path(root).parts:
        raise ValueError("ssd_offload_path must be an absolute path without '..'")
    if os.path.realpath(root) != os.path.normpath(root):
        raise ValueError("ssd_offload_path must not contain symbolic links")

    try:
        budget = int(buffer_bytes)
    except (TypeError, ValueError) as exc:
        raise ValueError("SSD buffer size must be an integer number of bytes") from exc
    if isinstance(buffer_bytes, bool) or budget <= 0:
        raise ValueError("SSD buffer size must be positive")

    # Different DP engines can all have TP rank 0 on the same host.
    path = os.path.join(root, f"dp_{dp_rank}_rank_{tp_rank}")
    os.makedirs(path, mode=0o700, exist_ok=True)
    return {"enable_ssd_offload": True, "ssd_offload_path": path}, budget
