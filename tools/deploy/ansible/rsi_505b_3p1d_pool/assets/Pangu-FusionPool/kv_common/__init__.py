# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""kv_common: shared building blocks for the KV modules (kv_offload, fusion_pool).

Deliberately light on import: importing ``kv_common`` must NOT pull in the
heavy submodules. Import from ``kv_common.npu_kv_views`` (or whatever
submodule) explicitly.
"""

__version__ = "0.1.0"

__all__ = ["__version__"]
