# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Training/inference accuracy-alignment helpers.

The package intentionally contains the implementation while the vLLM patch
tree only contains the small runtime entry point needed by PatchManager.
"""

from .config import (
    AlignmentConfigError,
    PureRealConfig,
    TrainRefConfig,
    alignment_requested,
    pure_real_requested,
    real_requested,
    requested_mode,
    train_ref_requested,
)

__all__ = [
    "AlignmentConfigError",
    "PureRealConfig",
    "TrainRefConfig",
    "alignment_requested",
    "pure_real_requested",
    "real_requested",
    "requested_mode",
    "train_ref_requested",
]
