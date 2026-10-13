# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

"""Process-local handle for the worker's ``RoutedExpertsCapturer``.

The model runner publishes the capturer for use by NPU MoE layers. ``None``
means capture is disabled or not initialized.
"""

from typing import Any, Optional

_CAPTURER: Optional[Any] = None


def publish_capturer(capturer: Any) -> None:
    """Publish or replace the worker's capturer."""
    global _CAPTURER
    _CAPTURER = capturer


def get_capturer() -> Optional[Any]:
    """Return the published capturer, if available."""
    return _CAPTURER


def clear_capturer() -> None:
    """Drop the handle. Used by tests and by teardown paths."""
    global _CAPTURER
    _CAPTURER = None
