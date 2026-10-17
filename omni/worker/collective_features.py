# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Generic allowlisted feature dispatcher for NPUWorker collective RPC."""
from __future__ import annotations

from collections.abc import Callable

FeatureHandler = Callable[..., dict]
_FEATURE_HANDLERS: dict[str, dict[str, FeatureHandler]] = {}


def register_collective_feature(feature: str, action: str):
    """Register one allowlisted feature action."""
    if not feature or not action:
        raise ValueError("feature and action names must be non-empty")

    def register(handler: FeatureHandler) -> FeatureHandler:
        actions = _FEATURE_HANDLERS.setdefault(feature, {})
        if action in actions:
            raise RuntimeError(
                f"collective feature {feature!r} already registers action {action!r}")
        actions[action] = handler
        return handler

    return register


def collective_feature(worker, feature, action="configure", **options):
    """Dispatch an allowlisted runtime feature control request."""
    if not isinstance(feature, str) or not feature:
        raise ValueError("'feature' must be a non-empty string")
    if not isinstance(action, str) or not action:
        raise ValueError("'action' must be a non-empty string")
    actions = _FEATURE_HANDLERS.get(feature)
    if actions is None:
        raise ValueError(
            f"unsupported collective feature {feature!r}; "
            f"supported features: {tuple(_FEATURE_HANDLERS)}")
    handler = actions.get(action)
    if handler is None:
        raise ValueError(
            f"unsupported action {action!r} for feature {feature!r}; "
            f"supported actions: {tuple(actions)}")
    result = handler(worker, **options)
    return {
        "feature": feature,
        "action": action,
        "global_rank": getattr(worker, "rank", None),
        "result": result,
    }
