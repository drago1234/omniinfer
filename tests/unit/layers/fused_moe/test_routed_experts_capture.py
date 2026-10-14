# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT
"""Tests for the process-local routed-experts capturer handle."""
import importlib.util
from pathlib import Path

import pytest

_HANDLE_PATH = (
    Path(__file__).resolve().parents[4]
    / "omni"
    / "layers"
    / "fused_moe"
    / "routed_experts_handle.py"
)


@pytest.fixture
def handle():
    spec = importlib.util.spec_from_file_location(
        "_t5_routed_experts_handle_under_test", _HANDLE_PATH
    )
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class _Capturer:
    pass


def test_handle_starts_empty(handle):
    assert handle.get_capturer() is None


def test_publish_then_get_round_trips(handle):
    capturer = _Capturer()
    handle.publish_capturer(capturer)
    assert handle.get_capturer() is capturer


def test_clear_resets_to_none(handle):
    handle.publish_capturer(_Capturer())
    handle.clear_capturer()
    assert handle.get_capturer() is None


def test_publish_replaces_previous_capturer(handle):
    """Publishing a new capturer replaces the stale handle."""
    first, second = _Capturer(), _Capturer()
    handle.publish_capturer(first)
    handle.publish_capturer(second)
    assert handle.get_capturer() is second
