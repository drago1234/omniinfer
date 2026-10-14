# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: MIT
"""Unit tests for OpenPanguV2MOE W4A8 flag detection (pangu_v2_moe.py).

``OpenPanguV2MOE.__init__`` selects the W4A8 forward branch with:

    self._is_w4a8 = hasattr(self.experts, "w13_weight_int4_scale")

``__init__`` builds the full MoE runner, so these tests bypass it with
``__new__`` and run that same check against mock experts.
"""
from types import SimpleNamespace

import pytest
import torch

# Real import (the test env has vLLM installed, like test_pangu_v2_moe_residual).
from omni_npu.v1.models.pangu import pangu_v2_moe as model_mod


pytestmark = pytest.mark.unit


def _detect_w4a8(experts):
    """Mirror of the production detection line in OpenPanguV2MOE.__init__."""
    moe = model_mod.OpenPanguV2MOE.__new__(model_mod.OpenPanguV2MOE)
    moe.experts = experts
    moe._is_w4a8 = hasattr(moe.experts, "w13_weight_int4_scale")
    return moe


def test_w4a8_flag_false_without_int4_scale():
    """W8A8 experts have no int4 scale, so the W4A8 branch stays off."""
    experts = SimpleNamespace(w13_weight=torch.zeros(2, 4, 4, dtype=torch.int8))
    assert _detect_w4a8(experts)._is_w4a8 is False


@pytest.mark.parametrize("offset", [None, "present"])
def test_w4a8_flag_true_with_int4_scale(offset):
    """int4 scale selects W4A8 whether or not a weight offset is registered."""
    experts = SimpleNamespace(
        w13_weight=torch.zeros(2, 4, 4, dtype=torch.int8),
        w13_weight_int4_scale=torch.ones(2, 1, 8, dtype=torch.int64),
        w13_weight_offset=None if offset is None else torch.ones(2, 1, 8),
    )
    assert _detect_w4a8(experts)._is_w4a8 is True
