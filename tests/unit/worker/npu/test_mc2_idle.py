# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""PAD-only batches disable MC2 rows without replacing the graph input."""
from types import SimpleNamespace

import pytest
import torch
from vllm.v1.attention.backends.utils import PAD_SLOT_ID

from omni_npu.attention.backends.utils import generate_mc2_active_mask


@pytest.mark.parametrize("slots", [[], [PAD_SLOT_ID] * 4])
def test_pad_only_batch_clears_persistent_mask(slots):
    mask = torch.tensor([True, True, False, False])
    result = generate_mc2_active_mask(SimpleNamespace(mc2_mask=mask), 2, torch.tensor(slots, dtype=torch.int64))
    assert result is mask
    assert not mask.any()


@pytest.mark.parametrize("slots", [[0, 1, 2, 3], [PAD_SLOT_ID, 0, PAD_SLOT_ID, 2]])
def test_valid_slot_preserves_prefix_mask(slots):
    mask = torch.tensor([True, True, False, False])
    expected = mask.clone()
    assert generate_mc2_active_mask(SimpleNamespace(mc2_mask=mask), 2, torch.tensor(slots)) is mask
    assert torch.equal(mask, expected)


def test_idle_to_real_batch_restores_same_buffer():
    positions = torch.arange(8)
    mask = positions < 4
    pointer = mask.data_ptr()
    builder = SimpleNamespace(mc2_mask=mask, mc2_mask_positions=positions)
    generate_mc2_active_mask(builder, 4, torch.full((4,), PAD_SLOT_ID))
    assert not mask.any()
    generate_mc2_active_mask(builder, 3, torch.tensor([0, 1, 2, PAD_SLOT_ID]))
    assert mask.data_ptr() == pointer
    assert torch.equal(mask, positions < 3)
