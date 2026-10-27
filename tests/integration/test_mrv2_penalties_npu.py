# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Real-NPU penalty regression; run explicitly on an idle NPU.

Excluded from default CI collection in tests/conftest.py; invoke this file
explicitly for local NPU validation.

Checks the MRv2 state entry point against a CPU reference with mixed request
parameters, reordered slots, MTP prefixes, packed prompt bits and vocab tails.
"""

import os
from types import SimpleNamespace

import numpy as np
import pytest
import torch

pytest.importorskip("torch_npu")

from omni_npu.vllm_patches.patches.common import patch_mrv2_sampler
from omni_npu.worker.npu.ops import penalties as npu_penalties
import vllm.v1.worker.gpu.sample.penalties as up_penalties

pytestmark = [pytest.mark.integration, pytest.mark.npu]


@pytest.fixture
def patched(monkeypatch):
    if not torch.npu.is_available():
        pytest.skip("NPU required")
    torch.npu.set_device(int(os.environ.get("OMNI_TEST_NPU_DEVICE", "0")))
    monkeypatch.setattr(up_penalties, "apply_penalties", up_penalties.apply_penalties)
    ownership = dict(getattr(up_penalties, "_omni_npu_applied_patches", {}))
    ownership.pop("apply_penalties", None)
    monkeypatch.setattr(up_penalties, "_omni_npu_applied_patches", ownership, raising=False)
    patch_mrv2_sampler.MRv2PenaltiesPatch.apply()


def test_binding_preserves_mrv1(patched):
    from vllm.v1.sample.ops import penalties as v1_penalties

    assert up_penalties.apply_penalties is npu_penalties.apply_penalties
    assert v1_penalties.apply_all_penalties.__module__ == "vllm.v1.sample.ops.penalties"
    assert up_penalties.PenaltiesState.apply_penalties.__globals__["apply_penalties"] is npu_penalties.apply_penalties


@pytest.mark.parametrize("vocab", [257, 8209, 151552])
@pytest.mark.parametrize("width", [1, 8])
def test_penalties_cpu_reference(patched, vocab, width):
    # Per-slot (frequency, presence, repetition); include a neutral request.
    params = [(0., 0., 1.), (.3, 0., 1.), (0., .3, 1.),
              (.3, .3, 1.), (.3, .3, 1.2), (-.3, -.3, .8)]
    slots = [3, 0, 5, 2, 1, 4]
    mapping = torch.tensor([s for s in slots for _ in range(width)], dtype=torch.int32)
    positions = torch.arange(width, dtype=torch.int32).repeat(len(slots))
    ids = torch.tensor(([4, 5, 5, 7, 9, 31, 32, vocab - 1][:width]) * len(slots), dtype=torch.int64)
    rows = len(mapping)
    logits = torch.linspace(-3., 3., vocab).repeat(rows, 1)
    logits[:, 7] = 2.
    logits[:, 9] = 1.
    counts = torch.zeros(len(slots), vocab, dtype=torch.int32)
    prompt = torch.zeros(len(slots), (vocab + 31) // 32, dtype=torch.int32)
    prompt_tokens = [9, 31, 32, vocab - 1]
    for slot in range(len(slots)):
        counts[slot, 5] = slot + 1
        counts[slot, 7] = 1
        counts[slot, vocab - 1] = 2
        for token in prompt_tokens:
            bit = 1 << (token % 32)
            prompt[slot, token // 32] |= bit if bit < 2**31 else bit - 2**32

    expected = logits.clone()
    for row, slot in enumerate(mapping.tolist()):
        freq, pres, rep = params[slot]
        histogram = counts[slot].clone()
        pos = int(positions[row])
        for token in ids[row - pos + 1:row + 1]:
            histogram[token] += 1
        seen = histogram > 0
        repeated = seen.clone()
        repeated[prompt_tokens] = True
        # Independent selective updates, with both positive and negative logits.
        positive = repeated & (expected[row] > 0)
        negative = repeated & ~positive
        expected[row, positive] /= rep
        expected[row, negative] *= rep
        expected[row] -= freq * histogram
        expected[row] -= pres * seen

    # Exercise a padded row stride as well as request remapping.
    storage = torch.empty(rows, vocab + 8, device="npu", dtype=torch.float32)
    actual = storage[:, :vocab]
    actual.copy_(logits)
    state = SimpleNamespace(
        use_penalty=np.array([f != 0 or p != 0 or r != 1 for f, p, r in params]),
        repetition_penalty=SimpleNamespace(gpu=torch.tensor([p[2] for p in params], device="npu")),
        frequency_penalty=SimpleNamespace(gpu=torch.tensor([p[0] for p in params], device="npu")),
        presence_penalty=SimpleNamespace(gpu=torch.tensor([p[1] for p in params], device="npu")),
        prompt_bin_mask=prompt.npu(),
        output_bin_counts=counts.npu(),
    )
    before_counts = state.output_bin_counts.cpu().clone()
    before_prompt = state.prompt_bin_mask.cpu().clone()
    up_penalties.PenaltiesState.apply_penalties(
        state, actual, mapping.npu(), np.array(slots), ids.npu(), positions.npu()
    )
    torch.npu.synchronize()
    torch.testing.assert_close(actual.cpu(), expected, atol=2e-6, rtol=2e-6)
    assert torch.equal(state.output_bin_counts.cpu(), before_counts)
    assert torch.equal(state.prompt_bin_mask.cpu(), before_prompt)
