# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Single-NPU sampling checks; run explicitly outside normal CI discovery."""

import os

import pytest
import torch

pytest.importorskip("torch_npu")
from omni_npu.sample.ops import draft_sampling as module


pytestmark = [pytest.mark.integration, pytest.mark.npu]


@pytest.fixture(scope="module", autouse=True)
def npu_device():
    if not torch.npu.is_available():
        pytest.skip("NPU required")
    torch.npu.set_device(int(os.environ.get("OMNI_TEST_NPU_DEVICE", "0")))


@pytest.mark.parametrize("filtering", ["none", "k", "p", "both"])
def test_prepare_kernel_scales_strides_and_expands_filters(filtering):
    rows, vocab, slots, block = 4, 4101, 2, 2048
    x = torch.arange(rows * vocab * 2, dtype=torch.float32, device="npu").reshape(rows, -1)[:, ::2]
    # Non-unit strides and non-zero offsets in request metadata too.
    temperatures = torch.tensor([-1., 0., -1., 0.5], device="npu")[1::2]
    k = torch.tensor([-1, 1, -1, 7], dtype=torch.int32, device="npu")[1::2] if filtering in ("k", "both") else None
    p = torch.tensor([-1., 0.8, -1., 0.9], device="npu")[1::2] if filtering in ("p", "both") else None
    enabled = filtering != "none"
    out = torch.full((rows, vocab), float("nan"), dtype=torch.bfloat16 if enabled else torch.float32, device="npu")
    ko = torch.full((rows,), -1, dtype=torch.int32, device="npu")
    po = torch.full((rows,), float("nan"), dtype=out.dtype, device="npu")
    before = x.clone()
    module._prepare_draft_logits[(rows, 3)](
           x, temperatures, k, p, out, ko, po,
           *x.stride(), temperatures.stride(0),
           k.stride(0) if k is not None else 0,
           p.stride(0) if p is not None else 0,
           vocab, slots, k is not None, p is not None, enabled, block)
    scale = torch.tensor([1., 1., 0.5, 0.5], device="npu")
    torch.testing.assert_close(out, (x / scale[:, None]).to(out.dtype), rtol=0, atol=0)
    torch.testing.assert_close(x, before, rtol=0, atol=0)
    if enabled:
        expected_k = k.repeat_interleave(slots) if k is not None else torch.full_like(ko, vocab)
        expected_p = p.repeat_interleave(slots).to(po.dtype) if p is not None else torch.ones_like(po)
        torch.testing.assert_close(ko, expected_k, rtol=0, atol=0)
        torch.testing.assert_close(po, expected_p, rtol=0, atol=0)
    else:
        assert (ko == -1).all() and po.isnan().all()


def _sample(x, temperature, noise, scale, all_random):
    return module.sample_processed_draft_logits(
        x, temperature, noise, all_random, 1e-5, scale_temperature=scale,
    )


@pytest.mark.parametrize("scale,all_random", [(False, True), (False, False), (True, True), (True, False)])
def test_reduction_kernels_match_dense_reference(scale, all_random):
    # Avoid nearly tied maxima from paired linspace inputs; this case checks
    # numerical agreement with a clear winner.
    generator = torch.Generator().manual_seed(37)
    x = torch.randn(4, 4101 * 2, generator=generator).npu()[:, ::2]
    t = torch.tensor([0., 0.7], device="npu")
    u = (0.02 + 0.96 * torch.rand(4, 4101, generator=generator)).npu()
    before, noise_before = x.clone(), u.clone()
    ids, q = _sample(x, t, u, scale, all_random)
    expanded = t.cpu().double().repeat_interleave(2)
    logits = x.cpu().double()
    if scale:
        logits = logits / torch.where(expanded == 0, 1., expanded)[:, None]
    expected_q = logits.softmax(-1)
    scores = logits - (-(1 - u.cpu().double()).log()).log()
    if not all_random:
        scores = torch.where((expanded < 1e-5)[:, None], logits, scores)
    top_two = scores.topk(2, dim=-1).values
    assert bool(((top_two[:, 0] - top_two[:, 1]) > 1e-4).all())
    torch.testing.assert_close(q.cpu(), expected_q.float(), rtol=2e-6, atol=1e-7)
    torch.testing.assert_close(ids.cpu(), scores.argmax(-1), rtol=0, atol=0)
    torch.testing.assert_close(x, before, rtol=0, atol=0)
    torch.testing.assert_close(u, noise_before, rtol=0, atol=0)
