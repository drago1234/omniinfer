# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""NPU top-k/top-p filtering: FP32 semantics and noncontiguous layouts.

Run on an idle device with OMNI_TEST_NPU_DEVICE (default: 0).
"""
import os
import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")
from omni_npu.worker.npu.sampler import apply_top_k_top_p
from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch


pytestmark = [pytest.mark.integration, pytest.mark.npu]

@pytest.fixture(scope="module", autouse=True)
def npu_device():
    if not torch.npu.is_available():
        pytest.skip("NPU required")
    torch.npu.set_device(int(os.environ.get("OMNI_TEST_NPU_DEVICE", "0")))


def assert_matches(logits, k, p, result):
    expected = apply_top_k_top_p_pytorch(
        logits.cpu().clone(), None if k is None else k.cpu().clone(),
        None if p is None else p.cpu().clone(), allow_cpu_sync=True,
    )
    actual = result.cpu()
    assert actual.dtype == logits.dtype
    assert torch.equal(torch.isneginf(actual), torch.isneginf(expected))
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@pytest.mark.parametrize("vocab", [1031, 151552])
@pytest.mark.parametrize("use_k,use_p", [(True, True), (True, False), (False, True), (False, False)])
def test_filter_matches_fp32(vocab, use_k, use_p):
    logits = torch.randn(3, vocab, generator=torch.Generator().manual_seed(17)).npu()
    k = torch.tensor([1, 100, vocab], dtype=torch.int32, device="npu") if use_k else None
    p = torch.tensor([0.8, 0.6, 1.0], device="npu") if use_p else None
    original = logits.clone()
    result = apply_top_k_top_p(logits, k, p)
    assert_matches(original, k, p, result)
    if k is None and p is None:
        assert result is logits


def test_filter_real_batch_and_noncontiguous_inputs():
    # Match a target verification batch: 3 requests * (7 proposals + bonus).
    rows, vocab = 24, 151552
    logits = torch.randn(rows, vocab * 2, generator=torch.Generator().manual_seed(23)).npu()[:, ::2]
    k = torch.full((rows * 2,), 100, dtype=torch.int64, device="npu")[::2]
    p = torch.full((rows * 2,), 0.8, device="npu")[::2]
    original = logits.clone()
    result = apply_top_k_top_p(logits, k, p)
    assert_matches(original, k, p, result)
