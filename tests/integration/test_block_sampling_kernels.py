# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Single-NPU sampling checks; run explicitly outside normal CI discovery."""

import os

import pytest
import torch

pytest.importorskip("torch_npu")
from omni_npu.sample.ops import block_acceptance as acceptance
from omni_npu.sample.ops import block_selection as selection


pytestmark = [pytest.mark.integration, pytest.mark.npu]


@pytest.fixture(scope="module", autouse=True)
def npu_device():
    if not torch.npu.is_available():
        pytest.skip("NPU required")
    torch.npu.set_device(int(os.environ.get("OMNI_TEST_NPU_DEVICE", "0")))


class TestBlockKernelCases:
    @pytest.mark.parametrize("uniform, first_tau, first_output", [
        ([.9, .2], 2, [0, 0, 10]),
        ([.1, .9], 1, [0, 5, -1]),
    ])
    def test_acceptance_and_selection(self, uniform, first_tau, first_output):
        # Four requests: later-prefix success, no drafts, zero target mass,
        # and a greedy row whose existing output must remain untouched.
        ids = torch.tensor([0, 0, 0, 1, 1], dtype=torch.int32, device="npu")
        cu = torch.tensor([2, 2, 3, 5], dtype=torch.int32, device="npu")
        p = torch.tensor([[.25, .75], [.4, .6], [0., 1.], [.2, .8], [.2, .8]], device="npu")
        q = torch.tensor([[.5, .5], [.8, .2], [1., 0.], [0., 0.], [0., 0.]], device="npu")
        greedy = torch.tensor([False, False, False, True], device="npu")
        tau, accepted, scale = acceptance.compute_block_acceptance_triton(
            ids, cu, q, p, torch.tensor(uniform + [0., .5, .5], device="npu"), 2, greedy,
        )
        # Request 0 has c=(1/2,1/4), h=(1/6,1/4). A later prefix can
        # pass after an earlier failure; partial acceptance uses recovered[1].
        assert tau.tolist() == [first_tau, 0, 0, 0]
        assert accepted.tolist() == [True, first_tau == 2, False, False, False]
        assert torch.allclose(scale, torch.tensor([1., .5, 1., 1., 1.], device="npu"), rtol=2e-5, atol=2e-6)
        output = torch.full((4, 3), 99, dtype=torch.int32, device="npu")
        result = selection.select_block_tokens_triton(
            output, tau, cu, ids, torch.tensor([4, 5, 6, 7, 8], dtype=torch.int32, device="npu"),
            torch.tensor([[10], [11], [12], [13]], dtype=torch.int32, device="npu"), ~greedy, 2, -1,
        )
        assert output.tolist() == [first_output, [11, -1, -1], [6, -1, -1], [99, 99, 99]]
        assert result.dtype == torch.int64 and torch.equal(result, tau.long())

    def test_partial_residual_mass_covers_tail_tile(self):
        vocab = 4101
        p = torch.full((2, vocab), 0.1 / 4096, device="npu")
        p[:, 4096:] = 0.18
        q = torch.full_like(p, 1.0 / vocab)
        # A non-unit vocabulary stride must also be honored by the kernel.
        p = torch.stack((p, p), -1)[..., 0]
        partial = acceptance.compute_block_partial_z_triton(
            q, p, torch.tensor([0.5, 0.25], device="npu"), torch.tensor([2], device="npu"), 2,
        )
        residual = (0.5 * p[1] - q[1]).clamp_min(0)
        expected = torch.stack((residual[:4096].sum(), residual[4096:].sum()))
        assert partial.shape == (1, 1, 2) and expected[1] > 0
        assert torch.allclose(partial[0, 0], expected, rtol=2e-5, atol=2e-6)

    def test_empty_drafts_preserve_greedy_output(self):
        ids = torch.empty(0, dtype=torch.int32, device="npu")
        cu = torch.zeros(3, dtype=torch.int32, device="npu")
        tau, mask, scale = acceptance.compute_block_acceptance_triton(
            ids, cu, torch.empty(0, 2, device="npu"), torch.empty(0, 2, device="npu"), torch.empty(0, device="npu"), 0,
        )
        assert tau.tolist() == [0, 0, 0] and mask.numel() == scale.numel() == 0
        output = torch.full((3, 3), 99, dtype=torch.int32, device="npu")
        result = selection.select_block_tokens_triton(
            output, tau, cu, ids, ids, torch.tensor([[10], [11], [12]], dtype=torch.int32, device="npu"),
            torch.tensor([True, False, True], device="npu"), 2, -1,
        )
        assert result.tolist() == [0, 0, 0]
        assert output.tolist() == [[10, -1, -1], [99, 99, 99], [12, -1, -1]]
