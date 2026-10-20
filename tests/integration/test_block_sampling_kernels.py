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

    @pytest.mark.parametrize("batch", [0, 2])
    def test_single_draft_and_empty_batch(self, batch):
        ids = torch.zeros(batch, dtype=torch.int32, device="npu")
        cu = torch.arange(1, batch + 1, dtype=torch.int32, device="npu")
        probs = torch.full((batch, 2), .5, device="npu")
        tau, accepted, scale = acceptance.compute_block_acceptance_triton(
            ids, cu, probs, probs, torch.full((batch,), .9, device="npu"), 1,
        )
        assert tau.tolist() == [1] * batch
        assert accepted.tolist() == [True] * batch
        assert scale.tolist() == [1.] * batch
        output = torch.full((batch, 2), 99, dtype=torch.int32, device="npu")
        result = selection.select_block_tokens_triton(
            output, tau, cu, ids, torch.full_like(ids, 4),
            torch.full((batch, 1), 10, dtype=torch.int32, device="npu"), None, 1, -1,
        )
        assert output.tolist() == [[0, 10]] * batch
        assert result.tolist() == [1] * batch

    @pytest.mark.parametrize("field, match", [
        ("output_token_ids", "output_token_ids must have shape"),
        ("tau", "tau must have shape"),
        ("bonus_token_ids", "bonus_token_ids must have one token"),
        ("recovered_token_ids", "same number of elements"),
        ("draft_stride", "All token selection tensors must be contiguous"),
        ("fill_this_time", "fill_this_time must have shape"),
        ("fill_stride", "fill_this_time must be contiguous"),
    ])
    def test_selection_rejects_invalid_layout(self, field, match):
        output = torch.full((2, 3), 99, dtype=torch.int32, device="npu")
        args = dict(
            output_token_ids=output, tau=torch.ones(2, dtype=torch.int32, device="npu"),
            cu_num_draft_tokens=torch.tensor([2, 4], dtype=torch.int32, device="npu"),
            draft_token_ids=torch.arange(4, dtype=torch.int32, device="npu"),
            recovered_token_ids=torch.arange(4, dtype=torch.int32, device="npu"),
            bonus_token_ids=torch.zeros(2, 1, dtype=torch.int32, device="npu"),
            fill_this_time=torch.ones(2, dtype=torch.bool, device="npu"),
            max_spec_len=2, placeholder_token_id=-1,
        )
        if field == "draft_stride":
            args["draft_token_ids"] = torch.arange(8, dtype=torch.int32, device="npu")[::2]
        elif field == "fill_stride":
            args["fill_this_time"] = torch.ones(4, dtype=torch.bool, device="npu")[::2]
        else:
            args[field] = args[field][:1]
        with pytest.raises(ValueError, match=match):
            selection.select_block_tokens_triton(**args)
        assert torch.all(output == 99)

    @pytest.mark.parametrize("case, match", [
        ("draft_stride", "draft_token_ids must be contiguous"),
        ("cu_stride", "cu_num_draft_tokens must be contiguous"),
        ("greedy_stride", "is_greedy must be contiguous"),
        ("negative_length", "max_num_draft_tokens must be non-negative"),
        ("uniform_length", "uniform_probs and c must have the same number"),
        ("partial_batch", "partial_z must have one row per request"),
    ])
    def test_acceptance_rejects_invalid_layout(self, case, match):
        ids = torch.zeros(2, dtype=torch.int32, device="npu")
        cu = torch.tensor([1, 2], dtype=torch.int32, device="npu")
        probs = torch.full((2, 2), .5, device="npu")
        with pytest.raises(ValueError, match=match):
            if case == "negative_length":
                acceptance.compute_block_partial_z_triton(probs, probs, torch.ones(2, device="npu"), cu, -1)
            elif case in ("uniform_length", "partial_batch"):
                acceptance.finalize_block_acceptance_triton(
                    torch.ones(2, device="npu"), torch.zeros(1 if case == "partial_batch" else 2, 0, 1, device="npu"),
                    torch.zeros(1 if case == "uniform_length" else 2, device="npu"), cu, draft_token_ids=ids,
                )
            else:
                if case == "draft_stride":
                    ids = torch.zeros(4, dtype=torch.int32, device="npu")[::2]
                if case == "cu_stride":
                    cu = torch.tensor([1, 0, 2, 0], dtype=torch.int32, device="npu")[::2]
                greedy = torch.zeros(4, dtype=torch.bool, device="npu")[::2] if case == "greedy_stride" else None
                acceptance.compute_block_prefix_triton(ids, cu, probs, probs, greedy)
