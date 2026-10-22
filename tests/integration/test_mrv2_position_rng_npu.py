# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Current MRv2 seed/position RNG reproducibility on a single NPU.

These tests check the compiled RNG, not old/new random-sequence equivalence or
model logits. Arithmetic and layout cases are in test_mrv2_sampling_kernels;
exercise acceptance, direct/draft sampling and rejected/bonus selection here.
Run on an idle device with OMNI_TEST_NPU_DEVICE (default: 0).
"""

import os

import pytest
import torch


pytestmark = [pytest.mark.integration, pytest.mark.npu]


@pytest.fixture(scope="module")
def rng():
    pytest.importorskip("torch_npu")
    if not torch.npu.is_available():
        pytest.skip("NPU required")
    torch.npu.set_device(int(os.environ.get("OMNI_TEST_NPU_DEVICE", "0")))
    from integration.test_mrv2_sampling_kernels import acceptance_uniform
    return acceptance_uniform


def _tensor(values, dtype=torch.int64):
    return torch.tensor(values, dtype=dtype, device="npu")


def _assert_exact(actual, expected):
    # Copy only outside graph capture/replay; identical metadata must reproduce
    # exactly the same FP32 uniforms and sampled token IDs.
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("block", [False, True])
def test_acceptance_graph_replay_observes_rng_metadata(rng, block):
    base_mapping = [0, 1, -1, 0]
    base_seeds = [17, 2**32 + 19]
    base_positions = [8192, 99, 0, 8192]
    mapping = _tensor(base_mapping, torch.int32)
    seeds, positions = _tensor(base_seeds), _tensor(base_positions)
    steps = _tensor([0, 1, 2, 3])
    addresses = tuple(t.data_ptr() for t in (mapping, seeds, positions, steps))

    def run():
        return rng(4, "npu", mapping, seeds, positions, steps if block else None)

    for _ in range(3):
        run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = run()
    baseline = None
    for label in ["original", "seed", "position", "mapping", "step", "restore"]:
        if label == "seed":
            seeds.add_(2**32)
        elif label == "position":
            positions.add_(17)
        elif label == "mapping":
            mapping.copy_(_tensor([1, 0, -1, 1], torch.int32))
        elif label == "step":
            if not block:
                continue
            steps.add_(1)
        elif label == "restore":
            mapping.copy_(_tensor(base_mapping, torch.int32))
            seeds.copy_(_tensor(base_seeds))
            positions.copy_(_tensor(base_positions))
            steps.copy_(_tensor([0, 1, 2, 3]))
        assert tuple(t.data_ptr() for t in (mapping, seeds, positions, steps)) == addresses
        expected = run().cpu()
        graph.replay()
        torch.npu.synchronize()
        _assert_exact(captured, expected)
        if label == "original":
            baseline = expected
        elif label == "restore":
            _assert_exact(expected, baseline)
        else:
            assert not torch.equal(expected, baseline)


@pytest.mark.parametrize("drafting", [False, True])
def test_gumbel_graph_replay_observes_metadata(rng, drafting):
    from omni_npu.worker.npu.ops.gumbel import gumbel_sample
    vocab = 1031
    gen = torch.Generator().manual_seed(47)
    logits = torch.randn(4, vocab, generator=gen).npu()
    mapping = _tensor([0, 1, -1, 0], torch.int32)
    seeds = _tensor([17, 2**32 + 29])
    positions = _tensor([8192, 99, 0, 8192])
    temperature = torch.ones(2, device="npu")
    column = _tensor(0, torch.int32) if drafting else None
    cache = torch.empty(2, 3, vocab, device="npu") if drafting else None
    def run():
        return gumbel_sample(logits, mapping, temperature, seeds, positions, True,
                             output_processed_logits=cache,
                             output_processed_logits_col=column)
    for _ in range(3):
        run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = run()
    baseline = None
    for change in ["original", "seed", "position", "mapping", "column", "restore"]:
        if change == "seed": seeds.add_(2**32)
        if change == "position": positions.add_(17)
        if change == "mapping": mapping.copy_(_tensor([1, 0, -1, 1], torch.int32))
        if change == "column":
            if not drafting:
                continue
            column.fill_(2)
        if change == "restore":
            if drafting:
                column.zero_()
            seeds.copy_(_tensor([17, 2**32 + 29]))
            positions.copy_(_tensor([8192, 99, 0, 8192]))
            mapping.copy_(_tensor([0, 1, -1, 0], torch.int32))
        expected = run().cpu()
        graph.replay()
        torch.npu.synchronize()
        _assert_exact(captured, expected)
        if change == "original": baseline = expected
        elif change == "restore": _assert_exact(expected, baseline)
        else: assert not torch.equal(expected, baseline)


@pytest.mark.parametrize("block", [False, True])
def test_rejection_graph_replay_observes_selected_position(rng, block):
    from omni_npu.worker.npu.ops.rejection_sampler_utils import rejection_sample
    vocab, k = 1031, 3
    gen = torch.Generator().manual_seed(67)
    target = torch.randn(8, vocab, generator=gen).npu()
    draft = torch.randn(2, k, vocab, generator=gen).npu()
    ids = torch.zeros(8, dtype=torch.int64, device="npu")
    cu = _tensor([0, 4, 8], torch.int32)
    mapping = _tensor([0, 1], torch.int32)
    expanded = mapping.repeat_interleave(4)
    local = _tensor([0, 1, 2, 3] * 2, torch.int32)
    seeds, positions = _tensor([17, 2**32 + 29]), _tensor(list(range(8192, 8200)))
    temperature = torch.ones(2, device="npu")
    def run():
        return rejection_sample(target, draft, ids, cu, positions, mapping,
                                expanded, local, temperature, seeds, k,
                                use_block_verification=block)
    for _ in range(3): run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph): captured, counts = run()
    # Force all-accepted bonus and then an immediate rejection, at unchanged
    # tensor addresses, exercising the device-selected resampling position.
    for bonus in [True, False, True]:
        draft.copy_(target.reshape(2, 4, vocab)[:, :k])
        if not bonus:
            target[:, 0] = -100.
            draft[:, :, 0] = 100.
        seeds.add_(2**32)
        positions.add_(7)
        expected, expected_count = run()
        expected, expected_count = expected.cpu(), expected_count.cpu()
        graph.replay()
        torch.npu.synchronize()
        _assert_exact(counts, expected_count)
        assert torch.all(expected_count == (4 if bonus else 1))
        for row, count in enumerate(expected_count.tolist()):
            _assert_exact(captured[row, :count], expected[row, :count])
