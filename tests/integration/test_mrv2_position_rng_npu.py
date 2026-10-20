# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Current MRv2 seed/position RNG reproducibility on a single NPU.

These tests check the compiled RNG, not old/new random-sequence equivalence or
model logits. Arithmetic and layout cases are in test_mrv2_sampling_kernels;
keep one device-graph case for each distinct RNG path here.
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
    from omni_npu.worker.npu.ops import position_sampling_rng
    return position_sampling_rng.make_uniform


def _tensor(values, dtype=torch.int64):
    return torch.tensor(values, dtype=dtype, device="npu")


def _assert_exact(actual, expected):
    # Copy only outside graph capture/replay. Zero tolerance is intentional:
    # these FP32 uniforms are exact integer multiples of 2**-24.
    torch.testing.assert_close(actual.cpu(), expected.cpu(), rtol=0, atol=0)


@pytest.mark.parametrize("mode,vocab", [
    ("accept", 1),
    ("draft", 151552),
    ("resample", 151552),
])
def test_graph_replay_observes_rng_metadata(rng, mode, vocab):
    # Two requests, each with seven proposed positions and a bonus position.
    base_mapping = [0] * 8 + [1] * 8
    base_seeds = [17, 2**32 + 19]
    base_positions = list(range(8192, 8200)) + list(range(200, 208))
    mapping = _tensor(base_mapping, torch.int32)
    seeds, positions = _tensor(base_seeds), _tensor(base_positions)
    cu, rejected = _tensor([0, 8, 16], torch.int32), _tensor([0, 7], torch.int32)
    metadata = (mapping, seeds, positions, cu, rejected)
    addresses = tuple(value.data_ptr() for value in metadata)

    def run():
        if mode == "resample":
            return rng(2, vocab, "npu", mapping, seeds, positions,
                       cu_num_logits=cu, rejected_steps=rejected)
        return rng(16, vocab, "npu", mapping, seeds, positions)

    for _ in range(3):
        run()
    torch.npu.synchronize()
    graph = torch.npu.NPUGraph()
    with torch.npu.graph(graph):
        captured = run()

    cases = [
        ("original", base_mapping, base_seeds, base_positions, [0, 7]),
        ("seed", base_mapping, [value + 2**32 for value in base_seeds], base_positions, [0, 7]),
        ("position", base_mapping, base_seeds, [value + 2**32 for value in base_positions], [0, 7]),
        ("mapping", [1] * 8 + [0] * 8, base_seeds, base_positions, [0, 7]),
        ("rejected", base_mapping, base_seeds, base_positions, [7, 0]),
        ("restored", base_mapping, base_seeds, base_positions, [0, 7]),
    ]
    baseline = None
    for label, row_values, seed_values, pos_values, offsets in cases:
        mapping.copy_(_tensor(row_values, torch.int32))
        seeds.copy_(_tensor(seed_values))
        positions.copy_(_tensor(pos_values))
        rejected.copy_(_tensor(offsets, torch.int32))
        assert tuple(value.data_ptr() for value in metadata) == addresses
        expected = run().cpu()
        graph.replay()
        torch.npu.synchronize()
        actual = captured.cpu()
        _assert_exact(actual, expected)
        for _ in range(2):
            graph.replay()
            torch.npu.synchronize()
            _assert_exact(captured, actual)

        if mode == "resample":
            full = rng(16, vocab, "npu", mapping, seeds, positions).cpu()
            _assert_exact(actual, full[[offsets[0], 8 + offsets[1]]])
        if label == "original":
            baseline = actual
        elif label == "restored" or (label == "rejected" and mode != "resample"):
            _assert_exact(actual, baseline)
        else:
            assert not torch.equal(actual, baseline), f"Updated {label} was ignored"
