# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Real-NPU MRv2 sampling kernels checked against independent CPU references.

Run this file explicitly on a supported NPU environment; normal CI discovery
excludes it in tests/conftest.py. No interpreter or custom coverage collection.
"""

import os

import pytest
import torch

pytest.importorskip("torch_npu")

from vllm.triton_utils import tl, triton
from omni_npu.worker.npu.ops.gumbel import (
    gumbel_sample, murmur3_uniform32, spec_step_seed,
)
from omni_npu.worker.npu.ops import rejection_sampler_utils as rejection
from omni_npu.worker.npu.ops.rejection_sampler_utils import tl_rand32


pytestmark = [pytest.mark.integration, pytest.mark.npu]


@pytest.fixture(scope="module", autouse=True)
def npu_device():
    if not torch.npu.is_available():
        pytest.skip("NPU required")
    torch.npu.set_device(int(os.environ.get("OMNI_TEST_NPU_DEVICE", "0")))


def philox_reference(seed, position):
    """Independent integer oracle: full 64-bit products, not split multiplies."""
    mask = (1 << 32) - 1
    c = [position & mask, 0, 0, 0]
    key = [seed & mask, (seed >> 32) & mask]
    for _ in range(10):
        left, right = 0xD2511F53 * c[0], 0xCD9E8D57 * c[2]
        c = [((right >> 32) ^ c[1] ^ key[0]) & mask, right & mask,
             ((left >> 32) ^ c[3] ^ key[1]) & mask, left & mask]
        key = [(key[0] + 0x9E3779B9) & mask, (key[1] + 0xBB67AE85) & mask]
    # Triton folds signed int32 into nonnegative values before FP32 scaling.
    word = c[0] if c[0] < 2**31 else mask - c[0]
    return max(float(torch.tensor(word, dtype=torch.float32) * 4.6566127342e-10),
               4.6566127342e-10)


def uniform_reference(mapping, seeds, positions):
    return torch.tensor([
        0.5 if state < 0 else philox_reference(int(seeds[state]), int(pos))
        for state, pos in zip(mapping.tolist(), positions.tolist())
    ], dtype=torch.float32).reshape(-1, 1)


@triton.jit
def _acceptance_uniform_probe(out, mapping, seeds, positions, steps,
                              BLOCK: tl.constexpr):
    # Test-only probe of the production scalar helpers, not an RNG implementation.
    row = tl.program_id(0)
    state = tl.load(mapping + row)
    u = 0.5
    if state >= 0:
        seed = tl.load(seeds + state)
        if BLOCK:
            seed = spec_step_seed(seed, tl.load(steps + row))
        u = tl_rand32(seed, tl.load(positions + row), includes_zero=False)
    tl.store(out + row, u)


def acceptance_uniform(num_rows, device, mapping, seeds, positions, steps=None):
    out = torch.empty((num_rows, 1), dtype=torch.float32, device=device)
    if num_rows:
        _acceptance_uniform_probe[(num_rows,)](
            out, mapping, seeds, positions, steps, BLOCK=steps is not None)
    return out


def murmur_uniform_reference(mapping, seeds, positions, vocab, drafting=False):
    mask = 0xFFFFFFFF
    def rotate(x, n):
        return ((x << n) | (x >> (32 - n))) & mask
    def sample(seed, pos, offset):
        h = 0
        if drafting:
            pos += 1 << 30
        for key in [seed & mask, (seed >> 32) & mask, pos & mask, offset]:
            key = rotate((key * 0xCC9E2D51) & mask, 15)
            h ^= (key * 0x1B873593) & mask
            h = (rotate(h, 13) * 5 + 0xE6546B64) & mask
        h ^= 16
        h ^= h >> 16
        h = (h * 0x85EBCA6B) & mask
        h ^= h >> 13
        h = (h * 0xC2B2AE35) & mask
        h ^= h >> 16
        return (h + .5) / 2**32
    return torch.tensor([
        [.5 if state < 0 else sample(int(seeds[state]), int(pos), i)
         for i in range(vocab)]
        for state, pos in zip(mapping.tolist(), positions.tolist())
    ], dtype=torch.float32)


class TestMRv2KernelCases:
    def test_philox_reference_padding_and_reordering(self):
        seeds = torch.tensor([0, -1, 2**32 + 19, 2**63 - 1])
        base_positions = torch.tensor([0, 2**32 + 11, -1, 2**63 - 1, 8192, 0])
        base_mapping = torch.tensor([0, 2, 1, 3, -1, 0], dtype=torch.int32)
        # Check scalar RNG against the oracle and preserve request ordering.
        num_rows = 6
        positions, mapping = base_positions, base_mapping
        actual = acceptance_uniform(
            num_rows, "npu", mapping.npu(), seeds.npu(), positions.npu()
        ).cpu()
        assert actual.shape == (num_rows, 1)
        torch.testing.assert_close(actual, uniform_reference(mapping, seeds, positions), rtol=0, atol=6e-8)
        order = torch.tensor([3, 1, 0, 5])
        reordered = acceptance_uniform(
            4, "npu", mapping[order].npu(), seeds.npu(), positions[order].npu()
        ).cpu()
        assert torch.equal(reordered, actual[order])
        assert torch.equal(actual[0], actual[5])

    @pytest.mark.parametrize("changed_input", ["seed", "position"])
    def test_rng_repeats_after_metadata_update_and_restore(self, changed_input):
        # Determinism is indexed by request metadata, not call history.
        seeds = torch.tensor([17, 29])
        positions = torch.tensor([99, 2**32 + 99, 0, 99])
        mapping = torch.tensor([0, 1, -1, 0], dtype=torch.int32)

        def run():
            return acceptance_uniform(4, "npu", mapping.npu(), seeds.npu(), positions.npu()).cpu()

        baseline = run()
        torch.testing.assert_close(baseline, uniform_reference(mapping, seeds, positions), rtol=0, atol=6e-8)
        for _ in range(3):
            assert torch.equal(run(), baseline)
        metadata, index = (seeds, 0) if changed_input == "seed" else (positions, 1)
        original = metadata[index].clone()
        metadata[index] += 2**32 if changed_input == "seed" else 17
        changed = run()
        torch.testing.assert_close(changed, uniform_reference(mapping, seeds, positions), rtol=0, atol=6e-8)
        affected = [0, 3] if changed_input == "seed" else [1]
        unaffected = [i for i in range(4) if i not in affected]
        assert not torch.equal(changed[affected], baseline[affected])
        assert torch.equal(changed[unaffected], baseline[unaffected])
        metadata[index] = original
        assert torch.equal(run(), baseline)

    def test_rng_request_slot_relocation(self):
        seeds = torch.tensor([17, -1, 2**32 + 29])
        positions = torch.tensor([8192, 17, 2**32 + 9, 0, 8192])
        mapping = torch.tensor([0, 1, 2, -1, 0], dtype=torch.int32)
        baseline = acceptance_uniform(5, "npu", mapping.npu(), seeds.npu(), positions.npu()).cpu()
        relocated = torch.where(mapping >= 0, 2 - mapping, mapping)
        actual = acceptance_uniform(
            5, "npu", relocated.npu(), seeds.flip(0).npu(), positions.npu(),
        ).cpu()
        assert torch.equal(actual, baseline)
        torch.testing.assert_close(actual, uniform_reference(mapping, seeds, positions), rtol=0, atol=6e-8)


    @pytest.mark.parametrize("apply_temperature", [False, True])
    @pytest.mark.parametrize("column", [None, 2])
    def test_gumbel_against_independent_noise_and_processed_logits(self, apply_temperature, column):
        vocab = 1031
        logits = torch.randn(4, vocab, generator=torch.Generator().manual_seed(37))
        mapping = torch.tensor([2, 0, -1, 1], dtype=torch.int32)
        temperature = torch.tensor([0., .7, 1.3])
        seeds = torch.tensor([11, 2**32 + 17, -23])
        positions = torch.tensor([8192, 17, 0, 2**32 + 19])
        processed = torch.full((3, vocab) if column is None else (3, 4, vocab), 123.)
        processed_npu = processed.npu()
        result = gumbel_sample(
            logits.npu(), mapping.npu(), temperature.npu(), seeds.npu(), positions.npu(),
            apply_temperature, processed_npu,
            None if column is None else torch.tensor(column, dtype=torch.int32, device="npu"),
        ).cpu()
        processed = processed_npu.cpu()
        scaled = logits.clone()
        expected = logits.clone()
        noise_seeds = seeds if column is None else seeds ^ (column << 32)
        noise = murmur_uniform_reference(mapping, noise_seeds, positions, vocab, drafting=column is not None)
        for row, state in enumerate(mapping.tolist()):
            if state >= 0 and temperature[state] != 0:
                if apply_temperature:
                    scaled[row] /= temperature[state]
                expected[row] = scaled[row] - torch.log(-torch.log1p(-noise[row].double()))
            if state >= 0:
                written = processed[state] if column is None else processed[state, column]
                assert torch.allclose(written, scaled[row], rtol=1e-5, atol=1e-6)
        assert torch.equal(result, expected.argmax(-1))
        if column is not None:
            assert torch.all(processed[:, [0, 1, 3]] == 123.)

    @pytest.mark.parametrize("block", [False, True])
    @pytest.mark.parametrize("kind", ["bonus", "reject", "no_draft_logits", "stochastic"])
    def test_rejection_and_bonus_against_cpu_distribution(self, block, kind):
        n, k, vocab = 2, 3, 1031
        gen = torch.Generator().manual_seed(43)
        target = torch.randn(n, k + 1, vocab, generator=gen)
        draft = torch.randn(n, k, vocab, generator=gen)
        ids = torch.randint(vocab, (n, k + 1), generator=gen)
        temperature = torch.ones(n)
        if kind == "bonus":
            draft.copy_(target[:, :k])
        if kind == "reject":
            # Multiple residual candidates exercise noise-based selection.
            # Row 0 rejects immediately; row 1 accepts one token then rejects.
            target.fill_(-100.)
            target[:, :, 0] = 1.
            target[:, :, 2] = .5
            target[:, :, 3] = -.3
            draft.fill_(-100.)
            draft[:, :, 1] = 1.
            ids[:, 1:] = 1
            draft[1, 0].copy_(target[1, 0])
            ids[1, 1] = 0
        mapping = torch.tensor([1, 0], dtype=torch.int32)
        # Draft storage uses request slots, target rows use batch order.
        draft_slots = draft[mapping.long()].contiguous()
        expanded = mapping.repeat_interleave(k + 1)
        seeds = torch.tensor([17, 2**32 + 29])
        positions = torch.arange(n * (k + 1)) + 8192
        cu = torch.arange(n + 1, dtype=torch.int32) * (k + 1)
        local = torch.arange(k + 1, dtype=torch.int32).repeat(n)
        out, count = rejection.rejection_sample(
            target.reshape(-1, vocab).npu(), None if kind == "no_draft_logits" else draft_slots.npu(),
            ids.flatten().npu(), cu.npu(), positions.npu(), mapping.npu(), expanded.npu(),
            local.npu(), temperature.npu(), seeds.npu(), k, use_block_verification=block,
        )
        out, count = out.cpu(), count.cpu()
        p, q = target.double().softmax(-1), draft.double().softmax(-1)
        if kind == "no_draft_logits":
            q.zero_(); q.scatter_(-1, ids[:, 1:, None], 1.)
        for row in range(n):
            accepted = 0
            prefix, running = [], 1.
            for step in range(k):
                token = int(ids[row, step + 1])
                running = min(running * float(p[row, step, token] / q[row, step, token]), 1.)
                prefix.append(running)
            for step in range(k):
                token = int(ids[row, step + 1])
                step_seed = int(seeds[mapping[row]]) ^ ((step << 32) if block else 0)
                u = philox_reference(step_seed, int(positions[row * (k + 1) + step]))
                if block:
                    if step == k - 1:
                        threshold = prefix[step]
                    else:
                        mass = float((prefix[step] * p[row, step + 1] - q[row, step + 1]).clamp_min(0).sum())
                        denominator = mass + 1 - prefix[step]
                        threshold = mass / denominator if denominator > 0 else 1.
                    accept = u <= threshold
                else:
                    accept = u < float(p[row, step, token] / q[row, step, token])
                if accept:
                    accepted = step + 1
                elif not block:
                    break
            if kind == "bonus":
                assert accepted == k
            elif kind == "reject":
                assert accepted == row
            elif kind == "no_draft_logits":
                assert accepted < k
            assert int(count[row]) == accepted + 1
            assert torch.equal(out[row, :accepted], ids[row, 1:accepted + 1])
            z = target[row, accepted].double()
            if accepted < k:
                scale = prefix[accepted - 1] if block and accepted else 1.
                z = (scale * p[row, accepted] - q[row, accepted]).clamp_min(0).log()
            u = murmur_uniform_reference(expanded, seeds, positions, vocab).reshape(n, k + 1, vocab)[row, accepted].double()
            z -= torch.log(-torch.log1p(-u))
            assert int(out[row, accepted]) == int(z.argmax())


@triton.jit
def _murmur_uniform_probe(out, seeds, positions, V: tl.constexpr, B: tl.constexpr):
    row = tl.program_id(0)
    block = tl.program_id(1) * B + tl.arange(0, B)
    seed = tl.load(seeds + row)
    pos = tl.load(positions + row)
    tl.store(out + row * V + block, murmur3_uniform32(seed, pos, block), block < V)


def test_murmur_uniform_matches_community_integer_reference():
    # One independent oracle covers signed seeds, seed high bits, position
    # truncation (upstream semantics), and a partial final vocabulary tile.
    seeds = torch.tensor([0, -1, 2**32 + 19, 2**63 - 1, 19])
    positions = torch.tensor([0, 8192, 2**32 + 99, 2**30 + 99, 99])
    vocab = 4101
    actual = torch.empty(5, vocab, device="npu")
    _murmur_uniform_probe[(5, triton.cdiv(vocab, 1024))](
        actual, seeds.npu(), positions.npu(), vocab, 1024)
    expected = murmur_uniform_reference(torch.arange(5), seeds, positions, vocab)
    torch.testing.assert_close(actual.cpu(), expected, rtol=0, atol=0)


@pytest.mark.parametrize("block", [False, True])
def test_acceptance_rng_matches_reference_and_step_seed(block):
    seeds = torch.tensor([0, -1, 2**32 + 19, 2**63 - 1])
    mapping = torch.arange(4, dtype=torch.int32).repeat_interleave(8)
    steps = torch.arange(8).repeat(4)
    positions = steps + 8192
    actual = acceptance_uniform(32, "npu", mapping.npu(), seeds.npu(),
                                positions.npu(), steps.npu() if block else None).cpu()
    expected = torch.tensor([
        philox_reference(int(seeds[state]) ^ ((int(step) << 32) if block else 0), int(pos))
        for state, step, pos in zip(mapping, steps, positions)
    ]).reshape(-1, 1)
    torch.testing.assert_close(actual, expected, rtol=0, atol=6e-8)
    truncated = acceptance_uniform(32, "npu", mapping.npu(), seeds.npu(),
                                   (positions + 2**32).npu(), steps.npu() if block else None).cpu()
    assert torch.equal(actual, truncated)


@pytest.mark.parametrize("block", [False, True])
def test_rejection_greedy_matches_target_argmax(block):
    # One request accepts all drafts; the other rejects the first draft.
    # Both use target argmax, including the bonus selected by the insert kernel.
    n, k, vocab = 2, 3, 1031
    target = torch.full((n * (k + 1), vocab), -10., device="npu")
    target[:, 2] = 10.
    draft = torch.zeros((n, k, vocab), device="npu")
    ids = torch.full((n, k + 1), 2, dtype=torch.int64, device="npu")
    ids[1, 1] = 1
    mapping = torch.arange(n, dtype=torch.int32, device="npu")
    sampled, counts = rejection.rejection_sample(
        target, draft, ids.flatten(),
        torch.arange(n + 1, dtype=torch.int32, device="npu") * (k + 1),
        torch.arange(n * (k + 1), device="npu"), mapping,
        mapping.repeat_interleave(k + 1),
        torch.arange(k + 1, dtype=torch.int32, device="npu").repeat(n),
        torch.zeros(n, device="npu"),
        torch.tensor([17, 29], dtype=torch.int64, device="npu"), k,
        use_block_verification=block,
    )
    sampled, counts = sampled.cpu(), counts.cpu()
    assert counts.tolist() == [k + 1, 1]
    assert sampled[0, :k + 1].tolist() == [2] * (k + 1)
    assert sampled[1, 0].item() == 2
