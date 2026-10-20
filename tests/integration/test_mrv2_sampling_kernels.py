# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Real-NPU MRv2 sampling kernels checked against independent CPU references.

Run this file explicitly on a supported NPU environment; normal CI discovery
excludes it in tests/conftest.py. No interpreter or custom coverage collection.
"""

import os

import pytest
import torch

torch_npu = pytest.importorskip("torch_npu")

from omni_npu.worker.npu import sampler
from omni_npu.worker.npu.ops import rejection_sampler_utils as rejection
from omni_npu.worker.npu.ops.position_sampling_rng import make_uniform


pytestmark = [pytest.mark.integration, pytest.mark.npu]


@pytest.fixture(scope="module", autouse=True)
def npu_device():
    if not torch.npu.is_available():
        pytest.skip("NPU required")
    torch.npu.set_device(int(os.environ.get("OMNI_TEST_NPU_DEVICE", "0")))


def philox_reference(seed, position, vocabulary_index):
    """Independent integer oracle: full 64-bit products, not split multiplies."""
    mask = (1 << 32) - 1
    c = [vocabulary_index // 4, 0, position & mask, (position >> 32) & mask]
    key = [seed & mask, (seed >> 32) & mask]
    for _ in range(10):
        left, right = 0xD2511F53 * c[0], 0xCD9E8D57 * c[2]
        c = [((right >> 32) ^ c[1] ^ key[0]) & mask, right & mask,
             ((left >> 32) ^ c[3] ^ key[1]) & mask, left & mask]
        key = [(key[0] + 0x9E3779B9) & mask, (key[1] + 0xBB67AE85) & mask]
    return (c[vocabulary_index % 4] >> 8) / (1 << 24)


def uniform_reference(mapping, seeds, positions, vocab):
    return torch.tensor([
        [0.5 if state < 0 else philox_reference(int(seeds[state]), int(pos), i)
         for i in range(vocab)]
        for state, pos in zip(mapping.tolist(), positions.tolist())
    ], dtype=torch.float32)


class TestMRv2KernelCases:
    @pytest.mark.parametrize("vocab", [1, 3, 1031, 4101])
    def test_philox_reference_padding_and_reordering(self, vocab):
        seeds = torch.tensor([0, -1, 2**32 + 19, 2**63 - 1])
        positions = torch.tensor([0, 2**32 + 11, -1, 2**63 - 1, 8192, 0])
        mapping = torch.tensor([0, 2, 1, 3, -1, 0], dtype=torch.int32)
        actual = make_uniform(6, vocab, "npu", mapping.npu(), seeds.npu(), positions.npu()).cpu()
        expected = uniform_reference(mapping, seeds, positions, vocab)
        assert torch.equal(actual, expected)
        order = torch.tensor([3, 1, 0, 5])
        reordered = make_uniform(4, vocab, "npu", mapping[order].npu(), seeds.npu(), positions[order].npu()).cpu()
        assert torch.equal(reordered, actual[order])
        first = make_uniform(6, 1, "npu", mapping.npu(), seeds.npu(), positions.npu()).cpu()
        assert torch.equal(first, actual[:, :1])
        assert torch.equal(actual[0], actual[5])

    @pytest.mark.parametrize("vocab", [1, 1031])
    @pytest.mark.parametrize("changed_input", ["seed", "position"])
    def test_rng_repeats_after_metadata_update_and_restore(self, vocab, changed_input):
        # Determinism is indexed by request metadata, not call history.
        seeds = torch.tensor([17, 29])
        positions = torch.tensor([99, 2**32 + 99, 0, 99])
        mapping = torch.tensor([0, 1, -1, 0], dtype=torch.int32)

        def run():
            return make_uniform(4, vocab, "npu", mapping.npu(), seeds.npu(), positions.npu()).cpu()

        baseline = run()
        assert torch.equal(baseline, uniform_reference(mapping, seeds, positions, vocab))
        for _ in range(3):
            assert torch.equal(run(), baseline)
        metadata, index = (seeds, 0) if changed_input == "seed" else (positions, 1)
        original = metadata[index].clone()
        metadata[index] += 2**32  # Exercise the high word independently.
        changed = run()
        assert torch.equal(changed, uniform_reference(mapping, seeds, positions, vocab))
        affected = [0, 3] if changed_input == "seed" else [1]
        unaffected = [i for i in range(4) if i not in affected]
        assert not torch.equal(changed[affected], baseline[affected])
        assert torch.equal(changed[unaffected], baseline[unaffected])
        metadata[index] = original
        assert torch.equal(run(), baseline)

    @pytest.mark.parametrize("vocab", [1, 1031])
    def test_rng_request_slot_relocation(self, vocab):
        seeds = torch.tensor([17, -1, 2**32 + 29])
        positions = torch.tensor([8192, 17, 2**32 + 9, 0, 8192])
        mapping = torch.tensor([0, 1, 2, -1, 0], dtype=torch.int32)
        baseline = make_uniform(5, vocab, "npu", mapping.npu(), seeds.npu(), positions.npu()).cpu()
        relocated = torch.where(mapping >= 0, 2 - mapping, mapping)
        actual = make_uniform(
            5, vocab, "npu", relocated.npu(), seeds.flip(0).npu(), positions.npu(),
        ).cpu()
        assert torch.equal(actual, baseline)
        assert torch.equal(actual, uniform_reference(mapping, seeds, positions, vocab))

    def test_acceptance_tile_tail_and_selected_rejection_rows(self):
        n = 259
        mapping = torch.arange(n, dtype=torch.int32) % 3
        seeds = torch.tensor([17, 2**32 + 23, -19])
        positions = torch.arange(n) + 8192
        actual = make_uniform(n, 1, "npu", mapping.npu(), seeds.npu(), positions.npu()).cpu()
        assert torch.equal(actual, uniform_reference(mapping, seeds, positions, 1))
        cu = torch.tensor([0, 3, 7], dtype=torch.int32)
        rejected = torch.tensor([0, 3], dtype=torch.int32)
        for steps in ([0, 3], [2, 0]):
            rejected.copy_(torch.tensor(steps))
            selected = cu[:2].long() + rejected.long()
            result = make_uniform(
                2, 1031, "npu", mapping.npu(), seeds.npu(), positions.npu(),
                cu_num_logits=cu.npu(), rejected_steps=rejected.npu(),
            ).cpu()
            assert torch.equal(result, uniform_reference(mapping[selected], seeds, positions[selected], 1031))

    @pytest.mark.parametrize("rows,vocab", [(0, 10), (2, 0)])
    def test_empty_uniform(self, rows, vocab):
        empty = torch.empty(0, dtype=torch.int64, device="npu")
        out = make_uniform(rows, vocab, "npu", empty, empty, empty)
        assert out.shape == (rows, vocab) and out.dtype == torch.float32

    @pytest.mark.parametrize("kwargs", [dict(num_rows=-1), dict(vocab_size=-1),
                                       dict(cu_num_logits=torch.tensor([0])),
                                       dict(rejected_steps=torch.tensor([0]))])
    def test_invalid_uniform_arguments(self, kwargs):
        args = dict(num_rows=1, vocab_size=1, device="npu",
                    mapping=torch.tensor([0], device="npu"),
                    seeds=torch.tensor([1], device="npu"),
                    positions=torch.tensor([2], device="npu"))
        args.update(kwargs)
        with pytest.raises(ValueError):
            make_uniform(**args)

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
        result = sampler.gumbel_sample(
            logits.npu(), mapping.npu(), temperature.npu(), seeds.npu(), positions.npu(),
            apply_temperature, processed_npu,
            None if column is None else torch.tensor(column, dtype=torch.int32, device="npu"),
        ).cpu()
        processed = processed_npu.cpu()
        scaled = logits.clone()
        expected = logits.clone()
        noise = uniform_reference(mapping, seeds, positions, vocab)
        for row, state in enumerate(mapping.tolist()):
            if state >= 0 and temperature[state] != 0:
                if apply_temperature:
                    scaled[row] /= temperature[state]
                expected[row] = scaled[row] - torch.log(-torch.log(noise[row] + 1e-20) + 1e-20)
            if state >= 0:
                written = processed[state] if column is None else processed[state, column]
                assert torch.allclose(written, scaled[row], rtol=1e-5, atol=1e-6)
        assert torch.equal(result, expected.argmax(-1))
        if column is not None:
            assert torch.all(processed[:, [0, 1, 3]] == 123.)

    @pytest.mark.parametrize("block", [False, True])
    @pytest.mark.parametrize("kind", ["random", "bonus", "reject", "greedy", "one_hot"])
    def test_rejection_and_bonus_against_cpu_distribution(self, block, kind):
        n, k, vocab = 2, 3, 1031
        gen = torch.Generator().manual_seed(43)
        target = torch.randn(n, k + 1, vocab, generator=gen)
        draft = torch.randn(n, k, vocab, generator=gen)
        ids = torch.randint(vocab, (n, k + 1), generator=gen)
        temperature = torch.zeros(n) if kind == "greedy" else torch.ones(n)
        if kind == "bonus":
            draft.copy_(target[:, :k])
        if kind == "reject":
            target.fill_(-100.); target[:, :, 0] = 100.
            draft.fill_(-100.); draft[:, :, 1] = 100.
            ids[:, 1:] = 1
        if kind == "greedy":
            ids[:, 1:] = target[:, :k].argmax(-1)
            ids[0, 2] = (ids[0, 2] + 1) % vocab
        mapping = torch.tensor([1, 0], dtype=torch.int32)
        # Draft storage uses request slots, target rows use batch order.
        draft_slots = draft[mapping.long()].contiguous()
        expanded = mapping.repeat_interleave(k + 1)
        seeds = torch.tensor([17, 2**32 + 29])
        positions = torch.arange(n * (k + 1)) + 8192
        cu = torch.arange(n + 1, dtype=torch.int32) * (k + 1)
        local = torch.arange(k + 1, dtype=torch.int32).repeat(n)
        out, count = rejection.rejection_sample(
            target.reshape(-1, vocab).npu(), None if kind == "one_hot" else draft_slots.npu(),
            ids.flatten().npu(), cu.npu(), positions.npu(), mapping.npu(), expanded.npu(),
            local.npu(), temperature.npu(), seeds.npu(), k, use_block_verification=block,
        )
        out, count = out.cpu(), count.cpu()
        uniforms = uniform_reference(expanded, seeds, positions, vocab).reshape(n, k + 1, vocab)
        p, q = target.double().softmax(-1), draft.double().softmax(-1)
        if kind == "one_hot":
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
                u = max(float(uniforms[row, step, 0]), 4.6566127342e-10)
                if kind == "greedy":
                    accept = token == int(target[row, step].argmax())
                elif block:
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
                elif not block or kind == "greedy":
                    break
            assert int(count[row]) == accepted + 1
            assert torch.equal(out[row, :accepted], ids[row, 1:accepted + 1])
            z = target[row, accepted].double()
            if kind != "greedy" and accepted < k:
                scale = prefix[accepted - 1] if block and accepted else 1.
                z = (scale * p[row, accepted] - q[row, accepted]).clamp_min(0).log()
            if kind != "greedy":
                u = uniforms[row, accepted].double()
                z -= torch.log(-torch.log(u + 1e-20) + 1e-20)
            assert int(out[row, accepted]) == int(z.argmax())

    @pytest.mark.parametrize("use_k,use_p", [(False, False), (True, False), (False, True), (True, True)])
    def test_native_filter_argument_contract(self, monkeypatch, use_k, use_p):
        logits = torch.arange(24, dtype=torch.float32).reshape(3, 8)[:, ::2]
        k = torch.tensor([1, 9, 2, 9, 3, 9])[::2] if use_k else None
        p = torch.tensor([.8, 0., .7, 0., 1., 0.], dtype=torch.float64)[::2] if use_p else None
        calls = []

        def native(x, actual_k, actual_p, *, q, is_need_logits):
            calls.append(True)
            assert x.is_contiguous() and actual_k.is_contiguous() and actual_p.is_contiguous()
            assert actual_k.dtype == torch.int32 and actual_p.dtype == logits.dtype
            assert actual_k.tolist() == ([1, 2, 3] if use_k else [4, 4, 4])
            assert torch.equal(actual_p, p.float() if use_p else torch.ones(3))
            assert q is None and is_need_logits
            return None, x.double()

        monkeypatch.setattr(torch_npu, "npu_top_k_top_p_sample", native)
        result = sampler.apply_top_k_top_p(logits, k, p)
        assert result.dtype == logits.dtype and torch.equal(result, logits)
        assert len(calls) == int(use_k or use_p)
        if not use_k and not use_p:
            assert result is logits
