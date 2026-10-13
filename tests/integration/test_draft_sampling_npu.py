# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Real-NPU checks for draft sampling, RNG, output lifetime and numerical edges."""
from types import SimpleNamespace

import pytest
import torch

from omni_npu.sample.ops import draft_sampling as ds
from omni_npu.sample.rejection_sampler import compute_probs
from omni_npu.model_config.config_loader.loader import model_extra_config
from omni_npu.vllm_patches.patches.common.patch_eagle import EagleProposerPatch
from vllm.v1.sample.sampler import _SAMPLING_EPS
from vllm.v1.spec_decode.llm_base_proposer import compute_probs_and_sample_next_token


def entrypoint():
    return EagleProposerPatch._sample_from_logits


def metadata(temps, k=None, p=None, all_random=True, generators=None):
    return SimpleNamespace(
        temperature=torch.tensor(temps, device='npu', dtype=torch.float32),
        top_k=None if k is None else torch.tensor(k, device='npu', dtype=torch.int32),
        top_p=None if p is None else torch.tensor(p, device='npu', dtype=torch.float32),
        all_random=all_random, all_greedy=False, generators=generators or {})


def reference(logits, meta, uniform):
    rows = logits.shape[0]
    slots = rows // meta.temperature.numel()
    cu = torch.arange(slots, rows + 1, slots, device=logits.device, dtype=torch.int32)
    filtered = compute_probs(logits.float().clone(), cu, meta, use_npu_sample=True)
    q = filtered.softmax(-1, dtype=torch.float32)
    exp = -(1 - uniform).clamp(2**-24, 1-2**-24).log()
    ids = (q / exp).argmax(-1)
    if not meta.all_random:
        ids = torch.where((meta.temperature < _SAMPLING_EPS).repeat_interleave(slots),
                          q.argmax(-1), ids)
    return filtered, ids, q


# Cover each filtering branch once, then the two large-vocabulary reduction paths.
@pytest.mark.parametrize('vocab,slots,dtype,noncontiguous,filtering', [
    pytest.param(127, 1, torch.float32, False, 'none', id='small-unfiltered'),
    pytest.param(127, 1, torch.float32, False, 'k', id='top-k-only'),
    pytest.param(127, 1, torch.float32, False, 'p', id='top-p-only'),
    pytest.param(4099, 3, torch.bfloat16, True, 'both', id='strided-parallel-filtered'),
    pytest.param(151552, 1, torch.bfloat16, False, 'none', id='large-unfiltered'),
    pytest.param(153601, 2, torch.float16, False, 'both', id='large-filtered-tail'),
])
def test_entrypoint_q_and_fixed_uniform(monkeypatch, vocab, slots, dtype, noncontiguous, filtering):
    torch.manual_seed(918)
    monkeypatch.setattr(model_extra_config, 'dtype', torch.bfloat16)
    meta = metadata([0., 0.7, 1.3],
                    [1, 17, vocab] if filtering in ('k', 'both') else None,
                    [1., 0.8, 0.95] if filtering in ('p', 'both') else None,
                    all_random=False)
    x = torch.randn(3 * slots, vocab * (2 if noncontiguous else 1), device='npu', dtype=dtype)
    if noncontiguous:
        x = x[:, ::2]
    before = x.clone()
    u = torch.empty(x.shape, device='npu').uniform_()
    filtered, expected, q_ref = reference(x, meta, u)
    prepared = ds.prepare_draft_logits(x, meta.temperature, meta.top_k, meta.top_p, model_extra_config.dtype)
    torch.testing.assert_close(prepared, filtered, rtol=0, atol=0)
    monkeypatch.setattr(ds, 'make_uniform_noise', lambda *args: u)
    ids, q = entrypoint()(SimpleNamespace(_enable_probabilistic_draft_probs=True,
                                        _temperature_only=False), x, meta)
    torch.testing.assert_close(q, q_ref, rtol=3e-6, atol=2e-7)
    torch.testing.assert_close(q.sum(-1), torch.ones(x.shape[0], device='npu'), rtol=2e-6, atol=2e-6)
    torch.testing.assert_close(ids, expected, rtol=0, atol=0)
    torch.testing.assert_close(x, before, rtol=0, atol=0)
    assert ids.dtype == torch.int64 and q.dtype == torch.float32


def test_endpoints_filtered_tiles_and_ties():
    x = torch.full((3, 4099), -float('inf'), device='npu')
    x[:, 17] = 0
    x[:, 4098] = 0
    t = torch.tensor([1., 0., 1.], device='npu')
    u = torch.zeros_like(x)
    u[2].fill_(1.)  # Defensive endpoint, although uniform_ itself excludes 1.
    ids, q = ds.sample_processed_draft_logits(x, t, u, False, _SAMPLING_EPS)
    torch.testing.assert_close(q, x.softmax(-1), rtol=0, atol=0)
    assert ids.cpu().tolist() == [17, 17, 17]
    assert bool(torch.isfinite(q).all())


@pytest.mark.parametrize('seeded_rows', [[1], [0, 1, 2]])
def test_rng_state_and_seed_reproducibility(seeded_rows):
    x = torch.empty(3, 127, device='npu')
    generators = {i: torch.Generator(device='npu').manual_seed(100 + i) for i in seeded_rows}
    torch.npu.manual_seed(918)
    initial = torch.npu.get_rng_state()
    states = {i: g.get_state() for i, g in generators.items()}
    actual = ds.make_uniform_noise(x, generators, 1)
    after = torch.npu.get_rng_state()
    after_seeded = {i: g.get_state() for i, g in generators.items()}
    torch.npu.set_rng_state(initial)
    for i, g in generators.items():
        g.set_state(states[i])
    expected = torch.empty_like(x)
    if len(generators) != 3:
        expected.uniform_()
    for i, g in generators.items():
        expected[i:i+1].uniform_(generator=g)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(after, torch.npu.get_rng_state())
    for i, g in generators.items():
        assert torch.equal(after_seeded[i], g.get_state())


def test_parallel_rng_keeps_default_generator():
    x = torch.empty(6, 127, device='npu')
    g = torch.Generator(device='npu').manual_seed(8)
    state = g.get_state()
    default_state = torch.npu.get_rng_state()
    actual = ds.make_uniform_noise(x, {0: g}, 3)
    after = torch.npu.get_rng_state()
    assert torch.equal(state, g.get_state())
    torch.npu.set_rng_state(default_state)
    expected = torch.empty_like(x).uniform_()
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)
    assert torch.equal(after, torch.npu.get_rng_state())


@pytest.mark.parametrize('probabilistic,all_greedy,temperature_only', [
    (False, False, False), (True, True, False), (True, True, True),
])
def test_greedy_does_not_consume_rng(probabilistic, all_greedy, temperature_only):
    x = torch.randn(3, 127, device='npu')
    meta = metadata([1., 1., 1.])
    meta.all_greedy = all_greedy
    state = torch.npu.get_rng_state()
    ids, q = entrypoint()(SimpleNamespace(_enable_probabilistic_draft_probs=probabilistic,
                                        _temperature_only=temperature_only), x, meta)
    torch.testing.assert_close(ids, x.argmax(-1), rtol=0, atol=0)
    assert q is None and torch.equal(state, torch.npu.get_rng_state())


@pytest.mark.parametrize('temps', [[0.7, 1.3], [0., 1.3]])
def test_temperature_only_matches_upstream(temps):
    x = torch.randn(2, 127, device='npu', dtype=torch.float32)
    meta = metadata(temps, k=[1, 1], p=[.01, .01], all_random=all(t > 0 for t in temps))
    proposer = SimpleNamespace(_enable_probabilistic_draft_probs=True,
                               _temperature_only=True, use_fp64_gumbel=False)
    state = torch.npu.get_rng_state()
    ids, q = entrypoint()(proposer, x.clone(), meta)
    after = torch.npu.get_rng_state()
    torch.npu.set_rng_state(state)
    ref_ids, ref_q = compute_probs_and_sample_next_token(x.clone(), meta, False)
    torch.testing.assert_close(ids, ref_ids, rtol=0, atol=0)
    torch.testing.assert_close(q, ref_q, rtol=0, atol=0)
    assert torch.equal(after, torch.npu.get_rng_state())
    assert q.dtype == torch.float32 and (q > 0).all()  # top-k/top-p are ignored.


def test_step_outputs_survive_later_calls():
    x = torch.randn(1, 4099, device='npu', dtype=torch.bfloat16)
    meta = SimpleNamespace(temperature=torch.ones(1, device='npu'),
        top_k=torch.tensor([100], device='npu', dtype=torch.int32),
        top_p=torch.tensor([.8], device='npu'), generators={}, all_random=True)
    saved = []
    for i in range(3):
        x.neg_()
        meta.temperature.fill_(.5 + i / 5)
        ids, q = ds.sample_draft_logits(x, meta, torch.bfloat16, 1e-5)
        saved.append((ids, q, ids.clone(), q.clone()))
    torch.npu.synchronize()
    for ids, q, ids_copy, q_copy in saved:
        torch.testing.assert_close(ids, ids_copy, rtol=0, atol=0)
        torch.testing.assert_close(q, q_copy, rtol=0, atol=0)


def test_extreme_finite_logits():
    # The winner is in the partially populated last tile; other masses underflow.
    x = torch.full((1, 4099), -1000., device='npu')
    x[0, 4098] = 0
    u = torch.full_like(x, 0.5)
    t = torch.ones(1, device='npu')
    ids, q = ds.sample_processed_draft_logits(x, t, u, True, 1e-5)
    torch.testing.assert_close(q, x.softmax(-1), rtol=0, atol=0)
    assert ids.item() == 4098


def test_large_common_logit_offset_keeps_random_selection():
    x = torch.full((1, 4099), 1e20, device='npu')
    u = torch.full_like(x, .5)
    u[0, 3000] = 0
    ids, q = ds.sample_processed_draft_logits(x, torch.ones(1, device='npu'), u, True, 1e-5)
    assert ids.item() == 3000
    torch.testing.assert_close(q, x.softmax(-1))
