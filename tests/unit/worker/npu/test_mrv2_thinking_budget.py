# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Strict budget enforcement and MRv2 request/row lifecycle on CPU."""

from types import SimpleNamespace as NS

import numpy as np
import pytest
import torch
from vllm import SamplingParams

from omni_npu.worker.npu.thinking_budget import MRv2ThinkingBudget
from omni_npu.vllm_patches.patches.common import patch_mrv2_thinking_budget as patch


def make_tracker(end=(9,), start=(8,)):
    config = NS(reasoning_start_token_ids=list(start), reasoning_end_token_ids=list(end))
    return MRv2ThinkingBudget(config, torch.device("cpu"))


def request(budget, prompt=(8,), outputs=()):
    return NS(sampling_params=SamplingParams(thinking_token_budget=budget),
              prompt_token_ids=list(prompt), prefill_token_ids=list(prompt)+list(outputs))


def history(rows):
    raw = torch.zeros(len(rows), 128, dtype=torch.int32)
    for i, tokens in enumerate(rows):
        raw[i, :len(tokens)] = torch.tensor(tokens, dtype=torch.int32)
    return NS(all_token_ids=NS(gpu=raw),
              total_len=NS(gpu=torch.tensor([len(t) for t in rows], dtype=torch.int32)))


def commit(states, slot, token):
    pos = int(states.total_len.gpu[slot])
    states.all_token_ids.gpu[slot, pos] = token
    states.total_len.gpu[slot] += 1


def sample(tracker, states, slots):
    logits = torch.zeros(len(slots), 16)
    logits[:, 3] = 2e9  # Stronger than the native end-token boost.
    tracker.apply(logits, states, np.array(slots))
    return logits.argmax(-1).tolist(), logits


@pytest.mark.parametrize("budget", [0, 1, 2, 5])
def test_exact_budget_then_end_then_answer(budget):
    tracker = make_tracker()
    tracker.add_request(0, request(budget))
    states = history([[8]])
    emitted = []
    for _ in range(budget+2):
        tokens, logits = sample(tracker, states, [0])
        emitted.extend(tokens)
        commit(states, 0, tokens[0])
    assert emitted == [3]*budget + [9, 3]


def test_multi_token_start_and_end_sequences():
    tracker = make_tracker(start=(7, 8), end=(10, 11))
    tracker.add_request(0, request(2, prompt=(1,)))
    states = history([[1]])
    for token in [7, 8, 3, 3]:
        commit(states, 0, token)
    emitted = []
    for _ in range(3):
        tokens, _ = sample(tracker, states, [0])
        emitted.extend(tokens)
        commit(states, 0, tokens[0])
    assert emitted == [10, 11, 3]


def test_natural_end_preserves_answer_logits():
    tracker = make_tracker()
    tracker.add_request(0, request(5))
    states = history([[8, 3, 9]])
    tokens, logits = sample(tracker, states, [0])
    assert tokens == [3]
    assert torch.isfinite(logits).all()


def test_reordered_mixed_batch_and_unbudgeted_slot():
    tracker = make_tracker()
    tracker.add_request(0, request(0))
    tracker.add_request(2, request(2))
    states = history([[8], [8], [8]])
    tokens, logits = sample(tracker, states, [2, 1, 0])
    assert tokens == [3, 3, 9]
    assert torch.isfinite(logits[:2]).all()
    for idx, token in zip([2, 1, 0], tokens):
        commit(states, idx, token)
    tokens, _ = sample(tracker, states, [0, 2, 1])
    assert tokens == [3, 3, 3]


def test_prefill_discard_does_not_spend_budget_or_end_sequence():
    tracker = make_tracker(end=(10, 11))
    tracker.add_request(0, request(0))
    states = history([[8]])
    for _ in range(4):
        # Non-final chunks do not commit their sampled token to RequestState.
        assert sample(tracker, states, [0])[0] == [10]
    commit(states, 0, 10)
    assert sample(tracker, states, [0])[0] == [11]


def test_resumption_counts_committed_output_and_prompt_thinking():
    tracker = make_tracker()
    tracker.add_request(0, request(3, prompt=(8, 3), outputs=(3, 3)))
    states = history([[8, 3, 3, 3]])
    assert sample(tracker, states, [0])[0] == [9]


def test_slot_reuse_and_removal_clear_budget():
    tracker = make_tracker()
    tracker.add_request(0, request(0))
    tracker.remove_request(0)
    states = history([[8]])
    assert sample(tracker, states, [0])[0] == [3]
    tracker.add_request(0, request(0))
    tracker.add_request(0, request(None))
    assert sample(tracker, states, [0])[0] == [3]


def test_forced_end_can_be_sampled_with_top_k_top_p():
    from vllm.v1.sample.ops.topk_topp_sampler import apply_top_k_top_p_pytorch
    tracker = make_tracker()
    tracker.add_request(0, request(0))
    _, logits = sample(tracker, history([[8]]), [0])
    filtered = apply_top_k_top_p_pytorch(logits, torch.tensor([1]), torch.tensor([0.1]))
    assert filtered.argmax(-1).tolist() == [9]
    assert filtered.softmax(-1)[0, 9].item() == 1


def processor(v2=True, speculative=None, enabled=True):
    return NS(use_v2_model_runner=v2, speculative_config=speculative,
              vllm_config=NS(reasoning_config=NS(enabled=enabled,
                  reasoning_start_token_ids=[8], reasoning_end_token_ids=[9])))


def test_validation_preserves_budget_and_other_native_checks(monkeypatch):
    p = processor()
    params = SamplingParams(thinking_token_budget=2)
    def validate(self, actual, tasks):
        assert self is not p and not self.use_v2_model_runner
        assert actual is params and actual.thinking_token_budget == 2
        raise ValueError("unrelated native check")
    monkeypatch.setattr(patch, "_validate_params", validate)
    with pytest.raises(ValueError, match="unrelated native check"):
        patch.MRv2ThinkingBudgetValidationPatch._validate_params(p, params, ("generate",))
    assert p.use_v2_model_runner


def test_native_validation_accepts_v2_budget_but_requires_reasoning(monkeypatch):
    monkeypatch.setattr(SamplingParams, "verify", lambda *args: None)
    p = processor()
    p.model_config = p.structured_outputs_config = p.tokenizer = None
    params = SamplingParams(thinking_token_budget=2)
    patch.MRv2ThinkingBudgetValidationPatch._validate_params(p, params, ("generate",))
    assert params.thinking_token_budget == 2 and p.use_v2_model_runner
    p.vllm_config.reasoning_config.enabled = False
    with pytest.raises(ValueError, match="reasoning_config is not configured"):
        patch.MRv2ThinkingBudgetValidationPatch._validate_params(p, params, ("generate",))


def test_validation_rejects_synthetic_acceptance_budgets():
    with pytest.raises(ValueError, match="standard rejection"):
        patch.MRv2ThinkingBudgetValidationPatch._validate_params(
            processor(speculative=NS(rejection_sample_method="synthetic")),
            SamplingParams(thinking_token_budget=2),
            ("generate",))


def test_validation_accepts_mtp7_standard(monkeypatch):
    p = processor(speculative=NS(rejection_sample_method="standard",
                                num_speculative_tokens=7))
    monkeypatch.setattr(patch, "_validate_params", lambda self, params, tasks: params)
    params = SamplingParams(thinking_token_budget=2)
    assert patch.MRv2ThinkingBudgetValidationPatch._validate_params(
        p, params, ("generate",)) is params


def test_validation_rejects_missing_end_tokens():
    p = processor()
    p.vllm_config.reasoning_config.reasoning_end_token_ids = []
    with pytest.raises(ValueError, match="start/end token IDs"):
        patch.MRv2ThinkingBudgetValidationPatch._validate_params(
            p, SamplingParams(thinking_token_budget=2), ("generate",))


@pytest.mark.parametrize("v2,budget", [(False, 2), (True, None)])
def test_unaffected_requests_use_native_validator(monkeypatch, v2, budget):
    p = processor(v2=v2)
    monkeypatch.setattr(patch, "_validate_params", lambda self, params, tasks: self)
    assert patch.MRv2ThinkingBudgetValidationPatch._validate_params(
        p, SamplingParams(thinking_token_budget=budget), ("generate",)) is p


def test_sampler_applies_budget_after_other_processors(monkeypatch):
    tracker = make_tracker()
    tracker.add_request(0, request(0))
    sampler = NS(thinking_budget=tracker, req_states=history([[8]]))
    def native(self, logits, *args, **kwargs):
        result = logits.clone()
        result[:, 9] = float("-inf")  # Simulate an allowed-token mask.
        return result
    monkeypatch.setattr(patch, "_apply_sampling_params", native)
    logits = torch.zeros(1, 16)
    result = patch.MRv2ThinkingBudgetSamplerPatch.apply_sampling_params(
        sampler, logits, None, np.array([0]), None, None, None)
    assert result.argmax(-1).tolist() == [9]
    assert torch.isfinite(logits).all()


def expanded_sample(tracker, states, slots, draft_inputs):
    logits = torch.zeros(len(slots), 16)
    logits[:, 3] = 2e9
    tracker.apply(logits, states, np.array(list(dict.fromkeys(slots))),
                  torch.tensor(slots), torch.tensor(draft_inputs))
    return logits.argmax(-1).tolist(), logits


@pytest.mark.parametrize("budget", range(11))
def test_mtp7_forces_correct_candidate_or_bonus_position(budget):
    tracker = make_tracker()
    tracker.add_request(0, request(budget))
    tokens, logits = expanded_sample(tracker, history([[8]]), [0]*8, [8]+[3]*7)
    if budget < 8:
        assert tokens[:budget] == [3]*budget
        assert tokens[budget] == 9
        assert logits[budget].softmax(-1)[9].item() == 1
    else:
        assert tokens == [3]*8
    assert tracker.requests[0][1] == []  # No speculative token committed.


@pytest.mark.parametrize("end,drafts,expected", [
    ((9,), [8, 3, 3, 9, 3, 3, 3, 3], [3, 3, 9, 3, 3, 3, 3, 3]),
    ((10, 11), [8, 3, 3, 10, 11, 3, 3, 3], [3, 3, 10, 11, 3, 3, 3, 3]),
])
def test_accepted_end_marker_allows_answer_in_same_mtp_round(end, drafts, expected):
    tracker = make_tracker(end=end)
    tracker.add_request(0, request(2))
    assert expanded_sample(tracker, history([[8]]), [0]*8, drafts)[0] == expected


def test_rejection_does_not_commit_candidate_budget_or_end_progress():
    tracker = make_tracker(end=(10, 11))
    tracker.add_request(0, request(2))
    states = history([[8]])
    expanded_sample(tracker, states, [0]*8, [8, 3, 3, 10, 11, 3, 3, 3])
    # Reject the first draft and commit only a resampled thinking token.
    commit(states, 0, 4)
    assert sample(tracker, states, [0])[0] == [3]
    commit(states, 0, 3)
    assert sample(tracker, states, [0])[0] == [10]
    # A discarded hypothetical closing sequence cannot advance end_count.
    expanded_sample(tracker, states, [0]*4, [3, 10, 11, 3])
    assert sample(tracker, states, [0])[0] == [10]
    commit(states, 0, 10)
    assert sample(tracker, states, [0])[0] == [11]


def test_mtp_start_marker_inside_candidate_prefix():
    tracker = make_tracker(start=(7, 8))
    tracker.add_request(0, request(2, prompt=(1,)))
    tokens, _ = expanded_sample(tracker, history([[1]]), [0]*8,
                                [1, 7, 8, 3, 3, 9, 3, 3])
    assert tokens[4] == 9
    assert tokens[5:] == [3]*3


def test_mixed_reordered_variable_length_mtp_rows():
    tracker = make_tracker()
    tracker.add_request(0, request(0))
    tracker.add_request(2, request(2))
    states = history([[8], [8], [8]])
    slots = [2]*8 + [1] + [0]*4
    inputs = [8, 3, 3, 9, 3, 3, 3, 3] + [8] + [8, 9, 3, 3]
    assert expanded_sample(tracker, states, slots, inputs)[0] == (
        [3, 3, 9, 3, 3, 3, 3, 3] + [3] + [9, 3, 3, 3])


def test_runner_tracks_after_sampler_created_and_removes_slot(monkeypatch):
    from vllm.v1.worker.gpu.model_runner import GPUModelRunner
    from omni_npu.worker.npu.model_runner import NPUModelRunnerV2

    monkeypatch.setattr(GPUModelRunner, "add_requests", lambda *args: None)
    monkeypatch.setattr(GPUModelRunner, "_remove_request", lambda *args: True)
    runner = object.__new__(NPUModelRunnerV2)
    runner.vllm_config = NS(reasoning_config=make_tracker().reasoning_config)
    runner.device = torch.device("cpu")
    runner.req_states = NS(req_id_to_index={"r": 0})
    runner.sampler = NS()  # Upstream constructs this in load_model, after init.
    data = request(0)
    data.req_id = "r"
    runner.add_requests(NS(scheduled_new_reqs=[data]))
    assert 0 in runner.sampler.thinking_budget.requests
    assert runner._remove_request("r") is True
    assert not runner.sampler.thinking_budget.requests
