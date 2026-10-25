# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU regressions for the vLLM #46662 MRv1 backport."""

from types import MethodType
from types import SimpleNamespace as NS

import pytest
import torch
from omni_npu.v1.sample.thinking_ban_state import ThinkingBanStateHolder
from omni_npu.v1.sample.thinking_budget import update_thinking_budget_state
from omni_npu.vllm_patches.patches.common.patch_thinking_ban import (
    RejectionSamplerPatch,
    SamplerPatch,
    ThinkingBudgetCommittedOutputsPatch,
)
from vllm.v1.sample.logits_processor.builtin import MinTokensLogitsProcessor
from vllm.v1.sample.logits_processor.interface import BatchUpdate, MoveDirectionality
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.sample.sampler import Sampler
from vllm.v1.sample.thinking_budget_state import ThinkingBudgetStateHolder

START, END, TOOL = 10, 11, 12
VOCAB = 32
_NATIVE_UPDATE = ThinkingBudgetStateHolder.update_state
_NATIVE_SAMPLER_PROCESSORS = Sampler.apply_logits_processors
_NATIVE_REJECTION_PROCESSORS = RejectionSampler.apply_logits_processors


class _Sampler:
    def __init__(self):
        self.penalty_calls = []

    _combine_outputs_with_spec_tokens = staticmethod(
        Sampler._combine_outputs_with_spec_tokens
    )

    def apply_penalties(self, logits, metadata, *args):
        self.penalty_calls.append(args)
        return logits


class _RejectionSampler(_Sampler):
    _combine_outputs_with_spec_tokens = staticmethod(
        RejectionSampler._combine_outputs_with_spec_tokens
    )


def _config(end_ids=None):
    return NS(
        reasoning_start_token_ids=[START], reasoning_end_token_ids=end_ids or [END]
    )


def _holder(budget, *, num_spec=3, end_ids=None, prompt=None):
    holder = ThinkingBudgetStateHolder(
        _config(end_ids), 4, num_spec, torch.device("cpu"), False
    )
    holder.update_state = MethodType(update_thinking_budget_state, holder)
    holder.sync_batch(
        BatchUpdate(
            batch_size=1,
            removed=[],
            moved=[],
            added=[
                (
                    0,
                    NS(thinking_token_budget=budget),
                    [START] if prompt is None else prompt,
                    [],
                )
            ],
        )
    )
    return holder


def _metadata(holder, outputs, drafts, **overrides):
    fields = {
        "thinking_budget_state_holder": holder,
        "thinking_ban_state_holder": None,
        "output_token_ids": outputs,
        "spec_token_ids": drafts,
        "no_penalties": True,
        "bad_words_token_ids": {},
        "allowed_token_ids_mask": None,
        "logitsprocs": NS(non_argmax_invariant=[]),
    }
    fields.update(overrides)
    return NS(**fields)


def _step(holder, outputs, drafts, **overrides):
    metadata = _metadata(holder, outputs, drafts, **overrides)
    sampler = _Sampler()
    bonus = SamplerPatch.apply_logits_processors(
        sampler, torch.zeros(len(outputs), VOCAB), metadata, bool(holder.in_spec_mode)
    )
    target = RejectionSamplerPatch.apply_logits_processors(
        _RejectionSampler(),
        torch.zeros(sum(map(len, drafts)), VOCAB),
        metadata,
        NS(num_draft_tokens=list(map(len, drafts))),
    )
    return bonus, target


def test_budget_only_does_not_build_host_indices_or_combine_history(monkeypatch):
    holder = _holder(65536)
    metadata = _metadata(holder, [[7] * 1000], [[2, 3, 4]])
    sampler = _Sampler()
    bonus, target = torch.zeros(1, VOCAB), torch.zeros(3, VOCAB)

    def unexpected(*args, **kwargs):
        pytest.fail("budget-only sampling constructed indices or copied history")

    monkeypatch.setattr(sampler, "_combine_outputs_with_spec_tokens", unexpected)
    monkeypatch.setattr(torch, "tensor", unexpected)
    monkeypatch.setattr(torch, "arange", unexpected)
    monkeypatch.setattr(torch.Tensor, "to", unexpected)
    SamplerPatch.apply_logits_processors(sampler, bonus, metadata, True)
    RejectionSamplerPatch.apply_logits_processors(
        sampler, target, metadata, NS(num_draft_tokens=[3])
    )
    assert holder._state[0]["output_tok_ids"] is metadata.output_token_ids[0]
    assert not bonus.any() and not target.any()


@pytest.mark.parametrize(
    "budget,row,bonus_forced",
    [
        (0, 0, False),
        (1, 0, False),
        (2, 1, False),
        (3, 2, False),
        (4, None, True),
        (65536, None, False),
    ],
)
def test_spec_budget_forces_correct_target_or_bonus(budget, row, bonus_forced):
    bonus, target = _step(_holder(budget), [[7]], [[2, 3, 4]])
    expected_target = torch.zeros_like(target)
    if row is not None:
        expected_target[row, END] = 1e9
    expected_bonus = torch.zeros_like(bonus)
    if bonus_forced:
        expected_bonus[0, END] = 1e9
    torch.testing.assert_close(target, expected_target)
    torch.testing.assert_close(bonus, expected_bonus)


@pytest.mark.parametrize(
    "budget,outputs,forced",
    [
        (0, [], True),
        (1, [7], True),
        (2, [7], False),
        (2, [7, 8], True),
    ],
)
def test_non_spec_budget_counts_committed_outputs(budget, outputs, forced):
    holder = _holder(budget, num_spec=0)
    metadata = _metadata(holder, [outputs], [[]])
    logits = SamplerPatch.apply_logits_processors(
        _Sampler(), torch.zeros(1, VOCAB), metadata, False
    )
    assert logits[0, END].item() == (1e9 if forced else 0)
    assert holder._state[0]["output_tok_ids"] is outputs


def test_rejected_forced_end_is_retried_and_multi_token_end_completes():
    holder = _holder(2, end_ids=[END, END + 2])
    # The forced END at target position 1 was rejected at position 0.
    _, target = _step(holder, [[7]], [[2, 3, 4]])
    assert target[1, END] == 1e9
    _, target = _step(holder, [[7, 8]], [[2, 3, 4]])
    assert target[0, END] == 1e9
    # Once END is accepted, continue the forced closing sequence.
    _, target = _step(holder, [[7, 8, END]], [[2, 3, 4]])
    assert target[0, END + 2] == 1e9
    bonus, target = _step(holder, [[7, 8, END, END + 2]], [[2, 3, 4]])
    assert not bonus.any() and not target.any()


def test_natural_end_does_not_force_more_tokens():
    bonus, target = _step(_holder(4), [[7, END, 9]], [[2, 3, 4]])
    assert not bonus.any() and not target.any()


@pytest.mark.parametrize("budget", [0, 1, 2, 4, 65536])
@pytest.mark.parametrize("num_spec", [0, 1, 3])
@pytest.mark.parametrize("end_ids", [[END], [END, END + 2]])
def test_logits_match_native_budget_across_decode_steps(budget, num_spec, end_ids):
    """Compare full logits with 0.25.1 through rejection and forced-end retries."""
    reference = _holder(budget, num_spec=num_spec, end_ids=end_ids)
    reference.update_state = MethodType(_NATIVE_UPDATE, reference)
    backport = _holder(budget, num_spec=num_spec, end_ids=end_ids)
    drafts = [[2, 3, 4][:num_spec]]
    traces = [[], [7], [7, 8], [7, 8, *end_ids], [7, 8, *end_ids, 9]]
    for outputs in traces:
        metadata = _metadata(reference, [outputs], drafts)
        expected_bonus = _NATIVE_SAMPLER_PROCESSORS(
            _Sampler(), torch.zeros(1, VOCAB), metadata, bool(num_spec)
        )
        expected_target = _NATIVE_REJECTION_PROCESSORS(
            _RejectionSampler(),
            torch.zeros(num_spec, VOCAB),
            metadata,
            NS(num_draft_tokens=[num_spec]),
        )
        bonus, target = _step(backport, [outputs], drafts)
        torch.testing.assert_close(bonus, expected_bonus)
        torch.testing.assert_close(target, expected_target)
        assert backport._state == reference._state


def test_mixed_budget_rows_follow_variable_draft_lengths():
    holder = _holder(2)
    holder.sync_batch(
        BatchUpdate(
            batch_size=3,
            removed=[],
            moved=[],
            added=[
                (1, NS(thinking_token_budget=None), [START], []),
                (2, NS(thinking_token_budget=1), [START], []),
            ],
        )
    )
    bonus, target = _step(holder, [[7], [7], [7]], [[2, 3], [], [4, 5, 6]])
    expected = torch.zeros_like(target)
    expected[1, END] = expected[2, END] = 1e9
    torch.testing.assert_close(target, expected)
    assert not bonus.any()


@pytest.mark.parametrize(
    "penalties,allowlist", [(True, False), (False, True), (True, True)]
)
def test_required_repeat_indices_and_penalty_history_are_preserved(
    penalties, allowlist
):
    sampler = _RejectionSampler()
    mask = torch.zeros(2, VOCAB, dtype=torch.bool) if allowlist else None
    if mask is not None:
        mask[0, 5], mask[1, 6] = True, True
    metadata = _metadata(
        None,
        [[7], [8]],
        [[2, 3], [4]],
        no_penalties=not penalties,
        allowed_token_ids_mask=mask,
    )
    target = RejectionSamplerPatch.apply_logits_processors(
        sampler, torch.zeros(3, VOCAB), metadata, NS(num_draft_tokens=[2, 1])
    )
    _, indices, history = sampler.penalty_calls[0]
    assert indices.tolist() == [0, 0, 1]
    assert history == ([[7], [7, 2], [8]] if penalties else [[7], [8]])
    if allowlist:
        assert target[0, 5] == target[1, 5] == target[2, 6] == -float("inf")


def test_budget_forcing_still_overrides_allowed_token_mask():
    mask = torch.zeros(1, VOCAB, dtype=torch.bool)
    mask[0, END] = True
    bonus, target = _step(_holder(2), [[7]], [[2, 3, 4]], allowed_token_ids_mask=mask)
    assert bonus[0, END] == -float("inf")
    assert target[1, END] == 1e9


def test_min_tokens_masks_are_preserved_except_for_forced_budget_end():
    processor = MinTokensLogitsProcessor(NS(), torch.device("cpu"), False)
    processor.update_state(
        BatchUpdate(
            batch_size=1,
            removed=[],
            moved=[],
            added=[(0, NS(min_tokens=20, all_stop_token_ids={END, 5}), [START], [7])],
        )
    )
    bonus, target = _step(
        _holder(2),
        [[7]],
        [[2, 3, 4]],
        logitsprocs=NS(non_argmax_invariant=[processor]),
    )
    assert bonus[0, END] == -float("inf")
    assert target[1, END] == 1e9
    assert torch.isneginf(target[:, 5]).all()


@pytest.mark.parametrize(
    "prompt,outputs,budget",
    [
        ([START, 7, 8], [9], 3),
        ([7], [START, 8, 9], 2),
    ],
)
def test_budget_counts_thinking_started_in_prompt_or_output(prompt, outputs, budget):
    holder = _holder(budget, prompt=prompt)
    _, target = _step(holder, [outputs], [[2, 3, 4]])
    assert target[0, END] == 1e9


def test_unbudgeted_request_does_not_force_logits():
    bonus, target = _step(_holder(None), [[7]], [[2, 3, 4]])
    assert not bonus.any() and not target.any()


def test_bad_words_still_see_bonus_and_target_draft_prefixes():
    sampler = _Sampler()
    metadata = _metadata(None, [[7]], [[2, 3]], bad_words_token_ids={0: [[7, 2, 3]]})
    bonus = SamplerPatch.apply_logits_processors(
        sampler, torch.zeros(1, VOCAB), metadata, True
    )
    # The full draft ends in 3, so bonus token 3 is allowed.
    assert bonus[0, 3] == 0
    target = RejectionSamplerPatch.apply_logits_processors(
        _RejectionSampler(),
        torch.zeros(2, VOCAB),
        metadata,
        NS(num_draft_tokens=[2]),
    )
    assert target[0, 3] == 0
    assert target[1, 3] == -float("inf")


def test_budget_and_thinking_ban_coexist():
    config = _config()
    config.ban_tool_start_in_thinking = True
    ban = ThinkingBanStateHolder(
        reasoning_config=config,
        tool_call_start_tid=TOOL,
        tool_call_end_tid=None,
        max_num_seqs=4,
        num_spec_tokens=3,
        device=torch.device("cpu"),
        is_pin_memory=False,
    )
    ban.sync_batch(
        BatchUpdate(
            batch_size=1,
            removed=[],
            moved=[],
            added=[(0, NS(thinking_token_budget=3), [START], [])],
        )
    )
    bonus, target = _step(_holder(3), [[7]], [[2, 3, 4]], thinking_ban_state_holder=ban)
    assert bonus[0, TOOL] == -float("inf")
    assert torch.isneginf(target[:, TOOL]).all()
    assert target[2, END] == 1e9


def test_request_move_and_removal_keep_budget_state_consistent():
    holder = _holder(2)
    holder.sync_batch(
        BatchUpdate(
            batch_size=2,
            removed=[],
            added=[],
            moved=[(0, 1, MoveDirectionality.UNIDIRECTIONAL)],
        )
    )
    bonus, target = _step(holder, [[], [7]], [[], [2, 3]])
    assert target[1, END] == 1e9 and not bonus.any()
    holder.sync_batch(BatchUpdate(batch_size=0, removed=[1], added=[], moved=[]))
    assert not holder.has_tracked_requests()


@pytest.mark.parametrize("patch_cls", [SamplerPatch, RejectionSamplerPatch])
def test_sampler_patch_installation_also_installs_matching_state_update(
    monkeypatch, patch_cls
):
    target = patch_cls._target
    for sampler_target in (SamplerPatch._target, RejectionSamplerPatch._target):
        monkeypatch.setattr(
            sampler_target, "_omni_npu_applied_patches", {}, raising=False
        )
        monkeypatch.setattr(
            sampler_target, "apply_logits_processors", lambda *args: None
        )
    monkeypatch.setattr(
        ThinkingBudgetStateHolder, "_omni_npu_applied_patches", {}, raising=False
    )
    monkeypatch.setattr(ThinkingBudgetStateHolder, "update_state", lambda *args: None)
    patch_cls.apply()
    assert ThinkingBudgetStateHolder.update_state is update_thinking_budget_state
    assert target.apply_logits_processors is patch_cls.apply_logits_processors
    assert (
        SamplerPatch._target.apply_logits_processors
        is SamplerPatch.apply_logits_processors
    )
    # Default ALL may reach the bonus sampler again after resolving dependencies.
    SamplerPatch.apply()
    ThinkingBudgetCommittedOutputsPatch.apply()
