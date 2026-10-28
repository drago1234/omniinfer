# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# SPDX-License-Identifier: Apache-2.0
"""Budget-only H2D removal must retain the vLLM 0.25.1 state contract."""
from copy import deepcopy
from types import SimpleNamespace as NS

import pytest
import torch
from omni_npu.vllm_patches.patches.common import patch_thinking_ban as patch
from vllm.v1.sample.logits_processor.builtin import MinTokensLogitsProcessor
from vllm.v1.sample.logits_processor.interface import BatchUpdate
from vllm.v1.sample.rejection_sampler import RejectionSampler
from vllm.v1.sample.sampler import Sampler
from vllm.v1.sample.thinking_budget_state import ThinkingBudgetStateHolder

START, END, TOOL, VOCAB = 10, 11, 12, 32


@pytest.fixture(autouse=True)
def cpu_processor_device():
    # These tests exercise CPU processor logic, not NPU kernels. Do not
    # inherit a default NPU device left by other modules in a combined run.
    with torch.device("cpu"):
        yield


class Processor:
    combine = staticmethod(Sampler._combine_outputs_with_spec_tokens)

    def __init__(self):
        self.combined = []
        self.penalty_indices = []

    def _combine_outputs_with_spec_tokens(self, outputs, drafts):
        combined = self.combine(outputs, drafts)
        self.combined.append(deepcopy(combined))
        return combined

    def apply_penalties(self, logits, metadata, *args):
        # Penalty math is outside this change; observe the actual row mapping.
        if len(args) == 3:
            self.penalty_indices.append(args[1].tolist())
        return logits


class RejectionProcessor(Processor):
    combine = staticmethod(RejectionSampler._combine_outputs_with_spec_tokens)


def holder(budgets, num_spec=3, end_ids=None):
    result = ThinkingBudgetStateHolder(
        NS(reasoning_start_token_ids=[START], reasoning_end_token_ids=end_ids or [END]),
        len(budgets), num_spec, torch.device("cpu"), False,
    )
    result.sync_batch(BatchUpdate(
        batch_size=len(budgets), removed=[], moved=[],
        added=[(i, NS(thinking_token_budget=b), [START], []) for i, b in enumerate(budgets)],
    ))
    return result


def metadata(budget_holder, outputs, drafts, **overrides):
    fields = dict(thinking_budget_state_holder=budget_holder, thinking_ban_state_holder=None,
                  output_token_ids=outputs, spec_token_ids=drafts, no_penalties=True,
                  bad_words_token_ids={}, allowed_token_ids_mask=None,
                  logitsprocs=NS(non_argmax_invariant=[]))
    fields.update(overrides)
    return NS(**fields)


def bonus(m):
    return patch.SamplerPatch.apply_logits_processors(
        Processor(), torch.zeros(len(m.output_token_ids), VOCAB), m, True,
    )


def target(m, processor=None):
    return patch.RejectionSamplerPatch.apply_logits_processors(
        processor or RejectionProcessor(), torch.zeros(sum(map(len, m.spec_token_ids)), VOCAB),
        m, NS(num_draft_tokens=list(map(len, m.spec_token_ids))),
    )


def test_budget_only_skips_index_transfer_and_keeps_history_combine(monkeypatch):
    outputs, drafts = [[7, 8]], [[2, 3, 4]]
    m = metadata(holder([262144]), outputs, drafts)
    bonus(m)
    before = deepcopy(m.thinking_budget_state_holder._state)
    p = RejectionProcessor()

    def unexpected_transfer(*args, **kwargs):
        pytest.fail("budget-only rejection must not transfer repeat indices")

    monkeypatch.setattr(torch.Tensor, "to", unexpected_transfer)
    result = target(m, p)
    assert not result.any()
    assert p.combined == [[[7, 8], [7, 8, 2], [7, 8, 2, 3]]]
    assert not p.penalty_indices
    assert m.thinking_budget_state_holder._state == before
    assert outputs == [[7, 8]] and drafts == [[2, 3, 4]]


@pytest.mark.parametrize("budget,forced_row,force_bonus", [
    (0, 0, False), (1, 0, False), (2, 1, False), (3, 2, False),
    (4, None, True), (65536, None, False), (262144, None, False),
])
def test_native_budget_forcing_remains_correct(budget, forced_row, force_bonus):
    m = metadata(holder([budget]), [[7]], [[2, 3, 4]])
    b = bonus(m)
    t = target(m)
    expected_b, expected_t = torch.zeros_like(b), torch.zeros_like(t)
    if force_bonus:
        expected_b[0, END] = 1e9
    if forced_row is not None:
        expected_t[forced_row, END] = 1e9
    torch.testing.assert_close(b, expected_b)
    torch.testing.assert_close(t, expected_t)


@pytest.mark.parametrize("allowed,penalties", [(True, False), (False, True), (True, True)])
def test_real_index_consumers_still_receive_expanded_rows(allowed, penalties):
    mask = torch.zeros(2, VOCAB, dtype=torch.bool) if allowed else None
    if allowed:
        mask[0, TOOL], mask[1, TOOL + 1] = True, True
    m = metadata(holder([262144, 262144]), [[], []], [[2, 3], [4, 5, 6]],
                 allowed_token_ids_mask=mask, no_penalties=not penalties)
    p = RejectionProcessor()
    out = target(m, p)
    # DeviceContext can re-enter a monkeypatched Tensor.to through
    # __torch_function__. Assert the consumer's indices, not wrapper calls.
    assert p.penalty_indices == [[0, 0, 1, 1, 1]]
    if allowed:
        assert torch.isneginf(out[:2, TOOL]).all()
        assert torch.isneginf(out[2:, TOOL + 1]).all()


@pytest.mark.parametrize("budget", [None, 0, 2, 4, 262144])
@pytest.mark.parametrize("draft_counts", [[3], [0, 3], [3, 1, 0]])
@pytest.mark.parametrize("end_ids", [[END], [END, 13]])
def test_logits_and_state_match_native_across_steps(budget, draft_counts, end_ids):
    budgets = [budget if i % 2 == 0 else None for i in range(len(draft_counts))]
    holders = [holder(budgets, end_ids=end_ids) for _ in range(2)]
    outputs = [[7] for _ in draft_counts]
    for step in range(5):
        drafts = [[2 + step % 2] * n for n in draft_counts]
        ms = [metadata(h, deepcopy(outputs), deepcopy(drafts)) for h in holders]
        bs = [bonus(m) for m in ms]
        torch.testing.assert_close(*bs)
        expected = patch._original_rejection_apply_logits_processors(
            RejectionProcessor(), torch.zeros(sum(draft_counts), VOCAB), ms[0],
            NS(num_draft_tokens=draft_counts),
        )
        torch.testing.assert_close(target(ms[1]), expected)
        assert holders[0]._state == holders[1]._state
        for row in outputs:
            row.append(end_ids[step % len(end_ids)] if step in (1, 2) else 7)


def test_bad_words_and_min_tokens_are_preserved():
    processors = []
    for _ in range(2):
        minimum = MinTokensLogitsProcessor(None, torch.device("cpu"), False)
        minimum.min_toks[0] = (10, [7], {END})
        processors.append(minimum)
    ms = [metadata(holder([262144]), [[7]], [[2, 3, 4]],
                   bad_words_token_ids={0: [[TOOL]]},
                   logitsprocs=NS(non_argmax_invariant=[p])) for p in processors]
    for m in ms:
        bonus(m)
    expected = patch._original_rejection_apply_logits_processors(
        RejectionProcessor(), torch.zeros(3, VOCAB), ms[0], NS(num_draft_tokens=[3]),
    )
    got = target(ms[1])
    torch.testing.assert_close(got, expected)
    assert torch.isneginf(got[:, TOOL]).all()
    assert torch.isneginf(got[:, END]).all()


def test_existing_thinking_ban_still_runs_after_budget():
    from omni_npu.v1.sample.thinking_ban_state import ThinkingBanStateHolder

    ban = ThinkingBanStateHolder(
        reasoning_config=NS(reasoning_start_token_ids=[START], reasoning_end_token_ids=[END],
                            ban_tool_start_in_thinking=True),
        tool_call_start_tid=TOOL, tool_call_end_tid=None, max_num_seqs=1, num_spec_tokens=3,
        device=torch.device("cpu"), is_pin_memory=False,
    )
    ban.sync_batch(BatchUpdate(batch_size=1, removed=[], moved=[],
                              added=[(0, NS(extra_args=None), [START], [])]))
    m = metadata(holder([262144]), [[7]], [[2, 3, 4]], thinking_ban_state_holder=ban)
    bonus(m)
    out = target(m)
    assert torch.isneginf(out[:, TOOL]).all()


def test_sampler_and_state_updater_keep_native_ownership():
    sampler_source = patch._original_sampler_apply_logits_processors.__code__.co_filename.replace("\\", "/")
    update_source = ThinkingBudgetStateHolder.update_state.__code__.co_filename.replace("\\", "/")
    assert sampler_source.endswith("vllm/v1/sample/sampler.py")
    assert update_source.endswith("vllm/v1/sample/thinking_budget_state.py")
