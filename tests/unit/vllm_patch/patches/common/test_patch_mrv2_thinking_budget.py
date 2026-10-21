# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""MRv2 thinking-budget hooks: registration, forwarding order, state-absent no-ops.

``NPUThinkingBudgetState`` is attached by ``NPUModelRunnerV2`` only on the last
PP rank / non-pooling models, so every hook must survive a sampler without the
attribute -- that path is the difference between "MRv2 keeps working" and a
startup crash on ranks that never run sampling.
"""

from __future__ import annotations

import pytest

from omni_npu.vllm_patches.patches.common import (
    patch_mrv2_thinking_budget as patch_mod,
)
from unit.vllm_patch.patches.patch_test_utils import applied_patches

PATCH = patch_mod.MRv2ThinkingBudgetSamplerPatch


class _FakeState:
    """Records what the sampler forwards into the NPU state."""

    def __init__(self):
        self.calls = []

    def add_request(self, req_idx, sampling_params):
        self.calls.append(("add_request", req_idx, sampling_params))

    def apply_staged_writes(self):
        self.calls.append(("apply_staged_writes",))

    def apply(self, logits, expanded_idx_mapping, idx_mapping_np, input_ids,
              expanded_local_pos):
        self.calls.append(("apply", logits, len(idx_mapping_np)))
        return "ignored-return"  # the hook returns the upstream logits


class _FakeSampler:
    """Upstream side of the chain: the three methods being wrapped."""

    def __init__(self):
        self.calls = []

    def add_request(self, req_idx, prompt_len, sampling_params):
        self.calls.append(("up.add_request", req_idx, prompt_len))

    def apply_staged_writes(self):
        self.calls.append(("up.apply_staged_writes",))

    def apply_sampling_params(self, logits, expanded_idx_mapping, idx_mapping_np,
                              pos, input_ids, expanded_local_pos,
                              skip_top_k_top_p=False):
        self.calls.append(("up.apply_sampling_params", skip_top_k_top_p))
        return "upstream-logits"


@pytest.fixture
def chain(monkeypatch):
    """Point the wrappers at _FakeSampler instead of the real vllm Sampler."""
    monkeypatch.setattr(
        PATCH, "_upstream_add_request", _FakeSampler.add_request, raising=False
    )
    monkeypatch.setattr(
        PATCH, "_upstream_apply_staged_writes",
        _FakeSampler.apply_staged_writes, raising=False,
    )
    monkeypatch.setattr(
        PATCH, "_upstream_apply_sampling_params",
        _FakeSampler.apply_sampling_params, raising=False,
    )
    return PATCH


def test_patch_is_registered_on_the_v2_sampler():
    """All three lifecycle methods are hooked on the 0.25.1 Sampler."""
    import vllm.v1.worker.gpu.sample.sampler as up_sampler

    assert PATCH._target is up_sampler.Sampler
    assert PATCH._attr_names_to_apply == [
        "add_request",
        "apply_staged_writes",
        "apply_sampling_params",
    ]


def test_apply_installs_every_hook_and_keeps_the_chain():
    """apply() must bypass the conflict check yet still capture upstream."""
    with applied_patches([PATCH]):
        import vllm.v1.worker.gpu.sample.sampler as up_sampler

        for name in PATCH._attr_names_to_apply:
            assert getattr(up_sampler.Sampler, name) is PATCH.__dict__[name]
            assert hasattr(PATCH, f"_upstream_{name}")


def test_add_request_forwards_before_staging_the_budget(chain):
    """Upstream first: ``ReqState`` row must exist before the budget is staged."""
    sampler = _FakeSampler()
    sampler.thinking_budget_state = _FakeState()
    params = object()

    chain.add_request(sampler, 3, 17, params)

    assert sampler.calls == [("up.add_request", 3, 17)]
    state_calls = sampler.thinking_budget_state.calls
    assert state_calls == [("add_request", 3, params)]


def test_apply_staged_writes_syncs_the_state_after_upstream(chain):
    sampler = _FakeSampler()
    sampler.thinking_budget_state = _FakeState()

    chain.apply_staged_writes(sampler)

    assert sampler.calls == [("up.apply_staged_writes",)]
    assert sampler.thinking_budget_state.calls == [("apply_staged_writes",)]


def test_apply_sampling_params_returns_upstream_logits_and_calls_apply(chain):
    """The state writes into the upstream FP32 copy; its return value is unused."""
    sampler = _FakeSampler()
    sampler.thinking_budget_state = _FakeState()
    rows = [0, 1]

    out = chain.apply_sampling_params(
        sampler, "logits", "expanded", [7, 7], "pos", "input_ids", "local_pos"
    )

    assert out == "upstream-logits"
    assert sampler.calls == [("up.apply_sampling_params", False)]
    assert sampler.thinking_budget_state.calls == [("apply", "upstream-logits", 2)]


def test_apply_sampling_params_passes_the_skip_flag_through(chain):
    """``sample()`` sets skip_top_k_top_p=True; the flag must reach upstream."""
    sampler = _FakeSampler()

    chain.apply_sampling_params(
        sampler, "logits", "expanded", [0], "pos", "input_ids", "local_pos",
        skip_top_k_top_p=True,
    )

    assert sampler.calls == [("up.apply_sampling_params", True)]


def test_hooks_no_op_without_a_state(chain):
    """Pooling models / non-last PP ranks have no state: forward and return."""
    sampler = _FakeSampler()

    chain.add_request(sampler, 0, 5, object())
    chain.apply_staged_writes(sampler)
    out = chain.apply_sampling_params(
        sampler, "logits", "expanded", [0], "pos", "input_ids", "local_pos"
    )

    assert out == "upstream-logits"
    assert sampler.calls == [
        ("up.add_request", 0, 5),
        ("up.apply_staged_writes",),
        ("up.apply_sampling_params", False),
    ]
    assert not hasattr(sampler, "thinking_budget_state")
