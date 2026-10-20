# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Tests for omni_npu.worker.npu.thinking_budget (V2 thinking budget).

Two tiers:

* State-management tests run on CPU: enablement gating, budget staging,
  marker-cache resets, and the no-budget fast path.
* Kernel tests (ported from upstream vllm#46727
  ``tests/v1/worker/test_gpu_thinking_budget.py``) need a real accelerator
  and are skipped when NPU is unavailable.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest
import torch

from omni_npu.worker.npu.thinking_budget import NPUThinkingBudgetState

START = [100]
NATURAL_END = [200]
FORCED_END = [201]

VOCAB = 256


def _npu_available() -> bool:
    try:
        import torch_npu  # noqa: F401
    except ImportError:
        return False
    return hasattr(torch, "npu") and torch.npu.is_available()


requires_npu = pytest.mark.skipif(not _npu_available(), reason="NPU is unavailable")


@pytest.fixture(autouse=True)
def npu_uva(monkeypatch):
    """Install the production buffer adapters without loading every patch."""
    if not _npu_available():
        return
    from vllm.v1.worker.gpu import buffer_utils
    from omni_npu.worker.npu.buffer_utils import (
        NPUUvaBuffer,
        copy_to_uva_with_slot_event,
    )

    monkeypatch.setattr(buffer_utils, "UvaBuffer", NPUUvaBuffer)
    monkeypatch.setattr(
        buffer_utils.UvaBufferPool, "copy_to_uva", copy_to_uva_with_slot_event
    )


@pytest.fixture
def cpu_uva(monkeypatch):
    """Exercise the real UvaBackedTensor/pool with CPU storage in state tests."""
    from vllm.v1.worker.gpu import buffer_utils

    class CPUBuffer:
        def __init__(self, size, dtype):
            self.cpu = torch.zeros(size, dtype=dtype)
            self.np = self.cpu.numpy()
            self.uva = self.cpu

        def wait_for_host_reuse(self):
            pass

    monkeypatch.setattr(buffer_utils, "UvaBuffer", CPUBuffer)


def _make_config(start=START, end=FORCED_END, natural=NATURAL_END):
    """Reasoning-config stand-in; omit ``natural`` to emulate the native
    (non-superset) 0.25.1 ReasoningConfig that predates the field."""
    cfg = SimpleNamespace()
    cfg.reasoning_start_token_ids = start
    cfg.reasoning_end_token_ids = end
    if natural is not None:
        cfg.natural_reasoning_end_token_ids = natural
    return cfg


def _make_req_states(num_reqs=4, max_model_len=64, device=None, token_rows=None,
                     total_lens=None):
    device = device or torch.device("cpu")
    all_tokens = torch.zeros(num_reqs, max_model_len, dtype=torch.int32,
                             device=device)
    for i, row in enumerate(token_rows or []):
        all_tokens[i, : len(row)] = torch.tensor(row, dtype=torch.int32)
    total_len = torch.zeros(num_reqs, dtype=torch.int32, device=device)
    for i, length in enumerate(total_lens or []):
        total_len[i] = length
    return SimpleNamespace(
        max_num_reqs=num_reqs,
        max_model_len=max_model_len,
        device=device,
        all_token_ids=SimpleNamespace(gpu=all_tokens),
        total_len=SimpleNamespace(gpu=total_len),
    )


def _params(budget):
    return SimpleNamespace(thinking_token_budget=budget)


def _run_apply(state, logits, req_rows):
    """Drive state.apply for the given (req_idx, local_pos) logits rows."""
    expanded_idx = torch.tensor([r for r, _ in req_rows], dtype=torch.int32,
                                 device=state.device)
    local_pos = torch.tensor([p for _, p in req_rows], dtype=torch.int32,
                             device=state.device)
    idx_mapping_np = np.array([r for r, _ in req_rows])
    input_ids = torch.zeros(len(req_rows), dtype=torch.int32, device=state.device)
    state.apply(logits, expanded_idx, idx_mapping_np, input_ids, local_pos)


# ---------------------------------------------------------------------------
# Enablement gating (CPU)
# ---------------------------------------------------------------------------


def test_disabled_without_reasoning_config():
    state = NPUThinkingBudgetState(_make_req_states(), None)
    assert not state.enabled


def test_disabled_when_natural_end_missing():
    state = NPUThinkingBudgetState(_make_req_states(), _make_config(natural=None))
    assert not state.enabled


@pytest.mark.usefixtures("cpu_uva")
def test_enabled_initial_state():
    state = NPUThinkingBudgetState(_make_req_states(), _make_config())
    assert state.enabled
    assert state.thinking_token_budget.np.tolist() == [-1] * 4
    assert state.use_thinking_budget.tolist() == [False] * 4
    assert state.cached_last_start.tolist() == [-1] * 4
    assert state.cached_last_end.tolist() == [-1] * 4
    assert state.cached_scan_pos.tolist() == [0] * 4
    assert state.reasoning_start_token_ids.tolist() == START
    assert state.natural_reasoning_end_token_ids.tolist() == NATURAL_END
    assert state.reasoning_end_token_ids.tolist() == FORCED_END


# ---------------------------------------------------------------------------
# Budget staging (CPU)
# ---------------------------------------------------------------------------


@pytest.mark.usefixtures("cpu_uva")
def test_add_request_with_budget():
    state = NPUThinkingBudgetState(_make_req_states(), _make_config())
    state.add_request(2, _params(128))
    assert state.use_thinking_budget[2]
    assert state.thinking_token_budget.np[2] == 128
    assert state._reset_reqs == [2]
    assert state._budget_dirty


@pytest.mark.usefixtures("cpu_uva")
def test_add_request_without_budget():
    state = NPUThinkingBudgetState(_make_req_states(), _make_config())
    state.add_request(1, _params(None))
    assert not state.use_thinking_budget[1]
    assert state.thinking_token_budget.np[1] == -1
    assert state._reset_reqs == []
    assert not state._budget_dirty


@pytest.mark.usefixtures("cpu_uva")
def test_add_request_clamps_oversized_budget():
    state = NPUThinkingBudgetState(_make_req_states(), _make_config())
    state.add_request(0, _params(np.iinfo(np.int64).max))
    assert state.thinking_token_budget.np[0] == np.iinfo(np.int32).max


@pytest.mark.usefixtures("cpu_uva")
def test_apply_staged_writes_resets_and_syncs():
    state = NPUThinkingBudgetState(_make_req_states(), _make_config())
    # Dirty the marker caches: req 0 stays dirty, req 2 is reset.
    state.cached_last_start.fill_(42)
    state.cached_last_end.fill_(42)
    state.cached_scan_pos.fill_(42)
    state.add_request(2, _params(7))
    state.apply_staged_writes()
    assert state.cached_last_start[0] == 42
    assert state.cached_last_start[2] == -1
    assert state.cached_last_end[2] == -1
    assert state.cached_scan_pos[2] == 0
    assert state._reset_reqs == []
    assert not state._budget_dirty
    assert state.thinking_token_budget.gpu[2] == 7
    assert state.thinking_token_budget.gpu[0] == -1


@pytest.mark.usefixtures("cpu_uva")
def test_apply_noop_without_budget_requests(monkeypatch):
    import omni_npu.worker.npu.thinking_budget as tb

    state = NPUThinkingBudgetState(_make_req_states(), _make_config())
    called = []
    monkeypatch.setattr(
        tb, "apply_thinking_budget", lambda *a, **k: called.append(a)
    )
    logits = torch.zeros(1, VOCAB)
    _run_apply(state, logits, [(0, 0)])
    assert called == []


@pytest.mark.usefixtures("cpu_uva")
def test_budget_staging_preserves_previous_step_and_clears_reused_slot():
    state = NPUThinkingBudgetState(_make_req_states(), _make_config())
    state.add_request(0, _params(7))
    state.apply_staged_writes()
    previous_step = state.thinking_token_budget.gpu

    state.add_request(0, _params(None))
    assert previous_step[0] == 7
    state.apply_staged_writes()
    assert previous_step[0] == 7
    assert state.thinking_token_budget.gpu[0] == -1
    assert not state.use_thinking_budget[0]


# ---------------------------------------------------------------------------
# Kernel semantics (NPU) — ported from upstream vllm#46727 tests
# ---------------------------------------------------------------------------


@requires_npu
def test_kernel_forces_end_after_budget_reached():
    # Committed: start marker at pos 3, four reasoning tokens after it.
    tokens = [1, 2, 3, 100, 5, 6, 7, 8]
    state = NPUThinkingBudgetState(
        _make_req_states(token_rows=[tokens], total_lens=[len(tokens)],
                         device=torch.device("npu")),
        _make_config(),
    )
    state.add_request(0, _params(4))
    state.apply_staged_writes()

    logits = torch.zeros(1, VOCAB, dtype=torch.float32, device=state.device)
    _run_apply(state, logits, [(0, 0)])
    assert logits[0, FORCED_END[0]] == 1.0e9
    others = logits.clone()
    others[0, FORCED_END[0]] = 0.0
    assert others.abs().max() == 0.0


@requires_npu
def test_kernel_allows_tokens_before_budget():
    tokens = [1, 2, 3, 100, 5, 6, 7, 8]
    state = NPUThinkingBudgetState(
        _make_req_states(token_rows=[tokens], total_lens=[len(tokens)],
                         device=torch.device("npu")),
        _make_config(),
    )
    state.add_request(0, _params(100))
    state.apply_staged_writes()

    logits = torch.zeros(1, VOCAB, dtype=torch.float32, device=state.device)
    _run_apply(state, logits, [(0, 0)])
    assert logits.abs().max() == 0.0


@requires_npu
def test_kernel_stops_after_natural_end():
    # Natural end marker (200) at pos 3 is later than the start marker.
    tokens = [1, 100, 5, 200, 8]
    state = NPUThinkingBudgetState(
        _make_req_states(token_rows=[tokens], total_lens=[len(tokens)],
                         device=torch.device("npu")),
        _make_config(),
    )
    state.add_request(0, _params(0))
    state.apply_staged_writes()

    logits = torch.zeros(1, VOCAB, dtype=torch.float32, device=state.device)
    _run_apply(state, logits, [(0, 0)])
    assert logits.abs().max() == 0.0


@requires_npu
def test_kernel_uses_latest_start_marker():
    # start, natural end, then a second start: the request is thinking again.
    tokens = [100, 1, 200, 100, 5]
    state = NPUThinkingBudgetState(
        _make_req_states(token_rows=[tokens], total_lens=[len(tokens)],
                         device=torch.device("npu")),
        _make_config(),
    )
    state.add_request(0, _params(0))
    state.apply_staged_writes()

    logits = torch.zeros(1, VOCAB, dtype=torch.float32, device=state.device)
    _run_apply(state, logits, [(0, 0)])
    assert logits[0, FORCED_END[0]] == 1.0e9


@requires_npu
def test_kernel_continues_multi_token_end_marker():
    # Forced sequence is [201, 202]; the tail already ends with 201.
    forced = [201, 202]
    tokens = [1, 2, 100, 201]
    state = NPUThinkingBudgetState(
        _make_req_states(token_rows=[tokens], total_lens=[len(tokens)],
                         device=torch.device("npu")),
        _make_config(end=forced),
    )
    state.add_request(0, _params(1))
    state.apply_staged_writes()

    logits = torch.zeros(1, VOCAB, dtype=torch.float32, device=state.device)
    _run_apply(state, logits, [(0, 0)])
    assert logits[0, 202] == 1.0e9
    assert logits[0, 201] == 0.0


@requires_npu
@pytest.mark.parametrize(
    "length,marker_pos",
    [(1024, 0), (1025, 0), (2048, 1023), (2049, 1024), (4097, 1)],
)
def test_kernel_cold_scan_finds_prompt_marker(length, marker_pos):
    # Multi-token starts also straddle the backward scan's block boundary.
    tokens = [10] * length
    tokens[marker_pos:marker_pos + 2] = [100, 101]
    state = NPUThinkingBudgetState(
        _make_req_states(max_model_len=length + 32, token_rows=[tokens],
                         total_lens=[length], device=torch.device("npu")),
        _make_config(start=[100, 101]),
    )
    state.add_request(0, _params(0))
    state.apply_staged_writes()

    logits = torch.zeros(1, VOCAB, dtype=torch.float32, device=state.device)
    _run_apply(state, logits, [(0, 0)])
    assert logits[0, FORCED_END[0]] == 1.0e9
    assert state.cached_last_start[0] == marker_pos
    assert state.cached_scan_pos[0] == length - 1


@requires_npu
def test_kernel_incremental_scan_reuses_cursor():
    tokens = [1, 2, 3, 100, 5, 6, 7, 8]
    state = NPUThinkingBudgetState(
        _make_req_states(token_rows=[tokens], total_lens=[len(tokens)],
                         device=torch.device("npu")),
        _make_config(),
    )
    state.add_request(0, _params(0))
    state.apply_staged_writes()

    logits = torch.zeros(1, VOCAB, dtype=torch.float32, device=state.device)
    _run_apply(state, logits, [(0, 0)])
    assert logits[0, FORCED_END[0]] == 1.0e9
    first_cursor = int(state.cached_scan_pos[0])

    # Commit one more token; the hot path scans only the new token.
    state.req_states.all_token_ids.gpu[0, len(tokens)] = 9
    state.req_states.total_len.gpu[0] = len(tokens) + 1
    logits2 = torch.zeros(1, VOCAB, dtype=torch.float32, device=state.device)
    _run_apply(state, logits2, [(0, 0)])
    assert logits2[0, FORCED_END[0]] == 1.0e9
    assert int(state.cached_scan_pos[0]) == first_cursor + 1


@requires_npu
def test_kernel_spec_decode_expanded_rows():
    # One request, two logits rows (local_pos 0 and 1, e.g. MTP verify).
    tokens = [1, 2, 3, 100, 5, 6, 7, 8]
    state = NPUThinkingBudgetState(
        _make_req_states(token_rows=[tokens], total_lens=[len(tokens)],
                         device=torch.device("npu")),
        _make_config(),
    )
    state.add_request(0, _params(4))
    state.apply_staged_writes()

    logits = torch.zeros(2, VOCAB, dtype=torch.float32, device=state.device)
    expanded_idx = torch.tensor([0, 0], dtype=torch.int32, device=state.device)
    local_pos = torch.tensor([0, 1], dtype=torch.int32, device=state.device)
    input_ids = torch.tensor([8, 9], dtype=torch.int32, device=state.device)
    state.apply(logits, expanded_idx, np.array([0, 0]), input_ids, local_pos)
    # Both the committed row and the draft-prefix row are forced.
    assert logits[0, FORCED_END[0]] == 1.0e9
    assert logits[1, FORCED_END[0]] == 1.0e9


@requires_npu
@pytest.mark.parametrize("draft_token,expected_end", [(201, 202), (200, None)])
def test_kernel_spec_decode_end_marker_in_draft(draft_token, expected_end):
    tokens = [100, 5, 6, 7, 8]
    state = NPUThinkingBudgetState(
        _make_req_states(token_rows=[tokens], total_lens=[len(tokens)],
                         device=torch.device("npu")),
        _make_config(end=[201, 202]),
    )
    state.add_request(0, _params(4))
    state.apply_staged_writes()

    logits = torch.zeros(2, VOCAB, dtype=torch.float32, device=state.device)
    state.apply(
        logits,
        torch.tensor([0, 0], dtype=torch.int32, device=state.device),
        np.array([0]),
        torch.tensor([8, draft_token], dtype=torch.int32, device=state.device),
        torch.tensor([0, 1], dtype=torch.int32, device=state.device),
    )
    assert logits[0, 201] == 1.0e9
    if expected_end is None:
        assert logits[1].abs().max() == 0.0
    else:
        assert logits[1, expected_end] == 1.0e9
        assert logits[1, 201] == 0.0
    # A speculative natural end must not enter the committed marker cache.
    assert state.cached_last_end[0] == -1


@requires_npu
def test_kernel_mixed_batch_only_forces_budget_requests():
    tokens0 = [1, 2, 3, 100, 5, 6, 7, 8]
    tokens1 = [1, 2, 100, 6, 7]
    state = NPUThinkingBudgetState(
        _make_req_states(token_rows=[tokens0, tokens1],
                         total_lens=[len(tokens0), len(tokens1)],
                         device=torch.device("npu")),
        _make_config(),
    )
    state.add_request(0, _params(0))
    state.add_request(1, _params(None))
    state.apply_staged_writes()

    logits = torch.zeros(2, VOCAB, dtype=torch.float32, device=state.device)
    _run_apply(state, logits, [(0, 0), (1, 0)])
    assert logits[0, FORCED_END[0]] == 1.0e9
    assert logits[1].abs().max() == 0.0
    # No budget: the marker cache is untouched for request 1.
    assert state.cached_scan_pos[1] == 0


def test_disabled_state_makes_lifecycle_hooks_no_ops():
    """Missing natural-end markers (native 0.25.1 config) disable the state.

    Both sampler hooks early-return then, so a launch without the superset
    ReasoningConfig costs nothing and writes nothing.
    """
    state = NPUThinkingBudgetState(_make_req_states(), _make_config(natural=None))
    assert not state.enabled

    state.add_request(0, _params(8))
    state.apply_staged_writes()

    # The constructor bails out before allocating anything, so the hooks have
    # no state to touch and must simply return.
    assert not hasattr(state, "thinking_token_budget")
    assert not hasattr(state, "_budget_dirty")


# ---------------------------------------------------------------------------
# Source-level guards (no accelerator needed)
# ---------------------------------------------------------------------------


def test_kernels_use_no_boolean_operators():
    """Jitted code must combine conditions with ``&``/``|``, not ``and``/``or``.

    Triton's AST->TTIR pass lowers at most two boolean operands and raises
    ``UnsupportedLanguageConstruct: chained boolean operators (A or B or C) are
    not supported`` for anything longer. That surfaced as a fatal engine error
    on the first real request, so the rule is kept strict (no ``and``/``or`` in
    kernels at all) and enforced statically here.
    """
    import ast
    import inspect

    from omni_npu.worker.npu import thinking_budget as tb

    tree = ast.parse(inspect.getsource(tb))

    def _is_jit(fn: ast.AST) -> bool:
        for dec in getattr(fn, "decorator_list", ()):
            name = ast.unparse(dec)
            if name.endswith("triton.jit") or name.endswith(".jit"):
                return True
        return False

    kernels = [
        node
        for node in tree.body
        if isinstance(node, ast.FunctionDef) and _is_jit(node)
    ]
    assert kernels, "expected at least one @triton.jit kernel in this module"

    offenders = []
    for fn in kernels:
        for node in ast.walk(fn):
            if isinstance(node, ast.BoolOp):
                offenders.append((node.lineno, ast.unparse(node)))
    assert not offenders, (
        "kernels must use `&` / `|` on int1 scalars instead of `and` / `or`; "
        f"offending conditions at {offenders}"
    )
