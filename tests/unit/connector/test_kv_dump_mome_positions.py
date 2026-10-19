# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.

from types import SimpleNamespace
from unittest.mock import patch

import pytest
import torch

from omni_npu.connector.kv_dump import (
    _cpu_ints,
    _get_mome_position_state,
    _mome_block_id,
    _update_mome_block_positions,
    maybe_note_mome_cache,
)

KV_DUMP = "omni_npu.connector.kv_dump"
pytestmark = pytest.mark.unit


def test_cpu_ints_and_block_id_edges():
    assert _cpu_ints(None) == []
    assert _cpu_ints(3) == [3]
    assert _cpu_ints(torch.tensor([1, 2]), 1) == [1]

    class _Listed:
        def tolist(self):
            return [4, 5]

    assert _cpu_ints(_Listed(), 1) == [4]
    row = torch.tensor([7])
    table = torch.tensor([[1, 2]])
    assert _mome_block_id(row, 0, 0) == 7
    assert _mome_block_id(row, 0, 1) is None
    assert _mome_block_id(table, 0, 1) == 2
    assert _mome_block_id(table, 0, 5) is None
    assert _mome_block_id(torch.zeros(1, 1, 1), 0, 0) is None


def test_update_mome_block_positions_three_ranges():
    assert _update_mome_block_positions(
        [], 2, 0, 2, 10, 4, 2, 3,
    ) == [8, 9, 10]
    assert _update_mome_block_positions(
        [1, 2, 3, 4], 1, 0, 2, 10, 4, 2, 3, position_shift=1,
    ) == [1, 2, 8, 9]
    assert _update_mome_block_positions(
        [1], 0, 1, 2, 10, 4, 2, 3,
    ) == [1]


def test_position_state_drops_requests_outside_registry():
    runner = SimpleNamespace(
        _mome_cache_positions={"live": {}, "dead": {}},
        requests={"live": object()},
        input_batch=SimpleNamespace(req_ids=["live", "dead"]),
    )
    state = _get_mome_position_state(runner)
    assert set(state) == {"live"}

    runner = SimpleNamespace(
        _mome_cache_positions={"a": {}},
        input_batch=SimpleNamespace(req_ids=["a", None]),
    )
    assert set(_get_mome_position_state(runner)) == {"a"}


def _runner(req_ids, prompts, spec=0, kv_transfer=None, window=3, requests=None):
    runner = SimpleNamespace(
        input_batch=SimpleNamespace(
            req_ids=req_ids,
            num_prompt_tokens=torch.tensor(prompts, dtype=torch.int32),
        ),
        num_spec_tokens=spec,
        vllm_config=SimpleNamespace(
            model_config=SimpleNamespace(
                hf_config=SimpleNamespace(router_sliding_window=window),
            ),
            kv_transfer_config=kv_transfer,
        ),
    )
    if requests is not None:
        runner.requests = requests
    return runner


def _meta(
    *,
    n=1,
    computed=128,
    qlen=1,
    accepted=1,
    block_idx=0,
    first=0,
    last=0,
    cache=None,
    num_decodes=None,
    block_size=128,
    pad_slot_id=-1,
    query_start=None,
):
    if cache is None:
        cache = torch.tensor([[5, 6, 7, 8]] * n)
    if isinstance(computed, int):
        computed = [computed] * n
    if isinstance(accepted, int):
        accepted = [accepted] * n
    if isinstance(block_idx, int):
        block_idx = [block_idx] * n
    if isinstance(first, int):
        first = [first] * n
    if isinstance(last, int):
        last = [last] * n
    if query_start is None:
        query_start = [0]
        for length in [qlen] * n:
            query_start.append(query_start[-1] + length)
    return SimpleNamespace(
        num_accepted_tokens=torch.tensor(accepted, dtype=torch.int32),
        cache_indices=cache,
        num_decodes=n if num_decodes is None else num_decodes,
        B_size=block_size,
        query_start_loc=torch.tensor(query_start, dtype=torch.int32),
        num_computed_tokens=torch.tensor(computed, dtype=torch.int32),
        block_idx_last_computed_token=torch.tensor(block_idx, dtype=torch.int32),
        block_idx_first_scheduled_token=torch.tensor(first, dtype=torch.int32),
        block_idx_last_scheduled_token=torch.tensor(last, dtype=torch.int32),
        pad_slot_id=pad_slot_id,
    )


def _note(runner, metadata, **kwargs):
    context = SimpleNamespace(attn_metadata=metadata)
    with (
        patch(f"{KV_DUMP}._kv_dump_path", "/tmp/kv_dump"),
        patch("vllm.forward_context.get_forward_context", return_value=context),
    ):
        maybe_note_mome_cache(runner, **kwargs)


def test_note_returns_when_dump_disabled_or_runner_missing():
    with patch(f"{KV_DUMP}._kv_dump_path", ""):
        maybe_note_mome_cache(object())
    with patch(f"{KV_DUMP}._kv_dump_path", "/tmp/kv_dump"):
        maybe_note_mome_cache(None)


def test_note_skips_unsupported_metadata_and_empty_batch():
    runner = _runner(["r0"], [128])
    _note(runner, 1)
    assert runner._mome_cache_positions == {}

    empty = _runner([], [])
    _note(empty, {"layer": _meta()})
    assert empty._mome_cache_positions == {}
    assert runner._mome_cache_positions == {}


def test_note_reports_when_metadata_yields_no_rows():
    runner = _runner(["r0"], [128])
    for metadata in (
        {"layer": SimpleNamespace()},
        {"layer": _meta(num_decodes=0)},
        {"layer": _meta(query_start=[0])},
        {"layer": _meta(qlen=0)},
        {"layer": _meta(cache=torch.tensor([[0, 1]]))},
        {"layer": _meta(cache=torch.tensor([[7]]), pad_slot_id=7)},
        [1, {"layer": _meta(num_decodes=0)}],
    ):
        _note(runner, metadata)
    assert runner._mome_cache_positions == {}


def test_note_seeds_decode_and_flags_later_mismatch():
    runner = _runner(["r0"], [128])
    meta = _meta()
    _note(runner, {"layer0": meta, "layer1": meta})
    tracked = runner._mome_cache_positions["r0"]
    assert tracked["non_mtp:layer0"][0]["positions"] == [128, 129]
    assert "non_mtp:layer1" in tracked

    _note(runner, {"layer0": meta})
    blocks = runner._mome_cache_positions["r0"]["non_mtp:layer0"]
    assert blocks[0]["block_id"] == 5
    assert blocks[0]["positions"] == [128, 129]


def test_note_first_decode_recompute_matches_prompt_tail():
    runner = _runner(["r0"], [128], spec=2)
    _note(runner, {"layer": _meta(computed=100, block_idx=0)})
    blocks = runner._mome_cache_positions["r0"]["non_mtp:layer"]
    assert blocks[0]["block_idx"] == 0
    assert blocks[0]["positions"] == [98, 99, 100, 101]


def test_note_untracked_remap_and_draft_step0():
    runner = _runner(["r0"], [128])
    _note(runner, {"layer": _meta(computed=200, block_idx=1)})
    blocks = runner._mome_cache_positions["r0"]["non_mtp:layer"]
    assert [block["block_idx"] for block in blocks] == [0]
    assert blocks[0]["block_id"] == 5

    runner = _runner(["r0"], [128])
    _note(runner, {"layer": _meta(cache=torch.tensor([[5, 6]]))})
    _note(runner, {"layer": _meta(cache=torch.tensor([[9, 6]]), last=0)})
    blocks = runner._mome_cache_positions["r0"]["non_mtp:layer"]
    assert blocks[-1]["block_id"] == 9
    assert all(block["block_id"] != 5 for block in blocks)

    runner = _runner(["r0"], [128], spec=7)
    _note(
        runner,
        {"layer": _meta(qlen=8, accepted=2, computed=128)},
        draft=True,
        draft_head=0,
        draft_step=0,
    )
    blocks = runner._mome_cache_positions["r0"]["mtp_head_0:layer"]
    assert blocks[0]["positions"][0] == 129
    assert len(blocks[0]["positions"]) == 9


def test_note_ndim1_draft_shift_and_microbatch():
    runner = _runner(["r0"], [128], spec=0, kv_transfer=object())
    _note(runner, {"layer": _meta(cache=torch.tensor([5]))})
    blocks = runner._mome_cache_positions["r0"]["non_mtp:layer"]
    assert blocks[0]["block_id"] == 5
    assert blocks[0]["positions"] == [127, 128, 129]

    runner = _runner(["a", "b"], [128, 128], spec=1)
    micro = [
        "skip",
        {"layer": _meta()},
        {"layer": _meta(computed=128)},
    ]
    _note(runner, micro, draft=True, draft_head=1, draft_step=1)
    assert "mtp_head_1:layer" in runner._mome_cache_positions["a"]
    assert "mtp_head_1:layer" in runner._mome_cache_positions["b"]


def test_note_skips_null_request_and_swallows_checker_errors():
    runner = _runner([None], [128])
    _note(runner, {"layer": _meta()})
    assert runner._mome_cache_positions == {}

    boom = _runner(["r0"], [128])
    with (
        patch(f"{KV_DUMP}._kv_dump_path", "/tmp/kv_dump"),
        patch(
            "vllm.forward_context.get_forward_context",
            side_effect=RuntimeError("boom"),
        ),
    ):
        maybe_note_mome_cache(boom)
    assert boom._mome_cache_positions == {}


def test_note_updates_crossed_block_and_keeps_two():
    runner = _runner(["r0"], [128], spec=0)
    _note(runner, {"layer": _meta(cache=torch.tensor([[5, 6, 7, 8]]), last=0)})
    _note(runner, {"layer": _meta(
        cache=torch.tensor([[5, 6, 7, 8]]), first=0, last=1,
    )})
    _note(runner, {"layer": _meta(
        cache=torch.tensor([[9, 6, 7, 8]]), first=0, last=3,
    )})
    blocks = runner._mome_cache_positions["r0"]["non_mtp:layer"]
    assert len(blocks) <= 2
    assert blocks[-1]["block_idx"] == 3
