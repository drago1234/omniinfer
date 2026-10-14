# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""CPU-only tests for the sharded DDR->NPU load planner and executor.

The executor runs on CPU tensors: H2D is a memmove on the same pointers the NPU
path hands to swap_blocks_batch, and the all-gather is a thread barrier, so the
byte layout, the slot reuse and the scatter are exercised exactly.
"""

import ctypes
import sys
import threading
import types

import numpy as np
import pytest
import torch

from omni_npu.v1.kv_offload.cpu import sharded_load as sl

FACTOR = 2
WIDTHS = [96, 96, 64]     # NPU row bytes of each KV tensor
PAGES = [96, 80, 64]      # tensor 1 only fills a row prefix
NUM_NPU = 64
NUM_CPU = 40
MIXED_PAGE = 72           # second page size inside tensor 0, like pangu hybrid groups


def _build_case(world, seed=0, mixed=False):
    rng = np.random.default_rng(seed)
    cpu = [torch.from_numpy(rng.integers(-128, 128, size=(NUM_CPU, w * FACTOR), dtype=np.int8))
           for w in WIDTHS]
    parts = {k: [] for k in ("t", "page", "cpu", "npu", "src")}
    for t, (w, p) in enumerate(zip(WIDTHS, PAGES)):
        if t == 2:
            cblocks = np.arange(0, NUM_CPU, world)[:7]      # every block owned by rank 0
        else:
            cblocks = rng.choice(NUM_CPU, size=19, replace=False)
        cb = np.repeat(cblocks, FACTOR)
        subs = np.tile(np.arange(FACTOR), cblocks.size)
        nb = rng.choice(NUM_NPU, size=cb.size, replace=False)
        pg = np.full(cb.size, p)
        if mixed and t == 0:
            pg[cb.size // 2 // FACTOR * FACTOR:] = MIXED_PAGE
        parts["t"].append(np.full(cb.size, t))
        parts["page"].append(pg)
        parts["cpu"].append(cb)
        parts["npu"].append(nb)
        parts["src"].append(cpu[t].data_ptr() + cb * cpu[t].stride(0) + subs * pg)
    case = {k: np.concatenate(v).astype(np.int64) for k, v in parts.items()}
    case["cpu_tensors"] = cpu
    return case


def _args(case):
    return case["t"], case["page"], case["cpu"], case["npu"], case["src"]


def _reference(case, init):
    out = [x.clone() for x in init]
    cpu = case["cpu_tensors"]
    for i in range(case["t"].size):
        t, cb, nb, p = (int(case[k][i]) for k in ("t", "cpu", "npu", "page"))
        off = int(case["src"][i]) - cpu[t].data_ptr() - cb * cpu[t].stride(0)
        out[t][nb, :p] = cpu[t][cb, off:off + p]
    return out


def _memmove_h2d(srcs, dsts, sizes):
    for s, d, n in zip(srcs.tolist(), dsts.tolist(), sizes.tolist()):
        ctypes.memmove(int(d), int(s), int(n))


class _Work:
    def wait(self):
        return None


class _FakeGroup:
    """All-gather across threads. It gathers when issued, the harshest order for
    the slot ring: chunk k+1 overwrites its receive slot before chunk k is
    scattered, so a slot mix-up corrupts the result."""

    def __init__(self, world):
        self.barrier = threading.Barrier(world)
        self.inputs = [None] * world
        self.calls = [0] * world

    def allgather(self, rank):
        def fn(out, inp):
            self.calls[rank] += 1
            self.inputs[rank] = inp.clone()
            self.barrier.wait()
            out.copy_(torch.cat(self.inputs))
            self.barrier.wait()
            return _Work()
        return fn


def _run_sharded(case, world, chunk_bytes, warm=True, seed=1):
    rng = np.random.default_rng(seed)
    init = [[torch.from_numpy(rng.integers(-128, 128, size=(NUM_NPU, w), dtype=np.int8))
             for w in WIDTHS] for _ in range(world)]
    npu = [[x.clone() for x in init[r]] for r in range(world)]
    plan = sl.plan_sharded_load(*_args(case), world, chunk_bytes)
    group = _FakeGroup(world)
    errors = []

    def worker(r):
        try:
            ex = sl.ShardedLoadExecutor(r, world, chunk_bytes, 0, torch.device("cpu"),
                                        group.allgather(r))
            if warm:
                ex.warmup()
            ex.run(plan, case["src"], case["npu"], npu[r], _memmove_h2d)
        except BaseException as exc:  # surface thread failures in the main thread
            errors.append(exc)
            group.barrier.abort()

    threads = [threading.Thread(target=worker, args=(r,)) for r in range(world)]
    for th in threads:
        th.start()
    for th in threads:
        th.join()
    assert not errors, errors
    assert group.calls == [(1 if warm else 0) + len(plan.chunks)] * world
    for r in range(world):
        want = _reference(case, init[r])
        for t in range(len(WIDTHS)):
            assert torch.equal(npu[r][t], want[t]), f"rank {r} tensor {t} differs"
    return plan


def _budgets(world):
    # One row per chunk, a few rows per chunk, everything in one chunk.
    return (world * max(PAGES), world * 700, 1 << 20)


@pytest.mark.parametrize("world", [2, 4, 16])
def test_roundtrip_matches_plain_load(world):
    case = _build_case(world)
    for chunk_bytes in _budgets(world):
        _run_sharded(case, world, chunk_bytes)


@pytest.mark.parametrize("world", [2, 4, 16])
def test_mixed_pages_in_one_tensor(world):
    case = _build_case(world, mixed=True)
    assert np.unique(case["page"][case["t"] == 0]).size == 2
    for chunk_bytes in _budgets(world):
        plan = _run_sharded(case, world, chunk_bytes)
        pages = {pc.page_size for c in plan.chunks for pc in c.pieces if pc.tensor_idx == 0}
        assert pages == {PAGES[0], MIXED_PAGE}


def test_run_warms_up_on_first_use():
    _run_sharded(_build_case(4), 4, 4 * 700, warm=False)


def test_exact_duplicate_rows_are_copied_once():
    world = 4
    case = _build_case(world, mixed=True)
    n = case["t"].size
    dup = np.arange(0, n, 5)
    for k in ("t", "page", "cpu", "npu", "src"):
        case[k] = np.concatenate([case[k], case[k][dup]])
    for chunk_bytes in _budgets(world):
        plan = _run_sharded(case, world, chunk_bytes)
        used = np.concatenate([pc.slot_desc.reshape(-1) for c in plan.chunks for pc in c.pieces])
        assert used.max() < n


def test_every_rank_uploads_exactly_the_blocks_it_owns():
    world = 4
    case = _build_case(world, mixed=True)
    n = case["t"].size
    owner = case["cpu"] % world
    for chunk_bytes in _budgets(world):
        plan = sl.plan_sharded_load(*_args(case), world, chunk_bytes)
        pieces = [pc for c in plan.chunks for pc in c.pieces]
        assert set(np.concatenate([pc.slot_desc.reshape(-1) for pc in pieces]).tolist()) \
            == set(range(n))
        for r in range(world):
            sent = np.concatenate([pc.slot_desc[r] for pc in pieces])
            assert set(sent[owner[sent] == r].tolist()) == set(np.flatnonzero(owner == r).tolist())
        for pc in pieces:
            same = (case["t"][pc.slot_desc] == pc.tensor_idx) & (case["page"][pc.slot_desc] == pc.page_size)
            assert same.all()


def test_chunks_fit_one_rank_share():
    world = 4
    case = _build_case(world)
    for chunk_bytes in _budgets(world):
        plan = sl.plan_sharded_load(*_args(case), world, chunk_bytes)
        for chunk in plan.chunks:
            assert chunk.send_bytes <= chunk_bytes // world
            offset = 0
            for pc in chunk.pieces:
                assert pc.send_offset == offset
                offset += pc.num_slots * pc.page_size
            assert offset == chunk.send_bytes


def test_rank_owning_nothing_in_a_tensor_only_pads():
    world = 4
    case = _build_case(world)
    plan = sl.plan_sharded_load(*_args(case), world, 1 << 20)
    owner = case["cpu"] % world
    for pc in (p for c in plan.chunks for p in c.pieces if p.tensor_idx == 2):
        assert (owner[pc.slot_desc[1:]] == 0).all()


def test_plan_refuses_unshardable_jobs():
    case = _build_case(4)
    assert sl.plan_sharded_load(*_args(case), 1, 1 << 20) is None
    empty = np.empty(0, dtype=np.int64)
    assert sl.plan_sharded_load(empty, empty, empty, empty, empty, 4, 1 << 20) is None

    conflict = dict(case)
    conflict["npu"] = case["npu"].copy()
    conflict["npu"][1] = conflict["npu"][0]    # same row, different source bytes
    why = []
    assert sl.plan_sharded_load(*_args(conflict), 4, 1 << 20, why) is None
    assert why == ["1 rows are written from different sources"]

    why = []
    assert sl.plan_sharded_load(*_args(case), 4, 4 * max(PAGES) - 1, why) is None
    assert why and "larger than one rank's share" in why[0]

    with pytest.raises(ValueError, match="same length"):
        sl.plan_sharded_load(case["t"], case["page"][:-1], case["cpu"], case["npu"],
                             case["src"], 4, 1 << 20)


def test_device_footprint_adds_hccl_scratch_and_one_chunk_of_workspace():
    chunk = 256 << 20
    assert sl.device_footprint_bytes(chunk, 16, 320 << 20) == (
        sl.staging_bytes(chunk, 16) + (320 << 20) + chunk
    )


def test_staging_bytes_matches_the_buffers_warmup_allocates():
    ex = sl.ShardedLoadExecutor(0, 16, 256 << 20, 48, torch.device("cpu"), lambda out, inp: _Work())
    ex.allocate()
    assert ex._send.numel() + ex._recv.numel() == sl.staging_bytes(256 << 20, 16)
    assert sl.staging_bytes(256 << 20, 16) == 544 << 20


def test_allocate_does_not_gather_and_probe_does():
    calls = []
    ex = sl.ShardedLoadExecutor(0, 4, 4 * 1000, 0, torch.device("cpu"),
                                lambda out, inp: calls.append(1) or _Work())
    ex.allocate()
    assert calls == []
    ex.probe()
    assert calls == [1]


def test_warmup_allocates_two_slots_and_gathers_once():
    calls = []

    def allgather(out, inp):
        calls.append((out.numel(), inp.numel()))
        return _Work()

    ex = sl.ShardedLoadExecutor(0, 4, 4 * 1000, 48, torch.device("cpu"), allgather)
    ex.warmup()
    slot = sl._align(1000)
    assert ex._send.numel() == 2 * slot
    assert ex._recv.numel() == 2 * slot * 4
    assert calls == [(4, 1)]
    assert ex.min_blocks == 48


def test_int32_view_falls_back_when_the_layout_does_not_divide():
    rng = np.random.default_rng(3)
    dst = torch.from_numpy(rng.integers(-128, 128, size=(8, 10), dtype=np.int8))
    region = torch.from_numpy(rng.integers(-128, 128, size=(2, 3, 6), dtype=np.int8))
    idx = torch.tensor([[0, 2, 4], [1, 3, 5]])
    want = dst.clone()
    want[idx.reshape(-1), :6] = region.reshape(-1, 6)
    assert sl._as_int32(region) is None
    sl.scatter_rows(dst, idx, region, 6)
    assert torch.equal(dst, want)


def test_int32_view_applies_on_the_production_layout():
    # Page sizes of pangu 505B at TP16. Every region the load scatters must take
    # the int32 view, or the scatter silently degrades to int8 on the card.
    world, rows = 16, 4096
    pages = (147456, 188416, 153600)
    rng = np.random.default_rng(5)
    t = np.repeat(np.arange(6), 300)
    page = np.array([pages[i % 3] for i in t])
    cpu = rng.integers(0, 100000, size=t.size)
    npu = np.concatenate([rng.permutation(rows)[:300] for _ in range(6)])
    src = np.arange(t.size) * (1 << 20)
    chunk_bytes = 256 << 20
    plan = sl.plan_sharded_load(t, page, cpu, npu, src, world, chunk_bytes)
    ex = sl.ShardedLoadExecutor(0, world, chunk_bytes, 0, torch.device("meta"), None)
    recv = torch.empty(ex._slot * 2 * world, dtype=torch.int8, device="meta")
    dst = torch.empty((8, max(pages)), dtype=torch.int8, device="meta")
    assert sl._as_int32(dst) is not None
    for k, chunk in enumerate(plan.chunks):
        ro = (k % 2) * ex._slot * world
        by_rank = recv[ro:ro + chunk.send_bytes * world].view(world, chunk.send_bytes)
        for pc in chunk.pieces:
            region = by_rank[:, pc.send_offset:pc.send_offset + pc.num_slots * pc.page_size]
            region = region.view(world, pc.num_slots, pc.page_size)
            assert sl._as_int32(region) is not None
            assert sl._as_int32(dst[:, :pc.page_size]) is not None


def test_executor_plan_uses_its_world_and_chunk_size():
    case = _build_case(4)
    ex = sl.ShardedLoadExecutor(0, 4, 4 * 700, 0, torch.device("cpu"), None)
    got = ex.plan(*_args(case))
    want = sl.plan_sharded_load(*_args(case), 4, 4 * 700)
    assert [c.send_bytes for c in got.chunks] == [c.send_bytes for c in want.chunks]
    for a, b in zip(got.chunks, want.chunks):
        for x, y in zip(a.pieces, b.pieces):
            assert (x.tensor_idx, x.page_size, x.send_offset) == (y.tensor_idx, y.page_size, y.send_offset)
            assert np.array_equal(x.slot_desc, y.slot_desc)


def test_scatter_rows_on_device_calls_npu_scatter_nd_update(monkeypatch):
    calls = []
    fake = types.SimpleNamespace(
        npu_scatter_nd_update_=lambda target, idx, region: calls.append((target, idx, region)))
    monkeypatch.setitem(sys.modules, "torch_npu", fake)
    dst = torch.empty((8, 16), dtype=torch.int8, device="meta")
    region = torch.empty((2, 3, 12), dtype=torch.int8, device="meta")
    idx = torch.empty((2, 3), dtype=torch.int64, device="meta")
    sl.scatter_rows(dst, idx, region, 12)
    ((target, got_idx, got_region),) = calls
    assert (target.dtype, tuple(target.shape)) == (torch.int32, (8, 3))
    assert (got_region.dtype, tuple(got_region.shape)) == (torch.int32, (2, 3, 3))
    assert tuple(got_idx.shape) == (2, 3, 1)


def test_warmup_on_device_synchronizes(monkeypatch):
    synced = []
    monkeypatch.setattr(torch.npu, "synchronize", lambda: synced.append(True), raising=False)
    ex = sl.ShardedLoadExecutor(0, 2, 2 * 1024, 0, torch.device("meta"), lambda out, inp: _Work())
    ex.warmup()
    assert synced == [True]
