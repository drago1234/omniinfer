# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Sharded DDR -> NPU load for replicated KV layouts.

Under the replicated layout every TP rank holds the same KV bytes, so the
plain load has each rank copy the whole job from the shared host pool and the
same bytes cross the host-to-device path tp times. Here rank r copies only the
CPU blocks with ``cpu_block % tp == r`` into a contiguous send buffer, the send
buffers are all-gathered over a dedicated HCCL group, and every rank scatters
the gathered rows into its KV cache.

The planner only looks at block ids and page sizes, so every rank derives the
same chunks and buffer layout without talking to the others.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import NamedTuple

import numpy as np
import torch

# Two send and two receive slots let chunk k+1 be copied and gathered while
# chunk k is still being gathered and scattered; see ShardedLoadExecutor.run.
_NUM_SLOTS = 2
# Slot sizes are aligned so that int32 views of the receive buffer line up.
_SLOT_ALIGN = 512


class ShardMeta(NamedTuple):
    """Per-descriptor block ids that ``plan_sharded_load`` needs.

    The descriptor arrays the plain load builds carry addresses and sizes only.
    These three are filled alongside them, in the same order.
    """

    tensor_idx: np.ndarray
    cpu_block: np.ndarray
    npu_block: np.ndarray


@dataclass
class ShardedPiece:
    """Rows of one (KV tensor, page size) pair inside one chunk."""

    tensor_idx: int
    page_size: int
    # Byte offset of the piece inside every rank's send buffer.
    send_offset: int
    # (world, slots) descriptor behind every slot. Ranks own different numbers
    # of rows, so the shorter ranks pad with a real descriptor of the piece: the
    # padding uploads that same host row and writes the same bytes to the same
    # KV row a second time.
    slot_desc: np.ndarray

    @property
    def num_slots(self) -> int:
        return int(self.slot_desc.shape[1])


@dataclass
class ShardedChunk:
    """One all-gather: every rank sends ``send_bytes`` and receives world times that."""

    send_bytes: int
    pieces: list[ShardedPiece] = field(default_factory=list)


@dataclass
class ShardedPlan:
    chunks: list[ShardedChunk]


def _align(n: int) -> int:
    return (int(n) + _SLOT_ALIGN - 1) // _SLOT_ALIGN * _SLOT_ALIGN


def _owner_matrix(idx: np.ndarray, owner: np.ndarray, world: int) -> np.ndarray:
    """(world, max rows per rank) descriptor indices, -1 padded.

    Row r holds rank r's descriptors in their original order, so any column
    range of it is exactly what rank r sends for those slots.
    """
    order = np.argsort(owner, kind="stable")
    ranks = owner[order]
    counts = np.bincount(ranks, minlength=world)
    starts = np.concatenate(([0], np.cumsum(counts)[:-1]))
    cols = np.arange(order.size) - starts[ranks]
    mat = np.full((world, int(counts.max())), -1, dtype=np.int64)
    mat[ranks, cols] = idx[order]
    return mat


def plan_sharded_load(
    tensor_idx,
    page_size,
    cpu_block,
    npu_block,
    src_ptr,
    world: int,
    chunk_bytes: int,
    why: list | None = None,
) -> ShardedPlan | None:
    """Split one load job into all-gather chunks, or return None for the plain load.

    Descriptor i copies ``page_size[i]`` bytes from host address ``src_ptr[i]``,
    inside CPU block ``cpu_block[i]``, to the start of row ``npu_block[i]`` of KV
    tensor ``tensor_idx[i]``. ``chunk_bytes`` bounds what one all-gather
    receives. ``why`` collects the reason a job is refused.
    """
    tensor_idx = np.asarray(tensor_idx, dtype=np.int64)
    page_size = np.asarray(page_size, dtype=np.int64)
    cpu_block = np.asarray(cpu_block, dtype=np.int64)
    npu_block = np.asarray(npu_block, dtype=np.int64)
    src_ptr = np.asarray(src_ptr, dtype=np.int64)
    n = int(tensor_idx.size)
    # world<=1 is refused at executor build time so the HCCL domain is never
    # paid for; this guard is for direct callers of the pure function.
    if world <= 1 or n == 0:
        return None
    if not (page_size.size == cpu_block.size == npu_block.size == src_ptr.size == n):
        raise ValueError("descriptor arrays must have the same length")

    # Layers that share one KV cache repeat a descriptor verbatim; one copy is
    # enough. Different sources for one row would make the bytes depend on the
    # write order, which the scatter does not preserve.
    span = int(npu_block.max()) + 1
    row_key = tensor_idx * span + npu_block
    _, first = np.unique(row_key, return_index=True)
    keep = np.arange(n, dtype=np.int64)
    if first.size != n:
        exact = np.unique(np.stack([row_key, page_size, src_ptr], axis=1), axis=0).shape[0]
        if exact != first.size:
            if why is not None:
                why.append("%d rows are written from different sources" % (exact - first.size))
            return None
        keep = np.sort(first)

    per_rank = int(chunk_bytes) // world
    if int(page_size[keep].max()) > per_rank:
        if why is not None:
            why.append("a page is larger than one rank's share of a chunk")
        return None

    owner = np.mod(cpu_block, world)
    # One piece has one slot size, so rows are grouped by (tensor, page size):
    # hybrid models let KV groups with different pages share one tensor.
    kt, kp = tensor_idx[keep], page_size[keep]
    base = int(kp.max()) + 1
    code = kt * base + kp
    order = np.argsort(code, kind="stable")
    keys, starts = np.unique(code[order], return_index=True)

    chunks: list[ShardedChunk] = []
    cur = ShardedChunk(send_bytes=0)
    for key, idx in zip(keys.tolist(), np.split(keep[order], starts[1:])):
        t, page = divmod(int(key), base)
        mat = _owner_matrix(idx, owner[idx], world)
        k = 0
        while k < mat.shape[1]:
            take = min((per_rank - cur.send_bytes) // page, mat.shape[1] - k)
            if take == 0:
                chunks.append(cur)
                cur = ShardedChunk(send_bytes=0)
                continue
            sub = mat[:, k:k + take]
            valid = sub >= 0
            pad = int(sub[valid][0])
            cur.pieces.append(ShardedPiece(t, page, cur.send_bytes, np.where(valid, sub, pad)))
            cur.send_bytes += take * page
            k += take
    if cur.pieces:
        chunks.append(cur)
    return ShardedPlan(chunks)


def staging_bytes(chunk_bytes: int, world: int) -> int:
    """Device bytes the staging buffers take: two send slots plus two receive slots."""
    return _align(max(int(chunk_bytes) // world, 1)) * _NUM_SLOTS * (1 + world)


def device_footprint_bytes(chunk_bytes: int, world: int, hccl_buffer_bytes: int) -> int:
    """Device bytes the sharded load holds once it has run.

    Staging, the group's HCCL scratch, and the scatter's workspace: the scatter
    gets a view strided across ranks and copies it contiguous first, up to one
    chunk.
    """
    return staging_bytes(chunk_bytes, world) + int(hccl_buffer_bytes) + int(chunk_bytes)


def _as_int32(t: torch.Tensor) -> torch.Tensor | None:
    """``t`` (int8) read as int32 elements, or None when its layout does not divide by 4."""
    row_aligned = t.shape[-1] % 4 == 0 and t.storage_offset() % 4 == 0
    strides_aligned = t.stride(-1) == 1 and all(s % 4 == 0 for s in t.stride()[:-1])
    if not (row_aligned and strides_aligned):
        return None
    return t.view(torch.int32)


def scatter_rows(dst: torch.Tensor, idx: torch.Tensor, region: torch.Tensor, page: int) -> None:
    """``dst[idx[r, s], :page] = region[r, s]`` for every rank r and slot s.

    ``region`` is the (world, slots, page) view of the all-gather output and
    ``idx`` the (world, slots) destination rows. The bytes move as int32 when
    the layout allows it, which the NPU scatters faster than int8.
    """
    target = dst if page == dst.shape[1] else dst[:, :page]
    target32, region32 = _as_int32(target), _as_int32(region)
    if target32 is not None and region32 is not None:
        target, region = target32, region32
    if dst.device.type == "cpu":
        target[idx.reshape(-1)] = region.reshape(-1, region.shape[-1])
        return
    import torch_npu

    torch_npu.npu_scatter_nd_update_(target, idx.unsqueeze(-1), region)


class ShardedLoadExecutor:
    """Runs sharded load plans for one TP rank on the caller's current stream.

    ``allgather_async(out, inp)`` must behave like
    ``torch.distributed.all_gather_into_tensor(out, inp, group=..., async_op=True)``.
    Jobs with fewer than ``min_blocks`` CPU blocks are left to the plain load.
    """

    def __init__(self, rank: int, world: int, chunk_bytes: int, min_blocks: int,
                 device: torch.device, allgather_async):
        self.rank = int(rank)
        self.world = int(world)
        self.chunk_bytes = int(chunk_bytes)
        self.min_blocks = int(min_blocks)
        self.device = device
        self._allgather = allgather_async
        self._slot = _align(max(self.chunk_bytes // self.world, 1))
        self._send: torch.Tensor | None = None
        self._recv: torch.Tensor | None = None

    def plan(self, tensor_idx, page_size, cpu_block, npu_block, src_ptr,
             why: list | None = None) -> ShardedPlan | None:
        return plan_sharded_load(tensor_idx, page_size, cpu_block, npu_block, src_ptr,
                                 self.world, self.chunk_bytes, why)

    def allocate(self) -> None:
        """Allocate the staging buffers. Raises when the device memory does not fit.

        Local only: callers agree on success across ranks before calling
        ``probe``, which is collective.
        """
        self._send = torch.empty(self._slot * _NUM_SLOTS, dtype=torch.int8, device=self.device)
        self._recv = torch.empty(self._slot * _NUM_SLOTS * self.world, dtype=torch.int8,
                                 device=self.device)

    def probe(self) -> None:
        """Build the HCCL communicator now instead of inside the first load.

        One tiny all-gather; every rank of the group has to call it.
        """
        self._allgather(self._recv[:self.world], self._send[:1]).wait()
        if self.device.type != "cpu":
            torch.npu.synchronize()

    def warmup(self) -> None:
        self.allocate()
        self.probe()

    def run(self, plan: ShardedPlan, src_ptr, npu_block, npu_tensors, h2d) -> list:
        """Enqueue every chunk's H2D copy, all-gather and scatter on the current stream.

        ``h2d(src_ptrs, dst_ptrs, sizes)`` copies host bytes to device bytes.
        Chunk k+1 is copied and its all-gather issued before chunk k is waited
        for, so the copies and scatters overlap the gathers. Chunk k+2 reuses
        chunk k's slots: its copy is enqueued after chunk k's wait, and its
        all-gather, which waits for the current stream when issued, after chunk
        k's scatters.

        Returns tensors that must stay alive until the stream passes the
        caller's end event.
        """
        if self._send is None:
            self.allocate()
        src_ptr = np.asarray(src_ptr, dtype=np.int64)
        npu_block = np.asarray(npu_block, dtype=np.int64)
        chunks = plan.chunks
        slot, world = self._slot, self.world

        # Destination row of every slot of every piece, uploaded once per job.
        offsets, rows, pos = [], [], 0
        for chunk in chunks:
            for pc in chunk.pieces:
                offsets.append(pos)
                rows.append(npu_block[pc.slot_desc.reshape(-1)])
                pos += int(pc.slot_desc.size)
        host_idx = torch.from_numpy(np.concatenate(rows))
        if self.device.type == "cpu":
            dev_idx = host_idx
        else:
            host_idx = host_idx.pin_memory()
            dev_idx = host_idx.to(self.device, non_blocking=True)

        send_base = int(self._send.data_ptr())

        def issue(k: int):
            chunk = chunks[k]
            base = send_base + (k % _NUM_SLOTS) * slot
            srcs, dsts, sizes = [], [], []
            for pc in chunk.pieces:
                m = pc.num_slots
                srcs.append(src_ptr[pc.slot_desc[self.rank]])
                dsts.append(base + pc.send_offset + np.arange(m, dtype=np.int64) * pc.page_size)
                sizes.append(np.full(m, pc.page_size, dtype=np.int64))
            h2d(np.concatenate(srcs), np.concatenate(dsts), np.concatenate(sizes))
            n = chunk.send_bytes
            so, ro = (k % _NUM_SLOTS) * slot, (k % _NUM_SLOTS) * slot * world
            return self._allgather(self._recv[ro:ro + n * world], self._send[so:so + n])

        works = {0: issue(0)}
        i = 0
        for k, chunk in enumerate(chunks):
            if k + 1 < len(chunks):
                works[k + 1] = issue(k + 1)
            works.pop(k).wait()
            n = chunk.send_bytes
            ro = (k % _NUM_SLOTS) * slot * world
            by_rank = self._recv[ro:ro + n * world].view(world, n)
            for pc in chunk.pieces:
                m = pc.num_slots
                region = by_rank[:, pc.send_offset:pc.send_offset + m * pc.page_size]
                idx = dev_idx[offsets[i]:offsets[i] + world * m].view(world, m)
                i += 1
                scatter_rows(npu_tensors[pc.tensor_idx], idx,
                             region.view(world, m, pc.page_size), pc.page_size)
        return [host_idx, dev_idx]
