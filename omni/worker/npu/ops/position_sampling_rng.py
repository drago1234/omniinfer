# SPDX-License-Identifier: MIT
# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""Batched Philox4x32-10 uniforms indexed only by request seed, position and vocabulary index.

No tl.rand/tl.randint, generator state, native operator loops, or position windows.
key=(seed.low32, seed.high32); counter=(vocab_index//4, 0, pos.low32, pos.high32).
The four Philox words map to four consecutive vocabulary entries. FP32 [0,1)
conversion uses the upper 24 bits, independent of batch shape and tile layout.
"""
import torch
from vllm.triton_utils import tl, triton

from omni_npu import envs


def position_rng_enabled() -> bool:
    """Read before graph capture; a captured graph keeps its selected RNG path."""
    return envs.OMNI_MRV2_POSITION_RNG


@triton.jit
def _mulhi16(x, A: tl.constexpr):
    # Exact high32(A*x). Split both operands into 16-bit limbs so CANN can
    # use ordinary 32-bit vector multiplies instead of extended multiplication.
    # Each partial sum below fits uint32, including its carry.
    lo = x & 0xffff
    hi = (x >> 16) & 0xffff
    p0 = lo * (A & 0xffff)
    p1 = hi * (A & 0xffff) + ((p0 >> 16) & 0xffff)
    p2 = lo * (A >> 16) + (p1 & 0xffff)
    return (hi * (A >> 16) + ((p1 >> 16) & 0xffff)
            + ((p2 >> 16) & 0xffff)).to(tl.uint32)


@triton.jit
def _philox4x32(c0, c1, c2, c3, k0, k1):
    for _ in tl.static_range(10):
        a = tl.full((), 0xD2511F53, tl.uint32)
        b = tl.full((), 0xCD9E8D57, tl.uint32)
        old0, old2 = c0, c2
        c0 = _mulhi16(old2, 0xCD9E8D57) ^ c1 ^ k0
        c2 = _mulhi16(old0, 0xD2511F53) ^ c3 ^ k1
        c1 = (b * old2).to(tl.uint32)
        c3 = (a * old0).to(tl.uint32)
        k0 = (k0 + tl.full((), 0x9E3779B9, tl.uint32)).to(tl.uint32)
        k1 = (k1 + tl.full((), 0xBB67AE85, tl.uint32)).to(tl.uint32)
    return c0, c1, c2, c3


@triton.jit
def _uniform(x):
    return ((x >> 8) & 0x00FFFFFF).to(tl.int32).to(tl.float32) * (1.0 / 16777216.0)


@triton.jit
def _batched_uniform(out, mapping, seeds, positions, cu, rejected, n, v,
                     TILES: tl.constexpr, FIRST_ONLY: tl.constexpr,
                     RESAMPLE: tl.constexpr, B: tl.constexpr):
    tile = tl.program_id(0)
    if FIRST_ONLY:
        row = tile * B + tl.arange(0, B)
        valid_row = row < n
        group = tl.full((B,), 0, tl.uint32)
    else:
        row = tile // TILES
        valid_row = row < n
        group = ((tile % TILES) * B + tl.arange(0, B)).to(tl.uint32)
    token = row.to(tl.int64)
    if RESAMPLE:
        token = (tl.load(cu + row, mask=valid_row, other=0).to(tl.int64)
                 + tl.load(rejected + row, mask=valid_row, other=0))
    state = tl.load(mapping + token, mask=valid_row, other=-1).to(tl.int64)
    valid = valid_row & (state >= 0)
    seed = tl.load(seeds + state, mask=valid, other=0).to(tl.uint64)
    pos = tl.load(positions + token, mask=valid, other=0).to(tl.uint64)
    c0, c1, c2, c3 = _philox4x32(group, tl.full((B,),0,tl.uint32),
                                pos.to(tl.uint32), (pos >> 32).to(tl.uint32),
                                seed.to(tl.uint32), (seed >> 32).to(tl.uint32))
    if FIRST_ONLY:
        tl.store(out + row, tl.where(valid, _uniform(c0), 0.5), mask=valid_row)
    else:
        words = tl.interleave(tl.interleave(c0, c2), tl.interleave(c1, c3))
        idx = (tile % TILES) * (B * 4) + tl.arange(0, B * 4)
        values = tl.where(valid, _uniform(words), 0.5)
        tl.store(out + row.to(tl.int64) * v + idx, values, mask=idx < v)



def make_uniform(
    num_rows: int,
    vocab_size: int,
    device: torch.device | str,
    mapping: torch.Tensor,
    seeds: torch.Tensor,
    positions: torch.Tensor,
    *,
    cu_num_logits: torch.Tensor | None = None,
    rejected_steps: torch.Tensor | None = None,
) -> torch.Tensor:
    """Return FP32 uniforms for contiguous MRv2 device metadata.

    mapping indexes request seed slots; a negative slot denotes padding and
    produces 0.5. Seeds and absolute positions are interpreted as 64-bit bit
    patterns. With vocab_size=1, return U[0] without generating the vocabulary.
    For resampling, cu_num_logits + rejected_steps selects rows on device.
    There is no generator state: graph replay reads updated metadata tensors.
    """
    if (cu_num_logits is None) != (rejected_steps is None):
        raise ValueError("cu_num_logits and rejected_steps must be provided together")
    if num_rows < 0 or vocab_size < 0:
        raise ValueError("Uniform output dimensions must be nonnegative")
    out = torch.empty((num_rows, vocab_size), device=device, dtype=torch.float32)
    if num_rows == 0 or vocab_size == 0:
        return out

    # A scalar per position uses a small row tile; full-vocabulary tiles
    # produce 4*1024 contiguous uniforms. Sampling kernels keep their blocks.
    block = 256 if vocab_size == 1 else 1024
    tiles = triton.cdiv(triton.cdiv(vocab_size, 4), block)
    grid = triton.cdiv(num_rows, block) if vocab_size == 1 else num_rows * tiles
    _batched_uniform[(grid,)](
        out, mapping, seeds, positions, cu_num_logits, rejected_steps,
        num_rows, vocab_size, TILES=tiles, FIRST_ONLY=vocab_size == 1,
        RESAMPLE=rejected_steps is not None, B=block,
        # Philox multiplies and key increments intentionally wrap uint32.
        sanitize_overflow=False,
    )
    return out
