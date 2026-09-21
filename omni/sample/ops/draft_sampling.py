# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
"""NPU draft sampling with external uniform RNG and tiled Triton reductions.

Keep the existing NPU top-k/top-p operator and its dtype/filtering semantics.
No request metadata is copied to the host, and no random numbers are generated
inside Triton. The dense FP32 proposal distribution is retained for p/q verify.
"""

import torch
import torch_npu
from vllm.triton_utils import tl, triton


@triton.jit
def _prepare_draft_logits(
    L, T, K, P, O, KO, PO,
    LS0: tl.constexpr, LS1: tl.constexpr,
    TS: tl.constexpr, KS: tl.constexpr, PS: tl.constexpr,
    V: tl.constexpr, SLOTS: tl.constexpr,
    HAS_K: tl.constexpr, HAS_P: tl.constexpr, FILTER: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    req = row // SLOTS
    cols = tile * BLOCK + tl.arange(0, BLOCK)
    temperature = tl.load(T + req * TS)
    # Match compute_probs: only exact zero is replaced for scaling.
    temperature = tl.where(temperature == 0, 1., temperature)
    values = tl.load(L + row * LS0 + cols * LS1, cols < V, 0).to(tl.float32)
    values = tl.div_rn(values, temperature)
    tl.store(O + row * V + cols, values, cols < V)
    if FILTER:
        if tile == 0:
            k = V
            p = 1.
            if HAS_K:
                k = tl.load(K + req * KS)
            if HAS_P:
                p = tl.load(P + req * PS)
            tl.store(KO + row, k)
            tl.store(PO + row, p)


@triton.jit
def _draft_stats_and_candidates(
    L, U, T, STATS, IDS,
    LS0: tl.constexpr, LS1: tl.constexpr, TS: tl.constexpr,
    V: tl.constexpr, SLOTS: tl.constexpr, TILES: tl.constexpr, STATS_STRIDE: tl.constexpr,
    ALL_RANDOM: tl.constexpr, EPS: tl.constexpr, SCALE: tl.constexpr,
    BLOCK: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    cols = tile * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(L + row * LS0 + cols * LS1, cols < V, -float("inf")).to(tl.float32)
    if SCALE or not ALL_RANDOM:
        temperature = tl.load(T + (row // SLOTS) * TS)
    if SCALE:
        x = tl.div_rn(x, tl.where(temperature == 0, 1., temperature))
    # Non-finite rows produce NaN q and torch.argmax selects token zero.
    m = tl.max(tl.where(x != x, float("inf"), x), 0)
    safe_m = tl.where(m == -float("inf"), 0., m)
    z = tl.sum(tl.exp(x - safe_m), 0)
    u = tl.load(U + row * V + cols, cols < V, 0.)
    survival = tl.minimum(tl.maximum(1. - u, 5.960464477539063e-8),
                          0.9999999403953552)
    # q/E and x-log(E) have the same mathematical argmax. Computing in
    # log space lets candidate selection run before the global softmax sum.
    score = (x - safe_m) - tl.log(-tl.log(survival))
    if not ALL_RANDOM:
        score = tl.where(temperature < EPS, x - safe_m, score)
    score = tl.where(cols < V, score, -float("inf"))
    best = tl.max(score, 0)
    index = tl.min(tl.where((score == best) & (cols < V), cols, 2147483647), 0)
    tl.store(STATS + row * TILES + tile, m)
    tl.store(STATS + STATS_STRIDE + row * TILES + tile, z)
    tl.store(STATS + 2 * STATS_STRIDE + row * TILES + tile, best)
    tl.store(IDS + row * TILES + tile, index)


@triton.jit
def _draft_normalize_and_select(
    L, T, STATS, IDS, Q, OUT,
    LS0: tl.constexpr, LS1: tl.constexpr, TS: tl.constexpr,
    V: tl.constexpr, SLOTS: tl.constexpr, TILES: tl.constexpr, STATS_STRIDE: tl.constexpr,
    SCALE: tl.constexpr, BLOCK: tl.constexpr, REDUCE: tl.constexpr,
):
    row = tl.program_id(0)
    tile = tl.program_id(1)
    parts = tl.arange(0, REDUCE)
    ms = tl.load(STATS + row * TILES + parts, parts < TILES, -float("inf"))
    zs = tl.load(STATS + STATS_STRIDE + row * TILES + parts, parts < TILES, 0.)
    maximum = tl.max(ms, 0)
    scales = tl.where(ms == -float("inf"), 0., tl.exp(ms - maximum))
    total = tl.sum(zs * scales, 0)
    cols = tile * BLOCK + tl.arange(0, BLOCK)
    x = tl.load(L + row * LS0 + cols * LS1, cols < V, -float("inf")).to(tl.float32)
    if SCALE:
        temperature = tl.load(T + (row // SLOTS) * TS)
        x = tl.div_rn(x, tl.where(temperature == 0, 1., temperature))
    q = tl.exp(x - maximum) / total
    tl.store(Q + row * V + cols, q, cols < V)
    if tile == 0:
        scores = tl.load(STATS + 2 * STATS_STRIDE + row * TILES + parts, parts < TILES, -float("inf"))
        scores = scores + (ms - maximum)
        ids = tl.load(IDS + row * TILES + parts, parts < TILES, 2147483647)
        best = tl.max(scores, 0)
        index = tl.min(tl.where(scores == best, ids, 2147483647), 0)
        index = tl.where((maximum == float("inf")) | (maximum == -float("inf")), 0, index)
        tl.store(OUT + row, index)


def prepare_draft_logits(logits, temperature, top_k, top_p, filter_dtype):
    """Scale in FP32, then cast exactly where the existing NPU filter casts."""
    rows, vocab = logits.shape
    slots = rows // temperature.numel()
    filtering = top_k is not None or top_p is not None
    processed = torch.empty((rows, vocab), device=logits.device,
                            dtype=filter_dtype if filtering else torch.float32)
    k = torch.empty(rows, device=logits.device, dtype=torch.int32) if filtering else None
    p = torch.empty(rows, device=logits.device, dtype=filter_dtype) if filtering else None
    _prepare_draft_logits[(rows, triton.cdiv(vocab, 2048))](
        logits, temperature, top_k, top_p, processed, k, p,
        logits.stride(0), logits.stride(1), temperature.stride(0),
        top_k.stride(0) if top_k is not None else 0,
        top_p.stride(0) if top_p is not None else 0,
        vocab, slots, top_k is not None, top_p is not None, filtering, 2048,
    )
    if filtering:
        _, processed = torch_npu.npu_top_k_top_p_sample(
            processed, k, p, q=None, is_need_logits=True)
    return processed


def make_uniform_noise(logits, generators, tokens_per_req):
    """Preserve request-generator routing and parallel-draft RNG ownership.

    uniform_ replaces exponential_; exact seeded sequences need not match the
    old algorithm. Parallel drafting retains the existing default generator.
    """
    noise = torch.empty(logits.shape, device=logits.device, dtype=torch.float32)
    if tokens_per_req != 1:
        noise.uniform_()
    else:
        if len(generators) != logits.shape[0]:
            noise.uniform_()
        for row, generator in generators.items():
            noise[row:row + 1].uniform_(generator=generator)
    return noise


def sample_processed_draft_logits(processed, temperature, uniform_noise,
                                  all_random, sampling_eps, scale_temperature=False):
    """Return (int64 IDs, dense FP32 q); never overwrite logits or noise."""
    rows, vocab = processed.shape
    block = 2048
    tiles = triton.cdiv(vocab, block)
    # Pack the three FP32 reductions without creating tensor views. IDs keep
    # their integer dtype; outputs remain independent across draft steps/graphs.
    stats = torch.empty((3, rows, tiles), device=processed.device, dtype=torch.float32)
    indices = torch.empty((rows, tiles), device=processed.device, dtype=torch.int32)
    probs = torch.empty((rows, vocab), device=processed.device, dtype=torch.float32)
    output = torch.empty(rows, device=processed.device, dtype=torch.int64)
    _draft_stats_and_candidates[(rows, tiles)](
        processed, uniform_noise, temperature, stats, indices,
        processed.stride(0), processed.stride(1), temperature.stride(0),
        vocab, rows // temperature.numel(), tiles, rows * tiles, all_random, sampling_eps,
        scale_temperature, block)
    _draft_normalize_and_select[(rows, tiles)](
        processed, temperature, stats, indices, probs, output,
        processed.stride(0), processed.stride(1), temperature.stride(0),
        vocab, rows // temperature.numel(), tiles, rows * tiles, scale_temperature,
        block, triton.next_power_of_2(tiles))
    return output, probs


def sample_draft_logits(logits, sampling_metadata, filter_dtype, sampling_eps):
    temperature = sampling_metadata.temperature
    if logits.shape[0] == 0:
        return (torch.empty(0, device=logits.device, dtype=torch.int64),
                torch.empty(logits.shape, device=logits.device, dtype=torch.float32))
    filtering = sampling_metadata.top_k is not None or sampling_metadata.top_p is not None
    # Without filtering, fuse temperature scaling into the reductions as well.
    processed = prepare_draft_logits(
        logits, temperature, sampling_metadata.top_k, sampling_metadata.top_p,
        filter_dtype) if filtering else logits
    uniform = make_uniform_noise(
        logits, sampling_metadata.generators, logits.shape[0] // temperature.numel())
    return sample_processed_draft_logits(
        processed, temperature, uniform, sampling_metadata.all_random, sampling_eps,
        scale_temperature=not filtering)
