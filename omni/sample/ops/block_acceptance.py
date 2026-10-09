# Copyright (c) 2025-2026 Huawei Technologies Co., Ltd. All Rights Reserved.

import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _compute_block_prefix_kernel(
    P_ptr,                 # [D, V]
    Q_ptr,                 # [D, V]
    draft_ids_ptr,         # [D]
    cu_num_draft_ptr,      # [B]，累计结束位置
    greedy_ptr,            # [B]，可选
    C_ptr,                 # [D]，输出 c_i
    scale_ptr,             # [D]，输出 c_(i-1)
    p_stride_row: tl.constexpr,
    p_stride_vocab: tl.constexpr,
    q_stride_row: tl.constexpr,
    q_stride_vocab: tl.constexpr,
    HAS_GREEDY: tl.constexpr,
):
    # 每个 program 负责一个请求。
    request = tl.program_id(0)

    end = tl.load(cu_num_draft_ptr + request).to(tl.int64)
    start = tl.load(
        cu_num_draft_ptr + request - 1,
        mask=request > 0,
        other=0,
    ).to(tl.int64)

    greedy = False
    if HAS_GREEDY:
        greedy = tl.load(greedy_ptr + request) != 0

    log_c = 0.0
    previous_c = 1.0
    valid_prefix = True

    # 循环在 NPU 内执行，不会逐步重新启动 kernel。
    for i in range(end - start):
        row = start + i
        token = tl.load(draft_ids_ptr + row).to(tl.int64)

        valid_prefix = valid_prefix & (token >= 0)
        safe_token = tl.maximum(token, 0)
        p = tl.load(
            P_ptr + row * p_stride_row + safe_token * p_stride_vocab,
            mask=valid_prefix,
            other=1.0,
        ).to(tl.float32)

        q = tl.load(
            Q_ptr + row * q_stride_row + safe_token * q_stride_vocab,
            mask=valid_prefix,
            other=1.0,
        ).to(tl.float32)

        # 与当前 Torch helper 保持一致。
        p = tl.where(greedy, 1.0, p)
        q = tl.where(greedy, 1.0, q)

        # 当前位置恢复采样使用前一个 c。
        tl.store(scale_ptr + row, previous_c)

        log_ratio = tl.log(p) - tl.log(q)
        log_c = tl.minimum(log_c + log_ratio, 0.0)
        current_c = tl.exp(log_c)

        tl.store(C_ptr + row, current_c)
        previous_c = current_c


@triton.jit
def _compute_block_partial_z_kernel(
    P_ptr,                  # [D, V]
    Q_ptr,                  # [D, V]
    C_ptr,                  # [D]，第一个 kernel 的输出
    cu_num_draft_ptr,       # [B]，累计结束位置
    partial_z_ptr,          # [B, MAX_PREFIXES, NUM_TILES]
    p_stride_row: tl.constexpr,
    p_stride_vocab: tl.constexpr,
    q_stride_row: tl.constexpr,
    q_stride_vocab: tl.constexpr,
    VOCAB_SIZE: tl.constexpr,
    MAX_PREFIXES: tl.constexpr,
    NUM_TILES: tl.constexpr,
    BLOCK_SIZE: tl.constexpr,
):
    # 将“请求 × 前缀位置”展开在第一个网格维度。
    request_prefix = tl.program_id(0)
    tile = tl.program_id(1)

    request = request_prefix // MAX_PREFIXES
    prefix = request_prefix % MAX_PREFIXES  # 从 0 开始

    end = tl.load(cu_num_draft_ptr + request).to(tl.int64)
    start = tl.load(
        cu_num_draft_ptr + request - 1,
        mask=request > 0,
        other=0,
    ).to(tl.int64)

    length = end - start
    partial = 0.0

    # 实际长度为 K 时，只计算 K-1 个 Z。
    if prefix + 1 < length:
        row = start + prefix
        next_row = row + 1

        # C[row] 是当前前缀的 c。
        c = tl.load(C_ptr + row).to(tl.float32)

        # 当前 program 负责的词表区间。
        cols = tile * BLOCK_SIZE + tl.arange(0, BLOCK_SIZE)
        mask = cols < VOCAB_SIZE

        # 读取下一个候选位置的 P/Q。
        p = tl.load(
            P_ptr
            + next_row * p_stride_row
            + cols * p_stride_vocab,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        q = tl.load(
            Q_ptr
            + next_row * q_stride_row
            + cols * q_stride_vocab,
            mask=mask,
            other=0.0,
        ).to(tl.float32)

        residual = tl.maximum(c * p - q, 0.0)
        partial = tl.sum(residual, axis=0)

    # 不属于有效前缀的位置也明确写 0。
    output_offset = request_prefix * NUM_TILES + tile
    tl.store(partial_z_ptr + output_offset, partial)


@triton.jit
def _finalize_block_acceptance_kernel(
    C_ptr,                  # [D]
    partial_z_ptr,          # [B, MAX_PREFIXES, NUM_TILES]
    uniform_ptr,            # [D]
    draft_ids_ptr,          # [D], negative IDs end the real prefix
    cu_num_draft_ptr,        # [B]
    greedy_ptr,             # [B]，可选
    tau_ptr,                # [B]
    accepted_ptr,           # [D]
    MAX_PREFIXES: tl.constexpr,
    NUM_TILES: tl.constexpr,
    PADDED_K: tl.constexpr,
    PADDED_TILES: tl.constexpr,
    HAS_GREEDY: tl.constexpr,
):
    request = tl.program_id(0)

    end = tl.load(cu_num_draft_ptr + request).to(tl.int64)
    start = tl.load(
        cu_num_draft_ptr + request - 1,
        mask=request > 0,
        other=0,
    ).to(tl.int64)

    length = end - start

    # 候选位置，从 0 开始。
    positions = tl.arange(0, PADDED_K)
    rows = start + positions
    valid = positions < length
    tokens = tl.load(draft_ids_ptr + rows, mask=valid, other=-1)
    real_length = tl.min(
        tl.where(valid & (tokens < 0), positions, length), axis=0
    )
    valid_prefix = positions < real_length

    c = tl.load(
        C_ptr + rows,
        mask=valid,
        other=0.0,
    ).to(tl.float32)

    u = tl.load(
        uniform_ptr + rows,
        mask=valid,
        other=0.0,
    ).to(tl.float32)

    # 1. 汇总各词表块，得到每个前缀的 Z。
    if MAX_PREFIXES > 0:
        tiles = tl.arange(0, PADDED_TILES)

        offsets = (
            (
                request.to(tl.int64) * MAX_PREFIXES
                + positions[:, None]
            )
            * NUM_TILES
            + tiles[None, :]
        )

        partial_mask = (
            (positions[:, None] < MAX_PREFIXES)
            & (positions[:, None] + 1 < length)
            & (tiles[None, :] < NUM_TILES)
        )

        partial = tl.load(
            partial_z_ptr + offsets,
            mask=partial_mask,
            other=0.0,
        )

        # [PADDED_K, PADDED_TILES] → [PADDED_K]
        z = tl.sum(partial, axis=1)
    else:
        # 最多一个 draft 时，没有需要计算的 Z。
        z = tl.full((PADDED_K,), 0.0, tl.float32)

    # 2. 计算 h，避免分母为零时直接相除。
    denominator = z + (1.0 - c)
    safe_denominator = tl.where(
        denominator == 0.0, 1.0, denominator
    )

    h = tl.where(
        denominator == 0.0,
        1.0,
        z / safe_denominator,
    )

    # 最后一个实际候选的阈值为 c_K。
    is_last = valid_prefix & (positions + 1 == real_length)
    h = tl.where(is_last, c, h)

    # 3. 选最长通过前缀，不能用“遇到失败就停止”。
    passed = valid_prefix & (u <= h) & (h > 0.0)

    candidate_lengths = tl.where(
        passed, positions + 1, 0
    )
    request_tau = tl.max(candidate_lengths, axis=0)

    # 与当前 Torch helper 保持一致：
    # greedy 请求的真正接受结果由原 greedy 分支决定。
    if HAS_GREEDY:
        greedy = tl.load(greedy_ptr + request) != 0
        request_tau = tl.where(greedy, 0, request_tau)

    tl.store(tau_ptr + request, request_tau)

    # 4. 将 τ 转换为连续前缀 mask。
    accepted = positions < request_tau
    tl.store(
        accepted_ptr + rows,
        accepted,
        mask=valid,
    )


def compute_block_prefix_triton(
    draft_token_ids,
    cu_num_draft_tokens,
    draft_probs,
    target_probs,
    is_greedy=None,
):
    # 一维输入按连续布局读取；P/Q 使用实际 stride。
    if not draft_token_ids.is_contiguous():
        raise ValueError("draft_token_ids must be contiguous")
    if not cu_num_draft_tokens.is_contiguous():
        raise ValueError("cu_num_draft_tokens must be contiguous")
    if is_greedy is not None:
        if not is_greedy.is_contiguous():
            raise ValueError("is_greedy must be contiguous")

    num_tokens = draft_token_ids.numel()
    batch_size = cu_num_draft_tokens.numel()

    c = torch.empty(
        num_tokens,
        dtype=torch.float32,
        device=target_probs.device,
    )
    target_scale = torch.empty_like(c)

    if num_tokens == 0:
        return c, target_scale

    _compute_block_prefix_kernel[(batch_size,)](
        target_probs,
        draft_probs,
        draft_token_ids,
        cu_num_draft_tokens,
        is_greedy,
        c,
        target_scale,
        target_probs.stride(0),
        target_probs.stride(1),
        draft_probs.stride(0),
        draft_probs.stride(1),
        HAS_GREEDY=is_greedy is not None,
        num_warps=1,
    )

    return c, target_scale


def compute_block_partial_z_triton(
    draft_probs,
    target_probs,
    c,
    cu_num_draft_tokens,
    max_num_draft_tokens,
):
    if not c.is_contiguous():
        raise ValueError("c must be contiguous")
    if not cu_num_draft_tokens.is_contiguous():
        raise ValueError("cu_num_draft_tokens must be contiguous")
    if not (max_num_draft_tokens >= 0):
        raise ValueError("max_num_draft_tokens must be non-negative")

    batch_size = cu_num_draft_tokens.numel()
    vocab_size = target_probs.shape[1]

    block_size = 4096
    num_tiles = triton.cdiv(vocab_size, block_size)
    max_prefixes = max(max_num_draft_tokens - 1, 0)

    partial_z = torch.empty(
        (batch_size, max_prefixes, num_tiles),
        dtype=torch.float32,
        device=target_probs.device,
    )

    # 没有请求，或者最多只有一个 draft：无需计算 Z。
    if batch_size == 0 or max_prefixes == 0:
        return partial_z

    grid = (batch_size * max_prefixes, num_tiles)

    _compute_block_partial_z_kernel[grid](
        target_probs,
        draft_probs,
        c,
        cu_num_draft_tokens,
        partial_z,
        target_probs.stride(0),
        target_probs.stride(1),
        draft_probs.stride(0),
        draft_probs.stride(1),
        VOCAB_SIZE=vocab_size,
        MAX_PREFIXES=max_prefixes,
        NUM_TILES=num_tiles,
        BLOCK_SIZE=block_size,
        num_warps=4,
    )

    return partial_z


def finalize_block_acceptance_triton(
    c,
    partial_z,
    uniform_probs,
    cu_num_draft_tokens,
    is_greedy=None,
    *,
    draft_token_ids,
):
    if not c.is_contiguous():
        raise ValueError("c must be contiguous")
    if not partial_z.is_contiguous():
        raise ValueError("partial_z must be contiguous")
    if not uniform_probs.is_contiguous():
        raise ValueError("uniform_probs must be contiguous")
    if uniform_probs.dtype != torch.float32:
        raise TypeError("uniform_probs must have dtype torch.float32")
    if not cu_num_draft_tokens.is_contiguous():
        raise ValueError("cu_num_draft_tokens must be contiguous")
    if uniform_probs.numel() != c.numel():
        raise ValueError("uniform_probs and c must have the same number of elements")
    if not draft_token_ids.is_contiguous():
        raise ValueError("draft_token_ids must be contiguous")
    if draft_token_ids.numel() != c.numel():
        raise ValueError("draft_token_ids and c must have the same number of elements")


    if is_greedy is not None:
        if not is_greedy.is_contiguous():
            raise ValueError("is_greedy must be contiguous")

    batch_size = cu_num_draft_tokens.numel()
    if partial_z.shape[0] != batch_size:
        raise ValueError("partial_z must have one row per request")

    max_prefixes = partial_z.shape[1]
    num_tiles = partial_z.shape[2]

    # Z 最多有 K-1 个，候选最多有 K 个。
    padded_k = triton.next_power_of_2(max_prefixes + 1)
    padded_tiles = triton.next_power_of_2(max(num_tiles, 1))

    tau = torch.empty(
        batch_size,
        dtype=torch.int32,
        device=c.device,
    )
    accepted = torch.empty(
        c.numel(),
        dtype=torch.bool,
        device=c.device,
    )

    if batch_size == 0:
        return tau, accepted
    if c.numel() == 0:
        tau.zero_()
        return tau, accepted

    _finalize_block_acceptance_kernel[(batch_size,)](
        c,
        partial_z,
        uniform_probs,
        draft_token_ids,
        cu_num_draft_tokens,
        is_greedy,
        tau,
        accepted,
        MAX_PREFIXES=max_prefixes,
        NUM_TILES=num_tiles,
        PADDED_K=padded_k,
        PADDED_TILES=padded_tiles,
        HAS_GREEDY=is_greedy is not None,
        num_warps=1,
    )

    return tau, accepted


def compute_block_acceptance_triton(
    draft_token_ids,
    cu_num_draft_tokens,
    draft_probs,
    target_probs,
    uniform_probs,
    max_num_draft_tokens,
    is_greedy=None,
):
    c, target_scale = compute_block_prefix_triton(
        draft_token_ids,
        cu_num_draft_tokens,
        draft_probs,
        target_probs,
        is_greedy,
    )

    partial_z = compute_block_partial_z_triton(
        draft_probs,
        target_probs,
        c,
        cu_num_draft_tokens,
        max_num_draft_tokens,
    )

    tau, accepted = finalize_block_acceptance_triton(
        c,
        partial_z,
        uniform_probs,
        cu_num_draft_tokens,
        is_greedy,
        draft_token_ids=draft_token_ids,
    )

    return tau, accepted, target_scale
