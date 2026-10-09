# SPDX-License-Identifier: MIT
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.


import torch
from vllm.triton_utils import tl, triton


@triton.jit
def _select_block_tokens_kernel(
    output_ptr,
    tau_ptr,
    cu_num_draft_ptr,
    draft_ids_ptr,
    recovered_ids_ptr,
    bonus_ids_ptr,
    fill_ptr,
    accepted_num_ptr,
    OUTPUT_WIDTH: tl.constexpr,
    PADDED_WIDTH: tl.constexpr,
    PLACEHOLDER: tl.constexpr,
    HAS_FILL_MASK: tl.constexpr,
):
    request = tl.program_id(0).to(tl.int64)

    end = tl.load(cu_num_draft_ptr + request).to(tl.int64)
    start = tl.load(
        cu_num_draft_ptr + request - 1,
        mask=request > 0,
        other=0,
    ).to(tl.int64)

    length = end - start
    tau = tl.load(tau_ptr + request).to(tl.int64)

    # 与原函数一样：只覆盖本次负责处理的请求。
    fill = True
    if HAS_FILL_MASK:
        fill = tl.load(fill_ptr + request) != 0

    positions = tl.arange(0, PADDED_WIDTH)

    # 已接受的 draft 前缀。
    draft = tl.load(
        draft_ids_ptr + start + positions,
        mask=fill & (positions < tau) & (positions < length),
        other=PLACEHOLDER,
    )

    # 有拒绝时，取第 tau 个候选位置预计算的恢复 token。
    # 全接受时 start + tau 可能越界，必须用 mask 保护。
    recovered = tl.load(
        recovered_ids_ptr + start + tau,
        mask=fill & (tau < length),
        other=PLACEHOLDER,
    )

    bonus = tl.load(
        bonus_ids_ptr + request,
        mask=fill,
        other=PLACEHOLDER,
    )

    boundary_token = tl.where(tau < length, recovered, bonus)

    output = tl.where(
        positions < tau,
        draft,
        tl.where(
            positions == tau,
            boundary_token,
            PLACEHOLDER,
        ),
    )

    tl.store(
        output_ptr + request * OUTPUT_WIDTH + positions,
        output,
        mask=fill & (positions < OUTPUT_WIDTH),
    )

    # 原 select_tokens_by_accepted 返回 int64 接受长度。
    tl.store(accepted_num_ptr + request, tau)


def select_block_tokens_triton(
    output_token_ids,
    tau,
    cu_num_draft_tokens,
    draft_token_ids,
    recovered_token_ids,
    bonus_token_ids,
    fill_this_time,
    max_spec_len,
    placeholder_token_id,
):
    batch_size = cu_num_draft_tokens.numel()
    output_width = max_spec_len + 1

    assert output_token_ids.shape == (batch_size, output_width)
    assert tau.shape == (batch_size,)
    assert bonus_token_ids.numel() == batch_size
    assert recovered_token_ids.numel() == draft_token_ids.numel()

    for tensor in (
        output_token_ids,
        tau,
        cu_num_draft_tokens,
        draft_token_ids,
        recovered_token_ids,
        bonus_token_ids,
    ):
        assert tensor.is_contiguous()

    if fill_this_time is not None:
        assert fill_this_time.shape == (batch_size,)
        assert fill_this_time.is_contiguous()

    accepted_num = torch.empty(
        batch_size,
        dtype=torch.int64,
        device=output_token_ids.device,
    )

    if batch_size == 0:
        return accepted_num

    if draft_token_ids.numel() == 0:
        # 没有 draft，因此接受的 draft 数量全部为 0。
        accepted_num.zero_()

        # 每条请求只输出一个 bonus，其余位置为占位值。
        empty_output = torch.full_like(
            output_token_ids,
            placeholder_token_id,
        )
        empty_output[:, 0].copy_(bonus_token_ids.reshape(batch_size))

        # 只更新当前分支负责的请求，保留其他行的原有结果。
        if fill_this_time is not None:
            empty_output = torch.where(
                fill_this_time.reshape(batch_size, 1) != 0,
                empty_output,
                output_token_ids,
            )

        output_token_ids.copy_(empty_output)
        return accepted_num

    _select_block_tokens_kernel[(batch_size,)](
        output_token_ids,
        tau,
        cu_num_draft_tokens,
        draft_token_ids,
        recovered_token_ids,
        bonus_token_ids,
        fill_this_time,
        accepted_num,
        OUTPUT_WIDTH=output_width,
        PADDED_WIDTH=triton.next_power_of_2(output_width),
        PLACEHOLDER=placeholder_token_id,
        HAS_FILL_MASK=fill_this_time is not None,
        num_warps=1,
    )

    return accepted_num
