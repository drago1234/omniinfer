#!/usr/bin/env python3
# coding: utf-8
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# This program is free software, you can redistribute it and/or modify it under the terms and conditions of
# CANN Open Software License Agreement Version 2.0 (the "License").
# Please refer to the License for details. You may not use this file except in compliance with the License.
# THIS SOFTWARE IS PROVIDED ON AN "AS IS" BASIS, WITHOUT WARRANTIES OF ANY KIND, EITHER EXPRESS OR IMPLIED,
# INCLUDING BUT NOT LIMITED TO NON-INFRINGEMENT, MERCHANTABILITY, OR FITNESS FOR A PARTICULAR PURPOSE.
# See LICENSE in the root of the software repository for the full text of the License.
# -----------------------------------------------------------------------------------------------------------
"""
Engram Forward PyPTO Kernel Implementation (Multi-Head, MHC).

多头 (m heads) Engram 前向算子: bs 合轴 + 逐头 Key/Value 投影 + RMSNorm + sign-sqrt gate。
BF16 输入, FP32 中间累加 (matmul 走 FP32 acc); output 输出 BF16, 中间量 score/gate 输出 FP32。
反向算子见 ai_infra_pypto_engram_proj_gate_grad (../ai_infra_pypto_engram_proj_gate_grad)。

计算路径 (per head m, bs = batch_size*seq_len):
  key       = embeddings_in @ Wk^(m)                              # [bs, h] FP32
  value     = embeddings_in @ Wv                                  # [bs, h] FP32
  rms_k     = sqrt(mean(key^2) + eps);  rms_q = sqrt(mean(q^2) + eps)
  score     = sum((key*q)*(gamma_key*gamma_query), -1) / (rms_k*rms_q) / sqrt(h)   # [bs] FP32
  gate      = sigmoid( sign(score) * sqrt(clamp(|score|, c)) )      # [bs] FP32
  output = (gate * value).to(BF16)                          # [bs, h] BF16

输出: output(BF16), score_cache_out(FP32), key_cache_out(BF16), value_cache_out(BF16), gate_cache_out(FP32)
"""

import math

import torch
import pypto


def sign_sqrt_clamp(tensor, clamp_value):
    """gate 预激活: |tensor|.clamp_min.sqrt * sign, 全程 FP32"""
    clamped = pypto.maximum(pypto.abs(tensor), clamp_value)
    sqrt_part = pypto.sqrt(clamped)
    return pypto.sign(tensor) * sqrt_part


@pypto.frontend.jit(
    pass_options={
        "cube_l1_reuse_setting": {"DEFAULT": 4},
        "vec_nbuffer_setting": {"DEFAULT": 4, "func7_3": 32, "func5_0": 32},
    },
    runtime_options={
        "stitch_function_max_num": 64,
    },
)
def ai_infra_engram_proj_gate_kernel(
    # inputs
    hidden_states_in: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16), # [bs, m, h]
    embeddings_in: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16),   # [bs, de]
    proj_weights_key_in: pypto.Tensor([], pypto.DT_BF16), # [m, de, h]
    proj_weights_value_in: pypto.Tensor([], pypto.DT_BF16), # [de, h]
    gamma_key_in: pypto.Tensor([], pypto.DT_BF16), # [m, h]
    gamma_query_in: pypto.Tensor([], pypto.DT_BF16),  # [m, h]
    # outputs
    value_cache_out: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16), # [bs, h]
    key_cache_out: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16), # [bs, m, h]
    score_cache_out: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32), # [bs, m]
    gate_cache_out: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_FP32), # [bs, m]
    output: pypto.Tensor([pypto.DYNAMIC, ...], pypto.DT_BF16), # [bs, m, h]
    # attrs
    clamp_value,
    eps,
    return_cache,
):
    pypto.experimental.set_operation_options(combine_axis=True)
    bs, m, h = hidden_states_in.shape
    sqrt_h = math.sqrt(h)
    unroll_list = [4, 8, 16, 32, 64, 128]
    first_axis = 8
    if h > 1536:
        first_axis = 4

    # 合轴为 bs, 单层 loop_unroll 直接遍历
    for bs_idx, tile in pypto.loop_unroll(0, bs, 1, unroll_list=unroll_list):
        offset = bs_idx
        pypto.set_semantic_label("value_proj")
        pypto.set_cube_tile_shapes([128, 128], [128, 256], [128, 128])
        embeddings = embeddings_in[offset:offset + tile]
        value_proj = pypto.matmul(embeddings, proj_weights_value_in, pypto.DT_BF16)
        if return_cache:
            pypto.set_vec_tile_shapes(first_axis, h)
            value_cache_out[offset:offset + tile] = value_proj

        # ── 逐头 key/query/gate/output ──
        for m_idx in pypto.loop(m):
            pypto.set_semantic_label("key_proj")
            pypto.set_cube_tile_shapes([128, 128], [128, 256], [128, 128])
            embeddings = embeddings_in[offset:offset + tile]
            key_proj = pypto.matmul(embeddings, proj_weights_key_in[m_idx], pypto.DT_BF16)
            if return_cache:
                pypto.set_vec_tile_shapes(first_axis, h)
                key_cache_out[offset:offset + tile, m_idx] = key_proj

            pypto.set_semantic_label("score")
            pypto.set_vec_tile_shapes(first_axis, h)
            key_proj_fp32 = pypto.cast(key_proj, pypto.DT_FP32)
            query = hidden_states_in[offset:offset + tile, m_idx]
            query_fp32 = pypto.cast(query, pypto.DT_FP32)
            rms_k = pypto.sqrt(pypto.sum(key_proj_fp32 * key_proj_fp32, dim=-1) * (1.0 / h) + eps)
            rms_q = pypto.sqrt(pypto.sum(query_fp32 * query_fp32, dim=-1) * (1.0 / h) + eps)
            pypto.set_pass_options(sg_set_scope=1)
            gamma_key = pypto.cast(gamma_key_in[m_idx:m_idx + 1], pypto.DT_FP32)
            gamma_query = pypto.cast(gamma_query_in[m_idx:m_idx + 1], pypto.DT_FP32)
            gamma_qk = gamma_key * gamma_query
            pypto.set_pass_options(sg_set_scope=-1)
            prod = (key_proj_fp32 * query_fp32) * gamma_qk
            score = pypto.div(pypto.div(pypto.sum(prod, dim=-1), (rms_k * rms_q),
                                        pypto.PrecisionType.INTRINSIC), sqrt_h, pypto.PrecisionType.INTRINSIC)
            if return_cache:
                score_cache_out[offset:offset + tile, m_idx] = score

            pypto.set_semantic_label("gate")
            pypto.set_vec_tile_shapes(first_axis)
            sign_qk = sign_sqrt_clamp(score, clamp_value)
            gate = sign_qk.sigmoid()
            if return_cache:
                gate_cache_out[offset:offset + tile, m_idx] = gate
            gate_unsq = gate.unsqueeze(-1)
            pypto.set_vec_tile_shapes(first_axis, h)
            value_proj_fp32 = pypto.cast(value_proj, pypto.DT_FP32)
            out = pypto.cast(gate_unsq * value_proj_fp32, pypto.DT_BF16)
            output[offset:offset + tile, m_idx] = out


def ai_infra_engram_proj_gate(
    hidden_states,       # [b, s, m, h] BF16
    embeddings,          # [b, s, de]   BF16
    proj_weights_key,    # [m, de, h]   BF16
    proj_weights_value,  # [de, h]      BF16
    gamma_key,           # [m, h]       BF16
    gamma_query,         # [m, h]       BF16
    clamp_value=1e-6,
    eps=1e-6,
    return_cache=True,
) -> tuple:
    """
    Inputs:
      hidden_states:      [b, s, m, h]      BF16
      embeddings:         [b, s, de]        BF16
      proj_weights_key:   [m, de, h]        BF16
      proj_weights_value: [de, h]           BF16
      gamma_key:          [m, h]            BF16
      gamma_query:        [m, h]            BF16
      return_cache:            推理场景置 False, 跳过 4 个 cache 的分配/写出/回传, 节省开销

    Returns:
      5 元组 (output, score_cache, key_cache, value_cache, gate_cache)
      return_cache=True (训练):  后 4 项为反向 cache
      return_cache=False (推理): 后 4 项为 None (返回签名一致, 调用方无需分支解包)
    """
    # ── 输入校验: 拦截空 tensor 和非连续 tensor ──
    for name, t in (("hidden_states", hidden_states),
                    ("embeddings", embeddings),
                    ("proj_weights_key", proj_weights_key),
                    ("proj_weights_value", proj_weights_value),
                    ("gamma_key", gamma_key),
                    ("gamma_query", gamma_query)):
        if t.numel() == 0:
            raise ValueError(f"ai_infra_engram_proj_gate: input '{name}' must not be empty, got shape {tuple(t.shape)}")
        if not t.is_contiguous():
            raise ValueError(f"ai_infra_engram_proj_gate: input '{name}' must be contiguous, "
                             f"got shape {tuple(t.shape)}")

    b, s, m, h = hidden_states.shape
    de = embeddings.shape[-1]
    device = hidden_states.device
    if return_cache:
        key_cache = torch.zeros([b * s, m, h], dtype=torch.bfloat16, device=device)
        value_cache = torch.zeros([b * s, h], dtype=torch.bfloat16, device=device)
        gate_cache = torch.zeros([b * s, m], dtype=torch.float32, device=device)
        score_cache = torch.zeros([b * s, m], dtype=torch.float32, device=device)
    else:
        # 推理场景: 不分配 cache, 传最小 dummy 占位 (kernel 不生成 cache 写出代码)
        key_cache = torch.zeros([1], dtype=torch.bfloat16, device=device)
        value_cache = torch.zeros([1], dtype=torch.bfloat16, device=device)
        gate_cache = torch.zeros([1], dtype=torch.float32, device=device)
        score_cache = torch.zeros([1], dtype=torch.float32, device=device)
    output = torch.zeros([b * s, m, h], dtype=torch.bfloat16, device=device)

    ai_infra_engram_proj_gate_kernel(
        hidden_states.view(b * s, m, h),
        embeddings.view(b * s, de),
        proj_weights_key,
        proj_weights_value,
        gamma_key,
        gamma_query,
        value_cache,
        key_cache,
        score_cache,
        gate_cache,
        output,
        clamp_value,
        eps,
        return_cache,
    )

    output = output.view(b, s, m, h)
    if not return_cache:
        # 推理场景: 返回签名与训练一致, 4 个 cache 以 None 占位
        return output, None, None, None, None
    score_cache = score_cache.view(b, s, m)
    key_cache = key_cache.view(b, s, m, h)
    value_cache = value_cache.view(b, s, h)
    gate_cache = gate_cache.view(b, s, m)
    return output, score_cache, key_cache, value_cache, gate_cache
