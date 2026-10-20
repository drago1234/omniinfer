#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# One startup command for the two-machine ModelArts D resource group.
# ModelArts executes it on both machines; POD_IP selects local DP ranks.
set -eo pipefail
MA_SCRIPTS_DIR="${MA_SCRIPTS_DIR:-$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/../scripts" && pwd)}"
COMMON_ARGS=(
    --max-model-len 524288 --no-disable-hybrid-kv-cache-manager
    --enable-expert-parallel --enable-prompt-tokens-details --enable-chunked-prefill
    --dtype bfloat16 --kv-cache-dtype li_int8_ds_mla
    --reasoning-parser pangu --enable-auto-tool-choice --tool-call-parser pangu
    --reasoning-config '{"reasoning_start_str":"<think>","reasoning_end_str":"</think>"}'
    --mamba-block-size 128 --enable-prefix-caching
    --speculative-config '{"method":"mtp","num_speculative_tokens":3}'
    --enable-logging-iteration-details --distributed-executor-backend mp
)
DECODE_ARGS=(--max-num-seqs 3
    --compilation-config '{"mode":3,"cudagraph_mode":"FULL","cudagraph_capture_sizes":[4,8,12],"backend":"","compile_sizes":[4,8,12]}'
    --additional-config '{"enable_low_latency":true,"npugraph_ex_config":{"enable":true,"super_kernel_optimize":false,"static_kernel_compile":true}}')
bash "$MA_SCRIPTS_DIR/decode.sh" "${COMMON_ARGS[@]}" "${DECODE_ARGS[@]}"
if [[ "${MA_DRY_RUN:-0}" != 1 ]]; then
    sleep infinity
fi
