#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
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
PREFILL_ARGS=(--max-num-batched-tokens 49152 --max-num-seqs 4
    --enforce-eager --additional-config '{"enable_low_latency":true}')
bash "$MA_SCRIPTS_DIR/prefill.sh" "${COMMON_ARGS[@]}" "${PREFILL_ARGS[@]}"
if [[ "${MA_DRY_RUN:-0}" != 1 ]]; then
    sleep infinity
fi
