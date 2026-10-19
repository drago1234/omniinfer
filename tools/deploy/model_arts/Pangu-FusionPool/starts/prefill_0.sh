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
if [[ "${MA_DRY_RUN:-0}" == 1 ]]; then
    bash "$MA_SCRIPTS_DIR/proxy.sh"
    exec bash "$MA_SCRIPTS_DIR/prefill.sh" "${COMMON_ARGS[@]}" "${PREFILL_ARGS[@]}"
fi
# Proxy starts the supervised Master; workers wait for its metadata/RPC ports.
bash "$MA_SCRIPTS_DIR/proxy.sh" &
proxy_pid=$!
trap 'kill -TERM "$proxy_pid" 2>/dev/null || true' EXIT
bash "$MA_SCRIPTS_DIR/prefill.sh" "${COMMON_ARGS[@]}" "${PREFILL_ARGS[@]}" &
prefill_pid=$!
trap 'kill -TERM "$prefill_pid" "$proxy_pid" 2>/dev/null || true' TERM INT
# omni_proxy.sh may daemonize nginx and return; a successful return is allowed.
status=0
wait -n "$proxy_pid" "$prefill_pid" || status=$?
if (( status != 0 )); then
    kill -TERM "$prefill_pid" "$proxy_pid" 2>/dev/null || true
    wait "$prefill_pid" 2>/dev/null || true
    exit "$status"
fi
if ! kill -0 "$prefill_pid" 2>/dev/null; then exit 1; fi
wait "$prefill_pid"
if [[ "${MA_DRY_RUN:-0}" != 1 ]]; then
    sleep infinity
fi
