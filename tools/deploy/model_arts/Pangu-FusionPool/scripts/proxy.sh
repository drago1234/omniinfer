#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
set -eo pipefail
source "$(dirname "${BASH_SOURCE[0]}")/runtime.sh"
init_runtime proxy
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export MOONCAKE_METADATA_PORT="${MOONCAKE_METADATA_PORT:-8081}"
export MOONCAKE_RPC_PORT="${MOONCAKE_RPC_PORT:-16752}"
export MOONCAKE_METRICS_PORT="${MOONCAKE_METRICS_PORT:-$((MOONCAKE_METADATA_PORT + 100))}"
export MOONCAKE_KV_EVENTS_PORT="${MOONCAKE_KV_EVENTS_PORT:-5599}"
if [[ "${MA_DRY_RUN:-0}" != 1 ]]; then
if [[ "${FUSION_POOL_START_MASTER:-true}" == true ]]; then
    bash "$SCRIPT_DIR/mooncake_master.sh"
fi
python3 - "$MOONCAKE_MASTER_IP" "$MOONCAKE_METADATA_PORT" "$MOONCAKE_RPC_PORT" "$MOONCAKE_METRICS_PORT" <<'PY'
import os, socket, sys, time
deadline = time.monotonic() + int(os.getenv("FUSION_POOL_MASTER_WAIT_SECONDS", "600"))
while True:
    try:
        for port in map(int, sys.argv[2:]):
            with socket.create_connection((sys.argv[1], port), timeout=2):
                pass
        break
    except OSError:
        if time.monotonic() >= deadline:
            raise SystemExit("ERROR: Mooncake master metadata/RPC/admin ports not ready")
        time.sleep(2)
PY
fi
log_dir="${LOG_PATH_PROXY:-/home/ma-user/log}/proxy/${POD_IP}"
mkdir -p "$log_dir"
if [[ "${MA_DRY_RUN:-0}" != 1 ]]; then
    cd "$CODE_PATH/components/omni-proxy/omni_proxy"
fi
cmd=(bash omni_proxy.sh \
    --listen-port "${PROXY_PORT:-7000}" \
    --prefill-endpoints "$PREFILL_API_SERVER_LIST" \
    --decode-endpoints "$DECODE_API_SERVER_LIST" \
    --log-file "$log_dir/nginx_error.log" --log-level info \
    --access-log-file "$log_dir/nginx_access.log" \
    --omni-proxy-model-path "$MODEL_PATH" \
    --core-num 16 --start-core-index 16 --subrequest-output-buffer-size 10M \
    --omni-proxy-pd-policy sequential --omni-proxy-max-batch-num-token 49152 \
    --omni-proxy-prefill-max-num-seqs 4 --omni-proxy-decode-max-num-seqs 3 \
    --omni-proxy-tokenizer-wide-affinity on \
    --omni-proxy-schedule-algo work_aware \
    --omni-proxy-work-aware-short-threshold 98304 \
    --omni-proxy-work-aware-service-floor-tokens 4096 \
    --omni-proxy-pool-aware-scheduling on \
    --omni-proxy-pool-master-events "$MOONCAKE_MASTER_IP:$MOONCAKE_KV_EVENTS_PORT" \
    --omni-proxy-pool-master-admin "$MOONCAKE_MASTER_IP:$MOONCAKE_METRICS_PORT" \
    --omni-proxy-pool-w-hbm 100 --omni-proxy-pool-w-local 90 --omni-proxy-pool-w-remote 80 \
    "$@")
if [[ "${MA_DRY_RUN:-0}" == 1 ]]; then
    printf '%q ' "${cmd[@]}"; printf '\n'
else
    exec "${cmd[@]}"
fi
