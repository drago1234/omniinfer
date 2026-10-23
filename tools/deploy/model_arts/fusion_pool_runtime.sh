#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Shared implementation selected only by ENABLE_FUSION_POOL=1.
init_runtime() {
    local role=$1 topology_exports
    if [[ -f ~/.bashrc ]]; then source ~/.bashrc; fi
    source /usr/local/Ascend/ascend-toolkit/set_env.sh 2>/dev/null || true
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    : "${MODEL_PATH:?Set MODEL_PATH in ModelArts}"
    : "${POD_IP:?ModelArts must provide POD_IP}"
    export CODE_PATH="${CODE_PATH:-/workspace/omniinfer}"
    export PANGU_FUSION_ROOT="${PANGU_FUSION_ROOT:-$CODE_PATH/components/Pangu-Fusion}"
    export SOCKET_IFNAME="${SOCKET_IFNAME:-eth0}"
    export GLOO_SOCKET_IFNAME="$SOCKET_IFNAME" TP_SOCKET_IFNAME="$SOCKET_IFNAME"
    export PYTHONHASHSEED="${PYTHONHASHSEED:-1234}"
    export ROLE="$role" OMNI_PD_ROLE="$role"
    unset http_proxy https_proxy HTTP_PROXY HTTPS_PROXY proxy
    topology_exports=$(python3 "$SCRIPT_DIR/get_node_ip_list.py" --role "$role") || return 1
    eval "$topology_exports"
    export LOG_PATH="${LOG_PATH:-/home/ma-user/log}/${role}/${POD_IP}"
    mkdir -p "$LOG_PATH"
    export ASCEND_PROCESS_LOG_PATH="$LOG_PATH/ascend"
    mkdir -p "$ASCEND_PROCESS_LOG_PATH"
    export OMNI_LLMDATADIST_ZMQ_PORT="${OMNI_LLMDATADIST_ZMQ_PORT:-5568}"
    export VLLM_LLMDATADIST_ZMQ_PORT="$OMNI_LLMDATADIST_ZMQ_PORT"
    export ASCEND_RT_VISIBLE_DEVICES="${ASCEND_RT_VISIBLE_DEVICES:-0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15}"
    if [[ "$ASCEND_RT_VISIBLE_DEVICES" != '0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15' ]]; then
        echo 'ERROR: this deployment expects visible logical devices 0..15' >&2
        return 1
    fi
    export HCCL_INTRA_PCIE_ENABLE=0 HCCL_INTRA_ROCE_ENABLE=1
    export VLLM_USE_V1=1
    export VLLM_WORKER_MULTIPROC_METHOD="${VLLM_WORKER_MULTIPROC_METHOD:-fork}"
    export OMNI_NPU_VLLM_PATCHES="${OMNI_NPU_VLLM_PATCHES:-ALL}"
    export OMNI_VLLM_PATCHES="${OMNI_VLLM_PATCHES:-ALL}"
    export VLLM_ENABLE_MC2="${VLLM_ENABLE_MC2:-1}"
    export NUM_DIE_PER_MACH=16
    export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
    if [[ "$role" != proxy && "${MA_DRY_RUN:-0}" != 1 ]]; then
        cd "$CODE_PATH/tools/deploy/start_server" || return 1
    fi
}

prepare_pool() {
    if [[ "${ENABLE_FUSION_POOL:-1}" != 1 || "${ENABLE_OMNI_CACHE:-0}" != 0 || "${kv_offload:-false}" != false ]]; then
        echo 'ERROR: this deployment requires KV Pool without Omni Cache/CPU offload' >&2
        return 1
    fi
    if [[ "${MA_DRY_RUN:-0}" == 1 ]]; then
        KV_TRANSFER_CONFIG=$(python3 "$SCRIPT_DIR/fusion_pool_config.py" "$ROLE" --dry-run) || return 1
        export KV_TRANSFER_CONFIG
        return 0
    fi
    local pool_root="$PANGU_FUSION_ROOT/Pangu-FusionPool"
    [[ -d "$pool_root/kv_common" && -d "$pool_root/fusion_pool" ]] || {
        echo "ERROR: missing kv_common/fusion_pool under $pool_root" >&2; return 1;
    }
    git config --global --add safe.directory "$CODE_PATH" || true
    pip3 install --no-build-isolation --no-deps -e "$pool_root/kv_common" -e "$pool_root/fusion_pool" || return 1
    # Reject conflicting installed registrations before importing the connector.
    python3 - <<'PY' || return 1
from importlib.metadata import entry_points
expected = "fusion_pool.connector:register_npumooncake_connector"
entries = [e for e in entry_points().select(group="omni_npu.kv_connectors")
           if e.name == "NPUMooncakeStoreConnector"]
if not entries or any(e.value != expected for e in entries):
    raise SystemExit("ERROR: incompatible connector registrations; use a clean matching image: "
                     + repr([(e.name, e.value) for e in entries]))
PY
    python3 -c 'from fusion_pool.connector.npu_mooncake_store_connector import NPUMooncakeStoreConnector; from fusion_pool.connector.pool_event_multi_connector import MultiConnector' || return 1
    export LD_LIBRARY_PATH="/opt/openEuler/gcc-toolset-12/root/usr/lib64:/usr/local/lib/python3.11/site-packages/mooncake:/usr/local/lib:${LD_LIBRARY_PATH:-}"
    export MOONCAKE_CONFIG_PATH="${MOONCAKE_CONFIG_PATH:-/tmp/mooncake.json}"
    # ModelArts containers cannot reserve/mount host hugepages by default.
    if [[ "${FUSION_POOL_HUGEPAGE_ENABLED:-false}" == true ]]; then
        export MC_STORE_USE_HUGEPAGE=1 MC_STORE_HUGEPAGE_SIZE=2MB
        MAP_SIZE_BYTES=$(( (${FUSION_POOL_GLOBAL_SEGMENT_SIZE_GB:-64} + ${FUSION_POOL_LOCAL_BUFFER_SIZE_GB:-8}) * 1024 * 1024 * 1024 * 16 )) \
            bash "$CODE_PATH/tools/deploy/start_server/setup_hugetlbfs.sh" > "$LOG_PATH/fusion_pool_setup.log" 2>&1 || return 1
    elif [[ "${FUSION_POOL_HUGEPAGE_ENABLED:-false}" == false ]]; then
        # Mooncake may enable hugepages merely when this variable exists.
        unset MC_STORE_USE_HUGEPAGE MC_STORE_HUGEPAGE_SIZE
    else
        echo 'ERROR: FUSION_POOL_HUGEPAGE_ENABLED must be true or false' >&2; return 1
    fi
    KV_TRANSFER_CONFIG=$(python3 "$SCRIPT_DIR/fusion_pool_config.py" "$ROLE") || return 1
    export KV_TRANSFER_CONFIG
}

run_vllm() {
    local rank start devices pid status=0
    local -a pids=() cmd
    if [[ "${MA_DRY_RUN:-0}" != 1 ]]; then
        python3 - "$API_PORT" "$num_servers" <<'PY'
import socket, sys
ports = range(int(sys.argv[1]), int(sys.argv[1]) + int(sys.argv[2]))
sockets = []
try:
    for port in ports:
        sock = socket.socket()
        sockets.append(sock)
        sock.bind(("0.0.0.0", port))
finally:
    for sock in sockets:
        sock.close()
PY
    fi
    trap 'kill -TERM "${pids[@]}" 2>/dev/null || true' TERM INT
    for ((rank=0; rank<num_servers; rank++)); do
        export VLLM_DP_SIZE
        export VLLM_DP_RANK=$((SERVER_OFFSET + rank))
        export VLLM_DP_RANK_LOCAL="$VLLM_DP_RANK"
        export VLLM_DP_MASTER_IP="$MASTER_IP" VLLM_DP_MASTER_PORT="$MASTER_PORT"
        start=$((rank * tp))
        devices=$(seq -s, "$start" "$((start + tp - 1))")
        cmd=(vllm serve "$MODEL_PATH" --trust-remote-code
            --pipeline-parallel-size 1 --tensor-parallel-size "$tp"
            --gpu-memory-utilization "${gpu_util:-0.9}"
            --data-parallel-address "$MASTER_IP" --data-parallel-rpc-port "$MASTER_PORT"
            --port "$((API_PORT + rank))" --served-model-name "${MODEL_NAME:-pangu-v2-505b}"
            --load-format auto --block-size 128 --kv-transfer-config "$KV_TRANSFER_CONFIG")
        if [[ "$ROLE" == decode ]]; then
            cmd+=(--data-parallel-size "$VLLM_DP_SIZE" --data-parallel-rank "$VLLM_DP_RANK")
        fi
        if [[ "${OMNI_ENABLE_KV_EVENTS:-1}" == 1 ]]; then
            cmd+=(--kv-events-config "{\"enable_kv_cache_events\":true,\"publisher\":\"zmq\",\"topic\":\"kv-events\",\"endpoint\":\"tcp://*:$((API_PORT + 100))\"}")
        fi
        cmd+=("$@")
        printf '%q ' "${cmd[@]}" > "$LOG_PATH/command_${rank}.txt"
        printf '\n' >> "$LOG_PATH/command_${rank}.txt"
        if [[ "${MA_DRY_RUN:-0}" == 1 ]]; then
            cat "$LOG_PATH/command_${rank}.txt"
            continue
        fi
        ASCEND_RT_VISIBLE_DEVICES="$devices" "${cmd[@]}" > "$LOG_PATH/server_${rank}.log" 2>&1 &
        pids+=("$!")
    done
    if [[ "${MA_DRY_RUN:-0}" == 1 ]]; then return 0; fi
    # Propagate engine failures to the ModelArts job and stop sibling engines.
    wait -n "${pids[@]}" || status=$?
    kill -TERM "${pids[@]}" 2>/dev/null || true
    for pid in "${pids[@]}"; do wait "$pid" 2>/dev/null || true; done
    (( status != 0 )) || status=1
    return "$status"
}

# The Proxy and P0 are started by the same JSON unit, as in the validated bundle.
fusion_pool_proxy() {
export TORCH_DEVICE_BACKEND_AUTOLOAD=0
export MOONCAKE_METADATA_PORT="${MOONCAKE_METADATA_PORT:-8081}"
export MOONCAKE_RPC_PORT="${MOONCAKE_RPC_PORT:-16752}"
export MOONCAKE_METRICS_PORT="${MOONCAKE_METRICS_PORT:-$((MOONCAKE_METADATA_PORT + 100))}"
export MOONCAKE_KV_EVENTS_PORT="${MOONCAKE_KV_EVENTS_PORT:-5599}"
if [[ "${MA_DRY_RUN:-0}" != 1 ]]; then
if [[ "${FUSION_POOL_START_MASTER:-true}" == true ]]; then
    bash "$SCRIPT_DIR/fusion_pool_master.sh"
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
}

fusion_pool_main() {
    local role=$1
    shift
    if [[ "${ENABLE_OMNI_CACHE:-0}" != 0 || "${kv_offload:-false}" != false ]]; then
        echo 'ERROR: FusionPool cannot be combined with Omni Cache or CPU KV offload' >&2
        return 1
    fi
    init_runtime "$role" || return 1
    if [[ "$role" == proxy ]]; then
        fusion_pool_proxy "$@"
        return $?
    fi
    if [[ "$role" == prefill ]]; then
        export PREFILL_SERVER_LIST="$ASCEND_RT_VISIBLE_DEVICES"
        export OMNI_WORKER_CPUSET_MAP="${OMNI_WORKER_CPUSET_MAP:-auto}"
    else
        export DECODE_SERVER_LIST="$ASCEND_RT_VISIBLE_DEVICES"
        # One device list per D machine, without a hard-coded two-machine list.
        export DECODE_DATA_PARALLEL_SIZE
        DECODE_DATA_PARALLEL_SIZE=$(python3 -c 'import os; print(",".join([os.environ["DECODE_SERVER_LIST"]] * len(os.environ["LOCAL_DECODE_SERVER_IP_LIST"].split(","))))') || return 1
    fi
    prepare_pool || return 1
    run_vllm "$@"
}
