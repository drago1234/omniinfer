#!/bin/bash
# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Huawei Technologies Co., Ltd. All Rights Reserved.
# Shared initialization for the fixed ModelArts 4P1D deployment.
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
        KV_TRANSFER_CONFIG=$(python3 "$SCRIPT_DIR/pool_config.py" "$ROLE" --dry-run) || return 1
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
    KV_TRANSFER_CONFIG=$(python3 "$SCRIPT_DIR/pool_config.py" "$ROLE") || return 1
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
