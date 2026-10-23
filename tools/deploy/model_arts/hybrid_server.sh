#!/usr/bin/env bash
# 通用混部 vLLM 服务启动脚本。
# 模型及部署参数通过部署 JSON 的 envs 传入；cmd 中脚本路径后的命令行参数将原样追加到 vLLM serve 命令末尾。

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 解析cmd指令
USER_EXTRA_ARGS=("$@")

required_vars=(
    MODEL_NAME
    MODEL_PATH
    PORT
    ASCEND_RT_VISIBLE_DEVICES
    TP_SIZE
    DP_SIZE
    MAX_MODEL_LEN
    MAX_NUM_BATCHED_TOKENS
    MAX_NUM_SEQS
    GPU_MEMORY_UTILIZATION
    VLLM_PLUGINS
)
for name in "${required_vars[@]}"; do
    if [[ -z "${!name:-}" ]]; then
        echo "ERROR: required environment variable is not set: $name" >&2
        exit 2
    fi
done

HOST="${HOST:-0.0.0.0}"
DTYPE="${DTYPE:-bfloat16}"
DISTRIBUTED_EXECUTOR_BACKEND="${DISTRIBUTED_EXECUTOR_BACKEND:-mp}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-1}"
ADD_DATA_PARALLEL_ARG="${ADD_DATA_PARALLEL_ARG:-1}"
PREFIX_CACHING="${PREFIX_CACHING:-auto}"
CHUNKED_PREFILL="${CHUNKED_PREFILL:-auto}"
EXECUTION_MODE="${EXECUTION_MODE:-auto}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-0}"
ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-0}"

export ASCEND_RT_VISIBLE_DEVICES
export ASCEND_GLOBAL_LOG_LEVEL="${ASCEND_GLOBAL_LOG_LEVEL:-3}"
export HCCL_BUFFSIZE="${HCCL_BUFFSIZE:-500}"
export HCCL_CONNECT_TIMEOUT="${HCCL_CONNECT_TIMEOUT:-1800}"
export HCCL_OP_EXPANSION_MODE="${HCCL_OP_EXPANSION_MODE:-AIV}"
export PYTORCH_NPU_ALLOC_CONF="${PYTORCH_NPU_ALLOC_CONF:-expandable_segments:True}"
export VLLM_DISABLE_COMPILE_CACHE="${VLLM_DISABLE_COMPILE_CACHE:-1}"
export VLLM_DISABLE_PYNCCL="${VLLM_DISABLE_PYNCCL:-1}"
export VLLM_PLUGINS

if [[ "${CLEAN_COMPILE_CACHE:-0}" == "1" ]]; then
    cache_dir="${VLLM_COMPILE_CACHE_DIR:-/root/.cache/vllm/torch_compile_cache}"
    if [[ "$cache_dir" != /root/.cache/vllm/* && "$cache_dir" != /tmp/* ]]; then
        echo "ERROR: refusing to remove unsafe cache path: $cache_dir" >&2
        exit 2
    fi
    rm -rf -- "$cache_dir"
fi

cmd=(
    vllm serve "$MODEL_PATH"
    --served-model-name "$MODEL_NAME"
    --host "$HOST"
    --port "$PORT"
    --dtype "$DTYPE"
    --max-model-len "$MAX_MODEL_LEN"
    --max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS"
    --max-num-seqs "$MAX_NUM_SEQS"
    --distributed-executor-backend "$DISTRIBUTED_EXECUTOR_BACKEND"
    --gpu-memory-utilization "$GPU_MEMORY_UTILIZATION"
    --tensor-parallel-size "$TP_SIZE"
)

if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
    cmd+=(--trust-remote-code)
fi
if [[ "$ADD_DATA_PARALLEL_ARG" == "1" ]]; then
    cmd+=(--data-parallel-size "$DP_SIZE")
fi
if [[ -n "${ALLOWED_LOCAL_MEDIA_PATH:-}" ]]; then
    cmd+=(--allowed-local-media-path "$ALLOWED_LOCAL_MEDIA_PATH")
fi
if [[ -n "${MEDIA_IO_KWARGS:-}" ]]; then
    cmd+=(--media-io-kwargs "$MEDIA_IO_KWARGS")
fi
if [[ -n "${MM_PROCESSOR_KWARGS:-}" ]]; then
    cmd+=(--mm-processor-kwargs "$MM_PROCESSOR_KWARGS")
fi

case "$PREFIX_CACHING" in
    enable) cmd+=(--enable-prefix-caching) ;;
    disable) cmd+=(--no-enable-prefix-caching) ;;
    auto) ;;
    *) echo "ERROR: PREFIX_CACHING must be enable, disable, or auto" >&2; exit 2 ;;
esac
case "$CHUNKED_PREFILL" in
    enable) cmd+=(--enable-chunked-prefill) ;;
    disable) cmd+=(--no-enable-chunked-prefill) ;;
    auto) ;;
    *) echo "ERROR: CHUNKED_PREFILL must be enable, disable, or auto" >&2; exit 2 ;;
esac
case "$EXECUTION_MODE" in
    eager) cmd+=(--enforce-eager) ;;
    graph)
        if [[ -z "${COMPILATION_CONFIG:-}" ]]; then
            echo "ERROR: COMPILATION_CONFIG is required when EXECUTION_MODE=graph" >&2
            exit 2
        fi
        cmd+=(--compilation-config "$COMPILATION_CONFIG")
        ;;
    auto) ;;
    *) echo "ERROR: EXECUTION_MODE must be eager, graph, or auto" >&2; exit 2 ;;
esac

if [[ "$ASYNC_SCHEDULING" == "1" ]]; then
    cmd+=(--async-scheduling)
fi
if [[ "$ENABLE_EXPERT_PARALLEL" == "1" ]]; then
    cmd+=(--enable-expert-parallel)
fi
if [[ "${NO_DISABLE_HYBRID_KV_CACHE_MANAGER:-0}" == "1" ]]; then
    cmd+=(--no-disable-hybrid-kv-cache-manager)
fi
if [[ "${TRUST_REQUEST_CHAT_TEMPLATE:-0}" == "1" ]]; then
    cmd+=(--trust-request-chat-template)
fi
if [[ "${DISABLE_CHUNKED_MM_INPUT:-0}" == "1" ]]; then
    cmd+=(--disable-chunked-mm-input)
fi

optional_value_args=(
    API_SERVER_COUNT:--api-server-count
    ADDITIONAL_CONFIG:--additional-config
    MM_PROCESSOR_CACHE_TYPE:--mm-processor-cache-type
    MM_PROCESSOR_CACHE_GB:--mm-processor-cache-gb
    MM_SHM_CACHE_MAX_OBJECT_SIZE_MB:--mm-shm-cache-max-object-size-mb
    LIMIT_MM_PER_PROMPT:--limit-mm-per-prompt
    BLOCK_SIZE:--block-size
    KV_CACHE_DTYPE:--kv-cache-dtype
    SPECULATIVE_CONFIG:--speculative-config
    LOAD_FORMAT:--load-format
    HF_OVERRIDES:--hf-overrides
    TOOL_CALL_PARSER:--tool-call-parser
)
for spec in "${optional_value_args[@]}"; do
    name="${spec%%:*}"
    flag="${spec#*:}"
    if [[ -n "${!name:-}" ]]; then
        cmd+=("$flag" "${!name}")
    fi
done

# 追加 JSON cmd 中传入的额外 vLLM 参数。
cmd+=("${USER_EXTRA_ARGS[@]}")


# 日志存储
start_time="$(date +'%Y-%m-%d %H:%M:%S')"
log_timestamp="$(date +'%Y%m%d_%H%M%S')"
safe_model_name="${MODEL_NAME//[^a-zA-Z0-9._-]/_}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/hybrid_${safe_model_name}_port${PORT}_${log_timestamp}.log}"

mkdir -p "$(dirname "$LOG_FILE")"
touch "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1
exec "${cmd[@]}"
