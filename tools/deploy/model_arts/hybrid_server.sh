#!/usr/bin/env bash
# 通用混部 vLLM 服务启动脚本。
# 模型及部署参数通过部署 JSON 的 envs 传入；cmd 中脚本路径后的命令行参数将原样追加到 vLLM serve 命令末尾。

set -Eeuo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

# 解析cmd指令
USER_EXTRA_ARGS=("$@")

required_vars=(
    MODEL_PATH
)
for name in "${required_vars[@]}"; do
    if [[ -z "${!name:-}" ]]; then
        echo "ERROR: required environment variable is not set: $name" >&2
        exit 2
    fi
done

HOST="${HOST:-}"
DTYPE="${DTYPE:-}"
DISTRIBUTED_EXECUTOR_BACKEND="${DISTRIBUTED_EXECUTOR_BACKEND:-}"
TRUST_REMOTE_CODE="${TRUST_REMOTE_CODE:-0}"
PREFIX_CACHING="${PREFIX_CACHING:-auto}"
CHUNKED_PREFILL="${CHUNKED_PREFILL:-auto}"
ASYNC_SCHEDULING="${ASYNC_SCHEDULING:-0}"
ENABLE_EXPERT_PARALLEL="${ENABLE_EXPERT_PARALLEL:-0}"

ENFORCE_EAGER="${ENFORCE_EAGER:-0}"

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
)

if [[ -n "${MODEL_NAME:-}" ]]; then
    cmd+=(--served-model-name "$MODEL_NAME")
fi
if [[ -n "${ALLOWED_LOCAL_MEDIA_PATH:-}" ]]; then
    cmd+=(--allowed-local-media-path "$ALLOWED_LOCAL_MEDIA_PATH")
fi
if [[ -n "${MEDIA_IO_KWARGS:-}" ]]; then
    cmd+=(--media-io-kwargs "$MEDIA_IO_KWARGS")
fi
if [[ -n "$HOST" ]]; then
    cmd+=(--host "$HOST")
fi
if [[ -n "${PORT:-}" ]]; then
    cmd+=(--port "$PORT")
fi
if [[ -n "$DTYPE" ]]; then
    cmd+=(--dtype "$DTYPE")
fi
if [[ -n "${MAX_MODEL_LEN:-}" ]]; then
    cmd+=(--max-model-len "$MAX_MODEL_LEN")
fi
if [[ -n "${MAX_NUM_BATCHED_TOKENS:-}" ]]; then
    cmd+=(--max-num-batched-tokens "$MAX_NUM_BATCHED_TOKENS")
fi
if [[ -n "${MAX_NUM_SEQS:-}" ]]; then
    cmd+=(--max-num-seqs "$MAX_NUM_SEQS")
fi
if [[ -n "$DISTRIBUTED_EXECUTOR_BACKEND" ]]; then
    cmd+=(--distributed-executor-backend "$DISTRIBUTED_EXECUTOR_BACKEND")
fi
if [[ -n "${GPU_MEMORY_UTILIZATION:-}" ]]; then
    cmd+=(--gpu-memory-utilization "$GPU_MEMORY_UTILIZATION")
fi
if [[ -n "${TP_SIZE:-}" ]]; then
    cmd+=(--tensor-parallel-size "$TP_SIZE")
fi
if [[ "$TRUST_REMOTE_CODE" == "1" ]]; then
    cmd+=(--trust-remote-code)
fi
if [[ -n "${DP_SIZE:-}" ]]; then
    cmd+=(--data-parallel-size "$DP_SIZE")
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
if [[ "$ENFORCE_EAGER" == "1" ]]; then
    cmd+=(--enforce-eager)
elif [[ "$ENFORCE_EAGER" != "0" ]]; then
    echo "ERROR: ENFORCE_EAGER must be 0 or 1" >&2
    exit 2
fi

if [[ -n "${COMPILATION_CONFIG:-}" ]]; then
    if [[ "$COMPILATION_CONFIG" == *'${MAX_NUM_SEQS}'* ]]; then
        if [[ -z "${MAX_NUM_SEQS:-}" ]]; then
            echo "ERROR: MAX_NUM_SEQS is required when COMPILATION_CONFIG uses \${MAX_NUM_SEQS}" >&2
            exit 2
        fi
        COMPILATION_CONFIG="${COMPILATION_CONFIG//\$\{MAX_NUM_SEQS\}/${MAX_NUM_SEQS}}"
    fi
    cmd+=(--compilation-config "$COMPILATION_CONFIG")
fi

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
log_timestamp="$(date +'%Y%m%d_%H%M%S')"
log_model_name="${MODEL_NAME:-${MODEL_PATH##*/}}"
safe_model_name="${log_model_name//[^a-zA-Z0-9._-]/_}"
LOG_DIR="${LOG_DIR:-${SCRIPT_DIR}/logs}"
log_port="${PORT:-default}"
LOG_FILE="${LOG_FILE:-${LOG_DIR}/hybrid_${safe_model_name}_port${log_port}_${log_timestamp}.log}"

mkdir -p "$(dirname "$LOG_FILE")"
touch "$LOG_FILE"
exec > >(tee -a "$LOG_FILE") 2>&1
exec "${cmd[@]}"
