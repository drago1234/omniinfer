export PROXY_NODE_PORT=7710

export MODEL_NAME=pangu_ultra_moe
export npu=16                      # Local NPU count per pod
export PREFILL_INSTANCE_NUM=1      # Number of prefill groups in rank table
export DECODE_INSTANCE_NUM=1       # Number of decode groups in rank table
export dtype=bfloat16

# Common Runtime & Performance
export ASCEND_GLOBAL_LOG_LEVEL=3
export TASK_QUEUE_ENABLE=1
export PYTORCH_NPU_ALLOC_CONF=expandable_segments:True
export HCCL_CONNECT_TIMEOUT=1800
export HCCL_EXEC_TIMEOUT=1800
export HCCL_OP_RETRY_ENABLE="L0:0,L1:0,L2:0"
export HCCL_DETERMINISTIC=STRICT
export LCCL_DETERMINISTIC=TRUE
export CLOSE_MATMUL_K_SHIFT=0
export HCCL_OP_EXPANSION_MODE=AI_CPU
export VLLM_ENABLE_MC2=1
export VLLM_SERVER_DEV_MODE=1

# Patch & Model Plugins
export OMNI_NPU_PATCHES_DIR=pangu_v2_hybrid
export OMNI_NPU_VLLM_PATCHES=ALL
export VLLM_PLUGINS=omni-npu,omni_npu_patches,omni_custom_models

export MODEL_PATH="/mnt/bucket-910c-6055/lijun/505B/index_cache/iter_0009525"
export ENABLE_OMNI_CACHE=1
export ENABLE_HOST_MAPPING=0
export DISABLE_GATHER_SELECTION=1
export OMNI_CACHE_PREFILL_MMAP_FILE=omni_cache_p
export OMNI_CACHE_DECODE_MMAP_FILE=omni_cache_d
export OMNI_CACHE_LAYER_BYTES=13958643712
export MAP_SIZE_BYTES=251255586816
export NUM_DIE_PER_MACH=16
export BASE_PORT=16077
export ZMQ_BASE_PORT=16555
export OMNI_CACHE_MLA_SWA_DEBUG=1
export OMNI_CACHE_PREFILL_NUM_GPU_BLOCKS_OVERRIDE=77439
export HYBRID_ATTN_GROUP_SIZE=18

export BASE_API_PORT=9000
export MASTER_PORT=8000
export gpu_util=0.65
export max_model_len=163840
export max_num_seqs=32
export max_num_batched_tokens=65536
export HCCL_BUFFSIZE=200
export CUSTOM_MODEL_CONFIG_PATH="high_throughout/openpangu_v2/openpangu_v2_505b_bf16_a3_1p1d_precision_consistency_p.json"
export kv_transfer_config='{"kv_connector":"OmniCacheConnector","kv_role":"kv_consumer","kv_rank":1,"kv_parallel_size":33}'

bash /mnt/bucket-910c-6055/o00961519/model_arts_scripts/proxy.sh \
  --omni-proxy-pd-policy sequential \
  --omni-proxy-max-batch-num-token 100000 \
  --omni-proxy-prefill-max-num-seqs 32 \
  --omni-proxy-decode-max-num-seqs 8 &&
bash /mnt/bucket-910c-6055/o00961519/model_arts_scripts/prefill.sh \
  --no-disable-hybrid-kv-cache-manager \
  --enforce-eager \
  --enable-expert-parallel \
  --enable-prefix-caching \
  --no-async-scheduling \
  --reasoning-parser pangu \
  --enable-auto-tool-choice \
  --tool-call-parser pangu \
  --speculative-config '{"method":"mtp","num_speculative_tokens":3}'