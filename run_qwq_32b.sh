# Full Attention
# VLLM_ATTENTION_BACKEND="FLASH_ATTN" VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_USE_V1=0 vllm serve Qwen/QwQ-32B --tensor-parallel-size 8 --no-enable-chunked-prefill --enforce-eager
# DCA
# VLLM_ATTENTION_BACKEND="DUAL_CHUNK_FLASH_ATTN" VLLM_DCA_RECOVER_RATE=0.9 VLLM_USE_V1=0 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve Qwen/QwQ-32B --tensor-parallel-size 8 --no-enable-chunked-prefill --enforce-eager
# XAttn
# VLLM_ATTENTION_BACKEND="FLASH_ATTN" VLLM_SKIP_DCA_CONFIG=1 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_USE_V1=0 VLLM_FA_SPARSE_PREFILL=1 vllm serve Qwen/QwQ-32B --tensor-parallel-size 8 --no-enable-chunked-prefill --enforce-eager
# FlexPrefill
# VLLM_ATTENTION_BACKEND="FLASH_ATTN" VLLM_SKIP_DCA_CONFIG=1 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_USE_V1=0 VLLM_FA_SPARSE_PREFILL=3 vllm serve Qwen/QwQ-32B --tensor-parallel-size 8 --enforce-eager --no-enable-chunked-prefill
# SpargeAttn
# VLLM_ATTENTION_BACKEND="FLASH_ATTN" VLLM_SKIP_DCA_CONFIG=1 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_USE_V1=0 VLLM_FA_SPARSE_PREFILL=4 vllm serve /data/shanhaikang.shk/modelscope/qwq32b --tensor-parallel-size 8 --enforce-eager --no-enable-chunked-prefill
VLLM_ATTENTION_BACKEND="FLASH_ATTN" VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_USE_V1=0 vllm serve /data/shanhaikang.shk/modelscope/qwq32b --tensor-parallel-size 8 --no-enable-chunked-prefill --enforce-eager #--enable-pooling --pooling-blk-size 2
# VLLM_BLEND_METAFILE_PATH="/data/shanhaikang.shk/vllm/cacheblend/meta.hdf5" VLLM_BLEND_KVCACHE_DIR="/data/shanhaikang.shk/vllm/cacheblend" VLLM_ATTENTION_BACKEND="FLASH_ATTN" VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_USE_V1=0 vllm serve /data/shanhaikang.shk/modelscope/qwq32b --tensor-parallel-size 8 --no-enable-chunked-prefill --enforce-eager --enable-blend-prepare
# VLLM_BLEND_METAFILE_PATH="/data/shanhaikang.shk/vllm/cacheblend/meta.hdf5" VLLM_BLEND_KVCACHE_DIR="/data/shanhaikang.shk/vllm/cacheblend" VLLM_ATTENTION_BACKEND="FLASH_ATTN" VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_USE_V1=0 vllm serve /data/shanhaikang.shk/modelscope/qwq32b --tensor-parallel-size 8 --no-enable-chunked-prefill --enforce-eager --max-model-len 66000 --gpu-memory-utilization 0.8 --enable-cache-blend 