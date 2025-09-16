# Full Attention
# VLLM_ATTENTION_BACKEND="FLASH_ATTN" VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_USE_V1=0 vllm serve Qwen/QwQ-32B --tensor-parallel-size 8 --no-enable-chunked-prefill --enforce-eager
# DCA
# VLLM_ATTENTION_BACKEND="DUAL_CHUNK_FLASH_ATTN" VLLM_DCA_RECOVER_RATE=0.9 VLLM_USE_V1=0 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve Qwen/QwQ-32B --tensor-parallel-size 8 --no-enable-chunked-prefill --enforce-eager
# XAttn
# VLLM_ATTENTION_BACKEND="FLASH_ATTN" VLLM_SKIP_DCA_CONFIG=1 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_USE_V1=0 VLLM_FA_SPARSE_PREFILL=1 vllm serve Qwen/QwQ-32B --tensor-parallel-size 8 --no-enable-chunked-prefill --enforce-eager
# FlexPrefill
# VLLM_ATTENTION_BACKEND="FLASH_ATTN" VLLM_SKIP_DCA_CONFIG=1 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_USE_V1=0 VLLM_FA_SPARSE_PREFILL=3 vllm serve Qwen/QwQ-32B --tensor-parallel-size 8 --enforce-eager --no-enable-chunked-prefill
# SpargeAttn
# VLLM_ATTENTION_BACKEND="FLASH_ATTN" VLLM_SKIP_DCA_CONFIG=1 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_USE_V1=0 VLLM_FA_SPARSE_PREFILL=4 vllm serve Qwen/QwQ-32B --tensor-parallel-size 8 --enforce-eager --no-enable-chunked-prefill
VLLM_ATTENTION_BACKEND="FLASH_ATTN" VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 VLLM_USE_V1=0 vllm serve Qwen/QwQ-32B --tensor-parallel-size 8 --no-enable-chunked-prefill --enforce-eager --enable-pooling --pooling-blk-size 2