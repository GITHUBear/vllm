# VLLM_USE_V1=0 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --tensor-parallel-size 8 --enable-chunked-prefill --max-num-batched-tokens 131072 --max_model_len 1000000 --enforce-eager

# VLLM_FA_DECODE_RECOVER_RATE=0.9 VLLM_USE_V1=0 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --tensor-parallel-size 8 --enable-chunked-prefill --max-num-batched-tokens 131072 --max_model_len 1000000 --enforce-eager

# VLLM_FA_DUMP_DECODE_ATTN=1 VLLM_FA_DUMP_DECODE_STEP=5000 VLLM_USE_V1=0 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --tensor-parallel-size 8 --enable-chunked-prefill --max-num-batched-tokens 131072 --max_model_len 1000000 --enforce-eager

# VLLM_USE_V1=0 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --tensor-parallel-size 8 --enable-chunked-prefill --max-num-batched-tokens 131072 --max_model_len 1000000 --enforce-eager --speculative_config '{"method":"standalone", "kv_compress_recover_rate":0.8, "num_speculative_tokens": 16}'

VLLM_USE_V1=0 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --tensor-parallel-size 8 --enable-chunked-prefill --max-num-batched-tokens 131072 --max_model_len 1000000 --enforce-eager

# VLLM_USE_V1=0 VLLM_ALLOW_LONG_MAX_MODEL_LEN=1 vllm serve deepseek-ai/DeepSeek-R1-Distill-Qwen-14B --tensor-parallel-size 8 --enable-chunked-prefill --max-num-batched-tokens 131072 --max_model_len 1000000 --enforce-eager --speculative_config '{"method":"standalone", "block_sparse_mode":true, "num_speculative_tokens": 4}'