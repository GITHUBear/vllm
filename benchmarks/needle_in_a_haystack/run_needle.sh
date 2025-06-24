# flash_attn
# python ./needle_test.py --model_name Qwen/Qwen2.5-7B-Instruct-1M --max_length 500000 --min_length 10000 --trust_remote_code --enable_chunked_prefill --tensor_parallel_size 4
# DCA
# python ./needle_test.py --model_name Qwen/Qwen2.5-7B-Instruct-1M --max_length 500000 --min_length 10000 --trust_remote_code --enable_chunked_prefill --tensor_parallel_size 4 --enable_dca --enforce_eager
# X_attn
# python ./needle_test.py --model_name Qwen/Qwen2.5-7B-Instruct-1M --max_length 500000 --min_length 10000 --trust_remote_code --enable_chunked_prefill --tensor_parallel_size 4 --enforce_eager --sparse_prefill_type 1 --run_name x_attn

# X_attn2
python ./needle_test.py --model_name Qwen/Qwen2.5-7B-Instruct-1M --max_length 400000 --min_length 10000 --trust_remote_code --enable_chunked_prefill --tensor_parallel_size 4 --enforce_eager --sparse_prefill_type 1 --run_name x_attn_fixed --max_model_len 1010000 --max_num_batched_tokens 131072 --max_num_seqs 1

# FlexPrefill
# CUDA_VISIBLE_DEVICES=4,5,6,7 python ./needle_test.py --model_name Qwen/Qwen2.5-7B-Instruct-1M --max_length 10000 --min_length 10000 --trust_remote_code --tensor_parallel_size 4 --enforce_eager --sparse_prefill_type 3 --run_name FlexPrefill --max_model_len 402320 --max_num_batched_tokens 402320 --max_num_seqs 1