# flash_attn
# python ./needle_test.py --model_name Qwen/Qwen2.5-7B-Instruct-1M --max_length 500000 --min_length 10000 --trust_remote_code --enable_chunked_prefill --tensor_parallel_size 4
# DCA
# python ./needle_test.py --model_name Qwen/Qwen2.5-7B-Instruct-1M --max_length 500000 --min_length 10000 --trust_remote_code --enable_chunked_prefill --tensor_parallel_size 4 --enable_dca --enforce_eager
# X_attn
python ./needle_test.py --model_name Qwen/Qwen2.5-7B-Instruct-1M --max_length 500000 --min_length 10000 --trust_remote_code --enable_chunked_prefill --tensor_parallel_size 4 --enforce_eager --sparse_prefill_type 1 --run_name x_attn