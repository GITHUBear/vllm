python ./run_math.py \
--dataset_path ./data/aime24.jsonl \
--save_path ./results/output_full.jsonl \
--model_path deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
--eval_batch_size 1 \
--enforce_eager \
--enable_chunked_prefill