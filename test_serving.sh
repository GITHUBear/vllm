python benchmarks/benchmark_serving.py \
  --backend vllm \
  --model deepseek-ai/DeepSeek-R1-Distill-Qwen-14B \
  --endpoint /v1/completions \
  --dataset-name random \
  --random-input-len 256 \
  --random-output-len 65536 \
  --num-prompts 10 \
  --max-concurrency 10 \
  --ignore-eos