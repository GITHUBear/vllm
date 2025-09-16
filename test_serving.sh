python benchmarks/benchmark_serving.py \
  --backend vllm \
  --model Qwen/QwQ-32B \
  --endpoint /v1/completions \
  --dataset-name random \
  --random-input-len 32768 \
  --random-output-len 1 \
  --num-prompts 10 \
  --max-concurrency 10 \
  --ignore-eos