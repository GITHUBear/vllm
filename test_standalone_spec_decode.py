import os
from vllm import LLM, SamplingParams

os.environ["VLLM_USE_V1"] = "0"

prompts = [
    "The future of AI is",
    "Hello, World! "
]
sampling_params = SamplingParams(temperature=0)

llm = LLM(
    model="Qwen/Qwen2.5-7B-Instruct",
    tensor_parallel_size=4,
    speculative_config={
        "method": "standalone",
        "kv_compress_recover_rate": 0.8,
        "num_speculative_tokens": 4,
    },
)

outputs = llm.generate(prompts, sampling_params)

for output in outputs:
    prompt = output.prompt
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")