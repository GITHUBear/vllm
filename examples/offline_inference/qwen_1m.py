# SPDX-License-Identifier: Apache-2.0
import os
from urllib.request import urlopen

from vllm import LLM, SamplingParams

os.environ["VLLM_ATTENTION_BACKEND"] = "DUAL_CHUNK_FLASH_ATTN"
os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
os.environ["VLLM_USE_V1"] = "0"
os.environ["VLLM_DCA_RECOVER_RATE"] = "0.9"
# os.environ["CUDA_VISIBLE_DEVICES"]="4,5,6,7"

# os.environ["VLLM_FA_SPARSE_PREFILL"] = "1"
# os.environ["VLLM_ENABLE_LAST_ATTN_MAP_DUMP"] = "1"
# os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
# os.environ["VLLM_SKIP_DCA_CONFIG"] = "1"
# os.environ["VLLM_USE_V1"] = "0"


def load_prompt() -> str:
    # Test cases with various lengths can be found at:
    #
    # https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-1M/test-data/64k.txt
    # https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-1M/test-data/200k.txt
    # https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-1M/test-data/600k.txt
    # https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-1M/test-data/1m.txt

    with urlopen(
            # "https://qianwen-res.oss-cn-beijing.aliyuncs.com"
            # "/Qwen2.5-1M/test-data/600k.txt",
            "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-1M/test-data/200k.txt",
            # "https://qianwen-res.oss-cn-beijing.aliyuncs.com/Qwen2.5-1M/test-data/64k.txt",
            timeout=5) as response:
        prompt = response.read().decode('utf-8')
    return prompt


# Processing the prompt.
def process_requests(llm: LLM, prompts: list[str]) -> None:
    # Create a sampling params object.
    sampling_params = SamplingParams(
        temperature=0,
        top_p=0.8,
        top_k=20,
        repetition_penalty=1.05,
        detokenize=True,
        max_tokens=256,
    )
    # Generate texts from the prompts.
    outputs = llm.generate(prompts, sampling_params)
    # Print the outputs.
    for output in outputs:
        prompt_token_ids = output.prompt_token_ids
        generated_text = output.outputs[0].text
        print(f"Prompt length: {len(prompt_token_ids)}, "
              f"Generated text: {generated_text!r}")


# Create an LLM.
def initialize_engine() -> LLM:
    # llm = LLM(model="Qwen/Qwen2.5-7B-Instruct-1M",
    #           max_model_len=402320,
    #           tensor_parallel_size=4,
    #           enforce_eager=True,
    #           enable_chunked_prefill=True,
    #           max_num_batched_tokens=402320,
    #           max_num_seqs=1)
    llm = LLM(model="Qwen/Qwen2.5-7B-Instruct-1M",
              max_model_len=1010000,
              tensor_parallel_size=4,
              enforce_eager=True,
              enable_chunked_prefill=True,
              max_num_batched_tokens=131072,
              max_num_seqs=1,
            #   speculative_config={
            #     "method": "standalone",
            #     "block_sparse_mode": True,
            #     "num_speculative_tokens": 4,
            #   }
              )
    return llm


def main():
    llm = initialize_engine()
    # process_requests(llm, ["Hello, world!"])
    prompt = load_prompt()
    process_requests(llm, [prompt])


if __name__ == '__main__':
    main()
