import os
import json
import random
import argparse

import numpy as np
from tqdm import tqdm

import torch
from vllm import LLM, SamplingParams

dataset2key = {
    "gsm8k": ["question", "answer"],
    "aime24": ["question", "answer"],
    "math": ["problem", "answer"],
}

dataset2max_length = {
    "gsm8k": 8192,
    "aime24": 16384,
    "math": 8192,
}


def set_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.manual_seed_all(seed)


prompt_template = "You are given a math problem.\n\nProblem: {question}\n\n You need to solve the problem step by step. First, you need to provide the chain-of-thought, then provide the final answer.\n\n Provide the final answer in the format: Final answer:  \\boxed{{}}"

def main(
    dataset_path: str,
    dataset_name: str,
    result_path: str,
    model_path: str,
    max_length: int,
    eval_batch_size: int,
    # model_config
    max_model_len: int,
    max_num_batched_tokens: int,
    max_num_seqs: int,
    # 
    tensor_parallel_size: int,
    enforce_eager: bool = False,
    enable_chunked_prefill: bool = False,
    sparse_prefill_type = None,
):
    fout = open(result_path, "w")

    prompts = []
    test_data = []

    with open(dataset_path) as f:
        for index, line in enumerate(f):
            example = json.loads(line)
            question_key = dataset2key[dataset_name][0]

            question = example[question_key]
            example["question"] = question
            prompt = prompt_template.format(**example)

            example["prompt"] = prompt
            example["index"] = index
            prompts.append(prompt)
            test_data.append(example)

    if sparse_prefill_type is not None:
        os.environ["VLLM_USE_V1"] = "0"
        os.environ["VLLM_FA_SPARSE_PREFILL"] = sparse_prefill_type
    model = LLM(
        model=model_path,
        max_num_seqs=max_num_seqs,
        max_num_batched_tokens=max_num_batched_tokens,
        max_model_len=max_model_len,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=enforce_eager,
        enable_chunked_prefill=enable_chunked_prefill,
    )
    generation_config = SamplingParams(
        temperature=0,
        max_tokens=max_length
    )

    for i in tqdm(range(0, len(prompts), eval_batch_size)):
        batch_prompts = prompts[i : i + eval_batch_size]

        outs = model.generate(
            batch_prompts,
            sampling_params=generation_config,
        )

        for j in range(len(outs)):
            sample_idx = i + j
            test_data[sample_idx]["prompt"] = batch_prompts[j]
            test_data[sample_idx]["output"] = outs[j].outputs[0].text
            test_data[sample_idx]["prefill_tokens"] = len(outs[j].prompt_token_ids)
            test_data[sample_idx]["output_tokens"] = len(outs[j].outputs[0].token_ids)
            test_data[sample_idx]["total_tokens"] = len(outs[j].prompt_token_ids) + len(outs[j].outputs[0].token_ids)
            test_data[sample_idx]["sample_idx"] = sample_idx

            fout.write(json.dumps(test_data[sample_idx], ensure_ascii=False) + "\n")

    fout.close()

def parse_arguments():
    parser = argparse.ArgumentParser()

    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--dataset_path", type=str)
    parser.add_argument("--save_path", type=str)
    parser.add_argument("--model_path", type=str)
    parser.add_argument("--max_length", type=int, default=-1)
    parser.add_argument("--eval_batch_size", type=int, default=1)

    # parser.add_argument("--enable_dca", action="store_true")
    parser.add_argument("--max_model_len", type=int, default=None)
    parser.add_argument("--max_num_batched_tokens", type=int, default=131072)
    parser.add_argument("--max_num_seqs", type=int, default=None)
    parser.add_argument("--tensor_parallel_size", type=int, default=8)
    parser.add_argument("--enforce_eager", action="store_true", default=False)
    parser.add_argument("--enable_chunked_prefill", action="store_true", default=False)
    parser.add_argument("--sparse_prefill_type", type=str, default=None)

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    set_seed(args.seed)

    args.dataset_name = args.dataset_path.split("/")[-1].split(".")[0]
    if args.max_length == -1: args.max_length = dataset2max_length[args.dataset_name]

    main(
        args.dataset_path,
        args.dataset_name,
        args.save_path,
        args.model_path,
        args.max_length,
        args.eval_batch_size,
        args.max_model_len,
        args.max_num_batched_tokens,
        args.max_num_seqs,
        args.tensor_parallel_size,
        args.enforce_eager,
        args.enable_chunked_prefill,
        args.sparse_prefill_type
    )
