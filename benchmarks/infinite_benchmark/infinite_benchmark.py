# Copyright (c) 2024-2025 Microsoft
# Licensed under The MIT License [see LICENSE for details]

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, List, Tuple

import torch
from args import parse_args
from compute_scores import compute_scores
from eval_utils import (
    DATA_NAME_TO_MAX_NEW_TOKENS,
    check_benchmark_availability,
    create_prompt,
    dump_jsonl,
    get_answer,
    load_data,
)
from torch import Tensor
from tqdm import tqdm
from transformers import (
    AutoConfig,
    AutoModelForCausalLM,
    AutoTokenizer,
    GenerationConfig,
    LlamaForCausalLM,
)
from transformers.cache_utils import SinkCache
from transformers.modeling_outputs import BaseModelOutputWithPast
from vllm import LLM, SamplingParams

# sampling_params = SamplingParams(temperature=0.8, top_p=0.95)
def truncate_input(input: list, max_length: int, manner="middle"):
    if max_length < 0:
        return input
    if len(input) <= max_length:
        return input
    if manner == "middle":
        split = max_length // 2
        return input[0:split] + input[-split:]
    else:
        return None


def truncate_by_tokens(input, tok, max_tokens, manner: str = "middle"):
    tokens = tok.encode(input)
    len_before = len(tokens)
    print(f"# tokens before: {len_before}")
    tokens = truncate_input(tokens, max_length=max_tokens, manner=manner)
    len_after = len(tokens)  # type: ignore
    print(f"# tokens after: {len_after}")
    assert len_after <= len_before
    assert len_after <= max_tokens or max_tokens < 0
    return tokens


def get_pred(
    model,
    tok: AutoTokenizer,
    input_text: str,
    max_input_length: int,
    verbose: bool = False,
    generation_config: GenerationConfig = None,
) -> str:
    """
    Truncate down to 128k then make inference.
    """
    input_tokens = truncate_by_tokens(input_text, tok, max_input_length)
    if verbose:
        print("# tokens:", len(input_tokens))
        print("=============== Input ===============")
        print(tok.decode(input_tokens[:200]))
        print("...")
        print(tok.decode(input_tokens[-200:]))
        print("=====================================")
    if len(input_tokens) != 1:
        input_tokens = [input_tokens]
    outputs = model.generate(
        prompt_token_ids=input_tokens,
        sampling_params=generation_config,
    )
    output = outputs[0].outputs[0].text
    output = output.strip()
    # print(input_text[:5000], input_text[-5000:])
    print("Chunked generation:", output)
    return output


def load_model(
    model_name: str,
    trust_remote_code: bool = False,
    tensor_parallel_size: int = 1,
    max_model_len: int = 1048576,
    max_num_batched_tokens: int = 131072,
    enforce_eager: bool = False,
    enable_chunked_prefill: bool = False,
    enable_dca: bool = False,
    sparse_prefill_type: str = None,
    max_num_seqs: int = None,
):
    tok = AutoTokenizer.from_pretrained(
        model_name, resume_download=None, trust_remote_code=trust_remote_code
    )
    tok.pad_token = tok.eos_token

    if enable_dca:
        os.environ["VLLM_ATTENTION_BACKEND"] = "DUAL_CHUNK_FLASH_ATTN"
        os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
        llm = LLM(
            model=model_name,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=enforce_eager,
            enable_chunked_prefill=enable_chunked_prefill,
        )
    else:
        os.environ["VLLM_SKIP_DCA_CONFIG"] = "1"
        os.environ["VLLM_ALLOW_LONG_MAX_MODEL_LEN"] = "1"
        os.environ["VLLM_USE_V1"] = "0"

        if sparse_prefill_type is not None:
            os.environ["VLLM_FA_SPARSE_PREFILL"] = sparse_prefill_type
            print(f"================ SPARSE PREFILL ENABLED: {sparse_prefill_type} ================")

        llm = LLM(
            model=model_name,
            max_num_seqs=max_num_seqs,
            max_num_batched_tokens=max_num_batched_tokens,
            max_model_len=max_model_len,
            tensor_parallel_size=tensor_parallel_size,
            enforce_eager=enforce_eager,
            enable_chunked_prefill=enable_chunked_prefill,
        )

    print("Model and tokenizer loaded.")
    return llm, tok

def gen_test_tag(enable_dca: bool, sparse_prefill_type):
    if enable_dca:
        return "DCA"
    if sparse_prefill_type is None:
        return "woDCA"
    if sparse_prefill_type == "1":
        return "XATTN"
    if sparse_prefill_type == "3":
        return "FLEX_PREFILL"
    if sparse_prefill_type == "4":
        return "SPARGE_ATTN"
    raise ValueError(f"Invalid sparse prefill type")

if __name__ == "__main__":
    args = parse_args()

    check_benchmark_availability(args.data_dir)
    model_name = args.model_name_or_path
    max_seq_length = args.max_seq_length
    real_model_name = model_name.split("/")[-1]
    data_name = args.task

    if "," in data_name:
        data_names = data_name.split(",")
    else:
        data_names = [data_name]

    # Model
    model, tok = load_model(
        model_name,
        trust_remote_code=args.trust_remote_code,
        tensor_parallel_size=args.tensor_parallel_size,
        max_model_len=max_seq_length,
        max_num_batched_tokens=args.max_num_batched_tokens,
        enforce_eager=args.enforce_eager,
        enable_chunked_prefill=args.enable_chunked_prefill,
        enable_dca=args.enable_dca,
        sparse_prefill_type=args.sparse_prefill_type,
        max_num_seqs=args.max_num_seqs,
    )
    results = {}

    for data_name in data_names:
        max_new_tokens = DATA_NAME_TO_MAX_NEW_TOKENS[data_name]
        if max_new_tokens >= max_seq_length:
            max_new_tokens = 500

        generation_config = SamplingParams(
            temperature=0,
            max_tokens=max_new_tokens,
        )
        
        # Data
        tag = gen_test_tag(args.enable_dca, sparse_prefill_type=args.sparse_prefill_type)
        result_dir = Path(args.output_dir, f"{real_model_name}_{tag}")
        result_dir.mkdir(exist_ok=True, parents=True)
        output_path = result_dir / f"prediction_{data_name}.jsonl"
        examples = load_data(data_name, data_dir=args.data_dir)

        if args.num_eval_examples != -1:
            num_eval_examples = min(args.num_eval_examples, len(examples))
            examples = examples[:num_eval_examples]

        preds = []
        print("==== Evaluation ====")
        print(f"# examples: {len(examples)}")
        print(f"Num eval examples: {args.num_eval_examples}")
        print(f"Verbose: {args.verbose}")
        print(f"Max new tokens: {max_new_tokens}")
        print(f"examples len: {len(examples)} {data_name} {args.data_dir}")

        if os.path.exists(output_path) and not args.rewrite:
            print(f"Output file {output_path} exists. Loading from file.")
            compute_scores(output_path, data_name, real_model_name, max_seq_length)
            with open(output_path) as f:
                preds = [json.loads(ii) for ii in f.readlines()]

        for i, eg in tqdm(enumerate(examples)):
            if i < args.start_example_id or i < len(preds):
                continue
            input_text = create_prompt(eg, data_name, real_model_name, args.data_dir)
            ground_truth = get_answer(eg, data_name)

            msgs = [dict(role="system", content=input_text)]
            input_text = tok.apply_chat_template(
                msgs, add_generation_prompt=True, tokenize=False
            )
            pred = get_pred(
                model,
                tok,
                input_text,
                max_input_length=max_seq_length - max_new_tokens,
                verbose=args.verbose,
                generation_config=generation_config,
            )
            print("Ground Truth", get_answer(eg, data_name))
            if args.verbose:
                print(pred)
            preds.append(
                {
                    "id": i,
                    "prediction": pred,
                    "ground_truth": get_answer(eg, data_name),
                }
            )
            dump_jsonl(preds, output_path)
            torch.cuda.empty_cache()

        result_file_path = f"{real_model_name}"
        score = compute_scores(output_path, data_name, result_file_path)
        results[data_name] = score

    print("==== Results ====")
    print(json.dumps(results, indent=2))
