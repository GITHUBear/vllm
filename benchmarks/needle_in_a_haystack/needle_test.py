# Copyright (c) 2024 Microsoft
# Licensed under The MIT License [see LICENSE for details]

import argparse
import os
from dataclasses import dataclass
from datetime import datetime

from needle_tools import LLMNeedleHaystackTester
from needle_viz import plot_needle_viz

@dataclass
class Config:
    # wget https://github.com/liyucheng09/LatestEval/releases/download/pg19/pg19_mini.jsonl
    haystack_file: str = "./data/pg19_mini.jsonl"  # Path to the haystack file
    model_name: str = "Qwen/Qwen2.5-7B-Instruct-1M"
    run_name: str = None  # Name of the run, used for the output file
    context_lengths_min: int = 30_000
    context_lengths_max: int = 100_000
    n_context_length_intervals: int = 15  # Number of intervals between min and max
    n_document_depth_intervals: int = 10  # position of the needle in the haystack
    n_rounds: int = 3
    seed: int = 42
    output_path: str = "./results"
    jobs: str = None
    # kv_cache_cpu: bool = False
    trust_remote_code: bool = False
    # kv_cache_cpu_device: str = "cpu"
    # kv_type: str = "dense"
    enable_dca: bool = False
    dca_recover_rate: float = None
    max_model_len: int = 1048576
    max_num_batched_tokens: int = 131072
    max_num_seqs: int = None
    tensor_parallel_size: int = 4
    enforce_eager: bool = False
    enable_chunked_prefill: bool = False
    sparse_prefill_type: str = None

    def __post_init__(self):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if not os.path.exists(self.output_path):
            os.makedirs(self.output_path)
        dca_tag = "DCA" if self.enable_dca else "woDCA"
        output_file = f"needle_res_{self.model_name.replace('/', '-')}_{self.run_name if self.run_name is not None else ''}_{dca_tag}_{self.jobs if self.jobs is not None else ''}_{timestamp}_{self.context_lengths_min}_{self.context_lengths_max}.json"
        self.output_file = os.path.join(self.output_path, output_file)


def main(
    model_name: str,
    run_name: str = None,
    output_path: str = "./results",
    rounds: int = 3,
    jobs: str = None,
    max_length: int = 100000,
    min_length: int = 1000,
    # kv_cache_cpu: bool = False,
    trust_remote_code: bool = False,
    # kv_cache_cpu_device: str = "cpu",
    # kv_type: str = "dense",
    enable_dca: bool = False,
    dca_recover_rate: float = None,
    max_model_len: int = 1048576,
    max_num_batched_tokens: int = 131072,
    max_num_seqs: int = None,
    tensor_parallel_size: int = 4,
    enforce_eager: bool = False,
    enable_chunked_prefill: bool = False,
    sparse_prefill_type = None,
):
    config = Config(
        model_name=model_name,
        run_name=run_name,
        output_path=output_path,
        n_rounds=rounds,
        jobs=jobs,
        context_lengths_min=min_length,
        context_lengths_max=max_length,
        # kv_cache_cpu=kv_cache_cpu,
        trust_remote_code=trust_remote_code,
        # kv_cache_cpu_device=kv_cache_cpu_device,
        # kv_type=kv_type,
        enable_dca=enable_dca,
        dca_recover_rate=dca_recover_rate,
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
        max_num_seqs=max_num_seqs,
        tensor_parallel_size=tensor_parallel_size,
        enforce_eager=enforce_eager,
        enable_chunked_prefill=enable_chunked_prefill,
        sparse_prefill_type=sparse_prefill_type,
    )
    # kwargs = {
    #     "max_model_len": 1048576,
    #     "tensor_parallel_size": 4,
    #     "enforce_eager": True,
    #     "enable_chunked_prefill": True,
    #     "max_num_batched_tokens": 131072,
    # }
    ht = LLMNeedleHaystackTester(config)
    ht.start_test()

    print("making plot...")
    plot_needle_viz(
        config.output_file,
        (
            config.model_name.replace("/", "-") + f"_{config.run_name}"
            if config.run_name is not None
            else ""
        ),
        "DCA" if enable_dca else "woDCA",
        config.context_lengths_min,
        config.context_lengths_max,
        output_path=config.output_path,
    )


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--model_name", type=str, required=True)
    args.add_argument("--run_name", type=str, default=None)
    args.add_argument("--output_path", type=str, default="./results")
    args.add_argument("--rounds", type=int, default=3)
    args.add_argument("--jobs", type=str, default=None)
    args.add_argument("--max_length", type=int, default=100000)
    args.add_argument("--min_length", type=int, default=1000)
    args.add_argument("--enable_dca", action="store_true")
    args.add_argument("--dca_recover_rate", type=float, default=None)
    # args.add_argument("--kv_cache_cpu", action="store_true")
    # args.add_argument("--kv_cache_cpu_device", type=str, default="cpu")
    args.add_argument("--trust_remote_code", action="store_true")
    args.add_argument("--max_model_len", type=int, default=1048576)
    args.add_argument("--max_num_batched_tokens", type=int, default=131072)
    args.add_argument("--max_num_seqs", type=int, default=None)
    args.add_argument("--tensor_parallel_size", type=int, default=4)
    args.add_argument("--enforce_eager", action="store_true")
    args.add_argument("--enable_chunked_prefill", action="store_true")
    args.add_argument("--sparse_prefill_type", type=str, default=None)
    args = args.parse_args()

    main(
        model_name=args.model_name,
        run_name=args.run_name,
        output_path=args.output_path,
        rounds=args.rounds,
        jobs=args.jobs,
        max_length=args.max_length,
        min_length=args.min_length,
        # kv_cache_cpu=args.kv_cache_cpu,
        trust_remote_code=args.trust_remote_code,
        # kv_cache_cpu_device=args.kv_cache_cpu_device,
        enable_dca=args.enable_dca,
        dca_recover_rate=args.dca_recover_rate,
        max_model_len=args.max_model_len,
        max_num_batched_tokens=args.max_num_batched_tokens,
        max_num_seqs=args.max_num_seqs,
        tensor_parallel_size=args.tensor_parallel_size,
        enforce_eager=args.enforce_eager,
        enable_chunked_prefill=args.enable_chunked_prefill,
        sparse_prefill_type=args.sparse_prefill_type,
    )
