from typing import List

import torch

from vllm.config import DeviceConfig, ModelConfig, ParallelConfig, SpeculativeConfig
from vllm.utils import LayerBlockType, get_dtype_size

class SparseIndexBlockCacheEngine:
    def __init__(
        self,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        device_config: DeviceConfig,
        speculative_config: SpeculativeConfig,
    ) -> None:
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.device_config = device_config
        self.speculative_config = speculative_config

        self.num_attention_layers = model_config.get_num_layers_by_block_type(
            parallel_config, LayerBlockType.attention)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        self.sparse_index_num_gpu_blocks = speculative_config.sparse_index_num_gpu_blocks
        self.sparse_index_max_vertical_slash_topk = speculative_config.sparse_index_max_vertical_slash_topk

        assert self.sparse_index_num_gpu_blocks is not None
        self.dtype = torch.int32

        # slash & vertical index
        (self.block_count_gpu_cache, self.block_index_gpu_cache,
         self.column_count_gpu_cach, self.column_index_gpu_cache) = self._allocate_sparse_index_cache(
            device=self.device_config.device_type
        )

    def _allocate_sparse_index_cache(
        self,
        device: str,
    ) -> List[torch.Tensor]:
        block_count_cache_shape = (
            self.sparse_index_num_gpu_blocks,
            self.num_attention_layers,
            self.num_kv_heads,
        )
        block_index_cache_shape = (
            self.sparse_index_num_gpu_blocks,
            self.num_attention_layers,
            self.num_kv_heads,
            self.sparse_index_max_vertical_slash_topk,
        )
        column_count_cache_shape = (
            self.sparse_index_num_gpu_blocks,
            self.num_attention_layers,
            self.num_kv_heads,
        )
        column_index_cache_shape = (
            self.sparse_index_num_gpu_blocks,
            self.num_attention_layers,
            self.num_kv_heads,
            self.sparse_index_max_vertical_slash_topk,
        )
        block_count_gpu_cache = torch.zeros(
            block_count_cache_shape,
            dtype=self.dtype,
            device=device,
        )
        block_index_gpu_cache = torch.zeros(
            block_index_cache_shape,
            dtype=self.dtype,
            device=device,
        )
        column_count_gpu_cache = torch.zeros(
            column_count_cache_shape,
            dtype=self.dtype,
            device=device,
        )
        column_index_gpu_cache = torch.zeros(
            column_index_cache_shape,
            dtype=self.dtype,
            device=device,
        )
        return (
            block_count_gpu_cache, block_index_gpu_cache,
            column_count_gpu_cache, column_index_gpu_cache,
        )

    @staticmethod
    def get_cache_block_size(
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        speculative_config: SpeculativeConfig,
    ) -> int:
        assert speculative_config.sparse_index_max_vertical_slash_topk is not None
        num_heads = model_config.get_num_kv_heads(parallel_config)
        num_attention_layers = model_config.get_num_layers_by_block_type(
            parallel_config, LayerBlockType.attention)
        # TODO: Maybe we can use a small data type later.
        dtype = torch.int32

        total = num_attention_layers * num_heads * 2 * (
            1 + speculative_config.sparse_index_max_vertical_slash_topk)
        dtype_size = get_dtype_size(dtype)
        return dtype_size * total