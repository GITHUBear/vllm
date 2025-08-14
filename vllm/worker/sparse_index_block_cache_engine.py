from typing import List

import torch

from vllm.config import DeviceConfig, ModelConfig, ParallelConfig, SpeculativeConfig, CacheConfig
from vllm.utils import LayerBlockType, get_dtype_size

class SparseIndexBlockCacheEngine:
    def __init__(
        self,
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        device_config: DeviceConfig,
        speculative_config: SpeculativeConfig,
    ) -> None:
        self.cache_config = cache_config
        self.model_config = model_config
        self.parallel_config = parallel_config
        self.device_config = device_config
        self.speculative_config = speculative_config

        self.num_attention_layers = model_config.get_num_layers_by_block_type(
            parallel_config, LayerBlockType.attention)
        self.num_kv_heads = model_config.get_num_kv_heads(parallel_config)
        self.num_heads = model_config.get_num_attention_heads(parallel_config)
        self.sparse_index_num_gpu_blocks = speculative_config.sparse_index_num_gpu_blocks
        self.sparse_index_max_vertical_slash_topk = speculative_config.sparse_index_max_vertical_slash_topk

        assert self.sparse_index_num_gpu_blocks is not None
        self.dtype = torch.int32

        # slash & vertical index
        (self.block_count_gpu_cache, self.block_index_gpu_cache,
         self.column_count_gpu_cache, self.column_index_gpu_cache) = self._allocate_sparse_index_cache(
            device=self.device_config.device_type
        )

    def _allocate_sparse_index_cache(
        self,
        device: str,
    ) -> List[torch.Tensor]:
        if not self.speculative_config.block_sparse_mode:
            block_count_gpu_cache = []
            block_index_gpu_cache = []
            column_count_gpu_cache = []
            column_index_gpu_cache = []

            block_count_cache_shape = (
                self.sparse_index_num_gpu_blocks,
                self.num_heads,
            )
            block_index_cache_shape = (
                self.sparse_index_num_gpu_blocks,
                self.num_heads,
                self.sparse_index_max_vertical_slash_topk,
            )
            column_count_cache_shape = (
                self.sparse_index_num_gpu_blocks,
                self.num_heads,
            )
            column_index_cache_shape = (
                self.sparse_index_num_gpu_blocks,
                self.num_heads,
                self.sparse_index_max_vertical_slash_topk,
            )

            for _ in range(self.num_attention_layers):
                layer_block_count_gpu_cache = torch.zeros(
                    block_count_cache_shape,
                    dtype=self.dtype,
                    device=device,
                )
                block_count_gpu_cache.append(layer_block_count_gpu_cache)

            for _ in range(self.num_attention_layers):
                layer_block_index_gpu_cache = torch.zeros(
                    block_index_cache_shape,
                    dtype=self.dtype,
                    device=device,
                )
                block_index_gpu_cache.append(layer_block_index_gpu_cache)

            for _ in range(self.num_attention_layers):
                layer_column_count_gpu_cache = torch.zeros(
                    column_count_cache_shape,
                    dtype=self.dtype,
                    device=device,
                )
                column_count_gpu_cache.append(layer_column_count_gpu_cache)
            
            for _ in range(self.num_attention_layers):
                layer_column_index_gpu_cache = torch.zeros(
                    column_index_cache_shape,
                    dtype=self.dtype,
                    device=device,
                )
                column_index_gpu_cache.append(layer_column_index_gpu_cache)

            return (
                block_count_gpu_cache, block_index_gpu_cache,
                column_count_gpu_cache, column_index_gpu_cache,
            )
        else:
            page_compress_caches = []
            blocks = self.speculative_config.block_sparse_token_budget // self.cache_config.block_size

            page_compress_cache_shape = (
                self.sparse_index_num_gpu_blocks,
                self.num_kv_heads,
                blocks,
            )

            for _ in range(self.num_attention_layers):
                layer_page_compress_cache = torch.zeros(
                    page_compress_cache_shape,
                    dtype=self.dtype,
                    device=device
                )
                page_compress_caches.append(layer_page_compress_cache)

            return (None, page_compress_caches, None, None)

    @staticmethod
    def get_cache_block_size(
        cache_config: CacheConfig,
        model_config: ModelConfig,
        parallel_config: ParallelConfig,
        speculative_config: SpeculativeConfig,
    ) -> int:
        if not speculative_config.block_sparse_mode:
            assert speculative_config.sparse_index_max_vertical_slash_topk is not None
            num_heads = model_config.get_num_attention_heads(parallel_config)
            num_attention_layers = model_config.get_num_layers_by_block_type(
                parallel_config, LayerBlockType.attention)
            # TODO: Maybe we can use a small data type later.
            dtype = torch.int32

            total = num_attention_layers * num_heads * 2 * (
                1 + speculative_config.sparse_index_max_vertical_slash_topk)
            dtype_size = get_dtype_size(dtype)
            return dtype_size * total
        else:
            # block_size = cache_config.block_size
            # token_budget = speculative_config.block_sparse_token_budget
            # assert token_budget % block_size == 0
            # blocks = token_budget // block_size
            # num_heads = model_config.get_num_attention_heads(parallel_config)
            # num_attention_layers = model_config.get_num_layers_by_block_type(
            #     parallel_config, LayerBlockType.attention)
            # dtype = torch.int32

            # total = num_attention_layers * num_heads * (1 + blocks)
            # dtype_size = get_dtype_size(dtype)
            # return dtype_size * total
            assert speculative_config.block_sparse_mode
            block_size = cache_config.block_size
            token_budget = speculative_config.block_sparse_token_budget
            assert token_budget % block_size == 0
            blocks = token_budget // block_size   # topk
            num_kv_heads = model_config.get_num_kv_heads(parallel_config)
            num_attention_layers = model_config.get_num_layers_by_block_type(
                parallel_config, LayerBlockType.attention)
            dtype = torch.int32

            total = num_attention_layers * num_kv_heads * blocks
            dtype_size = get_dtype_size(dtype)
            return dtype_size * total
            