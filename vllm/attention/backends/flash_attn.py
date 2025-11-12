# SPDX-License-Identifier: Apache-2.0
"""Attention layer with FlashAttention."""
from collections import defaultdict
from dataclasses import dataclass
from itertools import accumulate
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn.functional as F
import numpy as np

from vllm import _custom_ops as ops
# yapf conflicts with isort for this block
# yapf: disable
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer,
                                              AttentionMetadata,
                                              AttentionMetadataBuilder,
                                              AttentionType,
                                              is_quantized_kv_cache)
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding
# yapf: enable
from vllm.attention.backends.utils import (
    PAD_SLOT_ID, CommonAttentionState, compute_slot_mapping,
    compute_slot_mapping_start_idx, get_num_prefill_decode_query_kv_tokens,
    get_seq_len_block_table_args, is_all_cross_attn_metadata_set,
    is_all_encoder_attn_metadata_set, is_block_tables_empty)
from vllm.attention.utils.fa_utils import (flash_attn_supports_fp8,
                                           get_flash_attn_version)
from vllm.logger import init_logger
from vllm.multimodal import MultiModalPlaceholderMap
from vllm.utils import async_tensor_h2d, make_tensor_with_pad
from vllm.vllm_flash_attn import (flash_attn_varlen_func,
                                  flash_attn_with_kvcache,
                                  sparse_attn_func)

import os
import math
from enum import IntEnum
from .x_attn import Xattention_prefill
from .minference import Minference_prefill
from .flex_prefill_attention import flex_prefill_attention
from spas_sage_attn import spas_sage2_attn_meansim_cuda

if TYPE_CHECKING:
    from vllm.worker.model_runner import (ModelInputForGPUBuilder,
                                          ModelInputForGPUWithSamplingMetadata)

logger = init_logger(__name__)

class FlashAttentionBackend(AttentionBackend):

    accept_output_buffer: bool = True

    @staticmethod
    def get_supported_head_sizes() -> List[int]:
        return [32, 64, 96, 128, 160, 192, 224, 256]

    @staticmethod
    def get_name() -> str:
        return "FLASH_ATTN"

    @staticmethod
    def get_impl_cls() -> Type["FlashAttentionImpl"]:
        return FlashAttentionImpl

    @staticmethod
    def get_metadata_cls() -> Type["AttentionMetadata"]:
        return FlashAttentionMetadata

    @staticmethod
    def get_builder_cls() -> Type["FlashAttentionMetadataBuilder"]:
        return FlashAttentionMetadataBuilder

    @staticmethod
    def get_state_cls() -> Type["CommonAttentionState"]:
        return CommonAttentionState

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
    ) -> Tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        return (2, num_blocks, block_size, num_kv_heads, head_size)

    @staticmethod
    def swap_blocks(
        src_kv_cache: torch.Tensor,
        dst_kv_cache: torch.Tensor,
        src_to_dst: torch.Tensor,
    ) -> None:
        src_key_cache = src_kv_cache[0]
        dst_key_cache = dst_kv_cache[0]
        ops.swap_blocks(src_key_cache, dst_key_cache, src_to_dst)
        src_value_cache = src_kv_cache[1]
        dst_value_cache = dst_kv_cache[1]
        ops.swap_blocks(src_value_cache, dst_value_cache, src_to_dst)

    @staticmethod
    def copy_blocks(
        kv_caches: List[torch.Tensor],
        src_to_dists: torch.Tensor,
    ) -> None:
        key_caches = [kv_cache[0] for kv_cache in kv_caches]
        value_caches = [kv_cache[1] for kv_cache in kv_caches]

        ops.copy_blocks(key_caches, value_caches, src_to_dists)


@dataclass
class FlashAttentionMetadata(AttentionMetadata):
    """Metadata for FlashAttentionBackend.

    NOTE: Any python object stored here is not updated when it is
    cuda-graph replayed. If you have values that need to be changed
    dynamically, it should be stored in tensor. The tensor has to be
    updated from `CUDAGraphRunner.forward` API.
    """
    # (batch_size,). The sequence length per sequence. Sequence length means
    # the computed tokens + new tokens None if it is a decoding.
    seq_lens: Optional[List[int]]
    # seq_lens stored as a tensor.
    seq_lens_tensor: Optional[torch.Tensor]

    # NOTE(sang): Definition of context_len, query_len, and seq_len.
    # |---------- N-1 iteration --------|
    # |---------------- N iteration ---------------------|
    # |- tokenA -|......................|-- newTokens ---|
    # |---------- context_len ----------|
    # |-------------------- seq_len ---------------------|
    #                                   |-- query_len ---|

    # Maximum sequence length among prefill batch. 0 if there are decoding
    # requests only.
    max_prefill_seq_len: int
    # Maximum sequence length among decode batch. 0 if there are prefill
    # requests only.
    max_decode_seq_len: int
    # (batch_size,) A tensor of context lengths (tokens that are computed
    # so far).
    context_lens_tensor: Optional[torch.Tensor]

    # (batch_size, max_blocks_per_seq).
    # Block addresses per sequence. (Seq id -> list of physical block)
    # E.g., [0, 1, 2] means tokens are stored in 0th, 1st, and 2nd blocks
    # in the kv cache. Each block can contain up to block_size tokens.
    # 2nd dimensions are padded up to max_blocks_per_seq if it is cuda-graph
    # captured.
    block_tables: Optional[torch.Tensor]

    # Whether or not if cuda graph is enabled.
    # Cuda-graph is currently enabled for decoding only.
    # TODO(woosuk): Move `use_cuda_graph` out since it's unrelated to attention.

    use_cuda_graph: bool

    # Maximum query length in the batch.
    max_query_len: Optional[int] = None

    # Max number of query tokens among request in the batch.
    max_decode_query_len: Optional[int] = None

    # (batch_size + 1,). The cumulative subquery lengths of the sequences in
    # the batch, used to index into subquery. E.g., if the subquery length
    # is [4, 6], it is [0, 4, 10].
    query_start_loc: Optional[torch.Tensor] = None
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    seq_start_loc: Optional[torch.Tensor] = None

    prompt_lens: Optional[List[int]] = None

    max_sparse_index_decode_seq_len: Optional[int] = None
    max_sparse_index_decode_query_len: Optional[int] = None

    _cached_prefill_metadata: Optional["FlashAttentionMetadata"] = None
    _cached_decode_metadata: Optional["FlashAttentionMetadata"] = None
    _cached_sparse_index_decode_metadata: Optional["FlashAttentionMetadata"] = None

    # Begin encoder attn & enc/dec cross-attn fields...

    # Encoder sequence lengths representation
    encoder_seq_lens: Optional[List[int]] = None
    encoder_seq_lens_tensor: Optional[torch.Tensor] = None
    # (batch_size + 1,). The cumulative sequence lengths of the sequences in
    # the batch, used to index into sequence. E.g., if the sequence length is
    # [4, 6], it is [0, 4, 10].
    encoder_seq_start_loc: Optional[torch.Tensor] = None
    # Maximum sequence length among encoder sequences
    max_encoder_seq_len: Optional[int] = None
    # Number of tokens input to encoder
    num_encoder_tokens: Optional[int] = None

    # Cross-attention memory-mapping data structures: slot mapping
    # and block tables
    cross_slot_mapping: Optional[torch.Tensor] = None
    cross_block_tables: Optional[torch.Tensor] = None

    # num_sparse_index_decodes: int = 0
    # num_sparse_index_tokens: int = 0
    # sparse_index_kv_compress_recover_rate: Optional[float] = None
    num_sparse_index_recomputes: int = 0
    num_sparse_index_recompute_tokens: int = 0
    # flash attention 所需的新参数
    # page_compress_cache_ids: Optional[List[int]] = None
    page_compress_cache_ids_tensor: Optional[torch.Tensor] = None
    # num_compressed_pages: Optional[List[int]] = None
    num_compressed_pages_tensor: Optional[torch.Tensor] = None
    actual_seqlen_tensor: Optional[torch.Tensor] = None
    actual_max_num_blocks_per_seq: Optional[int] = None
    actual_max_decode_seq_len: Optional[int] = None
    # page selector 的参数
    # num_full_blocks_tensor 由 num_compressed_pages_tensor 的前 num_sparse_index_recomputes 个提供
    # 用于初始化 out tensor
    page_selector_max_block_size: Optional[int] = None
    page_compress_topk: Optional[int] = None


    update_meta_block_id_tensor: Optional[torch.Tensor] = None
    seq_len_after_pooling_for_decode_tensor: Optional[torch.Tensor] = None
    doc_token_ranges: List[Optional[List[tuple]]] = None
    docs_hash: List[Optional[list]] = None
    kvcache_path: List[Optional[list]] = None
    cached_offset: List[Optional[list]] = None
    cache_blend_static_index_cache: Optional[List] = None

    enable_blk_attn: bool = False
    # batch_idx_offset_for_blk_attn_tensor: Optional[torch.Tensor] = None
    blk_attn_prefill_cu_seqlens_q: Optional[torch.Tensor] = None
    blk_attn_max_prefill_q_len: Optional[int] = None
    blk_attn_prefill_seqused_k: Optional[torch.Tensor] = None
    blk_attn_max_prefill_kv_len: Optional[int] = None
    blk_attn_prefill_actual_chunked_seqlen_k: Optional[torch.Tensor] = None
    blk_attn_prefill_chunk_rotary_offset_positions: Optional[torch.Tensor] = None
    blk_attn_prefill_cu_num_chunks_k: Optional[torch.Tensor] = None

    blk_attn_decode_cu_seqlens_q: Optional[torch.Tensor] = None
    blk_attn_max_decode_q_len: Optional[int] = None
    blk_attn_decode_seqused_k: Optional[torch.Tensor] = None
    blk_attn_max_decode_kv_len: Optional[int] = None
    blk_attn_decode_actual_chunked_seqlen_k: Optional[torch.Tensor] = None
    blk_attn_decode_chunk_rotary_offset_positions: Optional[torch.Tensor] = None
    blk_attn_decode_cu_num_chunks_k: Optional[torch.Tensor] = None
    # blk_attn_seq_start_loc_tensor: Optional[torch.Tensor] = None

    @property
    def is_all_encoder_attn_metadata_set(self):
        '''
        All attention metadata required for encoder attention is set.
        '''
        return is_all_encoder_attn_metadata_set(self)

    @property
    def is_all_cross_attn_metadata_set(self):
        '''
        All attention metadata required for enc/dec cross-attention is set.

        Superset of encoder attention required metadata.
        '''
        return is_all_cross_attn_metadata_set(self)

    @property
    def prefill_metadata(self) -> Optional["FlashAttentionMetadata"]:
        if self.num_prefills == 0:
            return None

        if self._cached_prefill_metadata is not None:
            return self._cached_prefill_metadata

        assert ((self.seq_lens is not None)
                or (self.encoder_seq_lens is not None))
        assert ((self.seq_lens_tensor is not None)
                or (self.encoder_seq_lens_tensor is not None))

        # Compute some attn_metadata fields which default to None
        query_start_loc = (None if self.query_start_loc is None else
                           self.query_start_loc[:self.num_prefills + 1])
        slot_mapping = (None if self.slot_mapping is None else
                        self.slot_mapping[:self.num_prefill_tokens])
        seq_lens = (None if self.seq_lens is None else
                    self.seq_lens[:self.num_prefills])
        seq_lens_tensor = (None if self.seq_lens_tensor is None else
                           self.seq_lens_tensor[:self.num_prefills])
        seq_start_loc = (None if self.seq_start_loc is None else
                         self.seq_start_loc[:self.num_prefills + 1])
        context_lens_tensor = (None if self.context_lens_tensor is None else
                               self.context_lens_tensor[:self.num_prefills])
        block_tables = (None if self.block_tables is None else
                        self.block_tables[:self.num_prefills])
        prompt_lens = (None if self.prompt_lens is None else
                       self.prompt_lens[:self.num_prefills])

        self._cached_prefill_metadata = FlashAttentionMetadata(
            num_prefills=self.num_prefills,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=0,
            slot_mapping=slot_mapping,
            multi_modal_placeholder_index_maps=self.
            multi_modal_placeholder_index_maps,
            enable_kv_scales_calculation=self.enable_kv_scales_calculation,
            seq_lens=seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            max_query_len=self.max_query_len,
            max_prefill_seq_len=self.max_prefill_seq_len,
            max_decode_query_len=0,
            max_decode_seq_len=0,
            query_start_loc=query_start_loc,
            seq_start_loc=seq_start_loc,
            prompt_lens=prompt_lens,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables,
            use_cuda_graph=False,
            # Begin encoder & cross attn fields below...
            encoder_seq_lens=self.encoder_seq_lens,
            encoder_seq_lens_tensor=self.encoder_seq_lens_tensor,
            encoder_seq_start_loc=self.encoder_seq_start_loc,
            max_encoder_seq_len=self.max_encoder_seq_len,
            cross_slot_mapping=self.cross_slot_mapping,
            cross_block_tables=self.cross_block_tables,
            doc_token_ranges=self.doc_token_ranges,
            docs_hash=self.docs_hash,
            kvcache_path=self.kvcache_path,
            cached_offset=self.cached_offset,
            cache_blend_static_index_cache=self.cache_blend_static_index_cache,
            enable_blk_attn=self.enable_blk_attn,
            # batch_idx_offset_for_blk_attn_tensor=self.batch_idx_offset_for_blk_attn_tensor,
            blk_attn_prefill_cu_seqlens_q=self.blk_attn_prefill_cu_seqlens_q,
            blk_attn_max_prefill_q_len=self.blk_attn_max_prefill_q_len,
            blk_attn_prefill_seqused_k=self.blk_attn_prefill_seqused_k,
            blk_attn_max_prefill_kv_len=self.blk_attn_max_prefill_kv_len,
            blk_attn_prefill_actual_chunked_seqlen_k=self.blk_attn_prefill_actual_chunked_seqlen_k,
            blk_attn_prefill_chunk_rotary_offset_positions=self.blk_attn_prefill_chunk_rotary_offset_positions,
            blk_attn_prefill_cu_num_chunks_k=self.blk_attn_prefill_cu_num_chunks_k,
            # blk_attn_seq_start_loc_tensor=self.blk_attn_seq_start_loc_tensor
        )
        return self._cached_prefill_metadata

    @property
    def decode_metadata(self) -> Optional["FlashAttentionMetadata"]:
        if self.num_decode_tokens == 0:
            return None

        if self._cached_decode_metadata is not None:
            return self._cached_decode_metadata
        assert ((self.seq_lens_tensor is not None)
                or (self.encoder_seq_lens_tensor is not None))

        # Compute some attn_metadata fields which default to None
        slot_mapping = (None if self.slot_mapping is None else
                        self.slot_mapping[self.num_prefill_tokens:])
        seq_lens_tensor = (None if self.seq_lens_tensor is None else
                           self.seq_lens_tensor[self.num_prefills:])
        block_tables = (None if self.block_tables is None else
                        self.block_tables[self.num_prefills:])
        prompt_lens = (None if self.prompt_lens is None else
                       self.prompt_lens[self.num_prefills:])

        self._cached_decode_metadata = FlashAttentionMetadata(
            num_prefills=0,
            num_prefill_tokens=0,
            num_decode_tokens=self.num_decode_tokens,
            slot_mapping=slot_mapping,
            multi_modal_placeholder_index_maps=None,
            enable_kv_scales_calculation=True,
            seq_lens=self.seq_lens,
            seq_lens_tensor=seq_lens_tensor,
            max_decode_query_len=self.max_decode_query_len,
            max_query_len=self.max_query_len,
            max_prefill_seq_len=0,
            max_decode_seq_len=self.max_decode_seq_len,
            # Batch may be composed of prefill|decodes, adjust query start
            # indices to refer to the start of decodes. E.g.
            # in tokens:[3 prefills|6 decodes], query_start_loc=[3,9] => [0,6].
            query_start_loc=(self.query_start_loc[self.num_prefills:] -
                             self.query_start_loc[self.num_prefills])
            if self.query_start_loc is not None else None,
            seq_start_loc=self.seq_start_loc[self.num_prefills:]
            if self.seq_start_loc is not None else None,
            prompt_lens=prompt_lens,
            context_lens_tensor=None,
            block_tables=block_tables,
            use_cuda_graph=self.use_cuda_graph,
            # Begin encoder & cross attn fields below...
            encoder_seq_lens=self.encoder_seq_lens,
            encoder_seq_lens_tensor=self.encoder_seq_lens_tensor,
            encoder_seq_start_loc=self.encoder_seq_start_loc,
            max_encoder_seq_len=self.max_encoder_seq_len,
            cross_slot_mapping=self.cross_slot_mapping,
            cross_block_tables=self.cross_block_tables,
            num_sparse_index_recomputes=self.num_sparse_index_recomputes,
            page_compress_cache_ids_tensor=self.page_compress_cache_ids_tensor,
            num_compressed_pages_tensor=self.num_compressed_pages_tensor,
            actual_max_num_blocks_per_seq=self.actual_max_num_blocks_per_seq,
            actual_max_decode_seq_len=self.actual_max_decode_seq_len,
            actual_seqlen_tensor=self.actual_seqlen_tensor,
            page_selector_max_block_size=self.page_selector_max_block_size,
            page_compress_topk=self.page_compress_topk,
            seq_len_after_pooling_for_decode_tensor=self.seq_len_after_pooling_for_decode_tensor,

            enable_blk_attn=self.enable_blk_attn,
            blk_attn_decode_cu_seqlens_q=self.blk_attn_decode_cu_seqlens_q,
            blk_attn_max_decode_q_len=self.blk_attn_max_decode_q_len,
            blk_attn_decode_seqused_k=self.blk_attn_decode_seqused_k,
            blk_attn_max_decode_kv_len=self.blk_attn_max_decode_kv_len,
            blk_attn_decode_actual_chunked_seqlen_k=self.blk_attn_decode_actual_chunked_seqlen_k,
            blk_attn_decode_chunk_rotary_offset_positions=self.blk_attn_decode_chunk_rotary_offset_positions,
            blk_attn_decode_cu_num_chunks_k=self.blk_attn_decode_cu_num_chunks_k,
        )
        return self._cached_decode_metadata

    def advance_step(self,
                     model_input: "ModelInputForGPUWithSamplingMetadata",
                     sampled_token_ids: Optional[torch.Tensor],
                     block_size: int,
                     num_seqs: int,
                     num_queries: int,
                     turn_prefills_into_decodes: bool = False):
        """
        Update metadata in-place to advance one decode step.
        """
        # When using cudagraph, the num_seqs is padded to the next captured
        # batch sized, but num_queries tracks the actual number of requests in
        # the batch. For --enforce-eager mode, num_seqs == num_queries
        if num_seqs != num_queries:
            assert num_seqs > num_queries
            assert self.use_cuda_graph

        if turn_prefills_into_decodes:
            # When Multi-Step is enabled with Chunked-Prefill, prefills and
            # decodes are scheduled together. In the first step, all the
            # prefills turn into decodes. This update reflects that
            # conversion.
            assert self.num_decode_tokens + self.num_prefills == num_seqs
            self.num_decode_tokens += self.num_prefills
            self.num_prefills = 0
            self.num_prefill_tokens = 0
            self.max_prefill_seq_len = 0
            self.max_query_len = 1

            self.slot_mapping = self.slot_mapping[:num_seqs]
        else:
            assert self.seq_lens is not None
            assert self.max_decode_seq_len == max(self.seq_lens)

        assert self.num_prefills == 0
        assert self.num_prefill_tokens == 0
        assert self.num_decode_tokens == num_seqs
        assert self.slot_mapping.shape == (num_seqs, )

        assert self.seq_lens is not None
        assert len(self.seq_lens) == num_seqs
        assert self.seq_lens_tensor is not None
        assert self.seq_lens_tensor.shape == (num_seqs, )
        assert self.max_query_len == 1
        assert self.max_prefill_seq_len == 0

        assert self.query_start_loc is not None
        assert self.query_start_loc.shape == (num_queries + 1, )
        assert self.seq_start_loc is not None
        assert self.seq_start_loc.shape == (num_seqs + 1, )

        assert self.context_lens_tensor is not None
        assert self.context_lens_tensor.shape == (num_queries, )

        assert self.block_tables is not None
        assert self.block_tables.shape[0] == num_seqs

        # Update query lengths. Note that we update only queries and not seqs,
        # since tensors may be padded due to captured cuda graph batch size
        for i in range(num_queries):
            self.seq_lens[i] += 1
        self.max_decode_seq_len = max(self.seq_lens)

        ops.advance_step_flashattn(num_seqs=num_seqs,
                                   num_queries=num_queries,
                                   block_size=block_size,
                                   input_tokens=model_input.input_tokens,
                                   sampled_token_ids=sampled_token_ids,
                                   input_positions=model_input.input_positions,
                                   seq_lens=self.seq_lens_tensor,
                                   slot_mapping=self.slot_mapping,
                                   block_tables=self.block_tables)


class FlashAttentionMetadataBuilder(
        AttentionMetadataBuilder[FlashAttentionMetadata]):
    
    # Block Attention 需要处理的几个参数：
    # 1. q：对于 prefill 的请求，需要处理未命中的 chunk （rotary offsets == -1）
    #    对于 decode 请求，仅处理原始的 query len
    #    对于非 chunk 请求，仅处理原始的 query len
    # 2. cu_seqlens_q & max_seqlen_q：根据上述 query len 进行处理
    # 3. seqused_k：对于 prefill 请求，对于未命中的 chunk，设置为 -1，每个请求的最后一个 chunk 使用整个序列长度
    #    对于 decode 请求，仅处理最后一个 chunk，因此需要独立维护一个 kv_lens
    #    对于非 chunk 请求，seqlen
    # 4. max_seqlen_k：根据 kv_lens 获取
    # 5. block_tables：对于 prefill 请求，未命中的 chunk 需要截取，以便计算 slot id；decode无需处理；
    # 6. actual_chunked_seqlen_k：是 seqused_k 的 chunk 细分版本， 对于 prefill 请求，未命中的 chunk 设置为该 chunk 的长度，对于最后一个 chunk，设置为之前的 chunk 序列
    #    对于 decode 请求，仅处理最后一个 chunk，设置为之前的 chunk 序列
    #    对于非 chunk 请求，seqlen
    # 7. chunk_rotray_offset_positions：对于 prefill 请求，未命中的 chunk 设置为 0，最后一个 chunk 设置命中 chunk的offset，同时非命中的设置为0，并且最后设置为 0
    #    对于 decode 请求，按照 prefill 的最后一个块处理
    #    对于非 chunk 请求，0
    # 8. cu_num_chunks_k：对于 prefill 请求，未命中的 chunk 设置为 1，最后一个 chunk，设置为之前 chunk 序列长度 + 1
    #    对于 decode 请求，仅处理最后一个 chunk
    #    对于非 chunk 请求，1
    # 9. local_key & local_value & local_cu_seqlen_k：仅在 prefill 时使用

    def __init__(self, input_builder: "ModelInputForGPUBuilder"):
        self.input_builder = input_builder
        self.runner = input_builder.runner
        self.sliding_window = input_builder.sliding_window
        self.block_size = input_builder.block_size
        self.page_compress_topk = input_builder.page_compress_topk
        self.enble_blk_attn = input_builder.enable_blk_attn

    def prepare(self):
        self.slot_mapping: List[int] = []
        self.prefill_seq_lens: List[int] = []
        self.context_lens: List[int] = []
        self.block_tables: List[List[int]] = []
        self.curr_seq_lens: List[int] = []  # 前 num_sparse_index_recomputes 个用于计算 page_selector_max_block_size
        self.multimodal_placeholder_maps: Dict[
            str,
            MultiModalPlaceholderMap] = defaultdict(MultiModalPlaceholderMap)
        self.num_prefills = 0
        ##########
        # self.num_sparse_index_decodes = 0
        self.num_sparse_index_recomputes = 0
        ##########
        self.num_prefill_tokens = 0
        self.num_decode_tokens = 0
        self.has_prefix_cache_hit = False
        self.prompt_lens: List[int] = []
        self.num_sparse_index_recompute_tokens = 0
        # self.num_sparse_index_tokens = 0
        # self.use_sparse_index_seq_lens: List[int] = []
        # We assume that each sequence group has only one sequence.
        # 对于 decode 请求，每个请求的 sparse_index_block_id, 不存在则 -1
        self.sparse_index_blocks: List[int] = []    # 用于构建 page_compress_cache_ids_tensor
        # 对于 decode 请求，每个请求被压缩的 page 数量，不存在则 -1
        # 对于需要重新计算 compressed page 的请求，前 num_sparse_index_recomputes 个即用于 page_selector 参数 num_full_blocks
        self.num_compressed_pages: List[int] = []   # 用于构建 num_compressed_pages_tensor & num_full_blocks_tensor
        # 对于 decode 请求，每个请求被压缩之后的实际 seqlen
        self.actual_curr_seq_lens: List[int] = []   # 用于计算 actual_max_num_blocks_per_seq
        self.doc_token_ranges: List[List[tuple]] = []
        self.docs_hash: List[list] = []
        self.kvcache_path: List[list] = []
        self.cached_offset: List[list] = []
        # self.batch_idx_offset_for_blk_attn: List[int] = []
        # self.blk_attn_prefill_seq_lens: List[int] = []
        # blk_attn_query_lens 即每段 chunk 的
        self.chunked_prefill_query_lens: List[int] = []
        self.chunked_decode_query_lens: List[int] = []
        self.chunked_prefill_kv_lens: List[int] = []
        self.chunked_decode_kv_lens: List[int] = []
        self.detail_chunked_prefill_kv_lens: List[int] = []
        self.detail_chunked_decode_kv_lens: List[int] = []
        self.detail_chunked_prefill_rotary_offsets: List[int] = []
        self.detail_chunked_decode_rotary_offsets: List[int] = []
        self.detail_chunk_nums_prefill: List[int] = []
        self.detail_chunk_nums_decode: List[int] = []

    def _add_seq_group(
            self, inter_data: "ModelInputForGPUBuilder.InterDataForSeqGroup",
            chunked_prefill_enabled: bool, prefix_cache_hit: bool):
        """Add a sequence group to the metadata. Specifically update/append
        1. context length.
        2. block table.
        3. slot mapping.
        """
        is_prompt = inter_data.is_prompt
        is_sparse_index_recompute = inter_data.is_sparse_index_recompute
        is_sparse_index = inter_data.is_sparse_index
        is_recitify = inter_data.is_recitify
        pooling_token_delta = inter_data.pooling_token_delta
        doc_ranges = inter_data.doc_ranges
        docs_hash = inter_data.docs_hash
        kvcache_path = inter_data.kvcache_path
        cached_offset = inter_data.cached_offset
        rotary_offsets = inter_data.rotary_position_offsets
        assert (not is_sparse_index_recompute) or (not is_recitify)
        
        block_tables = inter_data.block_tables
        sparse_index_block = inter_data.sparse_index_block
        num_computed_token_to_compress = inter_data.num_computed_token_to_compress
        if is_sparse_index or is_sparse_index_recompute:
            assert num_computed_token_to_compress != -1

        if sparse_index_block and len(sparse_index_block) > 0:
            assert len(sparse_index_block) == 1
            sparse_index_block_id = next(iter(sparse_index_block.values()))
        else:
            sparse_index_block_id = -1

        for (seq_id, token_len, seq_len, curr_seq_len, query_len, context_len,
             curr_sliding_window_block, prompt_len) in zip(
                 inter_data.seq_ids, [len(t) for t in inter_data.input_tokens],
                 inter_data.orig_seq_lens, inter_data.seq_lens,
                 inter_data.query_lens, inter_data.context_lens,
                 inter_data.curr_sliding_window_blocks,
                 inter_data.prompt_lens):
            self.prompt_lens.append(prompt_len)
            self.context_lens.append(context_len)

            if is_prompt:
                mm_maps = inter_data.multi_modal_placeholder_maps
                if mm_maps:
                    for modality, placeholders in mm_maps.items():
                        self.multimodal_placeholder_maps[modality].extend(
                            placeholders)

                self.num_prefills += 1
                self.num_prefill_tokens += token_len
                self.prefill_seq_lens.append(seq_len)
                if not self.enble_blk_attn:
                    pass
                elif doc_ranges is None:
                    # 对于非 chunk 请求，仅处理原始的 query len
                    self.chunked_prefill_query_lens.append(query_len)
                    # 对于非 chunk 请求，seqlen
                    self.chunked_prefill_kv_lens.append(seq_len)
                    # 对于非 chunk 请求，seqlen
                    self.detail_chunked_prefill_kv_lens.append(seq_len)
                    # 对于非 chunk 请求，0
                    self.detail_chunked_prefill_rotary_offsets.append(0)
                    # 对于非 chunk 请求，1
                    self.detail_chunk_nums_prefill.append(1)
                    # assert context_len == 0
                    # self.blk_attn_prefill_seq_lens.append(seq_len)
                    # self.batch_idx_offset_for_blk_attn.append(0)
                else:
                    actual_chunk_lens_for_last_chunk = []
                    rotary_offsets_for_last_chunk = []
                    for doc_range, rotary_offset in zip(doc_ranges, rotary_offsets):
                        actual_chunk_lens_for_last_chunk.append(doc_range[2])
                        if rotary_offset == -1:
                            self.num_prefills += 1 # 这里需要更新 num_prefills
                            rotary_offsets_for_last_chunk.append(0)
                            # 未命中 chunk，对齐后的长度
                            self.chunked_prefill_query_lens.append(doc_range[1] - doc_range[0])
                            # 未命中 chunk，seqused_k = -1
                            self.chunked_prefill_kv_lens.append(-1)
                            # 未命中 chunk，actual_chunked_seqlen_k 可以设置为对齐后的长度
                            self.detail_chunked_prefill_kv_lens.append(doc_range[1] - doc_range[0])
                            # 未命中 chunk，rotary offset = 0
                            self.detail_chunked_prefill_rotary_offsets.append(0)
                            # 未命中 chunk，chunk num = 1
                            self.detail_chunk_nums_prefill.append(1)
                        else:
                            rotary_offsets_for_last_chunk.append(rotary_offset)
                    # last chunk
                    assert seq_len > doc_ranges[-1][1]
                    # 最后一块，query_len 为剩余长度
                    self.chunked_prefill_query_lens.append(seq_len - doc_ranges[-1][1])
                    # 最后一块，seqused_k = seq_len
                    self.chunked_prefill_kv_lens.append(seq_len)
                    # 最后一块，chunk，actual_chunked_seqlen_k
                    actual_chunk_lens_for_last_chunk.append(seq_len - doc_ranges[-1][1])
                    self.detail_chunked_prefill_kv_lens.extend(actual_chunk_lens_for_last_chunk)
                    # 最后一块，rotary_offsets
                    rotary_offsets_for_last_chunk.append(0)
                    self.detail_chunked_prefill_rotary_offsets.extend(rotary_offsets_for_last_chunk)
                    # 最后一块，chunk_num
                    assert len(actual_chunk_lens_for_last_chunk) == len(rotary_offsets_for_last_chunk)
                    self.detail_chunk_nums_prefill.append(len(actual_chunk_lens_for_last_chunk))
                    # assert context_len == 0
                    # self.blk_attn_prefill_seq_lens.extend([(doc_range[1] - doc_range[0]) for doc_range in doc_ranges])
                    # self.blk_attn_prefill_seq_lens.append(seq_len - doc_ranges[-1][1])
                    # self.batch_idx_offset_for_blk_attn.extend([0] * len(doc_ranges))
                    # self.batch_idx_offset_for_blk_attn.append(len(doc_ranges))
            else:
                # decode
                self.num_decode_tokens += query_len
                # [shk]
                self.curr_seq_lens.append(curr_seq_len - pooling_token_delta)

                # num_sparse_index_recomputes & num_sparse_index_recompute_tokens
                if is_sparse_index_recompute:
                    self.num_sparse_index_recomputes += 1
                    self.num_sparse_index_recompute_tokens += query_len
                
                # sparse_index_blocks
                if not is_recitify:
                    self.sparse_index_blocks.append(sparse_index_block_id)
                else:
                    self.sparse_index_blocks.append(-1)

                # num_compressed_pages
                if is_recitify or num_computed_token_to_compress == -1:
                    self.num_compressed_pages.append(-1)
                else:
                    self.num_compressed_pages.append(num_computed_token_to_compress // self.block_size)
                
                # actual_curr_seq_lens
                if (not is_recitify) and (is_sparse_index or is_sparse_index_recompute):
                    self.actual_curr_seq_lens.append(
                        curr_seq_len - (
                            num_computed_token_to_compress // self.block_size - self.page_compress_topk
                        ) * self.block_size
                    )
                else:
                    self.actual_curr_seq_lens.append(curr_seq_len)

                if not self.enble_blk_attn:
                    pass
                elif doc_ranges is None:
                    assert query_len == 1
                    # 对于非 chunk 请求，query len = 1
                    self.chunked_decode_query_lens.append(1)
                    # 对于非 chunk 请求，seqlen
                    self.chunked_decode_kv_lens.append(seq_len)
                    # 对于非 chunk 请求，seqlen
                    self.detail_chunked_decode_kv_lens.append(seq_len)
                    # 对于非 chunk 请求，0
                    self.detail_chunked_decode_rotary_offsets.append(0)
                    # 对于非 chunk 请求，1
                    self.detail_chunk_nums_decode.append(1)
                else:
                    assert query_len == 1
                    actual_chunk_lens_for_last_chunk = []
                    rotary_offsets_for_last_chunk = []
                    for doc_range, rotary_offset in zip(doc_ranges, rotary_offsets):
                        actual_chunk_lens_for_last_chunk.append(doc_range[2])
                        if rotary_offset == -1:
                            rotary_offsets_for_last_chunk.append(0)
                        else:
                            rotary_offsets_for_last_chunk.append(rotary_offset)
                    # last chunk
                    assert seq_len > doc_ranges[-1][1]
                    # 最后一块，query_len = 1
                    self.chunked_decode_query_lens.append(1)
                    # 最后一块，seqused_k = seq_len
                    self.chunked_decode_kv_lens.append(seq_len)
                    # 最后一块，chunk，actual_chunked_seqlen_k
                    actual_chunk_lens_for_last_chunk.append(seq_len - doc_ranges[-1][1])
                    self.detail_chunked_decode_kv_lens.extend(actual_chunk_lens_for_last_chunk)
                    # 最后一块，rotary_offsets
                    rotary_offsets_for_last_chunk.append(0)
                    self.detail_chunked_decode_rotary_offsets.extend(rotary_offsets_for_last_chunk)
                    # 最后一块，chunk_num
                    assert len(actual_chunk_lens_for_last_chunk) == len(rotary_offsets_for_last_chunk)
                    self.detail_chunk_nums_decode.append(len(actual_chunk_lens_for_last_chunk))

            # Compute block table.
            # TODO(sang): Combine chunked prefill and prefix caching by
            # only allowing multiple of block_size chunk size.
            # NOTE: This only works for oooooooxxx style attention.
            block_table = []
            chunked_block_ranges = []
            if prefix_cache_hit or (self.enble_blk_attn and block_tables is not None):
                # NOTE(woosuk): For flash-attn, the block table should
                # include the entries for the incoming prefill tokens.
                block_table = block_tables[seq_id]
                # 对于 prefill 请求，未命中的 chunk 需要截取 block_table；decode无需处理；
                if self.enble_blk_attn and is_prompt and doc_ranges is not None:
                    for doc_range, rotary_offset in zip(doc_ranges, rotary_offsets):
                        assert (doc_range[0] % self.block_size == 0) and (doc_range[1] % self.block_size == 0)
                        if rotary_offset == -1:
                            block_ranges = (doc_range[0] // self.block_size, doc_range[1] // self.block_size, doc_range[1] - doc_range[0])
                            chunked_block_ranges.append(block_ranges)
                            self.block_tables.append(block_table[block_ranges[0]:block_ranges[1]])
                    chunked_block_ranges.append((doc_ranges[-1][1] // self.block_size, len(block_table), seq_len - doc_ranges[-1][1]))

            elif ((chunked_prefill_enabled or not is_prompt)
                  and block_tables is not None):
                if curr_sliding_window_block == 0:
                    block_table = block_tables[seq_id]
                else:
                    block_table = block_tables[seq_id][
                        -curr_sliding_window_block:]
            self.block_tables.append(block_table)
            # if sparse_index_block and len(sparse_index_block) > 0:
            #     self.sparse_index_blocks.append(sparse_index_block_id)

            self.doc_token_ranges.append(doc_ranges)
            self.docs_hash.append(docs_hash)
            self.kvcache_path.append(kvcache_path)
            self.cached_offset.append(cached_offset)

            # Compute slot mapping.
            is_profile_run = is_block_tables_empty(block_tables)
            start_idx = compute_slot_mapping_start_idx(is_prompt, query_len,
                                                       context_len,
                                                       self.sliding_window)
            if len(chunked_block_ranges) > 0:
                assert start_idx == 0
                for chunked_block_range in chunked_block_ranges:
                    compute_slot_mapping(is_profile_run, self.slot_mapping, seq_id,
                                    chunked_block_range[2], 0, 0,
                                    self.block_size, inter_data.block_tables,
                                    pooling_token_delta=0,
                                    chunked_block_range=chunked_block_range)
            else:
                compute_slot_mapping(is_profile_run, self.slot_mapping, seq_id,
                                    seq_len, context_len, start_idx,
                                    self.block_size, inter_data.block_tables,
                                    pooling_token_delta=pooling_token_delta)

    def _get_graph_runner_block_tables(
            self, num_seqs: int,
            block_tables: List[List[int]]) -> torch.Tensor:
        # The shape of graph_block_tables is
        # [max batch size, max context len // block size].
        max_batch_size, max_blocks = self.runner.graph_block_tables.shape
        assert max_batch_size >= num_seqs

        graph_block_tables = self.runner.graph_block_tables[:num_seqs]
        for i, block_table in enumerate(block_tables):
            if block_table:
                num_blocks = len(block_table)
                if num_blocks <= max_blocks:
                    graph_block_tables[i, :num_blocks] = block_table
                else:
                    # It may be possible to have more blocks allocated due
                    # to lookahead slots of multi-step, however, they are
                    # not used anyway, so can be safely ignored.
                    graph_block_tables[
                        i, :max_blocks] = block_table[:max_blocks]

        return torch.from_numpy(graph_block_tables).to(
            device=self.runner.device, non_blocking=True)

    def build(self, seq_lens: List[int], query_lens: List[int],
              cuda_graph_pad_size: int, batch_size: int,
              seq_len_after_pooling_for_decode: Optional[List[int]] = None):
        """Build attention metadata with on-device tensors.

        Args:
            seq_lens: The maybe padded sequence lengths of the input sequences.
            query_lens: The query lengths of the input sequences.
            cuda_graph_pad_size: The padding size for cuda graph.
                                 -1 if cuda graph is not used.
            batch_size: The maybe padded batch size.
        """
        prefix_cache_hit = any([
            inter_data.prefix_cache_hit
            for inter_data in self.input_builder.inter_data_list
        ])
        for inter_data in self.input_builder.inter_data_list:
            self._add_seq_group(inter_data,
                                self.input_builder.chunked_prefill_enabled,
                                prefix_cache_hit)

        device = self.runner.device
        use_captured_graph = cuda_graph_pad_size != -1

        max_query_len = max(query_lens)
        
        # sparse_index_decode_query_lens = query_lens[self.num_prefills : self.num_prefills+self.num_sparse_index_decodes]
        # if len(sparse_index_decode_query_lens) > 0:
        #     max_sparse_index_decode_query_len = max(sparse_index_decode_query_lens)
        # else:
        #     max_sparse_index_decode_query_len = 1
        # sparse_index_kv_compress_reover_rate = self.input_builder.runner.sparse_index_kv_compress_recover_rate
        
        # TODO[shk]: decode query 长度 大于 1的情况目前暂不考虑
        # 开启监督采样后 page compress 需要禁用
        decode_query_lens = query_lens[self.num_prefills:]
        if len(decode_query_lens) > 0:
            max_decode_query_len = max(decode_query_lens)
        else:
            max_decode_query_len = 1
        
        max_prefill_seq_len = max(self.prefill_seq_lens, default=0)
        # max_sparse_index_decode_seq_len = max(self.use_sparse_index_seq_lens, default=0)
        max_decode_seq_len = max(self.curr_seq_lens, default=0)

        blk_attn_prefill_cu_seqlens_q = list(accumulate(self.chunked_prefill_query_lens, initial=0)) if self.enble_blk_attn else None
        blk_attn_max_prefill_q_len = max(self.chunked_prefill_query_lens, default=0) if self.enble_blk_attn else None
        blk_attn_prefill_seqused_k = self.chunked_prefill_kv_lens if self.enble_blk_attn else None
        blk_attn_max_prefill_kv_len = max(self.chunked_prefill_kv_lens, default=0) if self.enble_blk_attn else None
        blk_attn_prefill_actual_chunked_seqlen_k = self.detail_chunked_prefill_kv_lens if self.enble_blk_attn else None
        blk_attn_prefill_chunk_rotary_offset_positions = self.detail_chunked_prefill_rotary_offsets if self.enble_blk_attn else None
        blk_attn_prefill_cu_num_chunks_k = list(accumulate(self.detail_chunk_nums_prefill, initial=0)) if self.enble_blk_attn else None

        blk_attn_decode_cu_seqlens_q = list(accumulate(self.chunked_decode_query_lens, initial=0)) if self.enble_blk_attn else None
        blk_attn_max_decode_q_len = max(self.chunked_decode_query_lens, default=0) if self.enble_blk_attn else None
        blk_attn_decode_seqused_k = self.chunked_decode_kv_lens if self.enble_blk_attn else None
        blk_attn_max_decode_kv_len = max(self.chunked_decode_kv_lens, default=0) if self.enble_blk_attn else None
        blk_attn_decode_actual_chunked_seqlen_k = self.detail_chunked_decode_kv_lens if self.enble_blk_attn else None
        blk_attn_decode_chunk_rotary_offset_positions = self.detail_chunked_decode_rotary_offsets if self.enble_blk_attn else None
        blk_attn_decode_cu_num_chunks_k = list(accumulate(self.detail_chunk_nums_decode, initial=0)) if self.enble_blk_attn else None

        num_decode_tokens = self.num_decode_tokens
        query_start_loc = list(accumulate(query_lens, initial=0))
        seq_start_loc = list(accumulate(seq_lens, initial=0))
        # if self.enble_blk_attn:
        #     blk_attn_seq_start_loc = list(accumulate(self.blk_attn_prefill_seq_lens, initial=0))
        # else:
        #     blk_attn_seq_start_loc = []

        num_seqs = len(seq_lens)
        if use_captured_graph:
            self.slot_mapping.extend([PAD_SLOT_ID] * cuda_graph_pad_size)
            self.block_tables.extend([] * cuda_graph_pad_size)
            num_decode_tokens = batch_size - self.num_prefill_tokens
            block_tables = self._get_graph_runner_block_tables(
                num_seqs, self.block_tables)
        else:
            block_tables = make_tensor_with_pad(
                self.block_tables,
                pad=0,
                dtype=torch.int,
                device=device,
            )
        assert max_query_len > 0, ("query_lens: {}".format(query_lens))

        page_compress_cache_ids_tensor = None
        if len(self.sparse_index_blocks) > 0:
            assert all([sib >= 0 for sib in self.sparse_index_blocks[:self.num_sparse_index_recomputes]])
            page_compress_cache_ids_tensor = async_tensor_h2d(self.sparse_index_blocks, torch.int32,
                                                            device, self.runner.pin_memory)
        
        num_compressed_pages_tensor = None
        if len(self.num_compressed_pages) > 0:
            assert all([ncp >= 0 for ncp in self.num_compressed_pages[:self.num_sparse_index_recomputes]])
            num_compressed_pages_tensor = async_tensor_h2d(self.num_compressed_pages, torch.int32,
                                                            device, self.runner.pin_memory)
        
        actual_seqlen_tensor = None
        if len(self.actual_curr_seq_lens) > 0:
            actual_seqlen_tensor = async_tensor_h2d(self.actual_curr_seq_lens, torch.int,
                                                            device, self.runner.pin_memory)
        
        actual_max_num_blocks_per_seq = -1
        actual_max_decode_seq_len = -1
        if len(self.actual_curr_seq_lens) > 0:
            actual_max_decode_seq_len = max(self.actual_curr_seq_lens)
            actual_max_num_blocks_per_seq = (actual_max_decode_seq_len + self.block_size - 1) // self.block_size

        page_selector_max_block_size = None
        if self.num_sparse_index_recomputes > 0:
            assert self.num_sparse_index_recomputes <= len(self.curr_seq_lens)
            page_selector_max_block_size = (max(self.curr_seq_lens[:self.num_sparse_index_recomputes]) + self.block_size - 1) // self.block_size

        # TODO[shk]: replace 16
        assert self.block_size == 16
        update_meta_block_id = [id // 16 for id in self.slot_mapping if id % 16 == 15]
        update_meta_block_id_tensor = async_tensor_h2d(update_meta_block_id, torch.int, device, self.runner.pin_memory)

        assert device is not None
        context_lens_tensor = async_tensor_h2d(self.context_lens, torch.int,
                                               device, self.runner.pin_memory)
        seq_len_after_pooling_for_decode_tensor = async_tensor_h2d(
            seq_len_after_pooling_for_decode, torch.int, device,
            self.runner.pin_memory)
        seq_lens_tensor = async_tensor_h2d(seq_lens, torch.int, device,
                                           self.runner.pin_memory)
        slot_mapping_tensor = async_tensor_h2d(self.slot_mapping, torch.long,
                                               device, self.runner.pin_memory)
        query_start_loc_tensor = async_tensor_h2d(query_start_loc, torch.int32,
                                                  device,
                                                  self.runner.pin_memory)
        seq_start_loc_tensor = async_tensor_h2d(seq_start_loc, torch.int32,
                                                device, self.runner.pin_memory)
        placeholder_index_maps = {
            modality: placeholder_map.index_map()
            for modality, placeholder_map in
            self.multimodal_placeholder_maps.items()
        }
        # batch_idx_offset_for_blk_attn_tensor = async_tensor_h2d(
        #     self.batch_idx_offset_for_blk_attn,
        #     torch.int32,
        #     device, self.runner.pin_memory
        # )
        # blk_attn_seq_start_loc_tensor = async_tensor_h2d(
        #     blk_attn_seq_start_loc,
        #     torch.int32,
        #     device, self.runner.pin_memory
        # )

        if blk_attn_prefill_cu_seqlens_q is not None:
            blk_attn_prefill_cu_seqlens_q = async_tensor_h2d(blk_attn_prefill_cu_seqlens_q, torch.int32, device, self.runner.pin_memory)
        if blk_attn_prefill_seqused_k is not None:
            blk_attn_prefill_seqused_k = async_tensor_h2d(blk_attn_prefill_seqused_k, torch.int32, device, self.runner.pin_memory)
        if blk_attn_prefill_actual_chunked_seqlen_k is not None:
            blk_attn_prefill_actual_chunked_seqlen_k = async_tensor_h2d(blk_attn_prefill_actual_chunked_seqlen_k, torch.int32, device, self.runner.pin_memory)
        if blk_attn_prefill_chunk_rotary_offset_positions is not None:
            blk_attn_prefill_chunk_rotary_offset_positions = async_tensor_h2d(blk_attn_prefill_chunk_rotary_offset_positions, torch.int32, device, self.runner.pin_memory)
        if blk_attn_prefill_cu_num_chunks_k is not None:
            blk_attn_prefill_cu_num_chunks_k = async_tensor_h2d(blk_attn_prefill_cu_num_chunks_k, torch.int32, device, self.runner.pin_memory)

        if blk_attn_decode_cu_seqlens_q is not None:
            blk_attn_decode_cu_seqlens_q = async_tensor_h2d(blk_attn_decode_cu_seqlens_q, torch.int32, device, self.runner.pin_memory)
        if blk_attn_decode_seqused_k is not None:
            blk_attn_decode_seqused_k = async_tensor_h2d(blk_attn_decode_seqused_k, torch.int32, device, self.runner.pin_memory)
        if blk_attn_decode_actual_chunked_seqlen_k is not None:
            blk_attn_decode_actual_chunked_seqlen_k = async_tensor_h2d(blk_attn_decode_actual_chunked_seqlen_k, torch.int32, device, self.runner.pin_memory)
        if blk_attn_decode_chunk_rotary_offset_positions is not None:
            blk_attn_decode_chunk_rotary_offset_positions = async_tensor_h2d(blk_attn_decode_chunk_rotary_offset_positions, torch.int32, device, self.runner.pin_memory)
        if blk_attn_decode_cu_num_chunks_k is not None:
            blk_attn_decode_cu_num_chunks_k = async_tensor_h2d(blk_attn_decode_cu_num_chunks_k, torch.int32, device, self.runner.pin_memory)

        return FlashAttentionMetadata(
            num_prefills=self.num_prefills,
            slot_mapping=slot_mapping_tensor,
            num_prefill_tokens=self.num_prefill_tokens,
            num_decode_tokens=num_decode_tokens,
            seq_lens=seq_lens,
            multi_modal_placeholder_index_maps=placeholder_index_maps,
            enable_kv_scales_calculation=True,
            seq_lens_tensor=seq_lens_tensor,
            max_query_len=max_query_len,
            max_decode_query_len=max_decode_query_len,
            max_prefill_seq_len=max_prefill_seq_len,
            max_decode_seq_len=max_decode_seq_len,
            query_start_loc=query_start_loc_tensor,
            seq_start_loc=seq_start_loc_tensor,
            prompt_lens=self.prompt_lens,
            context_lens_tensor=context_lens_tensor,
            block_tables=block_tables,
            use_cuda_graph=use_captured_graph,
            num_sparse_index_recomputes=self.num_sparse_index_recomputes,
            num_sparse_index_recompute_tokens=self.num_sparse_index_recompute_tokens,
            actual_seqlen_tensor=actual_seqlen_tensor,
            page_compress_cache_ids_tensor=page_compress_cache_ids_tensor,
            num_compressed_pages_tensor=num_compressed_pages_tensor,
            actual_max_num_blocks_per_seq=actual_max_num_blocks_per_seq,
            actual_max_decode_seq_len=actual_max_decode_seq_len,
            page_selector_max_block_size=page_selector_max_block_size,
            page_compress_topk=self.page_compress_topk,
            update_meta_block_id_tensor=update_meta_block_id_tensor,
            seq_len_after_pooling_for_decode_tensor=seq_len_after_pooling_for_decode_tensor,
            doc_token_ranges=self.doc_token_ranges,
            docs_hash=self.docs_hash,
            kvcache_path=self.kvcache_path,
            cached_offset=self.cached_offset,
            cache_blend_static_index_cache=None,
            enable_blk_attn=self.enble_blk_attn,
            # batch_idx_offset_for_blk_attn_tensor=batch_idx_offset_for_blk_attn_tensor,
            # blk_attn_seq_start_loc_tensor=blk_attn_seq_start_loc_tensor,
            blk_attn_prefill_cu_seqlens_q=blk_attn_prefill_cu_seqlens_q,
            blk_attn_max_prefill_q_len=blk_attn_max_prefill_q_len,
            blk_attn_prefill_seqused_k=blk_attn_prefill_seqused_k,
            blk_attn_max_prefill_kv_len=blk_attn_max_prefill_kv_len,
            blk_attn_prefill_actual_chunked_seqlen_k=blk_attn_prefill_actual_chunked_seqlen_k,
            blk_attn_prefill_chunk_rotary_offset_positions=blk_attn_prefill_chunk_rotary_offset_positions,
            blk_attn_prefill_cu_num_chunks_k=blk_attn_prefill_cu_num_chunks_k,

            blk_attn_decode_cu_seqlens_q=blk_attn_decode_cu_seqlens_q,
            blk_attn_max_decode_q_len=blk_attn_max_decode_q_len,
            blk_attn_decode_seqused_k=blk_attn_decode_seqused_k,
            blk_attn_max_decode_kv_len=blk_attn_max_decode_kv_len,
            blk_attn_decode_actual_chunked_seqlen_k=blk_attn_decode_actual_chunked_seqlen_k,
            blk_attn_decode_chunk_rotary_offset_positions=blk_attn_decode_chunk_rotary_offset_positions,
            blk_attn_decode_cu_num_chunks_k=blk_attn_decode_cu_num_chunks_k,
        )

class BlendType(IntEnum):
    NAIVE = 0
    CACHE_BLEND_STATIC_INDEX = 1
    CACHE_BLEND_DYNAMIC_INDEX = 2
    CHANNEL_AWARE = 3
    ATTN_AWARE = 4
    CACHE_BLEND_ORIGIN = 5
    RECOMPUTE_LAST_LAYER = 6
    # RECOMPUTE_EVERY_LAYER = 7

@dataclass
class CacheBlendConfig:
    recomp_ratio:float = 0.18
    recomp_layer:int = 1
    val_diff_only:bool = False
    key_diff_only:bool = False
    query_diff_only:bool = False

@dataclass
class CacheBlendDynamicConfig:
    recomp_ratio:float = 0.18
    recomp_layer:int = 1
    val_diff_only:bool = False
    key_diff_only:bool = False
    query_diff_only:bool = False

    recomp_stride:int = 4

@dataclass
class ChannelAwareConfig:
    pivot_stride:int = 8
    sample_layer:int = 2

@dataclass
class AttnAwareConfig:
    topk_ratio:float = 0.18
    num_full_layer:int = 1

@dataclass
class CacheBlendOriginConfig:
    recomp_ratio:float = 0.18
    recomp_layer:int = 1

@dataclass
class RecomputeLastLayerConfig:
    recompute_layer: int = -1

# @dataclass
# class RecomputeEveryLayerConfig:
#     full_recompute_for_last_second_layer: bool = True
#     recomp_ratio:float = 0.18
#     recomp_layer:int = 1

class SparsePrefillType(IntEnum):
    FULL_ATTN = 0
    X_ATTN = 1
    MINFERENCE = 2
    FLEX_PREFILL = 3
    SPARGE_ATTN = 4

@dataclass
class XAttentionConfig:
    stride: int = 8
    threshold: float = 0.8
    block_size: int = 128
    chunk_size: int = 2048

@dataclass
class FlexPrefillConfig:
    block_size: int = 128
    min_budget: int = 1024
    max_budget: int = None
    gamma: float = 0.9
    tau: float = 0

@dataclass
class SpargeAttnConfig:
    simthreshd1: float = 0.6
    cdfthreshd: float = 0.98
    pvthreshd: int = 50

def _sum_all_diagonal_matrix(mat: torch.tensor):
    h, n, m = mat.shape
    # Zero matrix used for padding
    zero_mat = torch.zeros((h, n, n), device=mat.device)
    # pads the matrix on left and right
    mat_padded = torch.cat((zero_mat, mat, zero_mat), -1)
    # Change the strides
    mat_strided = mat_padded.as_strided((h, n, n + m),
                                        (n * (2 * n + m), 2 * n + m + 1, 1))
    # Sums the resulting matrix's columns
    sum_diags = torch.sum(mat_strided, 1)
    return sum_diags[:, 1:]

# @triton.jit
# def triton_calcuate_blk_similarity(
#     x_ptr,        # N * H * D
#     # xmean_ptr,
#     output_pool_ptr,
#     output_blk_table_bitmap_ptr,
#     sim_threshold: tl.float32,
#     sim_blk_size: tl.constexpr,
#     table_blk_size: tl.constexpr,
#     N: tl.constexpr,
#     D: tl.constexpr,
#     BS: tl.constexpr,
#     # fuse_mean: tl.constexpr,
# ):
#     nb, h = tl.program_id(0), tl.program_id(1)
#     NB, H = tl.num_programs(0), tl.num_programs(1)

#     block_offset = nb * BS * H * D + h * D
#     xmask = 

def group_mean_vectorized(tensor, BLK):
    N, H, d = tensor.shape
    num_full_groups = N // BLK
    remainder = N % BLK

    if num_full_groups > 0:
        full_part = tensor[:num_full_groups * BLK]
        full_reshaped = full_part.view(num_full_groups, BLK, H, d)
        full_means = full_reshaped.mean(dim=1)
    else:
        full_means = None

    if remainder > 0:
        remaining_part = tensor[num_full_groups * BLK:]
        remaining_mean = remaining_part.mean(dim=0, keepdim=True)  # (1, d)
    else:
        remaining_mean = None

    if full_means is not None and remaining_mean is not None:
        result = torch.cat([full_means, remaining_mean], dim=0)
    elif full_means is not None:
        result = full_means
    elif remaining_mean is not None:
        result = remaining_mean
    else:
        assert False

    return result

def get_attn_score_unrecompte_idx(
    Q, K, # N * h * d -> h * N * d
          # N * h * d -> h * d * N
    group_size,
    softmax_scale,
    num_key_head,
    head_dim,
    topk_ratio,
):
    key_expand = K.unsqueeze(2).repeat(1, 1, group_size, 1).reshape(-1, num_key_head * group_size, head_dim)
    S = (Q.transpose(0, 1) * softmax_scale) @ key_expand.permute(1, 2, 0)
    del key_expand
    P = F.softmax(S, dim=-1, dtype=torch.float32)
    del S
    score_sum_head_mean = P.sum(dim=-2).mean(dim=0)
    del P
    num_key = score_sum_head_mean.shape[0]
    num_unrecomputed = num_key - int(topk_ratio * num_key)
    unrecompute_idx = torch.topk(score_sum_head_mean, k=num_unrecomputed, largest=False).indices
    del score_sum_head_mean
    return unrecompute_idx

class FlashAttentionImpl(AttentionImpl):
    """
    If the input tensors contain prompt tokens, the layout is as follows:
    |<--------------- num_prefill_tokens ----------------->|	
    |<--prefill_0-->|<--prefill_1-->|...|<--prefill_N-1--->|

    Otherwise, the layout is as follows:	
    |<----------------- num_decode_tokens ------------------>|	
    |<--decode_0-->|..........|<--decode_M-1-->|<--padding-->|

    Generation tokens can contain padding when cuda-graph is used.
    Currently, prompt tokens don't contain any padding.

    The prompts might have different lengths, while the generation tokens
    always have length 1.

    If chunked prefill is enabled, prefill tokens and decode tokens can be
    batched together in a flattened 1D query.

    |<----- num_prefill_tokens ---->|<------- num_decode_tokens --------->|
    |<-prefill_0->|...|<-prefill_N-1->|<--decode_0-->|...|<--decode_M-1-->|

    Currently, cuda graph is disabled for chunked prefill, meaning there's no
    padding between prefill and decode tokens.
    """

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: Optional[List[float]],
        sliding_window: Optional[int],
        kv_cache_dtype: str,
        blocksparse_params: Optional[Dict[str, Any]] = None,
        logits_soft_cap: Optional[float] = None,
        attn_type: str = AttentionType.DECODER,
        use_irope: bool = False,
        layer_idx: int = -1,
        enable_pooling: bool = False,
        enable_blend_prepare: bool = False,
        enable_cache_blend: bool = False,
        pooling_blk_size: Optional[int] = None,
        dual_chunk_attention_config: Optional[Dict[str, Any]] = None,
        enable_attn_out_dump: bool = False,
        enable_last_attn_map_dump: bool = False,
        dump_last_query_len: int = 64,
        num_layers: Optional[int] = None,
    ) -> None:
        if blocksparse_params is not None:
            raise ValueError(
                "FlashAttention does not support block-sparse attention.")
        if use_irope:
            logger.warning(
                "Using irope in V0 is not supported yet, it will fall back "
                "to global attention for long context.")
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        self.sliding_window = ((sliding_window - 1,
                                0) if sliding_window is not None else (-1, -1))
        self.kv_cache_dtype = kv_cache_dtype
        self.vllm_flash_attn_version = get_flash_attn_version(
            requires_alibi=self.alibi_slopes is not None)
        if is_quantized_kv_cache(self.kv_cache_dtype) and (
                not self.kv_cache_dtype.startswith("fp8")
                or not flash_attn_supports_fp8()):
            raise NotImplementedError(
                f"FlashAttention does not support {self.kv_cache_dtype} "
                "kv-cache on this device "
                f"(FA supports fp8 = {flash_attn_supports_fp8()}).")
        if logits_soft_cap is None:
            # In flash-attn, setting logits_soft_cap as 0 means no soft cap.
            logits_soft_cap = 0
        self.logits_soft_cap = logits_soft_cap

        assert self.num_heads % self.num_kv_heads == 0
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads

        support_head_sizes = FlashAttentionBackend.get_supported_head_sizes()
        if head_size not in support_head_sizes:
            raise ValueError(
                f"Head size {head_size} is not supported by FlashAttention. "
                f"Supported head sizes are: {support_head_sizes}.")
        self.attn_type = attn_type

        self.layer_idx = layer_idx
        self.enable_pooling = enable_pooling
        self.enable_blend_prepare = enable_blend_prepare
        self.enable_cache_blend = enable_cache_blend
        self.pooling_blk_size = pooling_blk_size
        self.enable_attn_out_dump = enable_attn_out_dump
        self.enable_last_attn_map_dump = enable_last_attn_map_dump
        self.dump_last_query_len = dump_last_query_len
        from vllm.distributed.parallel_state import get_tensor_model_parallel_rank
        self.tp_rank = get_tensor_model_parallel_rank()
        self.dual_chunk_attention_config = dual_chunk_attention_config
        self.vertical_slash_config = None
        self.sparse_attention_threshold = None
        if self.dual_chunk_attention_config is not None:
            self.vertical_slash_config = self.dual_chunk_attention_config.get(
                "sparse_attention_config", None)
            self.sparse_attention_threshold = dual_chunk_attention_config.get(
                "sparse_attention_threshold", 32768)
            assert self.vertical_slash_config is not None
        
        if self.vertical_slash_config:
            self.vertical_slash_config = {
                int(i): j
                for i, j in self.vertical_slash_config[self.layer_idx].items()
            }
            start_head = self.num_heads * self.tp_rank
            end_head = start_head + self.num_heads
            # 当前层 start_head 到 end_head 的 sparse attention 配置
            self.vertical_slash_config = [
                self.vertical_slash_config[i]
                for i in range(start_head, end_head)
            ]

        self.dump_prefill_qkv = os.getenv("VLLM_FA_DUMP_PREFILL_QKV", None) is not None
        # self.enable_blk_compress_qkv = os.getenv("VLLM_FA_BLK_QKV_COMPRESS", None) is not None
        self.dump_decode_attn = os.getenv("VLLM_FA_DUMP_DECODE_ATTN", None) is not None
        self.dump_decode_which_step = int(os.getenv("VLLM_FA_DUMP_DECODE_STEP", 0))

        self.fa_sparse_decoding_recover_rate = os.getenv("VLLM_FA_DECODE_RECOVER_RATE", None)
        self.dump_cache_blend_path = os.getenv("VLLM_DUMP_CB_PATH", None)
        if self.fa_sparse_decoding_recover_rate is not None:
            self.fa_sparse_decoding_recover_rate = float(self.fa_sparse_decoding_recover_rate)
        self.sparse_prefill_attn_type = SparsePrefillType(int(os.getenv("VLLM_FA_SPARSE_PREFILL", 0)))
        self.sparse_prefill_attn_config = None
        if self.sparse_prefill_attn_type == SparsePrefillType.X_ATTN:
            self.sparse_prefill_attn_config = XAttentionConfig(stride=16)
        if self.sparse_prefill_attn_type == SparsePrefillType.FLEX_PREFILL:
            self.sparse_prefill_attn_config = FlexPrefillConfig()
        if self.sparse_prefill_attn_type == SparsePrefillType.SPARGE_ATTN:
            self.sparse_prefill_attn_config = SpargeAttnConfig()
        
        self.sparse_index_block_size = 64
        self.arange = torch.arange(self.sparse_index_block_size)
        self.last_q_mask = (self.arange[None, :, None]
                            >= self.arange[None, None, :])
        self.int32_max = torch.iinfo(torch.int32).max
        self.int32_min = torch.iinfo(torch.int32).min

        self.num_layers = num_layers
        self.blend_prepare_save_residual_layer = None
        self.blend_prepare_for_last_layer_recompute = os.getenv("VLLM_FA_BLEND_PREPARE_FOR_LAST_LAYER", None) is not None
        if self.blend_prepare_for_last_layer_recompute:
            self.blend_prepare_save_residual_layer = [self.num_layers - 2]
        self.blend_prepare_for_cache_blend_dynamic = int(os.getenv("VLLM_FA_BLEND_PREPARE_FOR_CB_DYN", -1))
        if self.blend_prepare_for_cache_blend_dynamic != -1:
            self.blend_prepare_save_residual_layer = list(range(0, self.num_layers, self.blend_prepare_for_cache_blend_dynamic))

        self.blend_type = BlendType(int(os.getenv("VLLM_FA_BLEND_TYPE", 0)))
        self.blend_config = None
        if self.blend_type == BlendType.CACHE_BLEND_STATIC_INDEX:
            self.blend_config = CacheBlendConfig(key_diff_only=True)
        if self.blend_type == BlendType.CACHE_BLEND_DYNAMIC_INDEX:
            assert self.blend_prepare_for_cache_blend_dynamic is not None
            self.blend_config = CacheBlendDynamicConfig(recomp_stride=self.blend_prepare_for_cache_blend_dynamic)
        if self.blend_type == BlendType.CHANNEL_AWARE:
            self.blend_config = ChannelAwareConfig()
        if self.blend_type == BlendType.ATTN_AWARE:
            self.blend_config = AttnAwareConfig()
        if self.blend_type == BlendType.CACHE_BLEND_ORIGIN:
            self.blend_config = CacheBlendOriginConfig()
        if self.blend_type == BlendType.RECOMPUTE_LAST_LAYER:
            assert self.num_layers is not None
            self.blend_config = RecomputeLastLayerConfig(recompute_layer=self.num_layers - 2)
        # if self.blend_type == BlendType.RECOMPUTE_EVERY_LAYER:
        #     assert self.num_layers is not None
        #     self.blend_config = RecomputeEveryLayerConfig()

        # 参数目前硬编码
        # TODO[shk]: 传入参数
        # self.blend_rotary = RotaryEmbedding(
        #     head_size=self.head_size,
        #     rotary_dim=self.head_size,
        #     max_position_embeddings=32768,
        #     base=1000000.0,
        #     is_neox_style=True,
        #     dtype=torch.bfloat16,
        # )
            
        self._cos_sin_cache = self._init_cos_sin_cache()

    def _init_cos_sin_cache(
        self,
        base = 1000000.0,
        rotary_dim = 128,
        max_position_embeddings = 32768,
    ):
        inv_freq = 1.0 / (base**(torch.arange(0, rotary_dim, 2, dtype=torch.float) / rotary_dim))
        t = torch.arange(max_position_embeddings, dtype=torch.float)
        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cos_sin_cache = torch.cat((cos, sin), dim=-1).to(dtype=torch.float16)
        assert cos_sin_cache.stride(-1) == 1
        return cos_sin_cache
    
    def apply_rotary(
        self,
        key:torch.Tensor,
        offset
    ):
        if offset == 0:
            return key
        tensor_len = key.shape[0]
        positions = torch.tensor([offset]*tensor_len, dtype=torch.int64, device=key.device)
        # key, _ = self.blend_rotary.forward_cuda(
        #     positions=positions,
        #     query=key,
        #     key=None,
        # )
        del positions
        return key

    def forward(
        self,
        layer: AttentionLayer,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: FlashAttentionMetadata,
        output: Optional[torch.Tensor] = None,
        key_meta_cache: Optional[torch.Tensor] = None,
        block_count_gpu_cache: Optional[torch.Tensor] = None,
        block_index_gpu_cache: Optional[torch.Tensor] = None,
        column_count_gpu_cache: Optional[torch.Tensor] = None,
        column_index_gpu_cache: Optional[torch.Tensor] = None,
        residual_to_cache: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Forward pass with FlashAttention.

        Args:
            query: shape = [num_tokens, num_heads, head_size]
            key: shape = [num_tokens, num_kv_heads, head_size]
            value: shape = [num_tokens, num_kv_heads, head_size]
            output: shape = [num_tokens, num_heads, head_size]
            kv_cache = [2, num_blocks, block_size, num_kv_heads, head_size]
                NOTE: kv_cache will be an empty tensor with shape [0]
                for profiling run.
            attn_metadata: Metadata for attention.
        NOTE: It in-place updates the output tensor.
        NOTE: FP8 quantization, flash-attn expect the size of
              {q,k,v}_descale to be (num_sequences, num_kv_heads).
              We use torch's .expand() to avoid duplicating values
        """
        assert output is not None, "Output tensor must be provided."

        # NOTE(woosuk): FlashAttention2 does not support FP8 KV cache.
        if not flash_attn_supports_fp8() or output.dtype != torch.bfloat16:
            assert (
                layer._k_scale_float == 1.0 and layer._v_scale_float == 1.0), (
                    "key/v_scale is only supported in FlashAttention 3 with "
                    "base dtype bfloat16")

        attn_type = self.attn_type
        if (attn_type == AttentionType.ENCODER
                and (not attn_metadata.is_all_encoder_attn_metadata_set)):
            raise AttributeError("Encoder attention requires setting "
                                 "encoder metadata attributes.")
        elif (attn_type == AttentionType.ENCODER_DECODER
              and (not attn_metadata.is_all_cross_attn_metadata_set)):
            raise AttributeError("Encoder/decoder cross-attention "
                                 "requires setting cross-attention "
                                 "metadata attributes.")

        kv_cache_dtype: str = self.kv_cache_dtype
        softmax_scale: float = self.scale
        window_size = self.sliding_window
        alibi_slopes: Optional[torch.Tensor] = self.alibi_slopes
        logits_soft_cap: Optional[float] = self.logits_soft_cap
        fp8_attention = kv_cache_dtype.startswith("fp8")

        if fp8_attention and not flash_attn_supports_fp8():
            raise NotImplementedError(
                "FlashAttention does not support FP8 kv-cache on this device.")

        (num_prefill_query_tokens, num_prefill_kv_tokens,
         num_decode_query_tokens) = \
            get_num_prefill_decode_query_kv_tokens(attn_metadata, attn_type)
        
        key_after_pooling = []
        val_after_pooling = []
        # cache_blend_recomp_every_layer_kept_key_index = None
        if prefill_meta := attn_metadata.prefill_metadata:
            if self.enable_blend_prepare:
                cu_seqlens_cpu = prefill_meta.seq_start_loc.cpu().tolist()
                doc_token_ranges = prefill_meta.doc_token_ranges
                docs_hash = prefill_meta.docs_hash
                kvcache_paths = prefill_meta.kvcache_path
                assert (len(doc_token_ranges) + 1 == len(cu_seqlens_cpu) and
                        len(docs_hash) + 1 == len(cu_seqlens_cpu) and
                        len(kvcache_paths) + 1 == len(cu_seqlens_cpu))

                for idx in range(0, len(cu_seqlens_cpu) - 1):
                    cur_doc_ranges = doc_token_ranges[idx]
                    doc_hash = docs_hash[idx]
                    kvcache_path = kvcache_paths[idx]

                    start_idx = cu_seqlens_cpu[idx]
                    end_idx = cu_seqlens_cpu[idx:idx+2][-1]
                    seqlen = end_idx - start_idx

                    if cur_doc_ranges is None:
                        continue
                        
                    assert len(cur_doc_ranges) == len(doc_hash) and len(cur_doc_ranges) == len(kvcache_path)
                    if self.blend_prepare_save_residual_layer is not None:
                        assert residual_to_cache is not None
                        assert self.num_layers is not None
                        cur_key = key[start_idx : end_idx]
                        cur_val = value[start_idx : end_idx]
                        if self.layer_idx in self.blend_prepare_save_residual_layer:
                            cur_q = query[start_idx : end_idx]
                            cur_res = residual_to_cache[start_idx : end_idx]
                        for doc_range, h, kvpath in zip(cur_doc_ranges, doc_hash, kvcache_path):
                            if kvpath is None:
                                continue
                            tp_kvpath = kvpath + f"_tp{self.tp_rank}.hdf5"
                            ds, de = doc_range
                            import h5py
                            with h5py.File(tp_kvpath, 'a') as f:
                                f.create_dataset(f'K_{h}_{self.layer_idx}', data=cur_key[ds:de].clone().detach().float().cpu().numpy())
                                f.create_dataset(f'V_{h}_{self.layer_idx}', data=cur_val[ds:de].clone().detach().float().cpu().numpy())
                                if self.layer_idx in self.blend_prepare_save_residual_layer:
                                    f.create_dataset(f'Q_{h}_{self.layer_idx}', data=cur_q[ds:de].clone().detach().float().cpu().numpy())
                                    f.create_dataset(f'R_{h}_{self.layer_idx}', data=cur_res[ds:de].clone().detach().float().cpu().numpy())
                    else:
                        cur_key = key[start_idx : end_idx]
                        cur_val = value[start_idx : end_idx]
                        cur_q = query[start_idx : end_idx]
                        for doc_range, h, kvpath in zip(cur_doc_ranges, doc_hash, kvcache_path):
                            if kvpath is None:
                                continue
                            tp_kvpath = kvpath + f"_tp{self.tp_rank}.hdf5"
                            ds, de = doc_range
                            import h5py
                            with h5py.File(tp_kvpath, 'a') as f:
                                f.create_dataset(f'K_{h}_{self.layer_idx}', data=cur_key[ds:de].clone().detach().float().cpu().numpy())
                                f.create_dataset(f'V_{h}_{self.layer_idx}', data=cur_val[ds:de].clone().detach().float().cpu().numpy())
                                f.create_dataset(f'Q_{h}_{self.layer_idx}', data=cur_q[ds:de].clone().detach().float().cpu().numpy())

            if self.enable_cache_blend and self.blend_type == BlendType.NAIVE and self.layer_idx > 0:
                cu_seqlens_cpu = prefill_meta.seq_start_loc.cpu().tolist()
                doc_token_ranges = prefill_meta.doc_token_ranges
                docs_hash = prefill_meta.docs_hash
                kvcache_paths = prefill_meta.kvcache_path
                cached_offsets = prefill_meta.cached_offset
                assert (len(doc_token_ranges) + 1 == len(cu_seqlens_cpu) and
                        len(docs_hash) + 1 == len(cu_seqlens_cpu) and
                        len(kvcache_paths) + 1 == len(cu_seqlens_cpu) and
                        len(cached_offsets) + 1 == len(cu_seqlens_cpu))
                
                for idx in range(0, len(cu_seqlens_cpu) - 1):
                    cur_doc_ranges = doc_token_ranges[idx]
                    doc_hash = docs_hash[idx]
                    kvcache_path = kvcache_paths[idx]
                    cur_offset = cached_offsets[idx]

                    start_idx = cu_seqlens_cpu[idx]
                    end_idx = cu_seqlens_cpu[idx:idx+2][-1]
                    seqlen = end_idx - start_idx

                    if kvcache_path is None:
                        continue

                    assert (len(cur_doc_ranges) == len(doc_hash) and 
                            len(cur_doc_ranges) == len(kvcache_path) and
                            len(cur_doc_ranges) == len(cur_offset))
                    import h5py
                    for doc_id, (doc_range, h, kvpath, offset) in enumerate(zip(cur_doc_ranges, doc_hash, kvcache_path, cur_offset)):
                        if kvpath is None or doc_id == 0:
                            continue
                        tp_kvpath = kvpath + f"_tp{self.tp_rank}.hdf5"
                        ds, de = doc_range
                        key_tag = f'K_{h}_{self.layer_idx}'
                        value_tag = f'V_{h}_{self.layer_idx}'
                        with h5py.File(tp_kvpath, 'a') as f:
                            key_np = np.array(f[key_tag])
                            cached_key = torch.from_numpy(key_np).to(device=key.device, dtype=key.dtype)
                            val_np = np.array(f[value_tag])
                            cached_val = torch.from_numpy(val_np).to(device=value.device, dtype=value.dtype)
                        key[start_idx + ds : start_idx + de] = self.apply_rotary(
                            cached_key, ds - offset
                        )
                        del cached_key
                        value[start_idx + ds : start_idx + de] = cached_val
                        del cached_val
            
            if (self.enable_cache_blend and self.blend_type == BlendType.CACHE_BLEND_STATIC_INDEX and
                self.layer_idx >= self.blend_config.recomp_layer):
                cu_seqlens_cpu = prefill_meta.seq_start_loc.cpu().tolist()
                doc_token_ranges = prefill_meta.doc_token_ranges
                docs_hash = prefill_meta.docs_hash
                kvcache_paths = prefill_meta.kvcache_path
                cached_offsets = prefill_meta.cached_offset
                cache_blend_indice_for_batches = None

                if self.layer_idx == self.blend_config.recomp_layer:
                    assert prefill_meta.cache_blend_static_index_cache is None
                elif self.layer_idx > self.blend_config.recomp_layer:
                    assert prefill_meta.cache_blend_static_index_cache is not None
                    cache_blend_indice_for_batches = prefill_meta.cache_blend_static_index_cache
                
                recomp_index_for_batches = []
                for idx in range(0, len(cu_seqlens_cpu) - 1):
                    cur_doc_ranges = doc_token_ranges[idx]
                    doc_hash = docs_hash[idx]
                    kvcache_path = kvcache_paths[idx]
                    cur_offset = cached_offsets[idx]
                    cur_cache_blend_indice = None
                    if self.layer_idx > self.blend_config.recomp_layer:
                        cur_cache_blend_indice = cache_blend_indice_for_batches[idx]

                    start_idx = cu_seqlens_cpu[idx]
                    end_idx = cu_seqlens_cpu[idx:idx+2][-1]
                    seqlen = end_idx - start_idx

                    if kvcache_path is None:
                        if self.layer_idx == self.blend_config.recomp_layer:
                            recomp_index_for_batches.append(None)
                        continue
                    
                    import h5py
                    recomp_index_per_batch = []
                    for doc_id,(doc_range, h, kvpath, offset) in enumerate(zip(cur_doc_ranges, doc_hash, kvcache_path, cur_offset)):
                        if kvpath is None or doc_id == 0:
                            if self.layer_idx == self.blend_config.recomp_layer:
                                recomp_index_per_batch.append(None)
                            continue
                        tp_kvpath = kvpath + f"_tp{self.tp_rank}.hdf5"
                        ds, de = doc_range
                        key_tag = f'K_{h}_{self.layer_idx}'
                        value_tag = f'V_{h}_{self.layer_idx}'
                        # q_tag = f'Q_{h}_{self.layer_idx}'
                        with h5py.File(tp_kvpath, 'a') as f:
                            key_np = np.array(f[key_tag])
                            cached_key = self.apply_rotary(
                                torch.from_numpy(key_np).to(device=key.device, dtype=key.dtype),
                                ds - offset
                            )
                            # q_np = np.array(f[q_tag])
                            # cached_q = self.apply_rotary(
                            #     torch.from_numpy(q_np).to(device=query.device, dtype=query.dtype),
                            #     ds - offset
                            # )
                            val_np = np.array(f[value_tag])
                            cached_val = torch.from_numpy(val_np).to(device=value.device, dtype=value.dtype)
                        
                        if self.layer_idx == self.blend_config.recomp_layer:
                            ntopk = (de - ds) - int(self.blend_config.recomp_ratio * (de - ds))
                            if self.blend_config.val_diff_only:
                                diff = torch.sum(
                                    (value[start_idx + ds : start_idx + de] - cached_val)**2, 
                                    dim=[1,2]
                                )
                            elif self.blend_config.key_diff_only:
                                diff = torch.sum(
                                    (key[start_idx + ds : start_idx + de] - cached_key)**2, 
                                    dim=[1,2]
                                )
                            # elif self.blend_config.query_diff_only:
                            #     diff = torch.sum(
                            #         (query[start_idx + ds : start_idx + de] - cached_q)**2, 
                            #         dim=[1,2]
                            #     )
                            else:
                                diff = torch.sum(
                                    (key[start_idx + ds : start_idx + de] - cached_key)**2 + (value[start_idx + ds : start_idx + de] - cached_val) ** 2, 
                                    dim=[1,2]
                                )
                            recomp_index_per_batch.append(torch.topk(diff, k=ntopk, largest=False).indices)
                            del diff
                            # del cached_q
                            del cached_key
                            del cached_val
                        else:
                            cache_blend_idx = cur_cache_blend_indice[doc_id]
                            key[cache_blend_idx + (start_idx + ds)] = cached_key[cache_blend_idx]
                            value[cache_blend_idx + (start_idx + ds)] = cached_val[cache_blend_idx]
                            del cached_key
                            del cached_val
                    
                    if self.layer_idx == self.blend_config.recomp_layer:
                        recomp_index_for_batches.append(recomp_index_per_batch)
                if self.layer_idx == self.blend_config.recomp_layer:
                    prefill_meta.cache_blend_static_index_cache = recomp_index_for_batches

                # if (self.dump_cache_blend_path is not None) and kv_cache.numel() > 0 and (prefill_meta.block_tables is not None):
                #     print("================== CACHE_BLEND DUMP KV ================")
                #     import h5py
                #     file_name = f"{self.dump_cache_blend_path}/tensor_{self.tp_rank}.hdf5"
                #     with h5py.File(file_name, 'a') as f:
                #         f.create_dataset(f'K_{self.layer_idx}', data=key.clone().detach().float().cpu().numpy())
                #         f.create_dataset(f'V_{self.layer_idx}', data=value.clone().detach().float().cpu().numpy())

            if (self.enable_cache_blend and self.blend_type == BlendType.RECOMPUTE_LAST_LAYER and
                self.layer_idx <= self.blend_config.recompute_layer):
                cu_seqlens_cpu = prefill_meta.seq_start_loc.cpu().tolist()
                doc_token_ranges = prefill_meta.doc_token_ranges
                docs_hash = prefill_meta.docs_hash
                kvcache_paths = prefill_meta.kvcache_path
                cached_offsets = prefill_meta.cached_offset
                need_full_recompute = (self.layer_idx == self.blend_config.recompute_layer)
                for idx in range(0, len(cu_seqlens_cpu) - 1):
                    cur_doc_ranges = doc_token_ranges[idx]
                    doc_hash = docs_hash[idx]
                    kvcache_path = kvcache_paths[idx]
                    cur_offset = cached_offsets[idx]

                    start_idx = cu_seqlens_cpu[idx]
                    end_idx = cu_seqlens_cpu[idx:idx+2][-1]
                    seqlen = end_idx - start_idx

                    if kvcache_path is None:
                        continue

                    import h5py
                    for doc_range, h, kvpath, offset in zip(cur_doc_ranges, doc_hash, kvcache_path, cur_offset):
                        if kvpath is None:
                            continue
                        tp_kvpath = kvpath + f"_tp{self.tp_rank}.hdf5"
                        ds, de = doc_range
                        key_tag = f'K_{h}_{self.layer_idx}'
                        value_tag = f'V_{h}_{self.layer_idx}'
                        if need_full_recompute:
                            q_tag = f'Q_{h}_{self.layer_idx}'
                            res_tag = f'R_{h}_{self.layer_idx}'
                        with h5py.File(tp_kvpath, 'a') as f:
                            key_np = np.array(f[key_tag])
                            cached_key = self.apply_rotary(
                                torch.from_numpy(key_np).to(device=key.device, dtype=key.dtype),
                                ds - offset
                            )
                            val_np = np.array(f[value_tag])
                            cached_val = torch.from_numpy(val_np).to(device=value.device, dtype=value.dtype)
                            if need_full_recompute:
                                assert residual_to_cache is not None
                                q_np = np.array(f[q_tag])
                                cached_q = self.apply_rotary(
                                    torch.from_numpy(q_np).to(device=query.device, dtype=query.dtype),
                                    ds - offset
                                )
                                res_np = np.array(f[res_tag])
                                cached_res = torch.from_numpy(res_np).to(device=residual_to_cache.device, dtype=residual_to_cache.dtype)
                            key[start_idx + ds : start_idx + de] = cached_key
                            value[start_idx + ds : start_idx + de] = cached_val
                            del cached_key
                            del cached_val
                            if need_full_recompute:
                                query[start_idx + ds : start_idx + de] = cached_q
                                residual_to_cache[start_idx + ds : start_idx + de] = cached_res
                                del cached_q
                                del cached_res

            if (self.enable_cache_blend and self.blend_type == BlendType.CACHE_BLEND_ORIGIN and 
                self.layer_idx >= self.blend_config.recomp_layer):
                cu_seqlens_cpu = prefill_meta.seq_start_loc.cpu().tolist()
                doc_token_ranges = prefill_meta.doc_token_ranges
                docs_hash = prefill_meta.docs_hash
                kvcache_paths = prefill_meta.kvcache_path
                cached_offsets = prefill_meta.cached_offset
                cache_blend_indice_for_batches = None

                if self.layer_idx == self.blend_config.recomp_layer:
                    assert prefill_meta.cache_blend_static_index_cache is None
                elif self.layer_idx > self.blend_config.recomp_layer:
                    assert prefill_meta.cache_blend_static_index_cache is not None
                    cache_blend_indice_for_batches = prefill_meta.cache_blend_static_index_cache
                
                recomp_index_for_batches = []
                for idx in range(0, len(cu_seqlens_cpu) - 1):
                    cur_doc_ranges = doc_token_ranges[idx]
                    doc_hash = docs_hash[idx]
                    kvcache_path = kvcache_paths[idx]
                    cur_offset = cached_offsets[idx]
                    cur_cache_blend_indice = None
                    if self.layer_idx > self.blend_config.recomp_layer:
                        cur_cache_blend_indice = cache_blend_indice_for_batches[idx]
                    
                    start_idx = cu_seqlens_cpu[idx]
                    end_idx = cu_seqlens_cpu[idx:idx+2][-1]
                    seqlen = end_idx - start_idx

                    if kvcache_path is None:
                        if self.layer_idx == self.blend_config.recomp_layer:
                            recomp_index_for_batches.append(None)
                        continue
                    
                    import h5py
                    if self.layer_idx == self.blend_config.recomp_layer:
                        shrinked_doc_range = []
                        total_doc_len = 0
                        last_len = 0
                        for (doc_range, kvpath) in zip(cur_doc_ranges, kvcache_path):
                            if kvpath is None:
                                shrinked_doc_range.append(None)
                                continue
                            ds, de = doc_range
                            doc_len = (de - ds)
                            total_doc_len += doc_len
                            shrinked_doc_range.append((last_len, last_len + doc_len))
                            last_len += doc_len
                        _, H, d = key.shape

                        ntopk = total_doc_len - int(self.blend_config.recomp_ratio * total_doc_len)
                        doc_keys_diff = torch.empty(
                            (total_doc_len,),
                            dtype=key.dtype,
                            device=key.device,
                        )
                    
                        for doc_id,(doc_range, h, kvpath, offset, sdoc_range) in enumerate(zip(cur_doc_ranges, doc_hash, kvcache_path, cur_offset, shrinked_doc_range)):
                            if kvpath is None:
                                continue
                            tp_kvpath = kvpath + f"_tp{self.tp_rank}.hdf5"
                            ds, de = doc_range
                            sds, sde = sdoc_range
                            key_tag = f'K_{h}_{self.layer_idx}'
                            with h5py.File(tp_kvpath, 'a') as f:
                                key_np = np.array(f[key_tag])
                                cached_key = self.apply_rotary(
                                    torch.from_numpy(key_np).to(device=key.device, dtype=key.dtype),
                                    ds - offset
                                )
                                doc_keys_diff[sds:sde] = torch.sum(
                                    (key[start_idx + ds : start_idx + de] - cached_key)**2, 
                                    dim=[1,2]
                                )
                                del cached_key

                        topk_tokens_idx = torch.topk(doc_keys_diff, k=ntopk, largest=False).indices
                        del doc_keys_diff
                        recomp_index_per_batch = []
                        for (doc_range, sdoc_range, kvpath) in zip(cur_doc_ranges, shrinked_doc_range, kvcache_path):
                            if kvpath is None:
                                recomp_index_per_batch.append(None)
                                continue
                            ds, _ = doc_range
                            sds, sde = sdoc_range
                            recomp_index_per_batch.append(topk_tokens_idx[(topk_tokens_idx >= sds) & (topk_tokens_idx < sde)] - sds)
                        
                        recomp_index_for_batches.append(recomp_index_per_batch)
                    else:
                        for doc_id,(doc_range, h, kvpath, offset) in enumerate(zip(cur_doc_ranges, doc_hash, kvcache_path, cur_offset)):
                            if kvpath is None or doc_id == 0:
                                continue
                            tp_kvpath = kvpath + f"_tp{self.tp_rank}.hdf5"
                            ds, de = doc_range
                            key_tag = f'K_{h}_{self.layer_idx}'
                            value_tag = f'V_{h}_{self.layer_idx}'
                            cache_blend_idx = cur_cache_blend_indice[doc_id]
                            if cache_blend_idx.shape[0] == 0:
                                continue
                            with h5py.File(tp_kvpath, 'a') as f:
                                key_np = np.array(f[key_tag])
                                cached_key = self.apply_rotary(
                                    torch.from_numpy(key_np).to(device=key.device, dtype=key.dtype),
                                    ds - offset
                                )
                                val_np = np.array(f[value_tag])
                                cached_val = torch.from_numpy(val_np).to(device=value.device, dtype=value.dtype)
                                key[cache_blend_idx + (start_idx + ds)] = cached_key[cache_blend_idx]
                                value[cache_blend_idx + (start_idx + ds)] = cached_val[cache_blend_idx]
                                del cached_key
                                del cached_val

                if self.layer_idx == self.blend_config.recomp_layer:
                    prefill_meta.cache_blend_static_index_cache = recomp_index_for_batches

            if (self.enable_cache_blend and self.blend_type == BlendType.CACHE_BLEND_DYNAMIC_INDEX and
                self.layer_idx >= self.blend_config.recomp_layer):
                full_compute_mode = ((self.layer_idx - self.blend_config.recomp_layer + 1) % self.blend_config.recomp_stride == 0)
                calc_compute_idx_mode = ((self.layer_idx - self.blend_config.recomp_layer) % self.blend_config.recomp_stride == 0)
                use_idx_mode = not (full_compute_mode or calc_compute_idx_mode)

                cu_seqlens_cpu = prefill_meta.seq_start_loc.cpu().tolist()
                doc_token_ranges = prefill_meta.doc_token_ranges
                docs_hash = prefill_meta.docs_hash
                kvcache_paths = prefill_meta.kvcache_path
                cached_offsets = prefill_meta.cached_offset
                cache_blend_indice_for_batches = None

                if use_idx_mode or full_compute_mode:
                    assert prefill_meta.cache_blend_static_index_cache is not None
                    cache_blend_indice_for_batches = prefill_meta.cache_blend_static_index_cache

                if full_compute_mode:
                    for idx in range(0, len(cu_seqlens_cpu) - 1):
                        cur_doc_ranges = doc_token_ranges[idx]
                        doc_hash = docs_hash[idx]
                        kvcache_path = kvcache_paths[idx]
                        cur_offset = cached_offsets[idx]
                        cur_cache_blend_indice = cache_blend_indice_for_batches[idx]

                        start_idx = cu_seqlens_cpu[idx]
                        end_idx = cu_seqlens_cpu[idx:idx+2][-1]
                        seqlen = end_idx - start_idx

                        if kvcache_path is None:
                            continue

                        import h5py
                        for doc_id,(doc_range, h, kvpath, offset) in enumerate(zip(cur_doc_ranges, doc_hash, kvcache_path, cur_offset)):
                            if kvpath is None or doc_id == 0:
                                continue
                            tp_kvpath = kvpath + f"_tp{self.tp_rank}.hdf5"
                            ds, de = doc_range
                            key_tag = f'K_{h}_{self.layer_idx}'
                            value_tag = f'V_{h}_{self.layer_idx}'
                            q_tag = f'Q_{h}_{self.layer_idx}'
                            res_tag = f'R_{h}_{self.layer_idx}'
                            with h5py.File(tp_kvpath, 'a') as f:
                                key_np = np.array(f[key_tag])
                                cached_key = self.apply_rotary(
                                    torch.from_numpy(key_np).to(device=key.device, dtype=key.dtype),
                                    ds - offset
                                )
                                q_np = np.array(f[q_tag])
                                cached_q = self.apply_rotary(
                                    torch.from_numpy(q_np).to(device=query.device, dtype=query.dtype),
                                    ds - offset
                                )
                                val_np = np.array(f[value_tag])
                                cached_val = torch.from_numpy(val_np).to(device=value.device, dtype=value.dtype)
                                res_np = np.array(f[res_tag])
                                cached_res = torch.from_numpy(res_np).to(device=residual_to_cache.device, dtype=residual_to_cache.dtype)
                            
                            cache_blend_idx = cur_cache_blend_indice[doc_id]
                            key[cache_blend_idx + (start_idx + ds)] = cached_key[cache_blend_idx]
                            value[cache_blend_idx + (start_idx + ds)] = cached_val[cache_blend_idx]
                            query[cache_blend_idx + (start_idx + ds)] = cached_q[cache_blend_idx]
                            residual_to_cache[cache_blend_idx + (start_idx + ds)] = cached_res[cache_blend_idx]
                            del cached_q
                            del cached_key
                            del cached_val
                            del cached_res
                else:
                    recomp_index_for_batches = []
                    for idx in range(0, len(cu_seqlens_cpu) - 1):
                        cur_doc_ranges = doc_token_ranges[idx]
                        doc_hash = docs_hash[idx]
                        kvcache_path = kvcache_paths[idx]
                        cur_offset = cached_offsets[idx]
                        cur_cache_blend_indice = None

                        if use_idx_mode:
                            cur_cache_blend_indice = cache_blend_indice_for_batches[idx]
                        
                        start_idx = cu_seqlens_cpu[idx]
                        end_idx = cu_seqlens_cpu[idx:idx+2][-1]
                        seqlen = end_idx - start_idx

                        if kvcache_path is None:
                            if calc_compute_idx_mode:
                                recomp_index_for_batches.append(None)
                            continue

                        import h5py
                        recomp_index_per_batch = []
                        for doc_id,(doc_range, h, kvpath, offset) in enumerate(zip(cur_doc_ranges, doc_hash, kvcache_path, cur_offset)):
                            if kvpath is None or doc_id == 0:
                                if calc_compute_idx_mode:
                                    recomp_index_per_batch.append(None)
                                continue
                            tp_kvpath = kvpath + f"_tp{self.tp_rank}.hdf5"
                            ds, de = doc_range
                            key_tag = f'K_{h}_{self.layer_idx}'
                            value_tag = f'V_{h}_{self.layer_idx}'
                            with h5py.File(tp_kvpath, 'a') as f:
                                key_np = np.array(f[key_tag])
                                cached_key = self.apply_rotary(
                                    torch.from_numpy(key_np).to(device=key.device, dtype=key.dtype),
                                    ds - offset
                                )
                                val_np = np.array(f[value_tag])
                                cached_val = torch.from_numpy(val_np).to(device=value.device, dtype=value.dtype)
                            
                            if calc_compute_idx_mode:
                                ntopk = (de - ds) - int(self.blend_config.recomp_ratio * (de - ds))
                                diff = torch.sum(
                                    (key[start_idx + ds : start_idx + de] - cached_key)**2, 
                                    dim=[1,2]
                                )
                                recomp_index_per_batch.append(torch.topk(diff, k=ntopk, largest=False).indices)
                                key[recomp_index_per_batch[-1] + (start_idx + ds)] = cached_key[recomp_index_per_batch[-1]]
                                value[recomp_index_per_batch[-1] + (start_idx + ds)] = cached_val[recomp_index_per_batch[-1]]
                                del diff
                            else:
                                cache_blend_idx = cur_cache_blend_indice[doc_id]
                                key[cache_blend_idx + (start_idx + ds)] = cached_key[cache_blend_idx]
                                value[cache_blend_idx + (start_idx + ds)] = cached_val[cache_blend_idx]
                            
                            del cached_key
                            del cached_val
                        if calc_compute_idx_mode:
                            recomp_index_for_batches.append(recomp_index_per_batch)
                    if calc_compute_idx_mode:
                        if prefill_meta.cache_blend_static_index_cache is not None:
                            del prefill_meta.cache_blend_static_index_cache
                        prefill_meta.cache_blend_static_index_cache = recomp_index_for_batches
            
            if (self.enable_cache_blend and self.blend_type == BlendType.CHANNEL_AWARE and
                self.layer_idx >= self.blend_config.sample_layer):
                cu_seqlens_cpu = prefill_meta.seq_start_loc.cpu().tolist()
                doc_token_ranges = prefill_meta.doc_token_ranges
                docs_hash = prefill_meta.docs_hash
                kvcache_paths = prefill_meta.kvcache_path
                cached_offsets = prefill_meta.cached_offset

                for idx in range(0, len(cu_seqlens_cpu) - 1):
                    cur_doc_ranges = doc_token_ranges[idx]
                    doc_hash = docs_hash[idx]
                    kvcache_path = kvcache_paths[idx]
                    cur_offset = cached_offsets[idx]

                    start_idx = cu_seqlens_cpu[idx]
                    end_idx = cu_seqlens_cpu[idx:idx+2][-1]
                    seqlen = end_idx - start_idx

                    if kvcache_path is None:
                        continue
                    
                    import h5py
                    for (doc_range, h, kvpath, offset) in zip(cur_doc_ranges, doc_hash, kvcache_path, cur_offset):
                        if kvpath is None:
                            continue
                        tp_kvpath = kvpath + f"_tp{self.tp_rank}.hdf5"
                        ds, de = doc_range
                        key_tag = f'K_{h}_{self.layer_idx}'
                        value_tag = f'V_{h}_{self.layer_idx}'
                        with h5py.File(tp_kvpath, 'a') as f:
                            key_np = np.array(f[key_tag])
                            cached_key = self.apply_rotary(
                                torch.from_numpy(key_np).to(device=key.device, dtype=key.dtype),
                                ds - offset
                            )
                            val_np = np.array(f[value_tag])
                            cached_val = torch.from_numpy(val_np).to(device=value.device, dtype=value.dtype)
                        
                        _, H, d = key.shape
                        unrecompute_mask = torch.ones(de - ds, dtype=torch.bool)
                        pivot_idx = torch.arange(0, (de - ds), self.blend_config.pivot_stride)
                        unrecompute_mask[pivot_idx] = False
                        unrecompute_idx = (torch.arange(0, (de - ds)))[unrecompute_mask]
                        # key_pooling = group_mean_vectorized(
                        #     key[start_idx + ds : start_idx + de],
                        #     self.blend_config.pivot_stride,
                        # )
                        # val_pooling = group_mean_vectorized(
                        #     value[start_idx + ds : start_idx + de],
                        #     self.blend_config.pivot_stride,
                        # )
                        # key_delta = key_pooling - cached_key[pivot_idx]
                        # key_delta_expand = (key_delta.unsqueeze(1).expand(-1, self.blend_config.pivot_stride, -1, -1).reshape(-1, H, d))[:de - ds]
                        # val_delta = val_pooling - cached_val[pivot_idx]
                        # val_delta_expand = (val_delta.unsqueeze(1).expand(-1, self.blend_config.pivot_stride, -1, -1).reshape(-1, H, d))[:de - ds]
                        # recitified_cache_key = cached_key + key_delta_expand
                        # recitified_cache_val = cached_val + val_delta_expand
                        # key[unrecompute_idx + (start_idx + ds)] = recitified_cache_key[unrecompute_idx]
                        # value[unrecompute_idx + (start_idx + ds)] = recitified_cache_val[unrecompute_idx]

                        key[unrecompute_idx + (start_idx + ds)] = cached_key[unrecompute_idx]
                        value[unrecompute_idx + (start_idx + ds)] = cached_val[unrecompute_idx]

                        # del val_pooling
                        # del key_pooling
                        del unrecompute_idx
                        del pivot_idx
                        del unrecompute_mask
                        del cached_key
                        del cached_val
            
            if (self.enable_cache_blend and self.blend_type == BlendType.ATTN_AWARE and
                self.layer_idx >= self.blend_config.num_full_layer):
                cu_seqlens_cpu = prefill_meta.seq_start_loc.cpu().tolist()
                doc_token_ranges = prefill_meta.doc_token_ranges
                docs_hash = prefill_meta.docs_hash
                kvcache_paths = prefill_meta.kvcache_path
                cached_offsets = prefill_meta.cached_offset
                cache_blend_indice_for_batches = None

                if self.layer_idx == self.blend_config.num_full_layer:
                    assert prefill_meta.cache_blend_static_index_cache is None
                elif self.layer_idx > self.blend_config.num_full_layer:
                    assert prefill_meta.cache_blend_static_index_cache is not None
                    cache_blend_indice_for_batches = prefill_meta.cache_blend_static_index_cache

                num_query_head = query.shape[-2]
                num_key_head = key.shape[-2]
                head_dim = query.shape[-1]
                group_size = num_query_head // num_key_head
                
                recomp_index_for_batches = []
                for idx in range(0, len(cu_seqlens_cpu) - 1):
                    cur_doc_ranges = doc_token_ranges[idx]
                    doc_hash = docs_hash[idx]
                    kvcache_path = kvcache_paths[idx]
                    cur_offset = cached_offsets[idx]
                    cur_cache_blend_indice = None
                    if self.layer_idx > self.blend_config.num_full_layer:
                        cur_cache_blend_indice = cache_blend_indice_for_batches[idx]

                    start_idx = cu_seqlens_cpu[idx]
                    end_idx = cu_seqlens_cpu[idx:idx+2][-1]
                    seqlen = end_idx - start_idx

                    if kvcache_path is None:
                        if self.layer_idx == self.blend_config.num_full_layer:
                            recomp_index_for_batches.append(None)
                        continue

                    import h5py
                    assert kvcache_path[-1] is not None
                    recomp_index_per_batch = []

                    question_query = None
                    if self.layer_idx == self.blend_config.num_full_layer:
                        question_query = query[start_idx + cur_doc_ranges[-1][1]:]
                    
                    for doc_id,(doc_range, h, kvpath, offset) in enumerate(zip(cur_doc_ranges, doc_hash, kvcache_path, cur_offset)):
                        if kvpath is None:
                            if self.layer_idx == self.blend_config.num_full_layer:
                                recomp_index_per_batch.append(None)
                            continue
                        tp_kvpath = kvpath + f"_tp{self.tp_rank}.hdf5"
                        ds, de = doc_range
                        key_tag = f'K_{h}_{self.layer_idx}'
                        value_tag = f'V_{h}_{self.layer_idx}'
                        q_tag = f'Q_{h}_{self.layer_idx}'
                        with h5py.File(tp_kvpath, 'a') as f:
                            key_np = np.array(f[key_tag])
                            cached_key = self.apply_rotary(
                                torch.from_numpy(key_np).to(device=key.device, dtype=key.dtype),
                                ds - offset
                            )
                            val_np = np.array(f[value_tag])
                            cached_val = torch.from_numpy(val_np).to(device=value.device, dtype=value.dtype)
                        if self.layer_idx == self.blend_config.num_full_layer:
                            recomp_index_per_batch.append(get_attn_score_unrecompte_idx(
                                question_query,
                                # cached_key,
                                key[start_idx + ds : start_idx + de],
                                group_size,
                                softmax_scale,
                                num_key_head,
                                head_dim,
                                self.blend_config.topk_ratio,
                            ))
                        else:
                            cache_blend_idx = cur_cache_blend_indice[doc_id]
                            key[cache_blend_idx + (start_idx + ds)] = cached_key[cache_blend_idx]
                            value[cache_blend_idx + (start_idx + ds)] = cached_val[cache_blend_idx]
                        del cached_key
                        del cached_val
                    
                    if self.layer_idx == self.blend_config.num_full_layer:
                        recomp_index_for_batches.append(recomp_index_per_batch)
                    
                    if question_query is not None:
                        del question_query
                
                if self.layer_idx == self.blend_config.num_full_layer:
                    prefill_meta.cache_blend_static_index_cache = recomp_index_for_batches

            # if (self.enable_cache_blend and self.blend_type == BlendType.RECOMPUTE_EVERY_LAYER and
            #     self.layer_idx >= self.blend_config.recomp_layer):
            #     num_query_head = query.shape[-2]
            #     num_key_head = key.shape[-2]
            #     group_size = num_query_head // num_key_head
            #     kept_indice = []

            #     cu_seqlens_cpu = prefill_meta.seq_start_loc.cpu().tolist()
            #     doc_token_ranges = prefill_meta.doc_token_ranges
            #     docs_hash = prefill_meta.docs_hash
            #     kvcache_paths = prefill_meta.kvcache_path
            #     cached_offsets = prefill_meta.cached_offset

            #     for idx in range(0, len(cu_seqlens_cpu) - 1):
            #         cur_doc_ranges = doc_token_ranges[idx]
            #         doc_hash = docs_hash[idx]
            #         kvcache_path = kvcache_paths[idx]
            #         cur_offset = cached_offsets[idx]

            #         start_idx = cu_seqlens_cpu[idx]
            #         end_idx = cu_seqlens_cpu[idx:idx+2][-1]
            #         seqlen = end_idx - start_idx

            #         import h5py
            #         for doc_id,(doc_range, h, kvpath, offset) in enumerate(zip(cur_doc_ranges, doc_hash, kvcache_path, cur_offset)):
            #             if kvpath is None or doc_id == 0:
            #                 continue
            #             tp_kvpath = kvpath + f"_tp{self.tp_rank}.hdf5"
            #             ds, de = doc_range
            #             key_tag = f'K_{h}_{self.layer_idx}'
            #             value_tag = f'V_{h}_{self.layer_idx}'
            #             with h5py.File(tp_kvpath, 'a') as f:
            #                 key_np = np.array(f[key_tag])
            #                 cached_key = self.apply_rotary(
            #                     torch.from_numpy(key_np).to(device=key.device, dtype=key.dtype),
            #                     ds - offset
            #                 )
            #                 val_np = np.array(f[value_tag])
            #                 cached_val = torch.from_numpy(val_np).to(device=value.device, dtype=value.dtype)
            #             ntopk = (de - ds) - int(self.blend_config.recomp_ratio * (de - ds))
            #             diff = torch.sum(
            #                 (key[start_idx + ds : start_idx + de] - cached_key)**2, 
            #                 dim=[1,2]
            #             )
            #             doc_kept_indice = (torch.topk(diff, k=ntopk, largest=False).indices + start_idx + ds).unsqueeze(1)
            #             topks = doc_kept_indice.shape[0]
            #             doc_kept_indice = ((torch.arange(0, topks * group_size) % group_size).reshape(topks, -1) + doc_kept_indice * group_size).flatten()
            #             kept_indice.append(doc_kept_indice)
            #             del diff
            #             del cached_key
            #             del cached_val
            #     cache_blend_recomp_every_layer_kept_key_index = torch.concat(kept_indice)

            if self.enable_pooling:
                cu_seqlens_cpu = prefill_meta.seq_start_loc.cpu().tolist()
                doc_token_ranges = prefill_meta.doc_token_ranges
                assert len(doc_token_ranges) + 1 == len(cu_seqlens_cpu)
                for idx in range(0, len(cu_seqlens_cpu) - 1):
                    cur_doc_ranges = doc_token_ranges[idx]
                    start_idx = cu_seqlens_cpu[idx]
                    end_idx = cu_seqlens_cpu[idx:idx+2][-1]
                    seqlen = end_idx - start_idx
                    if cur_doc_ranges is None:  
                        cur_pooling_key = group_mean_vectorized(key[start_idx : end_idx], self.pooling_blk_size)
                        cur_pooling_val = group_mean_vectorized(value[start_idx : end_idx], self.pooling_blk_size)
                        
                        key_after_pooling.append(cur_pooling_key)
                        val_after_pooling.append(cur_pooling_val)
                    else:
                        cur_key = key[start_idx : end_idx]
                        cur_val = value[start_idx : end_idx]
                        if cur_doc_ranges[0][0] != 0:
                            key_after_pooling.append(cur_key[:cur_doc_ranges[0][0]])
                            val_after_pooling.append(cur_val[:cur_doc_ranges[0][0]])

                        for (ds, de) in cur_doc_ranges:
                            key_after_pooling.append(
                                group_mean_vectorized(cur_key[ds:de], self.pooling_blk_size)
                            )
                            val_after_pooling.append(
                                group_mean_vectorized(cur_val[ds:de], self.pooling_blk_size)
                            )

                        if cur_doc_ranges[-1][1] != seqlen:
                            key_after_pooling.append(cur_key[cur_doc_ranges[-1][1]:])
                            val_after_pooling.append(cur_val[cur_doc_ranges[-1][1]:])
            else:
                key_after_pooling.append(key[:num_prefill_kv_tokens])
                val_after_pooling.append(value[:num_prefill_kv_tokens])
            
        if decode_meta := attn_metadata.decode_metadata:
            key_after_pooling.append(key[num_prefill_kv_tokens:])
            val_after_pooling.append(value[num_prefill_kv_tokens:])

        if kv_cache.numel() > 0:
            key_cache = kv_cache[0]
            value_cache = kv_cache[1]
            # We skip updating the KV cache under two conditions:
            #  a. When the Attention Type is ENCODER. In this phase, we compute
            #     only the encoder attention without updating the cache.
            #  b. When both Key and Value are None. This occurs during
            #     cross-attention computation in the decoding phase, where the
            #     KV cache is already populated with the cross-attention
            #     tensor. Thus, we skip cache updates during this time.
            if (attn_type != AttentionType.ENCODER) and (key is not None) and (
                    value is not None):
                if attn_type == AttentionType.ENCODER_DECODER:
                    # Update cross-attention KV cache (prefill-only)
                    updated_slot_mapping = attn_metadata.cross_slot_mapping
                else:
                    # Update self-attention KV cache (prefill/decode)
                    updated_slot_mapping = attn_metadata.slot_mapping

                if self.enable_pooling:
                    assert len(key_after_pooling) > 0 and len(val_after_pooling) > 0
                    key_after_pooling_tensor = torch.cat(key_after_pooling, dim=0)
                    val_after_pooling_tensor = torch.cat(val_after_pooling, dim=0)
                    torch.ops._C_cache_ops.reshape_and_cache_flash(
                        key_after_pooling_tensor,
                        val_after_pooling_tensor,
                        kv_cache[0],
                        kv_cache[1],
                        updated_slot_mapping.flatten(),  # type: ignore[union-attr]
                        kv_cache_dtype,
                        layer._k_scale,
                        layer._v_scale,
                    )
                else:
                    # Reshape the input keys and values and store them in the cache.
                    # If kv_cache is not provided, the new key and value tensors are
                    # not cached. This happens during the initial memory
                    # profiling run.
                    torch.ops._C_cache_ops.reshape_and_cache_flash(
                        key,
                        value,
                        kv_cache[0],
                        kv_cache[1],
                        updated_slot_mapping.flatten(),  # type: ignore[union-attr]
                        kv_cache_dtype,
                        layer._k_scale,
                        layer._v_scale,
                    )

                if (key_meta_cache is not None and key_meta_cache.size(0) > 0 and
                    attn_type == AttentionType.DECODER and attn_metadata.update_meta_block_id_tensor.size(0) != 0):
                    block_keys = key_cache[attn_metadata.update_meta_block_id_tensor, ...]
                    # num_block, num_kv_head, 2, head_dim
                    key_meta_cache[attn_metadata.update_meta_block_id_tensor, :, 0, :] = torch.max(block_keys, dim=-3).values
                    key_meta_cache[attn_metadata.update_meta_block_id_tensor, :, 1, :] = torch.min(block_keys, dim=-3).values

                if fp8_attention:
                    kv_cache = kv_cache.view(torch.float8_e4m3fn)
                    key_cache = key_cache.view(torch.float8_e4m3fn)
                    value_cache = value_cache.view(torch.float8_e4m3fn)

        if fp8_attention:
            num_tokens, num_heads, head_size = query.shape
            query, _ = ops.scaled_fp8_quant(
                query.reshape(
                    (num_tokens, num_heads * head_size)).contiguous(),
                layer._q_scale)
            query = query.reshape((num_tokens, num_heads, head_size))

        decode_query = query[num_prefill_query_tokens:]
        decode_output = output[num_prefill_query_tokens:]
        # QKV for prefill.
        query = query[:num_prefill_query_tokens]
        prefill_output = output[:num_prefill_query_tokens]
        assert query.shape[0] == num_prefill_query_tokens
        assert decode_query.shape[0] == num_decode_query_tokens

        if prefill_meta := attn_metadata.prefill_metadata:
            # Prompt run.
            if (kv_cache.numel() == 0 or prefill_meta.block_tables is None
                    or prefill_meta.block_tables.numel() == 0):
                # normal attention
                # When block_tables are not filled, it means q and k are the
                # prompt, and they have the same length.
                q_seq_start_loc, q_seq_len, k_seq_start_loc, k_seq_len = \
                    _get_query_key_seq_metadata(prefill_meta, True, attn_type)

                # if prefill_meta.enable_blk_attn:
                #     q_seq_start_loc = prefill_meta.blk_attn_seq_start_loc_tensor
                #     k_seq_start_loc = prefill_meta.blk_attn_seq_start_loc_tensor
                #     q_seq_len = prefill_meta.blk_attn_max_prefill_q_len
                #     k_seq_len = prefill_meta.blk_attn_max_prefill_kv_len

                key = key[:num_prefill_kv_tokens]
                value = value[:num_prefill_kv_tokens]

                if fp8_attention:
                    num_kv_tokens, num_kv_heads, head_size = key.shape

                    key, _ = ops.scaled_fp8_quant(
                        key.reshape((num_kv_tokens,
                                     num_kv_heads * head_size)).contiguous(),
                        layer._k_scale)
                    key = key.reshape((num_kv_tokens, num_kv_heads, head_size))

                    value, _ = ops.scaled_fp8_quant(
                        value.reshape((num_kv_tokens,
                                       num_kv_heads * head_size)).contiguous(),
                        layer._v_scale)
                    value = value.reshape(
                        (num_kv_tokens, num_kv_heads, head_size))
                
                # if self.enable_blk_compress_qkv:
                #     BLK_SIZE = 16

                #     pass

                descale_shape = (q_seq_start_loc.shape[0] - 1, key.shape[1])
                if self.sparse_prefill_attn_type == SparsePrefillType.FULL_ATTN or (
                    kv_cache.numel() == 0 or (prefill_meta.block_tables is None)):
                    # start_attn = torch.cuda.Event(enable_timing=True)
                    # end_attn = torch.cuda.Event(enable_timing=True)
                    # start_attn.record()
                    # TODO:[shk]
                    flash_attn_varlen_func(
                        q=query,
                        k=key,
                        v=value,
                        cu_seqlens_q=q_seq_start_loc,
                        cu_seqlens_k=k_seq_start_loc,
                        max_seqlen_q=q_seq_len,
                        max_seqlen_k=k_seq_len,
                        # batch_idx_offset_for_blk_attn=prefill_meta.batch_idx_offset_for_blk_attn_tensor if prefill_meta.enable_blk_attn else None,
                        softmax_scale=softmax_scale,
                        causal=_get_causal_option(attn_type),
                        window_size=window_size,
                        alibi_slopes=alibi_slopes,
                        softcap=logits_soft_cap,
                        out=prefill_output,
                        fa_version=self.vllm_flash_attn_version,
                        q_descale=layer._q_scale.expand(descale_shape),
                        k_descale=layer._k_scale.expand(descale_shape),
                        v_descale=layer._v_scale.expand(descale_shape),
                    )
                    # end_attn.record()
                    # torch.cuda.synchronize()
                    # attn_duration = start_attn.elapsed_time(end_attn)
                    # logger.info(f"===================== FA COST: attn:{attn_duration}ms ========================")
                    
                    if (self.dump_cache_blend_path is not None) and kv_cache.numel() > 0 and (prefill_meta.block_tables is not None):
                        print("================== CACHE_BLEND DUMP KV FA ================")
                        import h5py
                        file_name = f"{self.dump_cache_blend_path}/tensor_{self.tp_rank}.hdf5"
                        with h5py.File(file_name, 'a') as f:
                            f.create_dataset(f'K_{self.layer_idx}', data=key.clone().detach().float().cpu().numpy())
                            f.create_dataset(f'V_{self.layer_idx}', data=value.clone().detach().float().cpu().numpy())

                    if self.dump_prefill_qkv and kv_cache.numel() > 0 and (prefill_meta.block_tables is not None):
                        print("================== DUMP PREFILL QKV ================")
                        assert query.shape[0] == key.shape[0] and query.shape[0] == value.shape[0]
                        import h5py
                        file_name = f"/data/shanhaikang.shk/vllm/prefill_qkv_dump_rand1/tensor_{self.tp_rank}.hdf5"
                        with h5py.File(file_name, 'a') as f:
                            f.create_dataset(f'Q_{self.layer_idx}', data=query.clone().detach().float().cpu().numpy())
                            f.create_dataset(f'V_{self.layer_idx}', data=key.clone().detach().float().cpu().numpy())
                            f.create_dataset(f'K_{self.layer_idx}', data=value.clone().detach().float().cpu().numpy())
                elif self.sparse_prefill_attn_type == SparsePrefillType.X_ATTN:
                    cu_seqlens_q = prefill_meta.seq_start_loc
                    cu_seqlens_q_cpu = cu_seqlens_q.cpu().tolist()

                    # Because x_attn can only handle 1 batch_size now, we should do iteration here.
                    qlen = None
                    seqlen = None
                    for query_idx in range(0, len(cu_seqlens_q_cpu) - 1):
                        qs = cu_seqlens_q_cpu[query_idx]
                        qe = cu_seqlens_q_cpu[query_idx : query_idx + 2][-1]
                        qlen = qe - qs

                        current_q = query[qs : qe] # seq_len, num_head, head_dim
                        group_size = current_q.size(-2) // key.size(-2)
                        cur_key = key[qs : qe]
                        cur_value = value[qs : qe]
                        # x_attn can not handle GQA now, we should repeat key & value
                        current_q = current_q.permute(1, 0, 2).unsqueeze(0)
                        cur_key = torch.repeat_interleave(cur_key, repeats=group_size, dim=1).permute(1, 0, 2).unsqueeze(0)
                        cur_value = torch.repeat_interleave(cur_value, repeats=group_size, dim=1).permute(1, 0, 2).unsqueeze(0)

                        # copy partial_output to output
                        # print(f"============== stride={self.sparse_prefill_attn_config.stride} ==========")
                        partial_output = Xattention_prefill(
                            query_states=current_q,
                            key_states=cur_key,
                            value_states=cur_value,
                            stride=self.sparse_prefill_attn_config.stride,
                            threshold=self.sparse_prefill_attn_config.threshold,
                            block_size=self.sparse_prefill_attn_config.block_size,
                            chunk_size=self.sparse_prefill_attn_config.chunk_size,
                        ).squeeze(0)
                        # print(f"================= partial_output shape: {partial_output.shape} ===========")
                        output[qs : qe] = partial_output.permute(1, 0, 2)                        

                elif self.sparse_prefill_attn_type == SparsePrefillType.FLEX_PREFILL:
                    # Flex Prefill
                    max_seq_len = q_seq_len
                    block_size = self.sparse_prefill_attn_config.block_size
                    
                    if max_seq_len <= max(2 * block_size, math.ceil(self.sparse_prefill_attn_config.min_budget / block_size) * block_size):
                        flash_attn_varlen_func(
                            q=query,
                            k=key,
                            v=value,
                            cu_seqlens_q=q_seq_start_loc,
                            cu_seqlens_k=k_seq_start_loc,
                            max_seqlen_q=q_seq_len,
                            max_seqlen_k=k_seq_len,
                            softmax_scale=softmax_scale,
                            causal=_get_causal_option(attn_type),
                            window_size=window_size,
                            alibi_slopes=alibi_slopes,
                            softcap=logits_soft_cap,
                            out=prefill_output,
                            fa_version=self.vllm_flash_attn_version,
                            q_descale=layer._q_scale.expand(descale_shape),
                            k_descale=layer._k_scale.expand(descale_shape),
                            v_descale=layer._v_scale.expand(descale_shape),
                        )
                    else:
                        assert logits_soft_cap == 0.0
                        cu_seqlens_q = prefill_meta.seq_start_loc
                        cu_seqlens_q_cpu = cu_seqlens_q.cpu().tolist()

                        for query_idx in range(0, len(cu_seqlens_q_cpu) - 1):
                            qs = cu_seqlens_q_cpu[query_idx]
                            qe = cu_seqlens_q_cpu[query_idx : query_idx + 2][-1]
                            current_q = query[qs : qe] # seq_len, num_head, head_dim
                            # retrieve key & value from kv_cache
                            cur_key = key[qs : qe]
                            cur_value = value[qs : qe]

                            current_q = current_q.unsqueeze(0)
                            cur_key = cur_key.unsqueeze(0)
                            cur_value = cur_value.unsqueeze(0)
                            partial_output = flex_prefill_attention(
                                q=current_q,
                                k=cur_key,
                                v=cur_value,
                                gamma=self.sparse_prefill_attn_config.gamma,
                                tau=self.sparse_prefill_attn_config.tau,
                                min_budget=self.sparse_prefill_attn_config.min_budget,
                                max_budget=self.sparse_prefill_attn_config.max_budget,
                                softmax_scale=softmax_scale,
                                block_size=self.sparse_prefill_attn_config.block_size,
                            ).squeeze(0)
                            output[qs : qe] = partial_output
                elif self.sparse_prefill_attn_type == SparsePrefillType.SPARGE_ATTN:
                    # Sparge Attention
                    assert logits_soft_cap == 0.0
                    cu_seqlens_q = prefill_meta.seq_start_loc
                    cu_seqlens_q_cpu = cu_seqlens_q.cpu().tolist()
                    
                    max_seq_len = q_seq_len
                    has_short_query = False
                    for query_idx in range(0, len(cu_seqlens_q_cpu) - 1):
                        qs = cu_seqlens_q_cpu[query_idx]
                        qe = cu_seqlens_q_cpu[query_idx : query_idx + 2][-1]
                        if qe - qs < 128:
                            has_short_query = True
                            break
                    
                    if has_short_query:
                        flash_attn_varlen_func(
                            q=query,
                            k=key,
                            v=value,
                            cu_seqlens_q=q_seq_start_loc,
                            cu_seqlens_k=k_seq_start_loc,
                            max_seqlen_q=q_seq_len,
                            max_seqlen_k=k_seq_len,
                            softmax_scale=softmax_scale,
                            causal=_get_causal_option(attn_type),
                            window_size=window_size,
                            alibi_slopes=alibi_slopes,
                            softcap=logits_soft_cap,
                            out=prefill_output,
                            fa_version=self.vllm_flash_attn_version,
                            q_descale=layer._q_scale.expand(descale_shape),
                            k_descale=layer._k_scale.expand(descale_shape),
                            v_descale=layer._v_scale.expand(descale_shape),
                        )
                    else:
                        for query_idx in range(0, len(cu_seqlens_q_cpu) - 1):
                            qs = cu_seqlens_q_cpu[query_idx]
                            qe = cu_seqlens_q_cpu[query_idx : query_idx + 2][-1]

                            current_q = query[qs : qe] # seq_len, num_head, head_dim
                            group_size = current_q.size(-2) // key.size(-2)
                            cur_key = key[qs : qe]
                            cur_value = value[qs : qe]

                            cur_key = torch.repeat_interleave(cur_key, repeats=group_size, dim=1).unsqueeze(0)
                            cur_value = torch.repeat_interleave(cur_value, repeats=group_size, dim=1).unsqueeze(0)
                            current_q = current_q.unsqueeze(0)

                            partial_output = spas_sage2_attn_meansim_cuda(
                                q=current_q,
                                k=cur_key,
                                v=cur_value,
                                is_causal=True,
                                simthreshd1=self.sparse_prefill_attn_config.simthreshd1,
                                cdfthreshd=self.sparse_prefill_attn_config.cdfthreshd,
                                pvthreshd=self.sparse_prefill_attn_config.pvthreshd,
                                tensor_layout="NHD"
                            ).squeeze(0)
                            output[qs : qe] = partial_output
            else:
                if self.sparse_prefill_attn_type == SparsePrefillType.FULL_ATTN:
                    # prefix-enabled attention
                    assert attn_type == AttentionType.DECODER, (
                        "Only decoder-only models support prefix caching")
                    assert prefill_meta.seq_lens is not None
                    assert prefill_meta.query_start_loc is not None
                    max_seq_len = max(prefill_meta.seq_lens)
                    descale_shape = (prefill_meta.query_start_loc.shape[0] - 1,
                                    key.shape[1])

                    if prefill_meta.enable_blk_attn:
                        assert query.dtype == torch.float16
                        flash_attn_varlen_func(  # noqa
                            q=query,
                            k=key_cache,
                            v=value_cache,
                            cu_seqlens_q=prefill_meta.blk_attn_prefill_cu_seqlens_q,
                            max_seqlen_q=prefill_meta.blk_attn_max_prefill_q_len,
                            seqused_k=prefill_meta.blk_attn_prefill_seqused_k,
                            max_seqlen_k=prefill_meta.blk_attn_max_prefill_kv_len,
                            softmax_scale=softmax_scale,
                            causal=True,
                            window_size=window_size,
                            alibi_slopes=alibi_slopes,
                            block_table=prefill_meta.block_tables,

                            actual_chunked_seqlen_k=prefill_meta.blk_attn_prefill_actual_chunked_seqlen_k,
                            chunk_rotray_offset_positions=prefill_meta.blk_attn_prefill_chunk_rotary_offset_positions,
                            cu_num_chunks_k=prefill_meta.blk_attn_prefill_cu_num_chunks_k,
                            cos_sin_cache=self._cos_sin_cache,
                            local_key=key,
                            local_value=value,
                            local_cu_seqlen_k=prefill_meta.blk_attn_prefill_cu_seqlens_q,

                            softcap=logits_soft_cap,
                            out=prefill_output,
                            fa_version=self.vllm_flash_attn_version,
                            q_descale=layer._q_scale.expand(descale_shape),
                            k_descale=layer._k_scale.expand(descale_shape),
                            v_descale=layer._v_scale.expand(descale_shape),
                        )
                    else:
                        flash_attn_varlen_func(  # noqa
                            q=query,
                            k=key_cache,
                            v=value_cache,
                            cu_seqlens_q=prefill_meta.query_start_loc,
                            max_seqlen_q=prefill_meta.max_query_len,
                            seqused_k=prefill_meta.seq_lens_tensor,
                            max_seqlen_k=max_seq_len,
                            softmax_scale=softmax_scale,
                            causal=True,
                            window_size=window_size,
                            alibi_slopes=alibi_slopes,
                            block_table=prefill_meta.block_tables,
                            softcap=logits_soft_cap,
                            out=prefill_output,
                            fa_version=self.vllm_flash_attn_version,
                            q_descale=layer._q_scale.expand(descale_shape),
                            k_descale=layer._k_scale.expand(descale_shape),
                            v_descale=layer._v_scale.expand(descale_shape),
                        )

                    if self.enable_attn_out_dump:
                        print(f"================== ENABLE ATTN OUT DUMP {self.tp_rank} {self.layer_idx} ================")
                        assert len(prefill_meta.seq_lens) == 1
                        import h5py
                        file_name = f"/data/shanhaikang.shk/vllm/attn_out_dump/tensor_{self.tp_rank}.hdf5"
                        with h5py.File(file_name, 'a') as f:
                            f.create_dataset(f'{self.layer_idx}', data=prefill_output.clone().detach().float().cpu().numpy())
                    
                    if self.enable_last_attn_map_dump:
                        print(f"================== ENABLE LAST ATTN MAP DUMP {self.tp_rank} {self.layer_idx} ================")
                        assert len(prefill_meta.seq_lens) == 1
                        current_seq_len = prefill_meta.seq_lens[0]
                        current_block_table = prefill_meta.block_tables[0]
                        key = key_cache[current_block_table].view(-1, *key_cache.shape[-2:])[: current_seq_len]
                        last_query = query[-self.dump_last_query_len:, ...].transpose(0, 1)
                        last_attn_map = last_query @ key.permute(1, 2, 0)
                        file_name = f"/data/shanhaikang.shk/vllm/attn_map_dump2/tensor_{self.tp_rank}.hdf5"
                        import h5py
                        with h5py.File(file_name, 'a') as f:
                            f.create_dataset(f'{self.layer_idx}', data=last_attn_map.float().cpu().numpy())
                elif self.sparse_prefill_attn_type == SparsePrefillType.MINFERENCE:
                    assert self.sparse_attention_threshold is not None
                    assert self.vertical_slash_config is not None

                    min_seq_len = min(prefill_meta.seq_lens)
                    max_seq_len = max(prefill_meta.seq_lens)
                    if min_seq_len <= self.sparse_attention_threshold:
                        flash_attn_varlen_func(  # noqa
                            q=query,
                            k=key_cache,
                            v=value_cache,
                            cu_seqlens_q=prefill_meta.query_start_loc,
                            max_seqlen_q=prefill_meta.max_query_len,
                            seqused_k=prefill_meta.seq_lens_tensor,
                            max_seqlen_k=max_seq_len,
                            softmax_scale=softmax_scale,
                            causal=True,
                            window_size=window_size,
                            alibi_slopes=alibi_slopes,
                            block_table=prefill_meta.block_tables,
                            softcap=logits_soft_cap,
                            out=prefill_output,
                            fa_version=self.vllm_flash_attn_version,
                        )
                    else:
                        cu_seqlens_q = prefill_meta.query_start_loc
                        cu_seqlens_q_cpu = cu_seqlens_q.cpu().tolist()
                        assert (prefill_meta.seq_lens_tensor is not None and 
                                prefill_meta.seq_lens_tensor.shape[0] == len(cu_seqlens_q_cpu) - 1 and
                                prefill_meta.block_tables is not None and
                                prefill_meta.block_tables.shape[0] == len(cu_seqlens_q_cpu) - 1 and
                                prefill_meta.seq_lens is not None and
                                len(prefill_meta.seq_lens) == len(cu_seqlens_q_cpu) - 1)

                        for query_idx in range(0, len(cu_seqlens_q_cpu) - 1):
                            qs = cu_seqlens_q_cpu[query_idx]
                            qe = cu_seqlens_q_cpu[query_idx : query_idx + 2][-1]

                            current_q = query[qs : qe] # seq_len, num_head, head_dim
                            current_block_table = prefill_meta.block_tables[query_idx]
                            current_seq_len = prefill_meta.seq_lens[query_idx]
                            assert current_q.size(-2) % key_cache.size(-2) == 0
                            group_size = current_q.size(-2) // key_cache.size(-2)

                            key = key_cache[current_block_table].view(-1, *key_cache.shape[-2:])[: current_seq_len]
                            value = value_cache[current_block_table].view(-1, *value_cache.shape[-2:])[: current_seq_len]

                            current_q = current_q.permute(1, 0, 2).unsqueeze(0)
                            key = torch.repeat_interleave(key, repeats=group_size, dim=1).permute(1, 0, 2).unsqueeze(0)
                            value = torch.repeat_interleave(value, repeats=group_size, dim=1).permute(1, 0, 2).unsqueeze(0)

                            partial_output = Minference_prefill(
                                query_states=current_q,
                                key_states=key,
                                value_states=value,
                                vertical_slash_config=self.vertical_slash_config,
                            ).squeeze(0)
                            # print(f"================= partial_output shape: {partial_output.shape} ===========")
                            output[qs : qe] = partial_output.permute(1, 0, 2)
                elif self.sparse_prefill_attn_type == SparsePrefillType.X_ATTN:
                    # X-Attention
                    # For flash_attn: query -> nhd, while x_attn: query -> 1hnd
                    print("XXXXXXXXXXX ============== ENABLE X_ATTN ============")
                    cu_seqlens_q = prefill_meta.query_start_loc
                    cu_seqlens_q_cpu = cu_seqlens_q.cpu().tolist()
                    assert (prefill_meta.seq_lens_tensor is not None and 
                            prefill_meta.seq_lens_tensor.shape[0] == len(cu_seqlens_q_cpu) - 1 and
                            prefill_meta.block_tables is not None and
                            prefill_meta.block_tables.shape[0] == len(cu_seqlens_q_cpu) - 1 and
                            prefill_meta.seq_lens is not None and
                            len(prefill_meta.seq_lens) == len(cu_seqlens_q_cpu) - 1)

                    # Because x_attn can only handle 1 batch_size now, we should do iteration here.
                    qlen = None
                    seqlen = None
                    for query_idx in range(0, len(cu_seqlens_q_cpu) - 1):
                        qs = cu_seqlens_q_cpu[query_idx]
                        qe = cu_seqlens_q_cpu[query_idx : query_idx + 2][-1]
                        qlen = qe - qs

                        current_q = query[qs : qe] # seq_len, num_head, head_dim
                        current_block_table = prefill_meta.block_tables[query_idx]
                        current_seq_len = prefill_meta.seq_lens[query_idx]
                        seqlen = current_seq_len
                        assert current_q.size(-2) % key_cache.size(-2) == 0
                        group_size = current_q.size(-2) // key_cache.size(-2)
                        # kvcache_block_size = key_cache.size(1)
                        # retrieve key & value from kv_cache
                        key = key_cache[current_block_table].view(-1, *key_cache.shape[-2:])[: current_seq_len]
                        value = value_cache[current_block_table].view(-1, *value_cache.shape[-2:])[: current_seq_len]
                        # x_attn can not handle GQA now, we should repeat key & value
                        current_q = current_q.permute(1, 0, 2).unsqueeze(0)
                        key = torch.repeat_interleave(key, repeats=group_size, dim=1).permute(1, 0, 2).unsqueeze(0)
                        value = torch.repeat_interleave(value, repeats=group_size, dim=1).permute(1, 0, 2).unsqueeze(0)
                        # copy partial_output to output
                        # print(f"============== stride={self.sparse_prefill_attn_config.stride} ==========")
                        partial_output = Xattention_prefill(
                            query_states=current_q,
                            key_states=key,
                            value_states=value,
                            stride=self.sparse_prefill_attn_config.stride,
                            threshold=self.sparse_prefill_attn_config.threshold,
                            block_size=self.sparse_prefill_attn_config.block_size,
                            chunk_size=self.sparse_prefill_attn_config.chunk_size,
                        ).squeeze(0)
                        # print(f"================= partial_output shape: {partial_output.shape} ===========")
                        output[qs : qe] = partial_output.permute(1, 0, 2)

                    if self.enable_attn_out_dump:
                        print(f"================== ENABLE XATTN CHUNKED OUT DUMP {self.tp_rank} {self.layer_idx} {prefill_output.shape} {qlen} {seqlen} ================")
                        assert len(prefill_meta.seq_lens) == 1
                        import h5py
                        file_name = f"/data/shanhaikang.shk/vllm/xattn_chunked_out_dump/tensor_{self.tp_rank}.hdf5"
                        with h5py.File(file_name, 'a') as f:
                            f.create_dataset(f'{self.layer_idx}_{seqlen - qlen}', data=prefill_output.clone().detach().float().cpu().numpy())
                elif self.sparse_prefill_attn_type == SparsePrefillType.FLEX_PREFILL:
                    # Flex Prefill
                    print("XXXXXXXXXXXXXX ============== ENABLE FLEX_PREFILL ============")
                    max_seq_len = max(prefill_meta.seq_lens)
                    block_size = self.sparse_prefill_attn_config.block_size
                    
                    if max_seq_len <= max(2 * block_size, math.ceil(self.sparse_prefill_attn_config.min_budget / block_size) * block_size):
                        flash_attn_varlen_func(  # noqa
                            q=query,
                            k=key_cache,
                            v=value_cache,
                            cu_seqlens_q=prefill_meta.query_start_loc,
                            max_seqlen_q=prefill_meta.max_query_len,
                            seqused_k=prefill_meta.seq_lens_tensor,
                            max_seqlen_k=max_seq_len,
                            softmax_scale=softmax_scale,
                            causal=True,
                            window_size=window_size,
                            alibi_slopes=alibi_slopes,
                            block_table=prefill_meta.block_tables,
                            softcap=logits_soft_cap,
                            out=prefill_output,
                            fa_version=self.vllm_flash_attn_version,
                        )
                    else:
                        assert logits_soft_cap == 0.0
                        cu_seqlens_q = prefill_meta.query_start_loc
                        cu_seqlens_q_cpu = cu_seqlens_q.cpu().tolist()
                        assert (prefill_meta.seq_lens_tensor is not None and 
                                prefill_meta.seq_lens_tensor.shape[0] == len(cu_seqlens_q_cpu) - 1 and
                                prefill_meta.block_tables is not None and
                                prefill_meta.block_tables.shape[0] == len(cu_seqlens_q_cpu) - 1 and
                                prefill_meta.seq_lens is not None and
                                len(prefill_meta.seq_lens) == len(cu_seqlens_q_cpu) - 1)

                        for query_idx in range(0, len(cu_seqlens_q_cpu) - 1):
                            qs = cu_seqlens_q_cpu[query_idx]
                            qe = cu_seqlens_q_cpu[query_idx : query_idx + 2][-1]
                            current_q = query[qs : qe] # seq_len, num_head, head_dim
                            current_block_table = prefill_meta.block_tables[query_idx]
                            current_seq_len = prefill_meta.seq_lens[query_idx]
                            assert current_q.size(-2) % key_cache.size(-2) == 0
                            # retrieve key & value from kv_cache
                            key = key_cache[current_block_table].view(-1, *key_cache.shape[-2:])[: current_seq_len]
                            value = value_cache[current_block_table].view(-1, *value_cache.shape[-2:])[: current_seq_len]

                            current_q = current_q.unsqueeze(0)
                            key = key.unsqueeze(0)
                            value = value.unsqueeze(0)
                            partial_output = flex_prefill_attention(
                                q=current_q,
                                k=key,
                                v=value,
                                gamma=self.sparse_prefill_attn_config.gamma,
                                tau=self.sparse_prefill_attn_config.tau,
                                min_budget=self.sparse_prefill_attn_config.min_budget,
                                max_budget=self.sparse_prefill_attn_config.max_budget,
                                softmax_scale=softmax_scale,
                                block_size=self.sparse_prefill_attn_config.block_size,
                            ).squeeze(0)
                            output[qs : qe] = partial_output
                elif self.sparse_prefill_attn_type == SparsePrefillType.SPARGE_ATTN:
                    print("XXXXXXXXXXXXX ============== ENABLE SPARGE_ATTN ============")
                    # Sparge Attention
                    assert logits_soft_cap == 0.0
                    cu_seqlens_q = prefill_meta.query_start_loc
                    cu_seqlens_q_cpu = cu_seqlens_q.cpu().tolist()
                    assert (prefill_meta.seq_lens_tensor is not None and 
                            prefill_meta.seq_lens_tensor.shape[0] == len(cu_seqlens_q_cpu) - 1 and
                            prefill_meta.block_tables is not None and
                            prefill_meta.block_tables.shape[0] == len(cu_seqlens_q_cpu) - 1 and
                            prefill_meta.seq_lens is not None and
                            len(prefill_meta.seq_lens) == len(cu_seqlens_q_cpu) - 1)
                    
                    max_seq_len = max(prefill_meta.seq_lens)
                    has_short_query = False
                    for query_idx in range(0, len(cu_seqlens_q_cpu) - 1):
                        qs = cu_seqlens_q_cpu[query_idx]
                        qe = cu_seqlens_q_cpu[query_idx : query_idx + 2][-1]
                        if qe - qs < 128:
                            has_short_query = True
                            break
                    
                    if has_short_query:
                        flash_attn_varlen_func(  # noqa
                            q=query,
                            k=key_cache,
                            v=value_cache,
                            cu_seqlens_q=prefill_meta.query_start_loc,
                            max_seqlen_q=prefill_meta.max_query_len,
                            seqused_k=prefill_meta.seq_lens_tensor,
                            max_seqlen_k=max_seq_len,
                            softmax_scale=softmax_scale,
                            causal=True,
                            window_size=window_size,
                            alibi_slopes=alibi_slopes,
                            block_table=prefill_meta.block_tables,
                            softcap=logits_soft_cap,
                            out=prefill_output,
                            fa_version=self.vllm_flash_attn_version,
                        )
                    else:
                        for query_idx in range(0, len(cu_seqlens_q_cpu) - 1):
                            qs = cu_seqlens_q_cpu[query_idx]
                            qe = cu_seqlens_q_cpu[query_idx : query_idx + 2][-1]

                            current_q = query[qs : qe] # seq_len, num_head, head_dim
                            current_block_table = prefill_meta.block_tables[query_idx]
                            current_seq_len = prefill_meta.seq_lens[query_idx]
                            assert current_q.size(-2) % key_cache.size(-2) == 0
                            group_size = current_q.size(-2) // key_cache.size(-2)
                            key = key_cache[current_block_table].view(-1, *key_cache.shape[-2:])[: current_seq_len]
                            value = value_cache[current_block_table].view(-1, *value_cache.shape[-2:])[: current_seq_len]

                            key = torch.repeat_interleave(key, repeats=group_size, dim=1).unsqueeze(0)
                            value = torch.repeat_interleave(value, repeats=group_size, dim=1).unsqueeze(0)
                            current_q = current_q.unsqueeze(0)

                            partial_output = spas_sage2_attn_meansim_cuda(
                                q=current_q,
                                k=key,
                                v=value,
                                is_causal=True,
                                simthreshd1=self.sparse_prefill_attn_config.simthreshd1,
                                cdfthreshd=self.sparse_prefill_attn_config.cdfthreshd,
                                pvthreshd=self.sparse_prefill_attn_config.pvthreshd,
                                tensor_layout="NHD"
                            ).squeeze(0)
                            output[qs : qe] = partial_output
                else:
                    raise ValueError(f"Unsupported sparse prefill type: {self.sparse_prefill_attn_type}")


                
        if decode_meta := attn_metadata.decode_metadata:
            # Decoding run.
            # Use flash_attn_varlen_func kernel for speculative decoding
            # because different queries might have different lengths.

            assert decode_meta.max_decode_query_len is not None

            # start_event = torch.cuda.Event(enable_timing=True)
            # end_event = torch.cuda.Event(enable_timing=True)
            n_recomputes = decode_meta.num_sparse_index_recomputes
            num_compressed_page_tensor = decode_meta.num_compressed_pages_tensor
            page_compress_cache_ids_tensor = decode_meta.page_compress_cache_ids_tensor
            actual_seqlen_tensor = decode_meta.actual_seqlen_tensor
            actual_max_num_blocks_per_seq = decode_meta.actual_max_num_blocks_per_seq
            if n_recomputes > 0:
                assert (
                    attn_type == AttentionType.DECODER and 
                    key_meta_cache is not None and block_index_gpu_cache is not None and 
                    num_compressed_page_tensor is not None and page_compress_cache_ids_tensor is not None
                )
                # start_event.record()
                # print(f"==================== RECOMPUTE PAGE COMPRESS: actual_max_num_blocks_per_seq:{actual_max_num_blocks_per_seq} block_index_gpu_cache:{block_index_gpu_cache.data_ptr()} page_compress_cache_id:{page_compress_cache_ids_tensor[0]} num_compressed_page_tensor:{num_compressed_page_tensor[0]}  ==============")
                out = torch.full((n_recomputes, key_cache.shape[-2], decode_meta.block_tables.shape[-1]), 
                                    float('-inf'), 
                                    dtype=decode_query.dtype, device=decode_query.device)
                torch.ops._C.lserve_page_selector(
                    decode_query[:n_recomputes],
                    key_meta_cache,
                    decode_meta.block_tables[:n_recomputes],
                    num_compressed_page_tensor[:n_recomputes],
                    out,
                )
                block_index_gpu_cache[page_compress_cache_ids_tensor[:n_recomputes], :, :] = torch.gather(
                    decode_meta.block_tables[:n_recomputes].unsqueeze(1).expand(-1, out.shape[1], -1),
                    dim=-1,
                    index=torch.topk(out, k=decode_meta.page_compress_topk, sorted=False).indices
                )
                # block_index_gpu_cache[page_compress_cache_ids_tensor[:n_recomputes], :, :] = tmp

                # decode_seq_len = decode_meta.max_decode_seq_len
                # if decode_seq_len >= 4096 and decode_seq_len <= 8192:
                #     import h5py
                #     file_name = f"/data/shanhaikang.shk/vllm/topk_dump/tensor_{self.tp_rank}_{decode_seq_len}.hdf5"
                #     with h5py.File(file_name, 'a') as f:
                #         f.create_dataset(f'{self.layer_idx}', data=tmp.float().cpu().numpy())
                # end_event.record()
                # torch.cuda.synchronize()
                # elapsed_time = start_event.elapsed_time(end_event)
                # print(f"==================== Page Selector time cost: {elapsed_time:.4f}ms =================")
            
            # use only for actual varlen decoding
            if decode_meta.max_decode_query_len > 1:
                assert attn_type == AttentionType.DECODER, (
                    "Only decoder-only models support max_decode_query_len > 1"
                )
                assert decode_meta.query_start_loc is not None
                descale_shape = (decode_meta.query_start_loc.shape[0] - 1,
                                 key.shape[1])
                if block_index_gpu_cache is not None and block_index_gpu_cache.size(0) > 0:
                    flash_attn_varlen_func(
                        q=decode_query,
                        k=key_cache,
                        v=value_cache,
                        cu_seqlens_q=decode_meta.query_start_loc,
                        max_seqlen_q=decode_meta.max_decode_query_len,
                        seqused_k=actual_seqlen_tensor,   # 使用压缩后的 actual_seqlen_tensor
                        max_seqlen_k=decode_meta.actual_max_decode_seq_len,    # 使用压缩后的 actual_max_decode_seq_len
                        softmax_scale=softmax_scale,
                        causal=True,
                        window_size=window_size,
                        alibi_slopes=alibi_slopes,
                        softcap=logits_soft_cap,
                        block_table=decode_meta.block_tables,
                        page_compress_cache=block_index_gpu_cache,
                        page_compress_cache_ids=page_compress_cache_ids_tensor,
                        num_compressed_pages=num_compressed_page_tensor,
                        out=decode_output,
                        fa_version=self.vllm_flash_attn_version,
                        q_descale=layer._q_scale.expand(descale_shape),
                        k_descale=layer._k_scale.expand(descale_shape),
                        v_descale=layer._v_scale.expand(descale_shape),
                    )
                else:
                    flash_attn_varlen_func(
                        q=decode_query,
                        k=key_cache,
                        v=value_cache,
                        cu_seqlens_q=decode_meta.query_start_loc,
                        max_seqlen_q=decode_meta.max_decode_query_len,
                        seqused_k=decode_meta.seq_lens_tensor if not self.enable_pooling else decode_meta.seq_len_after_pooling_for_decode_tensor,
                        max_seqlen_k=decode_meta.max_decode_seq_len,
                        softmax_scale=softmax_scale,
                        causal=True,
                        window_size=window_size,
                        alibi_slopes=alibi_slopes,
                        softcap=logits_soft_cap,
                        block_table=decode_meta.block_tables,
                        out=decode_output,
                        fa_version=self.vllm_flash_attn_version,
                        q_descale=layer._q_scale.expand(descale_shape),
                        k_descale=layer._k_scale.expand(descale_shape),
                        v_descale=layer._v_scale.expand(descale_shape),
                    )
            else:
                # Use flash_attn_with_kvcache for normal decoding.
                (
                    seq_lens_arg,
                    _,
                    block_tables_arg,
                ) = get_seq_len_block_table_args(decode_meta, False, attn_type)
                descale_shape = (seq_lens_arg.shape[0], key_cache.shape[-2])

                if attn_type == AttentionType.DECODER and self.enable_pooling:
                    seq_lens_arg = decode_meta.seq_len_after_pooling_for_decode_tensor

                # TODO
                if self.dump_decode_attn:
                    assert seq_lens_arg.shape[0] == 1
                    decode_seq_len = seq_lens_arg[0].item()
                    current_block_table = block_tables_arg[0]
                    if self.dump_decode_which_step == decode_seq_len - decode_meta.prompt_lens[0]:
                        print(f"================ DUMP DECODE layer{self.layer_idx} tp{self.tp_rank} {decode_seq_len} {decode_meta.prompt_lens[0]} ================")
                        key = key_cache[current_block_table].view(-1, *key_cache.shape[-2:])[: decode_seq_len]
                        value = value_cache[current_block_table].view(-1, *value_cache.shape[-2:])[: decode_seq_len]
                        qk = (decode_query.transpose(0, 1) * softmax_scale) @ key.permute(1, 2, 0)
                        attn_map = F.softmax(qk, dim=-1)
                        attn_map = attn_map.squeeze(1)

                        file_name = f"/data/shanhaikang.shk/vllm/decode_attn_dump/{self.dump_decode_which_step}_tensor_{self.tp_rank}.hdf5"
                        import h5py
                        with h5py.File(file_name, 'a') as f:
                            f.create_dataset(f'{self.layer_idx}', data=attn_map.float().cpu().numpy())

                if self.fa_sparse_decoding_recover_rate is None:
                    # if (key_meta_cache is not None and key_meta_cache.size(0) > 0
                    #     and topk_blocks_per_head is not None):
                    #     group_size = decode_query.size(1) // key_meta_cache.size(1)
                    #     for head_i in range(decode_query.size(1)):
                    #         kv_head_i = head_i // group_size
                    #         flash_attn_with_kvcache(
                    #             q=decode_query[:, head_i:head_i+1, :].unsqueeze(1),
                    #             k_cache=key_cache,
                    #             v_cache=value_cache,
                    #             block_table=topk_blocks_per_head[kv_head_i:kv_head_i+1, :],
                    #             cache_seqlens=seq_lens_arg,
                    #             softmax_scale=softmax_scale,
                    #             causal=True,
                    #             window_size=window_size,
                    #             alibi_slopes=alibi_slopes,
                    #             softcap=logits_soft_cap,
                    #             out=decode_output[:, head_i:head_i+1, :].unsqueeze(1),
                    #             fa_version=self.vllm_flash_attn_version,
                    #             q_descale=layer._q_scale.expand(descale_shape),
                    #             k_descale=layer._k_scale.expand(descale_shape),
                    #             v_descale=layer._v_scale.expand(descale_shape),
                    #         )
                    # else:
                    # if dbg_num_compressed_page == 258:
                    #     print("dbg")
                    if decode_meta.enable_blk_attn:
                        assert query.dtype == torch.float16
                        flash_attn_varlen_func(  # noqa
                            q=decode_query,
                            k=key_cache,
                            v=value_cache,
                            cu_seqlens_q=decode_meta.blk_attn_decode_cu_seqlens_q,
                            max_seqlen_q=decode_meta.blk_attn_max_decode_q_len,
                            seqused_k=decode_meta.blk_attn_decode_seqused_k,
                            max_seqlen_k=decode_meta.blk_attn_max_decode_kv_len,
                            softmax_scale=softmax_scale,
                            causal=True,
                            window_size=window_size,
                            alibi_slopes=alibi_slopes,
                            block_table=block_tables_arg,

                            actual_chunked_seqlen_k=decode_meta.blk_attn_decode_actual_chunked_seqlen_k,
                            chunk_rotray_offset_positions=decode_meta.blk_attn_decode_chunk_rotary_offset_positions,
                            cu_num_chunks_k=decode_meta.blk_attn_decode_cu_num_chunks_k,
                            cos_sin_cache=self._cos_sin_cache,

                            softcap=logits_soft_cap,
                            out=decode_output,
                            fa_version=self.vllm_flash_attn_version,
                            q_descale=layer._q_scale.expand(descale_shape),
                            k_descale=layer._k_scale.expand(descale_shape),
                            v_descale=layer._v_scale.expand(descale_shape),
                        )
                        # flash_attn_with_kvcache(
                        #     q=decode_query.unsqueeze(1),
                        #     k_cache=key_cache,
                        #     v_cache=value_cache,
                        #     block_table=block_tables_arg,
                        #     cache_seqlens=seq_lens_arg,
                        #     softmax_scale=softmax_scale,
                        #     causal=True,
                        #     window_size=window_size,
                        #     alibi_slopes=alibi_slopes,
                        #     softcap=logits_soft_cap,
                        #     out=decode_output.unsqueeze(1),
                        #     fa_version=self.vllm_flash_attn_version,
                        #     q_descale=layer._q_scale.expand(descale_shape),
                        #     k_descale=layer._k_scale.expand(descale_shape),
                        #     v_descale=layer._v_scale.expand(descale_shape),
                        # )
                    else:
                        if block_index_gpu_cache is not None and block_index_gpu_cache.size(0) > 0:
                            flash_attn_with_kvcache(
                                q=decode_query.unsqueeze(1),
                                k_cache=key_cache,
                                v_cache=value_cache,
                                block_table=block_tables_arg,
                                page_compress_cache=block_index_gpu_cache,
                                page_compress_cache_ids=page_compress_cache_ids_tensor,
                                num_compressed_pages=num_compressed_page_tensor,
                                cache_seqlens=actual_seqlen_tensor,
                                softmax_scale=softmax_scale,
                                causal=True,
                                window_size=window_size,
                                alibi_slopes=alibi_slopes,
                                softcap=logits_soft_cap,
                                out=decode_output.unsqueeze(1),
                                fa_version=self.vllm_flash_attn_version,
                                q_descale=layer._q_scale.expand(descale_shape),
                                k_descale=layer._k_scale.expand(descale_shape),
                                v_descale=layer._v_scale.expand(descale_shape),
                                actual_max_num_blocks_per_seq=actual_max_num_blocks_per_seq,
                            )
                        else:
                            flash_attn_with_kvcache(
                                q=decode_query.unsqueeze(1),
                                k_cache=key_cache,
                                v_cache=value_cache,
                                block_table=block_tables_arg,
                                cache_seqlens=seq_lens_arg,
                                softmax_scale=softmax_scale,
                                causal=True,
                                window_size=window_size,
                                alibi_slopes=alibi_slopes,
                                softcap=logits_soft_cap,
                                out=decode_output.unsqueeze(1),
                                fa_version=self.vllm_flash_attn_version,
                                q_descale=layer._q_scale.expand(descale_shape),
                                k_descale=layer._k_scale.expand(descale_shape),
                                v_descale=layer._v_scale.expand(descale_shape),
                            )
                else:
                    # Just to check whether sparse pattern exists.
                    seq_len_cpu = seq_lens_arg.cpu().tolist()
                    assert len(seq_len_cpu) == 1
                    assert decode_query.shape[0] == 1
                    current_seq_len = seq_len_cpu[0]
                    current_block_table = block_tables_arg[0]
                    key = key_cache[current_block_table].view(-1, *key_cache.shape[-2:])[: current_seq_len]
                    value = value_cache[current_block_table].view(-1, *value_cache.shape[-2:])[: current_seq_len]

                    qk = (decode_query.transpose(0, 1) * softmax_scale) @ key.permute(1, 2, 0)
                    attn_map = F.softmax(qk, dim=-1)
                    # h,n
                    attn_sort_values, attn_sort_indices = attn_map.squeeze(1).sort(dim=-1, descending=True)

                    num_heads = decode_query.shape[1]
                    num_key_heads = key.shape[1]
                    group_size = num_heads // num_key_heads
                    
                    cum_attn_sort_values = attn_sort_values.cumsum(dim=-1)
                    targets = torch.ones((num_heads,), device=qk.device) * cum_attn_sort_values[..., -1] * self.fa_sparse_decoding_recover_rate
                    topk_per_head = torch.searchsorted(cum_attn_sort_values, targets.view(num_heads, 1), side='left')

                    for head_id in range(num_heads):
                        kv_head_id = head_id // group_size
                        hd_query = decode_query[:, head_id:head_id+1, :]

                        topk = min(topk_per_head[head_id, 0].item() + 1, current_seq_len)
                        hd_key = key[attn_sort_indices[head_id, :topk].sort().values, kv_head_id:kv_head_id+1, :]
                        hd_value = value[attn_sort_indices[head_id, :topk].sort().values, kv_head_id:kv_head_id+1, :]
                        
                        flash_attn_varlen_func(
                            q=hd_query,
                            k=hd_key,
                            v=hd_value,
                            softmax_scale=softmax_scale,
                            cu_seqlens_q=torch.tensor([0, 1],
                                                    dtype=torch.int32,
                                                    device=query.device),
                            max_seqlen_q=1,
                            cu_seqlens_k=torch.tensor([0, hd_key.shape[0]],
                                                    dtype=torch.int32,
                                                    device=query.device),
                            max_seqlen_k=hd_key.shape[0],
                            causal=True,
                            window_size=window_size,
                            alibi_slopes=alibi_slopes,
                            softcap=logits_soft_cap,
                            out=decode_output[:, head_id:head_id+1, :],
                            fa_version=self.vllm_flash_attn_version,
                            q_descale=layer._q_scale.expand(descale_shape),
                            k_descale=layer._k_scale.expand(descale_shape),
                            v_descale=layer._v_scale.expand(descale_shape),
                        )

        return output


def _get_query_key_seq_metadata(
    attn_metadata,
    is_prompt: bool,
    attn_type: str,
) -> tuple:
    """
    Returns sequence metadata for key and query based on the specified 
    attention type and whether input is a prompt.

    This function computes the starting locations and maximum sequence lengths 
    for key and query sequences for different attention types.

    Args:
        attn_metadata: The attention metadata object
        is_prompt (bool): A flag indicating if the input is a prompt
        attn_type (AttentionType): The type of attention being used.

    Returns:
        tuple: A tuple containing four integers:
            - Starting location for the query sequence.
            - Maximum sequence length for the query sequence.
            - Starting location for the key sequence.
            - Maximum sequence length for the key sequence.

    Raises:
        AttributeError: If an invalid attention type is provided.
    """
    if attn_type == AttentionType.DECODER:
        # Decoder self-attention
        # Choose max_seq_len based on whether we are in prompt_run
        if is_prompt:
            max_seq_len = attn_metadata.max_prefill_seq_len
        else:
            max_seq_len = attn_metadata.max_decode_seq_len
        return (attn_metadata.seq_start_loc, max_seq_len,
                attn_metadata.seq_start_loc, max_seq_len)

    elif attn_type == AttentionType.ENCODER_DECODER:
        # This is cross attention between the where the key
        # is the precomputed encoder attention and query
        # is the input sequence.
        # Choose query max length based on whether it is prompt
        # or not.
        if is_prompt:
            max_seq_len = attn_metadata.max_prefill_seq_len
        else:
            max_seq_len = attn_metadata.max_decode_seq_len
        return (attn_metadata.seq_start_loc, max_seq_len,
                attn_metadata.encoder_seq_start_loc,
                attn_metadata.max_encoder_seq_len)
    elif attn_type == AttentionType.ENCODER:
        # For encoder attention both the query and the key are same i.e the
        # encoder sequence.
        return (attn_metadata.encoder_seq_start_loc,
                attn_metadata.max_encoder_seq_len,
                attn_metadata.encoder_seq_start_loc,
                attn_metadata.max_encoder_seq_len)
    elif attn_type == AttentionType.ENCODER_ONLY:
        assert is_prompt, "Should not have decode for encoder only model."
        return (attn_metadata.seq_start_loc, attn_metadata.max_prefill_seq_len,
                attn_metadata.seq_start_loc, attn_metadata.max_prefill_seq_len)
    else:
        raise AttributeError(f"Invalid attention type {str(attn_type)}")


def _get_causal_option(attn_type: str) -> bool:
    """
    Determine whether the given attention type is suitable for causal 
    attention mechanisms.

    Args:
        attn_type (AttentionType): The type of attention being evaluated

    Returns:
        bool: Returns `True` if the attention type is suitable for causal 
        attention (i.e., not encoder, encoder-only, or encoder-decoder), 
        otherwise returns `False`.
    """
    return not (attn_type == AttentionType.ENCODER
                or attn_type == AttentionType.ENCODER_ONLY
                or attn_type == AttentionType.ENCODER_DECODER)
