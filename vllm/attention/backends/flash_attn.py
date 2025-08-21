# SPDX-License-Identifier: Apache-2.0
"""Attention layer with FlashAttention."""
from collections import defaultdict
from dataclasses import dataclass
from itertools import accumulate
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Tuple, Type

import torch
import torch.nn.functional as F

from vllm import _custom_ops as ops
# yapf conflicts with isort for this block
# yapf: disable
from vllm.attention.backends.abstract import (AttentionBackend, AttentionImpl,
                                              AttentionLayer,
                                              AttentionMetadata,
                                              AttentionMetadataBuilder,
                                              AttentionType,
                                              is_quantized_kv_cache)
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
# from .x_attn import Xattention_prefill
# from .minference import Minference_prefill
# from .flex_prefill_attention import flex_prefill_attention
# from spas_sage_attn import spas_sage2_attn_meansim_cuda

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
    # page selector 的参数
    # num_full_blocks_tensor 由 num_compressed_pages_tensor 的前 num_sparse_index_recomputes 个提供
    # 用于初始化 out tensor
    page_selector_max_block_size: Optional[int] = None
    page_compress_topk: Optional[int] = None


    update_meta_block_id_tensor: Optional[torch.Tensor] = None

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
            cross_block_tables=self.cross_block_tables)
        return self._cached_prefill_metadata
    
    # @property
    # def sparse_index_decode_metadata(self) -> Optional["FlashAttentionMetadata"]:
    #     if self.num_sparse_index_tokens == 0:
    #         return None
        
    #     if self._cached_sparse_index_decode_metadata is not None:
    #         return self._cached_sparse_index_decode_metadata
        
    #     assert ((self.seq_lens_tensor is not None)
    #             or (self.encoder_seq_lens_tensor is not None))
        
    #     token_ed = self.num_prefill_tokens + self.num_sparse_index_tokens
    #     query_ed = self.num_prefills + self.num_sparse_index_decodes
    #     slot_mapping = (None if self.slot_mapping is None else
    #                     self.slot_mapping[self.num_prefill_tokens:token_ed])
    #     seq_lens_tensor = (None if self.seq_lens_tensor is None else
    #                        self.seq_lens_tensor[self.num_prefills:query_ed])
    #     block_tables = (None if self.block_tables is None else
    #                     self.block_tables[self.num_prefills:query_ed])
    #     prompt_lens = (None if self.prompt_lens is None else
    #                    self.prompt_lens[self.num_prefills:query_ed])
        
    #     self._cached_sparse_index_decode_metadata = FlashAttentionMetadata(
    #         num_prefills=0,
    #         num_prefill_tokens=0,
    #         num_decode_tokens=self.num_sparse_index_tokens,
    #         slot_mapping=slot_mapping,
    #         multi_modal_placeholder_index_maps=None,
    #         enable_kv_scales_calculation=True,
    #         seq_lens=None,
    #         seq_lens_tensor=seq_lens_tensor,
    #         max_decode_query_len=self.max_sparse_index_decode_query_len,
    #         max_query_len=self.max_query_len,
    #         max_prefill_seq_len=0,
    #         max_decode_seq_len=self.max_sparse_index_decode_seq_len,
    #         # Batch may be composed of prefill|decodes, adjust query start
    #         # indices to refer to the start of decodes. E.g.
    #         # in tokens:[3 prefills|6 decodes], query_start_loc=[3,9] => [0,6].
    #         query_start_loc=(self.query_start_loc[self.num_prefills:query_ed+1] -
    #                          self.query_start_loc[self.num_prefills])
    #         if self.query_start_loc is not None else None,
    #         seq_start_loc=self.seq_start_loc[self.num_prefills:query_ed+1]
    #         if self.seq_start_loc is not None else None,
    #         prompt_lens=prompt_lens,
    #         context_lens_tensor=None,
    #         block_tables=block_tables,
    #         use_cuda_graph=self.use_cuda_graph,
    #         sparse_index_kv_compress_recover_rate=self.sparse_index_kv_compress_recover_rate,
    #         num_sparse_index_recomputes=self.num_sparse_index_recomputes,
    #         num_sparse_index_recompute_tokens=self.num_sparse_index_recompute_tokens,
    #         sparse_index_block=self.sparse_index_block,
    #         sparse_index_block_tensor=self.sparse_index_block_tensor)
    #     return self._cached_sparse_index_decode_metadata

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
            actual_seqlen_tensor=self.actual_seqlen_tensor,
            page_selector_max_block_size=self.page_selector_max_block_size,
            page_compress_topk=self.page_compress_topk)
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

    def __init__(self, input_builder: "ModelInputForGPUBuilder"):
        self.input_builder = input_builder
        self.runner = input_builder.runner
        self.sliding_window = input_builder.sliding_window
        self.block_size = input_builder.block_size
        self.page_compress_topk = input_builder.page_compress_topk

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
            else:
                # decode
                self.num_decode_tokens += query_len
                self.curr_seq_lens.append(curr_seq_len)

                # num_sparse_index_recomputes & num_sparse_index_recompute_tokens
                if is_sparse_index_recompute:
                    self.num_sparse_index_recomputes += 1
                    self.num_sparse_index_recompute_tokens += query_len
                
                # sparse_index_blocks
                self.sparse_index_blocks.append(sparse_index_block_id)

                # num_compressed_pages
                if num_computed_token_to_compress == -1:
                    self.num_compressed_pages.append(-1)
                else:
                    self.num_compressed_pages.append(num_computed_token_to_compress // self.block_size)
                
                # actual_curr_seq_lens
                if is_sparse_index or is_sparse_index_recompute:
                    self.actual_curr_seq_lens.append(
                        curr_seq_len - (
                            num_computed_token_to_compress // self.block_size - self.page_compress_topk
                        ) * self.block_size
                    )
                else:
                    self.actual_curr_seq_lens.append(curr_seq_len)

            # Compute block table.
            # TODO(sang): Combine chunked prefill and prefix caching by
            # only allowing multiple of block_size chunk size.
            # NOTE: This only works for oooooooxxx style attention.
            block_table = []
            if prefix_cache_hit:
                # NOTE(woosuk): For flash-attn, the block table should
                # include the entries for the incoming prefill tokens.
                block_table = block_tables[seq_id]
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

            # Compute slot mapping.
            is_profile_run = is_block_tables_empty(block_tables)
            start_idx = compute_slot_mapping_start_idx(is_prompt, query_len,
                                                       context_len,
                                                       self.sliding_window)
            compute_slot_mapping(is_profile_run, self.slot_mapping, seq_id,
                                 seq_len, context_len, start_idx,
                                 self.block_size, inter_data.block_tables)

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
              cuda_graph_pad_size: int, batch_size: int):
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

        num_decode_tokens = self.num_decode_tokens
        query_start_loc = list(accumulate(query_lens, initial=0))
        seq_start_loc = list(accumulate(seq_lens, initial=0))

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
        if len(self.actual_curr_seq_lens) > 0:
            actual_max_num_blocks_per_seq = (max(self.actual_curr_seq_lens) + self.block_size - 1) // self.block_size

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
            page_selector_max_block_size=page_selector_max_block_size,
            page_compress_topk=self.page_compress_topk,
            update_meta_block_id_tensor=update_meta_block_id_tensor,
        )

class SparsePrefillType(IntEnum):
    FULL_ATTN = 0
    X_ATTN = 1
    MINFERENCE = 2
    FLEX_PREFILL = 3
    SPARGE_ATTN = 4

@dataclass
class XAttentionConfig:
    stride: int = 8
    threshold: float = 0.95
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
        dual_chunk_attention_config: Optional[Dict[str, Any]] = None,
        enable_attn_out_dump: bool = False,
        enable_last_attn_map_dump: bool = False,
        dump_last_query_len: int = 64,
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

        self.dump_decode_attn = os.getenv("VLLM_FA_DUMP_DECODE_ATTN", None) is not None
        self.dump_decode_which_step = int(os.getenv("VLLM_FA_DUMP_DECODE_STEP", 0))

        self.fa_sparse_decoding_recover_rate = os.getenv("VLLM_FA_DECODE_RECOVER_RATE", None)
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
        column_index_gpu_cache: Optional[torch.Tensor] = None
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

        (num_prefill_query_tokens, num_prefill_kv_tokens,
        num_decode_query_tokens) = \
            get_num_prefill_decode_query_kv_tokens(attn_metadata, attn_type)
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

                descale_shape = (q_seq_start_loc.shape[0] - 1, key.shape[1])
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
                # prefix-enabled attention
                assert attn_type == AttentionType.DECODER, (
                    "Only decoder-only models support prefix caching")
                assert prefill_meta.seq_lens is not None
                assert prefill_meta.query_start_loc is not None
                max_seq_len = max(prefill_meta.seq_lens)
                descale_shape = (prefill_meta.query_start_loc.shape[0] - 1,
                                key.shape[1])

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

        # TODO[shk]: 将 vertical-slash 方式修改为 page 粒度计算
        # if sparse_index_decode_meta := attn_metadata.sparse_index_decode_metadata:
        #     assert attn_type == AttentionType.DECODER, (
        #         "Only decoder-only models support sparse index decode"
        #     )
        #     recover_rate = sparse_index_decode_meta.sparse_index_kv_compress_recover_rate
        #     sparse_index_block = sparse_index_decode_meta.sparse_index_block
        #     sparse_index_block_tensor = sparse_index_decode_meta.sparse_index_block_tensor
        #     assert (recover_rate is not None and sparse_index_block is not None and
        #             sparse_index_block_tensor is not None)
        #     assert (block_count_gpu_cache is not None and block_index_gpu_cache is not None and
        #                 column_count_gpu_cache is not None and column_index_gpu_cache is not None)
        #     # block_count_gpu_cache = block_count_gpu_cache.unsqueeze(-1)
        #     # block_index_gpu_cache = block_index_gpu_cache.unsqueeze(-2)
        #     # column_count_gpu_cache = column_count_gpu_cache.unsqueeze(-1)
        #     # column_index_gpu_cache = column_index_gpu_cache.unsqueeze(-2)
            
        #     if sparse_index_decode_meta.num_sparse_index_recomputes > 0:
        #         max_vertical_slash_topk = block_index_gpu_cache.size(-1)
        #         # For simple, we handle single batch.
        #         assert len(sparse_index_block) == 1
        #         query = sparse_index_decode_query[:self.sparse_index_block_size, ...]
        #         sparse_index_block_idx = sparse_index_block[0]
        #         sparse_index_decode_block_table = sparse_index_decode_meta.block_tables[0]
        #         seq_len = sparse_index_decode_meta.max_decode_seq_len
        #         n_heads = query.size(-2)

        #         group_size = query.size(-2) // key_cache.size(-2)
        #         key = key_cache[sparse_index_decode_block_table].view(-1, *key_cache.shape[-2:])[: seq_len]

        #         query = query.transpose(0, 1) # hnd
        #         # nhd -> hdn
        #         key = key.permute(1, 2, 0)
        #         assert key.size(0) == 1
        #         qk = (query * softmax_scale) @ key
        #         qk[:, :, -self.sparse_index_block_size:] = torch.where(
        #             self.last_q_mask,
        #             qk[:, :, -self.sparse_index_block_size:],
        #             -torch.inf,
        #         )
        #         qk = F.softmax(qk, dim=-1)

        #         # 
        #         vertical = qk.sum(-2)
        #         vertical_sorted = vertical.sort(dim=-1, descending=True).values
        #         cum_vertical_sorted = vertical_sorted.cumsum(dim=-1)
        #         vertical_targets = torch.ones((n_heads,), device=qk.device) * cum_vertical_sorted[..., -1] * recover_rate
        #         vindices = torch.searchsorted(cum_vertical_sorted, vertical_targets.view(n_heads, 1), side='left')
        #         vindices = vindices[..., 0] + 30
        #         # vertical count
        #         vindices = torch.clamp(vindices, max=min(max_vertical_slash_topk, seq_len))
        #         vertical[..., :30] = torch.inf
        #         # 
        #         slash = _sum_all_diagonal_matrix(qk)
        #         slash = slash[..., :-self.sparse_index_block_size + 1]
        #         slash_sorted = slash.sort(dim=-1, descending=True).values
        #         cum_slash_sorted = slash_sorted.cumsum(dim=-1)
        #         slash_targets = torch.ones((n_heads,), device=qk.device) * cum_slash_sorted[..., -1] * recover_rate
        #         sindices = torch.searchsorted(cum_slash_sorted, slash_targets.view(n_heads, 1), side='left')
        #         sindices = sindices[..., 0] + 100
        #         # slash_count
        #         sindices = torch.clamp(sindices, max=min(max_vertical_slash_topk, seq_len))
        #         slash[..., -100:] = torch.inf
                
        #         vertical_index_buff = torch.full(
        #             (n_heads, min(max_vertical_slash_topk, seq_len)),
        #             self.int32_max,
        #             dtype=torch.int32,
        #             device=qk.device,
        #         )
        #         slash_index_buff = torch.full(
        #             (n_heads, min(max_vertical_slash_topk, seq_len)),
        #             self.int32_max,
        #             dtype=torch.int32,
        #             device=qk.device,
        #         )
        #         for head_i in range(n_heads):
        #             vtopk = vindices[head_i].item()
        #             vertical_topk = torch.topk(vertical[head_i:head_i+1], vtopk, -1).indices
        #             vertical_index_buff[head_i, :vtopk] = vertical_topk

        #             stopk = sindices[head_i].item()
        #             slash_topk = torch.topk(slash[head_i:head_i+1], stopk, -1).indices
        #             slash_topk = (seq_len - 1) - slash_topk
        #             slash_index_buff[head_i, :stopk] = slash_topk
        #         vertical_index_buff = vertical_index_buff.sort(dim=-1, descending=False)[0]
        #         slash_index_buff = slash_index_buff.sort(dim=-1, descending=False)[0]

        #         # kv_seq_len = seq_len
        #         # context_size = self.sparse_index_block_size
        #         # q_seqlens = torch.tensor([context_size],
        #         #              dtype=torch.int32,
        #         #              device=query.device)
        #         # kv_seqlens = torch.tensor([kv_seq_len],
        #         #                         dtype=torch.int32,
        #         #                         device=query.device)
        #         vcount = vindices.to(torch.int32)
        #         scount = sindices.to(torch.int32)

        #         block_count_gpu_cache[sparse_index_block_idx, ...] = scount
        #         column_count_gpu_cache[sparse_index_block_idx, ...] = vcount
        #         block_index_gpu_cache[sparse_index_block_idx, :, :min(max_vertical_slash_topk, seq_len)] = slash_index_buff
        #         column_index_gpu_cache[sparse_index_block_idx, :, :min(max_vertical_slash_topk, seq_len)] = vertical_index_buff
                
        #         # block_count_gpu_cache[sparse_index_block_idx:sparse_index_block_idx+1, ...] = 
        #         # (
        #         #     block_count_gpu_cache[sparse_index_block_idx:sparse_index_block_idx+1, ...],
        #         #     block_index_gpu_cache[sparse_index_block_idx:sparse_index_block_idx+1, :, :, :seq_len],
        #         #     column_count_gpu_cache[sparse_index_block_idx:sparse_index_block_idx+1, ...],
        #         #     column_index_gpu_cache[sparse_index_block_idx:sparse_index_block_idx+1, :, :, :seq_len]
        #         # ) = ops.convert_vertical_slash_indexes_mergehead(
        #         #     q_seqlens, kv_seqlens, 
        #         #     vertical_index_buff.unsqueeze(0), slash_index_buff.unsqueeze(0), 
        #         #     vindices, sindices, context_size,
        #         #     self.sparse_index_block_size, self.sparse_index_block_size)

        #     assert sparse_index_decode_meta.max_decode_query_len is not None
        #     if sparse_index_decode_meta.max_decode_query_len > 1:
        #         assert attn_type == AttentionType.DECODER, (
        #             "Only decoder-only models support max_decode_query_len > 1"
        #         )
        #         assert sparse_index_decode_meta.query_start_loc is not None
        #         descale_shape = (sparse_index_decode_meta.query_start_loc.shape[0] - 1,
        #                          key.shape[1])
        #         flash_attn_varlen_func(
        #             q=sparse_index_decode_query,
        #             k=key_cache,
        #             v=value_cache,
        #             cu_seqlens_q=sparse_index_decode_meta.query_start_loc,
        #             max_seqlen_q=sparse_index_decode_meta.max_decode_query_len,
        #             seqused_k=sparse_index_decode_meta.seq_lens_tensor,
        #             max_seqlen_k=sparse_index_decode_meta.max_decode_seq_len,
        #             softmax_scale=softmax_scale,
        #             causal=True,
        #             window_size=window_size,
        #             alibi_slopes=alibi_slopes,
        #             softcap=logits_soft_cap,
        #             block_table=sparse_index_decode_meta.block_tables,
        #             out=sparse_index_decode_output,
        #             fa_version=self.vllm_flash_attn_version,
        #             q_descale=layer._q_scale.expand(descale_shape),
        #             k_descale=layer._k_scale.expand(descale_shape),
        #             v_descale=layer._v_scale.expand(descale_shape),
        #         )
        #     else:
        #         # if sparse_index_decode_meta.num_sparse_index_recomputes == 0:
        #         #     print("DEBUG")
        #         assert len(sparse_index_block) == 1
        #         sparse_index_block_idx = sparse_index_block[0]
        #         sparse_index_decode_block_table = sparse_index_decode_meta.block_tables[0]
        #         seq_len = sparse_index_decode_meta.max_decode_seq_len
        #         key = key_cache[sparse_index_decode_block_table].view(-1, *key_cache.shape[-2:])[: seq_len]
        #         value = value_cache[sparse_index_decode_block_table].view(-1, *value_cache.shape[-2:])[: seq_len]

        #         n_head = sparse_index_decode_query.size(-2)
        #         kv_head = key.size(-2)
        #         group_size = n_head // kv_head
        #         for head_i in range(n_head):
        #             khead_i = head_i // group_size
        #             block_count = block_count_gpu_cache[sparse_index_block_idx, head_i]
        #             column_count = column_count_gpu_cache[sparse_index_block, head_i]
        #             block_index = block_index_gpu_cache[sparse_index_block_idx, head_i, :block_count]
        #             column_index = column_index_gpu_cache[sparse_index_block_idx, head_i, :column_count]
        #             indices = torch.unique(torch.cat([((seq_len - 1) - block_index), column_index]))

        #             q = sparse_index_decode_query[:, head_i:head_i+1, :]
        #             k = key[indices, khead_i:khead_i+1, :]
        #             v = value[indices, khead_i:khead_i+1, :]
        #             flash_attn_varlen_func(
        #                 q=q,
        #                 k=k,
        #                 v=v,
        #                 softmax_scale=softmax_scale,
        #                 cu_seqlens_q=torch.tensor([0, 1],
        #                                         dtype=torch.int32,
        #                                         device=q.device),
        #                 max_seqlen_q=1,
        #                 cu_seqlens_k=torch.tensor([0, k.shape[0]],
        #                                         dtype=torch.int32,
        #                                         device=q.device),
        #                 max_seqlen_k=k.shape[0],
        #                 causal=True,
        #                 window_size=window_size,
        #                 alibi_slopes=alibi_slopes,
        #                 softcap=logits_soft_cap,
        #                 out=sparse_index_decode_output[:, head_i:head_i+1, :],
        #                 fa_version=self.vllm_flash_attn_version,
        #             )

        #         # # descale_shape = (sparse_index_decode_meta.seq_lens_tensor.shape[0], key_cache.shape[-2])
        #         # flash_attn_varlen_func(
        #         #     q=sparse_index_decode_query,
        #         #     k=key,
        #         #     v=value,
        #         #     softmax_scale=softmax_scale,
        #         #     cu_seqlens_q=torch.tensor([0, 1],
        #         #                             dtype=torch.int32,
        #         #                             device=query.device),
        #         #     max_seqlen_q=1,
        #         #     cu_seqlens_k=torch.tensor([0, key.shape[0]],
        #         #                             dtype=torch.int32,
        #         #                             device=query.device),
        #         #     max_seqlen_k=key.shape[0],
        #         #     causal=True,
        #         #     window_size=window_size,
        #         #     alibi_slopes=alibi_slopes,
        #         #     softcap=logits_soft_cap,
        #         #     out=sparse_index_decode_output,
        #         #     fa_version=self.vllm_flash_attn_version,
        #         #     # q_descale=layer._q_scale.expand(descale_shape),
        #         #     # k_descale=layer._k_scale.expand(descale_shape),
        #         #     # v_descale=layer._v_scale.expand(descale_shape),
        #         # )
        #         # block_count = block_count_gpu_cache[sparse_index_block_idx, ...].unsqueeze(-1)
        #         # block_offset = block_index_gpu_cache[sparse_index_block_idx, ...].unsqueeze(-2)
        #         # column_count = column_count_gpu_cache[sparse_index_block_idx, ...].unsqueeze(-1)
        #         # column_index = column_index_gpu_cache[sparse_index_block_idx, ...].unsqueeze(-2)
        #         # sparse_attn_func(
        #         #     q=sparse_index_decode_query.unsqueeze(0),
        #         #     k=key.unsqueeze(0),
        #         #     v=value.unsqueeze(0),
        #         #     block_count=block_count,
        #         #     block_offset=block_offset,
        #         #     column_count=column_count,
        #         #     column_index=column_index,
        #         #     causal=True,
        #         #     softmax_scale=softmax_scale,
        #         #     out=sparse_index_decode_output.unsqueeze(1)
        #         # )
        #         # seq_lens_arg = sparse_index_decode_meta.seq_lens_tensor
        #         # block_tables_arg = sparse_index_decode_meta.block_tables
        #         # descale_shape = (seq_lens_arg.shape[0], key_cache.shape[-2])
        #         # flash_attn_with_kvcache(
        #         #     q=sparse_index_decode_query.unsqueeze(1),
        #         #     k_cache=key_cache,
        #         #     v_cache=value_cache,
        #         #     block_table=block_tables_arg,
        #         #     cache_seqlens=seq_lens_arg,
        #         #     softmax_scale=softmax_scale,
        #         #     causal=True,
        #         #     window_size=window_size,
        #         #     alibi_slopes=alibi_slopes,
        #         #     softcap=logits_soft_cap,
        #         #     out=sparse_index_decode_output.unsqueeze(1),
        #         #     fa_version=self.vllm_flash_attn_version,
        #         #     q_descale=layer._q_scale.expand(descale_shape),
        #         #     k_descale=layer._k_scale.expand(descale_shape),
        #         #     v_descale=layer._v_scale.expand(descale_shape),
        #         # )

        if decode_meta := attn_metadata.decode_metadata:
            # Decoding run.
            # Use flash_attn_varlen_func kernel for speculative decoding
            # because different queries might have different lengths.

            assert decode_meta.max_decode_query_len is not None
            # use only for actual varlen decoding
            if decode_meta.max_decode_query_len > 1:
                assert attn_type == AttentionType.DECODER, (
                    "Only decoder-only models support max_decode_query_len > 1"
                )
                assert decode_meta.query_start_loc is not None
                descale_shape = (decode_meta.query_start_loc.shape[0] - 1,
                                 key.shape[1])
                flash_attn_varlen_func(
                    q=decode_query,
                    k=key_cache,
                    v=value_cache,
                    cu_seqlens_q=decode_meta.query_start_loc,
                    max_seqlen_q=decode_meta.max_decode_query_len,
                    seqused_k=decode_meta.seq_lens_tensor,
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

                # start_event = torch.cuda.Event(enable_timing=True)
                # end_event = torch.cuda.Event(enable_timing=True)
                n_recomputes = decode_meta.num_sparse_index_recomputes
                num_compressed_page_tensor = decode_meta.num_compressed_pages_tensor
                page_compress_cache_ids_tensor = decode_meta.page_compress_cache_ids_tensor
                actual_seqlen_tensor = decode_meta.actual_seqlen_tensor
                actual_max_num_blocks_per_seq = decode_meta.actual_max_num_blocks_per_seq
                if n_recomputes > 0:
                    assert (
                        key_meta_cache is not None and block_index_gpu_cache is not None and 
                        num_compressed_page_tensor is not None and page_compress_cache_ids_tensor is not None
                    )
                    # start_event.record()
                    # print(f"==================== RECOMPUTE PAGE COMPRESS: actual_max_num_blocks_per_seq:{actual_max_num_blocks_per_seq} block_index_gpu_cache:{block_index_gpu_cache.data_ptr()} page_compress_cache_id:{page_compress_cache_ids_tensor[0]} num_compressed_page_tensor:{num_compressed_page_tensor[0]}  ==============")
                    out = torch.full((n_recomputes, key_cache.shape[-2], block_tables_arg.shape[-1]), 
                                     float('-inf'), 
                                     dtype=decode_query.dtype, device=decode_query.device)
                    torch.ops._C.lserve_page_selector(
                        decode_query[:n_recomputes],
                        key_meta_cache,
                        block_tables_arg[:n_recomputes],
                        num_compressed_page_tensor[:n_recomputes],
                        out,
                    )
                    block_index_gpu_cache[page_compress_cache_ids_tensor[:n_recomputes], :, :] = torch.gather(
                        block_tables_arg[:n_recomputes].unsqueeze(1).expand(-1, out.shape[1], -1),
                        dim=-1,
                        index=torch.topk(out, k=decode_meta.page_compress_topk, sorted=False).indices
                    )
                    # end_event.record()
                    # torch.cuda.synchronize()
                    # elapsed_time = start_event.elapsed_time(end_event)
                    # print(f"==================== Page Selector time cost: {elapsed_time:.4f}ms =================")

                # topk = None
                # if key_meta_cache:
                #     torch.ops._C.lserve_page_selector(
                #         decode_query,
                #         key_meta_cache,
                #         block_tables_arg,
                #         # 在 build metadata 时创建两者
                #         num_full_blocks,
                #         out,
                #     )
                #     topk = torch.topk(out, k=256, largest=True).index

                # topk_blocks_per_head = None
                # SEQ_LEN_THRESHOLD = 8192
                # TOKEN_BUDGET = 8192
                # BLOCK_SIZE = 16
                # if key_meta_cache is not None and key_meta_cache.size(0) > 0:
                #     assert seq_lens_arg.size(0) == 1
                #     seq_len = decode_meta.seq_lens[0]
                #     block_table = block_tables_arg[0, :]
                #     if seq_len >= SEQ_LEN_THRESHOLD:
                #         num_full_filled_block = seq_len // BLOCK_SIZE
                #         non_full_block_tokens = seq_len % BLOCK_SIZE
                #         more_tokens = 1 if non_full_block_tokens > 0 else 0
                #         cur_meta = key_meta_cache[block_table[:num_full_filled_block], ...]
                #         pos_mask = decode_query >= 0
                #         neg_mask = decode_query < 0
                #         metas = torch.zeros(
                #             (num_full_filled_block, key_meta_cache.size(1), key_meta_cache.size(-1)),
                #             dtype=key_meta_cache.dtype,
                #             device=key_meta_cache.device,
                #         )
                #         group_size = decode_query.size(1) // key_meta_cache.size(1)

                #         topk = (TOKEN_BUDGET // BLOCK_SIZE) + more_tokens
                #         topk_blocks_per_head = torch.zeros(
                #             (decode_query.size(1), topk),
                #             dtype=torch.int32,
                #             device=key_meta_cache.device,
                #         )

                #         for head_i in range(decode_query.size(1)):
                #             head_pos_mask = pos_mask[0, head_i, :]
                #             head_neg_mask = neg_mask[0, head_i, :]
                #             kv_head_i = head_i // group_size
                #             metas[:, kv_head_i, head_pos_mask] = cur_meta[:, kv_head_i, 0, head_pos_mask]
                #             metas[:, kv_head_i, head_neg_mask] = cur_meta[:, kv_head_i, 1, head_neg_mask]
                #             scores = decode_query[:, head_i, :] @ metas[:, kv_head_i, :].transpose(0, 1)
                #             if more_tokens == 0:
                #                 topk_blocks_per_head[head_i, :] = block_table[scores.squeeze(0).topk(k=topk, dim=-1).indices]
                #             else:
                #                 topk_blocks_per_head[head_i, topk - 1] = block_table[num_full_filled_block]
                #                 topk_blocks_per_head[head_i, :topk - 1] = block_table[scores.squeeze(0).topk(k=topk-1, dim=-1).indices]
                #         seq_lens_arg[0] = TOKEN_BUDGET + non_full_block_tokens

                # TODO
                if self.dump_decode_attn:
                    assert seq_lens_arg.shape[0] == 1
                    decode_seq_len = seq_lens_arg[0].item()
                    current_block_table = block_tables_arg[0]
                    # if self.dump_decode_which_step == decode_seq_len - decode_meta.prompt_lens[0]:
                    #     print(f"================ DUMP DECODE layer{self.layer_idx} tp{self.tp_rank} {decode_seq_len} {decode_meta.prompt_lens[0]} ================")
                    #     key = key_cache[current_block_table].view(-1, *key_cache.shape[-2:])[: decode_seq_len]
                    #     value = value_cache[current_block_table].view(-1, *value_cache.shape[-2:])[: decode_seq_len]
                    #     qk = (decode_query.transpose(0, 1) * softmax_scale) @ key.permute(1, 2, 0)
                    #     attn_map = F.softmax(qk, dim=-1)
                    #     attn_map = attn_map.squeeze(1)

                    #     file_name = f"/data/shanhaikang.shk/vllm/decode_attn_dump/{self.dump_decode_which_step}_tensor_{self.tp_rank}.hdf5"
                    #     import h5py
                    #     with h5py.File(file_name, 'a') as f:
                    #         f.create_dataset(f'{self.layer_idx}', data=attn_map.float().cpu().numpy())

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
