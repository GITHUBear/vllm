# SPDX-License-Identifier: Apache-2.0
from dataclasses import dataclass

import torch


@dataclass
class CommonAttentionMetadata:
    """
    Attention metadata attributes that can be shared by layers in different KV
    cache groups and thus having different block table.
    """

    query_start_loc: torch.Tensor
    """(batch_size + 1,), the start location of each request in query Tensor"""
    seq_lens: torch.Tensor
    """(batch_size,), the length of each request including both computed tokens
    and newly scheduled tokens"""
    total_num_chunks: int
    """scalar, chunk 数量"""
    seq_chunk_len_gpu_tensor: torch.Tensor
    """"""
    seq_delta_rotarys_gpu_tensor: torch.Tensor
    """"""
    cu_num_chunk_gpu_tensor: torch.Tensor
    """"""
    block_table_offsets: torch.Tensor

