import torch
import torch.nn as nn
from typing import Union, Optional
from vllm.model_executor.custom_op import CustomOp

import random
import numpy as np

import triton
import triton.language as tl

def _rotate_neox(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., :x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    return torch.cat((-x2, x1), dim=-1)


def _rotate_gptj(x: torch.Tensor) -> torch.Tensor:
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    x = torch.stack((-x2, x1), dim=-1)
    return x.flatten(-2)

@triton.jit
def _rotary_embedding_kernel(
    positions: tl.tensor,   #[num_tokens, ]
    query: tl.tensor,       #[num_tokens, num_q_heads * head_dim]
    key: tl.tensor,         #[num_tokens, num_k_heads * head_dim]  
    cos_sin_cache: tl.tensor, #[max_pos_embedding, embed_dim + embed_dim]
    head_dim: tl.constexpr,
    query_head_dim: tl.constexpr,
    key_head_dim: tl.constexpr,
    rot_dim: tl.constexpr,
    # out_q,         #[num_tokens, num_q_heads * head_dim]
):
    assert rot_dim == head_dim
    assert query_head_dim % head_dim == 0
    assert key_head_dim % head_dim == 0

    token_id = tl.program_id(0)
    pos = tl.load(positions + token_id)

    query_per_head_ptr = (query + token_id * query_head_dim + tl.arange(0, query_head_dim // head_dim) * head_dim)[:, None]
    key_per_head_ptr = (key + token_id * key_head_dim + tl.arange(0, key_head_dim // head_dim) * head_dim)[:, None]
    cache_ptr = cos_sin_cache + pos * rot_dim

    q_first_half_ptr = query_per_head_ptr + (tl.arange(0, head_dim // 2))[None, :]
    q_first_half = tl.load(q_first_half_ptr)
    q_second_half_ptr = query_per_head_ptr + (tl.arange(head_dim // 2, head_dim))[None, :]
    q_second_half = tl.load(q_second_half_ptr)
    k_first_half_ptr = key_per_head_ptr + (tl.arange(0, head_dim // 2))[None, :]
    k_first_half = tl.load(k_first_half_ptr)
    k_second_half_ptr = key_per_head_ptr + (tl.arange(head_dim // 2, head_dim))[None, :]
    k_second_half = tl.load(k_second_half_ptr)
    cos_cache = tl.load(cache_ptr + (tl.arange(0, rot_dim // 2))[None, :])
    sin_cache = tl.load(cache_ptr + (tl.arange(rot_dim // 2, rot_dim))[None, :])

    qres1 = q_first_half * cos_cache + (-1) * q_second_half * sin_cache
    qres2 = q_second_half * cos_cache + q_first_half * sin_cache
    tl.store(q_first_half_ptr, qres1)
    tl.store(q_second_half_ptr, qres2)

    kres1 = k_first_half * cos_cache + (-1) * k_second_half * sin_cache
    kres2 = k_second_half * cos_cache + k_first_half * sin_cache
    tl.store(k_first_half_ptr, kres1)
    tl.store(k_second_half_ptr, kres2)



class DualChunkRotaryEmbedding(nn.Module):
    """Rotary positional embedding for Dual Chunk Attention."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: torch.dtype,
        chunk_size: int,
        local_size: int,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.chunk_size = chunk_size
        self.local_size = local_size
        self.dtype = dtype
        self.device = torch.device(f"cuda:{torch.cuda.current_device()}")
        (q_cache, qc_cache, k_cache, qc_no_clamp_cache,
         q_inter_cache) = self._compute_cos_sin_cache()

        self.register_buffer("cos_sin_q_cache", q_cache, persistent=False)
        self.register_buffer("cos_sin_qc_cache", qc_cache, persistent=False)
        self.register_buffer("cos_sin_k_cache", k_cache, persistent=False)
        self.register_buffer("cos_sin_qc_no_clamp_cache",
                             qc_no_clamp_cache,
                             persistent=False)
        self.register_buffer("cos_sin_q_inter_cache",
                             q_inter_cache,
                             persistent=False)

    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:
        """Compute the inverse frequency."""
        # NOTE(woosuk): The HF implementation uses `torch.arange(...).float()`.
        # However, we use `torch.arange(..., dtype=torch.float)` instead to
        # avoid numerical issues with large base values (e.g., 10000000).
        # This may cause a slight numerical difference between the HF
        # implementation and ours.
        # NOTE(woosuk): To exactly match the HF implementation, we need to
        # use CPU to compute the cache and then move it to GPU. However, we
        # create the cache on GPU for faster initialization. This may cause
        # a slight numerical difference between the HF implementation and ours.
        inv_freq = 1.0 / (base**(torch.arange(
            0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim))
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base)
        chunk_len = self.chunk_size - self.local_size
        # 这个应该是DCA chunk intra 的 cache
        q_t = torch.arange(chunk_len, dtype=torch.float)
        qc_t = (torch.arange(chunk_len, dtype=torch.float) +
                chunk_len).clamp(max=self.chunk_size)
        # k 的 position 继续沿用之前的 0 到 max_position_embeddings
        k_t = torch.arange(self.max_position_embeddings,
                           dtype=torch.float) % chunk_len

        # count from chunk_len, no clamp(self.chunk_size) restriction
        qc_no_clamp_t = torch.arange(chunk_len, dtype=torch.float) + chunk_len
        # count from self.chunk_size for q_inter's rope
        q_inter_t = torch.arange(chunk_len,
                                 dtype=torch.float) + self.chunk_size

        q_freqs = torch.outer(q_t, inv_freq)
        qc_freqs = torch.outer(qc_t, inv_freq)
        k_freqs = torch.outer(k_t, inv_freq)
        qc_no_clamp_freqs = torch.outer(qc_no_clamp_t, inv_freq)
        q_inter_freqs = torch.outer(q_inter_t, inv_freq)

        q_cos = q_freqs.cos()
        q_sin = q_freqs.sin()
        qc_cos = qc_freqs.cos()
        qc_sin = qc_freqs.sin()
        k_cos = k_freqs.cos()
        k_sin = k_freqs.sin()

        qc_no_clamp_cos = qc_no_clamp_freqs.cos()
        qc_no_clamp_sin = qc_no_clamp_freqs.sin()
        q_inter_cos = q_inter_freqs.cos()
        q_inter_sin = q_inter_freqs.sin()

        q_cache = torch.cat((q_cos, q_sin), dim=-1).to(dtype=self.dtype,
                                                       device=self.device)
        qc_cache = torch.cat((qc_cos, qc_sin), dim=-1).to(dtype=self.dtype,
                                                          device=self.device)
        k_cache = torch.cat((k_cos, k_sin), dim=-1).to(dtype=self.dtype,
                                                       device=self.device)
        qc_no_clamp_cache = torch.cat((qc_no_clamp_cos, qc_no_clamp_sin),
                                      dim=-1).to(dtype=self.dtype,
                                                 device=self.device)
        q_inter_cache = torch.cat((q_inter_cos, q_inter_sin),
                                  dim=-1).to(dtype=self.dtype,
                                             device=self.device)
        return q_cache, qc_cache, k_cache, qc_no_clamp_cache, q_inter_cache

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: torch.Tensor,
        offsets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        query = query.view(*query.shape[:-1], -1, self.head_size)
        key = key.view(*key.shape[:-1], -1, self.head_size)
        query_rot = query[..., :self.rotary_dim]
        key_rot = key[..., :self.rotary_dim]
        if self.rotary_dim < self.head_size:
            query_pass = query[..., self.rotary_dim:]
            key_pass = key[..., self.rotary_dim:]
        else:
            query_pass = None
            key_pass = None

        positions_with_offsets = (torch.add(positions, offsets)
                                  if offsets is not None else positions)
        key = self._apply_rotary_embedding(
            self.cos_sin_k_cache[positions_with_offsets], key_rot, key_pass)
        chunk_len = self.chunk_size - self.local_size
        query = self._apply_rotary_embedding(
            self.cos_sin_q_cache[positions_with_offsets % chunk_len],
            query_rot, query_pass)
        query_succ = self._apply_rotary_embedding(
            self.cos_sin_qc_cache[positions_with_offsets % chunk_len],
            query_rot, query_pass)
        query_inter = self._apply_rotary_embedding(
            self.cos_sin_qc_cache[chunk_len - 1].repeat(positions.shape[0], 1),
            query_rot, query_pass)
        query_succ_critical = self._apply_rotary_embedding(
            self.cos_sin_qc_no_clamp_cache[positions_with_offsets % chunk_len],
            query_rot, query_pass)
        query_inter_critical = self._apply_rotary_embedding(
            self.cos_sin_q_inter_cache[positions_with_offsets % chunk_len],
            query_rot, query_pass)

        # merge query into one tensor to simplify the interfaces
        query = torch.cat((
            query,
            query_succ,
            query_inter,
            query_succ_critical,
            query_inter_critical,
        ),
                          dim=-1)
        return query, key

    def _apply_rotary_embedding(self, cos_sin, hidden_rot, hidden_pass):
        cos, sin = cos_sin.chunk(2, dim=-1)
        print(f"1. {cos.shape} {sin.shape} {hidden_rot.shape}\n")
        if self.is_neox_style:
            # NOTE(woosuk): Here we assume that the positions tensor has the
            # shape [batch_size, seq_len].
            cos = cos.repeat(1, 1, 2).unsqueeze(-2)
            sin = sin.repeat(1, 1, 2).unsqueeze(-2)
            print(f"2. neox {cos.shape} {sin.shape}\n")
        else:
            cos = cos.repeat_interleave(2, dim=-1).unsqueeze(-2)
            sin = sin.repeat_interleave(2, dim=-1).unsqueeze(-2)
            print(f"2. simple {cos.shape} {sin.shape}\n")
        rotate_fn = _rotate_neox if self.is_neox_style else _rotate_gptj
        hidden_rot = hidden_rot * cos + rotate_fn(hidden_rot) * sin
        print(f"3. {hidden_rot.shape}\n")

        if self.rotary_dim < self.head_size:
            hidden = torch.cat((hidden_rot, hidden_pass), dim=-1)
        else:
            hidden = hidden_rot
        return hidden.flatten(-2).squeeze(0)

    def extra_repr(self) -> str:
        s = f"head_size={self.head_size}, rotary_dim={self.rotary_dim}"
        s += f", max_position_embeddings={self.max_position_embeddings}"
        s += f", base={self.base}, is_neox_style={self.is_neox_style}"
        s += f", chunk_size={self.chunk_size}, local_size={self.local_size}"
        return s

def _apply_rotary_emb_torch(
    x: torch.Tensor,
    cos: torch.Tensor,
    sin: torch.Tensor,
    is_neox_style: bool,
) -> torch.Tensor:
    cos = cos.unsqueeze(-2).to(x.dtype)
    sin = sin.unsqueeze(-2).to(x.dtype)
    if is_neox_style:
        x1, x2 = torch.chunk(x, 2, dim=-1)
    else:
        x1 = x[..., ::2]
        x2 = x[..., 1::2]
    o1 = x1 * cos - x2 * sin
    o2 = x2 * cos + x1 * sin
    if is_neox_style:
        return torch.cat((o1, o2), dim=-1)
    else:
        return torch.stack((o1, o2), dim=-1).flatten(-2)

class RotaryEmbedding(nn.Module):
    """Original rotary positional embedding."""

    def __init__(
        self,
        head_size: int,
        rotary_dim: int,
        max_position_embeddings: int,
        base: int,
        is_neox_style: bool,
        dtype: torch.dtype,
        triton_kernel: bool = True,
    ) -> None:
        super().__init__()
        self.head_size = head_size
        self.rotary_dim = rotary_dim
        self.max_position_embeddings = max_position_embeddings
        self.base = base
        self.is_neox_style = is_neox_style
        self.dtype = dtype
        self.triton_kernel = triton_kernel

        cache = self._compute_cos_sin_cache()
        cache = cache.to(dtype)
        self.cos_sin_cache: torch.Tensor
        self.register_buffer("cos_sin_cache", cache, persistent=False)

    def _compute_inv_freq(self, base: Union[int, float]) -> torch.Tensor:
        """Compute the inverse frequency."""
        # NOTE(woosuk): To exactly match the HF implementation, we need to
        # use CPU to compute the cache and then move it to GPU. However, we
        # create the cache on GPU for faster initialization. This may cause
        # a slight numerical difference between the HF implementation and ours.
        inv_freq = 1.0 / (base**(torch.arange(
            0, self.rotary_dim, 2, dtype=torch.float) / self.rotary_dim))
        return inv_freq

    def _compute_cos_sin_cache(self) -> torch.Tensor:
        """Compute the cos and sin cache."""
        inv_freq = self._compute_inv_freq(self.base)
        t = torch.arange(self.max_position_embeddings, dtype=torch.float)

        freqs = torch.einsum("i,j -> ij", t, inv_freq)
        cos = freqs.cos()
        sin = freqs.sin()
        cache = torch.cat((cos, sin), dim=-1)
        return cache

    def forward(
        self,
        positions: torch.Tensor,
        query: torch.Tensor,
        key: Optional[torch.Tensor] = None,
        offsets: Optional[torch.Tensor] = None,
    ) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
        """A PyTorch-native implementation of forward()."""
        if offsets is not None:
            positions = positions + offsets
        positions = positions.flatten()
        num_tokens = positions.shape[0]

        if self.triton_kernel:
            assert query.numel() % num_tokens == 0
            assert key.numel() % num_tokens == 0
            query_head_dim = query.numel() // num_tokens
            key_head_dim = key.numel() // num_tokens
            rot_dim = self.cos_sin_cache.shape[-1]
            # print(f"position shape: {positions.shape}")
            _rotary_embedding_kernel[(num_tokens,)](
                positions,
                query,
                key,
                self.cos_sin_cache,
                self.rotary_dim,
                query_head_dim,
                key_head_dim,
                rot_dim,
            )
        else:
            cos_sin = self.cos_sin_cache.index_select(0, positions)
            cos, sin = cos_sin.chunk(2, dim=-1)

            query_shape = query.shape
            query = query.view(num_tokens, -1, self.head_size)
            query_rot = query[..., :self.rotary_dim]
            query_pass = query[..., self.rotary_dim:]
            query_rot = _apply_rotary_emb_torch(query_rot, cos, sin,
                                                self.is_neox_style)
            query = torch.cat((query_rot, query_pass), dim=-1).reshape(query_shape)

            # key may be None in some cases, e.g. cross-layer KV sharing
            if key is not None:
                key_shape = key.shape
                key = key.view(num_tokens, -1, self.head_size)
                key_rot = key[..., :self.rotary_dim]
                key_pass = key[..., self.rotary_dim:]
                key_rot = _apply_rotary_emb_torch(key_rot, cos, sin,
                                                self.is_neox_style)
                key = torch.cat((key_rot, key_pass), dim=-1).reshape(key_shape)
            # from vllm import _custom_ops as ops

            # # __setattr__ in nn.Module (called by `self.cos_sin_cache = ...`)
            # # is expensive, so avoid calling it if possible
            # if self.cos_sin_cache.device != query.device or \
            #     self.cos_sin_cache.dtype != query.dtype:
            #     self.cos_sin_cache = self.cos_sin_cache.to(query.device,
            #                                             dtype=query.dtype)

            # # ops.rotary_embedding()/batched_rotary_embedding()
            # # are in-place operations that update the query and key tensors.
            # if offsets is not None:
            #     ops.batched_rotary_embedding(positions, query, key, self.head_size,
            #                                 self.cos_sin_cache,
            #                                 self.is_neox_style, self.rotary_dim,
            #                                 offsets)
            # else:
            #     ops.rotary_embedding(positions, query, key, self.head_size,
            #                         self.cos_sin_cache, self.is_neox_style)
        return query, key

def seed_everything(seed: int = 0):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

seed_everything()
num_head = 1
head_size = 128
rotary_dim = 128
max_position_embeddings = 1024000
base = 10000
is_neox_style = True
dtype = torch.bfloat16
chunk_size = 100
local_size = 20

seq_lens = [200000, 100000, 300000]

# torch.set_default_device("cuda:5")
with torch.device("cuda"):
    positions = [
        torch.arange(0, n)
        for n in seq_lens
    ]
    positions = torch.cat(positions, dim=-1)
    # print(positions.device)
    query = torch.rand(sum(seq_lens), head_size)
    key = torch.rand_like(query)

print(f"{query.shape} {key.shape}")
model = RotaryEmbedding(
    head_size=head_size,
    rotary_dim=rotary_dim,
    max_position_embeddings=max_position_embeddings,
    base=base,
    is_neox_style=is_neox_style,
    dtype=dtype,
    triton_kernel=False,
    # chunk_size=chunk_size,
    # local_size=local_size
).to("cuda")
import time
start = time.perf_counter()
torch.cuda.synchronize()
q, k = model(positions, query, key)
torch.cuda.synchronize()
end = time.perf_counter()
print(f"{end - start}")
print(q)
# print(q.cpu().numpy())
# print(k)
# print(f"{q.shape} & {k.shape}")