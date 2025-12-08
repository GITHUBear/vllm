# SPDX-License-Identifier: Apache-2.0

from collections import defaultdict
from dataclasses import dataclass
from typing import Optional, List
import xxhash

from vllm.distributed.kv_events import KVCacheEvent
from vllm.logger import init_logger
from vllm.utils import sha256
from vllm.v1.core.block_pool import BlockPool
from vllm.v1.core.kv_cache_utils import (BlockHashType, KVCacheBlocks,
                                         hash_request_tokens, KVCacheChunk,
                                         KVCacheChunkHitInfo, ChunkHitState)
from vllm.v1.core.single_type_kv_cache_manager import (
    get_manager_for_kv_cache_spec)
from vllm.v1.kv_cache_interface import KVCacheConfig
from vllm.v1.metrics.stats import PrefixCacheStats
from vllm.v1.request import Request, RequestStatus

logger = init_logger(__name__)

# def list_to_xxhash(nums: List[int]) -> str:
#     ba = bytearray()
#     for x in nums:
#         ba.extend(x.to_bytes(4, 'little', signed=True))
#     return str(xxhash.xxh64(ba).intdigest())


class KVCacheManager:

    def __init__(
        self,
        kv_cache_config: KVCacheConfig,
        max_model_len: int,
        enable_caching: bool = True,
        caching_hash_algo: str = "builtin",
        use_eagle: bool = False,
        log_stats: bool = False,
        enable_kv_cache_events: bool = False,
        enable_blk_caching: bool = False,
    ) -> None:
        assert len(kv_cache_config.kv_cache_groups) == 1, (
            "KVCacheManager does not support hybrid models with more than 1 "
            "kv cache group")
        kv_cache_spec = kv_cache_config.kv_cache_groups[0].kv_cache_spec
        self.block_size = kv_cache_spec.block_size
        self.num_gpu_blocks = kv_cache_config.num_blocks
        self.max_model_len = max_model_len

        self.enable_caching = enable_caching
        self.enable_blk_caching = enable_blk_caching
        self.caching_hash_fn = sha256 if caching_hash_algo == "sha256" else hash
        self.use_eagle = use_eagle
        self.log_stats = log_stats
        # FIXME: make prefix cache stats conditional on log_stats
        self.prefix_cache_stats = PrefixCacheStats() if log_stats else None

        self.block_pool = BlockPool(self.num_gpu_blocks, enable_caching,
                                    enable_kv_cache_events,
                                    enable_blk_caching=enable_blk_caching)

        self.single_type_manager = get_manager_for_kv_cache_spec(
            kv_cache_spec=kv_cache_spec,
            block_pool=self.block_pool,
            use_eagle=self.use_eagle,
            num_kv_cache_groups=1,
            caching_hash_fn=self.caching_hash_fn,
        )

        # Mapping from request ID to kv block hashes.
        # This is to avoid recomputing the block hashes for each call of
        # `get_computed_blocks` or `allocate_slots`.
        self.req_to_block_hashes: defaultdict[
            str, list[BlockHashType]] = defaultdict(list)
        
        # 只缓存一次，即使发生抢占
        self.req_to_chunk_hashes: Optional[defaultdict[str, list[str]]] = None
        # 在抢占并 resume 时重置
        self.blk_caching_req_to_chunk_ids: Optional[defaultdict[str, list[int]]] = None
        self.blk_caching_req_to_chunk_hit_info: Optional[defaultdict[str, list[KVCacheChunkHitInfo]]] = None
        if self.enable_blk_caching:
            self.req_to_chunk_hashes = defaultdict(list)
            self.blk_caching_req_to_chunk_ids = defaultdict(list)
            self.blk_caching_req_to_chunk_hit_info = defaultdict(list)

    @property
    def usage(self) -> float:
        """Get the KV cache usage.

        Returns:
            The KV cache usage (between 0.0 and 1.0).
        """
        return self.block_pool.get_usage()

    def make_prefix_cache_stats(self) -> Optional[PrefixCacheStats]:
        """Get (and reset) the prefix cache stats.

        Returns:
            The current prefix caching stats, or None if logging is disabled.
        """
        if not self.log_stats:
            return None
        stats = self.prefix_cache_stats
        self.prefix_cache_stats = PrefixCacheStats()
        return stats

    def _get_computed_blocks_for_prefix_caching(self, request: Request) -> tuple[KVCacheBlocks, int]:
        """Get the computed (cached) blocks for the request.
        Note that the computed blocks must be full.

        Args:
            request: The request to get the computed blocks.

        Returns:
            A tuple containing:
                - A list of blocks that are computed for the request.
                - The number of computed tokens.
        """
        # Prefix caching is disabled or
        # When the request requires prompt logprobs, we skip prefix caching.
        if (not self.enable_caching
                or request.sampling_params.prompt_logprobs is not None):
            return KVCacheBlocks.create_empty(), 0

        # The block hashes for the request may already be computed
        # if the scheduler has tried to schedule the request before.
        block_hashes = self.req_to_block_hashes[request.request_id]
        if not block_hashes:
            block_hashes = hash_request_tokens(self.caching_hash_fn,
                                               self.block_size, request)
            self.req_to_block_hashes[request.request_id] = block_hashes

        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.requests += 1

        # NOTE: When all tokens hit the cache, we must recompute the last token
        # to obtain logits. Thus, set max_cache_hit_length to prompt_length - 1.
        # This can trigger recomputation of an entire block, rather than just
        # the single last token, because allocate_slots() requires
        # num_computed_tokens to be block-size aligned. Removing this limitation
        # could slightly improve performance in the future.
        max_cache_hit_length = request.num_tokens - 1

        computed_blocks = self.single_type_manager.find_longest_cache_hit(
            block_hashes, max_cache_hit_length)
        # NOTE(woosuk): Since incomplete blocks are not eligible for
        # sharing, `num_computed_tokens` is always a multiple of
        # `block_size`.
        num_computed_tokens = len(computed_blocks) * self.block_size

        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.queries += request.num_tokens
            self.prefix_cache_stats.hits += num_computed_tokens

        return KVCacheBlocks(computed_blocks), num_computed_tokens

    def _get_computed_blocks_for_blk_caching(self, request: Request) -> tuple[List[KVCacheChunkHitInfo], int]:
        def list_to_xxhash(nums: List[int]) -> str:
            ba = bytearray()
            for x in nums:
                ba.extend(x.to_bytes(4, 'little', signed=True))
            return str(xxhash.xxh64(ba).intdigest())
        
        assert self.enable_blk_caching
        
        chunk_hashes = self.req_to_chunk_hashes[request.request_id]
        if not chunk_hashes:
            # chunk hashes 是无论被抢占多少次也一致的结果
            chunk_hashes = []
            for doc_range in request.doc_ranges:
                chunk_hashes.append(list_to_xxhash(request.all_token_ids[doc_range[0]:doc_range[1]]))
            self.req_to_chunk_hashes[request.request_id] = chunk_hashes

        if self.log_stats:
            # TODO[shk]: blk caching 统计信息
            pass

        computed_chunks = self.single_type_manager.find_chunk_hit(chunk_hashes)
        num_computed_tokens = 0
        computed_chunks_parsed: List[KVCacheChunkHitInfo] = []
        for chunk_hash, computed_chunk in zip(chunk_hashes, computed_chunks):
            if not computed_chunk:
                computed_chunks_parsed.append(KVCacheChunkHitInfo(
                    chunk_hash=chunk_hash,
                ))
                continue
            
            num_computed_tokens += len(computed_chunk._blocks) * self.block_size
            computed_chunks_parsed.append(KVCacheChunkHitInfo(
                hit_state=ChunkHitState.NATIVE_HIT,
                origin_rotary_offset=computed_chunk._rotary_offset,
                blocks=KVCacheBlocks(computed_chunk._blocks),
                chunk_hash=chunk_hash,
            ))

        # 每次被抢占后 resume 需要重新计算，所以这里直接覆盖
        native_hit_chunk_ids = []
        flattened_computed_chunks = []
        for idx, computed_chunk_parsed in enumerate(computed_chunks_parsed):
            if computed_chunk_parsed._hit_state == ChunkHitState.NATIVE_HIT:
                native_hit_chunk_ids.append(idx)
                flattened_computed_chunks.extend(computed_chunk_parsed._blocks.blocks)
        self.blk_caching_req_to_chunk_ids[request.request_id] = native_hit_chunk_ids
        self.blk_caching_req_to_chunk_hit_info[request.request_id] = computed_chunks_parsed
                
        if self.log_stats:
            # TODO[shk]: blk caching 统计信息
            pass

        return KVCacheBlocks(flattened_computed_chunks), num_computed_tokens

    def get_computed_blocks(self,
                            request: Request,
                            blk_caching_mode: bool = False) -> tuple[KVCacheBlocks | List[KVCacheChunkHitInfo], int]:
        if blk_caching_mode and self.enable_blk_caching:
            return self._get_computed_blocks_for_blk_caching(request)
        return self._get_computed_blocks_for_prefix_caching(request)
    
    def complete_block_caching_chunk_hit_infos(
        self,
        request: Request,
        external_hit_chunk_ids: List[int],
        external_rotary_offsets: List[int],
    ) -> None:
        if not self.enable_blk_caching:
            return
        
        assert len(external_hit_chunk_ids) == len(external_rotary_offsets)
        assert (request.request_id in self.blk_caching_req_to_chunk_ids and
                request.request_id in self.blk_caching_req_to_chunk_hit_info)

        doc_ranges = request.doc_ranges
        chunk_ids = self.blk_caching_req_to_chunk_ids[request.request_id]
        if len(doc_ranges) == len(chunk_ids):
            return
        
        chunk_ids.extend(external_hit_chunk_ids)
        chunk_ids_set = set(chunk_ids)
        assert len(chunk_ids_set) == len(chunk_ids)
        for idx in range(len(doc_ranges)):
            if idx not in chunk_ids_set:
                chunk_ids.append(idx)
        
        assert len(doc_ranges) == len(self.blk_caching_req_to_chunk_ids[request.request_id])

        chunk_hit_infos = self.blk_caching_req_to_chunk_hit_info[request.request_id]
        for chunk_id, rotary_offset in zip(external_hit_chunk_ids, external_rotary_offsets):
            chunk_hit_infos[chunk_id]._hit_state = ChunkHitState.EXTERNAL_HIT
            chunk_hit_infos[chunk_id]._origin_rotary_offset = rotary_offset

    def _allocate_slots_for_prefix_caching(
        self,
        request: Request,
        num_new_tokens: int,    # remote_key 还没准备好时，需要提前分配 external 命中的缓存
        num_new_computed_tokens: int = 0,   # num_native_computed_tokens
        new_computed_blocks: Optional[KVCacheBlocks] = None,    # new_computed_blocks
        num_lookahead_tokens: int = 0,
        delay_cache_blocks: bool = False,
    ) -> Optional[KVCacheBlocks]:
        """Add slots for a request with new tokens to append.

        Args:
            request: The request to allocate slots.
            num_new_tokens: The number of tokens to allocate, including external
                tokens. Note that this does not include tokens that have
                already been computed locally (i.e. new_computed_blocks).
            num_new_computed_tokens: The number of new computed tokens just
                hitting the prefix caching, excluding external tokens.
            new_computed_blocks: The cached blocks for the above new computed 
                tokens.
            num_lookahead_tokens: The number of speculative tokens to allocate.
                This is used by spec decode proposers with kv-cache such 
                as eagle.
            delay_cache_blocks: Whether to skip caching the blocks. This is
                used by P/D when allocating blocks used in a KV transfer
                which will complete in a future step.

        Blocks layout:
        ```
        -----------------------------------------------------------------------
        | < computed > | < new computed > |    < new >    | < pre-allocated > |
        -----------------------------------------------------------------------
        |                  < required >                   |
        --------------------------------------------------
        |                    < full >                  |
        ------------------------------------------------
                                          | <new full> |
                                          --------------
        ```
        The following *_blocks are illustrated in this layout.

        Returns:
            A list of new allocated blocks.
        """
        if num_new_tokens == 0:
            raise ValueError("num_new_tokens must be greater than 0")

        if new_computed_blocks is not None:
            new_computed_block_list = new_computed_blocks.blocks
        else:
            new_computed_block_list = []

        # Free the blocks that are skipped during the attention computation
        # (e.g., tokens outside the sliding window).
        # We can do this even if we cannot schedule this request due to
        # insufficient free blocks.
        # Should call this function before allocating new blocks to reduce
        # the number of evicted blocks.
        self.single_type_manager.remove_skipped_blocks(
            request.request_id, request.num_computed_tokens)

        # The number of computed tokens is the number of computed tokens plus
        # the new prefix caching hits
        num_computed_tokens = (request.num_computed_tokens +
                               num_new_computed_tokens)
        num_tokens_need_slot = min(
            num_computed_tokens + num_new_tokens + num_lookahead_tokens,
            self.max_model_len)
        num_blocks_to_allocate = (
            self.single_type_manager.get_num_blocks_to_allocate(
                request_id=request.request_id,
                num_tokens=num_tokens_need_slot,
                new_computed_blocks=new_computed_block_list,
            ))

        if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
            # Cannot allocate new blocks
            return None

        # Touch the computed blocks to make sure they won't be evicted.
        if self.enable_caching:
            self.block_pool.touch(new_computed_block_list)
        else:
            assert not new_computed_block_list, (
                "Computed blocks should be empty when "
                "prefix caching is disabled")

        # Append the new computed blocks to the request blocks until now to
        # avoid the case where the new blocks cannot be allocated.
        self.single_type_manager.save_new_computed_blocks(
            request.request_id, new_computed_block_list)

        new_blocks = self.single_type_manager.allocate_new_blocks(
            request.request_id, num_tokens_need_slot)

        # P/D: delay caching blocks if we have to recv from
        # remote. Update state for locally cached blocks.
        if not self.enable_caching or delay_cache_blocks:
            return KVCacheBlocks(new_blocks)

        # Speculated tokens might be rejected in the future, so we does
        # not cache any speculated tokens. We only cache blocks with
        # generated (accepted) tokens.
        self.single_type_manager.cache_blocks(
            request, self.req_to_block_hashes[request.request_id],
            num_computed_tokens + num_new_tokens - len(request.spec_token_ids))

        return KVCacheBlocks(new_blocks)
    
    def _allocate_slots_for_blk_caching(
        self,
        request: Request,
        num_new_tokens: int,    # remote_key 还没准备好时，需要提前分配 external 命中的缓存
        num_new_computed_tokens: int = 0,   # num_native_computed_tokens
        new_computed_blocks: Optional[KVCacheBlocks] = None,    # new_computed_blocks
        num_lookahead_tokens: int = 0,
        delay_cache_blocks: bool = False,
    ) -> Optional[KVCacheBlocks]:
        assert self.enable_blk_caching

        if num_new_tokens == 0:
            raise ValueError("num_new_tokens must be greater than 0")
        
        if new_computed_blocks is not None:
            new_computed_block_list = new_computed_blocks.blocks
        else:
            new_computed_block_list = []

        # block attention 尚不支持 slide window
        # self.single_type_manager.remove_skipped_blocks(
        #     request.request_id, request.num_computed_tokens)

        num_computed_tokens = (request.num_computed_tokens +
                               num_new_computed_tokens)
        num_tokens_need_slot = min(
            num_computed_tokens + num_new_tokens + num_lookahead_tokens,
            self.max_model_len)
        num_blocks_to_allocate = (
            self.single_type_manager.get_num_blocks_to_allocate(
                request_id=request.request_id,
                num_tokens=num_tokens_need_slot,
                new_computed_blocks=new_computed_block_list,
            ))
        
        if num_blocks_to_allocate > self.block_pool.get_num_free_blocks():
            # Cannot allocate new blocks
            return None
        
        assert (request.request_id in self.blk_caching_req_to_chunk_ids and
                request.request_id in self.blk_caching_req_to_chunk_hit_info)
        cache_ordered_chunk_idx = self.blk_caching_req_to_chunk_ids[request.request_id]
        chunk_hit_info = self.blk_caching_req_to_chunk_hit_info[request.request_id]
        doc_ranges = request.doc_ranges
        assert doc_ranges is not None

        self.block_pool.touch(new_computed_block_list)
        self.single_type_manager.save_new_computed_blocks(
            request.request_id, new_computed_block_list)
        if num_new_computed_tokens > 0:
            hit_tokens_num_for_check = 0
            for chunk_idx in cache_ordered_chunk_idx:
                if chunk_hit_info[chunk_idx]._hit_state != ChunkHitState.NATIVE_HIT:
                    break
                hit_tokens_num_for_check += (doc_ranges[chunk_idx][1] - doc_ranges[chunk_idx][0])
                cached_chunk = self.block_pool.get_cached_chunk(chunk_hit_info[chunk_idx]._chunk_hash)
                assert cached_chunk is not None
                self.block_pool.touch_chunk([cached_chunk])
            assert hit_tokens_num_for_check == num_new_computed_tokens
        
        if num_computed_tokens > doc_ranges[-1][1]:
            # chunk 区域全部分配过了
            # 正常分配即可
            new_blocks = self.single_type_manager.allocate_new_blocks(
                request_id=request.request_id,
                num_tokens=num_tokens_need_slot,
            )
            return KVCacheBlocks(new_blocks)

        remain_computed_tokens = num_computed_tokens
        remain_chunk_idxs = []
        need_refresh_block_ids = (num_computed_tokens + num_new_tokens >= doc_ranges[-1][1])
        for idx, chunk_idx in enumerate(cache_ordered_chunk_idx):
            chunk_len = doc_ranges[chunk_idx][1] - doc_ranges[chunk_idx][0]
            if remain_computed_tokens < chunk_len:
                remain_chunk_idxs = cache_ordered_chunk_idx[idx:]
                break
            remain_computed_tokens -= chunk_len

        # 要求必须对 chunk 对齐
        assert remain_computed_tokens == 0
        remain_new_tokens = num_new_tokens
        for idx, chunk_idx in enumerate(remain_chunk_idxs):
            chunk_len = doc_ranges[chunk_idx][1] - doc_ranges[chunk_idx][0]
            if remain_new_tokens < chunk_len:
                remain_chunk_idxs = remain_chunk_idxs[:idx]
                break
            remain_new_tokens -= chunk_len
        if not need_refresh_block_ids:
            assert remain_new_tokens == 0

        all_new_blocks = []
        cur_num_tokens_need_slot = num_computed_tokens
        for chunk_idx in remain_chunk_idxs:
            chunk_len = doc_ranges[chunk_idx][1] - doc_ranges[chunk_idx][0]
            cur_num_tokens_need_slot += chunk_len
            # 分配 chunk_len 长度的 slot
            new_blocks = self.single_type_manager.allocate_new_blocks(
                request_id=request.request_id,
                num_tokens=cur_num_tokens_need_slot,
            )
            all_new_blocks.extend(new_blocks)
            hit_info = chunk_hit_info[chunk_idx]
            hit_info._blocks = KVCacheBlocks(new_blocks)
            if hit_info._hit_state == ChunkHitState.EXTERNAL_HIT and delay_cache_blocks:
                # 如果是外部命中且是异步load，先不着急构造 KVCacheChunk 结构体加入缓存
                pass
            else:
                # 说明是没命中的 chunk，构造 KVCacheChunk 加入缓存
                new_chunk = KVCacheChunk(
                    blocks=new_blocks,
                    block_size=self.block_size,
                    chunk_hash=hit_info._chunk_hash,
                    chunk_start_token_offset=doc_ranges[chunk_idx][0],
                    chunk_end_token_offset=doc_ranges[chunk_idx][1],
                    num_tokens=doc_ranges[chunk_idx][2],
                    rotary_offset=doc_ranges[chunk_idx][3]
                )
                self.block_pool.touch_chunk([new_chunk])
                self.block_pool.cache_chunk(
                    chunk_hash=hit_info._chunk_hash,
                    chunk=new_chunk,
                )

        if remain_new_tokens > 0:
            cur_num_tokens_need_slot += (remain_new_tokens + num_lookahead_tokens)
            new_blocks = self.single_type_manager.allocate_new_blocks(
                request_id=request.request_id,
                num_tokens=cur_num_tokens_need_slot,
            )
            all_new_blocks.extend(new_blocks)

        if need_refresh_block_ids:
            ordered_chunk_blocks = []
            delta_rotary_offsets = []
            for hit_info, doc_range in zip(chunk_hit_info, doc_ranges):
                assert hit_info._blocks is not None
                ordered_chunk_blocks.extend(hit_info._blocks.blocks)
                if hit_info._hit_state == ChunkHitState.MISSED:
                    delta_rotary_offsets.append(0)
                else:
                    delta_rotary_offsets.append(doc_range[3] - hit_info._origin_rotary_offset)
            self.single_type_manager.refresh_blocks(
                request_id=request.request_id,
                new_blocks=ordered_chunk_blocks,
            )
            request.delta_rotary_offsets = delta_rotary_offsets
 
        return KVCacheBlocks(all_new_blocks)

    def allocate_slots(
        self,
        request: Request,
        num_new_tokens: int,    # remote_key 还没准备好时，需要提前分配 external 命中的缓存
        num_new_computed_tokens: int = 0,   # num_native_computed_tokens
        new_computed_blocks: Optional[KVCacheBlocks] = None,    # new_computed_blocks
        num_lookahead_tokens: int = 0,
        delay_cache_blocks: bool = False,
        blk_caching_mode: bool = False
    ) -> Optional[KVCacheBlocks]:
        if blk_caching_mode and self.enable_blk_caching:
            return self._allocate_slots_for_blk_caching(
                request,
                num_new_tokens,
                num_new_computed_tokens=num_new_computed_tokens,
                new_computed_blocks=new_computed_blocks,
                num_lookahead_tokens=num_lookahead_tokens,
                delay_cache_blocks=delay_cache_blocks,
            )
        return self._allocate_slots_for_prefix_caching(
            request,
            num_new_tokens,
            num_new_computed_tokens=num_new_computed_tokens,
            new_computed_blocks=new_computed_blocks,
            num_lookahead_tokens=num_lookahead_tokens,
            delay_cache_blocks=delay_cache_blocks,
        )

    # (chunked_new_len, offset, rotary_offset, context_len)
    def get_req_new_tokens_chunk_info(
        self,
        request: Request,
        num_computed_tokens: int,
        num_new_tokens: int,
        seq_chunk_lens: list[int],
        seq_delta_rotary_offsets: list[int],
        seq_chunk_num: list[int],
        seq_block_table_offset: list[tuple],
    ):  
        if (not self.enable_blk_caching) or (request.doc_ranges is None):
            seq_chunk_lens.append(num_computed_tokens + num_new_tokens)
            seq_delta_rotary_offsets.append(0)
            seq_chunk_num.append(1)
            seq_block_table_offset.append(0)
            return [(num_new_tokens, num_computed_tokens, 
                     num_computed_tokens, num_computed_tokens)]
        
        doc_ranges = request.doc_ranges
        assert request.request_id in self.blk_caching_req_to_chunk_ids
        cache_ordered_chunk_idx = self.blk_caching_req_to_chunk_ids[request.request_id]

        if num_computed_tokens > doc_ranges[-1][1]:
            for doc_range in doc_ranges:
                seq_chunk_lens.append(doc_range[2])
            if num_computed_tokens > doc_ranges[-1][1]:
                seq_chunk_lens.append(num_computed_tokens + num_new_tokens - doc_ranges[-1][1])

            assert request.delta_rotary_offsets is not None
            seq_delta_rotary_offsets.extend(request.delta_rotary_offsets)
            if num_computed_tokens > doc_ranges[-1][1]:
                seq_delta_rotary_offsets.append(0)
            seq_chunk_num.append(len(seq_chunk_lens))

            seq_block_table_offset.append(0)

            # 注意这里的 rotary offset 的计算
            return [(num_new_tokens, 
                     num_computed_tokens,
                     num_computed_tokens + doc_ranges[-1][3] + doc_ranges[-1][2] - doc_ranges[-1][1],
                     num_computed_tokens)]

        remain_computed_tokens = num_computed_tokens
        remain_chunk_idxs = []
        for idx, chunk_idx in enumerate(cache_ordered_chunk_idx):
            chunk_len = doc_ranges[chunk_idx][1] - doc_ranges[chunk_idx][0]
            if remain_computed_tokens < chunk_len:
                assert remain_computed_tokens == 0
                remain_chunk_idxs = cache_ordered_chunk_idx[idx:]
                break
            remain_computed_tokens -= chunk_len

        remain_new_tokens = num_new_tokens
        chunk_infos = []
        for idx, chunk_idx in enumerate(remain_chunk_idxs):
            chunk_len = doc_ranges[chunk_idx][1] - doc_ranges[chunk_idx][0]
            if remain_new_tokens < chunk_len:
                assert remain_new_tokens == 0
                return chunk_infos
            seq_chunk_lens.append(chunk_len)
            seq_delta_rotary_offsets.append(0)
            seq_chunk_num.append(1)
            seq_block_table_offset.append(doc_ranges[chunk_idx][0] // self.block_size)
            chunk_infos.append((chunk_len, doc_ranges[chunk_idx][0], 
                                doc_ranges[chunk_idx][3], 0))
            remain_new_tokens -= chunk_len
        
        if remain_new_tokens > 0:
            for doc_range in doc_ranges:
                seq_chunk_lens.append(doc_range[2])
            seq_chunk_lens.append(num_computed_tokens + num_new_tokens - doc_ranges[-1][1])
            assert request.delta_rotary_offsets is not None
            seq_delta_rotary_offsets.extend(request.delta_rotary_offsets)
            seq_delta_rotary_offsets.append(0)
            seq_chunk_num.append(len(doc_ranges) + 1)
            seq_block_table_offset.append(0)
            chunk_infos.append((remain_new_tokens,
                                doc_ranges[-1][1], 
                                doc_ranges[-1][3] + doc_ranges[-1][2],
                                doc_ranges[-1][1]))

        return chunk_infos
        

    def cache_external_async_load_chunks(
        self,
        request: Request,
    ) -> None:
        assert (request.request_id in self.blk_caching_req_to_chunk_hit_info)
        chunk_hit_info = self.blk_caching_req_to_chunk_hit_info[request.request_id]
        doc_ranges = request.doc_ranges
        assert doc_ranges is not None

        for hit_info, doc_range in zip(chunk_hit_info, doc_ranges):
            assert hit_info._blocks is not None
            new_chunk = KVCacheChunk(
                blocks=hit_info._blocks.blocks,
                block_size=self.block_size,
                chunk_hash=hit_info._chunk_hash,
                chunk_start_token_offset=doc_range[0],
                chunk_end_token_offset=doc_range[1],
                num_tokens=doc_range[2],
                rotary_offset=doc_range[3]
            )
            self.block_pool.touch_chunk([new_chunk])
            self.block_pool.cache_chunk(
                chunk_hash=hit_info._chunk_hash,
                chunk=new_chunk,
            )

    def free(self, request: Request) -> None:
        """Free the blocks allocated for the request.
        We free the blocks in reverse order so that he tail blocks are evicted 
        first when caching is enabled.

        Args:
            request: The request to free the blocks.
        """
        self.single_type_manager.free(request.request_id)

        if self.req_to_chunk_hashes is not None:
            chunk_hashes = self.req_to_chunk_hashes.pop(request.request_id, [])
            self.block_pool.free_chunks(chunk_hashes)
        if self.blk_caching_req_to_chunk_ids is not None:
            self.blk_caching_req_to_chunk_ids.pop(request.request_id, [])
        if self.blk_caching_req_to_chunk_hit_info is not None:
            self.blk_caching_req_to_chunk_hit_info.pop(request.request_id, [])

    def reset_prefix_cache(self) -> bool:
        """Reset prefix cache. This function may be used in RLHF
        flows to invalidate prefix caching after the weights are updated,
        or used for resetting prefix caching status for benchmarking.

        Returns:
            bool: True if the prefix cache is successfully reset,
            False otherwise.
        """
        if not self.block_pool.reset_prefix_cache():
            return False
        if self.log_stats:
            assert self.prefix_cache_stats is not None
            self.prefix_cache_stats.reset = True
        return True

    def get_num_common_prefix_blocks(
        self,
        request: Request,
        num_running_requests: int,
    ) -> int:
        """Calculate the number of common prefix blocks shared by all requests
        in the RUNNING state.

        The function determines this by selecting any request and iterating
        through its blocks.  A block is considered a common prefix block if its
        `ref_cnt` equals the total number of requests in the RUNNING state.

        NOTE(woosuk): The number of requests in the RUNNING state is **greater
        than or equal to** the number of requests scheduled in the current step.
        This is because the RUNNING state only indicates that:
        1. The request has not yet finished, and
        2. The request holds its blocks unfreed.

        While all scheduled requests must be in the RUNNING state, the inverse
        is not necessarily true. There may be RUNNING requests that are not
        scheduled in the current step.

        This can result in an edge case where the number of common prefix blocks
        is 0, even though all scheduled requests share a common prefix. This
        occurs because there may be unscheduled RUNNING requests that do not
        share the common prefix. Currently, this case cannot be easily detected,
        so the function returns 0 in such cases.

        Args:
            request: Any request in the RUNNING state, used to identify the
                common prefix blocks.
            num_running_requests: The total number of requests in the RUNNING
                state. This can be different from the number of scheduled
                requests in the current step.

        Returns:
            int: The number of common prefix blocks.
        """
        assert request.status == RequestStatus.RUNNING
        return self.single_type_manager.get_num_common_prefix_blocks(
            request.request_id, num_running_requests)

    def free_block_hashes(self, request: Request) -> None:
        """Discard the block hashes for the request.

        NOTE: Unlike `free`, this method should be called only when the request
        is finished, not when it is preempted.
        """
        self.req_to_block_hashes.pop(request.request_id, None)

    def take_events(self) -> list[KVCacheEvent]:
        """Take the KV cache events from the block pool.

        Returns:
            A list of KV cache events.
        """
        return self.block_pool.take_events()

    def get_block_ids(self, request_id: str) -> list[int]:
        """Get the block ids of a request."""
        assert request_id in self.single_type_manager.req_to_blocks
        return [
            block.block_id
            for block in self.single_type_manager.req_to_blocks[request_id]
        ]
