# SPDX-License-Identifier: Apache-2.0
import xxhash
import enum
from collections import deque
from typing import Deque, FrozenSet, Iterable, List, Optional, Tuple, Union, Dict

from vllm.core.block.common import (BlockPool, CopyOnWriteTracker, RefCounter,
                                    get_all_blocks_recursively)
from vllm.core.block.interfaces import Block, BlockAllocator, BlockId, Device
from vllm.core.evictor import LRUChunkBasedEvictor, make_evictor, EvictionPolicy

Refcount = int

class ChunkMeta:
    def __init__(
        self,
        block_ids: List[int],
        block_size: int,
        chunk_hash: str,
        chunk_start_token_offset: int,
        chunk_end_token_offset: int,
        num_tokens: int,
        rotary_offset: int,
        block_ref_counter: RefCounter,
    ):
        assert (chunk_start_token_offset % block_size == 0) and (chunk_end_token_offset % block_size == 0)
        self._block_ids = block_ids
        self._chunk_start_token_offset = chunk_start_token_offset
        self._chunk_end_token_offset = chunk_end_token_offset
        self._num_tokens = num_tokens
        self._rotary_offset = rotary_offset
        self._block_size = block_size
        self._chunk_hash = chunk_hash

        self._ref_count = 0
        self._computed = False
        self._block_ref_counter = block_ref_counter

        self._last_accessed = -1.0  # 是否直接由 ChunkMeta 维护待定

        for block_id in self._block_ids:
            self._block_ref_counter.incr(block_id)

    def is_valid(self):
        return self._computed
    
    def incr_ref_count(self):
        self._ref_count += 1
        return self._ref_count

    def decr_ref_count(self):
        self._ref_count -= 1
        return self._ref_count
    
    def mark_as_computed(self):
        self._computed = True


class ChunkAllocationState(enum.Enum):
    NOT_A_CHUNK = 0
    HIT_CHUNK = 1
    FUTURE_CHUNK = 2


class ChunkAllocationInfo:
    def __init__(
        self,
        state: ChunkAllocationState,
        chunk_start_token_offset: Optional[int],
        chunk_end_token_offset: Optional[int],
        rotary_offset: Optional[int],
        chunk_hash: Optional[str],
        origin_chunk_rotary_offset: Optional[int],
    ):
        self._state = state
        self._chunk_hash = chunk_hash
        self._chunk_start_token_offset = chunk_start_token_offset
        self._chunk_end_token_offset = chunk_end_token_offset
        self._rotary_offset = rotary_offset
        self._delta_rotray_offset = None
        if origin_chunk_rotary_offset is not None:
            assert rotary_offset is not None
            self._delta_rotray_offset = rotary_offset - origin_chunk_rotary_offset
    
    def __repr__(self):
        return (
            f"[\nstate:{self._state}\n"
            f"chunk_start:{self._chunk_start_token_offset}\n"
            f"chunk_end:{self._chunk_end_token_offset}\n"
            f"chunk_hash:{self._chunk_hash}\n"
            f"rotary_offset:{self._rotary_offset}\n"
            f"delta_rotray_offset:{self._delta_rotray_offset}\n]"
        )


class ChunkCachingBlockAllocator(BlockAllocator):
    def __init__(
        self,
        create_block: Block.Factory,
        num_blocks: int,
        block_size: int,
        block_ids: Optional[Iterable[int]] = None,
        block_pool: Optional[BlockPool] = None,
    ):
        if block_ids is None:
            block_ids = range(num_blocks)
        
        self._free_block_indices: Deque[BlockId] = deque(block_ids)
        self._all_block_indices = frozenset(block_ids)
        assert len(self._all_block_indices) == num_blocks

        self._refcounter = RefCounter(
            all_block_indices=self._free_block_indices)
        self._block_size = block_size

        # 由于缓存的 chunk 一定 block_size 对齐，所以 chunk 内的 block 一定不会发生 COW
        # 此处保留仅为 beam search
        self._cow_tracker = CopyOnWriteTracker(
            refcounter=self._refcounter.as_readonly())

        if block_pool is None:
            extra_factor = 4
            # Pre-allocate "num_blocks * extra_factor" block objects.
            # The "* extra_factor" is a buffer to allow more block objects
            # than physical blocks
            self._block_pool = BlockPool(self._block_size, create_block, self,
                                         num_blocks * extra_factor)
        else:
            # In this case, the block pool is provided by the caller,
            # which means that there is most likely a need to share
            # a block pool between allocators
            self._block_pool = block_pool
        
        self._cached_chunk: Dict[str, ChunkMeta] = {}
        self._chunk_evictor: Optional[LRUChunkBasedEvictor] = make_evictor(
            eviction_policy=EvictionPolicy.LRU, 
            chunk_based=True,
            block_ref_counter=self._refcounter
        )
        self._touched_chunk: List[str] = []
    
    @staticmethod
    def _list_to_xxhash(nums: List[int]) -> str:
        ba = bytearray()
        for x in nums:
            ba.extend(x.to_bytes(4, 'little', signed=True))
        return str(xxhash.xxh64(ba).intdigest())
    

    def allocate_immutable_block(self,
                                 prev_block: Optional[Block],
                                 token_ids: List[int],
                                 extra_hash: Optional[int] = None,
                                 device: Optional[Device] = None) -> Block:
        """Allocates a new immutable block with the given token IDs, linked to
        the previous block.

        Args:
            prev_block (Optional[Block]): The previous block in the sequence. If
                None, then the block to be allocated is the first block in the
                sequence.
            token_ids (List[int]): The token IDs to be stored in the new block.

        Returns:
            Block: The newly allocated immutable block.
        """
        # 目前这个方法不会被调用
        assert False
        assert device is None
        block = self.allocate_mutable_block(prev_block=prev_block)
        block.append_token_ids(token_ids)
        return block
    
    def _incr_refcount_cached_chunk(self, chunk_meta: ChunkMeta):
        chunk_meta.incr_ref_count()
        if chunk_meta._ref_count == 1:
            if chunk_meta._chunk_hash in self._chunk_evictor:
                self._chunk_evictor.remove(chunk_meta._chunk_hash)
    
    def _decr_refcount_cached_chunk(self, chunk_hash: str):
        assert chunk_hash in self._cached_chunk
        chunk_meta = self._cached_chunk[chunk_hash]
        assert chunk_meta.is_valid()

        refcount = chunk_meta.decr_ref_count()
        if refcount > 0:
            return
        
        assert refcount == 0
        self._chunk_evictor.add(
            chunk_hash=chunk_hash,
            block_ids=chunk_meta._block_ids,
            last_accessed=chunk_meta._last_accessed,
        )

    # 调用此接口的位置需要改动
    def allocate_immutable_blocks(
        self,
        prev_block: Optional[Block],
        block_token_ids: List[List[int]],
        extra_hash: Optional[int] = None,
        device: Optional[Device] = None,
        
        doc_range: Optional[Tuple] = None,
        chunk_alloc_states: Optional[List[ChunkAllocationInfo]] = None,
        can_be_cached: bool = True,
        chunk_hash_cached: Optional[str] = None) -> List[Block]:
        assert device is None
        # assert doc_range is not None
        blocks = []

        if doc_range is not None:
            assert chunk_alloc_states is not None
            # 是可命中缓存或者留作未来命中的 chunk
            if chunk_hash_cached is None:
                token_ids = [token_id for block_token_id in block_token_ids for token_id in block_token_id]
                chunk_hash = ChunkCachingBlockAllocator._list_to_xxhash(token_ids)
            else:
                chunk_hash = chunk_hash_cached

            promote_to_cache = True
            if chunk_hash in self._cached_chunk:
                if self._cached_chunk[chunk_hash].is_valid():
                    # 缓存命中
                    chunk_meta = self._cached_chunk[chunk_hash]
                    assert chunk_meta._num_tokens == doc_range[2] and len(chunk_meta._block_ids) == len(block_token_ids)
                    self._incr_refcount_cached_chunk(chunk_meta)

                    for block_token_id, cached_block_id in zip(block_token_ids, chunk_meta._block_ids):
                        self._refcounter.incr(cached_block_id)
                        prev_block = self._block_pool.init_block(
                            prev_block=prev_block,
                            token_ids=block_token_id,
                            block_size=self._block_size,
                            physical_block_id=cached_block_id
                        )
                        blocks.append(prev_block)
                    chunk_alloc_states.append(
                        ChunkAllocationInfo(
                            state=ChunkAllocationState.HIT_CHUNK,
                            chunk_start_token_offset=doc_range[0],
                            chunk_end_token_offset=doc_range[1],
                            rotary_offset=doc_range[3],
                            chunk_hash=chunk_hash,
                            origin_chunk_rotary_offset=chunk_meta._rotary_offset,
                        )
                    )
                    return blocks

                # 命中了还在 compute 的块，不需要登记为可命中的 chunk
                promote_to_cache = False
            
            # 分配块
            block_ids = []
            for i in range(len(block_token_ids)):
                allocated_block_id = self._allocate_block_id()
                prev_block = self._block_pool.init_block(
                    prev_block=prev_block,
                    token_ids=block_token_ids[i],
                    block_size=self._block_size,
                    physical_block_id=allocated_block_id,
                )
                blocks.append(prev_block)
                block_ids.append(allocated_block_id)

            # 缓存未命中，且可以登记为可命中的 chunk
            if promote_to_cache and can_be_cached:
                new_chunk_meta = ChunkMeta(
                    block_ids=block_ids,
                    block_size=self._block_size,
                    chunk_hash=chunk_hash,
                    chunk_start_token_offset=doc_range[0],
                    chunk_end_token_offset=doc_range[1],
                    num_tokens=doc_range[2],
                    rotary_offset=doc_range[3],
                    block_ref_counter=self._refcounter,
                )
                self._incr_refcount_cached_chunk(new_chunk_meta)
                self._cached_chunk[chunk_hash] = new_chunk_meta
            
            chunk_alloc_states.append(
                ChunkAllocationInfo(
                    state=ChunkAllocationState.FUTURE_CHUNK,
                    chunk_start_token_offset=doc_range[0],
                    chunk_end_token_offset=doc_range[1],
                    rotary_offset=doc_range[3],
                    chunk_hash=chunk_hash,
                    origin_chunk_rotary_offset=None,
                )
            )
            self._touched_chunk.append(chunk_hash)
            return blocks
        
        for i in range(len(block_token_ids)):
            prev_block = self._block_pool.init_block(
                prev_block=prev_block,
                token_ids=block_token_ids[i],
                block_size=self._block_size,
                physical_block_id=self._allocate_block_id(),
            )
            blocks.append(prev_block)
        return blocks

    def allocate_mutable_block(self,
                               prev_block: Optional[Block],
                               extra_hash: Optional[int] = None,
                               device: Optional[Device] = None) -> Block:
        assert device is None
        block_id = self._allocate_block_id()
        block = self._block_pool.init_block(prev_block=prev_block,
                                            token_ids=[],
                                            block_size=self._block_size,
                                            physical_block_id=block_id)
        return block

    def _maybe_allocate_common_block_id(self) -> Optional[BlockId]:
        if not self._free_block_indices:
            return None

        block_id = self._free_block_indices.popleft()
        self._refcounter.incr(block_id)
        return block_id
    
    def _maybe_allocate_evicted_block_id(self) -> Optional[BlockId]:
        if self._chunk_evictor.num_blocks == 0:
            return None
        
        evicted_block_ids, chunk_hash = self._chunk_evictor.evict()
        assert chunk_hash in self._cached_chunk and len(evicted_block_id) > 0
        chunk_meta = self._cached_chunk[chunk_hash]
        assert chunk_meta._ref_count == 0 and chunk_meta.is_valid()
        self._cached_chunk.pop(chunk_hash)
        for evicted_block_id in evicted_block_ids:
            assert self._refcounter.get(evicted_block_id) == 0
            self._free_block_indices.appendleft(evicted_block_id)
        
        block_id = self._free_block_indices.popleft()
        self._refcounter.incr(block_id)
        return block_id

    def _allocate_block_id(self) -> BlockId:
        common_block_id = self._maybe_allocate_common_block_id()
        if common_block_id is not None:
            return common_block_id
        
        evicted_block_id = self._maybe_allocate_evicted_block_id()
        if evicted_block_id is not None:
            return evicted_block_id
        
        raise BlockAllocator.NoFreeBlocksError()
    
    def _free_block_id(self, block: Union[Block, BlockId]) -> None:
        if isinstance(block, Block):
            block_id = block.block_id
            block.block_id = None
        else:
            block_id = block
        assert block_id is not None

        refcount = self._refcounter.decr(block_id)
        if refcount == 0:
            self._free_block_indices.appendleft(block_id)

    def free(self, block: Block, keep_block_object: bool = False) -> None:
        # Release the physical block id
        self._free_block_id(block)

        # Release the block object
        if not keep_block_object:
            self._block_pool.free_block(block)

    def free_block_id(self, block_id: BlockId) -> None:
        self._free_block_id(block_id)

    def free_chunks(self, chunk_hashes: Optional[List[str]] = None) -> None:
        if chunk_hashes is not None:
            for chunk_hash in chunk_hashes:
                self._decr_refcount_cached_chunk(chunk_hash)
    
    # 调用此接口的位置需要改动
    def fork(self, last_block: Block, chunk_hashes: Optional[List[str]] = None) -> List[Block]:
        """Creates a new sequence of blocks that shares the same underlying
        memory as the original sequence.

        Args:
            last_block (Block): The last block in the original sequence.

        Returns:
            List[Block]: The new sequence of blocks that shares the same memory
                as the original sequence.
        """
        source_blocks = get_all_blocks_recursively(last_block)

        forked_blocks: List[Block] = []
        prev_block = None
        for block in source_blocks:

            # Increment refcount for each block.
            assert block.block_id is not None
            refcount = self._refcounter.incr(block.block_id)
            assert refcount != 1, "can't fork free'd block"

            forked_block = self._block_pool.init_block(
                prev_block=prev_block,
                token_ids=block.token_ids,
                block_size=self._block_size,
                physical_block_id=block.block_id)

            forked_blocks.append(forked_block)
            prev_block = forked_blocks[-1]

        if chunk_hashes is not None:
            for chunk_hash in chunk_hashes:
                assert chunk_hash in self._cached_chunk
                self._incr_refcount_cached_chunk(self._cached_chunk[chunk_hash])
        
        return forked_blocks
    
    def get_num_free_blocks(self) -> int:
        return len(self._free_block_indices) + self._chunk_evictor.num_blocks
    
    def get_num_total_blocks(self) -> int:
        return len(self._all_block_indices)
    
    def get_physical_block_id(self, absolute_id: int) -> int:
        """Returns the zero-offset block id on certain block allocator
        given the absolute block id.

        Args:
            absolute_id (int): The absolute block id for the block 
            in whole allocator.

        Returns:
            int: The zero-offset block id on certain device.
        """
        return sorted(self._all_block_indices).index(absolute_id)
    
    @property
    def refcounter(self):
        return self._refcounter

    @property
    def all_block_ids(self) -> FrozenSet[int]:
        return self._all_block_indices
    
    def cow_block_if_not_appendable(self, block: Block) -> BlockId:
        """Performs a copy-on-write operation on the given block if it is not
        appendable.

        Args:
            block (Block): The block to check for copy-on-write.

        Returns:
            BlockId: The block index of the new block if a copy-on-write 
                operation was performed, or the original block index if
                no copy-on-write was necessary.
        """
        src_block_id = block.block_id
        assert src_block_id is not None

        if self._cow_tracker.is_appendable(block):
            return src_block_id

        self._free_block_id(block)
        trg_block_id = self._allocate_block_id()

        self._cow_tracker.record_cow(src_block_id, trg_block_id)

        return trg_block_id

    def clear_copy_on_writes(self) -> List[Tuple[BlockId, BlockId]]:
        """Returns the copy-on-write source->destination mapping and clears it.

        Returns:
            List[Tuple[BlockId, BlockId]]: A list mapping source
                block indices to destination block indices.
        """
        return self._cow_tracker.clear_cows()
    
    # 调用此接口的位置需要改动
    def mark_blocks_as_computed(self, block_ids: List[int]) -> None:
        """Mark blocks as computed, used in prefix caching.

        Since the naive allocator does not implement prefix caching, we do
        nothing.
        """
        for chunk_hash in self._touched_chunk:
            self._cached_chunk[chunk_hash]._computed = True
        self._touched_chunk.clear()

    # 调用此接口的位置需要改动
    def mark_blocks_as_accessed(self, block_ids: List[int],
                                now: float,
                                chunk_hashes: Optional[List[str]] = None) -> None:
        """Mark blocks as accessed, used in prefix caching.

        Since the naive allocator does not implement prefix caching, we do
        nothing.
        """
        assert chunk_hashes is not None
        for chunk_hash in chunk_hashes:
            self._cached_chunk[chunk_hash]._last_accessed = now
    
    def get_common_computed_block_ids(
            self, computed_seq_block_ids: List[List[int]]) -> List[int]:
        """Determine blocks that can be skipped in prefill.

        Since the naive allocator does not support prefix caching, always return
        an empty list.
        """
        return []

    def promote_to_immutable_block(self, block: Block) -> BlockId:
        raise NotImplementedError("There is no promotion for naive blocks")

    def get_num_full_blocks_touched(self, blocks: List[Block]) -> int:
        """Returns the number of full blocks that will be touched by
        swapping in/out.

        Args:
            blocks: List of blocks to be swapped.
        Returns:
            int: the number of full blocks that will be touched by
                swapping in/out the given blocks. Non full blocks are ignored
                when deciding the number of blocks to touch.
        """
        raise ValueError("Not implemented!")
        # NOTE: for naive block, we use set to eliminate common blocks among
        # seqs, also we compare the empty slots in the mutable blocks with
        # lookahead slots to get the number of unique new block that are
        # needed.
        old_block_set = set()
        for block in blocks:
            if block.is_full:
                old_block_set.add(block)
        return len(old_block_set)

    def swap_out(self, blocks: List[Block]) -> None:
        raise ValueError("Not implemented!")
        for block in blocks:
            self._free_block_id(block)

    def swap_in(self, blocks: List[Block]) -> None:
        raise ValueError("Not implemented!")
        for block in blocks:
            # Here we allocate either immutable or mutable block and then
            # extract its block_id. Note that the block object is released
            # and the block_id is assigned to "block" to allow reusing the
            # existing "block" object
            if block.is_full:
                tmp_block = self.allocate_immutable_block(
                    prev_block=block.prev_block, token_ids=block.token_ids)
            else:
                tmp_block = self.allocate_mutable_block(
                    prev_block=block.prev_block)
                tmp_block.append_token_ids(block.token_ids)

            block_id = tmp_block.block_id
            tmp_block.block_id = None
            self._block_pool.free_block(tmp_block)

            block.block_id = block_id  # Assign block_id

    def get_prefix_cache_hit_rate(self) -> float:
        return -1

    def reset_prefix_cache(self) -> bool:
        """No prefix cache for naive block allocator."""
        return True

    def find_cached_blocks_prefix(self, block_hashes: List[int]) -> List[int]:
        # Not applicable for naive block allocator.
        return []

# allocate_immutable_blocks
# 传入一个 chunk 块内的按照 block 分开的 token_ids 数组
# 判断这些 token_ids 的 hash 值是否命中缓存，命中则复用这些 block_id,
# 否则分配新的 block_ids
    
class ChunkAllocationTracker:
    def __init__(self, allocator: ChunkCachingBlockAllocator):
        self._allocator = allocator
        self._seq_chunk_alloc_info: Dict[int, List[ChunkAllocationInfo]] = {}
    
    def add_seq(self, seq_id: int, chunk_alloc_info: List[ChunkAllocationInfo]) -> None:
        """Start tracking seq_id
        """
        assert seq_id not in self._seq_chunk_alloc_info
        self._seq_chunk_alloc_info[seq_id] = chunk_alloc_info
    
    def get_rotary_position_offsets(self, seq_id: int):
        assert seq_id in self._seq_chunk_alloc_info
        chunk_alloc_info = self._seq_chunk_alloc_info[seq_id]
        # 未命中设置为 -1
        if len(chunk_alloc_info) == 0:
            return [-1]
        rotary_pos_offsets = []
        for info in chunk_alloc_info:
            if info._state == ChunkAllocationState.HIT_CHUNK:
                rotary_pos_offsets.append(info._delta_rotray_offset)
            else:
                rotary_pos_offsets.append(-1)
        return rotary_pos_offsets

    def remove_seq(self, seq_id: int) -> None:
        """Stop tracking seq_id
        """
        assert seq_id in self._seq_chunk_alloc_info
        chunk_alloc_info = self._seq_chunk_alloc_info[seq_id]
        chunk_hashes = [info._chunk_hash for info in chunk_alloc_info]
        self._allocator.free_chunks(chunk_hashes)
        del self._seq_chunk_alloc_info[seq_id]
    
    def get_chunk_hashes(self, seq_id: int) -> List[str]:
        assert seq_id in self._seq_chunk_alloc_info
        chunk_alloc_info = self._seq_chunk_alloc_info[seq_id]
        return [info._chunk_hash for info in chunk_alloc_info]
