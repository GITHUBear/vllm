import logging
from typing import Dict

from vllm.core.interfaces import AllocStatus
from vllm.sequence import Sequence, SequenceGroup, SequenceStatus

SeqId = int
BlockId = int

logger = logging.getLogger(__name__)

class SparseIndexBlockManager:
    def __init__(
        self,
        num_gpu_blocks: int,
        seqlen_threshold: int,
    ):
        self.num_gpu_blocks = num_gpu_blocks
        self.seqlen_threshold = seqlen_threshold
        self.free_block_ids = [
            i for i in range(self.num_gpu_blocks)
        ]
        self.seqid_block_id_mapping: Dict[SeqId, BlockId] = {}

    def can_allocate(self, seq_group: SequenceGroup) -> AllocStatus:
        assert (not seq_group.is_encoder_decoder())
        num_seq_need_allocate = 0
        for seq in seq_group.seqs:
            if seq.seq_id in self.seqid_block_id_mapping:
                continue
            if len(seq.get_token_ids()) < self.seqlen_threshold:
                continue
            num_seq_need_allocate += 1

        if num_seq_need_allocate > self.num_gpu_blocks:
            logger.warn(f"num_seq_need_allocate is {num_seq_need_allocate} too long "
                        f"and exceeds the capacity of sparse_index_block_manager")
            return AllocStatus.NEVER
        if num_seq_need_allocate <= len(self.free_block_ids):
            return AllocStatus.OK
        return AllocStatus.LATER

    def allocate(self, seq_group: SequenceGroup):
        for seq in seq_group.seqs:
            if seq.seq_id in self.seqid_block_id_mapping:
                continue
            if len(seq.get_token_ids()) < self.seqlen_threshold:
                continue

            blk_id = self.free_block_ids[0]
            self.free_block_ids = self.free_block_ids[1:]
            self.seqid_block_id_mapping[seq.seq_id] = blk_id
            logger.info(f"Allocate Sparse Index Block:{blk_id} for req:{seq_group.request_id} seq:{seq.seq_id}")

    def free(self, seq: Sequence):
        if seq.seq_id in self.seqid_block_id_mapping:
            blk_id = self.seqid_block_id_mapping[seq.seq_id]
            self.free_block_ids.append(blk_id)
            self.seqid_block_id_mapping.pop(seq.seq_id)
            logger.info(f"Free Sparse Index Block:{blk_id} for seq:{seq.seq_id}")
