import logging
from typing import Dict

from vllm.core.interfaces import AllocStatus
from vllm.sequence import (Sequence, SequenceGroup, SequenceData,
                           SequenceGroupMetadata, SequenceGroupMetadataDelta)

SeqId = int
BlockId = int

logger = logging.getLogger(__name__)

class SparseIndexBlockManager:
    def __init__(
        self,
        num_gpu_blocks: int,
        seqlen_threshold: int,
        recompute_step: int,
        num_sample_tokens: int,
        block_size: int,
    ):
        self.num_gpu_blocks = num_gpu_blocks
        self.seqlen_threshold = seqlen_threshold
        self.recompute_step = recompute_step
        self.num_sample_tokens = num_sample_tokens
        self.block_size = block_size
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
            if seq.is_prefill():
                continue
            # 考虑 decode 额外增加的一个 token
            if seq.get_num_computed_tokens() + 1 < self.seqlen_threshold:
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
                # 如果这个 seq 需要重新更新 compress page，则需要更新 _num_compressed_page
                seq.data.set_num_computed_tokens_to_compress(
                    need_check=True,
                    recompute_index_step=self.recompute_step,
                    block_size=self.block_size,
                )
                continue
            if seq.is_prefill():
                continue
            # 考虑 decode 额外增加的一个 token
            if seq.get_num_computed_tokens() + 1 < self.seqlen_threshold:
                continue
            # if seq.get_num_computed_tokens() - seq.get_prompt_len() < self.num_sample_tokens:
            #     continue


            blk_id = self.free_block_ids[0]
            self.free_block_ids = self.free_block_ids[1:]
            self.seqid_block_id_mapping[seq.seq_id] = blk_id
            # TODO[shk]: 修改为参与压缩的页面数量
            # seq.data.set_num_computed_tokens_when_enable_sparse_index()
            seq.data.set_num_computed_tokens_to_compress()
            logger.info(f"Allocate Sparse Index Block:{blk_id} for req:{seq_group.request_id} seq:{seq.seq_id}")

    def build_seq_group_meta_sparse_index_table(self, sgm: SequenceGroupMetadata):
        sgm.sparse_index_table = {}
        for seq_id in sgm.seq_data.keys():
            if seq_id not in self.seqid_block_id_mapping:
                continue
            assert (sgm.seq_data[seq_id]._num_computed_tokens_to_compress is not None)
            blk_id = self.seqid_block_id_mapping[seq_id]
            sgm.sparse_index_table[seq_id] = blk_id

    def build_seq_group_metadelta_sparse_index_table(
            self, seq_data: Dict[int, SequenceData], 
            sgmd: SequenceGroupMetadataDelta):
        sgmd.sparse_index_table = {}
        for seq_id in seq_data.keys():
            if seq_id not in self.seqid_block_id_mapping:
                continue
            assert (seq_data[seq_id]._num_computed_tokens_to_compress is not None)
            blk_id = self.seqid_block_id_mapping[seq_id]
            sgmd.sparse_index_table[seq_id] = blk_id

    def free(self, seq: Sequence):
        if seq.seq_id in self.seqid_block_id_mapping:
            blk_id = self.seqid_block_id_mapping[seq.seq_id]
            self.free_block_ids.append(blk_id)
            self.seqid_block_id_mapping.pop(seq.seq_id)
            seq.data.reset_num_computed_tokens_to_compress()
            # seq.data.reset_num_computed_tokens_when_enable_sparse_index()
            logger.info(f"Free Sparse Index Block:{blk_id} for seq:{seq.seq_id}")
