import copy
import weakref
from typing import Dict, List, Set, Tuple

import torch

from vllm.model_executor.layers.sampler import SamplerOutput
from vllm.sequence import (ExecuteModelRequest, HiddenStates, SequenceData,
                           SequenceGroupMetadata)
from vllm.spec_decode.interfaces import (SpeculativeProposals,
                                         SpeculativeProposer)
from vllm.spec_decode.proposer_worker_base import ProposerWorkerBase
from vllm.spec_decode.top1_proposer import Top1Proposer
from vllm.worker.worker_base import RefWorkerBase, WorkerBase

class StandaloneMultiStepWorker(ProposerWorkerBase, RefWorkerBase):
    def __init__(
        self,
        referred_worker: WorkerBase,
        standalone_kv_compress_recover_rate: float
    ):
        RefWorkerBase.__init__(self, referred_worker)
        assert hasattr(self.worker, "model_runner")
        self.worker.model_runner = getattr(self.worker, "model_runner").model_runner
        self.standalone_kv_compress_recover_rate = standalone_kv_compress_recover_rate
        # Lazy initialization list.
        self._proposer: SpeculativeProposer
    
    def init_device(self) -> None:
        self.worker.init_device()
        self._proposer = Top1Proposer(
            weakref.proxy(self),  # type: ignore[arg-type]
            self.device,
            self.vocab_size,
            max_proposal_len=self.max_model_len,
        )
    
    def set_include_gpu_probs_tensor(self) -> None:
        self.model_runner.sampler.include_gpu_probs_tensor = True
        if hasattr(self.model_runner.model, "sampler"):
            (self.model_runner.model.sampler.include_gpu_probs_tensor) = True

    def set_should_modify_greedy_probs_inplace(self) -> None:
        self.model_runner.sampler.should_modify_greedy_probs_inplace = True
        if hasattr(self.model_runner.model, "sampler"):
            (self.model_runner.model.sampler.should_modify_greedy_probs_inplace
             ) = True
    
    @staticmethod
    def _shallow_copy_seq_group_metadata(
        seq_group_metadata: SequenceGroupMetadata, ) -> SequenceGroupMetadata:
        new_seq_group_metadata = copy.copy(seq_group_metadata)

        # We must shallow-copy seq_data as we will append token ids
        new_seq_data: Dict[int, SequenceData] = {}
        for seq_id, old_seq_data in seq_group_metadata.seq_data.items():
            new_seq_data[seq_id] = copy.copy(old_seq_data)
            new_seq_data[seq_id].output_token_ids =\
                old_seq_data.output_token_ids[:]

        new_seq_group_metadata.seq_data = new_seq_data
        return new_seq_group_metadata

    @torch.inference_mode()
    def sampler_output(
        self,
        execute_model_req: ExecuteModelRequest,
        sample_len: int,
        seq_ids_with_bonus_token_in_last_step: Set[int],
    ) -> Tuple[List[SamplerOutput], bool]:
        self._raise_if_unsupported(execute_model_req)

        # Difference from MultiStepWorker, 
        # it is not necessary to handle the bonus token.
        # The target model and scorer model is the same, so 
        # kvcache has been prefill by previous decoding step.

        # Standalone mode also does not need hidden states.

        seq_group_metadata_list_copy = []
        new_execute_model_req = execute_model_req.clone(seq_group_metadata_list_copy)
        for sg in execute_model_req.seq_group_metadata_list:
            seq_group_metadata_list_copy.append(
                StandaloneMultiStepWorker._shallow_copy_seq_group_metadata(sg)
            )
        new_execute_model_req.seq_group_metadata_list = seq_group_metadata_list_copy

        # Run model sample_len times.
        model_outputs: List[SamplerOutput] = []
        for _ in range(sample_len):
            model_output: List[SamplerOutput] = self.worker.execute_model(
                    execute_model_req=new_execute_model_req)
            assert (len(model_output) == 1
                        ), "composing multistep workers not supported"
            model_output = model_output[0]

            self._append_new_tokens(model_output, new_execute_model_req.seq_group_metadata_list)
            model_outputs.append(model_output)
        
        return model_outputs, True

    
    def get_spec_proposals(
        self,
        execute_model_req: ExecuteModelRequest,
        seq_ids_with_bonus_token_in_last_step: set,
    ) -> SpeculativeProposals:
        """Produce speculations given an input batch of sequences. The number of
        speculative tokens per sequence is determined by max_proposal_len.
        """
        return self._proposer.get_spec_proposals(
            execute_model_req, seq_ids_with_bonus_token_in_last_step)

    def _raise_if_unsupported(
        self,
        execute_model_req: ExecuteModelRequest,
    ) -> None:
        if any([
                execute_model_req.blocks_to_swap_in,
                execute_model_req.blocks_to_swap_out,
                execute_model_req.blocks_to_copy
        ]):
            raise NotImplementedError(
                "StandaloneMultiStepWorker does not support cache operations")

        if any(
                len(seq_group_metadata.seq_data.keys()) != 1
                for seq_group_metadata in
                execute_model_req.seq_group_metadata_list):
            raise NotImplementedError(
                "StandaloneMultiStepWorker does not support beam search.")
    
    @staticmethod
    def _append_new_tokens(
            model_output: List[SamplerOutput],
            seq_group_metadata_list: List[SequenceGroupMetadata]) -> None:
        for _, (seq_group_metadata, sequence_group_outputs) in enumerate(
                zip(seq_group_metadata_list, model_output)):
            seq_group_metadata.is_prompt = False

            for seq_output in sequence_group_outputs.samples:
                seq = seq_group_metadata.seq_data[seq_output.parent_seq_id]

                token_id = seq_output.output_token
                token_logprob = seq_output.logprobs[token_id]

                seq.append_token_id(token_id, token_logprob.logprob,
                                    seq_output.output_embed)
                seq.update_num_computed_tokens(1)
