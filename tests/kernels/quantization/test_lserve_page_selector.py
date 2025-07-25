import torch
import numpy as np
import random
from vllm import _custom_ops

def seed_everything(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# seed_everything(0)

batch_size = 10
num_q_head = 8
head_dim = 128
num_kv_head = 1
num_block = 4000
max_block_size = 4000
data_type = torch.float16

with torch.device("cuda:0"):
    # q = torch.randint(1, 41, (batch_size, num_q_head, head_dim))
    # q = q.to(dtype=torch.bfloat16)
    # key_meta_cache = torch.randint(1, 41, (num_block, num_kv_head, 2, head_dim))
    # key_meta_cache = key_meta_cache.to(dtype=torch.bfloat16)
    q = torch.rand((batch_size, num_q_head, head_dim), dtype=data_type)
    key_meta_cache = torch.rand((num_block, num_kv_head, 2, head_dim), dtype=data_type)
    # for i in range(num_q_head):
    #     q[:, i, :] = torch.ones((head_dim), dtype=torch.bfloat16) * (i + 1)
    # for i in range(num_kv_head):
    #     key_meta_cache[:, i, :, :] = torch.ones((head_dim), dtype=torch.bfloat16) * (i + 1)
    block_table = torch.randint(low=0, high=num_block, size=(batch_size, max_block_size), dtype=torch.int)
    # block_table = torch.tensor([[0]], dtype=torch.int)
    # num_full_blocks = torch.tensor([1], dtype=torch.int)
    num_full_blocks = torch.randint(low=max_block_size // 2, high=max_block_size+1, size=(batch_size,), dtype=torch.int)
    # num_full_blocks = torch.ones((batch_size,), dtype=torch.int) * max_block_size
    out_std = torch.zeros((batch_size, num_q_head, max_block_size), dtype=data_type)
    out = torch.zeros((batch_size, num_q_head, max_block_size), dtype=data_type)

    print(q)
    print()
    print(key_meta_cache)
    print()
    print(block_table)
    print()
    print(num_full_blocks)
    print()

    head_group_size = num_q_head // num_kv_head
    start_event1 = torch.cuda.Event(enable_timing=True)
    end_event1 = torch.cuda.Event(enable_timing=True)

    start_event2 = torch.cuda.Event(enable_timing=True)
    end_event2 = torch.cuda.Event(enable_timing=True)

    start_event1.record()
    for bi in range(batch_size):
        num_full_blk = num_full_blocks[bi].item()
        block_id = block_table[bi, :num_full_blk]
        kmc = key_meta_cache[block_id, ...]
        
        for qhead_id in range(num_q_head):
            # print(f"batch_id:{bi} qhead_id:{qhead_id}")
            khead_id = qhead_id // head_group_size
            cur_q = q[bi, qhead_id, :]
            cur_kmc_max = kmc[:, khead_id, 0, :]
            cur_kmc_min = kmc[:, khead_id, 1, :]
            # print(f"cur_q.shape={cur_q.shape}  cur_kmc_max.shape={cur_kmc_max.shape} cur_kmc_min.shape={cur_kmc_min.shape}")
            max_qk = cur_q * cur_kmc_max
            # print(max_qk)
            min_qk = cur_q * cur_kmc_min
            # print(min_qk)
            # print(min_qk.shape)
            concated_qk = torch.concat([max_qk.unsqueeze(1), min_qk.unsqueeze(1)], dim=1).max(dim=1).values
            # out_std[bi, qhead_id, ]
            cur_block_size = concated_qk.size(0)
            out_std[bi, qhead_id, :cur_block_size] = concated_qk.sum(dim=-1)
            # print(concated_qk.shape)
            # print(f"batch_size: {bi} qhead_id: {qhead_id} res: {concated_qk.sum(dim=-1)}")
            # print()
    end_event1.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event1.elapsed_time(end_event1)
    print(f"torch cost: {elapsed_time_ms}ms")
    print(f"std: {out_std}")
    print()

    print("=======================================")
    start_event2.record()
    for _ in range(20):
        torch.ops._C.lserve_page_selector(
            q,
            key_meta_cache,
            block_table,
            num_full_blocks,
            out,
        )
    end_event2.record()
    torch.cuda.synchronize()
    elapsed_time_ms = start_event2.elapsed_time(end_event2)
    print(f"cuda cost: {elapsed_time_ms/20}ms")
    print(out)
    print()

    print(torch.abs(out - out_std).flatten().max())
    # print(f"matched? {torch.allclose(out, out_std, atol=1)}")
