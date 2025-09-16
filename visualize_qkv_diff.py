import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import os
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding

tp_size = 8
layer_num = 48
num_head_per_rank = 5
num_key_head_per_rank = 1
dirname1 = "./prefill_qkv_dump/"
dirname2 = "./prefill_qkv_dump2/"
dirname3 = "./prefill_qkv_dump_rand1/"
single_dirname_list = [
    f"./prefill_qkv_dump_single{i}/"
    for i in range(10)
]

doc_ids1 = [i for i in range(10)]
doc_map1 = {
    0:0, 1:1, 2:2,
    3:3, 4:4, 5:5,
    6:6, 7:7, 8:8,
    9:9
}
# 978, 278, 2420, 2092, 1264, 169, 1518, 1980, 386, 601
doc_split_pos1 = [(39, 1017), (1017, 1295), (1295, 3715), (3715, 5807), (5807, 7071), (7071, 7240), (7240, 8758), (8758, 10738), (10738, 11124), (11124, 11725)]
doc_ids2 = [5, 1, 2, 3, 4, 0, 6, 7, 8, 9]
doc_map2 = {
    0:5, 1:1, 2:2,
    3:3, 4:4, 5:0,
    6:6, 7:7, 8:8,
    9:9
}
doc_split_pos2 = [(39, 208), (208, 486), (486, 2906), (2906, 4998), (4998, 6262), (6262, 7240), (7240, 8758), (8758, 10738), (10738, 11124), (11124, 11725)]
doc_ids3 = [4, 9, 6, 3, 0, 1, 8, 7, 5, 2]
doc_map3 = {
    0:4, 1:5, 2:9,
    3:3, 4:0, 5:8,
    6:2, 7:7, 8:6,
    9:1
}
# [1264, 600, 1518, 2092, 978, 278, 386, 1980, 169, 2421]
doc_split_pos3 = [(39, 1303), (1303, 1903), (1903, 3421), (3421, 5513), (5513, 6491), (6491, 6769), (6769, 7155), (7155, 9135), (9135, 9304), (9304, 11725)]
# dl = [(e - s) for (s, e) in doc_split_pos3]

single_doc_map = {
    0:[(39, 1017)],
    1:[(39, 317)],
    2:[(39, 2459)],
    3:[(39, 2131)],
    4:[(39, 1303)],
    5:[(39, 208)],
    6:[(39, 1557)],
    7:[(39, 2019)],
    8:[(39, 425)],
    9:[(39, 639)]
}

def plot_heatmap(ax, data, title):
    """封装：绘制一个带对称归一化的热力图"""
    vmin, vmax = data.min(), data.max()
    # 确保包含 0，并对称映射
    norm = TwoSlopeNorm(vcenter=0.0, vmin=vmin, vmax=vmax)
    im = ax.imshow(data, cmap='RdBu_r', aspect='auto', norm=norm)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('Hidden state dim')
    if ax.get_subplotspec().colspan.start == 0:  # 第一列才加 ylabel
        ax.set_ylabel('Doc Length')
    return im

rotary = RotaryEmbedding(
    head_size=128,
    rotary_dim=128,
    max_position_embeddings=131072,
    base=1000000.0,
    is_neox_style=True,
    dtype=torch.bfloat16,
)

def apply_rotary(
    query:torch.Tensor,
    key:torch.Tensor,
    offset
):
    if offset == 0:
        return query, key
    origin_type = query.dtype
    query = query.to(dtype=torch.bfloat16).cuda(device="cuda:0")
    key = key.to(dtype=torch.bfloat16).cuda(device="cuda:0")
    tensor_len = query.shape[0]
    assert key.shape[0] == tensor_len
    positions = torch.tensor([offset]*tensor_len, dtype=torch.int64, device=query.device)
    query, key = rotary.forward_cuda(
        positions=positions,
        query=query,
        key=key
    )
    query = query.to(dtype=origin_type).cpu()
    key = key.to(dtype=origin_type).cpu()
    return query, key
    

tp_rank = 0
ordered_data_path = dirname1 + f"tensor_{tp_rank}.hdf5"
simple_swap_data_path = dirname2 + f"tensor_{tp_rank}.hdf5"
rand_data_path = dirname3 + f"tensor_{tp_rank}.hdf5"
with h5py.File(ordered_data_path, 'r') as ordered_data, h5py.File(simple_swap_data_path, 'r') as simple_swap_data, h5py.File(rand_data_path, 'r') as rand_data:
    for doc_id in range(10):
        single_data_path = single_dirname_list[doc_id] + f"tensor_{tp_rank}.hdf5"
        with h5py.File(single_data_path, 'r') as single_data:
            for j in range(layer_num):
                qkey = f"Q_{j}"
                kkey = f"V_{j}"
                vkey = f"K_{j}"
                single_q_tensor = torch.from_numpy(np.array(single_data[qkey])).to(device="cuda:0")
                single_k_tensor = torch.from_numpy(np.array(single_data[kkey])).to(device="cuda:0")
                single_v_tensor = torch.from_numpy(np.array(single_data[vkey])).to(device="cuda:0")
                ordered_q_tensor = torch.from_numpy(np.array(ordered_data[qkey])).to(device="cuda:0")
                ordered_k_tensor = torch.from_numpy(np.array(ordered_data[kkey])).to(device="cuda:0")
                ordered_v_tensor = torch.from_numpy(np.array(ordered_data[vkey])).to(device="cuda:0")
                swap_q_tensor = torch.from_numpy(np.array(simple_swap_data[qkey])).to(device="cuda:0")
                swap_k_tensor = torch.from_numpy(np.array(simple_swap_data[kkey])).to(device="cuda:0")
                swap_v_tensor = torch.from_numpy(np.array(simple_swap_data[vkey])).to(device="cuda:0")
                rand_q_tensor = torch.from_numpy(np.array(rand_data[qkey])).to(device="cuda:0")
                rand_k_tensor = torch.from_numpy(np.array(rand_data[kkey])).to(device="cuda:0")
                rand_v_tensor = torch.from_numpy(np.array(rand_data[vkey])).to(device="cuda:0")

                single_s, single_e = single_doc_map[doc_id][0]
                ordered_s, ordered_e = doc_split_pos1[doc_map1[doc_id]]
                swap_s, swap_e = doc_split_pos2[doc_map2[doc_id]]
                rand_s, rand_e = doc_split_pos3[doc_map3[doc_id]]
                doc_len = single_e - single_s
                # print(f"doc_len: {doc_len} order_len:{ordered_e - ordered_s} swap_len:{swap_e - swap_s} rand_len:{rand_e - rand_s}")
                # assert (ordered_e - ordered_s) == doc_len and (swap_e - swap_s) == doc_len
                act_len = min(doc_len, rand_e - rand_s)
                single_e = single_s + act_len
                ordered_e = ordered_s + act_len
                swap_e = swap_s + act_len
                rand_e = rand_s + act_len

                single_q = single_q_tensor[single_s:single_e, 1, :].cpu()
                ordered_q = ordered_q_tensor[ordered_s:ordered_e, 1, :].cpu()
                swap_q = swap_q_tensor[swap_s:swap_e, 1, :].cpu()
                rand_q = rand_q_tensor[rand_s:rand_e, 1, :].cpu()

                single_k = single_k_tensor[single_s:single_e, 0, :].cpu()
                ordered_k = ordered_k_tensor[ordered_s:ordered_e, 0, :].cpu()
                swap_k = swap_k_tensor[swap_s:swap_e, 0, :].cpu()
                rand_k = rand_k_tensor[rand_s:rand_e, 0, :].cpu()

                single_v = single_v_tensor[single_s:single_e, 0, :].cpu()
                ordered_v = ordered_v_tensor[ordered_s:ordered_e, 0, :].cpu()
                swap_v = swap_v_tensor[swap_s:swap_e, 0, :].cpu()
                rand_v = rand_v_tensor[rand_s:rand_e, 0, :].cpu()

                fig, axes = plt.subplots(3, 3, figsize=(18, 18))
                
                single_q_align_ordered, single_k_align_ordered = apply_rotary(
                    query=single_q,
                    key=single_k,
                    offset=ordered_s - single_s,
                )
                im1 = plot_heatmap(axes[0, 0], ordered_q - single_q_align_ordered, 'Ordered Q diff')
                fig.colorbar(im1, ax=axes[0,0], location='right', shrink=0.6, pad=0.02)
                im2 = plot_heatmap(axes[0, 1], ordered_k - single_k_align_ordered, 'Ordered K diff')
                fig.colorbar(im2, ax=axes[0,1], location='right', shrink=0.6, pad=0.02)
                im3 = plot_heatmap(axes[0, 2], ordered_v - single_v, 'Ordered V diff')
                fig.colorbar(im3, ax=axes[0,2], location='right', shrink=0.6, pad=0.02)

                single_q_align_swap, single_k_align_swap = apply_rotary(
                    query=single_q,
                    key=single_k,
                    offset=swap_s - single_s,
                )
                im4 = plot_heatmap(axes[1, 0], swap_q - single_q_align_swap, 'Swap Q diff')
                fig.colorbar(im4, ax=axes[1,0], location='right', shrink=0.6, pad=0.02)
                im5 = plot_heatmap(axes[1, 1], swap_k - single_k_align_swap, 'Swap K diff')
                fig.colorbar(im5, ax=axes[1,1], location='right', shrink=0.6, pad=0.02)
                im6 = plot_heatmap(axes[1, 2], swap_v - single_v, 'Swap V diff')
                fig.colorbar(im6, ax=axes[1,2], location='right', shrink=0.6, pad=0.02)

                single_q_align_rand, single_k_align_rand = apply_rotary(
                    query=single_q,
                    key=single_k,
                    offset=rand_s - single_s,
                )
                im7 = plot_heatmap(axes[2, 0], rand_q - single_q_align_rand, 'Random Q diff')
                fig.colorbar(im7, ax=axes[2,0], location='right', shrink=0.6, pad=0.02)
                im8 = plot_heatmap(axes[2, 1], rand_k - single_k_align_rand, 'Random K diff')
                fig.colorbar(im8, ax=axes[2,1], location='right', shrink=0.6, pad=0.02)
                im9 = plot_heatmap(axes[2, 2], rand_v - single_v, 'Random V diff')
                fig.colorbar(im9, ax=axes[2,2], location='right', shrink=0.6, pad=0.02)

                plt.tight_layout()
                plt.savefig(f"./qkv_diff3/heatmap_doc_{doc_id}_level_{j}.png", dpi=150, bbox_inches='tight')
                print(f"output figure: doc_id:{doc_id} level:{j}...")
                plt.close()
                # exit(0)
