import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import os
from vllm.model_executor.layers.rotary_embedding import RotaryEmbedding

tp_size = 8
layer_num = 64
num_head_per_rank = 5
num_key_head_per_rank = 1
common_dir = "./common_dump/"
cache_blend_dir = "./blend_dump/"

# doc_ranges = [(57, 1030), (1035, 1308), (1313, 3728), (3733, 5820), (5825, 7084), (7089, 7253), (7258, 8771), (8776, 10751), (10756, 11137), (11143, 11737)]
doc_ranges = [(1035, 1308), (1313, 3728), (3733, 5820), (5825, 7084), (7089, 7253), (7258, 8771), (8776, 10751), (10756, 11137), (11143, 11737)]

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

tp_rank = 0
common_data_path = common_dir + f"tensor_{tp_rank}.hdf5"
cb_data_path = cache_blend_dir + f"tensor_{tp_rank}.hdf5"
print(common_data_path)
print(cb_data_path)
with h5py.File(common_data_path, 'r') as common_data, h5py.File(cb_data_path, 'r') as cb_data:
    for idx, doc_range in enumerate(doc_ranges):
        (start_idx, end_idx) = doc_range
        for j in range(2, layer_num):
            kkey = f"K_{j}"
            vkey = f"V_{j}"
            common_k_tensor = torch.from_numpy(np.array(common_data[kkey])).to(device="cuda:0")
            common_v_tensor = torch.from_numpy(np.array(common_data[vkey])).to(device="cuda:0")
            cb_k_tensor = torch.from_numpy(np.array(cb_data[kkey])).to(device="cuda:0")
            cb_v_tensor = torch.from_numpy(np.array(cb_data[vkey])).to(device="cuda:0")

            common_k = common_k_tensor[start_idx:end_idx, 0, :].cpu()
            common_v = common_v_tensor[start_idx:end_idx, 0, :].cpu()
            cb_k = cb_k_tensor[start_idx:end_idx, 0, :].cpu()
            cb_v = cb_v_tensor[start_idx:end_idx, 0, :].cpu()

            fig, axes = plt.subplots(1, 2, figsize=(18, 18))
            # k_diff = cb_k - common_k
            # print(cb_k)
            # print(common_k)
            im1 = plot_heatmap(axes[0], cb_k - common_k, 'K diff')
            fig.colorbar(im1, ax=axes[0], location='right', shrink=0.6, pad=0.02)
            im2 = plot_heatmap(axes[1], cb_v - common_v, 'V diff')
            fig.colorbar(im2, ax=axes[1], location='right', shrink=0.6, pad=0.02)

            plt.tight_layout()
            plt.savefig(f"./cb_diff_fig/heatmap_doc_{idx}_level_{j}.png", dpi=150, bbox_inches='tight')
            print(f"output figure: doc_id:{idx} level:{j}...")
            plt.close()
            # exit(0)