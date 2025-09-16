import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

tp_size = 8
layer_num = 48
num_head_per_rank = 5
num_key_head_per_rank = 1
dirname = "./prefill_qkv_dump_single1/"
# doc_ids = [i for i in range(10)]
# doc_split_pos = [(39, 1017), (1017, 1295), (1295, 3715), (3715, 5807), (5807, 7071), (7071, 7240), (7240, 8758), (8758, 10738), (10738, 11124), (11124, 11725)]
# doc_ids = [5, 1, 2, 3, 4, 0, 6, 7, 8, 9]
doc_split_pos = [(39, 208), (208, 486), (486, 2906), (2906, 4998), (4998, 6262), (6262, 7240), (7240, 8758), (8758, 10738), (10738, 11124), (11124, 11725)]
# [(39, 208), (208, 1186), (1186, 2450), (2450, 3050), (3050, 3328), (3328, 5308), (5308, 5694), (5694, 7786), (7786, 10206), (10206, 11725)]
# 0:[(39, 1017)]
# 1:[(39, 317)]
# 2:[(39, 2459)]
# 3:[(39, 2131)]
# 4:[(39, 1303)]
# 5:[(39, 208)]
# 6:[(39, 1557)]
# 7:[(39, 2019)]
# 8:[(39, 425)]
# 9:[(39, 639)]
# for i in range(tp_size):
i = 0
data_path = dirname + f"tensor_{i}.hdf5"
with h5py.File(data_path, 'r') as data:
    for j in range(layer_num):
        qkey = f"Q_{j}"
        kkey = f"V_{j}"
        vkey = f"K_{j}"
        q_tensor = torch.from_numpy(np.array(data[qkey])).to(device="cuda:0")
        k_tensor = torch.from_numpy(np.array(data[kkey])).to(device="cuda:0")
        v_tensor = torch.from_numpy(np.array(data[vkey])).to(device="cuda:0")
        # print(q_tensor.shape)
        # print(k_tensor[:, 0, :])
        # print(v_tensor[:, 0, :].shape)
        for doc_idx, (doc_s_id, doc_e_id) in enumerate(doc_split_pos):
            q = q_tensor[doc_s_id:doc_e_id, 1, :].cpu()
            k = k_tensor[doc_s_id:doc_e_id, 0, :].cpu()
            v = v_tensor[doc_s_id:doc_e_id, 0, :].cpu()

            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            im1 = axes[0].imshow(q, cmap='RdBu_r', aspect='auto')
            axes[0].set_title('Q', fontsize=14)
            axes[0].set_xlabel('Hidden state dim')
            axes[0].set_ylabel('Doc Length')

            im2 = axes[1].imshow(k, cmap='RdBu_r', aspect='auto')
            axes[1].set_title('K', fontsize=14)
            axes[1].set_xlabel('Hidden state dim')

            im3 = axes[2].imshow(v, cmap='RdBu_r', aspect='auto')
            axes[2].set_title('V', fontsize=14)
            axes[2].set_xlabel('Hidden state dim')

            # fig.colorbar(label="Value")
            fig.colorbar(im3, ax=axes, location='right', shrink=0.6, pad=0.02).set_label('Value', fontsize=12)
            plt.savefig(f"./qkv_single1/heatmap_doc_{doc_idx}_level_{j}.png", dpi=150, bbox_inches='tight')
            plt.close()
            print(f"output figure: doc_id:{doc_idx} level:{j}...")
