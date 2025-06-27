import h5py
import torch
import numpy as np
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

tp_size = 8
layer_num = 48
num_head_per_rank = 5
dirname = "./decode_attn_dump/"
head_dim = 128
insight_step = 5000

recover_rate = 0.95

for head_id in range(num_head_per_rank * tp_size):
    tp_rank = head_id // num_head_per_rank
    file_name = f"{insight_step}_tensor_{tp_rank}.hdf5"
    data_name = dirname + file_name

    attn_sims = []
    attn_tensors = []
    with h5py.File(data_name, 'r') as f_attn:
        for layer_id in range(layer_num):
            key = f"{layer_id}"
            attn_np = np.array(f_attn[key])
            attn_tensor = torch.from_numpy(attn_np).to(device="cuda:0")
            attn_tensors.append(attn_tensor)

    for part in range(6):
        plt.figure(figsize=(12, 6))
        plt.xlabel('Layer ID')
        plt.ylabel('Attention Score')
        plt.grid(True, linestyle='--', alpha=0.6)
        for layer_id in range(part * 8, part * 8 + 8):
            cur_layer_hd_attn = attn_tensors[layer_id][head_id % num_head_per_rank, :]
            cur_layer_hd_attn_sorted_values, cur_layer_hd_attn_sorted_indices = torch.sort(cur_layer_hd_attn, dim=-1, descending=True)

            cum_sorted_hd_values = cur_layer_hd_attn_sorted_values.cumsum(dim=-1)
            topk = torch.searchsorted(cum_sorted_hd_values, cum_sorted_hd_values[-1].item() * recover_rate, side='left')

            kept_indices = cur_layer_hd_attn_sorted_indices[:min(topk + 1, cur_layer_hd_attn_sorted_indices.shape[0])]

            cur_sim = []
            for succ_layer_id in range(layer_id, layer_num):
                cur_sim.append(attn_tensors[succ_layer_id][head_id % num_head_per_rank, kept_indices].sum(dim=-1).item())
            
            x = np.arange(layer_id, layer_id + len(cur_sim))
            y = np.array(cur_sim)
            plt.plot(x, y, linewidth=1.5, label=f'Layer {layer_id}')
        
        print(f"================ OUTPUT FIGURE {insight_step}_HEAD-{head_id}_p{part}.jpg... ===========")
        plt.legend()
        plt.savefig(f"./decode_attn_visual/{insight_step}_Head_{head_id}_p{part}.jpg")
        plt.close()