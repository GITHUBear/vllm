import h5py
import torch
import numpy as np
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt

tp_size = 4
layer_num = 28
num_head_per_rank = 7
dirname = "./attn_map_dump2/"
head_dim = 128

recover_rate = 0.95

def sum_all_diagonal_matrix(mat: torch.tensor):
    h, n, m = mat.shape
    # Zero matrix used for padding
    zero_mat = torch.zeros((h, n, n), device=mat.device)
    # pads the matrix on left and right
    mat_padded = torch.cat((zero_mat, mat, zero_mat), -1)
    # Change the strides
    mat_strided = mat_padded.as_strided((h, n, n + m),
                                        (n * (2 * n + m), 2 * n + m + 1, 1))
    # Sums the resulting matrix's columns
    sum_diags = torch.sum(mat_strided, 1)
    return sum_diags[:, 1:]  # drop left bottom corner

for i in range(tp_size):
    file_name = f"tensor_{i}.hdf5"
    attn_dataname = dirname + file_name

    with h5py.File(attn_dataname, 'r') as f_attn:
        for j in range(27, layer_num, 1):
            key = f"{j}"
            attn_np = np.array(f_attn[key])
            attn_tensor = torch.from_numpy(attn_np).to(device="cuda:0")
            
            num_head, qlen, klen = attn_tensor.shape
            attn_tensor[:, :, -qlen:] = torch.where(
                torch.tril(torch.ones((num_head, qlen, qlen), dtype=torch.bool, device="cuda:0")),
                attn_tensor[:, :, -qlen:],
                -torch.inf
            )
            attn_tensor = attn_tensor / (head_dim ** 0.5)
            attn_tensor = F.softmax(attn_tensor, dim=-1)

            vertical_sum = attn_tensor.sum(dim=-2)
            vertical_sorted_attn = vertical_sum.sort(dim=-1, descending=False).values
            vertical_sorted_attn_desc = vertical_sum.sort(dim=-1, descending=True).values
            cum_vertical = vertical_sorted_attn.cumsum(dim=-1)
            cum_vertical_desc_np = vertical_sorted_attn_desc.cumsum(dim=-1).cpu().numpy()

            slash_sum = sum_all_diagonal_matrix(attn_tensor)[..., :-qlen + 1]
            slash_sorted_attn = slash_sum.sort(dim=-1, descending=False).values
            slash_sorted_attn_desc = slash_sum.sort(dim=-1, descending=True).values
            cum_slash = slash_sorted_attn.cumsum(dim=-1)
            print(cum_slash.shape)
            cum_slash_desc_np = slash_sorted_attn_desc.cumsum(dim=-1).cpu().numpy()

            for head_id in range(num_head_per_rank):
                actual_head_id = i * num_head_per_rank + head_id
                v_total = cum_vertical_desc_np[head_id][-1]
                vidx = np.searchsorted(cum_vertical_desc_np[head_id], v_total * recover_rate)
                s_total = cum_slash_desc_np[head_id][-1]
                sidx = np.searchsorted(cum_slash_desc_np[head_id], s_total * recover_rate)
                print(f"Layer {j} Head {actual_head_id} need {vidx} {vidx / klen * 100}% verticals & {sidx} {sidx / klen * 100}% slashes to recover")

                vdatas = cum_vertical[head_id, ...].cpu().numpy()
                sdatas = cum_slash[head_id, ...].cpu().numpy()
                assert len(vdatas == sdatas)
                x = np.arange(len(vdatas))
                vy = vdatas
                sy = sdatas

                plt.figure(figsize=(12, 6))
                plt.plot(x, vy, color='blue', linewidth=1.5, label='Vertical CumSum')
                plt.plot(x, sy, color='red', linewidth=1.5, label='Slash CumSum')

                plt.title(f"Layer_{j} Head_{actual_head_id} Vertical & Slash Cumulative Sum")
                plt.xlabel('Index')
                plt.ylabel('Value')
                plt.grid(True, linestyle='--', alpha=0.6)
                plt.legend()

                plt.tight_layout()
                print(f"save Layer {j} Head {actual_head_id} ...")
                plt.savefig(f"./minference_fig2/vertical_L{j}_H{actual_head_id}_{recover_rate}.jpg")
                plt.close()
            exit(0)