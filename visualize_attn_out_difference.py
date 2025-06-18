import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt

tp_size = 4
layer_num = 28
num_head_per_rank = 7
full_attn_dirname = "./attn_out_dump/"
xattn_dirname = "./xattn_out_dump/"
xattn_chunked_dirname = "./xattn_chunked_out_dump/"

for i in range(tp_size):
    file_name = f"tensor_{i}.hdf5"
    full_attn_dataname = full_attn_dirname + file_name
    xattn_chunked_dataname = xattn_chunked_dirname + file_name
    with h5py.File(full_attn_dataname, 'r') as f_full, h5py.File(xattn_chunked_dataname, 'r') as f_xattn_chunked:
        layer_dict = {}
        for key in f_xattn_chunked.keys():
            key_splits = key.split('_')
            layer_id = int(key_splits[0])
            if layer_id not in layer_dict:
                layer_dict[layer_id] = [key]
            else:
                layer_dict[layer_id].append(key)

        for j in range(layer_num):
            key = f"{j}"
            full_attn_np = np.array(f_full[key])

            xattn_chunks = []
            for chunk_key in layer_dict[j]:
                xattn_chunks.append(
                    torch.from_numpy(np.array(f_xattn_chunked[chunk_key])).to(device="cuda:0")
                )
            
            full_attn_out = torch.from_numpy(full_attn_np).to(device="cuda:0")
            xattn_chunk_out = torch.concat(xattn_chunks, dim=0)
            assert full_attn_out.shape == xattn_chunk_out.shape
            square_diff = (xattn_chunk_out - full_attn_out)**2
            diff_sum = square_diff.sum(dim=-1)

            for head_id in range(num_head_per_rank):
                actual_head_id = i * num_head_per_rank + head_id
                datas = diff_sum[:, head_id].cpu().numpy()
                sample_rate = 200
                sampled_data = datas[::sample_rate]

                # 绘图
                x = np.arange(len(sampled_data))
                plt.figure(figsize=(16, 6))
                plt.bar(x, sampled_data, width=1.0, color='red', edgecolor='none')
                plt.title(f"Attention output diff on layer-{j} head-{actual_head_id}")
                plt.xlabel("Index (every 200th point)")
                plt.ylabel("Diff")
                plt.tight_layout()
                print(f"save Layer {j} Head {actual_head_id} ...")
                plt.savefig(f"./diff_chunk_fig/L{j}_H{actual_head_id}.jpg")
                plt.close()




# for i in range(tp_size):
#     file_name = f"tensor_{i}.hdf5"
#     full_attn_dataname = full_attn_dirname + file_name
#     xattn_dataname = xattn_dirname + file_name
#     with h5py.File(full_attn_dataname, 'r') as f_full, h5py.File(xattn_dataname, 'r') as f_xattn:
#         for j in range(layer_num):
#             key = f"{j}"
#             full_attn_np = np.array(f_full[key])
#             xattn_np = np.array(f_xattn[key])

#             full_attn_out = torch.from_numpy(full_attn_np).to(device="cuda:0")
#             xattn_out = torch.from_numpy(xattn_np).to(device="cuda:0")
#             square_diff = (xattn_out - full_attn_out)**2
#             diff_sum = square_diff.sum(dim=-1)

#             for head_id in range(num_head_per_rank):
#                 actual_head_id = i * num_head_per_rank + head_id
#                 datas = diff_sum[:, head_id].cpu().numpy()
#                 sample_rate = 200
#                 sampled_data = datas[::sample_rate]

#                 # 绘图
#                 x = np.arange(len(sampled_data))
#                 plt.figure(figsize=(16, 6))
#                 plt.bar(x, sampled_data, width=1.0, color='skyblue', edgecolor='none')
#                 plt.title(f"Attention output diff on layer-{j} head-{actual_head_id}")
#                 plt.xlabel("Index (every 200th point)")
#                 plt.ylabel("Diff")
#                 plt.tight_layout()
#                 print(f"save Layer {j} Head {actual_head_id} ...")
#                 plt.savefig(f"./diff_fig/L{j}_H{actual_head_id}.jpg")
#                 plt.close()


