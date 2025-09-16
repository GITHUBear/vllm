import h5py
import torch
import numpy as np
import matplotlib.pyplot as plt
import os

tp_size = 8
layer_num = 48
num_head_per_rank = 5
num_key_head_per_rank = 1
dirname = "./topk_dump/"
start_step = 4096
stride = 32

for i in range(tp_size):
    cur_step = start_step
    topk_per_layer = [[] for _ in range(layer_num)]
    hit_rate_per_layer = [[] for _ in range(layer_num)]
    while os.path.exists(dirname + f"tensor_{i}_{cur_step}.hdf5"):
        data_path = dirname + f"tensor_{i}_{cur_step}.hdf5"

        with h5py.File(data_path, 'r') as data:
            for j in range(layer_num):
                key = f"{j}"
                topk_per_head_list = (np.array(data[key]).tolist())[0][0]
                # topk_per_head_tensor = torch.from_numpy(topk_per_head).to(device="cuda:0")
                topk_per_layer[j].append(sorted(topk_per_head_list))
        cur_step += stride

    # print("finish read")
    for j in range(layer_num):
        for k in range(1, len(topk_per_layer[j]), 1):
            cur_topk = topk_per_layer[j][k]
            pre_topk = topk_per_layer[j][k - 1]
            pre_set = set(pre_topk)
            hit_cnt = 0
            for topk in cur_topk:
                if topk in pre_set:
                    hit_cnt += 1
            hit_rate = hit_cnt / len(cur_topk)
            hit_rate_per_layer[j].append(hit_rate)
            
    plt.figure(figsize=(12, 8))  # 设置图像大小

    x = list(range(len(hit_rate_per_layer[0])))

    for j in range(len(hit_rate_per_layer)):
        plt.plot(x, hit_rate_per_layer[j])

    # 添加标题和标签
    plt.title("Topk hit rate", fontsize=16)
    plt.xlabel("Step", fontsize=12)
    plt.ylabel("Hit rate", fontsize=12)

    # plt.legend().set_visible(False)  # 关闭图例，保持画面清晰

    # 显示网格（可选）
    plt.grid(True, alpha=0.3)

    # 展示图像
    plt.tight_layout()  # 自动调整布局
    plt.savefig(f"./topk_visual/head_{i}.png", dpi=150, bbox_inches='tight')