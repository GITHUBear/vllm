import torch
import torch.distributed as dist
import argparse
import time

def test_bandwidth():
    # 初始化分布式环境
    dist.init_process_group(backend='nccl')
    rank = dist.get_rank()
    world_size = dist.get_world_size()

    if world_size != 2:
        raise ValueError("该测试需要恰好 2 个进程，请使用 `--nproc_per_node=2` 启动")

    # 参数设置：数据大小和迭代次数
    size = 100 * 7 * 128 * 1  # 1GB 数据，默认 float32（4 字节/元素）
    iterations = 1

    # 分配 GPU 设备
    device = torch.device('cuda', rank)
    torch.cuda.set_device(device)

    # 创建发送和接收的张量
    send_tensor = torch.rand(size // 4, device=device, dtype=torch.bfloat16)  # float32，元素数 = 1GB / 4 字节
    recv_tensor = torch.zeros_like(send_tensor)

    # 预热（预热减少初始化延迟影响）
    for _ in range(5):
        if rank == 0:
            dist.send(send_tensor, dst=1)
            dist.recv(recv_tensor, src=1)
        else:
            dist.recv(recv_tensor, src=0)
            dist.send(send_tensor, dst=0)
    torch.cuda.synchronize()  # 确保预热完成

    # 开始时间测量
    start_time = time.time()
    for _ in range(iterations):
        if rank == 0:
            dist.send(send_tensor, dst=1)
            dist.recv(recv_tensor, src=1)
        else:
            dist.recv(recv_tensor, src=0)
            dist.send(send_tensor, dst=0)
    torch.cuda.synchronize()  # 确保所有操作完成
    end_time = time.time()

    # 计算带宽（双向通信，总数据量 = size * iterations * 2）
    total_data_bytes = size * iterations * 2
    time_seconds = end_time - start_time
    print(f"=========== cost: {time_seconds} =============")
    bandwidth_gb_s = (total_data_bytes / time_seconds) / (1024**3)

    if rank == 0:
        print(f"双向通信带宽: {bandwidth_gb_s:.2f} GB/s")

if __name__ == "__main__":
    test_bandwidth()
