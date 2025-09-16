import triton
import triton.language as tl

@triton.jit
def triton_calc_blk_similarity(
    x_ptr,              # N * H * D
    x_loc_ptr,          # B + 1,
    output_pool_ptr,    # ceil(N, BLK_SIZE) * H * D
    output_map_size_ptr,  # ceil(N, BLK_SIZE) * H 值为 16或者
    PAGE_SIZE: tl.constexpr,     # page attention 的页面大小
    BLK_SIZE: tl.constexpr,      # 必须是 PAGE SIZE 的整倍数，目前先考虑 BLK_SIZE == PAGE_SIZE
    B: tl.constexpr,    # batch size
):
    pass