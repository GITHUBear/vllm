#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>

#define CHECK_DEVICE(x) TORCH_CHECK(x.device().type() == torch::kCUDA, #x " must be on CUDA")
#define CHECK_SHAPE(x, ...) TORCH_CHECK(x.sizes() == torch::IntArrayRef({__VA_ARGS__}), #x " must have shape (" #__VA_ARGS__ ")")
#define CHECK_CONTIGUOUS(x) TORCH_CHECK(x.is_contiguous(), #x " must be contiguous")

#define DISPATCH_HALF_AND_BF16(TYPE, NAME, ...)                            \
  if (TYPE == at::ScalarType::Half) {                                      \
    using scalar_t = at::Half;                                             \
    __VA_ARGS__();                                                         \
  } else if (TYPE == at::ScalarType::BFloat16) {                           \
    using scalar_t = at::BFloat16;                                         \
    __VA_ARGS__();                                                         \
  } else {                                                                 \
    AT_ERROR(#NAME, " not implemented for type '", toString(TYPE), "'"); \
  }

#define DISPATCH_WITH_HEAD_DIM(HEAD_DIM_NAME, NAME, ...)                        \
    if (HEAD_DIM_NAME == 32) {                                           \
        constexpr int HEAD_DIM = 32;                                       \
        __VA_ARGS__();                                                     \
    } else if (HEAD_DIM_NAME == 64) {                                           \
        constexpr int HEAD_DIM = 64;                                       \
        __VA_ARGS__();                                                     \
    } else if (HEAD_DIM_NAME == 128) {                                          \
        constexpr int HEAD_DIM = 128;                                      \
        __VA_ARGS__();                                                     \
    } else if (HEAD_DIM_NAME == 256) {                                          \
        constexpr int HEAD_DIM = 256;                                      \
        __VA_ARGS__();                                                     \
    } else {                                                                \
        AT_ERROR(#NAME, " not implemented head dim"); \
    }

template<typename T>
struct SATypeConverter {
    using Type = T;
};

template<>
struct SATypeConverter<at::Half> {
    using Type = uint16_t;
};

template<>
struct SATypeConverter<at::BFloat16> {
    using Type = __nv_bfloat16;
};

/////////////////////////////// Q_VEC transfer type definition /////////////////////////
template <typename T, int Dh_MAX>
struct Q_VEC_TRANSFER
{
};

template <>
struct Q_VEC_TRANSFER<uint16_t, 32>
{
    using Type = uint32_t;
};

template <>
struct Q_VEC_TRANSFER<uint16_t, 64>
{
    using Type = uint32_t;
};

template <>
struct Q_VEC_TRANSFER<uint16_t, 128>
{
    using Type = uint2;
};

template <>
struct Q_VEC_TRANSFER<uint16_t, 256>
{
    using Type = uint4;
};

template <>
struct Q_VEC_TRANSFER<__nv_bfloat16, 32>
{
    using Type = uint32_t;
};

template <>
struct Q_VEC_TRANSFER<__nv_bfloat16, 64>
{
    using Type = uint32_t;
};

template <>
struct Q_VEC_TRANSFER<__nv_bfloat16, 128>
{
    using Type = uint2;
};

template <>
struct Q_VEC_TRANSFER<__nv_bfloat16, 256>
{
    using Type = uint4;
};
///////////////////////////////////////////////////////////////////

/////////////////////////////// CALC_VEC type definition /////////////////////////
template <typename T, int VEC_SIZE>
struct CALC_VEC
{
};

template <>
struct CALC_VEC<uint16_t, 2>
{
    using Type = uint32_t;
};

template <>
struct CALC_VEC<uint16_t, 4>
{
    using Type = uint2;
};

template <>
struct CALC_VEC<uint16_t, 8>
{
    using Type = uint4;
};

template <>
struct CALC_VEC<__nv_bfloat16, 2>
{
    using Type = uint32_t;
};

template <>
struct CALC_VEC<__nv_bfloat16, 4>
{
    using Type = uint2;
};

template <>
struct CALC_VEC<__nv_bfloat16, 8>
{
    using Type = uint4;
};
///////////////////////////////////////////////////////////////////

/////////////////////////////// QkDotMinMaxTypeConverter /////////////////////////
template <typename T, typename VecT>
struct QkDotMinMaxTypeConverter
{
};

template <>
struct QkDotMinMaxTypeConverter<uint16_t, uint4>
{
    using Type = half2;
};

template <>
struct QkDotMinMaxTypeConverter<__nv_bfloat16, uint4>
{
    using Type = __nv_bfloat162;
};
///////////////////////////////////////////////////////////////////

template <typename T>
inline __device__ __host__ constexpr unsigned threads_per_value(unsigned dh)
{
    return dh * sizeof(T) / 16;
}

template <int THREAD_PER_KEY>
inline __device__ float qk_hmma_dot_min_max(const half2* q, const half2* k_max, const half2* k_min)
{
    half2 acc_max = __hmul2(q[0], k_max[0]);
    half2 acc_min = __hmul2(q[0], k_min[0]);
    half2 acc = __hmax2(acc_max, acc_min);
#pragma unroll
    for (int ii = 1; ii < 4; ++ii)
    {
        acc_max = __hmul2(q[ii], k_max[ii]);
        acc_min = __hmul2(q[ii], k_min[ii]);
        acc = __hadd2(acc, __hmax2(acc_max, acc_min));
    }
    float qk_min_max = __half2float(__hadd(acc.x, acc.y));

#pragma unroll
    for (int mask = THREAD_PER_KEY / 2; mask >= 1; mask /= 2)
    {
        qk_min_max += __shfl_xor_sync(uint32_t(-1), qk_min_max, mask);
    }

    return qk_min_max;
}

template <int THREAD_PER_KEY>
inline __device__ float qk_hmma_dot_min_max(const __nv_bfloat162* q, const __nv_bfloat162* k_max, const __nv_bfloat162* k_min)
{
    __nv_bfloat162 acc_max = __hmul2(q[0], k_max[0]);
    __nv_bfloat162 acc_min = __hmul2(q[0], k_min[0]);
    __nv_bfloat162 acc = __hmax2(acc_max, acc_min);
#pragma unroll
    for (int ii = 1; ii < 4; ++ii)
    {
        acc_max = __hmul2(q[ii], k_max[ii]);
        acc_min = __hmul2(q[ii], k_min[ii]);
        acc = __hadd2(acc, __hmax2(acc_max, acc_min));
    }
    float qk_min_max = __bfloat162float(__hadd(acc.x, acc.y));

#pragma unroll
    for (int mask = THREAD_PER_KEY / 2; mask >= 1; mask /= 2)
    {
        qk_min_max += __shfl_xor_sync(uint32_t(-1), qk_min_max, mask);
    }

    return qk_min_max;
}

template <
    typename T, 
    int HEAD_DIM, 
    unsigned THREADS_PER_BLOCK = 256,
    unsigned THREAD_PER_KEY = threads_per_value<T>(HEAD_DIM),
    unsigned META_CACHE_BLOCKS_PER_THREAD_BLOCK = 64>  // 目前设置为 64
__global__ void lserve_page_selector_kernel(
    const T* q,
    const T* key_meta_cache,
    const int* block_table,
    const int* num_full_blocks,
    T* out,
    const int num_q_head,
    const int num_kv_head,
    const int max_block_size,

    const int64_t qstride0, const int64_t qstride1,
    const int64_t kmc_stride0, const int64_t kmc_stride1, const int64_t kmc_stride2,
    const int64_t out_stride0, const int64_t out_stride1
) {
    const auto tid = threadIdx.x;
    const auto batch_id = blockIdx.x;
    const auto qhead_id = blockIdx.y;
    const auto block_tile_id = blockIdx.z;
    const auto head_group_size = num_q_head / num_kv_head;
    const auto khead_id = qhead_id / head_group_size;

    // 0.0 判断现在要处理的 META_CACHE_BLOCKS_PER_THREAD_BLOCK 个 meta cache block 是否超过了 num_full_blocks
    // 如果超过则直接return
    const unsigned logical_meta_cache_blocks_id = META_CACHE_BLOCKS_PER_THREAD_BLOCK * block_tile_id;
    int num_max_blocks = num_full_blocks[batch_id];
    if (logical_meta_cache_blocks_id >= num_max_blocks) {
        return;
    }

    // 0.1 读取 block table 到 smem
    static_assert(THREADS_PER_BLOCK >= META_CACHE_BLOCKS_PER_THREAD_BLOCK);
    __shared__ __align__(sizeof(int)) int BLOCK_TABLE_FOR_THREAD_BLOCK[META_CACHE_BLOCKS_PER_THREAD_BLOCK];
    const int* block_table_gmem_ptr = block_table + batch_id * max_block_size + logical_meta_cache_blocks_id;
    if (logical_meta_cache_blocks_id + tid < num_max_blocks) {
        BLOCK_TABLE_FOR_THREAD_BLOCK[tid] = *(block_table_gmem_ptr + tid);
    } else if (tid < META_CACHE_BLOCKS_PER_THREAD_BLOCK) {
        // 设置为 0 号 block，以免后续引入复杂的边界检查
        BLOCK_TABLE_FOR_THREAD_BLOCK[tid] = 0;
    }
    // __syncthreads();

    using q_vec = typename Q_VEC_TRANSFER<T, HEAD_DIM>::Type;
    constexpr unsigned Q_VEC_SIZE = sizeof(q_vec) / sizeof(T);
    const auto qvec_offset = tid * Q_VEC_SIZE;
    const T* query_gmem_ptr = q + qstride0 * batch_id + qstride1 * qhead_id;
    __shared__ __align__(sizeof(q_vec)) T q_smem[HEAD_DIM];
    // 1. 装载 query 到 smem
    if (qvec_offset < HEAD_DIM) {
        *reinterpret_cast<q_vec*>(&(q_smem[qvec_offset])) = *reinterpret_cast<const q_vec*>(query_gmem_ptr + qvec_offset);
    }
    __syncthreads();

    // 2. 从 smem 装载到寄存器
    constexpr auto CALC_VEC_SIZE = 16u / sizeof(T);
    constexpr auto ELEMENTS_PER_CHUNK = THREAD_PER_KEY * CALC_VEC_SIZE;
    // 目前的检查下可以保证刚好可以处理一整个 HEAD_DIM
    static_assert(ELEMENTS_PER_CHUNK == HEAD_DIM);
    // 每个线程的 CALC_VEC 寄存器
    using cvec = typename CALC_VEC<T, CALC_VEC_SIZE>::Type;
    cvec calc_qvec;
    // 每个线程只需要独立处理 query 的 CALC_VEC_SIZE 大小即可
    const auto calc_qvec_offset = (tid % THREAD_PER_KEY) * CALC_VEC_SIZE;
    calc_qvec = *reinterpret_cast<cvec*>(&(q_smem[calc_qvec_offset]));

    // constexpr unsigned WARP_SIZE = 32;
    constexpr unsigned NUM_META_BLOCK_PER_THREAD_BLOCK = THREADS_PER_BLOCK / THREAD_PER_KEY;
    // constexpr unsigned NUM_META_BLOCK_PER_WARP = WARP_SIZE / THREAD_PER_KEY;
    static_assert(META_CACHE_BLOCKS_PER_THREAD_BLOCK % NUM_META_BLOCK_PER_THREAD_BLOCK == 0);
    constexpr auto ITERS = META_CACHE_BLOCKS_PER_THREAD_BLOCK / NUM_META_BLOCK_PER_THREAD_BLOCK;
    // 3. 从 gmem 中向量化 load 每个线程需要的数据
    cvec calc_max_kvec[ITERS];
    cvec calc_min_kvec[ITERS];
#pragma unroll
    for (int block_table_idx = tid / THREAD_PER_KEY, iter = 0; 
        block_table_idx < META_CACHE_BLOCKS_PER_THREAD_BLOCK; 
        block_table_idx += NUM_META_BLOCK_PER_THREAD_BLOCK, ++iter) 
    {
        int block_id = BLOCK_TABLE_FOR_THREAD_BLOCK[block_table_idx];
        const T* key_max_cache_gmem_ptr = key_meta_cache + block_id * kmc_stride0 + khead_id * kmc_stride0;
        const T* key_min_cache_gmem_ptr = key_max_cache_gmem_ptr + kmc_stride2;
        calc_max_kvec[iter] = *reinterpret_cast<const cvec*>(key_max_cache_gmem_ptr + calc_qvec_offset);
        calc_min_kvec[iter] = *reinterpret_cast<const cvec*>(key_min_cache_gmem_ptr + calc_qvec_offset);
    }
    __syncthreads();

    // 4. 每个线程计算局部向量点积，再reduce求和
    using qk_type = typename QkDotMinMaxTypeConverter<T, cvec>::Type;
    T* out_gmem_ptr = out + batch_id * out_stride0 + qhead_id * out_stride1;
    unsigned logical_block_id = logical_meta_cache_blocks_id + (tid / THREAD_PER_KEY);
#pragma unroll
    for (int iter = 0; iter < ITERS && logical_block_id < num_max_blocks;
         ++iter, logical_block_id += NUM_META_BLOCK_PER_THREAD_BLOCK) {
        qk_type* q = reinterpret_cast<qk_type*>(&(calc_qvec));
        qk_type* k_max = reinterpret_cast<qk_type*>(&(calc_max_kvec[iter]));
        qk_type* k_min = reinterpret_cast<qk_type*>(&(calc_min_kvec[iter]));
        float qk_min_max = qk_hmma_dot_min_max<THREAD_PER_KEY>(q, k_max, k_min);

        // 5. 结果写回 out
        if (calc_qvec_offset == 0) {
            if constexpr(std::is_same<T, uint16_t>::value) {
                *(out_gmem_ptr + logical_block_id) = __float2half(qk_min_max);
            } else {
                *(out_gmem_ptr + logical_block_id) = __float2bfloat16(qk_min_max);
            }
        }
    }
}

template<typename T, int HEAD_DIM>
void lserve_page_selector_kernel_launch(
    const T* q,
    const T* key_meta_cache,
    const int* block_table,
    const int* num_full_blocks,
    T* out,
    const int batch_size,
    const int num_q_head,
    const int num_kv_head,
    const int head_dim,
    const int max_block_size,

    const int64_t qstride0, const int64_t qstride1
) {
    // auto constexpr threads_per_value = threads_per_value<T>(HEAD_DIM);
    constexpr unsigned META_CACHE_BLOCKS_PER_THREAD_BLOCK = 64;
    const unsigned num_block_tiles = (max_block_size + META_CACHE_BLOCKS_PER_THREAD_BLOCK - 1) / META_CACHE_BLOCKS_PER_THREAD_BLOCK;
    const int64_t kmc_stride0 = ((int64_t)1) * num_kv_head * 2 * head_dim;
    const int64_t kmc_stride1 = ((int64_t)1) * 2 * head_dim;
    const int64_t kmc_stride2 = ((int64_t)1) * head_dim;
    const int64_t out_stride0 = ((int64_t)1) * num_q_head * max_block_size;
    const int64_t out_stride1 = ((int64_t)1) * max_block_size;
    dim3 grid(batch_size, num_q_head, num_block_tiles);
    dim3 block(256);
    lserve_page_selector_kernel<T, HEAD_DIM><<<grid, block>>>(
        q, key_meta_cache, block_table,
        num_full_blocks, out,
        num_q_head,
        num_kv_head,
        max_block_size,
        qstride0, qstride1, 
        kmc_stride0, kmc_stride1, kmc_stride2,
        out_stride0, out_stride1
    );
}

template<typename T>
void lserve_page_selector_head_dim_dispatcher(
    const T* q,
    const T* key_meta_cache,
    const int* block_table,
    const int* num_full_blocks,
    T* out,
    const int batch_size,
    const int num_q_head,
    const int num_kv_head,
    const int head_dim,
    const int max_block_size,

    const int64_t qstride0, const int64_t qstride1
) {
    DISPATCH_WITH_HEAD_DIM(head_dim, "lserve_page_selector_head_dim_dispatcher", [&] {
        lserve_page_selector_kernel_launch<T, HEAD_DIM>(
            q, key_meta_cache, block_table,
            num_full_blocks, out,
            batch_size, 
            num_q_head,
            num_kv_head,
            max_block_size,
            head_dim,
            qstride0, qstride1
        );
    });
}


void lserve_page_selector(
    const torch::Tensor& q,                   // [batch_size, num_q_head, head_dim]. Only for decode. bf16
    const torch::Tensor& key_meta_cache,      // [num_block, num_kv_head, 2, head_dim] bf16
    const torch::Tensor& block_table,         // [batch_size, max_block_size] int32
    const torch::Tensor& num_full_blocks,     // [batch_size, ] int32
    torch::Tensor& out                        // [batch_size, num_q_head, max_block_size] bf16
) {
    CHECK_DEVICE(q); CHECK_DEVICE(key_meta_cache); 
    CHECK_DEVICE(block_table); CHECK_DEVICE(num_full_blocks);
    CHECK_DEVICE(out);

    // q 不是连续的，需要通过 stride 寻址
    // key_meta_cache 每层连续
    CHECK_CONTIGUOUS(key_meta_cache); CHECK_CONTIGUOUS(block_table);
    CHECK_CONTIGUOUS(num_full_blocks); CHECK_CONTIGUOUS(out);
    TORCH_CHECK(q.stride(-1) == 1);

    TORCH_CHECK(block_table.dtype() == torch::kInt32);
    TORCH_CHECK(num_full_blocks.dtype() == torch::kInt32);

    int batch_size = q.size(0);
    int num_q_head = q.size(1);
    int head_dim = q.size(-1);
    int num_kv_head = key_meta_cache.size(0);
    TORCH_CHECK(num_q_head % num_kv_head == 0);

    int max_block_size = block_table.size(-1);
    TORCH_CHECK(out.size(-1) == max_block_size);
    // 1. 目前先支持 fp16/bf16 类型
    // 2. 每个线程处理 16B 数据, 8个fp16, 同时希望一个 query key 能够被一个 warp 处理, 那么 head_dim 最长只能是 32 * 8 = 256
    //    其次，希望一个 warp 能够处理整数个 qk， 所以需要是 256 的因数
    //    所以限制 head_dim 需要是 16/32/64/128/256，且额外要求大于等于32
    TORCH_CHECK(
        head_dim == 32 ||
        head_dim == 64 || head_dim == 128 || 
        head_dim == 256,
        "query head_dim must be in {16, 32, 64, 128, 256}"
    );
    TORCH_CHECK(
        block_table.size(0) == batch_size &&
        num_full_blocks.size(0) == batch_size &&
        out.size(0) == batch_size,
        "must have the same batch_size"
    );
    TORCH_CHECK(
        key_meta_cache.size(-1) == head_dim,
        "key_meta_cache must have the same head_dim with q"
    );

    const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    DISPATCH_HALF_AND_BF16(q.scalar_type(), "lserve_page_selector", [&] {
        using DataType = typename SATypeConverter<scalar_t>::Type;

        lserve_page_selector_head_dim_dispatcher<DataType>(
            reinterpret_cast<DataType*>(q.data_ptr()),
            reinterpret_cast<DataType*>(key_meta_cache.data_ptr()),
            block_table.data_ptr<int>(),
            num_full_blocks.data_ptr<int>(),
            reinterpret_cast<DataType*>(out.data_ptr()),
            batch_size,
            num_q_head,
            num_kv_head,
            head_dim,
            max_block_size,
            q.stride(0),
            q.stride(1)
        );
    });
}