#include <torch/all.h>
#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <cute/tensor.hpp>
#include "cutlass/cutlass.h"
#include "cutlass/layout/layout.h"
#include <cutlass/numeric_types.h>

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
    using Type = __half;
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
struct Q_VEC_TRANSFER<__half, 32>
{
    using Type = uint32_t;
};

template <>
struct Q_VEC_TRANSFER<__half, 64>
{
    using Type = uint32_t;
};

template <>
struct Q_VEC_TRANSFER<__half, 128>
{
    using Type = uint2;
};

template <>
struct Q_VEC_TRANSFER<__half, 256>
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
struct CALC_VEC<__half, 2>
{
    using Type = uint32_t;
};

template <>
struct CALC_VEC<__half, 4>
{
    using Type = uint2;
};

template <>
struct CALC_VEC<__half, 8>
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
struct QkDotMinMaxTypeConverter<__half, uint4>
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
inline __device__ __half qk_hmma_dot_min_max(const half2* q, const half2* k_max, const half2* k_min)
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
    float acc_for_shuffle = *(reinterpret_cast<float*>(&(acc)));
#pragma unroll
    for (int mask = THREAD_PER_KEY / 2; mask >= 1; mask /= 2) 
    {
        acc_for_shuffle = __shfl_xor_sync(uint32_t(-1), acc_for_shuffle, mask);
        acc = __hadd2(acc, *(reinterpret_cast<half2*>(&(acc_for_shuffle))));
        acc_for_shuffle = *(reinterpret_cast<float*>(&(acc)));
    }
    return __hadd(acc.x, acc.y);

//     float qk_min_max = __half2float(__hadd(acc.x, acc.y));

// #pragma unroll
//     for (int mask = THREAD_PER_KEY / 2; mask >= 1; mask /= 2)
//     {
//         qk_min_max += __shfl_xor_sync(uint32_t(-1), qk_min_max, mask);
//     }

//     return qk_min_max;
}

template <int THREAD_PER_KEY>
inline __device__ __nv_bfloat16 qk_hmma_dot_min_max(const __nv_bfloat162* q, const __nv_bfloat162* k_max, const __nv_bfloat162* k_min)
{
    static_assert(sizeof(float) == sizeof(__nv_bfloat162));
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
    float acc_for_shuffle = *(reinterpret_cast<float*>(&(acc)));
#pragma unroll
    for (int mask = THREAD_PER_KEY / 2; mask >= 1; mask /= 2) 
    {
        acc_for_shuffle = __shfl_xor_sync(uint32_t(-1), acc_for_shuffle, mask);
        acc = __hadd2(acc, *(reinterpret_cast<__nv_bfloat162*>(&(acc_for_shuffle))));
        acc_for_shuffle = *(reinterpret_cast<float*>(&(acc)));
    }
    return __hadd(acc.x, acc.y);
//     float qk_min_max = __bfloat162float(__hadd(acc.x, acc.y));

// #pragma unroll
//     for (int mask = THREAD_PER_KEY / 2; mask >= 1; mask /= 2)
//     {
//         qk_min_max += __shfl_xor_sync(uint32_t(-1), qk_min_max, mask);
//     }

//     return qk_min_max;
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
    if (tid < META_CACHE_BLOCKS_PER_THREAD_BLOCK) {
        BLOCK_TABLE_FOR_THREAD_BLOCK[tid] = (logical_meta_cache_blocks_id + tid < num_max_blocks) ? 
                                            *(block_table_gmem_ptr + tid) : 0;
    }
    // __syncthreads();

    // q_vec 能够保证是一个 WARP 处理装载, 不出现 WARP DIVERGENCE
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
        const T* key_max_cache_gmem_ptr = key_meta_cache + block_id * kmc_stride0 + khead_id * kmc_stride1;
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
        auto qk_min_max = qk_hmma_dot_min_max<THREAD_PER_KEY>(q, k_max, k_min);

        // 5. 结果写回 out
        if (calc_qvec_offset == 0) {
            *(out_gmem_ptr + logical_block_id) = qk_min_max;
            // if constexpr(std::is_same<T, uint16_t>::value) {
            //     *(out_gmem_ptr + logical_block_id) = qk_min_max;
            // } else {
            //     *(out_gmem_ptr + logical_block_id) = qk_min_max;
            // }
        }
    }
}

template<int kNWarps_, int kHeadDim_, int kBlockQ_, int kBlockK_, typename elem_type=__nv_bfloat16, typename output_type=float>
struct Kernel_Traits {
    using Element = elem_type;
    using ElementO = output_type;

    static constexpr int kNWarps = kNWarps_;
    static constexpr int kHeadDim = kHeadDim_;
    static constexpr int kBlockQ = kBlockQ_;
    static constexpr int kBlockK = kBlockK_;
    static constexpr int kNThreads = kNWarps_ * 32;
    static constexpr int kGmemElemsPerLoad = sizeof(cute::uint128_t) / sizeof(Element);
    static constexpr int kBlockKSmem = kHeadDim % 64 == 0 ? 64 : 32;
    static constexpr int kGmemThreadsPerRow = kBlockKSmem / kGmemElemsPerLoad;
    static constexpr int kSwizzle = kBlockKSmem == 32 ? 2 : 3;
    static constexpr int kGmemRowsPerThread = kBlockK / (kNThreads / kGmemThreadsPerRow);

    // TODO[shk]:支持不同架构使用不同的 atom op
    using Gmem_copy_struct = SM80_CP_ASYNC_CACHEGLOBAL<cute::uint128_t>;
    using SmemCopyAtom = Copy_Atom<SM75_U32x4_LDSM_N, Element>;
    using SmemCopyAtomO = Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementO>;
    using MMA_Atom_Arch = MMA_Atom<SM80_16x8x16_F32F16F16F32_TN>;

    // gmem 拷贝的 thread 布局 16 * 8
    using GmemLayoutAtom = Layout<Shape <Int<kNThreads / kGmemThreadsPerRow>, Int<kGmemThreadsPerRow>>,
                                  Stride<Int<kGmemThreadsPerRow>, _1>>;
    using GmemTiledCopy = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));
    // 
    using GmemTiledCopyPaged = decltype(
        make_tiled_copy(Copy_Atom<Gmem_copy_struct, Element>{},
                        GmemLayoutAtom{},
                        Layout<Shape<Int<kGmemRowsPerThread>, _8>, Stride<_8, _1>>));
    using GmemTiledCopyO = decltype(
        make_tiled_copy(Copy_Atom<AutoVectorizingCopyWithAssumedAlignment<128>, ElementO>{},
                        GmemLayoutAtom{},
                        Layout<Shape<_1, _8>>{}));

    using SmemLayoutQ = decltype(
        tile_to_shape(
            SmemLayoutAtom{},
            Shape<Int<kBlockQ>, Int<kHeadDim>>{}
        )
    );
    using SmemLayoutK = decltype(
        tile_to_shape(
            SmemLayoutAtom{},
            Shape<Int<kBlockK>, Int<kHeadDim>>{}
        )
    );
    using SmemLayoutO = decltype(
        tile_to_shape(
            SmemLayoutAtom{},
            Shape<Int<kBlockQ>, Int<kBlockK / 2>>{}
        )
    );

    using TiledMma = TiledMMA<
        MMA_Atom_Arch,
        Layout<Shape<Int<kNWarps>,_1,_1>>,  // 4x1x1 or 8x1x1 thread group
        Tile<Int<16 * kNWarps>, _16, _16>>;
};

template <class Shape, class Stride>
__forceinline__ __device__
auto reshape_thread_tile(Layout<Shape, Stride> l) {
    return make_layout(append(get<0>(l.shape()), get<2>(l.shape())),
                        append(get<0>(l.stride()), get<2>(l.stride())));
}

template <bool Clear_OOB_MN=false,
          typename TiledCopy, typename Engine0, typename Layout0, typename Engine1, typename Layout1,
          typename Engine2, typename Layout2>
__forceinline__ __device__ void identity_copy(TiledCopy tiled_copy, Tensor<Engine0, Layout0> const &S,
                            Tensor<Engine1, Layout1> &D, Tensor<Engine2, Layout2> const &identity_MN,
                            const int max_MN=0) {
    // max_MN 即当前块内最大的 token offset
    CUTE_STATIC_ASSERT_V(rank(S) == Int<3>{});
    CUTE_STATIC_ASSERT_V(rank(D) == Int<3>{});
    CUTE_STATIC_ASSERT_V(size<0>(S) == size<0>(D));      
    CUTE_STATIC_ASSERT_V(size<1>(S) == size<1>(D));
    CUTE_STATIC_ASSERT_V(size<2>(S) == size<2>(D));
    #pragma unroll
    // 在 token 维度上迭代
    for (int m = 0; m < size<1>(S); ++m) {
        if (get<0>(identity_MN(0, m, 0)) < max_MN) {
            // 仅在 token 没有超过最大 token offset 时
            #pragma unroll
            for (int k = 0; k < size<2>(S); ++k) {
                cute::copy(tiled_copy, S(_, m, k), D(_, m, k));
            }
        } else if (Clear_OOB_MN) {
            cute::clear(D(_, m, _));
        }
    }
}

template <typename Kernel_Traits>
__global__ void lserve_page_selector_kernel_v2(
    const T* q,                     // [batch_size, seqlen_q, num_head, head_dim]
    const T* key_meta_cache,        // [num_block, num_head, 2, head_dim]
    const int* block_table,         // [batch_size, max_block_size]
    const int* num_full_blocks,     // [batch_size, ]
    T* out,                         // [batch_size, num_head, seqlen_q, max_block_size]
    const int batch_size,
    const int seqlen_q,
    const int num_head,
    const int max_block_size,

    const int64_t qstride0, const int64_t qstride1, const int64_t qstride2,
    const int64_t kmc_stride0, const int64_t kmc_stride1, const int64_t kmc_stride2,
    const int64_t out_stride0, const int64_t out_stride1, const int64_t out_stride2
) {
    using Element = Kernel_Traits::Element;
    using ElementO = Kernel_Traits::ElementO;
    constexpr int kBlockQ = Kernel_Traits::kBlockQ;
    constexpr int kBlockK = Kernel_Traits::kBlockK;
    constexpr int kBlockO = kBlockK >> 1;
    constexpr int kHeadDim = Kernel_Traits::kHeadDim;
    constexpr int kGmemThreadsPerRow = Kernel_traits::kGmemThreadsPerRow; // 每行 8 个 thread
    constexpr int kGmemRowsPerThread = Kernel_traits::kGmemRowsPerThread; // 每个 thread 处理 8 个 row
    static_assert(kGmemRowsPerThread % 2 == 0);
    constexpr int kGmemElemsPerLoad = Kernel_traits::kGmemElemsPerLoad; // 8

    const auto tidx = threadIdx.x;
    const auto bidx = blockIdx.x;
    const auto hidx = blockIdx.y;
    const auto block_tile_idx = blockIdx.z;
    const auto global_row_idx = block_tile_idx * kBlockK;
    const auto global_page_idx = global_row_idx >> 1;
    const int num_max_pages = num_full_blocks[bidx];
    const int64_t col_offset = tidx % kGmemThreadsPerRow * kGmemElemsPerLoad;  // 处理数据的行内偏移 （head_dim维度）
    const int* block_table_cur_batch = block_table + bidx * max_block_size;

    if (global_page_idx >= num_max_pages) {
        return;
    }

    cute::Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(q) + bidx * qstride0),
                            make_shape(seqlen_q, num_head, kHeadDim),
                            make_stride(qstride1, qstride2, _1{}));
    // 因为 seqlen_q <= kBlockQ, 所以 local_tile 只会有一个分块，坐标取 (0, 0)
    cute::Tensor gQ = local_tile(mQ(_, hidx, _), Shape<Int<kBlockQ>, Int<kHeadDim>>{}, 
                           make_coord(0, 0));
    // 同一个 key head 下面一个 page 中参与计算的只有1个，所以这里stride用 kmc_stride2 即可
    cute::Tensor gK = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(key_meta_cache) + hidx * kmc_stride1),
                            Shape<Int<kBlockK>, Int<kHeadDim>>{},
                            make_stride(kmc_stride2, _1{}));
    cute::Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<ElementO*>(out) + bidx * out_stride0),
                            make_shape(num_head, seqlen_q, max_block_size),
                            make_stride(out_stride1, out_stride2, _1{}));
    cute::Tensor gO = local_tile(mO(hidx, _, _), Shape<Int<kBlockQ, Int<kBlockO>>>{},
                           make_coord(0, block_tile_idx));
    
    static_assert((int64_t)(kBlockQ + kBlockK) * kHeadDim * sizeof(Element) >= (int64_t)kBlockQ * kBlockO * sizeof(ElementO));
    __shared__ __align__(sizeof(__uint128_t)) char smem_[(int64_t)(kBlockQ + kBlockK) * kHeadDim * sizeof(Element)];
    cute::Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(smem_)),
                            typename Kernel_Traits::SmemLayoutQ{});
    cute::Tensor sK = make_tensor(make_smem_ptr(sQ.data() + size(sQ)),
                            typename Kernel_Traits::SmemLayoutK{});
    cute::Tensor sO = make_tensor(make_smem_ptr(reinterpret_cast<Element*>(smem_)),
                            typename Kernel_Traits::SmemLayoutO{});
    
    typename Kernel_Traits::GmemTiledCopy gmem_tiled_copy_Q;
    auto thr_gmem_tiled_copy_Q = gmem_tiled_copy_Q.get_thread_slice(tidx);
    typename Kernel_Traits::GmemTiledCopyPaged gmem_tiled_copy_K;
    auto thr_gmem_tiled_copy_K = gmem_tiled_copy_K.get_thread_slice(tidx);

    cute::Tensor tQgQ = thr_gmem_tiled_copy_Q.partition_S(gQ);
    cute::Tensor tQsQ = thr_gmem_tiled_copy_Q.partition_D(sQ);

    cute::Tensor tKgK_ = thr_gmem_tiled_copy_K.partition_S(gK);
    cute::Tensor tKsK_ = thr_gmem_tiled_copy_K.partition_D(sK);
    cute::Tensor tKgK = make_tensor(tKgK_.data(), reshape_thread_tile(tKgK_.layout()));
    cute::Tensor tKsK = make_tensor(tKsK_.data(), reshape_thread_tile(tKsK_.layout()));

    // tQgQ 根据 seqlen_q 来进行 load -> tQsQ
    cute::Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
    cute::Tensor tQcQ = thr_gmem_tiled_copy_Q.partition_S(cQ);
    identity_copy(gmem_tiled_copy_Q, tQgQ, tQsQ, tQcQ, seqlen_q);
    
    cute::Tensor cK = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));
    cute::Tensor tKcK_ = thr_gmem_tiled_copy_K.partition_S(cK);
    cute::Tensor tKcK = make_tensor(tKcK_.data(), reshape_thread_tile(tKcK_.layout()));
    // block_tile_idx * kBlockO 当前 K tile 的 logical page 起始id
    // block_tile_idx * kBlockO + m / 2
    #pragma unroll
    for (int m = 0; m < size<1>(tKgK); ++m) {
        auto block_row_idx = get<0>(tKcK(0, m, 0));
        auto block_page_idx = block_row_idx >> 1;
        auto logical_page_idx = global_page_idx + block_page_idx;
        if (logical_page_idx < num_max_pages) {
            if (m % 2 == 0) {
                auto phy_page_idx = block_table_cur_batch[logical_page_idx];
                tKgK.data() = gK.data() + phy_page_idx * kmc_stride0 + col_offset;
            }
            #pragma unroll
            for (int k = 0; k < size<2>(tKgK); ++k) {
                cute::copy(gmem_tiled_copy_K, tKgK(_, m, k), tKsK(_, m, k));
            }
        }
    }

    cute::cp_async_fence();

    typename Kernel_Traits::TiledMma tiledMMa;
    auto thr_mma = tiledMMa.get_thread_slice(tidx);
    cute::Tensor tOrQ = thr_mma.partition_fragment_A(sQ);
    cute::Tensor tOrK = thr_mma.partition_fragment_B(sK);
    cute::Tensor cC = make_identity_tensor(Shape<Int<kBlockQ>, Int<kBlockK>>{});
    // cute::Tensor tOcC = thr_mma.partition_fragment_C(cC);
    // cute::Tensor tOrC = partition_fragment_C(tiledMMa, Shape<Int<kBlockQ>, Int<kBlockK>>{});
    cute::clear(tOrC);

    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_Traits::SmemCopyAtom{}, tiledMMa);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tid);
    cute::Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);
    cute::Tensor tOrQ_copy_view = smem_thr_copy_Q.retile_D(tOrQ);
    
    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_Traits::SmemCopyAtom{}, tiledMMa);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tid);
    cute::Tensor tSsK = smem_thr_copy_K.partition_S(sK);
    cute::Tensor tOrK_copy_view = smem_thr_copy_K.retile_D(tOrK);

    cute::cp_async_wait<0>();
    __syncthreads();

    cute::copy(smem_tiled_copy_Q, tSsQ(_, _, _0{}), tOrQ_copy_view(_, _, _0{}));
    cute::copy(smem_tiled_copy_K, tSsK(_, _, _0{}), tOrK_copy_view(_, _, _0{}));
    for (int i = 0; i < size<2>(tOrQ); ++i) {
        if (i < size<2>(tOrQ) - 1) {
            cute::copy(smem_tiled_copy_Q, tSsQ(_, _, i + 1), tOrQ_copy_view(_, _, i + 1));
            cute::copy(smem_tiled_copy_K, tSsB(_, _, i + 1), tOrK_copy_view(_, _, i + 1));
        }
        cute::gemm(tiledMMa, tOrQ(_, _, i), tOrK(_, _, i), tOrC);
    }
    __syncthreads();

    // auto smem_tile_copy_C = make_tiled_copy_C(typename Kernel_Traits::SmemCopyAtomO{}, tiledMMa);
    // auto smem_thr_copy_C = smem_tile_copy_C.get_thread_slice(tid);
    // 
    
}

template<typename T, int HEAD_DIM>
void lserve_page_selector_kernel_launch(
    const T* q,                     // [batch_size, seqlen_q, num_head, head_dim]
    const T* key_meta_cache,        // [num_block, num_head, 2, head_dim]
    const int* block_table,         // [batch_size, max_block_size]
    const int* num_full_blocks,     // [batch_size, ]
    float* out,                         // [batch_size, num_head, seqlen_q, max_block_size]
    const int batch_size,
    const int seqlen_q,
    const int num_head,
    const int max_block_size,

    const int64_t qstride0,
    const int64_t qstride1,
    const int64_t qstride2
) {
    TORCH_CHECK(seqlen_q <= 64);
    constexpr int kBlockK = 128;
    using kernel_traits = Kernel_Traits<4, HEAD_DIM, 64, kBlockK, T>;
    // 先用固定分块 64，后面再改进为 splitkv
    constexpr unsigned META_CACHE_BLOCKS_PER_THREAD_BLOCK = kBlockK >> 1;
    const unsigned num_block_tiles = (max_block_size + META_CACHE_BLOCKS_PER_THREAD_BLOCK - 1) / META_CACHE_BLOCKS_PER_THREAD_BLOCK;
    const int64_t kmc_stride0 = ((int64_t)1) * num_head * 2 * HEAD_DIM;
    const int64_t kmc_stride1 = ((int64_t)1) * 2 * HEAD_DIM;
    const int64_t kmc_stride2 = ((int64_t)1) * HEAD_DIM;
    const int64_t out_stride0 = ((int64_t)1) * num_head * seqlen_q * max_block_size;
    const int64_t out_stride1 = ((int64_t)1) * seqlen_q * max_block_size;
    const int64_t out_stride2 = ((int64_t)1) * max_block_size;
    dim3 grid(batch_size, num_head, num_block_tiles);
    dim3 block(128);
    // lserve_page_selector_kernel<T, HEAD_DIM><<<grid, block>>>(
    //     q, key_meta_cache, block_table,
    //     num_full_blocks, out,
    //     num_q_head,
    //     num_kv_head,
    //     max_block_size,
    //     qstride0, qstride1, 
    //     kmc_stride0, kmc_stride1, kmc_stride2,
    //     out_stride0, out_stride1
    // );
    lserve_page_selector_kernel_v2<kernel_traits><<<grid, block>>>(
        q, key_meta_cache, block_table,
        num_full_blocks, out,
        batch_size,
        seqlen_q,
        num_head,
        max_block_size,
        qstride0, qstride1, qstride2,
        kmc_stride0, kmc_stride1, kmc_stride2,
        out_stride0, out_stride1, out_stride2
    );
}

// TODO[shk]: block_table 可能不是连续的?
template<typename T>
void lserve_page_selector_head_dim_dispatcher(
    const T* q,                   // [batch_size, seqlen_q, num_head, head_dim]
    const T* key_meta_cache,      // [num_block, num_head, 2, head_dim]
    const int* block_table,       // [batch_size, max_block_size]
    const int* num_full_blocks,   // [batch_size, ]
    float* out,                       // [batch_size, num_head, max_block_size]
    const int batch_size,
    const int seqlen_q,
    const int num_head,
    const int head_dim,
    const int max_block_size,

    const int64_t qstride0,
    const int64_t qstride1,
    const int64_t qstride2
) {
    DISPATCH_WITH_HEAD_DIM(head_dim, "lserve_page_selector_head_dim_dispatcher", [&] {
        lserve_page_selector_kernel_launch<T, HEAD_DIM>(
            q, key_meta_cache, block_table,
            num_full_blocks, out,
            batch_size, 
            seqlen_q,
            num_head,
            max_block_size,
            qstride0, qstride1, qstride2
        );
    });
}


void lserve_page_selector(
    const torch::Tensor& q,                   // [batch_size, num_q_head, head_dim]. Only for decode. bf16/fp16
    const torch::Tensor& key_meta_cache,      // [num_block, num_kv_head, 2, head_dim] bf16/fp16
    const torch::Tensor& block_table,         // [batch_size, max_block_size] int32
    const torch::Tensor& num_full_blocks,     // [batch_size, ] int32
    torch::Tensor& out                        // [batch_size, num_kv_head, max_block_size] float
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
    int num_kv_head = key_meta_cache.size(1);
    int seqlen_q = 1;
    int num_heads = num_q_head;
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

    // 类似 FlashAttention 的做法 reshape q
    {
        const int ngroups = num_q_head / num_kv_head;
        q = q.reshape({batch_size, num_kv_head, ngroups, head_dim}).transpose(1, 2);
        seqlen_q = ngroups;
        num_heads = num_kv_head;
        TORCH_CHECK(seqlen_q <= 64);
    }

    const at::cuda::OptionalCUDAGuard device_guard(device_of(q));
    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();

    DISPATCH_HALF_AND_BF16(q.scalar_type(), "lserve_page_selector", [&] {
        using DataType = typename SATypeConverter<scalar_t>::Type;

        lserve_page_selector_head_dim_dispatcher<DataType>(
            reinterpret_cast<DataType*>(q.data_ptr()),
            reinterpret_cast<DataType*>(key_meta_cache.data_ptr()),
            block_table.data_ptr<int>(),
            num_full_blocks.data_ptr<int>(),
            reinterpret_cast<float*>(out.data_ptr()),
            batch_size,
            seqlen_q,
            num_heads,
            head_dim,
            max_block_size,
            q.stride(0),
            q.stride(1),
            q.stride(2)
        );
    });
}