/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 * Copyright (c) 2025, Jiading Gai — DualKV training backward kernel.
 *
 * Forked from flash_bwd_kernel.h:compute_dq_dk_dv_1colblock.
 * Split KV into context (bs=1, shared) + decoded (bs=N, per-sequence).
 * No dropout, alibi, softcap, or local attention.
 *
 * Key difference from standard backward:
 * - K/V loaded from kcontext_ptr or kdecoded_ptr based on physical block index
 * - dK/dV for context blocks: atomicAdd (shared across batch)
 * - dK/dV for decoded blocks: direct write (per-sequence)
 * - Causal mask uses logical K position (context blocks + decoded blocks)
 ******************************************************************************/

#pragma once

#include "namespace_config.h"

#include <cute/tensor.hpp>

#include <cutlass/cutlass.h>
#include <cutlass/array.h>
#include <cutlass/numeric_types.h>

#include "block_info.h"
#include "kernel_traits.h"
#include "utils.h"
#include "softmax.h"
#include "mask.h"

namespace FLASH_NAMESPACE {

using namespace cute;

////////////////////////////////////////////////////////////////////////////////////////////////////
// Reuse these from flash_bwd_kernel.h — they need to be visible in our TU.
// Since flash_bwd_kernel.h uses #pragma once, including it is safe.

template <int MMA_N, class... Args, class TiledMMA>
CUTE_HOST_DEVICE auto
make_tiled_copy_B_warpcontiguousN_dualkv(Copy_Atom<Args...> const& copy_atom, TiledMMA const& tiled_mma) {
    constexpr int TileShape_N = decltype(tiled_mma.template tile_size_mnk<1>())::value;
    constexpr int TileShape_K = decltype(tiled_mma.template tile_size_mnk<2>())::value;
    using AtomShape_MNK = typename TiledMMA::AtomShape_MNK;
    constexpr int AtomShape_N = decltype(size<1>(AtomShape_MNK{}))::value;
    constexpr int kNWarpsN = TileShape_N / AtomShape_N / 2;
    constexpr int MMAStride_N = MMA_N * AtomShape_N * 2;
    auto t = make_tile(Layout<Shape<Int<AtomShape_N>, Int<kNWarpsN>, _2>,
                              Stride<_1, Int<MMAStride_N>, _8> >{},
                       make_layout(Int<TileShape_K>{}));
    return make_tiled_copy_impl(copy_atom, tiled_mma.get_layoutB_TV(), t);
}

template <int MMA_N, class... Args, class TiledMMA>
CUTE_HOST_DEVICE auto
make_tiled_copy_C_warpcontiguousN_dualkv(Copy_Atom<Args...> const& copy_atom, TiledMMA const& tiled_mma) {
    constexpr int TileShape_M = decltype(tiled_mma.template tile_size_mnk<0>())::value;
    constexpr int TileShape_N = decltype(tiled_mma.template tile_size_mnk<1>())::value;
    using AtomShape_MNK = typename TiledMMA::AtomShape_MNK;
    constexpr int AtomShape_N = decltype(size<1>(AtomShape_MNK{}))::value;
    constexpr int kNWarpsN = TileShape_N / AtomShape_N / 2;
    constexpr int MMAStride_N = MMA_N * AtomShape_N * 2;
    auto t = make_tile(make_layout(Int<TileShape_M>{}),
                       Layout<Shape<Int<AtomShape_N>, Int<kNWarpsN>, _2>,
                              Stride<_1, Int<MMAStride_N>, _8> >{});
    return make_tiled_copy_impl(copy_atom, tiled_mma.get_layoutC_TV(), t);
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_causal, bool Is_even_K, bool Is_first, bool Is_last, typename Params>
inline __device__ void compute_dq_dk_dv_1colblock_dualkv_training(
    const Params &params, const int bidb, const int bidh, const int n_block)
{
    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    extern __shared__ char smem_[];
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int MMA_N_SdP = kBlockN / decltype(typename Kernel_traits::TiledMmaSdP{}.template tile_size_mnk<1>())::value;
    constexpr int AtomLayoutMS = Kernel_traits::AtomLayoutMSdP;
    constexpr bool Double_buffer = !Kernel_traits::No_double_buffer;

    // --- Sequence info ---
    const int sum_s_q = params.cu_seqlens_q == nullptr ? 0 : params.cu_seqlens_q[bidb];
    const int actual_seqlen_q = params.cu_seqlens_q == nullptr
        ? params.seqlen_q
        : params.cu_seqlens_q[bidb + 1] - params.cu_seqlens_q[bidb];

    const int context_seqlen = params.seqlen_k_context;
    const int sum_s_k_dec = params.cu_seqlens_k_decoded == nullptr ? 0 : params.cu_seqlens_k_decoded[bidb];
    const int actual_seqlen_k_decoded = params.cu_seqlens_k_decoded == nullptr
        ? params.seqlen_k_decoded
        : params.cu_seqlens_k_decoded[bidb + 1] - params.cu_seqlens_k_decoded[bidb];
    const int actual_seqlen_k = context_seqlen + actual_seqlen_k_decoded;

    const int n_blocks_ctx = cute::ceil_div(context_seqlen, kBlockN);
    const int n_blocks_dec = cute::ceil_div(actual_seqlen_k_decoded, kBlockN);

    // Is this a context block or decoded block?
    const bool is_context_block = (n_block < n_blocks_ctx);

    // Check if this block is within bounds
    if (is_context_block) {
        if (n_block * kBlockN >= context_seqlen) return;
    } else {
        const int dec_block = n_block - n_blocks_ctx;
        if (dec_block * kBlockN >= actual_seqlen_k_decoded) return;
    }

    // Number of valid K rows in this block
    const int valid_k = is_context_block
        ? std::min(kBlockN, context_seqlen - n_block * kBlockN)
        : std::min(kBlockN, actual_seqlen_k_decoded - (n_block - n_blocks_ctx) * kBlockN);

    // Logical K position for the start of this block
    const int k_pos_base = is_context_block
        ? n_block * kBlockN
        : context_seqlen + (n_block - n_blocks_ctx) * kBlockN;

    // --- m_block_max: highest Q block that could attend to this K block ---
    int m_block_max = cute::ceil_div(actual_seqlen_q, kBlockM);
    // For causal, Q at position m can only attend to K at position <= m
    // So K at position k_pos_base + kBlockN - 1 can only be attended by Q at position >= k_pos_base + kBlockN - 1
    // And the last Q that could attend to us is actual_seqlen_q - 1
    // m_block_max is already correct for non-causal

    // m_block_min: for causal, this K block can only be attended by Q positions >= k_pos_base
    int m_block_min = Is_causal
        ? std::max(0, (k_pos_base + actual_seqlen_q - actual_seqlen_k) / kBlockM)
        : 0;

    // Early exit: no Q blocks to process
    if (m_block_max <= m_block_min) {
        // Write zeros to dK and dV
        // We need to know which output pointer to use
        Element *dk_out_ptr, *dv_out_ptr;
        index_t dk_row_stride, dv_row_stride;
        if (is_context_block) {
            dk_out_ptr = reinterpret_cast<Element *>(params.dk_context_ptr);
            dv_out_ptr = reinterpret_cast<Element *>(params.dv_context_ptr);
            dk_row_stride = params.dk_context_row_stride;
            dv_row_stride = params.dv_context_row_stride;
        } else {
            dk_out_ptr = reinterpret_cast<Element *>(params.dk_decoded_ptr);
            dv_out_ptr = reinterpret_cast<Element *>(params.dv_decoded_ptr);
            dk_row_stride = params.dk_decoded_row_stride;
            dv_row_stride = params.dv_decoded_row_stride;
        }
        // For context: no need to write zeros since we zero-init before launch
        // For decoded: write zeros to this batch's portion
        if (!is_context_block) {
            const int dec_block = n_block - n_blocks_ctx;
            const index_t offset_dk = sum_s_k_dec * dk_row_stride + dec_block * kBlockN * dk_row_stride + bidh * params.dk_decoded_head_stride;
            const index_t offset_dv = sum_s_k_dec * dv_row_stride + dec_block * kBlockN * dv_row_stride + bidh * params.dv_decoded_head_stride;
            Tensor gdK = make_tensor(make_gmem_ptr(dk_out_ptr + offset_dk),
                                     Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                     make_stride(dk_row_stride, _1{}));
            Tensor gdV = make_tensor(make_gmem_ptr(dv_out_ptr + offset_dv),
                                     Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                     make_stride(dv_row_stride, _1{}));
            typename Kernel_traits::GmemTiledCopydKV gmem_tiled_copy_dKV;
            auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);
            Tensor tdKgdK = gmem_thr_copy_dKV.partition_D(gdK);
            Tensor tdVgdV = gmem_thr_copy_dKV.partition_D(gdV);
            Tensor tdKrdK = make_tensor<Element>(shape(tdKgdK));
            Tensor tdVrdV = make_tensor<Element>(shape(tdVgdV));
            clear(tdKrdK);
            clear(tdVrdV);
            Tensor cdKV = make_identity_tensor(make_shape(size<0>(gdK), size<1>(gdK)));
            Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);
            Tensor tdKVpdKV = make_tensor<bool>(make_shape(size<2>(tdKgdK)));
            #pragma unroll
            for (int k = 0; k < size(tdKVpdKV); ++k) { tdKVpdKV(k) = get<1>(tdKVcdKV(0, 0, k)) < params.d; }
            FLASH_NAMESPACE::copy</*Is_even_MN=*/false, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
                gmem_tiled_copy_dKV, tdKrdK, tdKgdK, tdKVcdKV, tdKVpdKV, valid_k);
            FLASH_NAMESPACE::copy</*Is_even_MN=*/false, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
                gmem_tiled_copy_dKV, tdVrdV, tdVgdV, tdKVcdKV, tdKVpdKV, valid_k);
        }
        return;
    }

    // --- K/V pointers for this block ---
    const int kv_head = bidh / params.h_h_k_ratio;
    Element *k_ptr, *v_ptr;
    index_t k_row_stride, v_row_stride;
    if (is_context_block) {
        k_ptr = reinterpret_cast<Element *>(params.kcontext_ptr)
            + n_block * kBlockN * params.kcontext_row_stride
            + kv_head * params.kcontext_head_stride;
        v_ptr = reinterpret_cast<Element *>(params.vcontext_ptr)
            + n_block * kBlockN * params.vcontext_row_stride
            + kv_head * params.vcontext_head_stride;
        k_row_stride = params.kcontext_row_stride;
        v_row_stride = params.vcontext_row_stride;
    } else {
        const int dec_block = n_block - n_blocks_ctx;
        k_ptr = reinterpret_cast<Element *>(params.kdecoded_ptr)
            + sum_s_k_dec * params.kdecoded_row_stride
            + dec_block * kBlockN * params.kdecoded_row_stride
            + kv_head * params.kdecoded_head_stride;
        v_ptr = reinterpret_cast<Element *>(params.vdecoded_ptr)
            + sum_s_k_dec * params.vdecoded_row_stride
            + dec_block * kBlockN * params.vdecoded_row_stride
            + kv_head * params.vdecoded_head_stride;
        k_row_stride = params.kdecoded_row_stride;
        v_row_stride = params.vdecoded_row_stride;
    }

    // --- Q, dO, O, dQ offsets (cast to index_t to prevent int32 overflow) ---
    const int m_block = m_block_max - 1;
    const index_t row_offset_q = (index_t)sum_s_q * params.q_row_stride
        + (index_t)m_block * kBlockM * params.q_row_stride + bidh * params.q_head_stride;
    const index_t row_offset_do = (index_t)sum_s_q * params.do_row_stride
        + (index_t)m_block * kBlockM * params.do_row_stride + bidh * params.do_head_stride;
    const index_t row_offset_o = (index_t)sum_s_q * params.o_row_stride
        + (index_t)m_block * kBlockM * params.o_row_stride + bidh * params.o_head_stride;
    const index_t row_offset_dq = (index_t)sum_s_q * params.dq_row_stride
        + (index_t)m_block * kBlockM * params.dq_row_stride + bidh * params.dq_head_stride;
    const index_t row_offset_dq_accum = ((index_t)sum_s_q * params.h * params.d_rounded)
        + ((index_t)m_block * kBlockM + 128ll * bidb) * params.h * params.d_rounded + bidh * params.d_rounded
        + (!params.deterministic ? 0 : blockIdx.x * params.dq_accum_split_stride);
    const index_t row_offset_lse = (index_t)bidh * params.total_q + sum_s_q + m_block * kBlockM;
    const index_t row_offset_dpsum = (index_t)bidh * (params.total_q + 128 * params.b) + sum_s_q + 128 * bidb + m_block * kBlockM;

    Tensor gQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.q_ptr) + row_offset_q),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.q_row_stride, _1{}));
    Tensor gK = make_tensor(make_gmem_ptr(k_ptr),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(k_row_stride, _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(v_ptr),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(v_row_stride, _1{}));
    Tensor gdO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.do_ptr) + row_offset_do),
                             Shape<Int<kBlockM>, Int<kHeadDim>>{},
                             make_stride(params.do_row_stride, _1{}));
    Tensor gO = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.o_ptr) + row_offset_o),
                            Shape<Int<kBlockM>, Int<kHeadDim>>{},
                            make_stride(params.o_row_stride, _1{}));
    Tensor gdQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dq_ptr) + row_offset_dq),
                             Shape<Int<kBlockM>, Int<kHeadDim>>{},
                             make_stride(params.dq_row_stride, _1{}));
    Tensor gdQaccum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dq_accum_ptr) + row_offset_dq_accum),
                                  Shape<Int<kBlockM>, Int<kHeadDim>>{},
                                  make_stride(params.h * params.d_rounded, _1{}));
    Tensor gLSE = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.softmax_lse_ptr) + row_offset_lse),
                              Shape<Int<kBlockM>>{}, Stride<_1>{});
    Tensor gdPsum = make_tensor(make_gmem_ptr(reinterpret_cast<ElementAccum *>(params.dsoftmax_sum) + row_offset_dpsum),
                                Shape<Int<kBlockM>>{}, Stride<_1>{});

    // --- Shared memory ---
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQdO{});
    Tensor sQt = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQdOtransposed{});
    Tensor sQtNoSwizzle = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutQdOtransposedNoSwizzle{});
    Tensor sdO = make_tensor(sQ.data() + (Double_buffer ? 2 : 1) * size(sQ), typename Kernel_traits::SmemLayoutQdO{});
    Tensor sdOt = make_tensor(sdO.data(), typename Kernel_traits::SmemLayoutQdOtransposed{});
    Tensor sdOtransposedNoSwizzle = make_tensor(sdO.data(), typename Kernel_traits::SmemLayoutQdOtransposedNoSwizzle{});
    Tensor sK = make_tensor(sdO.data() + size(sdO), typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sKt = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKtransposed{});
    Tensor sKtNoSwizzle = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutKtransposedNoSwizzle{});
    Tensor sdS = make_tensor(!Kernel_traits::Is_V_in_regs ? sV.data() + size(sV) : sK.data() + size(sK),
                             typename Kernel_traits::SmemLayoutPdS{});
    Tensor sdSt = make_tensor(sdS.data(), typename Kernel_traits::SmemLayoutPdStransposed{});
    Tensor sdStNoSwizzle = make_tensor(sdS.data(), typename Kernel_traits::SmemLayoutPdStransposedNoSwizzle{});
    Tensor sP = make_tensor(sdS.data() + size(sdS), typename Kernel_traits::SmemLayoutPdS{});
    Tensor sPt = make_tensor(sP.data(), typename Kernel_traits::SmemLayoutPdStransposed{});
    Tensor sPtNoSwizzle = make_tensor(sP.data(), typename Kernel_traits::SmemLayoutPdStransposedNoSwizzle{});
    Tensor sdQ = make_tensor(sP.data(), typename Kernel_traits::SmemLayoutdQ{});

    // --- Thread-level partitioning ---
    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);
    using GmemTiledCopydO = std::conditional_t<
        Is_first,
        typename Kernel_traits::GmemTiledCopydO,
        typename Kernel_traits::GmemTiledCopyQKV
    >;
    GmemTiledCopydO gmem_tiled_copy_dO;
    auto gmem_thr_copy_dO = gmem_tiled_copy_dO.get_thread_slice(tidx);
    typename Kernel_traits::GmemTiledCopydQ gmem_tiled_copy_dQ;
    auto gmem_thr_copy_dQ = gmem_tiled_copy_dQ.get_thread_slice(tidx);
    using GmemLayoutAtomdQaccum = std::conditional_t<
        true, // Always Seq_parallel for DualKV seqk_parallel path
        typename Kernel_traits::GmemTiledCopydQaccumAtomicAdd,
        typename Kernel_traits::GmemTiledCopydQaccum
    >;
    GmemLayoutAtomdQaccum gmem_tiled_copy_dQaccum;
    auto gmem_thr_copy_dQaccum = gmem_tiled_copy_dQaccum.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tdOgdO = gmem_thr_copy_dO.partition_S(gdO);
    Tensor tdOsdO = gmem_thr_copy_dO.partition_D(sdO);
    Tensor tdOgO = gmem_thr_copy_dO.partition_S(gO);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);
    Tensor tdQsdQ = gmem_thr_copy_dQ.partition_S(sdQ);
    Tensor tdQgdQ = gmem_thr_copy_dQ.partition_D(gdQ);
    Tensor tdQgdQaccum = gmem_thr_copy_dQaccum.partition_D(gdQaccum);

    typename Kernel_traits::TiledMmaSdP tiled_mma_sdp;
    auto thr_mma_sdp = tiled_mma_sdp.get_thread_slice(tidx);
    Tensor tSrQ = thr_mma_sdp.partition_fragment_A(sQ);
    Tensor tSrK = thr_mma_sdp.partition_fragment_B(sK);
    Tensor tdPrdO = thr_mma_sdp.partition_fragment_A(sdO);
    Tensor tdPrV = thr_mma_sdp.partition_fragment_B(sV);

    typename Kernel_traits::TiledMmadKV tiled_mma_dkv;
    auto thr_mma_dkv = tiled_mma_dkv.get_thread_slice(tidx);
    Tensor tdKrdSt = thr_mma_dkv.partition_fragment_A(sdStNoSwizzle);
    Tensor tdKrQt = thr_mma_dkv.partition_fragment_B(sQtNoSwizzle);
    Tensor tdVrPt = thr_mma_dkv.partition_fragment_A(sPtNoSwizzle);
    Tensor tdVrdO = thr_mma_dkv.partition_fragment_B(sdOtransposedNoSwizzle);

    typename Kernel_traits::TiledMmadQ tiled_mma_dq;
    auto thr_mma_dq = tiled_mma_dq.get_thread_slice(tidx);
    Tensor tdQrdS = thr_mma_dq.partition_fragment_A(sdS);
    Tensor tdQrKt = thr_mma_dq.partition_fragment_B(sKtNoSwizzle);

    Tensor acc_dk = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});
    Tensor acc_dv = partition_fragment_C(tiled_mma_dkv, Shape<Int<kBlockN>, Int<kHeadDim>>{});

    // --- Copy atom retiling ---
    auto smem_tiled_copy_QdO = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto smem_thr_copy_QdO = smem_tiled_copy_QdO.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_QdO.partition_S(sQ);
    Tensor tdPsdO = smem_thr_copy_QdO.partition_S(sdO);

    auto smem_tiled_copy_KV = make_tiled_copy_B_warpcontiguousN_dualkv<MMA_N_SdP>(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_sdp);
    auto smem_thr_copy_KV = smem_tiled_copy_KV.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_KV.partition_S(sK);
    Tensor tdPsV = smem_thr_copy_KV.partition_S(sV);

    auto smem_tiled_copy_PdS = make_tiled_copy_C_warpcontiguousN_dualkv<MMA_N_SdP>(typename Kernel_traits::SmemCopyAtomPdS{}, tiled_mma_sdp);
    auto smem_thr_copy_PdS = smem_tiled_copy_PdS.get_thread_slice(tidx);
    Tensor tPsP = smem_thr_copy_PdS.partition_D(sP);
    Tensor tdSsdS = smem_thr_copy_PdS.partition_D(sdS);

    auto smem_tiled_copy_PdSt = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_dkv);
    auto smem_thr_copy_PdSt = smem_tiled_copy_PdSt.get_thread_slice(tidx);
    Tensor tdVsPt = smem_thr_copy_PdSt.partition_S(sPt);
    Tensor tdKsdSt = smem_thr_copy_PdSt.partition_S(sdSt);

    auto smem_tiled_copy_QdOt = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_dkv);
    auto smem_thr_copy_QdOt = smem_tiled_copy_QdOt.get_thread_slice(tidx);
    Tensor tdVsdOt = smem_thr_copy_QdOt.partition_S(sdOt);
    Tensor tdKsQt = smem_thr_copy_QdOt.partition_S(sQt);

    auto smem_tiled_copy_dS = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma_dq);
    auto smem_thr_copy_dS = smem_tiled_copy_dS.get_thread_slice(tidx);
    Tensor tdQsdS = smem_thr_copy_dS.partition_S(sdS);

    auto smem_tiled_copy_Kt = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma_dq);
    auto smem_thr_copy_Kt = smem_tiled_copy_Kt.get_thread_slice(tidx);
    Tensor tdQsKt = smem_thr_copy_Kt.partition_S(sKt);

    auto smem_tiled_copy_dQ = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdQ{}, tiled_mma_dq);
    auto smem_thr_copy_dQ = smem_tiled_copy_dQ.get_thread_slice(tidx);
    Tensor taccdQsdQ = smem_thr_copy_dQ.partition_D(sdQ);

    // --- Predicates ---
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));
    Tensor tQcQ = gmem_thr_copy_QKV.partition_D(cQ);
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_D(cKV);

    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    // --- Prologue ---
    // Advance gdQ and gdQaccum by one block (we iterate backward)
    tdQgdQ.data() = tdQgdQ.data() + kBlockM * params.dq_row_stride;
    tdQgdQaccum.data() = tdQgdQaccum.data() + kBlockM * params.h * params.d_rounded;

    int m_blk = m_block_max - 1;

    if (Double_buffer && m_blk % 2 == 1) {
        tQsQ.data() = tQsQ.data() + size(sQ);
        tSsQ.data() = tSsQ.data() + size(sQ);
        tdKsQt.data() = tdKsQt.data() + size(sQ);
    }

    if ((!Is_first) || params.deterministic) { __syncthreads(); }

    if (Kernel_traits::Is_V_in_regs) {
        FLASH_NAMESPACE::copy</*Is_even_MN=*/false, Is_even_K, /*Clear_OOB_MN=*/true>(
            gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV, valid_k);
        FLASH_NAMESPACE::cp_async_fence();
    }

    Tensor tdOrdO = make_fragment_like(tdOgdO);
    Tensor tdOrO = make_fragment_like(tdOgO);
    if (!Is_first) {
        FLASH_NAMESPACE::copy</*Is_even_MN=*/false, Is_even_K, /*Clear_OOB_MN=*/true>(
            gmem_tiled_copy_dO, tdOgdO, tdOsdO, tQcQ, tQpQ, actual_seqlen_q - m_blk * kBlockM);
    } else {
        FLASH_NAMESPACE::copy</*Is_even_MN=*/false, Is_even_K, /*Clear_OOB_MN=*/true>(
            gmem_tiled_copy_dO, tdOgdO, tdOrdO, tQcQ, tQpQ, actual_seqlen_q - m_blk * kBlockM);
        FLASH_NAMESPACE::copy</*Is_even_MN=*/false, Is_even_K, /*Clear_OOB_MN=*/true>(
            gmem_tiled_copy_dO, tdOgO, tdOrO, tQcQ, tQpQ, actual_seqlen_q - m_blk * kBlockM);
    }
    FLASH_NAMESPACE::copy</*Is_even_MN=*/false, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ, actual_seqlen_q - m_blk * kBlockM);

    Tensor caccS = make_identity_tensor(Shape<Int<kBlockM>, Int<kBlockN>>{});
    Tensor taccScS = thr_mma_sdp.partition_C(caccS);
    static_assert(decltype(size<0>(taccScS))::value == 4);
    Tensor taccScS_row = logical_divide(taccScS, Shape<_2>{})(make_coord(0, _), _, 0);
    Tensor lse = make_tensor<ElementAccum>(Shape<Int<decltype(size(taccScS_row))::value>>{});
    #pragma unroll
    for (int mi = 0; mi < size(lse); ++mi) {
        const int row = get<0>(taccScS_row(mi));
        lse(mi) = row < actual_seqlen_q - m_blk * kBlockM ? gLSE(row) : INFINITY;
    }

    FLASH_NAMESPACE::copy</*Is_even_MN=*/false, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV, valid_k);
    if (!Kernel_traits::Is_V_in_regs) {
        FLASH_NAMESPACE::copy</*Is_even_MN=*/false, Is_even_K, /*Clear_OOB_MN=*/true>(
            gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV, valid_k);
    }
    FLASH_NAMESPACE::cp_async_fence();

    if (Is_first) {
        cute::copy(tdOrdO, tdOsdO);
        dot_do_o<Kernel_traits::kGmemThreadsPerRow>(tdOrdO, tdOrO, gdPsum,
                                                    Kernel_traits::kNThreads / (Kernel_traits::kGmemThreadsPerRow), /*p_dropout=*/1.0f);
    }

    if (Kernel_traits::Is_V_in_regs) {
        cute::cp_async_wait<1>();
        __syncthreads();
        Tensor tdPrV_copy_view = smem_thr_copy_KV.retile_D(tdPrV);
        CUTE_STATIC_ASSERT_V(size<1>(tdPsV) == size<1>(tdPrV_copy_view));
        cute::copy(smem_tiled_copy_KV, tdPsV, tdPrV_copy_view);
    }

    clear(acc_dv);
    clear(acc_dk);

    FLASH_NAMESPACE::Mask<Is_causal, /*Is_local=*/false, /*Has_alibi=*/false> mask(
        actual_seqlen_k, actual_seqlen_q, /*window_left=*/-1, /*window_right=*/-1, /*alibi_slope=*/0.0f);

    // --- Main loop over Q blocks ---
    for (; m_blk >= m_block_min; --m_blk) {
        Tensor acc_s = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s);
        cute::cp_async_wait<0>();
        __syncthreads();

        Tensor dP_sum = make_fragment_like(lse);
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) { dP_sum(mi) = gdPsum(get<0>(taccScS_row(mi))); }

        // S = Q @ K^T
        FLASH_NAMESPACE::gemm(acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma_sdp,
                    smem_tiled_copy_QdO, smem_tiled_copy_KV, smem_thr_copy_QdO, smem_thr_copy_KV);

        // Apply mask using LOGICAL K position
        Tensor scores = make_tensor(acc_s.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_s.layout()));
        if constexpr (Is_causal) {
            // Causal mask: Q at position q_pos can attend to K at position <= q_pos
            // Use logical K position (k_pos_base) for correct masking
            FLASH_NAMESPACE::apply_mask_causal(scores,
                k_pos_base + (tidx / 32 / AtomLayoutMS) * MMA_N_SdP * 16,
                actual_seqlen_k,
                m_blk * kBlockM + get<0>(taccScS_row(0)),
                actual_seqlen_q,
                AtomLayoutMS * 16);
        } else {
            // Bounds mask for non-causal: mask out OOB K positions
            if (k_pos_base + kBlockN > actual_seqlen_k) {
                FLASH_NAMESPACE::apply_mask(scores, actual_seqlen_k,
                    k_pos_base + (tidx / 32 / AtomLayoutMS) * MMA_N_SdP * 16);
            }
        }

        // Per-block OOB masking: context and decoded blocks have independent
        // padding, so "ghost" positions (K=0 but logically valid) need -inf.
        if (valid_k < kBlockN) {
            FLASH_NAMESPACE::apply_mask(scores, valid_k,
                (tidx / 32 / AtomLayoutMS) * MMA_N_SdP * 16);
        }

        // P = exp(S - LSE)
        FLASH_NAMESPACE::scale_apply_exp2</*scale_max=*/false>(scores, lse, params.scale_softmax_log2);

        // Convert P to fp16
        Tensor rP = FLASH_NAMESPACE::convert_type<Element>(acc_s);
        Tensor tPrP = make_tensor(rP.data(), FLASH_NAMESPACE::convert_layout_acc_Aregs<Kernel_traits::TiledMmaSdP>(rP.layout()));
        Tensor tPaP = smem_thr_copy_PdS.retile_S(tPrP);
        cute::copy(smem_tiled_copy_PdS, tPaP, tPsP);

        // dP = dO @ V^T
        Tensor acc_dp = partition_fragment_C(tiled_mma_sdp, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_dp);
        FLASH_NAMESPACE::gemm</*A_in_regs=*/false, /*B_in_regs=*/Kernel_traits::Is_V_in_regs>(
            acc_dp, tdPrdO, tdPrV, tdPsdO, tdPsV, tiled_mma_sdp,
            smem_tiled_copy_QdO, smem_tiled_copy_KV, smem_thr_copy_QdO, smem_thr_copy_KV);

        // dS = P * (dP - D)
        Tensor dS = make_tensor(acc_dp.data(), scores.layout());
        #pragma unroll
        for (int mi = 0; mi < size<0>(dS); ++mi) {
            #pragma unroll
            for (int ni = 0; ni < size<1>(dS); ++ni) {
                dS(mi, ni) = scores(mi, ni) * (dS(mi, ni) - dP_sum(mi));
            }
        }

        // dQ accumulation
        Tensor acc_dq = partition_fragment_C(tiled_mma_dq, Shape<Int<kBlockM>, Int<kHeadDim>>{});
        tdQgdQaccum.data() = tdQgdQaccum.data() + (-int(kBlockM * params.h * params.d_rounded));
        // Always Seq_parallel for DualKV
        clear(acc_dq);

        if (Double_buffer && m_blk > m_block_min) {
            const int sQ_offset = m_blk % 2 == 0 ? size(sQ) : -size(sQ);
            tQsQ.data() = tQsQ.data() + sQ_offset;
            tSsQ.data() = tSsQ.data() + sQ_offset;
            tQgQ.data() = tQgQ.data() + (-int(kBlockM * params.q_row_stride));
            FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ);
            FLASH_NAMESPACE::cp_async_fence();
        }

        // Convert dS to fp16 and write to smem
        Tensor dS_reshaped = make_tensor(dS.data(), acc_dp.layout());
        Tensor tdSrdS = FLASH_NAMESPACE::convert_type<Element>(dS_reshaped);
        Tensor tdSadS = smem_thr_copy_PdS.retile_S(tdSrdS);
        cute::copy(smem_tiled_copy_PdS, tdSadS, tdSsdS);
        __syncthreads();

        // dV += P^T @ dO
        FLASH_NAMESPACE::gemm(acc_dv, tdVrPt, tdVrdO, tdVsPt, tdVsdOt, tiled_mma_dkv,
                    smem_tiled_copy_PdSt, smem_tiled_copy_QdOt, smem_thr_copy_PdSt, smem_thr_copy_QdOt);

        __syncthreads();

        if (m_blk > m_block_min) {
            tdOgdO.data() = tdOgdO.data() + (-int(kBlockM * params.do_row_stride));
            if (Is_first) {
                tdOgO.data() = tdOgO.data() + (-int(kBlockM * params.o_row_stride));
                FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_dO, tdOgdO, tdOrdO, tQcQ, tQpQ);
                FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_dO, tdOgO, tdOrO, tQcQ, tQpQ);
            } else {
                FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_dO, tdOgdO, tdOsdO, tQcQ, tQpQ);
                FLASH_NAMESPACE::cp_async_fence();
            }
        }

        // dQ += dS @ K
        FLASH_NAMESPACE::gemm(acc_dq, tdQrdS, tdQrKt, tdQsdS, tdQsKt, tiled_mma_dq,
                    smem_tiled_copy_dS, smem_tiled_copy_Kt, smem_thr_copy_dS, smem_thr_copy_Kt);

        if (m_blk > m_block_min) {
            gLSE.data() = gLSE.data() + (-int(kBlockM));
            #pragma unroll
            for (int mi = 0; mi < size(lse); ++mi) { lse(mi) = gLSE(get<0>(taccScS_row(mi))); }
            gdPsum.data() = gdPsum.data() + (-int(kBlockM));
        }

        // Write dQ (always Seq_parallel = atomicAdd)
        // Reshape acc_dq and atomicAdd to dQaccum
        {
            CUTE_STATIC_ASSERT_V(size(acc_dq) == size(tdQgdQaccum));
            #pragma unroll
            for (int i = 0; i < size(acc_dq); ++i) { atomicAdd(&tdQgdQaccum(i), acc_dq(i)); }
        }

        // dK += dS^T @ Q
        FLASH_NAMESPACE::gemm(acc_dk, tdKrdSt, tdKrQt, tdKsdSt, tdKsQt, tiled_mma_dkv,
                    smem_tiled_copy_PdSt, smem_tiled_copy_QdOt, smem_thr_copy_PdSt, smem_thr_copy_QdOt);

        if (Double_buffer) {
            tdKsQt.data() = tdKsQt.data() + (m_blk % 2 == 0 ? size(sQ) : -size(sQ));
        }
        if (!Double_buffer && m_blk > m_block_min) {
            __syncthreads();
            tQgQ.data() = tQgQ.data() + (-int(kBlockM * params.q_row_stride));
            FLASH_NAMESPACE::copy</*Is_even_MN=*/true, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ);
            FLASH_NAMESPACE::cp_async_fence();
        }

        if (Is_first && m_blk > m_block_min) {
            cute::copy(tdOrdO, tdOsdO);
            dot_do_o<Kernel_traits::kGmemThreadsPerRow>(tdOrdO, tdOrO, gdPsum,
                                                        Kernel_traits::kNThreads / (Kernel_traits::kGmemThreadsPerRow), /*p_dropout=*/1.0f);
        }
    }

    // --- Epilogue: write dK, dV ---
    // No dropout, so no rp_dropout scaling for dV
    #pragma unroll
    for (int i = 0; i < size(acc_dk); ++i) { acc_dk(i) *= params.scale_softmax; }

    if (is_context_block) {
        // Context: atomicAdd fp32 accumulators directly to gmem fp32 buffer.
        // This avoids the fp32->fp16->fp32 round-trip that loses precision when
        // multiple batch elements accumulate to the same context positions.
        const index_t dk_accum_offset = n_block * kBlockN * params.dk_context_accum_row_stride
            + bidh * params.dk_context_accum_head_stride;
        const index_t dv_accum_offset = n_block * kBlockN * params.dv_context_accum_row_stride
            + bidh * params.dv_context_accum_head_stride;
        float *dk_accum_base = reinterpret_cast<float *>(params.dk_context_accum_ptr) + dk_accum_offset;
        float *dv_accum_base = reinterpret_cast<float *>(params.dv_context_accum_ptr) + dv_accum_offset;

        Tensor caccC = make_identity_tensor(Shape<Int<kBlockN>, Int<kHeadDim>>{});
        Tensor taccCrC = thr_mma_dkv.partition_C(caccC);

        CUTE_STATIC_ASSERT_V(size(acc_dk) == size(taccCrC));
        #pragma unroll
        for (int i = 0; i < size(acc_dk); ++i) {
            const int row = get<0>(taccCrC(i));
            const int col = get<1>(taccCrC(i));
            if (row < valid_k && (Is_even_K || col < params.d)) {
                atomicAdd(&dk_accum_base[row * params.dk_context_accum_row_stride + col],
                          acc_dk(i));
                atomicAdd(&dv_accum_base[row * params.dv_context_accum_row_stride + col],
                          acc_dv(i));
            }
        }
    } else {
        // Decoded: fp16 via smem, direct write to gmem (no cross-batch accumulation)
        Tensor rdK = FLASH_NAMESPACE::convert_type<Element>(acc_dk);
        Tensor rdV = FLASH_NAMESPACE::convert_type<Element>(acc_dv);

        Tensor sdK = make_tensor(sK.data(), typename Kernel_traits::SmemLayoutdKV{});
        Tensor sdV = make_tensor(sdK.data() + size(sdK), typename Kernel_traits::SmemLayoutdKV{});

        auto smem_tiled_copy_dKV = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomdKV{}, tiled_mma_dkv);
        auto smem_thr_copy_dKV = smem_tiled_copy_dKV.get_thread_slice(tidx);
        Tensor taccdKrdK = smem_thr_copy_dKV.retile_S(rdK);
        Tensor taccdKsdK = smem_thr_copy_dKV.partition_D(sdK);
        Tensor taccdVrdV = smem_thr_copy_dKV.retile_S(rdV);
        Tensor taccdVsdV = smem_thr_copy_dKV.partition_D(sdV);

        __syncthreads();

        cute::copy(smem_tiled_copy_dKV, taccdKrdK, taccdKsdK);
        cute::copy(smem_tiled_copy_dKV, taccdVrdV, taccdVsdV);

        const int dec_block = n_block - n_blocks_ctx;
        const index_t dk_offset = sum_s_k_dec * params.dk_decoded_row_stride
            + dec_block * kBlockN * params.dk_decoded_row_stride + bidh * params.dk_decoded_head_stride;
        const index_t dv_offset = sum_s_k_dec * params.dv_decoded_row_stride
            + dec_block * kBlockN * params.dv_decoded_row_stride + bidh * params.dv_decoded_head_stride;

        Tensor gdK_out = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dk_decoded_ptr) + dk_offset),
                                     Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                     make_stride(params.dk_decoded_row_stride, _1{}));
        Tensor gdV_out = make_tensor(make_gmem_ptr(reinterpret_cast<Element *>(params.dv_decoded_ptr) + dv_offset),
                                     Shape<Int<kBlockN>, Int<kHeadDim>>{},
                                     make_stride(params.dv_decoded_row_stride, _1{}));

        typename Kernel_traits::GmemTiledCopydKV gmem_tiled_copy_dKV;
        auto gmem_thr_copy_dKV = gmem_tiled_copy_dKV.get_thread_slice(tidx);
        Tensor tdKsdK_out = gmem_thr_copy_dKV.partition_S(sdK);
        Tensor tdKgdK_out = gmem_thr_copy_dKV.partition_D(gdK_out);
        Tensor tdVsdV_out = gmem_thr_copy_dKV.partition_S(sdV);
        Tensor tdVgdV_out = gmem_thr_copy_dKV.partition_D(gdV_out);

        __syncthreads();
        Tensor tdKrdK_out = make_tensor<Element>(shape(tdKgdK_out));
        cute::copy(gmem_tiled_copy_dKV, tdKsdK_out, tdKrdK_out);
        Tensor tdVrdV_out = make_tensor<Element>(shape(tdVgdV_out));
        cute::copy(gmem_tiled_copy_dKV, tdVsdV_out, tdVrdV_out);

        Tensor cdKV = make_identity_tensor(make_shape(size<0>(sdK), size<1>(sdK)));
        Tensor tdKVcdKV = gmem_thr_copy_dKV.partition_D(cdKV);
        Tensor tdKVpdKV = make_tensor<bool>(make_shape(size<2>(tdKgdK_out)));
        #pragma unroll
        for (int k = 0; k < size(tdKVpdKV); ++k) { tdKVpdKV(k) = get<1>(tdKVcdKV(0, 0, k)) < params.d; }

        FLASH_NAMESPACE::copy</*Is_even_MN=*/false, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dKV, tdKrdK_out, tdKgdK_out, tdKVcdKV, tdKVpdKV, valid_k);
        FLASH_NAMESPACE::copy</*Is_even_MN=*/false, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_dKV, tdVrdV_out, tdVgdV_out, tdKVcdKV, tdKVpdKV, valid_k);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_causal, bool Is_even_K, typename Params>
inline __device__ void compute_dq_dk_dv_seqk_parallel_dualkv_training(const Params &params) {
    const int bidb = blockIdx.y;
    const int bidh = blockIdx.z;

    // Total physical blocks
    const int n_blocks_ctx = cute::ceil_div(params.seqlen_k_context, (int)Kernel_traits::kBlockN);
    const int actual_seqlen_k_decoded = params.cu_seqlens_k_decoded == nullptr
        ? params.seqlen_k_decoded
        : params.cu_seqlens_k_decoded[bidb + 1] - params.cu_seqlens_k_decoded[bidb];
    const int n_blocks_dec = cute::ceil_div(actual_seqlen_k_decoded, (int)Kernel_traits::kBlockN);
    const int n_blocks_total = n_blocks_ctx + n_blocks_dec;

    for (int n_block = blockIdx.x; n_block < n_blocks_total; n_block += gridDim.x) {
        // Use Is_first=false, Is_last=false since we always use Seq_parallel (atomicAdd for dQ)
        compute_dq_dk_dv_1colblock_dualkv_training<Kernel_traits, Is_causal, Is_even_K, false, false>(params, bidb, bidh, n_block);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_causal, typename Params>
inline __device__ void compute_dq_dk_dv_dualkv_training(const Params &params) {
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    if (is_even_K) {
        compute_dq_dk_dv_seqk_parallel_dualkv_training<Kernel_traits, Is_causal, /*Is_even_K=*/true>(params);
    } else {
        compute_dq_dk_dv_seqk_parallel_dualkv_training<Kernel_traits, Is_causal, /*Is_even_K=*/false>(params);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace FLASH_NAMESPACE
