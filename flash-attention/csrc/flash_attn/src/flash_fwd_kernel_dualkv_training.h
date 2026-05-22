/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 * Copyright (c) 2025, Jiading Gai — DualKV training forward kernel.
 *
 * Forked from flash_fwd_kernel.h:compute_attn_1rowblock.
 * Split KV into context (bs=1, shared) + decoded (bs=N, per-sequence).
 * No dropout, alibi, softcap, local attention, or paged KV.
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

// DualKV training forward: one Q row block, iterating over split K/V blocks.
// K/V are split into context (shared, bs=1) and decoded (per-sequence, bs=N).
// Context: kcontext_ptr with shape (context_seqlen, nh_kv, hd)
// Decoded: kdecoded_ptr with varlen packing, cu_seqlens_k_decoded gives offsets.
//
// Physical block layout:
//   blocks 0 .. n_blocks_ctx-1  : context (from kcontext_ptr)
//   blocks n_blocks_ctx .. n_blocks_ctx+n_blocks_dec-1 : decoded (from kdecoded_ptr)
//
// Logical K position for block n:
//   n < n_blocks_ctx  : n * kBlockN + col
//   n >= n_blocks_ctx : context_seqlen + (n - n_blocks_ctx) * kBlockN + col

template<typename Kernel_traits, bool Is_causal, bool Is_even_K, typename Params>
inline __device__ void compute_attn_1rowblock_dualkv_training(const Params &params, const int bidb, const int bidh, const int m_block) {

    using Element = typename Kernel_traits::Element;
    using ElementAccum = typename Kernel_traits::ElementAccum;
    using index_t = typename Kernel_traits::index_t;

    extern __shared__ char smem_[];
    const int tidx = threadIdx.x;

    constexpr int kBlockM = Kernel_traits::kBlockM;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;
    constexpr int kNWarps = Kernel_traits::kNWarps;

    // --- Sequence info ---
    // Q uses standard varlen: cu_seqlens_q[bidb] .. cu_seqlens_q[bidb+1]
    const int sum_s_q = params.cu_seqlens_q == nullptr ? 0 : params.cu_seqlens_q[bidb];
    const int actual_seqlen_q = params.cu_seqlens_q == nullptr
        ? params.seqlen_q
        : params.cu_seqlens_q[bidb + 1] - params.cu_seqlens_q[bidb];

    // Context: shared, no per-batch offset
    const int context_seqlen = params.seqlen_k_context;

    // Decoded: per-sequence, varlen
    const int sum_s_k_dec = params.cu_seqlens_k_decoded == nullptr ? 0 : params.cu_seqlens_k_decoded[bidb];
    const int actual_seqlen_k_decoded = params.cu_seqlens_k_decoded == nullptr
        ? params.seqlen_k_decoded
        : params.cu_seqlens_k_decoded[bidb + 1] - params.cu_seqlens_k_decoded[bidb];

    const int actual_seqlen_k = context_seqlen + actual_seqlen_k_decoded;

    // Physical block counts (context and decoded padded independently)
    const int n_blocks_ctx = cute::ceil_div(context_seqlen, kBlockN);
    const int n_blocks_dec = cute::ceil_div(actual_seqlen_k_decoded, kBlockN);

    if (m_block * kBlockM >= actual_seqlen_q) return;

    // n_block_max: total physical blocks we need to visit
    int n_block_max = n_blocks_ctx + n_blocks_dec;
    if (Is_causal) {
        // Under causal with bottom-right alignment, Q position (m_block+1)*kBlockM - 1
        // can attend to K positions 0..(m_block+1)*kBlockM - 1 + (seqlen_k - seqlen_q).
        // The offset (seqlen_k - seqlen_q) is 0 for standard DualKV, P for SplitQ.
        const int causal_k_needed = actual_seqlen_k - actual_seqlen_q + (m_block + 1) * kBlockM;
        int causal_n_max;
        if (causal_k_needed <= context_seqlen) {
            causal_n_max = cute::ceil_div(causal_k_needed, kBlockN);
        } else {
            causal_n_max = n_blocks_ctx + cute::ceil_div(causal_k_needed - context_seqlen, kBlockN);
        }
        n_block_max = std::min(n_block_max, causal_n_max);
    }

    // Early exit: no K blocks to process
    if (n_block_max <= 0) {
        // Write zeros to O and inf to LSE
        const index_t row_offset_o = (params.cu_seqlens_q == nullptr ? bidb * params.o_batch_stride : sum_s_q * params.o_row_stride)
            + bidh * params.o_head_stride;
        Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr) + row_offset_o),
                                make_shape(actual_seqlen_q, params.d),
                                make_stride(params.o_row_stride, _1{}));
        Tensor gO = local_tile(mO, Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, 0));

        typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
        auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
        Tensor tOgO = gmem_thr_copy_O.partition_D(gO);
        Tensor tOrO = make_tensor<Element>(shape(tOgO));
        clear(tOrO);
        Tensor cO = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));
        Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
        Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
        if (!Is_even_K) {
            #pragma unroll
            for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
        }
        FLASH_NAMESPACE::copy</*Is_even_MN=*/false, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
            gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, actual_seqlen_q - m_block * kBlockM
        );
        // Write inf to LSE
        const bool varlen_q = params.unpadded_lse;
        auto lse_offset = varlen_q ? sum_s_q : 0;
        auto gmem_ptr_lse = make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.softmax_lse_ptr) + lse_offset);
        auto lse_shape = varlen_q ? make_shape(1, params.h, params.total_q) : make_shape(params.b, params.h, params.seqlen_q);
        auto lse_stride = varlen_q
            ? make_stride(params.h * params.total_q, params.total_q, 1)
            : make_stride(params.h * params.seqlen_q, params.seqlen_q, 1);
        Tensor mLSE = make_tensor(gmem_ptr_lse, make_layout(lse_shape, lse_stride));
        auto mLSE_slice = varlen_q ? mLSE(0, bidh, _) : mLSE(bidb, bidh, _);
        Tensor gLSE = local_tile(mLSE_slice, Shape<Int<kBlockM>>{}, make_coord(m_block));
        Tensor cO2 = make_identity_tensor(make_shape(size<0>(gO), size<1>(gO)));
        Tensor tOcO2 = gmem_thr_copy_O.partition_D(cO2);
        #pragma unroll
        for (int m = 0; m < size<1>(tOgO); ++m) {
            const int row = get<0>(tOcO2(0, m, 0));
            if (row < actual_seqlen_q - m_block * kBlockM && get<1>(tOcO2(0, m, 0)) == 0) { gLSE(row) = INFINITY; }
        }
        return;
    }

    // --- Helper lambdas for dual-pointer K/V loading ---
    // Returns K pointer for physical block n_block
    const int kv_head = bidh / params.h_h_k_ratio;
    auto get_k_ptr = [&](int n_block) -> Element* {
        if (n_block < n_blocks_ctx) {
            return reinterpret_cast<Element*>(params.kcontext_ptr)
                + n_block * kBlockN * params.kcontext_row_stride
                + kv_head * params.kcontext_head_stride;
        } else {
            const int dec_block = n_block - n_blocks_ctx;
            return reinterpret_cast<Element*>(params.kdecoded_ptr)
                + sum_s_k_dec * params.kdecoded_row_stride
                + dec_block * kBlockN * params.kdecoded_row_stride
                + kv_head * params.kdecoded_head_stride;
        }
    };
    auto get_v_ptr = [&](int n_block) -> Element* {
        if (n_block < n_blocks_ctx) {
            return reinterpret_cast<Element*>(params.vcontext_ptr)
                + n_block * kBlockN * params.vcontext_row_stride
                + kv_head * params.vcontext_head_stride;
        } else {
            const int dec_block = n_block - n_blocks_ctx;
            return reinterpret_cast<Element*>(params.vdecoded_ptr)
                + sum_s_k_dec * params.vdecoded_row_stride
                + dec_block * kBlockN * params.vdecoded_row_stride
                + kv_head * params.vdecoded_head_stride;
        }
    };
    // Returns logical K position for the start of physical block n_block
    auto get_k_pos = [&](int n_block) -> int {
        if (n_block < n_blocks_ctx) {
            return n_block * kBlockN;
        } else {
            return context_seqlen + (n_block - n_blocks_ctx) * kBlockN;
        }
    };
    // Returns number of valid K rows in physical block n_block
    auto get_valid_k = [&](int n_block) -> int {
        if (n_block < n_blocks_ctx) {
            return std::min(kBlockN, context_seqlen - n_block * kBlockN);
        } else {
            const int dec_block = n_block - n_blocks_ctx;
            return std::min(kBlockN, actual_seqlen_k_decoded - dec_block * kBlockN);
        }
    };
    // Returns K row stride for physical block n_block
    auto get_k_row_stride = [&](int n_block) -> index_t {
        return n_block < n_blocks_ctx ? params.kcontext_row_stride : params.kdecoded_row_stride;
    };
    auto get_v_row_stride = [&](int n_block) -> index_t {
        return n_block < n_blocks_ctx ? params.vcontext_row_stride : params.vdecoded_row_stride;
    };

    // --- Setup Q ---
    const index_t row_offset_q = (params.cu_seqlens_q == nullptr ? bidb * params.q_batch_stride : sum_s_q * params.q_row_stride)
        + bidh * params.q_head_stride;
    Tensor mQ = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.q_ptr) + row_offset_q),
                            make_shape(actual_seqlen_q, params.d),
                            make_stride(params.q_row_stride, _1{}));
    Tensor gQ = local_tile(mQ, Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, 0));

    // --- Setup K/V as flat tiles (repointed each iteration) ---
    // Start at n_block_max - 1 (we iterate in reverse)
    const int n_start = n_block_max - 1;
    Tensor gK = make_tensor(make_gmem_ptr(get_k_ptr(n_start)),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(get_k_row_stride(n_start), _1{}));
    Tensor gV = make_tensor(make_gmem_ptr(get_v_ptr(n_start)),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(get_v_row_stride(n_start), _1{}));

    // --- Shared memory ---
    Tensor sQ = make_tensor(make_smem_ptr(reinterpret_cast<Element *>(smem_)),
                            typename Kernel_traits::SmemLayoutQ{});
    Tensor sK = make_tensor(sQ.data() + (Kernel_traits::Share_Q_K_smem ? 0 : size(sQ)),
                            typename Kernel_traits::SmemLayoutKV{});
    Tensor sV = make_tensor(sK.data() + size(sK), typename Kernel_traits::SmemLayoutKV{});
    Tensor sVt = make_tensor(sV.data(), typename Kernel_traits::SmemLayoutVtransposed{});
    Tensor sVtNoSwizzle = make_tensor(sV.data().get(), typename Kernel_traits::SmemLayoutVtransposedNoSwizzle{});

    // --- Thread-level partitioning ---
    typename Kernel_traits::GmemTiledCopyQKV gmem_tiled_copy_QKV;
    auto gmem_thr_copy_QKV = gmem_tiled_copy_QKV.get_thread_slice(tidx);

    Tensor tQgQ = gmem_thr_copy_QKV.partition_S(gQ);
    Tensor tQsQ = gmem_thr_copy_QKV.partition_D(sQ);
    Tensor tKgK = gmem_thr_copy_QKV.partition_S(gK);
    Tensor tKsK = gmem_thr_copy_QKV.partition_D(sK);
    Tensor tVgV = gmem_thr_copy_QKV.partition_S(gV);
    Tensor tVsV = gmem_thr_copy_QKV.partition_D(sV);

    typename Kernel_traits::TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tidx);
    Tensor tSrQ  = thr_mma.partition_fragment_A(sQ);
    Tensor tSrK  = thr_mma.partition_fragment_B(sK);
    Tensor tOrVt = thr_mma.partition_fragment_B(sVtNoSwizzle);

    Tensor acc_o = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kHeadDim>>{});

    // --- Copy atom retiling ---
    auto smem_tiled_copy_Q = make_tiled_copy_A(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_Q = smem_tiled_copy_Q.get_thread_slice(tidx);
    Tensor tSsQ = smem_thr_copy_Q.partition_S(sQ);

    auto smem_tiled_copy_K = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtom{}, tiled_mma);
    auto smem_thr_copy_K = smem_tiled_copy_K.get_thread_slice(tidx);
    Tensor tSsK = smem_thr_copy_K.partition_S(sK);

    auto smem_tiled_copy_V = make_tiled_copy_B(typename Kernel_traits::SmemCopyAtomTransposed{}, tiled_mma);
    auto smem_thr_copy_V = smem_tiled_copy_V.get_thread_slice(tidx);
    Tensor tOsVt = smem_thr_copy_V.partition_S(sVt);

    // --- Predicates ---
    Tensor cQ = make_identity_tensor(make_shape(size<0>(sQ), size<1>(sQ)));
    Tensor cKV = make_identity_tensor(make_shape(size<0>(sK), size<1>(sK)));
    Tensor tQcQ = gmem_thr_copy_QKV.partition_S(cQ);
    Tensor tKVcKV = gmem_thr_copy_QKV.partition_S(cKV);

    Tensor tQpQ = make_tensor<bool>(make_shape(size<2>(tQsQ)));
    Tensor tKVpKV = make_tensor<bool>(make_shape(size<2>(tKsK)));
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tQpQ); ++k) { tQpQ(k) = get<1>(tQcQ(0, 0, k)) < params.d; }
        #pragma unroll
        for (int k = 0; k < size(tKVpKV); ++k) { tKVpKV(k) = get<1>(tKVcKV(0, 0, k)) < params.d; }
    }

    // --- Prologue: load Q ---
    FLASH_NAMESPACE::copy</*Is_even_MN=*/false, Is_even_K>(gmem_tiled_copy_QKV, tQgQ, tQsQ, tQcQ, tQpQ,
                                                           actual_seqlen_q - m_block * kBlockM);
    if (Kernel_traits::Is_Q_in_regs) { cute::cp_async_fence(); }

    if (Kernel_traits::Share_Q_K_smem) {
        FLASH_NAMESPACE::cp_async_wait<0>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
        __syncthreads();
    }

    // --- Load first K block (n_block_max - 1) ---
    int n_block = n_block_max - 1;
    FLASH_NAMESPACE::copy</*Is_even_MN=*/false, Is_even_K, /*Clear_OOB_MN=*/true>(
        gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV, get_valid_k(n_block)
    );
    cute::cp_async_fence();

    if (Kernel_traits::Is_Q_in_regs && !Kernel_traits::Share_Q_K_smem) {
        FLASH_NAMESPACE::cp_async_wait<1>();
        __syncthreads();
        Tensor tSrQ_copy_view = smem_thr_copy_Q.retile_D(tSrQ);
        CUTE_STATIC_ASSERT_V(size<1>(tSsQ) == size<1>(tSrQ_copy_view));
        cute::copy(smem_tiled_copy_Q, tSsQ, tSrQ_copy_view);
    }

    clear(acc_o);

    FLASH_NAMESPACE::Softmax<2 * size<1>(acc_o)> softmax;
    FLASH_NAMESPACE::Mask<Is_causal, /*Is_local=*/false, /*Has_alibi=*/false> mask(actual_seqlen_k, actual_seqlen_q, /*window_left=*/-1, /*window_right=*/0, /*alibi_slope=*/0.0f);

    // --- Main loop: iterate K/V blocks in reverse ---
    // Always use bounds-checked loads since context/decoded blocks have independent padding.
    for (; n_block >= 0; --n_block) {
        Tensor acc_s = partition_fragment_C(tiled_mma, Shape<Int<kBlockM>, Int<kBlockN>>{});
        clear(acc_s);
        FLASH_NAMESPACE::cp_async_wait<0>();
        __syncthreads();

        // Load V for current block
        // Repoint gV to current block
        tVgV.data() = gmem_thr_copy_QKV.partition_S(
            make_tensor(make_gmem_ptr(get_v_ptr(n_block)),
                        Shape<Int<kBlockN>, Int<kHeadDim>>{},
                        make_stride(get_v_row_stride(n_block), _1{}))
        ).data();
        FLASH_NAMESPACE::copy</*Is_even_MN=*/false, Is_even_K, /*Clear_OOB_MN=*/true>(
            gmem_tiled_copy_QKV, tVgV, tVsV, tKVcKV, tKVpKV, get_valid_k(n_block)
        );
        cute::cp_async_fence();

        // Compute S = Q @ K^T
        FLASH_NAMESPACE::gemm</*A_in_regs=*/Kernel_traits::Is_Q_in_regs>(
            acc_s, tSrQ, tSrK, tSsQ, tSsK, tiled_mma, smem_tiled_copy_Q, smem_tiled_copy_K,
            smem_thr_copy_Q, smem_thr_copy_K
        );

        // Apply causal/bounds mask using LOGICAL K position
        const int k_pos_base = get_k_pos(n_block);
        if constexpr (Is_causal) {
            mask.template apply_mask<Is_causal, /*Is_even_MN=*/false>(
                acc_s, k_pos_base, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
            );
        } else {
            // For non-causal, still need to mask OOB K positions
            mask.template apply_mask</*Causal_mask=*/false, /*Is_even_MN=*/false>(
                acc_s, k_pos_base, m_block * kBlockM + (tidx / 32) * 16 + (tidx % 32) / 4, kNWarps * 16
            );
        }

        // Per-block OOB masking: context and decoded blocks have independent
        // padding, so "ghost" positions within a block (K=0 but logically valid)
        // need explicit -inf masking.
        {
            const int valid_k = get_valid_k(n_block);
            if (valid_k < kBlockN) {
                Tensor acc_s_rowcol = make_tensor(acc_s.data(), FLASH_NAMESPACE::convert_layout_acc_rowcol(acc_s.layout()));
                FLASH_NAMESPACE::apply_mask(acc_s_rowcol, valid_k, /*col_idx_offset_=*/0);
            }
        }

        FLASH_NAMESPACE::cp_async_wait<0>();
        __syncthreads();

        // Prefetch next K block
        if (n_block > 0) {
            tKgK.data() = gmem_thr_copy_QKV.partition_S(
                make_tensor(make_gmem_ptr(get_k_ptr(n_block - 1)),
                            Shape<Int<kBlockN>, Int<kHeadDim>>{},
                            make_stride(get_k_row_stride(n_block - 1), _1{}))
            ).data();
            FLASH_NAMESPACE::copy</*Is_even_MN=*/false, Is_even_K, /*Clear_OOB_MN=*/true>(
                gmem_tiled_copy_QKV, tKgK, tKsK, tKVcKV, tKVpKV, get_valid_k(n_block - 1)
            );
            cute::cp_async_fence();
        }

        // Online softmax
        const bool is_first = (n_block == n_block_max - 1);
        if (is_first) {
            softmax.template softmax_rescale_o</*Is_first=*/true, /*Check_inf=*/Is_causal>(acc_s, acc_o, params.scale_softmax_log2);
        } else {
            softmax.template softmax_rescale_o</*Is_first=*/false, /*Check_inf=*/Is_causal>(acc_s, acc_o, params.scale_softmax_log2);
        }

        // Convert P to fp16 and accumulate O += P @ V
        Tensor rP = FLASH_NAMESPACE::convert_type<Element>(acc_s);
        Tensor tOrP = make_tensor(rP.data(), FLASH_NAMESPACE::convert_layout_acc_Aregs<typename Kernel_traits::TiledMma>(rP.layout()));
        FLASH_NAMESPACE::gemm_rs(acc_o, tOrP, tOrVt, tOsVt, tiled_mma, smem_tiled_copy_V, smem_thr_copy_V);
    }

    // --- Epilogue: normalize and write O, LSE ---
    Tensor lse = softmax.template normalize_softmax_lse</*Is_dropout=*/false>(acc_o, params.scale_softmax, /*rp_dropout=*/1.0f);

    Tensor rO = FLASH_NAMESPACE::convert_type<Element>(acc_o);
    Tensor sO = make_tensor(sQ.data(), typename Kernel_traits::SmemLayoutO{});
    auto smem_tiled_copy_O = make_tiled_copy_C(typename Kernel_traits::SmemCopyAtomO{}, tiled_mma);
    auto smem_thr_copy_O = smem_tiled_copy_O.get_thread_slice(tidx);
    Tensor taccOrO = smem_thr_copy_O.retile_S(rO);
    Tensor taccOsO = smem_thr_copy_O.partition_D(sO);

    if (Kernel_traits::Share_Q_K_smem) { __syncthreads(); }

    cute::copy(smem_tiled_copy_O, taccOrO, taccOsO);

    const index_t row_offset_o = (params.cu_seqlens_q == nullptr ? bidb * params.o_batch_stride : sum_s_q * params.o_row_stride)
        + bidh * params.o_head_stride;
    Tensor mO = make_tensor(make_gmem_ptr(reinterpret_cast<Element*>(params.o_ptr) + row_offset_o),
                            make_shape(actual_seqlen_q, params.d),
                            make_stride(params.o_row_stride, _1{}));
    Tensor gO = local_tile(mO, Shape<Int<kBlockM>, Int<kHeadDim>>{}, make_coord(m_block, 0));

    // Write LSE
    const bool varlen_q = params.unpadded_lse;
    auto lse_offset = varlen_q ? sum_s_q : 0;
    auto gmem_ptr_lse = make_gmem_ptr(reinterpret_cast<ElementAccum*>(params.softmax_lse_ptr) + lse_offset);
    auto lse_shape = varlen_q ? make_shape(1, params.h, params.total_q) : make_shape(params.b, params.h, params.seqlen_q);
    auto lse_stride = varlen_q
        ? make_stride(params.h * params.total_q, params.total_q, 1)
        : make_stride(params.h * params.seqlen_q, params.seqlen_q, 1);
    Tensor mLSE = make_tensor(gmem_ptr_lse, make_layout(lse_shape, lse_stride));
    auto mLSE_slice = varlen_q ? mLSE(0, bidh, _) : mLSE(bidb, bidh, _);
    Tensor gLSE = local_tile(mLSE_slice, Shape<Int<kBlockM>>{}, make_coord(m_block));

    typename Kernel_traits::GmemTiledCopyO gmem_tiled_copy_O;
    auto gmem_thr_copy_O = gmem_tiled_copy_O.get_thread_slice(tidx);
    Tensor tOsO = gmem_thr_copy_O.partition_S(sO);
    Tensor tOgO = gmem_thr_copy_O.partition_D(gO);

    __syncthreads();

    Tensor tOrO = make_tensor<Element>(shape(tOgO));
    cute::copy(gmem_tiled_copy_O, tOsO, tOrO);

    // Write LSE values
    Tensor caccO = make_identity_tensor(Shape<Int<kBlockM>, Int<kHeadDim>>{});
    Tensor taccOcO = thr_mma.partition_C(caccO);
    static_assert(decltype(size<0>(taccOcO))::value == 4);
    Tensor taccOcO_row = logical_divide(taccOcO, Shape<_2>{})(make_coord(0, _), _, 0);
    CUTE_STATIC_ASSERT_V(size(lse) == size(taccOcO_row));
    if (get<1>(taccOcO_row(0)) == 0) {
        #pragma unroll
        for (int mi = 0; mi < size(lse); ++mi) {
            const int row = get<0>(taccOcO_row(mi));
            if (row < actual_seqlen_q - m_block * kBlockM) { gLSE(row) = lse(mi); }
        }
    }

    // Write O
    Tensor cO = make_identity_tensor(make_shape(size<0>(sO), size<1>(sO)));
    Tensor tOcO = gmem_thr_copy_O.partition_D(cO);
    Tensor tOpO = make_tensor<bool>(make_shape(size<2>(tOgO)));
    if (!Is_even_K) {
        #pragma unroll
        for (int k = 0; k < size(tOpO); ++k) { tOpO(k) = get<1>(tOcO(0, 0, k)) < params.d; }
    }
    FLASH_NAMESPACE::copy</*Is_even_MN=*/false, Is_even_K, /*Clear_OOB_MN=*/false, /*Clear_OOB_K=*/false>(
        gmem_tiled_copy_O, tOrO, tOgO, tOcO, tOpO, actual_seqlen_q - m_block * kBlockM
    );
}

////////////////////////////////////////////////////////////////////////////////////////////////////

template<typename Kernel_traits, bool Is_causal, typename Params>
inline __device__ void compute_attn_dualkv_training(const Params &params) {
    const int m_block = blockIdx.x;
    const int bidb = blockIdx.y;
    const int bidh = blockIdx.z;
    const bool is_even_K = params.d == Kernel_traits::kHeadDim;
    // We always use Is_even_MN=false for DualKV (varlen, separate padding).
    // Dispatch Is_even_K at compile time.
    if (is_even_K) {
        compute_attn_1rowblock_dualkv_training<Kernel_traits, Is_causal, /*Is_even_K=*/true>(params, bidb, bidh, m_block);
    } else {
        compute_attn_1rowblock_dualkv_training<Kernel_traits, Is_causal, /*Is_even_K=*/false>(params, bidb, bidh, m_block);
    }
}

////////////////////////////////////////////////////////////////////////////////////////////////////

}  // namespace FLASH_NAMESPACE
