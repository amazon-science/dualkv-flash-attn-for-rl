/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 * Copyright (c) 2025, Jiading Gai — DualKV training backward launch template.
 ******************************************************************************/

#pragma once

#include "namespace_config.h"
#include <c10/cuda/CUDAException.h>

#include "static_switch.h"
#include "hardware_info.h"
#include "flash.h"
#include "flash_bwd_launch_template.h"
#include "flash_bwd_kernel_dualkv_training.h"

namespace FLASH_NAMESPACE {

#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
#ifndef ARCH_SUPPORTS_FLASH
#define ARCH_SUPPORTS_FLASH
#endif
#ifndef KERNEL_PARAM_MODIFIER
#define KERNEL_PARAM_MODIFIER __grid_constant__
#endif
#else
#ifndef KERNEL_PARAM_MODIFIER
#define KERNEL_PARAM_MODIFIER
#endif
#endif

#ifndef FLASH_UNSUPPORTED_ARCH
#define FLASH_UNSUPPORTED_ARCH printf("FATAL: FlashAttention requires building with sm version sm80-sm90, but was built for < 8.0!");
#endif

#ifndef DEFINE_FLASH_BACKWARD_KERNEL
#define DEFINE_FLASH_BACKWARD_KERNEL(kernelName, ...) \
template<typename Kernel_traits, __VA_ARGS__> \
__global__ void kernelName(KERNEL_PARAM_MODIFIER const Flash_bwd_params params)
#endif

DEFINE_FLASH_BACKWARD_KERNEL(flash_bwd_dq_dk_dv_dualkv_training_kernel, bool Is_causal) {
    #if defined(ARCH_SUPPORTS_FLASH)
        FLASH_NAMESPACE::compute_dq_dk_dv_dualkv_training<Kernel_traits, Is_causal>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

// Simple kernel to convert fp32 dK_ctx/dV_ctx accum buffers to fp16 output.
// Grid: (n_blocks_ctx, 1, num_heads)
// Each block handles one kBlockN chunk of context_seqlen for one head.
template<typename Kernel_traits>
__global__ void flash_bwd_convert_dkv_context_kernel(const Flash_bwd_params params) {
    using Element = typename Kernel_traits::Element;
    constexpr int kBlockN = Kernel_traits::kBlockN;
    constexpr int kHeadDim = Kernel_traits::kHeadDim;

    const int n_block = blockIdx.x;
    const int bidh = blockIdx.z;
    const int tidx = threadIdx.x;

    const int context_seqlen = params.seqlen_k_context;
    if (n_block * kBlockN >= context_seqlen) return;
    const int valid_n = min(kBlockN, context_seqlen - n_block * kBlockN);

    // fp32 accum source
    const auto dk_accum_offset = n_block * kBlockN * params.dk_context_accum_row_stride
        + bidh * params.dk_context_accum_head_stride;
    const auto dv_accum_offset = n_block * kBlockN * params.dv_context_accum_row_stride
        + bidh * params.dv_context_accum_head_stride;
    const float *dk_accum = reinterpret_cast<const float *>(params.dk_context_accum_ptr) + dk_accum_offset;
    const float *dv_accum = reinterpret_cast<const float *>(params.dv_context_accum_ptr) + dv_accum_offset;

    // fp16 output destination
    const auto dk_out_offset = n_block * kBlockN * params.dk_context_row_stride
        + bidh * params.dk_context_head_stride;
    const auto dv_out_offset = n_block * kBlockN * params.dv_context_row_stride
        + bidh * params.dv_context_head_stride;
    Element *dk_out = reinterpret_cast<Element *>(params.dk_context_ptr) + dk_out_offset;
    Element *dv_out = reinterpret_cast<Element *>(params.dv_context_ptr) + dv_out_offset;

    const int total_elems = kBlockN * kHeadDim;
    const int nthreads = Kernel_traits::kNThreads;
    for (int idx = tidx; idx < total_elems; idx += nthreads) {
        const int row = idx / kHeadDim;
        const int col = idx % kHeadDim;
        if (row < valid_n && col < params.d) {
            dk_out[row * params.dk_context_row_stride + col] =
                static_cast<Element>(dk_accum[row * params.dk_context_accum_row_stride + col]);
            dv_out[row * params.dv_context_row_stride + col] =
                static_cast<Element>(dv_accum[row * params.dv_context_accum_row_stride + col]);
        }
    }
}

template<typename Kernel_traits, bool Is_causal>
void run_flash_bwd_dualkv(Flash_bwd_params &params, cudaStream_t stream) {
#ifndef FLASHATTENTION_DISABLE_BACKWARD
    // Step 1: Compute D = rowsum(dO * O) — reuse standard preprocess kernel
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid_m(num_m_block, params.b, params.h);

    // Always clear dQ_accum first (needed for atomicAdd)
    flash_bwd_dot_do_o_kernel<true, Kernel_traits><<<grid_m, Kernel_traits::kNThreads, 0, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Step 2: Main backward kernel
    // Grid: (n_blocks_total_max, batch, heads)
    // n_blocks_total_max = ceil(context_seqlen / kBlockN) + ceil(max_seqlen_k_decoded / kBlockN)
    const int n_blocks_ctx = (params.seqlen_k_context + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN;
    const int n_blocks_dec = (params.seqlen_k_decoded + Kernel_traits::kBlockN - 1) / Kernel_traits::kBlockN;
    int gridDimx = n_blocks_ctx + n_blocks_dec;
    if (params.deterministic) {
        int num_sm = get_num_sm(get_current_device());
        gridDimx = (num_sm + params.b * params.h - 1) / (params.b * params.h);
    }
    dim3 grid_n(gridDimx, params.b, params.h);

    constexpr int smem_size = Kernel_traits::kSmemSize1colblock;
    auto kernel = &flash_bwd_dq_dk_dv_dualkv_training_kernel<Kernel_traits, Is_causal>;
    if (smem_size >= 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    kernel<<<grid_n, Kernel_traits::kNThreads, smem_size, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Step 3: Convert dQ from fp32 accumulator to fp16
    auto kernel_dq = &flash_bwd_convert_dq_kernel<Kernel_traits>;
    if (Kernel_traits::kSmemdQSize >= 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel_dq, cudaFuncAttributeMaxDynamicSharedMemorySize, Kernel_traits::kSmemdQSize));
    }
    kernel_dq<<<grid_m, Kernel_traits::kNThreads, Kernel_traits::kSmemdQSize, stream>>>(params, !params.deterministic ? 1 : gridDimx);
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    // Step 4: Convert dK_ctx/dV_ctx from fp32 accumulator to fp16
    if (params.dk_context_accum_ptr != nullptr) {
        dim3 grid_ctx(n_blocks_ctx, 1, params.h);
        flash_bwd_convert_dkv_context_kernel<Kernel_traits><<<grid_ctx, Kernel_traits::kNThreads, 0, stream>>>(params);
        C10_CUDA_KERNEL_LAUNCH_CHECK();
    }
#endif
}

// --- Per-headdim dispatch ---
// DualKV training bwd: no dropout, no alibi, no softcap, no local attention.
// Use same block sizes as FA2 non-dropout backward.

template<typename T, bool Is_causal>
void run_mha_bwd_dualkv_hdim64(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    run_flash_bwd_dualkv<Flash_bwd_kernel_traits<Headdim, 128, 128, 8, 4, 4, 4, false, false, T>, Is_causal>(params, stream);
}

template<typename T, bool Is_causal>
void run_mha_bwd_dualkv_hdim96(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 96;
    run_flash_bwd_dualkv<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 4, false, false, T>, Is_causal>(params, stream);
}

template<typename T, bool Is_causal>
void run_mha_bwd_dualkv_hdim128(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    run_flash_bwd_dualkv<Flash_bwd_kernel_traits<Headdim, 64, 128, 8, 2, 4, 4, false, false, T>, Is_causal>(params, stream);
}

template<typename T, bool Is_causal>
void run_mha_bwd_dualkv_hdim192(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 192;
    run_flash_bwd_dualkv<Flash_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 2, 4, false, false, T>, Is_causal>(params, stream);
}

template<typename T, bool Is_causal>
void run_mha_bwd_dualkv_hdim256(Flash_bwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 256;
    int device;
    cudaGetDevice(&device);
    int max_smem_per_block;
    cudaError status_ = cudaDeviceGetAttribute(
        &max_smem_per_block, cudaDevAttrMaxSharedMemoryPerBlockOptin, device);
    if (status_ != cudaSuccess) {
        C10_CUDA_CHECK(status_);
    }
    if (max_smem_per_block >= 176 * 1024) {  // H100
        run_flash_bwd_dualkv<Flash_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, false, false, T>, Is_causal>(params, stream);
    } else if (max_smem_per_block >= 144 * 1024) {  // A100
        run_flash_bwd_dualkv<Flash_bwd_kernel_traits<Headdim, 64, 64, 8, 4, 2, 2, false, true, T>, Is_causal>(params, stream);
    } else {  // sm86/sm89 (99 KB max smem): V in regs, no double buffer
        run_flash_bwd_dualkv<Flash_bwd_kernel_traits<Headdim, 64, 32, 8, 4, 1, 2, true, true, T>, Is_causal>(params, stream);
    }
}

}  // namespace FLASH_NAMESPACE
