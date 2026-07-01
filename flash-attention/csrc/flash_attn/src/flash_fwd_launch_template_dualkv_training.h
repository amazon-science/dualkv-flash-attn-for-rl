/******************************************************************************
 * Copyright (c) 2024, Tri Dao.
 * Copyright (c) 2025, Jiading Gai — DualKV training forward launch template.
 ******************************************************************************/

#pragma once

#include "namespace_config.h"
#include <c10/cuda/CUDAException.h>

#include "static_switch.h"
#include "flash.h"
#include "flash_fwd_kernel_dualkv_training.h"

namespace FLASH_NAMESPACE {

// Determine if the architecture supports FLASH and define a macro to handle parameter modifiers
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

#ifndef DEFINE_FLASH_FORWARD_KERNEL
#define DEFINE_FLASH_FORWARD_KERNEL(kernelName, ...) \
template<typename Kernel_traits, __VA_ARGS__> \
__global__ void kernelName(KERNEL_PARAM_MODIFIER const Flash_fwd_params params)
#endif

DEFINE_FLASH_FORWARD_KERNEL(flash_fwd_dualkv_training_kernel, bool Is_causal, bool Is_local) {
    #if defined(ARCH_SUPPORTS_FLASH)
        FLASH_NAMESPACE::compute_attn_dualkv_training<Kernel_traits, Is_causal, Is_local>(params);
    #else
        FLASH_UNSUPPORTED_ARCH
    #endif
}

template<typename Kernel_traits, bool Is_causal, bool Is_local>
void run_flash_fwd_dualkv(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr size_t smem_size = Kernel_traits::kSmemSize;

    // Grid: (num_m_blocks, batch_size, num_heads)
    // For varlen: seqlen_q is max across batch, each CTA checks its own bounds.
    const int num_m_block = (params.seqlen_q + Kernel_traits::kBlockM - 1) / Kernel_traits::kBlockM;
    dim3 grid(num_m_block, params.b, params.h);

    auto kernel = &flash_fwd_dualkv_training_kernel<Kernel_traits, Is_causal, Is_local>;
    if (smem_size >= 48 * 1024) {
        C10_CUDA_CHECK(cudaFuncSetAttribute(
            kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, smem_size));
    }
    kernel<<<grid, Kernel_traits::kNThreads, smem_size, stream>>>(params);
    C10_CUDA_KERNEL_LAUNCH_CHECK();
}

// --- Per-headdim dispatch functions ---
// DualKV training: no dropout, no alibi, no softcap. Use same block sizes as
// FA2 non-dropout path.
//
// Sliding-window (causal SWA) support: for hd <= 256 we dispatch a compile-time
// Is_local on (window_size_left >= 0), but ONLY under Is_causal (all DualKV use
// cases are causal; right-window == 0). This instantiates Is_local=true only for
// hd in {64,96,128,192,256} and only causal, exactly as designed. hd512 never
// gets a local instantiation (global layers are full-attention; guarded in
// flash_api.cpp). FA2 convention: window_size_left == -1 means no window.

// Dispatch causal-SWA: compile Is_local=true only when Is_causal; non-causal is
// always full (Is_local=false). Keeps the instantiation matrix to the design.
// Variadic so the commas inside Flash_fwd_kernel_traits<...> are absorbed by
// __VA_ARGS__ (a parenthesized type would be an invalid template argument).
#define DUALKV_FWD_LOCAL_DISPATCH(...)                                              \
    if constexpr (Is_causal) {                                                     \
        BOOL_SWITCH(params.window_size_left >= 0, Is_local, [&] {                  \
            run_flash_fwd_dualkv<__VA_ARGS__, Is_causal, Is_local>(params, stream); \
        });                                                                        \
    } else {                                                                       \
        run_flash_fwd_dualkv<__VA_ARGS__, Is_causal, /*Is_local=*/false>(params, stream); \
    }

template<typename T, bool Is_causal>
void run_mha_fwd_dualkv_hdim64(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 64;
    DUALKV_FWD_LOCAL_DISPATCH(Flash_fwd_kernel_traits<Headdim, 128, 128, 4, false, false, T>);
}

template<typename T, bool Is_causal>
void run_mha_fwd_dualkv_hdim96(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 96;
    DUALKV_FWD_LOCAL_DISPATCH(Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>);
}

template<typename T, bool Is_causal>
void run_mha_fwd_dualkv_hdim128(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 128;
    DUALKV_FWD_LOCAL_DISPATCH(Flash_fwd_kernel_traits<Headdim, 128, 64, 4, false, false, T>);
}

template<typename T, bool Is_causal>
void run_mha_fwd_dualkv_hdim192(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 192;
    DUALKV_FWD_LOCAL_DISPATCH(Flash_fwd_kernel_traits<Headdim, 128, 64, 8, false, false, T>);
}

template<typename T, bool Is_causal>
void run_mha_fwd_dualkv_hdim256(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 256;
    DUALKV_FWD_LOCAL_DISPATCH(Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>);
}

template<typename T, bool Is_causal>
void run_mha_fwd_dualkv_hdim512(Flash_fwd_params &params, cudaStream_t stream) {
    constexpr static int Headdim = 512;
    // hd512 = Gemma4 global (full-attention) layers only; never windowed.
    run_flash_fwd_dualkv<Flash_fwd_kernel_traits<Headdim, 64, 64, 4, false, false, T>, Is_causal, /*Is_local=*/false>(params, stream);
}

}  // namespace FLASH_NAMESPACE
