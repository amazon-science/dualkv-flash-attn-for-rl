#include "flash.h"
#include <cutlass/half.h>
#include <cutlass/bfloat16.h>

namespace FLASH_NAMESPACE {

template<typename T, int Headdim, bool Is_causal>
void run_mha_fwd_dualkv_(Flash_fwd_params &params, cudaStream_t stream) {
}

template<typename T, int Headdim, bool Is_causal>
void run_mha_bwd_dualkv_(Flash_bwd_params &params, cudaStream_t stream) {
}

// hdim32 stubs — not supported by DualKV kernels (fp16 and bf16)
template void run_mha_fwd_dualkv_<cutlass::half_t, 32, false>(Flash_fwd_params &, cudaStream_t);
template void run_mha_fwd_dualkv_<cutlass::half_t, 32, true>(Flash_fwd_params &, cudaStream_t);
template void run_mha_bwd_dualkv_<cutlass::half_t, 32, false>(Flash_bwd_params &, cudaStream_t);
template void run_mha_bwd_dualkv_<cutlass::half_t, 32, true>(Flash_bwd_params &, cudaStream_t);
template void run_mha_fwd_dualkv_<cutlass::bfloat16_t, 32, false>(Flash_fwd_params &, cudaStream_t);
template void run_mha_fwd_dualkv_<cutlass::bfloat16_t, 32, true>(Flash_fwd_params &, cudaStream_t);
template void run_mha_bwd_dualkv_<cutlass::bfloat16_t, 32, false>(Flash_bwd_params &, cudaStream_t);
template void run_mha_bwd_dualkv_<cutlass::bfloat16_t, 32, true>(Flash_bwd_params &, cudaStream_t);

}  // namespace FLASH_NAMESPACE
