#include "flash.h"
#include "static_switch.h"
#include <cutlass/half.h>
#include <cutlass/bfloat16.h>

namespace FLASH_NAMESPACE {

template<typename T, int Headdim, bool Is_causal>
void run_mha_fwd_splitkv_dispatch(Flash_fwd_params &params, cudaStream_t stream) {
}

template void run_mha_fwd_splitkv_dispatch<cutlass::half_t, 32, false>(Flash_fwd_params &, cudaStream_t);
template void run_mha_fwd_splitkv_dispatch<cutlass::half_t, 32, true>(Flash_fwd_params &, cudaStream_t);
template void run_mha_fwd_splitkv_dispatch<cutlass::half_t, 64, false>(Flash_fwd_params &, cudaStream_t);
template void run_mha_fwd_splitkv_dispatch<cutlass::half_t, 64, true>(Flash_fwd_params &, cudaStream_t);
template void run_mha_fwd_splitkv_dispatch<cutlass::half_t, 96, false>(Flash_fwd_params &, cudaStream_t);
template void run_mha_fwd_splitkv_dispatch<cutlass::half_t, 96, true>(Flash_fwd_params &, cudaStream_t);
template void run_mha_fwd_splitkv_dispatch<cutlass::half_t, 128, false>(Flash_fwd_params &, cudaStream_t);
template void run_mha_fwd_splitkv_dispatch<cutlass::half_t, 128, true>(Flash_fwd_params &, cudaStream_t);
template void run_mha_fwd_splitkv_dispatch<cutlass::half_t, 192, false>(Flash_fwd_params &, cudaStream_t);
template void run_mha_fwd_splitkv_dispatch<cutlass::half_t, 192, true>(Flash_fwd_params &, cudaStream_t);
template void run_mha_fwd_splitkv_dispatch<cutlass::half_t, 256, false>(Flash_fwd_params &, cudaStream_t);
template void run_mha_fwd_splitkv_dispatch<cutlass::half_t, 256, true>(Flash_fwd_params &, cudaStream_t);
template void run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 32, false>(Flash_fwd_params &, cudaStream_t);
template void run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 32, true>(Flash_fwd_params &, cudaStream_t);
template void run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 64, false>(Flash_fwd_params &, cudaStream_t);
template void run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 64, true>(Flash_fwd_params &, cudaStream_t);
template void run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 96, false>(Flash_fwd_params &, cudaStream_t);
template void run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 96, true>(Flash_fwd_params &, cudaStream_t);
template void run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 128, false>(Flash_fwd_params &, cudaStream_t);
template void run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 128, true>(Flash_fwd_params &, cudaStream_t);
template void run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 192, false>(Flash_fwd_params &, cudaStream_t);
template void run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 192, true>(Flash_fwd_params &, cudaStream_t);
template void run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 256, false>(Flash_fwd_params &, cudaStream_t);
template void run_mha_fwd_splitkv_dispatch<cutlass::bfloat16_t, 256, true>(Flash_fwd_params &, cudaStream_t);

}  // namespace FLASH_NAMESPACE
