// Copyright (c) 2025, Jiading Gai — DualKV training backward kernel instantiation.
#include "namespace_config.h"
#include "flash_bwd_launch_template_dualkv_training.h"

namespace FLASH_NAMESPACE {

template<>
void run_mha_bwd_dualkv_<cutlass::bfloat16_t, 64, true>(Flash_bwd_params &params, cudaStream_t stream) {
    run_mha_bwd_dualkv_hdim64<cutlass::bfloat16_t, true>(params, stream);
}

} // namespace FLASH_NAMESPACE
