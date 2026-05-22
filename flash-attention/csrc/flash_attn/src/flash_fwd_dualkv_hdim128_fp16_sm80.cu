// Copyright (c) 2025, Jiading Gai — DualKV training forward kernel instantiation.
#include "namespace_config.h"
#include "flash_fwd_launch_template_dualkv_training.h"

namespace FLASH_NAMESPACE {

template<>
void run_mha_fwd_dualkv_<cutlass::half_t, 128, false>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_dualkv_hdim128<cutlass::half_t, false>(params, stream);
}

} // namespace FLASH_NAMESPACE
