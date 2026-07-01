#include "namespace_config.h"
#include "flash_fwd_launch_template_dualkv_training.h"
namespace FLASH_NAMESPACE {
template<> void run_mha_fwd_dualkv_<cutlass::half_t, 512, false>(Flash_fwd_params &params, cudaStream_t stream) {
    run_mha_fwd_dualkv_hdim512<cutlass::half_t, false>(params, stream);
}
}
