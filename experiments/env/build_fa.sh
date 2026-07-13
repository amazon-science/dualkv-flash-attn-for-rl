#!/usr/bin/env bash
# Set these before running:
#   WORKDIR=<build root>  REPO=<gemma4-dev checkout>  VENV=<target venv dir>
: "${WORKDIR:?}" "${REPO:?}" "${VENV:?}"
set -x
V=${VENV}
NV=$V/lib/python3.12/site-packages/nvidia
REPO=${REPO}
source $V/bin/activate

# cu13 toolkit from the venv wheels
export CUDA_HOME=$NV/cu13
export PATH=$NV/cu13/bin:$PATH
export LIBRARY_PATH=$NV/cu13/lib:$LIBRARY_PATH          # memory: the key cmake-link fix
export LD_LIBRARY_PATH=$NV/cu13/lib:$LD_LIBRARY_PATH
export CPATH=$NV/cu13/include:$NV/cu13/include/cccl:$CPATH
export TORCH_CUDA_ARCH_LIST="9.0a"                       # H200 = sm90
export MAX_JOBS=64
export FLASH_ATTENTION_FORCE_BUILD=TRUE

# build deps
pip install -q ninja packaging wheel setuptools psutil einops

# CUTLASS at pinned commit
if [ ! -f "$REPO/flash-attention/csrc/cutlass/include/cutlass/cutlass.h" ]; then
  git clone -q https://github.com/NVIDIA/cutlass.git "$REPO/flash-attention/csrc/cutlass"
  git -C "$REPO/flash-attention/csrc/cutlass" checkout -q 7127592069c2fe01b041e174ba4345ef9b279671
fi

nvcc --version
echo "=== building flash-attn (DualKV) against cu13 ==="
time pip install --no-build-isolation -e "$REPO/flash-attention" 2>&1
echo "=== verify ==="
python -c "from flash_attn import flash_attn_dualkv_varlen_func; print(\"DUALKV OK\")"
