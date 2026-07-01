#!/usr/bin/env bash
set -x
WORKDIR=${WORKDIR:-/path/to/dualkv-flash-attn-for-rl}   # repo checkout with .venv (flash-attn DualKV + verl0.8)
source ${WORKDIR}/.venv/bin/activate
cd ${WORKDIR}
export GRPO_P=1024 GRPO_R=512 GRPO_N=8
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
OUT=${WORKDIR}/nsys_out/policy_update_dualkv
# nsys profile the whole torchrun (8 ranks, single node). CUDA+NVTX trace.
nsys profile \
  --trace=cuda,nvtx \
  --sample=none --cpuctxsw=none \
  --force-overwrite=true \
  --output=${OUT} \
  ${WORKDIR}/.venv/bin/torchrun --standalone --nproc_per_node=8 \
  test_dualkv_policy_update_31b_v8.py
echo "NSYS_EXIT=$?"
ls -la ${OUT}.nsys-rep
