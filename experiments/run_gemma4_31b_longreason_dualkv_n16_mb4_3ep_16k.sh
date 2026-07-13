#!/usr/bin/env bash
# Gemma4-31B DualKV GRPO on LongReason 16K — single 8xH200, N=16, mb=4, 3 epochs, SP=1.
# Verified: base val 0.748 -> ~0.80 (peak 0.814); peak mem ~96.8GB; DualKV fwd+bwd active.
# Set WORKDIR to your gemma4-dev checkout (.venv with flash-attn(DualKV)+verl0.8), DATA_DIR, MODEL_DIR.
# ============================================================================
# REPRODUCE (single 8xH200, gemma4-dev):
#  1. Build env (torch2.11+cu130 / vLLM0.23 / verl0.8 / DualKV flash-attn 2.8.4):
#       WORKDIR=<root> REPO=<gemma4-dev checkout> VENV=<venv> \
#         bash experiments/env/build_fa.sh && bash experiments/env/install_verl.sh
#     (build_fa.sh MUST pip install cu13 nvcc wheels first: nvidia-cuda-nvcc/crt/cccl/nvrtc/nvvm==13.0.x)
#  2. Data (LongReason 16K split, exact seed/ratio used for the verified run):
#       python experiments/preprocess_longreason.py --split 16k --train_ratio 0.6 --seed 42 \
#         --local_save_dir $DATA_DIR   # -> 476 train / 318 test parquet
#  3. Model: hf download google/gemma-4-31B-it --local-dir $MODEL_DIR
#  4. Run: WORKDIR=... DATA_DIR=... MODEL_DIR=... bash experiments/run_gemma4_31b_longreason_dualkv_n16_mb4_3ep_16k.sh
#  Verified result: held-out val 0.748 -> ~0.80 (peak 0.814); peak mem ~96.8GB; ~1220s/step; DualKV fwd+bwd active.
# ============================================================================
set -x
WORKDIR=${WORKDIR:?set to gemma4-dev checkout}
V=${WORKDIR}/.venv
REPO=${WORKDIR}
PYTHON=$V/bin/python3
source $V/bin/activate
NV=$V/lib/python3.12/site-packages/nvidia
export LD_LIBRARY_PATH=$NV/cu13/lib:$LD_LIBRARY_PATH
export NCCL_CUMEM_ENABLE=0
export VLLM_LOGGING_LEVEL=WARN

$PYTHON -c "import flash_attn,verl,transformers; print(flash_attn.__version__,verl.__version__,transformers.__version__)" || exit 1

DATA=${DATA_DIR:?set to dir with train.parquet+test.parquet}
MODEL=${MODEL_DIR:-google/gemma-4-31B-it}
REWARD_FN=$REPO/experiments/reward_longreason.py
CKPT=${WORKDIR}/ckpts/gemma4-31b-longreason-n16mb4

ray stop --force >/dev/null 2>&1
$V/bin/ray start --head --port=6379 --num-gpus=8 >/dev/null 2>&1
export RAY_ADDRESS=127.0.0.1:6379

$PYTHON -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    "++data.apply_chat_template_kwargs={enable_thinking: true}" \
    data.train_files=${DATA}/train.parquet \
    data.val_files=${DATA}/test.parquet \
    data.train_batch_size=64 \
    data.max_prompt_length=16384 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    actor_rollout_ref.model.path=${MODEL} \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=32 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    +actor_rollout_ref.model.use_dualkv=True \
    actor_rollout_ref.model.use_fused_kernels=True \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=1 \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=4 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.max_model_len=18432 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.agent.num_workers=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=False \
    actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=1 \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path=${REWARD_FN} \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    "trainer.logger=[\"console\"]" \
    trainer.project_name=gemma4-31b-fresh \
    trainer.experiment_name=dualkv-n16-mb4-3ep \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=10 \
    trainer.test_freq=1 \
    trainer.total_epochs=3 \
    trainer.total_training_steps=200 \
    trainer.default_local_dir=${CKPT} \
    trainer.rollout_data_dir=${WORKDIR}/rollout_dumps \
    trainer.log_val_generations=20 \
    "$@"
