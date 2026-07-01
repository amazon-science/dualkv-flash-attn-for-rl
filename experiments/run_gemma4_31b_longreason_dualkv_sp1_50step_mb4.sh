#!/usr/bin/env bash
# Gemma4-31B (dense) GRPO on LongReason 8k — 2x p5 (16x H100), DualKV enabled.
# Global layers (hd512) -> DualKV hdim512; sliding layers (hd256,W) -> FA2 windowed.
set -x
WORKDIR=${WORKDIR:-/path/to/dualkv-flash-attn-for-rl}   # repo checkout; .venv must have flash-attn(DualKV)+verl0.8
PYTHON=${WORKDIR}/.venv/bin/python3
source ${WORKDIR}/.venv/bin/activate
export RAY_ADDRESS=${RAY_ADDRESS:?set RAY_ADDRESS to <head-ip>:6379}   # 31B needs 2 nodes / 16x H100

$PYTHON -c "import flash_attn,verl,transformers; print(flash_attn.__version__,verl.__version__,transformers.__version__)" || exit 1

DATA=${DATA_DIR:?set DATA_DIR to a dir with train.parquet + test.parquet}
MODEL=${MODEL_DIR:-google/gemma-4-31b-it}   # HF id or local path to Gemma-4-31B
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REWARD_FN=${SCRIPT_DIR}/reward_longreason.py
CKPT=${WORKDIR}/ckpts/gemma4-31b-longreason-grpo

$PYTHON -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=${DATA}/train.parquet \
    data.val_files=${DATA}/test.parquet \
    data.train_batch_size=64 \
    data.max_prompt_length=8192 \
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
    actor_rollout_ref.actor.fsdp_config.param_offload=True \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=True \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=1 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.9 \
    actor_rollout_ref.rollout.max_model_len=12288 \
    actor_rollout_ref.rollout.n=16 \
    actor_rollout_ref.rollout.agent.num_workers=2 \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    actor_rollout_ref.ref.fsdp_config.ulysses_sequence_parallel_size=1 \
    algorithm.use_kl_in_reward=False \
    custom_reward_function.path=${REWARD_FN} \
    custom_reward_function.name=compute_score \
    trainer.critic_warmup=0 \
    "trainer.logger=[\"console\"]" \
    trainer.project_name=gemma4-31b-longreason \
    trainer.experiment_name=dualkv-2p5 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=2 \
    trainer.save_freq=1 \
    trainer.test_freq=10 \
    trainer.resume_mode=auto \
    trainer.total_epochs=1 \
    trainer.total_training_steps=50 \
    trainer.default_local_dir=${CKPT} \
    "$@"
