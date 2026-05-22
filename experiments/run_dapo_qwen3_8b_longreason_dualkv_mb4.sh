#!/usr/bin/env bash
# Qwen3-8B DAPO on LongReason 8k — single p5 (8x H100)
# DualKV enabled, micro_batch=4
set -x

WORKDIR=${WORKDIR:?Set WORKDIR to your working directory (see README)}
PYTHON=${WORKDIR}/dualkv_venv/bin/python3
VERL_SRC=${WORKDIR}/verl_src

# Verify flash_attn DualKV build and verl are importable
$PYTHON -c "
from flash_attn import flash_attn_dualkv_varlen_func
import verl
print(f'flash_attn OK: v{__import__(\"flash_attn\").__version__}')
print(f'verl OK: v{verl.__version__}')
" || exit 1

export WANDB_API_KEY=${WANDB_API_KEY:-}

DATA_DIR=${WORKDIR}/data/longreason
MODEL_DIR=${WORKDIR}/models/Qwen3-8B
CKPT_DIR=${WORKDIR}/ckpts/dapo-qwen3-8b-longreason
SCRIPT_DIR=$(cd "$(dirname "$0")" && pwd)
REWARD_FN=${SCRIPT_DIR}/reward_longreason.py

cd ${VERL_SRC}

$PYTHON -m recipe.dapo.main_dapo \
    algorithm.adv_estimator=grpo \
    algorithm.use_kl_in_reward=False \
    algorithm.filter_groups.enable=False \
    data.train_files=${DATA_DIR}/train.parquet \
    data.val_files=${DATA_DIR}/test.parquet \
    data.train_batch_size=128 \
    data.max_prompt_length=8192 \
    data.max_response_length=2048 \
    data.filter_overlong_prompts=True \
    data.truncation=error \
    "++data.apply_chat_template_kwargs={enable_thinking: true}" \
    actor_rollout_ref.model.path=${MODEL_DIR} \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.actor.ppo_mini_batch_size=64 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.actor.use_kl_loss=False \
    actor_rollout_ref.actor.kl_loss_coef=0.0 \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.clip_ratio_low=0.2 \
    actor_rollout_ref.actor.clip_ratio_high=0.28 \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.actor.loss_agg_mode=token-mean \
    +actor_rollout_ref.actor.use_dualkv=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.6 \
    actor_rollout_ref.rollout.n=32 \
    +actor_rollout_ref.ref.use_dualkv=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=8 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    custom_reward_function.path=${REWARD_FN} \
    custom_reward_function.name=compute_score \
    trainer.balance_batch=False \
    trainer.critic_warmup=0 \
    "trainer.logger=[\"console\"]" \
    trainer.project_name=dapo-qwen3-8b-longreason \
    trainer.experiment_name=dualkv-mb4-n32 \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=1 \
    trainer.save_freq=-1 \
    trainer.test_freq=2 \
    trainer.total_epochs=30 \
    trainer.total_training_steps=25 \
    trainer.default_local_dir=${CKPT_DIR} \
    "$@"
