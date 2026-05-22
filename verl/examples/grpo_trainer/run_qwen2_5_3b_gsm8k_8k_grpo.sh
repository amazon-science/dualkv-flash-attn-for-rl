#!/bin/bash
# GRPO training: Qwen2.5-3B-Instruct on GSM8K with 8K-token few-shot prompts
# Hardware: 8x p4d.24xlarge (64x A100-40GB)
# Purpose: Baseline profiling for DualKV shared-prompt optimization
#
# Config:
#   - 40 few-shot examples in system prompt (~8K tokens)
#   - N=20 rollouts per prompt, P=50 unique prompts per batch
#   - train_batch_size=1000, micro_batch=4/gpu
#   - max_prompt_length=8192, max_response_length=1024

set -x

export VLLM_ATTENTION_BACKEND=FLASH_ATTN
export VLLM_USE_V1=1

python3 -m verl.trainer.main_ppo \
    algorithm.adv_estimator=grpo \
    data.train_files=$HOME/data/gsm8k_8k/train.parquet \
    data.val_files=$HOME/data/gsm8k_8k/test.parquet \
    data.train_batch_size=1000 \
    data.max_prompt_length=8192 \
    data.max_response_length=1024 \
    data.filter_overlong_prompts=True \
    data.truncation='error' \
    actor_rollout_ref.model.path=Qwen/Qwen2.5-3B-Instruct \
    actor_rollout_ref.actor.optim.lr=1e-6 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.ppo_mini_batch_size=200 \
    actor_rollout_ref.actor.ppo_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.actor.use_kl_loss=True \
    actor_rollout_ref.actor.kl_loss_coef=0.001 \
    actor_rollout_ref.actor.kl_loss_type=low_var_kl \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.actor.fsdp_config.param_offload=False \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=False \
    actor_rollout_ref.rollout.log_prob_micro_batch_size_per_gpu=2 \
    actor_rollout_ref.rollout.tensor_model_parallel_size=2 \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.rollout.gpu_memory_utilization=0.4 \
    actor_rollout_ref.rollout.n=20 \
    actor_rollout_ref.rollout.enforce_eager=True \
    actor_rollout_ref.ref.log_prob_micro_batch_size_per_gpu=4 \
    actor_rollout_ref.ref.fsdp_config.param_offload=True \
    algorithm.use_kl_in_reward=False \
    trainer.critic_warmup=0 \
    trainer.logger='["console"]' \
    trainer.project_name='verl_dualkv_profiling' \
    trainer.experiment_name='qwen2.5_3b_gsm8k_8k_n20' \
    trainer.n_gpus_per_node=8 \
    trainer.nnodes=8 \
    trainer.save_freq=-1 \
    trainer.test_freq=1 \
    trainer.total_epochs=2 "$@"
