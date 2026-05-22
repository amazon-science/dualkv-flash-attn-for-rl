#!/usr/bin/env python3
"""Standalone replica of veRL's GRPO `update_policy` for memory/time calibration.

This script invokes `DataParallelPPOActor.update_policy(data)` directly with
synthetic inputs, exactly mirroring the memory footprint and wall-clock of
the veRL policy update in the §4.6 LongReason 8K apple-to-apple setup. No
rollout generation, no reward model, no reference-model forward — those live
outside `update_policy` and do not enter Table 7's measurement scope.

What this measures
------------------
One `update_policy` call, which:
  1. Splits the per-rank mini-batch into `ppo_micro_batch_size_per_gpu` micro-batches
  2. For each micro-batch: `_forward_micro_batch` → policy loss + KL loss → scaled backward
  3. `_optimizer_step`: grad clip + optimizer.step + zero_grad

What matches §4.6 by construction
---------------------------------
  - Qwen3-8B in bf16, FSDP2 full-shard, gradient checkpointing (`use_reentrant=False`)
  - AdamW(lr=1e-6, betas=[0.9, 0.95], weight_decay=0.01), real optimizer.step
  - `use_kl_loss=True`, `kl_loss_coef=0.001`, `kl_loss_type="low_var_kl"`,
    `entropy_coeff=0`, `loss_agg_mode="token-mean"`
  - Synthetic per-rank data: 8 sequences = 2 prompt groups of 4 rollouts,
    P=8192 prompt tokens, R=2048 response tokens, full-length (no pad)
  - DualKV path: `use_dualkv=True` + `_make_dualkv_flash_wrapper` monkey-patch
    installed on `transformers.integrations.flash_attention._flash_attention_forward`;
    `_dualkv_repack` and `_dualkv_extract_logprobs` fire inside `_forward_micro_batch`

What we provide synthetically (values are random, shapes match §4.6)
-------------------------------------------------------------------
  - `input_ids, attention_mask, position_ids, responses, response_mask`
  - `old_log_probs, advantages, ref_log_prob` — all fp32, correct shape; values
    do not affect memory or wall-clock, only the loss magnitude

Launch (8xH100):
    torchrun --standalone --nproc-per-node 8 benchmark_qwen3_update_policy.py \\
        --model $WORKDIR/models/Qwen3-8B --path both --profile
"""

import argparse
import gc
import logging
import os
import socket
import statistics
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import torch
import torch.distributed as dist
from omegaconf import OmegaConf
from tensordict import TensorDict
from torch.distributed.device_mesh import init_device_mesh
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoConfig, AutoModelForCausalLM

from verl import DataProto
from verl.utils.device import get_device_id, get_device_name
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group
from verl.utils.fsdp_utils import (
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    get_init_weight_context_manager,
)
from verl.utils.logger.aggregate_logger import print_rank_0
from verl.utils.profiler import log_gpu_memory_usage
from verl.workers.actor.dp_actor import DataParallelPPOActor


logging.basicConfig(
    format="%(levelname)s:%(asctime)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#                             Config B shapes                                 #
# --------------------------------------------------------------------------- #
# Per rank (§4.6 LongReason 8K apple-to-apple):
#   mb=4: ppo_micro_batch_size_per_gpu=4, grad_accum=2, 2 prompt groups of 4
#   mb=8: ppo_micro_batch_size_per_gpu=8, grad_accum=1, 1 prompt group of 8
# Both: ppo_mini_batch_size=8 per rank, P=8192, R=2048

P = 8192
R = 2048
PER_RANK_BATCH = 8

# These are set from CLI via --mb, updated at startup
MICRO_BATCH = 4
ROLLOUTS_PER_GROUP = 4
PROMPT_GROUPS_PER_RANK = 2


# --------------------------------------------------------------------------- #
#                                 Helpers                                     #
# --------------------------------------------------------------------------- #

@dataclass
class StepResult:
    wall_ms: float
    peak_mb: float


@contextmanager
def _peak_tracker():
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    yield
    torch.cuda.synchronize()


def _peak_mb() -> float:
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def _allreduce_max(val: float) -> float:
    t = torch.tensor([val], dtype=torch.float64, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


# --------------------------------------------------------------------------- #
#                          Synthetic DataProto                                #
# --------------------------------------------------------------------------- #

def _build_synthetic_data(vocab_size: int, device: str, seed: int = 42) -> DataProto:
    """Build a per-rank DataProto mirroring §4.6's shape contract.

    Layout: PER_RANK_BATCH sequences in PROMPT_GROUPS_PER_RANK groups of
    ROLLOUTS_PER_GROUP each (all within a group share a prompt).

    All fields have the shapes veRL's `update_policy` / `_forward_micro_batch`
    expects for full-length (no-pad) inputs. Values are random — they affect
    loss magnitude but not memory or wall-clock.
    """
    torch.manual_seed(seed)

    # One distinct prompt per group
    prompts = [
        torch.randint(1, vocab_size, (P,), device=device, dtype=torch.long)
        for _ in range(PROMPT_GROUPS_PER_RANK)
    ]

    responses = torch.randint(1, vocab_size, (PER_RANK_BATCH, R),
                              device=device, dtype=torch.long)

    input_ids = torch.empty(PER_RANK_BATCH, P + R, device=device, dtype=torch.long)
    for i in range(PER_RANK_BATCH):
        input_ids[i, :P] = prompts[i // ROLLOUTS_PER_GROUP]
        input_ids[i, P:] = responses[i]

    attention_mask = torch.ones(PER_RANK_BATCH, P + R,
                                device=device, dtype=torch.long)
    position_ids = torch.arange(P + R, device=device, dtype=torch.long) \
                        .unsqueeze(0).expand(PER_RANK_BATCH, -1).contiguous()
    response_mask = torch.ones(PER_RANK_BATCH, R,
                               device=device, dtype=torch.float32)

    old_log_probs = torch.randn(PER_RANK_BATCH, R,
                                device=device, dtype=torch.float32) * 0.1 - 1.0
    advantages = torch.randn(PER_RANK_BATCH, R,
                             device=device, dtype=torch.float32)
    ref_log_prob = torch.randn(PER_RANK_BATCH, R,
                               device=device, dtype=torch.float32) * 0.1 - 1.0

    tensor_dict = TensorDict({
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "position_ids": position_ids,
        "responses": responses,
        "response_mask": response_mask,
        "old_log_probs": old_log_probs,
        "advantages": advantages,
        "ref_log_prob": ref_log_prob,
    }, batch_size=[PER_RANK_BATCH])

    # uids: one distinct label per group, repeated ROLLOUTS_PER_GROUP times.
    uids = np.array(
        [f"prompt_{g}" for g in range(PROMPT_GROUPS_PER_RANK)
         for _ in range(ROLLOUTS_PER_GROUP)],
        dtype=object,
    )
    non_tensor_batch = {"uid": uids}

    meta_info = {
        "temperature": 1.0,
        "micro_batch_size": MICRO_BATCH,
    }
    return DataProto(batch=tensor_dict,
                     non_tensor_batch=non_tensor_batch,
                     meta_info=meta_info)


# --------------------------------------------------------------------------- #
#                          Actor construction                                 #
# --------------------------------------------------------------------------- #

def _build_actor_config(use_dualkv: bool):
    """Hand-built OmegaConf DictConfig with every field
    DataParallelPPOActor.__init__ / update_policy / _forward_micro_batch /
    _optimizer_step reads. Enumerated from `grep 'self\\.config\\.'`.
    """
    cfg = OmegaConf.create({
        # __init__
        "ulysses_sequence_parallel_size": 1,
        "entropy_from_logits_with_chunking": False,
        "fsdp_config": {"dtype": "bfloat16"},
        "use_dualkv": use_dualkv,
        "use_remove_padding": True,         # .get()
        "use_fused_kernels": False,         # .get()
        "use_torch_compile": False,         # .get()

        # update_policy
        "ppo_mini_batch_size": PER_RANK_BATCH,
        "ppo_micro_batch_size_per_gpu": MICRO_BATCH,
        "ppo_epochs": 1,
        "use_dynamic_bsz": False,
        "ppo_max_token_len_per_gpu": 16384,
        "calculate_entropy": False,
        "entropy_coeff": 0.0,
        "entropy_checkpointing": False,
        "loss_agg_mode": "token-mean",
        "policy_loss": {
            "loss_mode": "vanilla",
            "clip_ratio": 0.2,
            "clip_ratio_low": 0.2,
            "clip_ratio_high": 0.2,
            "clip_ratio_c": 3.0,
        },
        "use_kl_loss": True,
        "kl_loss_coef": 0.001,
        "kl_loss_type": "low_var_kl",

        # PPO vanilla loss (top-level, used by policy_loss_fn)
        "clip_ratio": 0.2,
        "clip_ratio_low": 0.2,
        "clip_ratio_high": 0.2,
        "clip_ratio_c": 3.0,
        "tau_pos": 1.0,
        "tau_neg": 1.05,
        "gamma": 1.0,
        "global_batch_info": {},

        # _optimizer_step
        "grad_clip": 1.0,
    })
    return cfg


def _load_and_wrap_fsdp2(model_path: str, device_mesh) -> torch.nn.Module:
    dtype = torch.bfloat16
    init_context = get_init_weight_context_manager(use_meta_tensor=True, mesh=device_mesh)
    log_gpu_memory_usage("Before model allocation", logger=logger)

    with init_context():
        model = AutoModelForCausalLM.from_pretrained(
            model_path,
            torch_dtype=dtype,
            attn_implementation="flash_attention_2",
        )
        model.config.use_cache = False
        model.gradient_checkpointing_enable(
            gradient_checkpointing_kwargs={"use_reentrant": False},
        )

    log_gpu_memory_usage("after model loading (pre-FSDP2)", logger=logger)

    mp_policy = MixedPrecisionPolicy(
        param_dtype=dtype,
        reduce_dtype=torch.bfloat16,
        cast_forward_inputs=True,
    )
    fsdp_kwargs = {
        "mesh": device_mesh,
        "mp_policy": mp_policy,
        "offload_policy": None,
        "reshard_after_forward": True,
    }
    fsdp_cfg = OmegaConf.create({"wrap_policy": {"min_num_params": 0}})
    full_state = model.state_dict()
    apply_fsdp2(model, fsdp_kwargs, fsdp_cfg)
    fsdp2_load_full_state_dict(model, full_state, device_mesh, None)
    model.train()
    log_gpu_memory_usage("after apply FSDP2", logger=logger)
    return model


def _install_dualkv_patch(actor_module):
    """Install the DualKV monkey-patch on HuggingFace's flash_attention integration.

    Must be called AFTER the model is loaded, since the patch wraps the
    existing `_flash_attention_forward`. Idempotent: returns the original
    function so the caller can restore it later.
    """
    from transformers.integrations import flash_attention
    from verl.models.transformers.monkey_patch import _make_dualkv_flash_wrapper

    orig = flash_attention._flash_attention_forward
    flash_attention._flash_attention_forward = _make_dualkv_flash_wrapper(orig)
    return orig


def _restore_flash_attention(orig_fn):
    from transformers.integrations import flash_attention
    flash_attention._flash_attention_forward = orig_fn


# --------------------------------------------------------------------------- #
#                                Benchmark                                    #
# --------------------------------------------------------------------------- #

def _run_update_policy(actor: DataParallelPPOActor, data: DataProto) -> StepResult:
    """One `update_policy` call, timed and memory-tracked."""
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)

    with _peak_tracker():
        start.record()
        with record_function("update_policy"):
            actor.update_policy(data)
        end.record()
        torch.cuda.synchronize()

    return StepResult(wall_ms=start.elapsed_time(end), peak_mb=_peak_mb())


def _benchmark_path(path: str, model_path: str, device_mesh,
                     args) -> StepResult:
    """Build actor for a given path (fa2 | dualkv), run warmup + measure iters."""
    print_rank_0(f"\n[{path}] building actor...")

    # Install DualKV patch if applicable, BEFORE constructing the FSDP model
    # (no — the patch is on HF's flash_attention function, unrelated to model
    # loading; install after FSDP so both paths use the same FSDP-wrapped model)
    orig_flash_fn = None
    if path == "dualkv":
        # Load a fresh model and wrap it (params are different for each path
        # only in terms of whether the DualKV patch is active; the model is
        # the same Qwen3-8B, but we build a fresh actor to keep optimizer
        # state clean).
        pass

    fsdp_model = _load_and_wrap_fsdp2(model_path, device_mesh)

    if path == "dualkv":
        orig_flash_fn = _install_dualkv_patch(fsdp_model)

    optimizer = torch.optim.AdamW(
        fsdp_model.parameters(),
        lr=1e-6, betas=(0.9, 0.95), weight_decay=0.01,
    )
    log_gpu_memory_usage("after optimizer creation", logger=logger)

    cfg = _build_actor_config(use_dualkv=(path == "dualkv"))
    actor = DataParallelPPOActor(
        config=cfg,
        actor_module=fsdp_model,
        actor_optimizer=optimizer,
    )

    data = _build_synthetic_data(
        vocab_size=151936,
        device=f"cuda:{get_device_id()}",
        seed=args.seed,
    )

    print_rank_0(f"[{path}] warmup ({args.warmup})...")
    for _ in range(args.warmup):
        _run_update_policy(actor, data)
        gc.collect()
        torch.cuda.empty_cache()

    print_rank_0(f"[{path}] measure ({args.measure})"
                 f"{' with profiler' if args.profile else ''}...")

    walls, peaks = [], []

    if args.profile:
        os.makedirs(args.profile_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        hostname = socket.gethostname()
        rank = int(os.environ["RANK"])
        trace_base = os.path.join(
            args.profile_dir,
            f"trace_{path}_{hostname}_rank{rank}_{timestamp}",
        )

        with profile(
            activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
            record_shapes=True,
            profile_memory=True,
            with_stack=True,       # required for export_memory_timeline
            with_modules=True,
        ) as prof:
            for i in range(args.measure):
                with record_function(f"iter_{i}_{path}"):
                    r = _run_update_policy(actor, data)
                walls.append(_allreduce_max(r.wall_ms))
                peaks.append(_allreduce_max(r.peak_mb))
                print_rank_0(f"  iter {i}: wall={walls[-1]:.1f} ms  "
                             f"peak={peaks[-1]:.0f} MB")

        trace_json = trace_base + ".json"
        prof.export_chrome_trace(trace_json)
        print_rank_0(f"[{path}] trace written: {trace_json}")
        try:
            mem_html = trace_base + "_memtimeline.html"
            prof.export_memory_timeline(mem_html, device=f"cuda:{get_device_id()}")
            print_rank_0(f"[{path}] memory timeline: {mem_html}")
        except Exception as e:
            print_rank_0(f"[{path}] export_memory_timeline failed: {e}")

        if rank == 0:
            print(f"\n[{path}] top CUDA ops (rank 0):")
            print(prof.key_averages().table(
                sort_by="cuda_time_total", row_limit=20,
            ))
    else:
        for i in range(args.measure):
            r = _run_update_policy(actor, data)
            walls.append(_allreduce_max(r.wall_ms))
            peaks.append(_allreduce_max(r.peak_mb))
            print_rank_0(f"  iter {i}: wall={walls[-1]:.1f} ms  "
                         f"peak={peaks[-1]:.0f} MB")
            gc.collect()
            torch.cuda.empty_cache()

    # Clean up before the next path
    if orig_flash_fn is not None:
        _restore_flash_attention(orig_flash_fn)
    del actor, fsdp_model, optimizer, data
    gc.collect()
    torch.cuda.empty_cache()

    return StepResult(wall_ms=statistics.median(walls), peak_mb=max(peaks))


# --------------------------------------------------------------------------- #
#                                   Main                                      #
# --------------------------------------------------------------------------- #

def get_args():
    parser = argparse.ArgumentParser()
    default_model = os.path.join(os.environ.get("WORKDIR", ""), "models/Qwen3-8B")
    parser.add_argument("--model", default=default_model or "Qwen/Qwen3-8B")
    parser.add_argument("--warmup", type=int, default=1)
    parser.add_argument("--measure", type=int, default=2)
    parser.add_argument("--path", choices=["fa2", "dualkv", "both"], default="both")
    parser.add_argument("--mb", type=int, choices=[4, 8], default=4,
                        help="ppo_micro_batch_size_per_gpu (§4.6 mb=4 or mb=8)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--profile-dir", default="/tmp/dualkv_update_policy_profile")
    return parser.parse_args()


def main(args):
    # Apply --mb to the module-level shape constants before anything else uses them.
    global MICRO_BATCH, ROLLOUTS_PER_GROUP, PROMPT_GROUPS_PER_RANK
    MICRO_BATCH = args.mb
    ROLLOUTS_PER_GROUP = args.mb
    PROMPT_GROUPS_PER_RANK = PER_RANK_BATCH // ROLLOUTS_PER_GROUP

    local_rank, rank, world_size = initialize_global_process_group()
    device_name = get_device_name()
    device_mesh = init_device_mesh(
        device_type=device_name,
        mesh_shape=(world_size,),
        mesh_dim_names=("fsdp",),
    )

    print_rank_0(f"Config: Qwen3-8B  per-rank mini-batch={PER_RANK_BATCH}  "
                 f"micro={MICRO_BATCH}  P={P}  R={R}  bf16  FSDP2({world_size})  "
                 f"AdamW + use_kl_loss")
    print_rank_0(f"Mirroring §4.6 LongReason 8K mb={args.mb} setup "
                 f"(grad_accum={PER_RANK_BATCH // MICRO_BATCH}).")

    config = AutoConfig.from_pretrained(args.model)
    assert config.num_hidden_layers == 36
    assert config.num_attention_heads == 32
    assert config.num_key_value_heads == 8

    results = {}
    paths = ["fa2", "dualkv"] if args.path == "both" else [args.path]
    for p in paths:
        results[p.upper()] = _benchmark_path(p, args.model, device_mesh, args)

    if rank == 0:
        print()
        print(f"{'Path':<8} {'wall (ms)':>12} {'peak (MB)':>12} {'peak (GB)':>10}")
        print("-" * 46)
        for name, r in results.items():
            print(f"{name:<8} {r.wall_ms:>12.1f} {r.peak_mb:>12.0f} "
                  f"{r.peak_mb/1024:>10.2f}")
        if "FA2" in results and "DUALKV" in results:
            fa2, dk = results["FA2"], results["DUALKV"]
            print()
            print(f"Speedup (FA2/DualKV): {fa2.wall_ms/dk.wall_ms:.2f}x")
            print(f"Peak memory reduction: {fa2.peak_mb - dk.peak_mb:.0f} MB "
                  f"({(1 - dk.peak_mb/fa2.peak_mb) * 100:.1f}%)")


if __name__ == "__main__":
    try:
        args = get_args()
        print_rank_0(f"ARGS: {args}")
        main(args)
    finally:
        destroy_global_process_group()
