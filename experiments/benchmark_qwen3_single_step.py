#!/usr/bin/env python3
"""Benchmark a single training step (forward + backward) on Qwen3-8B at the
Section 3.4 / Section 4.2 config: N=8, P=16384, R=2048, bf16.

Runs on 8 H100 GPUs via torchrun with FSDP2 parameter sharding.

Measures wall-clock (fwd, bwd, fwd+bwd) and peak GPU memory for two paths:
  1. FA2 baseline: N-copy packing, one flash_attn_varlen_func call per layer.
  2. DualKV:      single-prompt packing via _dualkv_repack + _dualkv_extract_logprobs,
                  DualKV kernel per layer through _make_dualkv_flash_wrapper monkey-patch.

Both paths execute the full Qwen3-8B forward, cross-entropy loss on response
positions, and backward through all 36 layers with gradient checkpointing on.

Launch (8xH100):
    torchrun --standalone --nproc-per-node 8 benchmark_qwen3_single_step.py \\
        --model $WORKDIR/models/Qwen3-8B --path both

    # DualKV only (FA2 may OOM at N=8,P=16384 on 8xH100):
    torchrun --standalone --nproc-per-node 8 benchmark_qwen3_single_step.py \\
        --model $WORKDIR/models/Qwen3-8B --path dualkv
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
from typing import List

import torch
import torch.distributed as dist
import torch.nn.functional as F
from omegaconf import OmegaConf
from torch.distributed.device_mesh import init_device_mesh
from torch.distributed.fsdp import MixedPrecision, ShardingStrategy
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.profiler import ProfilerActivity, profile, record_function
from transformers import AutoConfig, AutoModelForCausalLM

from verl.utils.device import get_device_id, get_device_name
from verl.utils.distributed import destroy_global_process_group, initialize_global_process_group
from verl.utils.fsdp_utils import (
    CPUOffloadPolicy,
    MixedPrecisionPolicy,
    apply_fsdp2,
    fsdp2_load_full_state_dict,
    get_fsdp_wrap_policy,
    get_init_weight_context_manager,
    init_fn,
)
from verl.utils.logger.aggregate_logger import print_rank_0
from verl.utils.profiler import log_gpu_memory_usage


logging.basicConfig(
    format="%(levelname)s:%(asctime)s %(message)s",
    level=logging.INFO,
    datefmt="%Y-%m-%d %H:%M:%S",
)
logger: logging.Logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
#                                 Helpers                                     #
# --------------------------------------------------------------------------- #

@dataclass
class StepResult:
    fwd_ms: float
    bwd_ms: float
    peak_mb: float


def _cuda_events():
    return torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)


def _zero_grads(model: torch.nn.Module) -> None:
    for p in model.parameters():
        if p.grad is not None:
            p.grad = None


@contextmanager
def _peak_memory_tracker():
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    yield
    torch.cuda.synchronize()


def _peak_mb() -> float:
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def _make_random_prompt_groups(n: int, p: int, r: int, vocab_size: int,
                                device: str, seed: int = 42):
    torch.manual_seed(seed)
    prompt_ids = torch.randint(1, vocab_size, (p,), device=device)
    responses = [torch.randint(1, vocab_size, (r,), device=device) for _ in range(n)]
    return [(prompt_ids, responses)]


# --------------------------------------------------------------------------- #
#                          FA2 baseline (N-copy path)                         #
# --------------------------------------------------------------------------- #

def _fa2_build_input(prompt_groups, device: str):
    all_ids = []
    all_pos = []
    response_slices = []
    flat_offset = 0

    for prompt_ids, responses in prompt_groups:
        p = prompt_ids.shape[0]
        for resp in responses:
            r = resp.shape[0]
            all_ids.append(torch.cat([prompt_ids, resp]))
            all_pos.append(torch.arange(p + r, device=device, dtype=torch.long))
            response_slices.append((flat_offset + p, flat_offset + p + r))
            flat_offset += p + r

    input_ids = torch.cat(all_ids).unsqueeze(0)
    position_ids = torch.cat(all_pos).unsqueeze(0)
    return input_ids, position_ids, response_slices


def _fa2_loss(logits: torch.Tensor, input_ids: torch.Tensor, response_slices):
    logits = logits.squeeze(0)
    flat_ids = input_ids.squeeze(0)
    loss_terms = []
    total = 0
    for (s, e) in response_slices:
        pred_logits = logits[s - 1 : e - 1].float()
        target_ids = flat_ids[s : e]
        lp = F.log_softmax(pred_logits, dim=-1)
        tok_lp = lp.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)
        loss_terms.append(-tok_lp.sum())
        total += (e - s)
    return torch.stack(loss_terms).sum() / total


def run_fa2_step(model, prompt_groups, device: str) -> StepResult:
    from transformers.integrations import flash_attention
    orig_fn = flash_attention._flash_attention_forward

    input_ids, position_ids, response_slices = _fa2_build_input(prompt_groups, device)
    _zero_grads(model)

    with _peak_memory_tracker():
        fwd_start, fwd_end = _cuda_events()
        bwd_start, bwd_end = _cuda_events()

        fwd_start.record()
        out = model(
            input_ids=input_ids,
            attention_mask=None,
            position_ids=position_ids,
            use_cache=False,
        )
        loss = _fa2_loss(out.logits, input_ids, response_slices)
        fwd_end.record()

        bwd_start.record()
        loss.backward()
        bwd_end.record()

        torch.cuda.synchronize()

    assert flash_attention._flash_attention_forward is orig_fn, \
        "FA2 baseline was contaminated by a DualKV monkey-patch"

    return StepResult(
        fwd_ms=fwd_start.elapsed_time(fwd_end),
        bwd_ms=bwd_start.elapsed_time(bwd_end),
        peak_mb=_peak_mb(),
    )


# --------------------------------------------------------------------------- #
#                   DualKV path (via _dualkv_repack + extract)                #
# --------------------------------------------------------------------------- #

def _dualkv_build_repack_input(prompt_groups, device: str):
    all_ids = []
    all_pos = []
    cu = [0]
    prompt_group_sizes = []
    prompt_lens = []

    for prompt_ids, responses in prompt_groups:
        p = prompt_ids.shape[0]
        prompt_lens.append(p)
        prompt_group_sizes.append(len(responses))
        for resp in responses:
            r = resp.shape[0]
            all_ids.append(torch.cat([prompt_ids, resp]))
            all_pos.append(torch.arange(p + r, device=device, dtype=torch.long))
            cu.append(cu[-1] + p + r)

    input_ids_rmpad = torch.cat(all_ids).unsqueeze(0)
    position_ids_rmpad = torch.cat(all_pos).unsqueeze(0)
    cu_seqlens = torch.tensor(cu, dtype=torch.int32, device=device)
    return input_ids_rmpad, cu_seqlens, position_ids_rmpad, prompt_lens, prompt_group_sizes


def run_dualkv_step(model, prompt_groups, device: str) -> StepResult:
    from transformers.integrations import flash_attention
    from verl.models.transformers.monkey_patch import _make_dualkv_flash_wrapper
    from verl.workers.actor.dp_actor import _dualkv_repack, _dualkv_extract_logprobs

    orig_fn = flash_attention._flash_attention_forward
    flash_attention._flash_attention_forward = _make_dualkv_flash_wrapper(orig_fn)

    try:
        (input_ids_rmpad, cu_seqlens, position_ids_rmpad,
         prompt_lens, prompt_group_sizes) = _dualkv_build_repack_input(prompt_groups, device)

        ids_packed, pos_packed, dualkv_ctx, repack_info = _dualkv_repack(
            input_ids_rmpad, cu_seqlens, position_ids_rmpad,
            prompt_lens, prompt_group_sizes,
        )

        total_seqs = sum(len(resps) for _, resps in prompt_groups)
        max_resp_len = max(r.shape[0] for _, resps in prompt_groups for r in resps)

        _zero_grads(model)

        with _peak_memory_tracker():
            fwd_start, fwd_end = _cuda_events()
            bwd_start, bwd_end = _cuda_events()

            fwd_start.record()
            out = model(
                input_ids=ids_packed,
                attention_mask=None,
                position_ids=pos_packed,
                use_cache=False,
                dualkv_context=dualkv_ctx,
            )
            logits_packed = out.logits.squeeze(0)

            log_probs, _entropy, nan_mask = _dualkv_extract_logprobs(
                logits_packed, repack_info,
                response_length=max_resp_len,
                batch_size=total_seqs,
                temperature=1.0,
                calculate_entropy=False,
                compute_entropy_fn=None,
            )

            length_mask = torch.zeros_like(log_probs, dtype=torch.bool)
            seq_idx = 0
            for _, responses in prompt_groups:
                for resp in responses:
                    length_mask[seq_idx, : resp.shape[0]] = True
                    seq_idx += 1
            mask = (~nan_mask) & length_mask
            loss = -(log_probs * mask.float()).sum() / mask.sum().clamp(min=1)
            fwd_end.record()

            bwd_start.record()
            loss.backward()
            bwd_end.record()

            torch.cuda.synchronize()

        return StepResult(
            fwd_ms=fwd_start.elapsed_time(fwd_end),
            bwd_ms=bwd_start.elapsed_time(bwd_end),
            peak_mb=_peak_mb(),
        )

    finally:
        flash_attention._flash_attention_forward = orig_fn


# --------------------------------------------------------------------------- #
#                               Model loading                                 #
# --------------------------------------------------------------------------- #

_FSDP_YAML = """
model:
  fsdp_config:
    model_dtype: bf16
    wrap_policy:
      min_num_params: 0
    cpu_offload: False
    offload_params: False
"""


def _load_and_wrap_fsdp2(model_path: str, dtype: torch.dtype, device_mesh):
    """Load Qwen3-8B and shard via FSDP2 (reshard_after_forward=True).

    Follows the pattern from bedrock/debug/run_gptoss_moe_profile.py: load the
    model inside a meta-tensor init context (non-rank-0 ranks don't allocate
    full params), then apply_fsdp2 with MixedPrecisionPolicy, then
    fsdp2_load_full_state_dict materializes the sharded params.
    """
    cfg = OmegaConf.create(_FSDP_YAML)

    init_context = get_init_weight_context_manager(
        use_meta_tensor=True,
        mesh=device_mesh,
    )
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
    cpu_offload = None

    fsdp_kwargs = {
        "mesh": device_mesh,
        "mp_policy": mp_policy,
        "offload_policy": cpu_offload,
        "reshard_after_forward": True,
    }
    full_state = model.state_dict()
    apply_fsdp2(model, fsdp_kwargs, cfg.model.fsdp_config)
    fsdp2_load_full_state_dict(model, full_state, device_mesh, cpu_offload)
    model.train()
    log_gpu_memory_usage("after apply FSDP2", logger=logger)
    return model


# --------------------------------------------------------------------------- #
#                                 Driver                                      #
# --------------------------------------------------------------------------- #

def _allreduce_max(t_ms: float) -> float:
    t = torch.tensor([t_ms], dtype=torch.float64, device="cuda")
    dist.all_reduce(t, op=dist.ReduceOp.MAX)
    return t.item()


def _benchmark(runner, model, prompt_groups, device: str,
                warmup: int, measure: int, label: str) -> StepResult:
    print_rank_0(f"[{label}] warmup ({warmup})...")
    for _ in range(warmup):
        runner(model, prompt_groups, device)
        gc.collect()
        torch.cuda.empty_cache()

    print_rank_0(f"[{label}] measure ({measure})...")
    fwds, bwds, peaks = [], [], []
    for i in range(measure):
        r = runner(model, prompt_groups, device)
        fwd_max = _allreduce_max(r.fwd_ms)
        bwd_max = _allreduce_max(r.bwd_ms)
        peak_max = _allreduce_max(r.peak_mb)
        fwds.append(fwd_max); bwds.append(bwd_max); peaks.append(peak_max)
        print_rank_0(
            f"  iter {i}: fwd={fwd_max:.1f} ms  bwd={bwd_max:.1f} ms  "
            f"peak={peak_max:.0f} MB"
        )
        gc.collect()
        torch.cuda.empty_cache()

    return StepResult(
        fwd_ms=statistics.median(fwds),
        bwd_ms=statistics.median(bwds),
        peak_mb=max(peaks),
    )


def get_args():
    parser = argparse.ArgumentParser()
    default_model = os.path.join(os.environ.get("WORKDIR", ""), "models/Qwen3-8B")
    parser.add_argument("--model", default=default_model or "Qwen/Qwen3-8B")
    parser.add_argument("--rollouts", type=int, default=8, help="N: rollouts per prompt")
    parser.add_argument("--prompt-len", type=int, default=16384, help="P")
    parser.add_argument("--resp-len", type=int, default=2048, help="R")
    parser.add_argument("--warmup", type=int, default=2)
    parser.add_argument("--measure", type=int, default=3)
    parser.add_argument("--path", choices=["fa2", "dualkv", "both"], default="both")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--dtype", choices=["bf16", "fp16"], default="bf16")
    parser.add_argument("--profile", action="store_true",
                        help="enable torch.profiler with CUDA+CPU activities; "
                             "exports chrome trace per rank to --profile-dir")
    parser.add_argument("--profile-dir", default="/tmp/dualkv_profile",
                        help="directory to write chrome trace files")
    parser.add_argument("--profile-memory", action="store_true",
                        help="profile memory allocations too (larger trace)")
    return parser.parse_args()


def _profile_run(runner, model, prompt_groups, device: str,
                  warmup: int, measure: int, label: str,
                  rank: int, profile_dir: str, profile_memory: bool) -> StepResult:
    """Run warmup iterations outside the profiler, then capture measure iterations
    inside torch.profiler. Export one chrome trace per rank.
    """
    print_rank_0(f"[{label}] profile warmup ({warmup})...")
    for _ in range(warmup):
        runner(model, prompt_groups, device)
        gc.collect()
        torch.cuda.empty_cache()

    print_rank_0(f"[{label}] profile measure ({measure}) with torch.profiler...")

    os.makedirs(profile_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    hostname = socket.gethostname()
    trace_path = os.path.join(
        profile_dir,
        f"trace_{label.lower()}_{hostname}_rank{rank}_{timestamp}.json",
    )

    fwds, bwds, peaks = [], [], []
    with profile(
        activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA],
        record_shapes=True,
        profile_memory=profile_memory,
        with_stack=False,
        with_modules=True,
    ) as prof:
        for i in range(measure):
            with record_function(f"step_{i}_{label}"):
                r = runner(model, prompt_groups, device)
            fwd_max = _allreduce_max(r.fwd_ms)
            bwd_max = _allreduce_max(r.bwd_ms)
            peak_max = _allreduce_max(r.peak_mb)
            fwds.append(fwd_max); bwds.append(bwd_max); peaks.append(peak_max)
            print_rank_0(
                f"  iter {i}: fwd={fwd_max:.1f} ms  bwd={bwd_max:.1f} ms  "
                f"peak={peak_max:.0f} MB"
            )

    prof.export_chrome_trace(trace_path)
    # Rank 0 prints the aggregate table; all ranks export their own trace.
    if rank == 0:
        print(f"\n[{label}] top ops (rank 0):")
        print(prof.key_averages().table(
            sort_by="cuda_time_total", row_limit=20,
        ))
    print_rank_0(f"[{label}] trace written: {trace_path}")

    return StepResult(
        fwd_ms=statistics.median(fwds),
        bwd_ms=statistics.median(bwds),
        peak_mb=max(peaks),
    )


def main(args):
    local_rank, rank, world_size = initialize_global_process_group()
    device_name = get_device_name()
    device_mesh = init_device_mesh(
        device_type=device_name,
        mesh_shape=(world_size,),
        mesh_dim_names=("fsdp",),
    )

    device = f"cuda:{local_rank}"
    dtype = torch.bfloat16 if args.dtype == "bf16" else torch.float16

    print_rank_0(
        f"Loading {args.model} ({args.dtype}) across {world_size} ranks (FSDP)..."
    )
    config = AutoConfig.from_pretrained(args.model)
    assert config.num_hidden_layers == 36
    assert config.num_attention_heads == 32
    assert config.num_key_value_heads == 8
    assert config.hidden_size == 4096
    assert config.head_dim == 128

    model = _load_and_wrap_fsdp2(args.model, dtype, device_mesh)

    prompt_groups = _make_random_prompt_groups(
        args.rollouts, args.prompt_len, args.resp_len,
        config.vocab_size, device,
        seed=args.seed + rank,
    )

    def _run(runner, label: str) -> StepResult:
        if args.profile:
            return _profile_run(
                runner, model, prompt_groups, device,
                args.warmup, args.measure, label, rank,
                args.profile_dir, args.profile_memory,
            )
        return _benchmark(
            runner, model, prompt_groups, device,
            args.warmup, args.measure, label,
        )

    results = {}
    if args.path in ("fa2", "both"):
        results["FA2"] = _run(run_fa2_step, "FA2")
    if args.path in ("dualkv", "both"):
        results["DualKV"] = _run(run_dualkv_step, "DualKV")

    if rank == 0:
        print()
        print(f"Config: Qwen3-8B  N={args.rollouts}  P={args.prompt_len}  "
              f"R={args.resp_len}  dtype={args.dtype}  FSDP ranks={world_size}")
        print(f"{'Path':<8} {'fwd (ms)':>10} {'bwd (ms)':>10} "
              f"{'f+b (ms)':>10} {'peak (MB)':>12}")
        print("-" * 54)
        for name, r in results.items():
            print(f"{name:<8} {r.fwd_ms:>10.1f} {r.bwd_ms:>10.1f} "
                  f"{r.fwd_ms + r.bwd_ms:>10.1f} {r.peak_mb:>12.0f}")

        if "FA2" in results and "DualKV" in results:
            fa2, dk = results["FA2"], results["DualKV"]
            print()
            print(f"Speedup (FA2 / DualKV):  "
                  f"fwd={fa2.fwd_ms/dk.fwd_ms:.2f}x  "
                  f"bwd={fa2.bwd_ms/dk.bwd_ms:.2f}x  "
                  f"f+b={(fa2.fwd_ms+fa2.bwd_ms)/(dk.fwd_ms+dk.bwd_ms):.2f}x")
            print(f"Peak memory reduction:   "
                  f"{fa2.peak_mb - dk.peak_mb:.0f} MB  "
                  f"({(1 - dk.peak_mb/fa2.peak_mb) * 100:.1f}%)")


if __name__ == "__main__":
    try:
        args = get_args()
        print_rank_0(f"ARGS: {args}")
        main(args)
    finally:
        destroy_global_process_group()
