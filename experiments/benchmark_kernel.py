#!/usr/bin/env python3
"""Kernel-level benchmarks for DualKV vs FA2 vs Prefix Grouper.

Reproduces:
  - Table 1: Isolated attention kernel fwd/bwd timing (varying N, P)
  - Table 2: Single transformer layer fwd+bwd (DualKV vs PG vs FA2, varying P, mb)

Usage (single GPU):
    # Table 1: kernel-level sweep
    python benchmark_kernel.py --experiment table1

    # Table 2: single-layer with Prefix Grouper comparison
    python benchmark_kernel.py --experiment table2

    # Both
    python benchmark_kernel.py --experiment both
"""

import argparse
import gc
import time
from dataclasses import dataclass
from typing import Optional

import torch
import torch.nn.functional as F


@dataclass
class BenchResult:
    fwd_ms: float
    bwd_ms: float
    peak_mb: float


def _sync():
    torch.cuda.synchronize()


def _peak_mb():
    return torch.cuda.max_memory_allocated() / (1024 * 1024)


def _reset_memory():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


# --------------------------------------------------------------------------- #
#  Table 1: Isolated kernel benchmarks (flash_attn_varlen_func calls)
# --------------------------------------------------------------------------- #

def _build_fa2_varlen_inputs(N, P, R, H, H_k, d, dtype, device):
    """Build N-copy packed inputs for standard FA2 varlen."""
    seqlen = P + R
    total_tokens = N * seqlen

    q = torch.randn(total_tokens, H, d, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(total_tokens, H_k, d, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(total_tokens, H_k, d, dtype=dtype, device=device, requires_grad=True)

    cu_seqlens = torch.arange(0, (N + 1) * seqlen, seqlen, dtype=torch.int32, device=device)
    max_seqlen = seqlen

    return q, k, v, cu_seqlens, max_seqlen


def _build_dualkv_inputs(N, P, R, H, H_k, d, dtype, device):
    """Build DualKV packed inputs: shared prompt once + N responses.

    Q layout: [prompt+resp0 (P+R), resp1 (R), resp2 (R), ..., respN-1 (R)]
    Total Q = P + N*R (vs FA2's N*(P+R))
    Context KV: P tokens shared across all sequences
    Decoded KV: R tokens per sequence (N*R total)
    """
    total_q_tokens = P + N * R

    q = torch.randn(total_q_tokens, H, d, dtype=dtype, device=device, requires_grad=True)
    k_context = torch.randn(P, H_k, d, dtype=dtype, device=device, requires_grad=True)
    v_context = torch.randn(P, H_k, d, dtype=dtype, device=device, requires_grad=True)
    k_decoded = torch.randn(N * R, H_k, d, dtype=dtype, device=device, requires_grad=True)
    v_decoded = torch.randn(N * R, H_k, d, dtype=dtype, device=device, requires_grad=True)

    # First sequence: P+R (prompt + first response), rest: R each
    cu_seqlens_q = torch.zeros(N + 1, dtype=torch.int32, device=device)
    cu_seqlens_q[1] = P + R
    for i in range(2, N + 1):
        cu_seqlens_q[i] = cu_seqlens_q[i - 1] + R

    cu_seqlens_k_decoded = torch.arange(0, (N + 1) * R, R, dtype=torch.int32, device=device)

    max_seqlen_q = P + R  # longest sequence in Q
    return (q, k_context, v_context, k_decoded, v_decoded,
            cu_seqlens_q, cu_seqlens_k_decoded, max_seqlen_q, P, R)


def bench_fa2_kernel(N, P, R, H, H_k, d, dtype, device, warmup=3, measure=5):
    from flash_attn import flash_attn_varlen_func

    q, k, v, cu_seqlens, max_seqlen = _build_fa2_varlen_inputs(N, P, R, H, H_k, d, dtype, device)

    for _ in range(warmup):
        out = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, causal=True)
        loss = out.sum()
        loss.backward()
        q.grad = k.grad = v.grad = None
        _sync()

    _reset_memory()
    fwd_times, bwd_times = [], []

    for _ in range(measure):
        _sync()
        t0 = time.perf_counter()
        out = flash_attn_varlen_func(q, k, v, cu_seqlens, cu_seqlens, max_seqlen, max_seqlen, causal=True)
        _sync()
        t1 = time.perf_counter()
        loss = out.sum()
        loss.backward()
        _sync()
        t2 = time.perf_counter()

        fwd_times.append((t1 - t0) * 1000)
        bwd_times.append((t2 - t1) * 1000)
        q.grad = k.grad = v.grad = None

    peak = _peak_mb()
    fwd_ms = sorted(fwd_times)[len(fwd_times) // 2]
    bwd_ms = sorted(bwd_times)[len(bwd_times) // 2]
    return BenchResult(fwd_ms=fwd_ms, bwd_ms=bwd_ms, peak_mb=peak)


def bench_dualkv_kernel(N, P, R, H, H_k, d, dtype, device, warmup=3, measure=5):
    from flash_attn import flash_attn_dualkv_varlen_func

    (q, k_ctx, v_ctx, k_dec, v_dec,
     cu_q, cu_k_dec, max_seqlen_q, ctx_len, dec_len) = _build_dualkv_inputs(
        N, P, R, H, H_k, d, dtype, device
    )

    for _ in range(warmup):
        out = flash_attn_dualkv_varlen_func(
            q, k_ctx, v_ctx, k_dec, v_dec,
            cu_q, cu_k_dec,
            max_seqlen_q, ctx_len, dec_len,
            causal=True,
        )
        loss = out.sum()
        loss.backward()
        q.grad = k_ctx.grad = v_ctx.grad = k_dec.grad = v_dec.grad = None
        _sync()

    _reset_memory()
    fwd_times, bwd_times = [], []

    for _ in range(measure):
        _sync()
        t0 = time.perf_counter()
        out = flash_attn_dualkv_varlen_func(
            q, k_ctx, v_ctx, k_dec, v_dec,
            cu_q, cu_k_dec,
            max_seqlen_q, ctx_len, dec_len,
            causal=True,
        )
        _sync()
        t1 = time.perf_counter()
        loss = out.sum()
        loss.backward()
        _sync()
        t2 = time.perf_counter()

        fwd_times.append((t1 - t0) * 1000)
        bwd_times.append((t2 - t1) * 1000)
        q.grad = k_ctx.grad = v_ctx.grad = k_dec.grad = v_dec.grad = None

    peak = _peak_mb()
    fwd_ms = sorted(fwd_times)[len(fwd_times) // 2]
    bwd_ms = sorted(bwd_times)[len(bwd_times) // 2]
    return BenchResult(fwd_ms=fwd_ms, bwd_ms=bwd_ms, peak_mb=peak)


def run_table1(device, dtype):
    """Table 1: kernel-level sweep over N, P."""
    H, H_k, d = 32, 8, 128  # Qwen3-8B dims
    R = 2048

    configs = [
        (28, 4096),
        (28, 16384),
        (16, 32768),
        (28, 32768),
        (16, 65536),
    ]

    print(f"\n{'='*80}")
    print(f"Table 1: Kernel-level benchmarks (H={H}, H_k={H_k}, d={d}, R={R}, {dtype})")
    print(f"{'='*80}")
    print(f"{'N':>4} {'P':>6} | {'FA2 fwd':>8} {'FA2 bwd':>8} {'FA2 f+b':>8} | "
          f"{'DK fwd':>8} {'DK bwd':>8} {'DK f+b':>8} | "
          f"{'fwd':>5} {'bwd':>5} {'f+b':>5} | {'Mem↓':>5}")
    print("-" * 100)

    for N, P in configs:
        _reset_memory()

        try:
            fa2 = bench_fa2_kernel(N, P, R, H, H_k, d, dtype, device)
            fa2_str = f"{fa2.fwd_ms:8.1f} {fa2.bwd_ms:8.1f} {fa2.fwd_ms+fa2.bwd_ms:8.1f}"
        except torch.cuda.OutOfMemoryError:
            fa2 = None
            fa2_str = f"{'OOM':>8} {'OOM':>8} {'OOM':>8}"
            _reset_memory()

        try:
            dk = bench_dualkv_kernel(N, P, R, H, H_k, d, dtype, device)
            dk_str = f"{dk.fwd_ms:8.1f} {dk.bwd_ms:8.1f} {dk.fwd_ms+dk.bwd_ms:8.1f}"
        except (torch.cuda.OutOfMemoryError, torch.AcceleratorError, RuntimeError) as e:
            dk = None
            dk_str = f"{'OOM':>8} {'OOM':>8} {'OOM':>8}"
            _reset_memory()

        if fa2 and dk:
            spd_fwd = f"{fa2.fwd_ms/dk.fwd_ms:.2f}x"
            spd_bwd = f"{fa2.bwd_ms/dk.bwd_ms:.2f}x"
            spd_fb = f"{(fa2.fwd_ms+fa2.bwd_ms)/(dk.fwd_ms+dk.bwd_ms):.2f}x"
            mem_red = f"{(1 - dk.peak_mb/fa2.peak_mb)*100:.0f}%"
        elif dk and not fa2:
            spd_fwd = spd_bwd = spd_fb = "∞"
            mem_red = "---"
        else:
            spd_fwd = spd_bwd = spd_fb = "---"
            mem_red = "---"

        print(f"{N:4d} {P:6d} | {fa2_str} | {dk_str} | "
              f"{spd_fwd:>5} {spd_bwd:>5} {spd_fb:>5} | {mem_red:>5}")
        _reset_memory()


# --------------------------------------------------------------------------- #
#  Table 2: Single-layer benchmark (DualKV vs PG vs FA2)
# --------------------------------------------------------------------------- #

def _build_single_layer_model(H, H_k, d, num_layers=1, dtype=torch.float16, device="cuda"):
    """Build a minimal single-layer Qwen3-like transformer for benchmarking."""
    from transformers import AutoConfig, AutoModelForCausalLM

    config = AutoConfig.from_pretrained("Qwen/Qwen3-8B")
    config.num_hidden_layers = num_layers
    model = AutoModelForCausalLM.from_config(config, torch_dtype=dtype, attn_implementation="flash_attention_2")
    model = model.to(device)
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
    model.train()
    return model


def bench_single_layer_fa2(model, N, P, R, dtype, device, warmup=3, measure=5):
    """FA2 baseline: N-copy packed single layer."""
    seqlen = P + R
    vocab_size = model.config.vocab_size

    input_ids = torch.randint(1, vocab_size, (1, N * seqlen), device=device)
    position_ids = torch.cat([torch.arange(seqlen, device=device) for _ in range(N)]).unsqueeze(0)

    for _ in range(warmup):
        out = model(input_ids=input_ids, position_ids=position_ids, use_cache=False)
        out.logits.sum().backward()
        model.zero_grad(set_to_none=True)
        _sync()

    _reset_memory()
    times = []
    for _ in range(measure):
        _sync()
        t0 = time.perf_counter()
        out = model(input_ids=input_ids, position_ids=position_ids, use_cache=False)
        out.logits.sum().backward()
        _sync()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
        model.zero_grad(set_to_none=True)

    peak = _peak_mb()
    ms = sorted(times)[len(times) // 2]
    return BenchResult(fwd_ms=ms, bwd_ms=0, peak_mb=peak)


def bench_single_layer_dualkv(model, N, P, R, dtype, device, warmup=3, measure=5):
    """DualKV: single-prompt packed single layer."""
    from transformers.integrations import flash_attention
    from verl.models.transformers.monkey_patch import _make_dualkv_flash_wrapper
    from verl.workers.actor.dp_actor import _dualkv_repack

    orig_fn = flash_attention._flash_attention_forward
    flash_attention._flash_attention_forward = _make_dualkv_flash_wrapper(orig_fn)

    try:
        vocab_size = model.config.vocab_size
        seqlen = P + R

        prompt_ids = torch.randint(1, vocab_size, (P,), device=device)
        all_ids = []
        for _ in range(N):
            resp = torch.randint(1, vocab_size, (R,), device=device)
            all_ids.append(torch.cat([prompt_ids, resp]))

        input_ids_rmpad = torch.cat(all_ids).unsqueeze(0)
        position_ids_rmpad = torch.cat([torch.arange(seqlen, device=device) for _ in range(N)]).unsqueeze(0)
        cu_seqlens = torch.arange(0, (N + 1) * seqlen, seqlen, dtype=torch.int32, device=device)
        prompt_lens = [P]
        prompt_group_sizes = [N]

        ids_packed, pos_packed, dualkv_ctx, repack_info = _dualkv_repack(
            input_ids_rmpad, cu_seqlens, position_ids_rmpad,
            prompt_lens, prompt_group_sizes,
        )

        for _ in range(warmup):
            out = model(input_ids=ids_packed, position_ids=pos_packed,
                        use_cache=False, dualkv_context=dualkv_ctx)
            out.logits.sum().backward()
            model.zero_grad(set_to_none=True)
            _sync()

        _reset_memory()
        times = []
        for _ in range(measure):
            _sync()
            t0 = time.perf_counter()
            out = model(input_ids=ids_packed, position_ids=pos_packed,
                        use_cache=False, dualkv_context=dualkv_ctx)
            out.logits.sum().backward()
            _sync()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
            model.zero_grad(set_to_none=True)

        peak = _peak_mb()
        ms = sorted(times)[len(times) // 2]
        return BenchResult(fwd_ms=ms, bwd_ms=0, peak_mb=peak)

    finally:
        flash_attention._flash_attention_forward = orig_fn


def bench_single_layer_pg(model, N, P, R, dtype, device, warmup=3, measure=5):
    """Prefix Grouper: shared-prefix attention optimization."""
    try:
        from prefix_grouper import PrefixGrouper
    except ImportError:
        return None

    vocab_size = model.config.vocab_size
    seqlen = P + R

    input_ids = torch.randint(1, vocab_size, (1, N * seqlen), device=device)
    position_ids = torch.cat([torch.arange(seqlen, device=device) for _ in range(N)]).unsqueeze(0)

    pg = PrefixGrouper(prefix_length=P, num_sequences=N)

    for _ in range(warmup):
        out = model(input_ids=input_ids, position_ids=position_ids,
                    use_cache=False, prefix_grouper=pg)
        out.logits.sum().backward()
        model.zero_grad(set_to_none=True)
        _sync()

    _reset_memory()
    times = []
    for _ in range(measure):
        _sync()
        t0 = time.perf_counter()
        out = model(input_ids=input_ids, position_ids=position_ids,
                    use_cache=False, prefix_grouper=pg)
        out.logits.sum().backward()
        _sync()
        t1 = time.perf_counter()
        times.append((t1 - t0) * 1000)
        model.zero_grad(set_to_none=True)

    peak = _peak_mb()
    ms = sorted(times)[len(times) // 2]
    return BenchResult(fwd_ms=ms, bwd_ms=0, peak_mb=peak)


def run_table2(device, dtype):
    """Table 2: Single-layer DualKV vs PG vs FA2."""
    R = 2048
    N = 32  # fixed rollout factor for this benchmark

    configs = [
        (5120, 32),
        (8192, 16),
        (16384, 8),
        (32768, 4),
        (65536, 4),
        (131072, 8),
    ]

    print(f"\n{'='*80}")
    print(f"Table 2: Single-layer fwd+bwd (Qwen3-8B dims, N={N}, R={R}, {dtype})")
    print(f"{'='*80}")
    print(f"{'P':>6} {'mb':>3} | {'FA2 (ms)':>9} {'DualKV':>9} {'PG':>9} | "
          f"{'DK/FA2':>7} {'DK/PG':>7} | {'DK mem':>7} {'PG mem':>7}")
    print("-" * 85)

    model = _build_single_layer_model(32, 8, 128, num_layers=1, dtype=dtype, device=device)

    for P, mb in configs:
        _reset_memory()

        try:
            fa2_r = bench_single_layer_fa2(model, mb, P, R, dtype, device)
            fa2_str = f"{fa2_r.fwd_ms:9.0f}"
        except torch.cuda.OutOfMemoryError:
            fa2_r = None
            fa2_str = f"{'OOM':>9}"
            _reset_memory()

        try:
            dk_r = bench_single_layer_dualkv(model, mb, P, R, dtype, device)
            dk_str = f"{dk_r.fwd_ms:9.0f}"
            dk_mem = f"{dk_r.peak_mb/1024:6.1f}G"
        except torch.cuda.OutOfMemoryError:
            dk_r = None
            dk_str = f"{'OOM':>9}"
            dk_mem = f"{'OOM':>7}"
            _reset_memory()

        try:
            pg_r = bench_single_layer_pg(model, mb, P, R, dtype, device)
            if pg_r is None:
                pg_str = f"{'N/A':>9}"
                pg_mem = f"{'N/A':>7}"
            else:
                pg_str = f"{pg_r.fwd_ms:9.0f}"
                pg_mem = f"{pg_r.peak_mb/1024:6.1f}G"
        except torch.cuda.OutOfMemoryError:
            pg_r = None
            pg_str = f"{'OOM':>9}"
            pg_mem = f"{'OOM':>7}"
            _reset_memory()

        if fa2_r and dk_r:
            spd_fa2 = f"{fa2_r.fwd_ms/dk_r.fwd_ms:.2f}x"
        elif dk_r:
            spd_fa2 = "∞"
        else:
            spd_fa2 = "---"

        if pg_r and dk_r:
            spd_pg = f"{pg_r.fwd_ms/dk_r.fwd_ms:.2f}x"
        elif dk_r:
            spd_pg = "∞" if pg_r is None else "---"
        else:
            spd_pg = "---"

        print(f"{P:6d} {mb:3d} | {fa2_str} {dk_str} {pg_str} | "
              f"{spd_fa2:>7} {spd_pg:>7} | {dk_mem:>7} {pg_mem:>7}")
        _reset_memory()


# --------------------------------------------------------------------------- #
#  Main
# --------------------------------------------------------------------------- #

def main():
    parser = argparse.ArgumentParser(description="Kernel-level benchmarks (Table 1 & 2)")
    parser.add_argument("--experiment", choices=["table1", "table2", "both"], default="both")
    parser.add_argument("--dtype", choices=["fp16", "bf16"], default="fp16",
                        help="Table 1 uses fp16 (paper); Table 2 uses bf16")
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()

    device = args.device
    torch.cuda.set_device(device)

    if args.experiment in ("table1", "both"):
        dtype = torch.float16  # Table 1 uses fp16 per paper
        run_table1(device, dtype)

    if args.experiment in ("table2", "both"):
        dtype = torch.bfloat16  # Table 2 uses bf16
        run_table2(device, dtype)


if __name__ == "__main__":
    main()
