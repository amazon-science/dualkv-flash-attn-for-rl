#!/usr/bin/env python3
"""Reproduce Paper Table 1: FA2 vs DualKV kernel-level fwd/bwd timing.

Hardware: 1x A100-40GB (paper) or 1x H100-80GB
Config: Qwen3-8B dims (H=32, H_k=8, d=128, GQA 4:1), fp16, R=2048

Usage:
    CUDA_VISIBLE_DEVICES=0 python reproduce_table1.py
"""

import gc
import time
import torch

WARMUP = 5
MEASURE = 10
H, H_k, d = 32, 8, 128
R = 2048
CONFIGS = [
    (28, 4096),
    (28, 16384),
    (16, 32768),
    (28, 32768),
    (16, 65536),
]


def _sync():
    torch.cuda.synchronize()


def _reset():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_peak_memory_stats()


def bench_fa2(N, P):
    from flash_attn import flash_attn_varlen_func

    seqlen = P + R
    total = N * seqlen
    dtype = torch.float16
    device = "cuda"

    q = torch.randn(total, H, d, dtype=dtype, device=device, requires_grad=True)
    k = torch.randn(total, H_k, d, dtype=dtype, device=device, requires_grad=True)
    v = torch.randn(total, H_k, d, dtype=dtype, device=device, requires_grad=True)
    cu = torch.arange(0, (N + 1) * seqlen, seqlen, dtype=torch.int32, device=device)

    for _ in range(WARMUP):
        out = flash_attn_varlen_func(q, k, v, cu, cu, seqlen, seqlen, causal=True)
        out.sum().backward()
        q.grad = k.grad = v.grad = None
        _sync()

    _reset()
    fwd_times, bwd_times = [], []
    for _ in range(MEASURE):
        _sync()
        t0 = time.perf_counter()
        out = flash_attn_varlen_func(q, k, v, cu, cu, seqlen, seqlen, causal=True)
        _sync()
        t1 = time.perf_counter()
        out.sum().backward()
        _sync()
        t2 = time.perf_counter()
        fwd_times.append((t1 - t0) * 1000)
        bwd_times.append((t2 - t1) * 1000)
        q.grad = k.grad = v.grad = None

    peak = torch.cuda.max_memory_allocated() / (1024**2)
    return sorted(fwd_times)[MEASURE // 2], sorted(bwd_times)[MEASURE // 2], peak


def bench_dualkv(N, P):
    from flash_attn import flash_attn_dualkv_varlen_func

    total_q = P + N * R
    dtype = torch.float16
    device = "cuda"

    q = torch.randn(total_q, H, d, dtype=dtype, device=device, requires_grad=True)
    k_ctx = torch.randn(P, H_k, d, dtype=dtype, device=device, requires_grad=True)
    v_ctx = torch.randn(P, H_k, d, dtype=dtype, device=device, requires_grad=True)
    k_dec = torch.randn(N * R, H_k, d, dtype=dtype, device=device, requires_grad=True)
    v_dec = torch.randn(N * R, H_k, d, dtype=dtype, device=device, requires_grad=True)

    cu_q = torch.zeros(N + 1, dtype=torch.int32, device=device)
    cu_q[1] = P + R
    for i in range(2, N + 1):
        cu_q[i] = cu_q[i - 1] + R
    cu_k_dec = torch.arange(0, (N + 1) * R, R, dtype=torch.int32, device=device)
    max_seqlen_q = P + R

    for _ in range(WARMUP):
        out = flash_attn_dualkv_varlen_func(
            q, k_ctx, v_ctx, k_dec, v_dec, cu_q, cu_k_dec,
            max_seqlen_q, P, R, causal=True,
        )
        out.sum().backward()
        q.grad = k_ctx.grad = v_ctx.grad = k_dec.grad = v_dec.grad = None
        _sync()

    _reset()
    fwd_times, bwd_times = [], []
    for _ in range(MEASURE):
        _sync()
        t0 = time.perf_counter()
        out = flash_attn_dualkv_varlen_func(
            q, k_ctx, v_ctx, k_dec, v_dec, cu_q, cu_k_dec,
            max_seqlen_q, P, R, causal=True,
        )
        _sync()
        t1 = time.perf_counter()
        out.sum().backward()
        _sync()
        t2 = time.perf_counter()
        fwd_times.append((t1 - t0) * 1000)
        bwd_times.append((t2 - t1) * 1000)
        q.grad = k_ctx.grad = v_ctx.grad = k_dec.grad = v_dec.grad = None

    peak = torch.cuda.max_memory_allocated() / (1024**2)
    return sorted(fwd_times)[MEASURE // 2], sorted(bwd_times)[MEASURE // 2], peak


def check_correctness(N, P):
    """Verify DualKV forward matches FA2 to within fp16 tolerance."""
    from flash_attn import flash_attn_varlen_func, flash_attn_dualkv_varlen_func

    dtype = torch.float16
    device = "cuda"
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    seqlen = P + R
    total_fa2 = N * seqlen
    total_dk = P + N * R

    # Shared random data for prompt and responses
    prompt_kv = torch.randn(P, H_k, d, dtype=dtype, device=device)
    prompt_v = torch.randn(P, H_k, d, dtype=dtype, device=device)

    # Build FA2 inputs: N copies of [prompt_q(P) + resp_q(R)] with [prompt_kv + resp_kv]
    q_parts, k_parts, v_parts = [], [], []
    resp_qs, resp_ks, resp_vs = [], [], []
    prompt_q = torch.randn(P, H, d, dtype=dtype, device=device)
    for i in range(N):
        rq = torch.randn(R, H, d, dtype=dtype, device=device)
        rk = torch.randn(R, H_k, d, dtype=dtype, device=device)
        rv = torch.randn(R, H_k, d, dtype=dtype, device=device)
        resp_qs.append(rq)
        resp_ks.append(rk)
        resp_vs.append(rv)
        q_parts.extend([prompt_q.clone(), rq])
        k_parts.extend([prompt_kv.clone(), rk])
        v_parts.extend([prompt_v.clone(), rv])

    q_fa2 = torch.cat(q_parts, dim=0)
    k_fa2 = torch.cat(k_parts, dim=0)
    v_fa2 = torch.cat(v_parts, dim=0)
    cu_fa2 = torch.arange(0, (N + 1) * seqlen, seqlen, dtype=torch.int32, device=device)

    with torch.no_grad():
        out_fa2 = flash_attn_varlen_func(q_fa2, k_fa2, v_fa2, cu_fa2, cu_fa2, seqlen, seqlen, causal=True)

    # Build DualKV inputs
    q_dk = torch.cat([prompt_q.clone()] + [rq for rq in resp_qs], dim=0)
    k_ctx = prompt_kv.clone()
    v_ctx = prompt_v.clone()
    k_dec = torch.cat(resp_ks, dim=0)
    v_dec = torch.cat(resp_vs, dim=0)

    cu_q = torch.zeros(N + 1, dtype=torch.int32, device=device)
    cu_q[1] = P + R
    for i in range(2, N + 1):
        cu_q[i] = cu_q[i - 1] + R
    cu_k_dec = torch.arange(0, (N + 1) * R, R, dtype=torch.int32, device=device)

    with torch.no_grad():
        out_dk = flash_attn_dualkv_varlen_func(
            q_dk, k_ctx, v_ctx, k_dec, v_dec, cu_q, cu_k_dec,
            P + R, P, R, causal=True,
        )

    # Compare response portions (skip prompt in FA2 since DualKV prompt goes through varlen separately)
    # DualKV output for responses: out_dk[P:] corresponds to resp tokens
    # FA2 output for responses: for each seq i, out_fa2[i*seqlen + P : (i+1)*seqlen]
    fa2_resp = torch.cat([out_fa2[i * seqlen + P: (i + 1) * seqlen] for i in range(N)], dim=0)
    dk_resp = out_dk[P:]

    match = torch.allclose(fa2_resp, dk_resp, atol=1e-3, rtol=1e-3)
    max_err = (fa2_resp - dk_resp).abs().max().item()
    return match, max_err


def main():
    device = "cuda:0"
    torch.cuda.set_device(device)
    print(f"GPU: {torch.cuda.get_device_name(0)}")

    # Phase 1: Numerical correctness
    print(f"\n{'='*96}")
    print(f"Numerical Correctness (torch.allclose atol=1e-3, rtol=1e-3)")
    print(f"{'='*96}")
    test_configs = [(4, 4096), (4, 16384)]
    all_pass = True
    for N, P in test_configs:
        match, max_err = check_correctness(N, P)
        status = "PASS" if match else "FAIL"
        print(f"  N={N}, P={P}: {status} (max_err={max_err:.2e})")
        all_pass &= match
    if not all_pass:
        print("  *** CORRECTNESS FAILURE — investigate before trusting performance numbers ***")
    print()

    # Phase 2: Performance
    print(f"{'='*96}")
    print(f"Table 1: FA2 vs DualKV kernel timing (H={H}, H_k={H_k}, d={d}, R={R}, fp16)")
    print(f"{'='*96}")
    print(f"{'N':>4} {'P':>6} | {'FA2 fwd':>8} {'FA2 bwd':>8} {'FA2 f+b':>8} | "
          f"{'DK fwd':>8} {'DK bwd':>8} {'DK f+b':>8} | "
          f"{'fwd':>5} {'bwd':>5} {'f+b':>5} | {'Mem↓':>5}")
    print("-" * 96)

    for N, P in CONFIGS:
        _reset()

        try:
            fa2_fwd, fa2_bwd, fa2_mem = bench_fa2(N, P)
            fa2_str = f"{fa2_fwd:8.1f} {fa2_bwd:8.1f} {fa2_fwd+fa2_bwd:8.1f}"
        except torch.cuda.OutOfMemoryError:
            fa2_fwd = fa2_bwd = fa2_mem = None
            fa2_str = f"{'OOM':>8} {'OOM':>8} {'OOM':>8}"
            _reset()

        try:
            dk_fwd, dk_bwd, dk_mem = bench_dualkv(N, P)
            dk_str = f"{dk_fwd:8.1f} {dk_bwd:8.1f} {dk_fwd+dk_bwd:8.1f}"
        except (torch.cuda.OutOfMemoryError, RuntimeError):
            dk_fwd = dk_bwd = dk_mem = None
            dk_str = f"{'OOM':>8} {'OOM':>8} {'OOM':>8}"
            _reset()

        if fa2_fwd and dk_fwd:
            spd_fwd = f"{fa2_fwd/dk_fwd:.2f}x"
            spd_bwd = f"{fa2_bwd/dk_bwd:.2f}x"
            spd_fb = f"{(fa2_fwd+fa2_bwd)/(dk_fwd+dk_bwd):.2f}x"
            mem_red = f"{(1 - dk_mem/fa2_mem)*100:.0f}%"
        elif dk_fwd and not fa2_fwd:
            spd_fwd = spd_bwd = spd_fb = "inf"
            mem_red = "---"
        else:
            spd_fwd = spd_bwd = spd_fb = "---"
            mem_red = "---"

        print(f"{N:4d} {P:6d} | {fa2_str} | {dk_str} | "
              f"{spd_fwd:>5} {spd_bwd:>5} {spd_fb:>5} | {mem_red:>5}")
        _reset()


if __name__ == "__main__":
    main()
