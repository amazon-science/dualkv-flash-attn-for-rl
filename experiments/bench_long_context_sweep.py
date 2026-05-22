"""
Single-layer benchmark: DualKV vs Prefix Grouper vs FA2 baseline.
Extended long-context sweep up to P=1M.

Setup:
  - Qwen3-8B decoder layer (hidden=4096, intermediate=14336, 32 heads, 8 KV heads)
  - N=32 rollouts per prompt
  - Sweep: P ∈ {5059, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576} × micro_batch ∈ {1, 2, 4, 8, 16, 32}
  - R ~ U(131, 2048) (realistic response distribution)
  - bf16, forward + backward, single H100 GPU

Usage:
  CUDA_VISIBLE_DEVICES=0 python bench_long_context_sweep.py
"""

import sys
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Tuple

# ============================================================
# Config
# ============================================================
@dataclass
class Qwen3_8B_Config:
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    dtype: torch.dtype = torch.bfloat16

RESPONSE_MIN = 131
RESPONSE_MAX = 2048
PROMPT_LENGTHS = [5059, 8192, 16384, 32768, 65536, 131072, 262144, 524288, 1048576]
MICRO_BATCHES = [1, 2, 4, 8, 16, 32]
WARMUP_ITERS = 2
BENCH_ITERS = 5

# ============================================================
# Model Components (single decoder layer)
# ============================================================
class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.eps = eps

    def forward(self, x):
        input_dtype = x.dtype
        variance = x.float().pow(2).mean(-1, keepdim=True)
        x = x.float() * torch.rsqrt(variance + self.eps)
        return (self.weight.float() * x).to(input_dtype)


class Qwen3MLP(nn.Module):
    def __init__(self, cfg: Qwen3_8B_Config):
        super().__init__()
        self.gate_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.up_proj = nn.Linear(cfg.hidden_size, cfg.intermediate_size, bias=False)
        self.down_proj = nn.Linear(cfg.intermediate_size, cfg.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))


class Qwen3Attention(nn.Module):
    def __init__(self, cfg: Qwen3_8B_Config):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.num_kv_groups = self.num_heads // self.num_kv_heads

        self.q_proj = nn.Linear(cfg.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=False)

    def forward(self, hidden_states):
        tokens = hidden_states.shape[0] if hidden_states.dim() == 2 else hidden_states.shape[0] * hidden_states.shape[1]
        flat = hidden_states.reshape(tokens, -1)
        q = self.q_proj(flat).view(tokens, self.num_heads, self.head_dim)
        k = self.k_proj(flat).view(tokens, self.num_kv_heads, self.head_dim)
        v = self.v_proj(flat).view(tokens, self.num_kv_heads, self.head_dim)
        return q, k, v

    def output_proj(self, attn_out):
        shape = attn_out.shape[:-2]
        return self.o_proj(attn_out.reshape(*shape, -1))


class Qwen3DecoderLayer(nn.Module):
    def __init__(self, cfg: Qwen3_8B_Config):
        super().__init__()
        self.input_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.post_attention_layernorm = RMSNorm(cfg.hidden_size, cfg.rms_norm_eps)
        self.self_attn = Qwen3Attention(cfg)
        self.mlp = Qwen3MLP(cfg)
        self.cfg = cfg

    def forward_fa2_varlen(self, hidden_states, cu_seqlens, max_seqlen):
        from flash_attn import flash_attn_varlen_func

        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        q, k, v = self.self_attn(h)

        attn_out = flash_attn_varlen_func(
            q, k, v,
            cu_seqlens_q=cu_seqlens,
            cu_seqlens_k=cu_seqlens,
            max_seqlen_q=max_seqlen,
            max_seqlen_k=max_seqlen,
            causal=True,
        )
        h = residual + self.self_attn.output_proj(attn_out)

        residual = h
        h = self.post_attention_layernorm(h)
        h = residual + self.mlp(h)
        return h

    def forward_dualkv(self, hidden_states, cu_seqlens_dec, max_dec, prefix_len):
        from flash_attn import flash_attn_varlen_func, flash_attn_dualkv_varlen_func

        residual = hidden_states
        h = self.input_layernorm(hidden_states)
        q, k, v = self.self_attn(h)

        P = prefix_len
        q_ctx, k_ctx, v_ctx = q[:P], k[:P], v[:P]
        q_dec, k_dec, v_dec = q[P:], k[P:], v[P:]

        cu_ctx = torch.tensor([0, P], device=q.device, dtype=torch.int32)
        ctx_out = flash_attn_varlen_func(
            q_ctx, k_ctx, v_ctx, cu_ctx, cu_ctx, P, P, causal=True,
        )

        dec_out = flash_attn_dualkv_varlen_func(
            q_dec, k_ctx, v_ctx, k_dec, v_dec,
            cu_seqlens_dec, cu_seqlens_dec,
            max_seqlen_q=max_dec,
            context_seqlen=P,
            max_seqlen_k_decoded=max_dec,
            causal=True,
        )

        attn_out = torch.cat([ctx_out, dec_out], dim=0)
        h = residual + self.self_attn.output_proj(attn_out)

        residual = h
        h = self.post_attention_layernorm(h)
        h = residual + self.mlp(h)
        return h

    def forward_prefix_grouper(self, hidden_states, prefix_len):
        from flash_attn import flash_attn_func

        residual = hidden_states  # (B, T, H)
        h = self.input_layernorm(hidden_states)

        B, T, H = h.shape
        P = prefix_len
        R_max = T - P

        q = self.self_attn.q_proj(h).view(B, T, self.self_attn.num_heads, self.self_attn.head_dim)
        k = self.self_attn.k_proj(h).view(B, T, self.self_attn.num_kv_heads, self.self_attn.head_dim)
        v = self.self_attn.v_proj(h).view(B, T, self.self_attn.num_kv_heads, self.self_attn.head_dim)

        q_prefix = q[:1, :P, :, :]
        k_prefix = k[:1, :P, :, :]
        v_prefix = v[:1, :P, :, :]

        prefix_out = flash_attn_func(
            q_prefix, k_prefix, v_prefix, causal=True,
        )

        q_suffix = q[:, P:, :, :]

        k_prefix_exp = k[:1, :P, :, :].expand(B, -1, -1, -1).contiguous()
        v_prefix_exp = v[:1, :P, :, :].expand(B, -1, -1, -1).contiguous()

        k_suffix = k[:, P:, :, :]
        v_suffix = v[:, P:, :, :]

        k_full = torch.cat([k_prefix_exp, k_suffix], dim=1)
        v_full = torch.cat([v_prefix_exp, v_suffix], dim=1)

        suffix_out = flash_attn_func(
            q_suffix, k_full, v_full, causal=True,
        )

        prefix_out_exp = prefix_out.expand(B, -1, -1, -1)
        attn_out = torch.cat([prefix_out_exp, suffix_out], dim=1)
        attn_out = attn_out.reshape(B, T, -1)

        h = residual + self.self_attn.o_proj(attn_out)

        residual = h
        h = self.post_attention_layernorm(h)
        h = residual + self.mlp(h)
        return h


# ============================================================
# Data Generation
# ============================================================
def generate_response_lengths(micro_batch_size: int, seed: int = 42) -> List[int]:
    rng = torch.Generator().manual_seed(seed)
    lengths = torch.randint(RESPONSE_MIN, RESPONSE_MAX + 1, (micro_batch_size,), generator=rng)
    return lengths.tolist()


def prepare_fa2_inputs(cfg, P, micro_batch_size, response_lens, device):
    cu_seqlens = [0]
    total_tokens = 0
    for R in response_lens:
        total_tokens += P + R
        cu_seqlens.append(total_tokens)

    cu_seqlens = torch.tensor(cu_seqlens, device=device, dtype=torch.int32)
    max_seqlen = P + max(response_lens)
    hidden_states = torch.randn(total_tokens, cfg.hidden_size, device=device, dtype=cfg.dtype)
    return hidden_states, cu_seqlens, max_seqlen


def prepare_dualkv_inputs(cfg, P, micro_batch_size, response_lens, device):
    total_tokens = P + sum(response_lens)
    cu_seqlens_dec = [0]
    for R in response_lens:
        cu_seqlens_dec.append(cu_seqlens_dec[-1] + R)

    cu_seqlens_dec = torch.tensor(cu_seqlens_dec, device=device, dtype=torch.int32)
    max_dec = max(response_lens)
    hidden_states = torch.randn(total_tokens, cfg.hidden_size, device=device, dtype=cfg.dtype)
    return hidden_states, cu_seqlens_dec, max_dec


def prepare_prefix_grouper_inputs(cfg, P, micro_batch_size, response_lens, device):
    R_max = max(response_lens)
    T = P + R_max
    B = micro_batch_size
    hidden_states = torch.randn(B, T, cfg.hidden_size, device=device, dtype=cfg.dtype)
    return hidden_states


# ============================================================
# Performance Benchmarking
# ============================================================
def bench_method(fn, warmup=WARMUP_ITERS, iters=BENCH_ITERS):
    for _ in range(warmup):
        loss = fn()
        loss.backward()

    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()

    times = []
    for _ in range(iters):
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        loss = fn()
        loss.backward()
        torch.cuda.synchronize()
        times.append(time.perf_counter() - t0)

    peak_mem = torch.cuda.max_memory_allocated() / (1024**3)
    avg_time = sum(times) / len(times) * 1000  # ms
    std_time = (sum((t*1000 - avg_time)**2 for t in times) / len(times)) ** 0.5

    return avg_time, std_time, peak_mem


def estimate_memory_gb(cfg, P, micro_batch_size, response_lens, method):
    """Conservative memory estimate in GB."""
    R_max = max(response_lens)
    hidden = cfg.hidden_size
    intermediate = cfg.intermediate_size
    bytes_per_elem = 2  # bf16

    if method == "fa2":
        tokens = sum(P + R for R in response_lens)
    elif method == "dualkv":
        tokens = P + sum(response_lens)
    else:  # prefix_grouper
        tokens = micro_batch_size * (P + R_max)

    # params (~0.4GB) + activations (fwd) + grads (bwd) ≈ 3x fwd activations
    est_gb = (tokens * (hidden + intermediate * 2) * bytes_per_elem * 3) / (1024**3)
    return est_gb


def run_perf_benchmark(P: int, micro_batch_size: int, device: str = "cuda"):
    cfg = Qwen3_8B_Config()
    response_lens = generate_response_lengths(micro_batch_size)
    R_max = max(response_lens)

    fa2_tokens = sum(P + R for R in response_lens)
    dualkv_tokens = P + sum(response_lens)
    pg_tokens = micro_batch_size * (P + R_max)

    results = {"P": P, "mb": micro_batch_size,
               "tokens_fa2": fa2_tokens, "tokens_dualkv": dualkv_tokens, "tokens_pg": pg_tokens}

    print(f"\n  P={P:>7,}, mb={micro_batch_size:<2} | tokens: FA2={fa2_tokens:>10,} DualKV={dualkv_tokens:>10,} PG={pg_tokens:>10,}")

    # --- FA2 Varlen ---
    est = estimate_memory_gb(cfg, P, micro_batch_size, response_lens, "fa2")
    if est < 70:
        layer = Qwen3DecoderLayer(cfg).to(device=device, dtype=cfg.dtype)
        h, cu, max_sl = prepare_fa2_inputs(cfg, P, micro_batch_size, response_lens, device)
        h.requires_grad_(True)
        try:
            t, std, mem = bench_method(lambda: layer.forward_fa2_varlen(h, cu, max_sl).sum())
            results["fa2"] = (t, std, mem)
            print(f"    FA2:    {t:>8.1f} ± {std:.1f} ms | {mem:.2f} GB")
        except torch.cuda.OutOfMemoryError:
            results["fa2"] = None
            print(f"    FA2:    OOM")
        del layer, h
        torch.cuda.empty_cache()
    else:
        results["fa2"] = None
        print(f"    FA2:    SKIP (est {est:.0f} GB > 70 GB)")

    # --- DualKV ---
    est = estimate_memory_gb(cfg, P, micro_batch_size, response_lens, "dualkv")
    if est < 70:
        layer = Qwen3DecoderLayer(cfg).to(device=device, dtype=cfg.dtype)
        h, cu_dec, max_dec = prepare_dualkv_inputs(cfg, P, micro_batch_size, response_lens, device)
        h.requires_grad_(True)
        try:
            t, std, mem = bench_method(lambda: layer.forward_dualkv(h, cu_dec, max_dec, P).sum())
            results["dualkv"] = (t, std, mem)
            print(f"    DualKV: {t:>8.1f} ± {std:.1f} ms | {mem:.2f} GB")
        except torch.cuda.OutOfMemoryError:
            results["dualkv"] = None
            print(f"    DualKV: OOM")
        del layer, h
        torch.cuda.empty_cache()
    else:
        results["dualkv"] = None
        print(f"    DualKV: SKIP (est {est:.0f} GB > 70 GB)")

    # --- Prefix Grouper ---
    est = estimate_memory_gb(cfg, P, micro_batch_size, response_lens, "prefix_grouper")
    if est < 70:
        layer = Qwen3DecoderLayer(cfg).to(device=device, dtype=cfg.dtype)
        h = prepare_prefix_grouper_inputs(cfg, P, micro_batch_size, response_lens, device)
        h.requires_grad_(True)
        try:
            t, std, mem = bench_method(lambda: layer.forward_prefix_grouper(h, P).sum())
            results["prefix_grouper"] = (t, std, mem)
            print(f"    PG:     {t:>8.1f} ± {std:.1f} ms | {mem:.2f} GB")
        except torch.cuda.OutOfMemoryError:
            results["prefix_grouper"] = None
            print(f"    PG:     OOM")
        del layer, h
        torch.cuda.empty_cache()
    else:
        results["prefix_grouper"] = None
        print(f"    PG:     SKIP (est {est:.0f} GB > 70 GB)")

    return results


# ============================================================
# Main
# ============================================================
def main():
    device = "cuda"
    print(f"Device: {torch.cuda.get_device_name(0)}")
    print(f"Config: Qwen3-8B single decoder layer, bf16")
    print(f"Sweep: P ∈ {PROMPT_LENGTHS} × micro_batch ∈ {MICRO_BATCHES}")
    print(f"Response lengths: U({RESPONSE_MIN}, {RESPONSE_MAX})")
    print(f"Warmup: {WARMUP_ITERS}, Bench iters: {BENCH_ITERS}")

    print("\n" + "=" * 70)
    print("  PERFORMANCE BENCHMARK — LONG CONTEXT SWEEP")
    print("=" * 70)

    all_results = []
    for P in PROMPT_LENGTHS:
        for mb in MICRO_BATCHES:
            r = run_perf_benchmark(P, mb, device)
            all_results.append(r)
            sys.stdout.flush()

    # ========================================
    # Final Summary Table
    # ========================================
    print("\n" + "=" * 70)
    print("  FINAL RESULTS")
    print("=" * 70)

    print(f"\n  {'P':<9} {'mb':<4} {'FA2 (ms)':<12} {'DualKV (ms)':<12} {'PG (ms)':<12} "
          f"{'DK vs FA2':<12} {'DK vs PG':<12} {'DK Mem':<8} {'PG Mem':<8}")
    print(f"  {'─'*100}")

    for r in all_results:
        P = r["P"]
        mb = r["mb"]
        fa2_str = f"{r['fa2'][0]:.1f}" if r.get("fa2") else "OOM"
        dk_str = f"{r['dualkv'][0]:.1f}" if r.get("dualkv") else "OOM"
        pg_str = f"{r['prefix_grouper'][0]:.1f}" if r.get("prefix_grouper") else "OOM"

        if r.get("fa2") and r.get("dualkv"):
            dk_vs_fa2 = f"{r['fa2'][0] / r['dualkv'][0]:.2f}x"
        else:
            dk_vs_fa2 = "—"

        if r.get("prefix_grouper") and r.get("dualkv"):
            dk_vs_pg = f"{r['prefix_grouper'][0] / r['dualkv'][0]:.2f}x"
        else:
            dk_vs_pg = "—"

        dk_mem = f"{r['dualkv'][2]:.1f}" if r.get("dualkv") else "—"
        pg_mem = f"{r['prefix_grouper'][2]:.1f}" if r.get("prefix_grouper") else "—"

        print(f"  {P:<9,} {mb:<4} {fa2_str:<12} {dk_str:<12} {pg_str:<12} "
              f"{dk_vs_fa2:<12} {dk_vs_pg:<12} {dk_mem:<8} {pg_mem:<8}")

    print(f"  {'─'*100}")

    # Token count summary
    print(f"\n  Token counts (total through MLP per micro-batch):")
    print(f"  {'P':<9} {'mb':<4} {'FA2':<12} {'DualKV':<12} {'PG':<12} {'FA2/DK':<8} {'PG/DK':<8}")
    print(f"  {'─'*65}")
    for r in all_results:
        ratio_fa2_dk = r['tokens_fa2'] / r['tokens_dualkv']
        ratio_pg_dk = r['tokens_pg'] / r['tokens_dualkv']
        print(f"  {r['P']:<9,} {r['mb']:<4} {r['tokens_fa2']:<12,} {r['tokens_dualkv']:<12,} "
              f"{r['tokens_pg']:<12,} {ratio_fa2_dk:<8.2f} {ratio_pg_dk:<8.2f}")

    # OOM summary
    print(f"\n  OOM Summary:")
    fa2_oom = sum(1 for r in all_results if r.get("fa2") is None)
    dk_oom = sum(1 for r in all_results if r.get("dualkv") is None)
    pg_oom = sum(1 for r in all_results if r.get("prefix_grouper") is None)
    total = len(all_results)
    print(f"    FA2:    {fa2_oom}/{total} configs OOM")
    print(f"    DualKV: {dk_oom}/{total} configs OOM")
    print(f"    PG:     {pg_oom}/{total} configs OOM")
    print()


if __name__ == "__main__":
    main()
