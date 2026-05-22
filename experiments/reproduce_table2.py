#!/usr/bin/env python3
"""Reproduce Paper Table 2: Single-layer DualKV vs Prefix Grouper vs FA2.

Hardware: 1x H100-80GB
Config: Qwen3-8B single decoder layer (hidden=4096, intermediate=14336,
        32 heads, 8 KV heads, d=128), fp16, forward + backward.
        Response lengths ~ U(131, 2048).

Sweep: P in {5059, 8192, 16384, 32768} x micro_batch in {4, 8, 16, 32}
Paper Table 2 reports the diagonal: (5K,32), (8K,16), (16K,8), (32K,4)
plus OOM configs (65K,4) and (131K,8).

Prefix Grouper is self-implemented here (no external package needed):
it uses padded [B, P+R_max, hidden] tensors with split prefix/suffix
attention via flash_attn_func, which is the best-case for PG since FA2
handles the attention itself.

Usage:
    CUDA_VISIBLE_DEVICES=0 python reproduce_table2.py
"""

import sys
import os
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List, Tuple


@dataclass
class Qwen3_8B_Config:
    hidden_size: int = 4096
    intermediate_size: int = 14336
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    rms_norm_eps: float = 1e-6
    dtype: torch.dtype = torch.float16


RESPONSE_MIN = 131
RESPONSE_MAX = 2048
PROMPT_LENGTHS = [5059, 8192, 16384, 32768]
MICRO_BATCHES = [4, 8, 16, 32]
WARMUP_ITERS = 3
BENCH_ITERS = 10


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
        """Prefix Grouper: padded [B, P+R_max, hidden].

        Uses flash_attn_func (padded FA2) for both prefix and suffix attention.
        This gives PG the best possible attention performance — the overhead
        comes from processing full padded tensors through projections and MLP.
        """
        from flash_attn import flash_attn_func

        residual = hidden_states
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

        prefix_out = flash_attn_func(q_prefix, k_prefix, v_prefix, causal=True)

        q_suffix = q[:, P:, :, :]
        k_prefix_exp = k[:1, :P, :, :].expand(B, -1, -1, -1).contiguous()
        v_prefix_exp = v[:1, :P, :, :].expand(B, -1, -1, -1).contiguous()
        k_suffix = k[:, P:, :, :]
        v_suffix = v[:, P:, :, :]

        k_full = torch.cat([k_prefix_exp, k_suffix], dim=1)
        v_full = torch.cat([v_prefix_exp, v_suffix], dim=1)

        suffix_out = flash_attn_func(q_suffix, k_full, v_full, causal=True)

        prefix_out_exp = prefix_out.expand(B, -1, -1, -1)
        attn_out = torch.cat([prefix_out_exp, suffix_out], dim=1)
        attn_out = attn_out.reshape(B, T, -1)

        h = residual + self.self_attn.o_proj(attn_out)

        residual = h
        h = self.post_attention_layernorm(h)
        h = residual + self.mlp(h)
        return h


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
    hidden_states = torch.randn(micro_batch_size, T, cfg.hidden_size, device=device, dtype=cfg.dtype)
    return hidden_states


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
    med_time = sorted(times)[len(times) // 2] * 1000
    return med_time, peak_mem


def can_fit_in_memory(cfg, P, micro_batch_size, response_lens, method):
    R_max = max(response_lens)
    hidden = cfg.hidden_size
    intermediate = cfg.intermediate_size

    if method == "fa2":
        tokens = sum(P + R for R in response_lens)
    elif method == "dualkv":
        tokens = P + sum(response_lens)
    else:
        tokens = micro_batch_size * (P + R_max)

    est_gb = (tokens * (hidden + intermediate * 2) * 2 * 3) / (1024**3)
    return est_gb < 70


def run_perf_benchmark(P: int, micro_batch_size: int, device: str = "cuda"):
    cfg = Qwen3_8B_Config()
    response_lens = generate_response_lengths(micro_batch_size)
    R_max = max(response_lens)

    fa2_tokens = sum(P + R for R in response_lens)
    dualkv_tokens = P + sum(response_lens)
    pg_tokens = micro_batch_size * (P + R_max)

    results = {"P": P, "mb": micro_batch_size,
               "tokens_fa2": fa2_tokens, "tokens_dualkv": dualkv_tokens, "tokens_pg": pg_tokens}

    print(f"\n  P={P}, mb={micro_batch_size} | tokens: FA2={fa2_tokens:,} DualKV={dualkv_tokens:,} PG={pg_tokens:,}")

    # FA2
    if can_fit_in_memory(cfg, P, micro_batch_size, response_lens, "fa2"):
        layer = Qwen3DecoderLayer(cfg).to(device=device, dtype=cfg.dtype)
        h, cu, max_sl = prepare_fa2_inputs(cfg, P, micro_batch_size, response_lens, device)
        h.requires_grad_(True)
        try:
            t, mem = bench_method(lambda: layer.forward_fa2_varlen(h, cu, max_sl).sum())
            results["fa2"] = (t, mem)
            print(f"    FA2:    {t:>8.1f} ms | {mem:.2f} GB")
        except torch.cuda.OutOfMemoryError:
            results["fa2"] = None
            print(f"    FA2:    OOM")
        del layer, h
        torch.cuda.empty_cache()
    else:
        results["fa2"] = None
        print(f"    FA2:    SKIP (est OOM)")

    # DualKV
    if can_fit_in_memory(cfg, P, micro_batch_size, response_lens, "dualkv"):
        layer = Qwen3DecoderLayer(cfg).to(device=device, dtype=cfg.dtype)
        h, cu_dec, max_dec = prepare_dualkv_inputs(cfg, P, micro_batch_size, response_lens, device)
        h.requires_grad_(True)
        try:
            t, mem = bench_method(lambda: layer.forward_dualkv(h, cu_dec, max_dec, P).sum())
            results["dualkv"] = (t, mem)
            print(f"    DualKV: {t:>8.1f} ms | {mem:.2f} GB")
        except torch.cuda.OutOfMemoryError:
            results["dualkv"] = None
            print(f"    DualKV: OOM")
        del layer, h
        torch.cuda.empty_cache()
    else:
        results["dualkv"] = None
        print(f"    DualKV: SKIP (est OOM)")

    # Prefix Grouper
    if can_fit_in_memory(cfg, P, micro_batch_size, response_lens, "prefix_grouper"):
        layer = Qwen3DecoderLayer(cfg).to(device=device, dtype=cfg.dtype)
        h = prepare_prefix_grouper_inputs(cfg, P, micro_batch_size, response_lens, device)
        h.requires_grad_(True)
        try:
            t, mem = bench_method(lambda: layer.forward_prefix_grouper(h, P).sum())
            results["prefix_grouper"] = (t, mem)
            print(f"    PG:     {t:>8.1f} ms | {mem:.2f} GB")
        except torch.cuda.OutOfMemoryError:
            results["prefix_grouper"] = None
            print(f"    PG:     OOM")
        del layer, h
        torch.cuda.empty_cache()
    else:
        results["prefix_grouper"] = None
        print(f"    PG:     SKIP (est OOM)")

    return results


def check_correctness(P: int, micro_batch_size: int, device: str = "cuda"):
    """Verify DualKV single-layer output matches FA2 (torch.allclose atol=1e-3, rtol=1e-3 in fp16)."""
    cfg = Qwen3_8B_Config()
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    layer = Qwen3DecoderLayer(cfg).to(device=device, dtype=cfg.dtype)
    layer.eval()

    response_lens = generate_response_lengths(micro_batch_size, seed=42)
    R_max = max(response_lens)

    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    dtype = cfg.dtype
    prompt_hidden = torch.randn(P, cfg.hidden_size, device=device, dtype=dtype)
    response_hiddens = [torch.randn(R, cfg.hidden_size, device=device, dtype=dtype) for R in response_lens]

    # FA2 varlen: pack as [P+R1, P+R2, ...]
    fa2_parts = []
    cu = [0]
    for i, R in enumerate(response_lens):
        fa2_parts.append(prompt_hidden.clone())
        fa2_parts.append(response_hiddens[i].clone())
        cu.append(cu[-1] + P + R)
    h_fa2 = torch.cat(fa2_parts, dim=0)
    cu_fa2 = torch.tensor(cu, device=device, dtype=torch.int32)

    # DualKV: pack as [P, R1, R2, ...]
    dk_parts = [prompt_hidden.clone()]
    cu_dec = [0]
    for i, R in enumerate(response_lens):
        dk_parts.append(response_hiddens[i].clone())
        cu_dec.append(cu_dec[-1] + R)
    h_dk = torch.cat(dk_parts, dim=0)
    cu_dec = torch.tensor(cu_dec, device=device, dtype=torch.int32)

    with torch.no_grad():
        out_fa2 = layer.forward_fa2_varlen(h_fa2, cu_fa2, P + R_max)
        out_dk = layer.forward_dualkv(h_dk, cu_dec, R_max, P)

    # Compare response portions
    fa2_responses = []
    offset = 0
    for i, R in enumerate(response_lens):
        fa2_responses.append(out_fa2[offset + P: offset + P + R])
        offset += P + R
    dk_responses = []
    offset = P
    for i, R in enumerate(response_lens):
        dk_responses.append(out_dk[offset: offset + R])
        offset += R

    fa2_cat = torch.cat(fa2_responses, dim=0)
    dk_cat = torch.cat(dk_responses, dim=0)
    match = torch.allclose(fa2_cat, dk_cat, atol=1e-3, rtol=1e-3)
    max_err = (fa2_cat - dk_cat).abs().max().item()

    del layer, h_fa2, h_dk, out_fa2, out_dk
    torch.cuda.empty_cache()
    return match, max_err


def main():
    device = "cuda:0"
    torch.cuda.set_device(device)
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"Config: Qwen3-8B single decoder layer, fp16")
    print(f"Sweep: P in {PROMPT_LENGTHS} x micro_batch in {MICRO_BATCHES}")
    print(f"Response lengths: U({RESPONSE_MIN}, {RESPONSE_MAX})")

    # Phase 1: Numerical correctness (DualKV vs FA2)
    print("\n" + "=" * 70)
    print("  NUMERICAL CORRECTNESS (DualKV vs FA2, fp16, atol=1e-3, rtol=1e-3)")
    print("=" * 70)
    all_pass = True
    for P in [4096, 8192, 16384]:
        for mb in [4, 8]:
            match, max_err = check_correctness(P, mb, device)
            status = "PASS" if match else "FAIL"
            print(f"  P={P}, mb={mb}: {status} (max_err={max_err:.2e})")
            all_pass &= match
    if not all_pass:
        print("  *** CORRECTNESS FAILURE ***")
    print()

    # Phase 2: Performance
    print("=" * 70)
    print("  PERFORMANCE BENCHMARK")
    print("=" * 70)

    all_results = []
    for P in PROMPT_LENGTHS:
        for mb in MICRO_BATCHES:
            r = run_perf_benchmark(P, mb, device)
            all_results.append(r)

    print("\n" + "=" * 70)
    print("  FINAL RESULTS (Paper Table 2 rows marked with *)")
    print("=" * 70)

    paper_configs = {(5059, 32), (8192, 16), (16384, 8), (32768, 4)}

    print(f"\n  {'':>1}{'P':<7} {'mb':<4} {'FA2 (ms)':<12} {'DualKV (ms)':<12} {'PG (ms)':<12} "
          f"{'DK vs FA2':<10} {'DK vs PG':<10} {'DK Mem':<8} {'PG Mem':<8}")
    print(f"  {'─'*95}")

    for r in all_results:
        P = r["P"]
        mb = r["mb"]
        mark = "*" if (P, mb) in paper_configs else " "
        fa2_str = f"{r['fa2'][0]:.0f}" if r.get("fa2") else "OOM"
        dk_str = f"{r['dualkv'][0]:.0f}" if r.get("dualkv") else "OOM"
        pg_str = f"{r['prefix_grouper'][0]:.0f}" if r.get("prefix_grouper") else "OOM"

        if r.get("fa2") and r.get("dualkv"):
            spd_fa2 = f"{r['fa2'][0] / r['dualkv'][0]:.2f}x"
        elif r.get("dualkv") and not r.get("fa2"):
            spd_fa2 = "inf"
        else:
            spd_fa2 = "---"

        if r.get("prefix_grouper") and r.get("dualkv"):
            spd_pg = f"{r['prefix_grouper'][0] / r['dualkv'][0]:.2f}x"
        elif r.get("dualkv") and not r.get("prefix_grouper"):
            spd_pg = "inf"
        else:
            spd_pg = "---"

        dk_mem = f"{r['dualkv'][1]:.1f}" if r.get("dualkv") else "---"
        pg_mem = f"{r['prefix_grouper'][1]:.1f}" if r.get("prefix_grouper") else "---"

        print(f"  {mark}{P:<7} {mb:<4} {fa2_str:<12} {dk_str:<12} {pg_str:<12} "
              f"{spd_fa2:<10} {spd_pg:<10} {dk_mem:<8} {pg_mem:<8}")

    print(f"  {'─'*95}")
    print(f"\n  * = configs reported in paper Table 2")


if __name__ == "__main__":
    main()
