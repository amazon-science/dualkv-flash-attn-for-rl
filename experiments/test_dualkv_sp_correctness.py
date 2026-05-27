#!/usr/bin/env python3
"""DualKV + Ulysses SP=2 correctness test.

Validates that the DualKV attention kernel composed with Ulysses all-to-all
produces numerically equivalent results to DualKV alone (which is already
validated against FA2 in reproduce_table2.py).

Usage:
    torchrun --nproc-per-node=2 test_dualkv_sp_correctness.py

Requires: 2 GPUs (or can run on 1 GPU with CUDA_VISIBLE_DEVICES trick).
"""

import os
import sys
import torch
import torch.distributed as dist
import torch.nn as nn
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List


@dataclass
class TestConfig:
    hidden_size: int = 4096
    num_attention_heads: int = 32
    num_key_value_heads: int = 8
    head_dim: int = 128
    dtype: torch.dtype = torch.float16


class TestAttention(nn.Module):
    def __init__(self, cfg: TestConfig):
        super().__init__()
        self.num_heads = cfg.num_attention_heads
        self.num_kv_heads = cfg.num_key_value_heads
        self.head_dim = cfg.head_dim
        self.q_proj = nn.Linear(cfg.hidden_size, self.num_heads * self.head_dim, bias=False)
        self.k_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.v_proj = nn.Linear(cfg.hidden_size, self.num_kv_heads * self.head_dim, bias=False)
        self.o_proj = nn.Linear(self.num_heads * self.head_dim, cfg.hidden_size, bias=False)

    def project(self, hidden_states):
        T = hidden_states.shape[0]
        q = self.q_proj(hidden_states).view(T, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(T, self.num_kv_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(T, self.num_kv_heads, self.head_dim)
        return q, k, v


def dualkv_attention_no_sp(attn: TestAttention, hidden_states: torch.Tensor,
                           prompt_len: int, cu_seqlens_dec: torch.Tensor,
                           max_dec: int) -> torch.Tensor:
    """Reference: DualKV without SP. Returns (T, hidden_size)."""
    from flash_attn import flash_attn_varlen_func, flash_attn_dualkv_varlen_func

    q, k, v = attn.project(hidden_states)
    P = prompt_len

    q_ctx, k_ctx, v_ctx = q[:P], k[:P], v[:P]
    q_dec, k_dec, v_dec = q[P:], k[P:], v[P:]

    cu_ctx = torch.tensor([0, P], device=q.device, dtype=torch.int32)
    ctx_out = flash_attn_varlen_func(
        q_ctx, k_ctx, v_ctx, cu_ctx, cu_ctx, P, P,
        causal=True, deterministic=True,
    )

    dec_out = flash_attn_dualkv_varlen_func(
        q_dec, k_ctx, v_ctx, k_dec, v_dec,
        cu_seqlens_dec, cu_seqlens_dec,
        max_seqlen_q=max_dec, context_seqlen=P, max_seqlen_k_decoded=max_dec,
        causal=True,
    )

    attn_out = torch.cat([ctx_out, dec_out], dim=0)
    return attn.o_proj(attn_out.reshape(attn_out.shape[0], -1))


def dualkv_attention_with_sp(attn: TestAttention, hidden_states_local: torch.Tensor,
                             prompt_len: int, cu_seqlens_dec: torch.Tensor,
                             max_dec: int, sp_group: dist.ProcessGroup,
                             pad_size: int) -> torch.Tensor:
    """DualKV + Ulysses SP=2. Each rank gets T/sp tokens, performs all-to-all.
    Returns (T/sp, hidden_size) per rank."""
    from flash_attn import flash_attn_varlen_func, flash_attn_dualkv_varlen_func

    sp_size = dist.get_world_size(sp_group)
    T_local = hidden_states_local.shape[0]

    q_local = attn.q_proj(hidden_states_local).view(T_local, attn.num_heads, attn.head_dim)
    k_local = attn.k_proj(hidden_states_local).view(T_local, attn.num_kv_heads, attn.head_dim)
    v_local = attn.v_proj(hidden_states_local).view(T_local, attn.num_kv_heads, attn.head_dim)

    # KV head replication if needed
    repeats = max(sp_size // attn.num_kv_heads, 1)
    if repeats > 1:
        k_local = k_local.repeat(1, repeats, 1)
        v_local = v_local.repeat(1, repeats, 1)

    # All-to-all: gather seq, scatter heads
    # (T/sp, H, d) → (T, H/sp, d)
    q_full = _all_to_all(q_local, scatter_dim=1, gather_dim=0, group=sp_group)
    k_full = _all_to_all(k_local, scatter_dim=1, gather_dim=0, group=sp_group)
    v_full = _all_to_all(v_local, scatter_dim=1, gather_dim=0, group=sp_group)

    # DualKV kernel on full sequence with partial heads
    P = prompt_len
    q_ctx, k_ctx, v_ctx = q_full[:P], k_full[:P], v_full[:P]
    q_dec, k_dec, v_dec = q_full[P:], k_full[P:], v_full[P:]

    cu_ctx = torch.tensor([0, P], device=q_full.device, dtype=torch.int32)
    ctx_out = flash_attn_varlen_func(
        q_ctx, k_ctx, v_ctx, cu_ctx, cu_ctx, P, P,
        causal=True, deterministic=True,
    )

    dec_out = flash_attn_dualkv_varlen_func(
        q_dec, k_ctx, v_ctx, k_dec, v_dec,
        cu_seqlens_dec, cu_seqlens_dec,
        max_seqlen_q=max_dec, context_seqlen=P, max_seqlen_k_decoded=max_dec,
        causal=True,
    )

    out_full = torch.cat([ctx_out, dec_out], dim=0)  # (T, H/sp, d)

    # All-to-all: gather heads, scatter seq
    # (T, H/sp, d) → (T/sp, H, d)
    out_local = _all_to_all(out_full, scatter_dim=0, gather_dim=1, group=sp_group)

    return attn.o_proj(out_local.reshape(out_local.shape[0], -1))


def _all_to_all(tensor: torch.Tensor, scatter_dim: int, gather_dim: int,
                group: dist.ProcessGroup) -> torch.Tensor:
    sp_size = dist.get_world_size(group)
    input_list = [t.contiguous() for t in torch.tensor_split(tensor, sp_size, scatter_dim)]
    output_list = [torch.empty_like(input_list[0]) for _ in range(sp_size)]
    dist.all_to_all(output_list, input_list, group=group)
    return torch.cat(output_list, dim=gather_dim).contiguous()


def run_test():
    dist.init_process_group(backend="nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    assert world_size == 2, f"This test requires exactly 2 processes, got {world_size}"

    local_device = torch.device(f"cuda:{rank}")
    torch.cuda.set_device(local_device)

    sp_group = dist.new_group(list(range(world_size)))

    cfg = TestConfig()
    test_cases = [
        (512, 4, [128, 256, 192, 300]),
        (1024, 3, [200, 150, 350]),
        (2048, 2, [512, 768]),
        (4096, 4, [256, 512, 384, 640]),
    ]

    all_pass = True

    for prompt_len, n_responses, response_lens in test_cases:
        torch.manual_seed(42)
        torch.cuda.manual_seed(42)

        attn = TestAttention(cfg).to(device=local_device, dtype=cfg.dtype)
        attn.eval()

        P = prompt_len
        T = P + sum(response_lens)
        hidden = torch.randn(T, cfg.hidden_size, device=local_device, dtype=cfg.dtype)

        cu_seqlens_dec = torch.tensor(
            [0] + list(torch.cumsum(torch.tensor(response_lens), dim=0).numpy()),
            device=local_device, dtype=torch.int32,
        )
        max_dec = max(response_lens)

        # --- Reference: DualKV no SP (run on all ranks with same data) ---
        with torch.no_grad():
            ref_out = dualkv_attention_no_sp(attn, hidden, P, cu_seqlens_dec, max_dec)

        # --- Test: DualKV + SP=2 ---
        # Pad to be divisible by sp_size
        sp_size = world_size
        pad_size = (sp_size - T % sp_size) % sp_size
        if pad_size > 0:
            hidden_padded = torch.cat([
                hidden,
                torch.zeros(pad_size, cfg.hidden_size, device=local_device, dtype=cfg.dtype)
            ], dim=0)
        else:
            hidden_padded = hidden

        T_padded = hidden_padded.shape[0]
        chunk_size = T_padded // sp_size
        hidden_local = hidden_padded[rank * chunk_size: (rank + 1) * chunk_size].contiguous()

        with torch.no_grad():
            sp_out_local = dualkv_attention_with_sp(
                attn, hidden_local, P, cu_seqlens_dec, max_dec, sp_group, pad_size,
            )

        # Gather SP outputs across ranks
        gathered = [torch.zeros_like(sp_out_local) for _ in range(sp_size)]
        dist.all_gather(gathered, sp_out_local, group=sp_group)
        sp_out_full = torch.cat(gathered, dim=0)

        # Remove padding
        if pad_size > 0:
            sp_out_full = sp_out_full[:T]

        # Compare
        match = torch.allclose(ref_out, sp_out_full, atol=1e-3, rtol=1e-3)
        max_err = (ref_out - sp_out_full).abs().max().item()
        mean_err = (ref_out - sp_out_full).abs().mean().item()

        if rank == 0:
            status = "PASS" if match else "FAIL"
            print(f"  P={P}, N={n_responses}, R={response_lens}: {status} "
                  f"(max_err={max_err:.2e}, mean_err={mean_err:.2e})")

        if not match:
            all_pass = False

        del attn, hidden, ref_out, sp_out_local
        torch.cuda.empty_cache()

    if rank == 0:
        print()
        if all_pass:
            print("ALL TESTS PASSED: DualKV+SP=2 matches DualKV reference.")
        else:
            print("*** SOME TESTS FAILED ***")
        sys.exit(0 if all_pass else 1)

    dist.destroy_process_group()


if __name__ == "__main__":
    run_test()
