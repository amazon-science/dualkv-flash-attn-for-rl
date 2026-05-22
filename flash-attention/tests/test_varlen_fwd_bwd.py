"""
Validate flash_attn_varlen_func forward + backward against
a tiled FA2 simulation (GPU, float32).

Test scenario: bs=20 sequences sharing the same 4k prompt + different 1k responses.
Reference: lean tiled FA2 (no N×N storage) on GPU in float32.
Test: flash_attn_varlen_func on GPU in fp16.
"""
import torch
import torch.nn.functional as F
import time


# ================================================================
# Lean tiled FA2 forward (GPU, float32, no S_full accumulation)
# ================================================================

def lean_flash_fwd(Q, K, V, causal=True, Br=128, Bc=128):
    """
    Tiled FA2 forward with online softmax. No S_full/P_full storage.
    Q: (bs, nh, N, d)  K: (bs, nh_kv, N, d)  V: (bs, nh_kv, N, d)
    Returns: O, L (logsumexp)
    """
    bs, nh, N, d = Q.shape
    nh_kv = K.shape[1]
    groups = nh // nh_kv
    scale = d ** -0.5

    Ke = K.repeat_interleave(groups, dim=1) if groups > 1 else K
    Ve = V.repeat_interleave(groups, dim=1) if groups > 1 else V

    O = torch.zeros_like(Q)
    m = torch.full((bs, nh, N), float('-inf'), dtype=Q.dtype, device=Q.device)
    ell = torch.zeros((bs, nh, N), dtype=Q.dtype, device=Q.device)

    Tr = (N + Br - 1) // Br
    Tc = (N + Bc - 1) // Bc

    for i in range(Tr):
        qi, qe = i * Br, min((i + 1) * Br, N)
        Qi = Q[:, :, qi:qe]
        mi = m[:, :, qi:qe].clone()
        li = ell[:, :, qi:qe].clone()
        Oi = O[:, :, qi:qe].clone()

        for j in range(Tc):
            kj, ke = j * Bc, min((j + 1) * Bc, N)
            if causal and kj > qe - 1:
                break

            Kj = Ke[:, :, kj:ke]
            Vj = Ve[:, :, kj:ke]

            Sij = (Qi @ Kj.transpose(-2, -1)) * scale

            if causal:
                qp = torch.arange(qi, qe, device=Q.device).unsqueeze(1)
                kp = torch.arange(kj, ke, device=Q.device).unsqueeze(0)
                cmask = (kp > qp).unsqueeze(0).unsqueeze(0)
                Sij = Sij.masked_fill(cmask, float('-inf'))

            mij = Sij.max(dim=-1).values
            new_m = torch.maximum(mi, mij)

            alpha = torch.exp(mi - new_m)
            alpha = torch.where(torch.isinf(mi) & (mi < 0), torch.zeros_like(alpha), alpha)

            Pij = torch.exp(Sij - new_m.unsqueeze(-1))
            Pij = torch.where(torch.isinf(Sij) & (Sij < 0), torch.zeros_like(Pij), Pij)

            lij = Pij.sum(dim=-1)

            Oi = Oi * alpha.unsqueeze(-1) + Pij @ Vj
            li = li * alpha + lij
            mi = new_m

        safe_li = torch.where(li == 0, torch.ones_like(li), li)
        O[:, :, qi:qe] = Oi / safe_li.unsqueeze(-1)
        m[:, :, qi:qe] = mi
        ell[:, :, qi:qe] = li

    safe_ell = torch.where(ell == 0, torch.ones_like(ell), ell)
    L = m + torch.log(safe_ell)
    return O, L


# ================================================================
# Lean tiled FA2 backward (GPU, float32, no S_full accumulation)
# ================================================================

def lean_flash_bwd(dO, Q, K, V, O, L, causal=True, Br=128, Bc=128):
    """
    Tiled FA2 backward. Recomputes S/P tile-by-tile from L.
    No dS_full/dP_full storage.
    Returns: dQ, dK, dV
    """
    bs, nh, N, d = Q.shape
    nh_kv = K.shape[1]
    groups = nh // nh_kv
    scale = d ** -0.5

    D = (dO * O).sum(dim=-1)  # (bs, nh, N)

    Ke = K.repeat_interleave(groups, dim=1) if groups > 1 else K
    Ve = V.repeat_interleave(groups, dim=1) if groups > 1 else V

    dQ = torch.zeros_like(Q)
    dK_e = torch.zeros(bs, nh, N, d, dtype=Q.dtype, device=Q.device)
    dV_e = torch.zeros(bs, nh, N, d, dtype=Q.dtype, device=Q.device)

    Tr = (N + Br - 1) // Br
    Tc = (N + Bc - 1) // Bc

    for j in range(Tc):
        kj, ke = j * Bc, min((j + 1) * Bc, N)
        Kj = Ke[:, :, kj:ke]
        Vj = Ve[:, :, kj:ke]

        dKj = torch.zeros_like(Kj)
        dVj = torch.zeros_like(Vj)

        for i in range(Tr):
            qi, qe = i * Br, min((i + 1) * Br, N)
            if causal and kj > qe - 1:
                continue
            if causal and qe - 1 < kj:
                continue

            Qi = Q[:, :, qi:qe]
            dOi = dO[:, :, qi:qe]
            Li = L[:, :, qi:qe]
            Di = D[:, :, qi:qe]

            Sij = (Qi @ Kj.transpose(-2, -1)) * scale

            if causal:
                qp = torch.arange(qi, qe, device=Q.device).unsqueeze(1)
                kp = torch.arange(kj, ke, device=Q.device).unsqueeze(0)
                cmask = (kp > qp).unsqueeze(0).unsqueeze(0)
                Sij = Sij.masked_fill(cmask, float('-inf'))

            Pij = torch.exp(Sij - Li.unsqueeze(-1))
            Pij = torch.nan_to_num(Pij, 0.0)

            dVj += Pij.transpose(-2, -1) @ dOi
            dPij = dOi @ Vj.transpose(-2, -1)
            dSij = Pij * (dPij - Di.unsqueeze(-1))
            if causal:
                dSij = dSij.masked_fill(cmask, 0.0)

            dQ[:, :, qi:qe] += (dSij @ Kj) * scale
            dKj += (dSij.transpose(-2, -1) @ Qi) * scale

        dK_e[:, :, kj:ke] = dKj
        dV_e[:, :, kj:ke] = dVj

    # Reduce GQA
    if groups > 1:
        dK = dK_e.reshape(bs, nh_kv, groups, N, d).sum(2)
        dV = dV_e.reshape(bs, nh_kv, groups, N, d).sum(2)
    else:
        dK = dK_e
        dV = dV_e

    return dQ, dK, dV


# ================================================================
# DualKV tile helpers
# ================================================================

def _load_kv_tile(K_ctx, K_dec, V_ctx, V_dec, kj, ke, P, bs, groups):
    """
    Load a KV tile spanning absolute positions [kj, ke).
    Context: 0..P-1 from K_ctx (bs=1, broadcast to bs).
    Decoded: P..P+R-1 from K_dec (bs=N).
    Boundary tiles: torch.cat context + decoded parts.
    Returns Kj, Vj both (bs, nh, tile_len, d) with GQA expansion.
    """
    if ke <= P:
        Kj = K_ctx[:, :, kj:ke].expand(bs, -1, -1, -1).repeat_interleave(groups, dim=1)
        Vj = V_ctx[:, :, kj:ke].expand(bs, -1, -1, -1).repeat_interleave(groups, dim=1)
    elif kj >= P:
        dj, de = kj - P, ke - P
        Kj = K_dec[:, :, dj:de].repeat_interleave(groups, dim=1)
        Vj = V_dec[:, :, dj:de].repeat_interleave(groups, dim=1)
    else:
        Kc = K_ctx[:, :, kj:P].expand(bs, -1, -1, -1).repeat_interleave(groups, dim=1)
        Vc = V_ctx[:, :, kj:P].expand(bs, -1, -1, -1).repeat_interleave(groups, dim=1)
        de = ke - P
        Kd = K_dec[:, :, :de].repeat_interleave(groups, dim=1)
        Vd = V_dec[:, :, :de].repeat_interleave(groups, dim=1)
        Kj = torch.cat([Kc, Kd], dim=2)
        Vj = torch.cat([Vc, Vd], dim=2)
    return Kj, Vj


def _scatter_kv_grad(dKj, dVj, dK_ctx_e, dV_ctx_e, dK_dec_e, dV_dec_e, kj, ke, P):
    """
    Scatter tile gradients back to context or decoded accumulators.
    dKj/dVj: (bs, nh, tile_len, d) — expanded heads.
    """
    if ke <= P:
        dK_ctx_e[:, :, kj:ke] += dKj
        dV_ctx_e[:, :, kj:ke] += dVj
    elif kj >= P:
        dj, de = kj - P, ke - P
        dK_dec_e[:, :, dj:de] += dKj
        dV_dec_e[:, :, dj:de] += dVj
    else:
        ctx_len = P - kj
        dK_ctx_e[:, :, kj:P] += dKj[:, :, :ctx_len]
        dV_ctx_e[:, :, kj:P] += dVj[:, :, :ctx_len]
        de = ke - P
        dK_dec_e[:, :, :de] += dKj[:, :, ctx_len:]
        dV_dec_e[:, :, :de] += dVj[:, :, ctx_len:]


# ================================================================
# Lean DualKV forward (GPU, float32, no S_full)
# ================================================================

def lean_dualkv_fwd(Q, K_ctx, K_dec, V_ctx, V_dec, causal=True, Br=128, Bc=128):
    """
    Tiled DualKV forward with online softmax. No S_full/P_full storage.
    Q:     (bs, nh, S, d)       S = P + R
    K_ctx: (1, nh_kv, P, d)     shared context
    K_dec: (bs, nh_kv, R, d)    per-sequence decoded
    V_ctx: (1, nh_kv, P, d)
    V_dec: (bs, nh_kv, R, d)
    Returns: O, L (logsumexp)
    """
    bs, nh, S, d = Q.shape
    nh_kv = K_ctx.shape[1]
    P = K_ctx.shape[2]
    R = K_dec.shape[2]
    assert S == P + R
    groups = nh // nh_kv
    scale = d ** -0.5

    O = torch.zeros_like(Q)
    m = torch.full((bs, nh, S), float('-inf'), dtype=Q.dtype, device=Q.device)
    ell = torch.zeros((bs, nh, S), dtype=Q.dtype, device=Q.device)

    Tr = (S + Br - 1) // Br
    Tc = (S + Bc - 1) // Bc

    for i in range(Tr):
        qi, qe = i * Br, min((i + 1) * Br, S)
        Qi = Q[:, :, qi:qe]
        mi = m[:, :, qi:qe].clone()
        li = ell[:, :, qi:qe].clone()
        Oi = O[:, :, qi:qe].clone()

        for j in range(Tc):
            kj, ke = j * Bc, min((j + 1) * Bc, S)
            if causal and kj > qe - 1:
                break

            Kj, Vj = _load_kv_tile(K_ctx, K_dec, V_ctx, V_dec, kj, ke, P, bs, groups)

            Sij = (Qi @ Kj.transpose(-2, -1)) * scale

            if causal:
                qp = torch.arange(qi, qe, device=Q.device).unsqueeze(1)
                kp = torch.arange(kj, ke, device=Q.device).unsqueeze(0)
                cmask = (kp > qp).unsqueeze(0).unsqueeze(0)
                Sij = Sij.masked_fill(cmask, float('-inf'))

            mij = Sij.max(dim=-1).values
            new_m = torch.maximum(mi, mij)

            alpha = torch.exp(mi - new_m)
            alpha = torch.where(torch.isinf(mi) & (mi < 0), torch.zeros_like(alpha), alpha)

            Pij = torch.exp(Sij - new_m.unsqueeze(-1))
            Pij = torch.where(torch.isinf(Sij) & (Sij < 0), torch.zeros_like(Pij), Pij)

            lij = Pij.sum(dim=-1)

            Oi = Oi * alpha.unsqueeze(-1) + Pij @ Vj
            li = li * alpha + lij
            mi = new_m

        safe_li = torch.where(li == 0, torch.ones_like(li), li)
        O[:, :, qi:qe] = Oi / safe_li.unsqueeze(-1)
        m[:, :, qi:qe] = mi
        ell[:, :, qi:qe] = li

    safe_ell = torch.where(ell == 0, torch.ones_like(ell), ell)
    L = m + torch.log(safe_ell)
    return O, L


# ================================================================
# Lean DualKV backward (GPU, float32, no dS_full)
# ================================================================

def lean_dualkv_bwd(dO, Q, K_ctx, K_dec, V_ctx, V_dec, O, L, causal=True, Br=128, Bc=128):
    """
    Tiled DualKV backward. Recomputes S/P tile-by-tile from L.
    dK_ctx/dV_ctx summed across batch (context is shared).
    Returns: dQ, dK_ctx, dK_dec, dV_ctx, dV_dec
    """
    bs, nh, S, d = Q.shape
    nh_kv = K_ctx.shape[1]
    P = K_ctx.shape[2]
    R = K_dec.shape[2]
    groups = nh // nh_kv
    scale = d ** -0.5

    D = (dO * O).sum(dim=-1)  # (bs, nh, S)

    dQ = torch.zeros_like(Q)
    # Expanded-head accumulators
    dK_ctx_e = torch.zeros(bs, nh, P, d, dtype=Q.dtype, device=Q.device)
    dV_ctx_e = torch.zeros(bs, nh, P, d, dtype=Q.dtype, device=Q.device)
    dK_dec_e = torch.zeros(bs, nh, R, d, dtype=Q.dtype, device=Q.device)
    dV_dec_e = torch.zeros(bs, nh, R, d, dtype=Q.dtype, device=Q.device)

    Tr = (S + Br - 1) // Br
    Tc = (S + Bc - 1) // Bc

    for j in range(Tc):
        kj, ke = j * Bc, min((j + 1) * Bc, S)

        Kj, Vj = _load_kv_tile(K_ctx, K_dec, V_ctx, V_dec, kj, ke, P, bs, groups)

        dKj = torch.zeros_like(Kj)
        dVj = torch.zeros_like(Vj)

        for i in range(Tr):
            qi, qe = i * Br, min((i + 1) * Br, S)
            if causal and kj > qe - 1:
                continue
            if causal and qe - 1 < kj:
                continue

            Qi = Q[:, :, qi:qe]
            dOi = dO[:, :, qi:qe]
            Li = L[:, :, qi:qe]
            Di = D[:, :, qi:qe]

            Sij = (Qi @ Kj.transpose(-2, -1)) * scale

            if causal:
                qp = torch.arange(qi, qe, device=Q.device).unsqueeze(1)
                kp = torch.arange(kj, ke, device=Q.device).unsqueeze(0)
                cmask = (kp > qp).unsqueeze(0).unsqueeze(0)
                Sij = Sij.masked_fill(cmask, float('-inf'))

            Pij = torch.exp(Sij - Li.unsqueeze(-1))
            Pij = torch.nan_to_num(Pij, 0.0)

            dVj += Pij.transpose(-2, -1) @ dOi
            dPij = dOi @ Vj.transpose(-2, -1)
            dSij = Pij * (dPij - Di.unsqueeze(-1))
            if causal:
                dSij = dSij.masked_fill(cmask, 0.0)

            dQ[:, :, qi:qe] += (dSij @ Kj) * scale
            dKj += (dSij.transpose(-2, -1) @ Qi) * scale

        _scatter_kv_grad(dKj, dVj, dK_ctx_e, dV_ctx_e, dK_dec_e, dV_dec_e, kj, ke, P)

    # Reduce GQA, then sum context across batch
    dK_ctx = dK_ctx_e.reshape(bs, nh_kv, groups, P, d).sum(2).sum(0, keepdim=True)  # (1, nh_kv, P, d)
    dV_ctx = dV_ctx_e.reshape(bs, nh_kv, groups, P, d).sum(2).sum(0, keepdim=True)
    dK_dec = dK_dec_e.reshape(bs, nh_kv, groups, R, d).sum(2)  # (bs, nh_kv, R, d)
    dV_dec = dV_dec_e.reshape(bs, nh_kv, groups, R, d).sum(2)

    return dQ, dK_ctx, dK_dec, dV_ctx, dV_dec


# ================================================================
# Test utilities
# ================================================================

def check(name, ref, test, tol, rtol=None):
    """Compare ref (float32) vs test (converted from fp16)."""
    abs_d = (ref - test).abs().max().item()
    ref_max = ref.abs().max().item()
    rel_d = abs_d / (ref_max + 1e-10) if rtol is not None else None
    use_tol = rtol if rtol is not None else tol
    val = rel_d if rtol is not None else abs_d
    ok = val < use_tol
    if rtol is not None:
        print(f"    {name:8s}: abs={abs_d:.2e}  rel={rel_d:.2e}  [{'PASS' if ok else 'FAIL'}]")
    else:
        print(f"    {name:8s}: abs={abs_d:.2e}  [{'PASS' if ok else 'FAIL'}]")
    return ok


def run_test(label, bs, P, R, nh, nh_kv, hd, causal=True, Br=128, Bc=128, fwd_tol=5e-3, bwd_rtol=5e-2):
    """
    Test flash_attn_varlen_func fwd+bwd against lean tiled simulation.
    All bs sequences share the same prompt of length P, each has a unique
    response of length R. Total seqlen S = P + R.
    """
    from flash_attn import flash_attn_varlen_func

    S = P + R
    groups = nh // nh_kv
    device = 'cuda'

    print(f"\n{'='*70}")
    print(f"{label}")
    print(f"  bs={bs}, P={P}, R={R}, S={S}, nh={nh}, nh_kv={nh_kv}, hd={hd}")
    print(f"  GQA ratio={groups}:1, causal={causal}, Br={Br}, Bc={Bc}")
    print(f"  fwd_tol={fwd_tol:.0e}, bwd_rtol={bwd_rtol:.0e}")
    print(f"{'='*70}")

    torch.manual_seed(42)

    # ---- Generate data with shared prompt ----
    # Prompt K/V: shared across batch (generate once, expand)
    K_prompt = torch.randn(1, nh_kv, P, hd, device=device, dtype=torch.float32)
    V_prompt = torch.randn(1, nh_kv, P, hd, device=device, dtype=torch.float32)

    # Response K/V: per-sequence
    K_resp = torch.randn(bs, nh_kv, R, hd, device=device, dtype=torch.float32)
    V_resp = torch.randn(bs, nh_kv, R, hd, device=device, dtype=torch.float32)

    # Full Q: per-sequence (covers both prompt and response positions)
    Q_full = torch.randn(bs, nh, S, hd, device=device, dtype=torch.float32)

    # Full K/V: concat shared prompt + per-sequence response
    K_full = torch.cat([K_prompt.expand(bs, -1, -1, -1), K_resp], dim=2)
    V_full = torch.cat([V_prompt.expand(bs, -1, -1, -1), V_resp], dim=2)

    # dO for backward
    dO_full = torch.randn(bs, nh, S, hd, device=device, dtype=torch.float32)

    # ---- Reference: lean tiled FA2 simulation (float32, GPU) ----
    t0 = time.perf_counter()
    O_ref, L_ref = lean_flash_fwd(Q_full, K_full, V_full, causal=causal, Br=Br, Bc=Bc)
    torch.cuda.synchronize()
    t_fwd_ref = time.perf_counter() - t0

    t0 = time.perf_counter()
    dQ_ref, dK_ref, dV_ref = lean_flash_bwd(
        dO_full, Q_full, K_full, V_full, O_ref, L_ref, causal=causal, Br=Br, Bc=Bc)
    torch.cuda.synchronize()
    t_bwd_ref = time.perf_counter() - t0

    print(f"\n  Reference (tiled sim, fp32): fwd={t_fwd_ref:.2f}s, bwd={t_bwd_ref:.2f}s")

    # ---- Test: flash_attn_varlen_func (fp16) ----
    # flash_attn_varlen_func expects packed layout: (total_tokens, nheads, headdim)
    # All sequences have same length S, so cu_seqlens = [0, S, 2S, ..., bs*S]
    cu_seqlens = torch.arange(0, (bs + 1) * S, S, device=device, dtype=torch.int32)

    # Convert to (bs*S, nheads, headdim) packed format, fp16
    # Q: (bs, nh, S, hd) -> (bs, S, nh, hd) -> (bs*S, nh, hd)
    Q_packed = Q_full.permute(0, 2, 1, 3).contiguous().reshape(bs * S, nh, hd).half().requires_grad_(True)
    K_packed = K_full.permute(0, 2, 1, 3).contiguous().reshape(bs * S, nh_kv, hd).half().requires_grad_(True)
    V_packed = V_full.permute(0, 2, 1, 3).contiguous().reshape(bs * S, nh_kv, hd).half().requires_grad_(True)
    dO_packed = dO_full.permute(0, 2, 1, 3).contiguous().reshape(bs * S, nh, hd).half()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    O_cuda, lse_cuda, _ = flash_attn_varlen_func(
        Q_packed, K_packed, V_packed,
        cu_seqlens, cu_seqlens, S, S,
        causal=causal, return_attn_probs=True
    )
    torch.cuda.synchronize()
    t_fwd_cuda = time.perf_counter() - t0

    t0 = time.perf_counter()
    O_cuda.backward(dO_packed)
    torch.cuda.synchronize()
    t_bwd_cuda = time.perf_counter() - t0

    print(f"  CUDA kernel (fp16):          fwd={t_fwd_cuda*1000:.1f}ms, bwd={t_bwd_cuda*1000:.1f}ms")

    # ---- Convert CUDA outputs to (bs, nh, S, hd) float32 for comparison ----
    O_cuda_cmp = O_cuda.detach().float().reshape(bs, S, nh, hd).permute(0, 2, 1, 3)
    # lse_cuda: (nheads, total_q) for varlen -> reshape to (bs, nh, S)
    # Actually flash_attn_varlen_func with return_attn_probs returns lse as (nheads, total_q)
    # but with unpadded_lse=True format
    L_cuda_cmp = lse_cuda.float()
    # Reshape L_ref to match: (bs, nh, S) -> (nh, bs*S)
    L_ref_cmp = L_ref.permute(1, 0, 2).reshape(nh, bs * S)

    dQ_cuda_cmp = Q_packed.grad.float().reshape(bs, S, nh, hd).permute(0, 2, 1, 3)
    dK_cuda_cmp = K_packed.grad.float().reshape(bs, S, nh_kv, hd).permute(0, 2, 1, 3)
    dV_cuda_cmp = V_packed.grad.float().reshape(bs, S, nh_kv, hd).permute(0, 2, 1, 3)

    # ---- Compare ----
    print("\n  Forward:")
    all_pass = True
    all_pass &= check("O", O_ref, O_cuda_cmp, tol=fwd_tol)
    all_pass &= check("L", L_ref_cmp, L_cuda_cmp, tol=fwd_tol)

    print("\n  Backward:")
    all_pass &= check("dQ", dQ_ref, dQ_cuda_cmp, tol=None, rtol=bwd_rtol)
    all_pass &= check("dK", dK_ref, dK_cuda_cmp, tol=None, rtol=bwd_rtol)
    all_pass &= check("dV", dV_ref, dV_cuda_cmp, tol=None, rtol=bwd_rtol)

    print(f"\n  {'ALL PASS' if all_pass else 'FAIL'}")
    return all_pass


def run_dualkv_test(label, bs, P, R, nh, nh_kv, hd, causal=True, Br=128, Bc=128, fwd_tol=5e-3, bwd_rtol=5e-2):
    """
    Test DualKV simulation fwd+bwd against:
      1. FA2 simulation (exact match expected — both float32)
      2. flash_attn_varlen_func CUDA kernel (fp16 tolerance)
    """
    from flash_attn import flash_attn_varlen_func

    S = P + R
    groups = nh // nh_kv
    device = 'cuda'

    print(f"\n{'='*70}")
    print(f"DualKV: {label}")
    print(f"  bs={bs}, P={P}, R={R}, S={S}, nh={nh}, nh_kv={nh_kv}, hd={hd}")
    print(f"  GQA ratio={groups}:1, causal={causal}, Br={Br}, Bc={Bc}")
    print(f"{'='*70}")

    torch.manual_seed(42)

    # ---- Generate data with shared prompt ----
    K_ctx = torch.randn(1, nh_kv, P, hd, device=device, dtype=torch.float32)
    V_ctx = torch.randn(1, nh_kv, P, hd, device=device, dtype=torch.float32)
    K_dec = torch.randn(bs, nh_kv, R, hd, device=device, dtype=torch.float32)
    V_dec = torch.randn(bs, nh_kv, R, hd, device=device, dtype=torch.float32)
    Q = torch.randn(bs, nh, S, hd, device=device, dtype=torch.float32)
    dO = torch.randn(bs, nh, S, hd, device=device, dtype=torch.float32)

    # Full K/V for FA2 sim and CUDA comparison
    K_full = torch.cat([K_ctx.expand(bs, -1, -1, -1), K_dec], dim=2)
    V_full = torch.cat([V_ctx.expand(bs, -1, -1, -1), V_dec], dim=2)

    # ---- DualKV simulation (float32, GPU) ----
    t0 = time.perf_counter()
    O_dk, L_dk = lean_dualkv_fwd(Q, K_ctx, K_dec, V_ctx, V_dec, causal=causal, Br=Br, Bc=Bc)
    torch.cuda.synchronize()
    t_fwd_dk = time.perf_counter() - t0

    t0 = time.perf_counter()
    dQ_dk, dK_ctx_dk, dK_dec_dk, dV_ctx_dk, dV_dec_dk = lean_dualkv_bwd(
        dO, Q, K_ctx, K_dec, V_ctx, V_dec, O_dk, L_dk, causal=causal, Br=Br, Bc=Bc)
    torch.cuda.synchronize()
    t_bwd_dk = time.perf_counter() - t0

    print(f"\n  DualKV sim (fp32):  fwd={t_fwd_dk:.2f}s, bwd={t_bwd_dk:.2f}s")

    # ---- FA2 simulation (float32, GPU) ----
    t0 = time.perf_counter()
    O_fa2, L_fa2 = lean_flash_fwd(Q, K_full, V_full, causal=causal, Br=Br, Bc=Bc)
    torch.cuda.synchronize()
    t_fwd_fa2 = time.perf_counter() - t0

    t0 = time.perf_counter()
    dQ_fa2, dK_fa2, dV_fa2 = lean_flash_bwd(
        dO, Q, K_full, V_full, O_fa2, L_fa2, causal=causal, Br=Br, Bc=Bc)
    torch.cuda.synchronize()
    t_bwd_fa2 = time.perf_counter() - t0

    print(f"  FA2 sim (fp32):     fwd={t_fwd_fa2:.2f}s, bwd={t_bwd_fa2:.2f}s")

    # ---- Reconstruct full dK/dV from DualKV for comparison ----
    # dK_ctx is (1, nh_kv, P, d) summed across batch; expand back for comparison
    # FA2's dK_full[:, :, :P] = sum of per-batch context grads (since K_full[:,:,:P] was broadcast)
    # But lean_flash_bwd returns per-batch dK — each batch element has its own dK[:,:,:P]
    # To compare: dK_dk_full per batch = dK_ctx expanded (divided by nothing — it's already the total)
    # Actually: FA2 sim treats K_full as (bs, nh_kv, S, d) independent per batch.
    # So dK_fa2[:, :, :P] is per-batch, NOT summed. We need per-batch DualKV context grads too.
    # But lean_dualkv_bwd sums context grads across batch. Let me reconstruct differently.

    # For the FA2 comparison, we need to compare O and L directly (these are identical regardless).
    # For gradients: dQ is per-batch in both. dK_dec and dV_dec are per-batch in both.
    # For context: FA2 gives dK_fa2[:,:,:P] per-batch; DualKV gives dK_ctx summed.
    # The relationship: dK_ctx_dk == dK_fa2[:,:,:P].sum(dim=0, keepdim=True)
    # This is correct because K_ctx was shared (broadcast), so the true gradient wrt the shared
    # parameter is the sum across batch elements.

    all_pass = True

    # ---- Comparison 1: DualKV sim vs FA2 sim (both fp32, expect exact match) ----
    print(f"\n  --- DualKV sim vs FA2 sim (fp32, expect exact match) ---")

    print("  Forward:")
    all_pass &= check("O", O_fa2, O_dk, tol=1e-5)
    all_pass &= check("L", L_fa2, L_dk, tol=1e-5)

    print("  Backward:")
    all_pass &= check("dQ", dQ_fa2, dQ_dk, tol=1e-5)
    # Context grads: FA2 per-batch vs DualKV summed
    dK_ctx_from_fa2 = dK_fa2[:, :, :P, :].sum(dim=0, keepdim=True)
    dV_ctx_from_fa2 = dV_fa2[:, :, :P, :].sum(dim=0, keepdim=True)
    all_pass &= check("dK_ctx", dK_ctx_from_fa2, dK_ctx_dk, tol=1e-4)
    all_pass &= check("dV_ctx", dV_ctx_from_fa2, dV_ctx_dk, tol=1e-4)
    # Decoded grads: both per-batch
    all_pass &= check("dK_dec", dK_fa2[:, :, P:, :], dK_dec_dk, tol=1e-5)
    all_pass &= check("dV_dec", dV_fa2[:, :, P:, :], dV_dec_dk, tol=1e-5)

    # ---- Comparison 2: DualKV sim vs CUDA kernel (fp32 vs fp16) ----
    print(f"\n  --- DualKV sim vs CUDA kernel (fp16 tolerance) ---")

    cu_seqlens = torch.arange(0, (bs + 1) * S, S, device=device, dtype=torch.int32)

    Q_packed = Q.permute(0, 2, 1, 3).contiguous().reshape(bs * S, nh, hd).half().requires_grad_(True)
    K_packed = K_full.permute(0, 2, 1, 3).contiguous().reshape(bs * S, nh_kv, hd).half().requires_grad_(True)
    V_packed = V_full.permute(0, 2, 1, 3).contiguous().reshape(bs * S, nh_kv, hd).half().requires_grad_(True)
    dO_packed = dO.permute(0, 2, 1, 3).contiguous().reshape(bs * S, nh, hd).half()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    O_cuda, lse_cuda, _ = flash_attn_varlen_func(
        Q_packed, K_packed, V_packed,
        cu_seqlens, cu_seqlens, S, S,
        causal=causal, return_attn_probs=True
    )
    torch.cuda.synchronize()
    t_fwd_cuda = time.perf_counter() - t0

    t0 = time.perf_counter()
    O_cuda.backward(dO_packed)
    torch.cuda.synchronize()
    t_bwd_cuda = time.perf_counter() - t0

    print(f"  CUDA kernel (fp16): fwd={t_fwd_cuda*1000:.1f}ms, bwd={t_bwd_cuda*1000:.1f}ms")

    O_cuda_cmp = O_cuda.detach().float().reshape(bs, S, nh, hd).permute(0, 2, 1, 3)
    dQ_cuda_cmp = Q_packed.grad.float().reshape(bs, S, nh, hd).permute(0, 2, 1, 3)
    dK_cuda_cmp = K_packed.grad.float().reshape(bs, S, nh_kv, hd).permute(0, 2, 1, 3)
    dV_cuda_cmp = V_packed.grad.float().reshape(bs, S, nh_kv, hd).permute(0, 2, 1, 3)

    print("  Forward:")
    all_pass &= check("O", O_dk, O_cuda_cmp, tol=fwd_tol)

    print("  Backward:")
    all_pass &= check("dQ", dQ_dk, dQ_cuda_cmp, tol=None, rtol=bwd_rtol)
    # For CUDA comparison, reconstruct full dK/dV from DualKV
    # DualKV dK_ctx is summed across batch, but CUDA dK is per-batch.
    # Use FA2 relationship: compare per-batch decoded + per-batch context from FA2
    # Actually simpler: just compare against dK_fa2 which we know matches DualKV.
    # Or: compare CUDA per-batch against FA2 per-batch (already validated in run_test).
    # For DualKV-specific: compare decoded grads directly, context grads summed.
    dK_ctx_cuda = dK_cuda_cmp[:, :, :P, :].sum(dim=0, keepdim=True)
    dV_ctx_cuda = dV_cuda_cmp[:, :, :P, :].sum(dim=0, keepdim=True)
    all_pass &= check("dK_ctx", dK_ctx_dk, dK_ctx_cuda, tol=None, rtol=bwd_rtol)
    all_pass &= check("dV_ctx", dV_ctx_dk, dV_ctx_cuda, tol=None, rtol=bwd_rtol)
    all_pass &= check("dK_dec", dK_dec_dk, dK_cuda_cmp[:, :, P:, :], tol=None, rtol=bwd_rtol)
    all_pass &= check("dV_dec", dV_dec_dk, dV_cuda_cmp[:, :, P:, :], tol=None, rtol=bwd_rtol)

    print(f"\n  {'ALL PASS' if all_pass else 'FAIL'}")
    return all_pass


# ================================================================
# DualKV CUDA kernel test (flash_attn_dualkv_varlen_func)
# ================================================================

def run_dualkv_cuda_test(label, bs, P, R, nh, nh_kv, hd, causal=True,
                         fwd_tol=5e-3, bwd_rtol=0.1):
    """
    Test flash_attn_dualkv_varlen_func CUDA kernel (fwd + bwd)
    against lean DualKV tiled simulation (float32 reference).
    """
    from flash_attn import flash_attn_dualkv_varlen_func

    S = P + R
    groups = nh // nh_kv
    device = 'cuda'

    print(f"\n{'='*70}")
    print(f"DualKV CUDA: {label}")
    print(f"  bs={bs}, P={P}, R={R}, S={S}, nh={nh}, nh_kv={nh_kv}, hd={hd}")
    print(f"  GQA ratio={groups}:1, causal={causal}")
    print(f"{'='*70}")

    torch.manual_seed(42)

    # Generate data (float32 for reference)
    K_ctx = torch.randn(1, nh_kv, P, hd, device=device, dtype=torch.float32)
    V_ctx = torch.randn(1, nh_kv, P, hd, device=device, dtype=torch.float32)
    K_dec = torch.randn(bs, nh_kv, R, hd, device=device, dtype=torch.float32)
    V_dec = torch.randn(bs, nh_kv, R, hd, device=device, dtype=torch.float32)
    Q = torch.randn(bs, nh, S, hd, device=device, dtype=torch.float32)
    dO = torch.randn(bs, nh, S, hd, device=device, dtype=torch.float32)

    # ---- Reference: lean DualKV simulation (float32) ----
    O_ref, L_ref = lean_dualkv_fwd(Q, K_ctx, K_dec, V_ctx, V_dec, causal=causal)
    dQ_ref, dK_ctx_ref, dK_dec_ref, dV_ctx_ref, dV_dec_ref = lean_dualkv_bwd(
        dO, Q, K_ctx, K_dec, V_ctx, V_dec, O_ref, L_ref, causal=causal)

    # ---- CUDA kernel (fp16) ----
    # Pack Q: (bs, nh, S, hd) -> (total_q, nh, hd)
    Q_packed = Q.permute(0, 2, 1, 3).contiguous().reshape(bs * S, nh, hd).half().requires_grad_(True)

    # K/V context: (1, nh_kv, P, hd) -> (P, nh_kv, hd)
    Kc_packed = K_ctx[0].permute(1, 0, 2).contiguous().half().requires_grad_(True)
    Vc_packed = V_ctx[0].permute(1, 0, 2).contiguous().half().requires_grad_(True)

    # K/V decoded: (bs, nh_kv, R, hd) -> (bs*R, nh_kv, hd) packed
    Kd_packed = K_dec.permute(0, 2, 1, 3).contiguous().reshape(bs * R, nh_kv, hd).half().requires_grad_(True)
    Vd_packed = V_dec.permute(0, 2, 1, 3).contiguous().reshape(bs * R, nh_kv, hd).half().requires_grad_(True)

    cu_seqlens_q = torch.arange(0, (bs + 1) * S, S, device=device, dtype=torch.int32)
    cu_seqlens_k_decoded = torch.arange(0, (bs + 1) * R, R, device=device, dtype=torch.int32)

    dO_packed = dO.permute(0, 2, 1, 3).contiguous().reshape(bs * S, nh, hd).half()

    torch.cuda.synchronize()
    O_cuda = flash_attn_dualkv_varlen_func(
        Q_packed, Kc_packed, Vc_packed, Kd_packed, Vd_packed,
        cu_seqlens_q, cu_seqlens_k_decoded,
        max_seqlen_q=S,
        context_seqlen=P,
        max_seqlen_k_decoded=R,
        causal=causal,
    )
    torch.cuda.synchronize()

    # Backward
    O_cuda.backward(dO_packed)
    torch.cuda.synchronize()

    # ---- Convert CUDA outputs for comparison ----
    O_cuda_cmp = O_cuda.detach().float().reshape(bs, S, nh, hd).permute(0, 2, 1, 3)

    dQ_cuda_cmp = Q_packed.grad.float().reshape(bs, S, nh, hd).permute(0, 2, 1, 3)
    dKc_cuda_cmp = Kc_packed.grad.float().permute(1, 0, 2).unsqueeze(0)  # (1, nh_kv, P, hd)
    dVc_cuda_cmp = Vc_packed.grad.float().permute(1, 0, 2).unsqueeze(0)
    dKd_cuda_cmp = Kd_packed.grad.float().reshape(bs, R, nh_kv, hd).permute(0, 2, 1, 3)
    dVd_cuda_cmp = Vd_packed.grad.float().reshape(bs, R, nh_kv, hd).permute(0, 2, 1, 3)

    # ---- Compare ----
    all_pass = True

    print("\n  Forward:")
    all_pass &= check("O", O_ref, O_cuda_cmp, tol=fwd_tol)

    print("\n  Backward:")
    all_pass &= check("dQ", dQ_ref, dQ_cuda_cmp, tol=None, rtol=bwd_rtol)
    all_pass &= check("dK_ctx", dK_ctx_ref, dKc_cuda_cmp, tol=None, rtol=bwd_rtol)
    all_pass &= check("dV_ctx", dV_ctx_ref, dVc_cuda_cmp, tol=None, rtol=bwd_rtol)
    all_pass &= check("dK_dec", dK_dec_ref, dKd_cuda_cmp, tol=None, rtol=bwd_rtol)
    all_pass &= check("dV_dec", dV_dec_ref, dVd_cuda_cmp, tol=None, rtol=bwd_rtol)

    print(f"\n  {'ALL PASS' if all_pass else 'FAIL'}")
    return all_pass


# ================================================================
# DualKV CUDA kernel test with non-uniform decoded lengths
# ================================================================

def run_dualkv_cuda_varlen_test(label, P, nh, nh_kv, hd, decoded_lens, causal=True,
                                atol=1e-2, rtol=1e-3):
    """
    Test flash_attn_dualkv_varlen_func with non-uniform decoded lengths per sequence.
    decoded_lens: list of int, one per batch element.
    Reference: per-sequence lean DualKV simulation (float32).
    """
    from flash_attn import flash_attn_dualkv_varlen_func

    bs = len(decoded_lens)
    max_R = max(decoded_lens)
    groups = nh // nh_kv
    device = 'cuda'

    print(f"\n{'='*70}")
    print(f"DualKV CUDA varlen: {label}")
    print(f"  bs={bs}, P={P}, decoded_lens={decoded_lens}, nh={nh}, nh_kv={nh_kv}, hd={hd}")
    print(f"  GQA ratio={groups}:1, causal={causal}, atol={atol}, rtol={rtol}")
    print(f"{'='*70}")

    torch.manual_seed(42)

    # Context KV: shared (1, nh_kv, P, hd)
    K_ctx = torch.randn(1, nh_kv, P, hd, device=device, dtype=torch.float32)
    V_ctx = torch.randn(1, nh_kv, P, hd, device=device, dtype=torch.float32)

    # Per-sequence decoded KV and Q with variable lengths
    # For reference: run lean_dualkv_fwd/bwd per sequence (each has S_i = P + R_i)
    O_refs = []
    L_refs = []
    dQ_refs = []
    dK_ctx_refs = []  # will sum across batch
    dV_ctx_refs = []
    dK_dec_refs = []
    dV_dec_refs = []

    # Generate all per-sequence data
    Qs = []
    K_decs = []
    V_decs = []
    dOs = []
    for i in range(bs):
        R_i = decoded_lens[i]
        S_i = P + R_i
        Qs.append(torch.randn(1, nh, S_i, hd, device=device, dtype=torch.float32))
        K_decs.append(torch.randn(1, nh_kv, R_i, hd, device=device, dtype=torch.float32))
        V_decs.append(torch.randn(1, nh_kv, R_i, hd, device=device, dtype=torch.float32))
        dOs.append(torch.randn(1, nh, S_i, hd, device=device, dtype=torch.float32))

    # Reference: per-sequence lean simulation
    for i in range(bs):
        R_i = decoded_lens[i]
        O_i, L_i = lean_dualkv_fwd(Qs[i], K_ctx, K_decs[i], V_ctx, V_decs[i], causal=causal)
        dQ_i, dKc_i, dKd_i, dVc_i, dVd_i = lean_dualkv_bwd(
            dOs[i], Qs[i], K_ctx, K_decs[i], V_ctx, V_decs[i], O_i, L_i, causal=causal)
        O_refs.append(O_i)
        L_refs.append(L_i)
        dQ_refs.append(dQ_i)
        dK_ctx_refs.append(dKc_i)
        dV_ctx_refs.append(dVc_i)
        dK_dec_refs.append(dKd_i)
        dV_dec_refs.append(dVd_i)

    # Sum context grads across batch (context is shared)
    dK_ctx_ref = sum(dK_ctx_refs)  # (1, nh_kv, P, hd)
    dV_ctx_ref = sum(dV_ctx_refs)

    # ---- Pack for CUDA kernel ----
    # Q: varlen packed (total_q, nh, hd)
    seqlens_q = [P + decoded_lens[i] for i in range(bs)]
    max_seqlen_q = max(seqlens_q)
    Q_packed_list = [Qs[i].squeeze(0).permute(1, 0, 2).contiguous() for i in range(bs)]  # each (S_i, nh, hd)
    Q_packed = torch.cat(Q_packed_list, dim=0).half().requires_grad_(True)

    cu_seqlens_q = torch.zeros(bs + 1, device=device, dtype=torch.int32)
    for i in range(bs):
        cu_seqlens_q[i + 1] = cu_seqlens_q[i] + seqlens_q[i]

    # K/V context: (P, nh_kv, hd)
    Kc_packed = K_ctx[0].permute(1, 0, 2).contiguous().half().requires_grad_(True)
    Vc_packed = V_ctx[0].permute(1, 0, 2).contiguous().half().requires_grad_(True)

    # K/V decoded: varlen packed (total_k_decoded, nh_kv, hd)
    Kd_list = [K_decs[i].squeeze(0).permute(1, 0, 2).contiguous() for i in range(bs)]  # each (R_i, nh_kv, hd)
    Vd_list = [V_decs[i].squeeze(0).permute(1, 0, 2).contiguous() for i in range(bs)]
    Kd_packed = torch.cat(Kd_list, dim=0).half().requires_grad_(True)
    Vd_packed = torch.cat(Vd_list, dim=0).half().requires_grad_(True)

    cu_seqlens_k_decoded = torch.zeros(bs + 1, device=device, dtype=torch.int32)
    for i in range(bs):
        cu_seqlens_k_decoded[i + 1] = cu_seqlens_k_decoded[i] + decoded_lens[i]

    # dO: varlen packed
    dO_packed_list = [dOs[i].squeeze(0).permute(1, 0, 2).contiguous() for i in range(bs)]
    dO_packed = torch.cat(dO_packed_list, dim=0).half()

    torch.cuda.synchronize()
    O_cuda = flash_attn_dualkv_varlen_func(
        Q_packed, Kc_packed, Vc_packed, Kd_packed, Vd_packed,
        cu_seqlens_q, cu_seqlens_k_decoded,
        max_seqlen_q=max_seqlen_q,
        context_seqlen=P,
        max_seqlen_k_decoded=max_R,
        causal=causal,
    )
    torch.cuda.synchronize()

    O_cuda.backward(dO_packed)
    torch.cuda.synchronize()

    # ---- Compare per-sequence ----
    def allclose_chk(name, ref, test):
        close = torch.allclose(ref, test, atol=atol, rtol=rtol)
        abs_err = (ref - test).abs().max().item()
        mask = ~torch.isclose(ref, test, atol=atol, rtol=rtol)
        n_fail = mask.sum().item()
        n_total = ref.numel()
        pct = 100.0 * n_fail / n_total
        status = "PASS" if close else "FAIL"
        print(f"    {name:10s}: allclose={close}  max_abs={abs_err:.2e}  fail={n_fail}/{n_total} ({pct:.2f}%)  [{status}]")
        return close

    all_pass = True
    print("\n  Forward:")

    offset_q = 0
    for i in range(bs):
        S_i = seqlens_q[i]
        O_cuda_i = O_cuda[offset_q:offset_q + S_i].detach().float().unsqueeze(0).permute(0, 2, 1, 3)
        all_pass &= allclose_chk(f"O[{i}]", O_refs[i], O_cuda_i)
        offset_q += S_i

    print("\n  Backward:")

    # dQ per-sequence
    offset_q = 0
    for i in range(bs):
        S_i = seqlens_q[i]
        dQ_i = Q_packed.grad[offset_q:offset_q + S_i].float().unsqueeze(0).permute(0, 2, 1, 3)
        all_pass &= allclose_chk(f"dQ[{i}]", dQ_refs[i], dQ_i)
        offset_q += S_i

    # dK_ctx, dV_ctx (summed across batch)
    dKc_cuda = Kc_packed.grad.float().permute(1, 0, 2).unsqueeze(0)
    dVc_cuda = Vc_packed.grad.float().permute(1, 0, 2).unsqueeze(0)
    all_pass &= allclose_chk("dK_ctx", dK_ctx_ref, dKc_cuda)
    all_pass &= allclose_chk("dV_ctx", dV_ctx_ref, dVc_cuda)

    # dK_dec, dV_dec per-sequence
    offset_k = 0
    for i in range(bs):
        R_i = decoded_lens[i]
        dKd_i = Kd_packed.grad[offset_k:offset_k + R_i].float().unsqueeze(0).permute(0, 2, 1, 3)
        dVd_i = Vd_packed.grad[offset_k:offset_k + R_i].float().unsqueeze(0).permute(0, 2, 1, 3)
        all_pass &= allclose_chk(f"dKd[{i}]", dK_dec_refs[i], dKd_i)
        all_pass &= allclose_chk(f"dVd[{i}]", dV_dec_refs[i], dVd_i)
        offset_k += R_i

    print(f"\n  {'ALL PASS' if all_pass else 'FAIL'}")
    return all_pass


# ================================================================
# Main
# ================================================================

if __name__ == "__main__":
    results = []

    # Small causal GQA
    results.append(run_test(
        "Small causal GQA 4:1",
        bs=4, P=64, R=16, nh=8, nh_kv=2, hd=64, causal=True, Br=32, Bc=32,
    ))

    # Small non-causal GQA
    results.append(run_test(
        "Small non-causal GQA 4:1",
        bs=4, P=64, R=16, nh=8, nh_kv=2, hd=64, causal=False, Br=32, Bc=32,
    ))

    # Medium causal GQA
    results.append(run_test(
        "Medium causal GQA 4:1, bs=20, P=256, R=64",
        bs=20, P=256, R=64, nh=32, nh_kv=8, hd=128, causal=True, Br=64, Bc=64,
    ))

    # Medium non-causal GQA
    results.append(run_test(
        "Medium non-causal GQA 4:1, bs=20, P=256, R=64",
        bs=20, P=256, R=64, nh=32, nh_kv=8, hd=128, causal=False, Br=64, Bc=64,
    ))

    # Full scale causal: bs=20, prompt=4096, response=1024
    results.append(run_test(
        "Full causal GQA 4:1, bs=20, P=4096, R=1024",
        bs=20, P=4096, R=1024, nh=32, nh_kv=8, hd=128, causal=True, Br=128, Bc=128,
    ))

    # Full scale non-causal: bs=20, prompt=4096, response=1024
    results.append(run_test(
        "Full non-causal GQA 4:1, bs=20, P=4096, R=1024",
        bs=20, P=4096, R=1024, nh=32, nh_kv=8, hd=128, causal=False, Br=128, Bc=128,
    ))

    # ---- Production model configs: FA2 ----

    # Qwen2-1.5B: nh=12, nh_kv=2, hd=128, GQA 6:1
    results.append(run_test(
        "Qwen2-1.5B config: GQA 6:1, causal",
        bs=20, P=4096, R=1024, nh=12, nh_kv=2, hd=128, causal=True, Br=128, Bc=128,
    ))

    # Llama-3-70B / Qwen2.5-72B: nh=64, nh_kv=8, hd=128, GQA 8:1
    results.append(run_test(
        "Llama-3-70B config: GQA 8:1, causal",
        bs=4, P=4096, R=1024, nh=64, nh_kv=8, hd=128, causal=True, Br=128, Bc=128,
    ))

    # ---- DualKV CUDA kernel tests (flash_attn_dualkv_varlen_func) ----

    # Small causal MHA
    results.append(run_dualkv_cuda_test(
        "Small causal MHA",
        bs=2, P=64, R=16, nh=4, nh_kv=4, hd=128, causal=True,
    ))

    # Small causal GQA
    results.append(run_dualkv_cuda_test(
        "Small causal GQA 4:1",
        bs=4, P=64, R=16, nh=8, nh_kv=2, hd=128, causal=True,
    ))

    # Small non-causal GQA
    results.append(run_dualkv_cuda_test(
        "Small non-causal GQA 4:1",
        bs=4, P=64, R=16, nh=8, nh_kv=2, hd=128, causal=False,
    ))

    # Medium causal GQA
    results.append(run_dualkv_cuda_test(
        "Medium causal GQA 4:1, bs=8, P=256, R=64",
        bs=8, P=256, R=64, nh=16, nh_kv=4, hd=128, causal=True,
    ))

    # Qwen2-1.5B config
    results.append(run_dualkv_cuda_test(
        "Qwen2-1.5B config: GQA 6:1, causal",
        bs=4, P=256, R=64, nh=12, nh_kv=2, hd=128, causal=True,
    ))

    # Larger batch + longer sequences
    results.append(run_dualkv_cuda_test(
        "Large causal GQA 4:1, bs=8, P=512, R=128",
        bs=8, P=512, R=128, nh=16, nh_kv=4, hd=128, causal=True,
    ))

    # ---- Gap 1: Head dimensions (hdim 64, 96, 192, 256) ----

    results.append(run_dualkv_cuda_test(
        "hdim=64, causal GQA 4:1",
        bs=4, P=64, R=16, nh=8, nh_kv=2, hd=64, causal=True,
    ))

    results.append(run_dualkv_cuda_test(
        "hdim=96, causal GQA 4:1",
        bs=4, P=64, R=16, nh=8, nh_kv=2, hd=96, causal=True,
    ))

    results.append(run_dualkv_cuda_test(
        "hdim=192, causal GQA 4:1",
        bs=4, P=64, R=16, nh=8, nh_kv=2, hd=192, causal=True,
    ))

    results.append(run_dualkv_cuda_test(
        "hdim=256, causal GQA 4:1",
        bs=4, P=64, R=16, nh=8, nh_kv=2, hd=256, causal=True,
    ))

    # ---- Gap 2: Edge cases (small P/R, unaligned to kBlockN=128) ----

    results.append(run_dualkv_cuda_test(
        "Edge: P=1, R=1",
        bs=2, P=1, R=1, nh=4, nh_kv=4, hd=128, causal=True,
    ))

    results.append(run_dualkv_cuda_test(
        "Edge: P=3, R=1",
        bs=2, P=3, R=1, nh=4, nh_kv=2, hd=128, causal=True,
    ))

    results.append(run_dualkv_cuda_test(
        "Edge: P=1, R=17 (unaligned)",
        bs=2, P=1, R=17, nh=4, nh_kv=4, hd=128, causal=True,
    ))

    results.append(run_dualkv_cuda_test(
        "Edge: P=33, R=7 (both unaligned)",
        bs=4, P=33, R=7, nh=8, nh_kv=2, hd=128, causal=True,
    ))

    results.append(run_dualkv_cuda_test(
        "Edge: P=127, R=129 (straddle kBlockN boundary)",
        bs=4, P=127, R=129, nh=8, nh_kv=2, hd=128, causal=True,
    ))

    results.append(run_dualkv_cuda_test(
        "Edge: P=128, R=1 (P exactly kBlockN)",
        bs=2, P=128, R=1, nh=4, nh_kv=4, hd=128, causal=True,
    ))

    results.append(run_dualkv_cuda_test(
        "Edge: P=129, R=127 (P just past kBlockN)",
        bs=4, P=129, R=127, nh=8, nh_kv=2, hd=128, causal=False,
    ))

    # ---- Gap 3: bs=1 (no cross-batch atomicAdd) ----

    results.append(run_dualkv_cuda_test(
        "bs=1, causal MHA",
        bs=1, P=64, R=16, nh=4, nh_kv=4, hd=128, causal=True,
    ))

    results.append(run_dualkv_cuda_test(
        "bs=1, causal GQA 4:1",
        bs=1, P=256, R=64, nh=8, nh_kv=2, hd=128, causal=True,
    ))

    results.append(run_dualkv_cuda_test(
        "bs=1, non-causal GQA 4:1",
        bs=1, P=128, R=32, nh=8, nh_kv=2, hd=128, causal=False,
    ))

    # ---- Gap 4: Large scale (P=4096+, heavy atomicAdd accumulation) ----

    results.append(run_dualkv_cuda_test(
        "Large: bs=16, P=4096, R=256, GQA 4:1",
        bs=16, P=4096, R=256, nh=16, nh_kv=4, hd=128, causal=True,
        bwd_rtol=0.15,  # relax slightly for heavy fp16 accumulation
    ))

    results.append(run_dualkv_cuda_test(
        "Large: bs=8, P=4096, R=1024, GQA 4:1",
        bs=8, P=4096, R=1024, nh=16, nh_kv=4, hd=128, causal=True,
        bwd_rtol=0.15,
    ))

    # ---- Gap 5: GQA 8:1 (Llama-3-70B style) ----

    results.append(run_dualkv_cuda_test(
        "GQA 8:1, small, causal",
        bs=4, P=64, R=16, nh=16, nh_kv=2, hd=128, causal=True,
    ))

    results.append(run_dualkv_cuda_test(
        "GQA 8:1, medium, causal",
        bs=4, P=256, R=64, nh=32, nh_kv=4, hd=128, causal=True,
    ))

    results.append(run_dualkv_cuda_test(
        "GQA 8:1, Llama-3-70B config",
        bs=4, P=512, R=128, nh=64, nh_kv=8, hd=128, causal=True,
    ))

    # ---- Gap 6: Non-uniform varlen decoded lengths ----

    results.append(run_dualkv_cuda_varlen_test(
        "Varlen: 3 seqs, R=[8,16,32]",
        P=64, nh=4, nh_kv=4, hd=128, decoded_lens=[8, 16, 32], causal=True,
    ))

    results.append(run_dualkv_cuda_varlen_test(
        "Varlen: 4 seqs, R=[1,7,33,64], GQA 4:1",
        P=128, nh=8, nh_kv=2, hd=128, decoded_lens=[1, 7, 33, 64], causal=True,
    ))

    results.append(run_dualkv_cuda_varlen_test(
        "Varlen: 4 seqs, R=[3,129,17,255], GQA 4:1, non-causal",
        P=64, nh=8, nh_kv=2, hd=128, decoded_lens=[3, 129, 17, 255], causal=False,
    ))

    results.append(run_dualkv_cuda_varlen_test(
        "Varlen: 5 seqs, R=[1,1,1,1,1], minimal decoded",
        P=64, nh=4, nh_kv=4, hd=128, decoded_lens=[1, 1, 1, 1, 1], causal=True,
    ))

    # ---- Risk-targeted variable-R configs ----

    # Risk 1: Block bounds at kBlockN=128 boundaries
    # R=1 (tiny), R=128 (exact kBlockN), R=127 (off-by-one), R=256 (2x kBlockN)
    results.append(run_dualkv_cuda_varlen_test(
        "Varlen boundary: R=[1,128,127,256], GQA 4:1",
        P=256, nh=8, nh_kv=2, hd=128, decoded_lens=[1, 128, 127, 256], causal=True,
    ))

    # Risk 2+3: Wide variance in S_i stresses Q iteration bounds + offset packing
    # Large spread: S_i = 266, 756, 259, 456 — very different m_block_max per sequence
    results.append(run_dualkv_cuda_varlen_test(
        "Varlen wide: R=[10,500,3,200], GQA 4:1",
        P=256, nh=8, nh_kv=2, hd=128, decoded_lens=[10, 500, 3, 200], causal=True,
    ))

    # Combined: boundary + wide variance + non-causal + GQA 8:1
    results.append(run_dualkv_cuda_varlen_test(
        "Varlen stress: R=[1,255,128,2,512], GQA 8:1",
        P=512, nh=16, nh_kv=2, hd=128, decoded_lens=[1, 255, 128, 2, 512], causal=True,
    ))

    # Non-causal variant (different masking path)
    results.append(run_dualkv_cuda_varlen_test(
        "Varlen wide non-causal: R=[10,500,3,200], GQA 4:1",
        P=256, nh=8, nh_kv=2, hd=128, decoded_lens=[10, 500, 3, 200], causal=False,
    ))

    # ---- DualKV simulation tests ----

    # Small causal DualKV GQA
    results.append(run_dualkv_test(
        "Small causal GQA 4:1",
        bs=4, P=64, R=16, nh=8, nh_kv=2, hd=64, causal=True, Br=32, Bc=32,
    ))

    # Small non-causal DualKV GQA
    results.append(run_dualkv_test(
        "Small non-causal GQA 4:1",
        bs=4, P=64, R=16, nh=8, nh_kv=2, hd=64, causal=False, Br=32, Bc=32,
    ))

    # Medium causal DualKV GQA
    results.append(run_dualkv_test(
        "Medium causal GQA 4:1, bs=20, P=256, R=64",
        bs=20, P=256, R=64, nh=32, nh_kv=8, hd=128, causal=True, Br=64, Bc=64,
    ))

    # Medium non-causal DualKV GQA
    results.append(run_dualkv_test(
        "Medium non-causal GQA 4:1, bs=20, P=256, R=64",
        bs=20, P=256, R=64, nh=32, nh_kv=8, hd=128, causal=False, Br=64, Bc=64,
    ))

    # Full scale causal DualKV
    results.append(run_dualkv_test(
        "Full causal GQA 4:1, bs=20, P=4096, R=1024",
        bs=20, P=4096, R=1024, nh=32, nh_kv=8, hd=128, causal=True, Br=128, Bc=128,
    ))

    # Full scale non-causal DualKV
    results.append(run_dualkv_test(
        "Full non-causal GQA 4:1, bs=20, P=4096, R=1024",
        bs=20, P=4096, R=1024, nh=32, nh_kv=8, hd=128, causal=False, Br=128, Bc=128,
    ))

    # ---- Production model configs: DualKV ----

    # Qwen2-1.5B: nh=12, nh_kv=2, hd=128, GQA 6:1
    results.append(run_dualkv_test(
        "Qwen2-1.5B config: GQA 6:1, causal",
        bs=20, P=4096, R=1024, nh=12, nh_kv=2, hd=128, causal=True, Br=128, Bc=128,
    ))

    # Llama-3-70B / Qwen2.5-72B: nh=64, nh_kv=8, hd=128, GQA 8:1
    results.append(run_dualkv_test(
        "Llama-3-70B config: GQA 8:1, causal",
        bs=4, P=4096, R=1024, nh=64, nh_kv=8, hd=128, causal=True, Br=128, Bc=128,
    ))

    print(f"\n{'='*70}")
    print(f"Summary: {sum(results)}/{len(results)} tests passed")
    print(f"{'='*70}")
