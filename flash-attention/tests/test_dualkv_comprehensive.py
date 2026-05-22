"""
Comprehensive correctness test for flash_attn_dualkv_varlen_func.
Tests forward-only, backward-only, and fwd+bwd across many parameter combos:
  - P (context length): block-boundary edge cases
  - R (decoded length): various sizes
  - bs (batch size): 1 to 16
  - hdim: 64, 96, 128, 192, 256
  - GQA ratios: MHA, 2:1, 4:1, 6:1, 8:1
  - causal vs non-causal
  - variable-length decoded sequences (different R per batch element)
  - dtype: fp16 (only supported dtype for DualKV kernels)

Reference: lean tiled DualKV simulation in float32.
"""
import sys, os, time
sys.path.insert(0, os.path.dirname(__file__))

import torch
from test_varlen_fwd_bwd import lean_dualkv_fwd, lean_dualkv_bwd
from flash_attn import flash_attn_dualkv_varlen_func


def allclose_check(name, ref, test, atol=1e-2, rtol=1e-3):
    close = torch.allclose(ref, test, atol=atol, rtol=rtol)
    abs_err = (ref - test).abs().max().item()
    mask = ~torch.isclose(ref, test, atol=atol, rtol=rtol)
    n_fail = mask.sum().item()
    n_total = ref.numel()
    pct = 100.0 * n_fail / n_total
    return close, abs_err, n_fail, n_total, pct


def run_one(bs, P, R, nh, nh_kv, hd, causal, mode="fwd_bwd", atol=1e-2, rtol=1e-3):
    """
    mode: "fwd_only", "bwd_only", or "fwd_bwd"
    Returns dict of {name: (ok, abs_err, n_fail, n_total, pct)}
    """
    S = P + R
    device = 'cuda'

    torch.manual_seed(42)

    K_ctx = torch.randn(1, nh_kv, P, hd, device=device, dtype=torch.float32)
    V_ctx = torch.randn(1, nh_kv, P, hd, device=device, dtype=torch.float32)
    K_dec = torch.randn(bs, nh_kv, R, hd, device=device, dtype=torch.float32)
    V_dec = torch.randn(bs, nh_kv, R, hd, device=device, dtype=torch.float32)
    Q = torch.randn(bs, nh, S, hd, device=device, dtype=torch.float32)
    dO = torch.randn(bs, nh, S, hd, device=device, dtype=torch.float32)

    # Reference (always compute both for bwd_only which needs O_ref, L_ref)
    O_ref, L_ref = lean_dualkv_fwd(Q, K_ctx, K_dec, V_ctx, V_dec, causal=causal)

    need_bwd = mode in ("bwd_only", "fwd_bwd")
    if need_bwd:
        dQ_ref, dK_ctx_ref, dK_dec_ref, dV_ctx_ref, dV_dec_ref = lean_dualkv_bwd(
            dO, Q, K_ctx, K_dec, V_ctx, V_dec, O_ref, L_ref, causal=causal)

    # CUDA kernel (fp16)
    need_grad = need_bwd
    Q_packed = Q.permute(0, 2, 1, 3).contiguous().reshape(bs * S, nh, hd).half().requires_grad_(need_grad)
    Kc_packed = K_ctx[0].permute(1, 0, 2).contiguous().half().requires_grad_(need_grad)
    Vc_packed = V_ctx[0].permute(1, 0, 2).contiguous().half().requires_grad_(need_grad)
    Kd_packed = K_dec.permute(0, 2, 1, 3).contiguous().reshape(bs * R, nh_kv, hd).half().requires_grad_(need_grad)
    Vd_packed = V_dec.permute(0, 2, 1, 3).contiguous().reshape(bs * R, nh_kv, hd).half().requires_grad_(need_grad)

    cu_seqlens_q = torch.arange(0, (bs + 1) * S, S, device=device, dtype=torch.int32)
    cu_seqlens_k_decoded = torch.arange(0, (bs + 1) * R, R, device=device, dtype=torch.int32)

    torch.cuda.synchronize()
    O_cuda = flash_attn_dualkv_varlen_func(
        Q_packed, Kc_packed, Vc_packed, Kd_packed, Vd_packed,
        cu_seqlens_q, cu_seqlens_k_decoded,
        max_seqlen_q=S, context_seqlen=P, max_seqlen_k_decoded=R, causal=causal,
    )
    torch.cuda.synchronize()

    if need_bwd:
        dO_packed = dO.permute(0, 2, 1, 3).contiguous().reshape(bs * S, nh, hd).half()
        O_cuda.backward(dO_packed)
        torch.cuda.synchronize()

    results = {}

    # Forward check
    if mode in ("fwd_only", "fwd_bwd"):
        O_cmp = O_cuda.detach().float().reshape(bs, S, nh, hd).permute(0, 2, 1, 3)
        ok, abs_err, n_fail, n_total, pct = allclose_check("O", O_ref, O_cmp, atol=atol, rtol=rtol)
        results["O"] = (ok, abs_err, n_fail, n_total, pct)

    # Backward checks
    if need_bwd:
        dQ_cmp = Q_packed.grad.float().reshape(bs, S, nh, hd).permute(0, 2, 1, 3)
        dKc_cmp = Kc_packed.grad.float().permute(1, 0, 2).unsqueeze(0)
        dVc_cmp = Vc_packed.grad.float().permute(1, 0, 2).unsqueeze(0)
        dKd_cmp = Kd_packed.grad.float().reshape(bs, R, nh_kv, hd).permute(0, 2, 1, 3)
        dVd_cmp = Vd_packed.grad.float().reshape(bs, R, nh_kv, hd).permute(0, 2, 1, 3)

        for name, ref, test in [
            ("dQ", dQ_ref, dQ_cmp),
            ("dK_ctx", dK_ctx_ref, dKc_cmp),
            ("dV_ctx", dV_ctx_ref, dVc_cmp),
            ("dK_dec", dK_dec_ref, dKd_cmp),
            ("dV_dec", dV_dec_ref, dVd_cmp),
        ]:
            ok, abs_err, n_fail, n_total, pct = allclose_check(name, ref, test, atol=atol, rtol=rtol)
            results[name] = (ok, abs_err, n_fail, n_total, pct)

    return results


def run_varlen(bs, P, R_per_seq, nh, nh_kv, hd, causal, atol=1e-2, rtol=1e-3):
    """
    Variable-length decoded sequences: each batch element has different R.
    R_per_seq: list of length bs with decoded lengths.
    """
    max_R = max(R_per_seq)
    device = 'cuda'

    torch.manual_seed(42)

    K_ctx_data = torch.randn(1, nh_kv, P, hd, device=device, dtype=torch.float32)
    V_ctx_data = torch.randn(1, nh_kv, P, hd, device=device, dtype=torch.float32)

    # Build packed Q, K_dec, V_dec and reference per-sequence
    all_Q_fp32 = []
    all_Kd_fp32 = []
    all_Vd_fp32 = []
    all_dO_fp32 = []
    O_refs = []
    dQ_refs = []
    dKd_refs = []
    dVd_refs = []
    dKc_ref_accum = torch.zeros(1, nh_kv, P, hd, device=device, dtype=torch.float32)
    dVc_ref_accum = torch.zeros(1, nh_kv, P, hd, device=device, dtype=torch.float32)

    for b in range(bs):
        R_b = R_per_seq[b]
        S_b = P + R_b
        Q_b = torch.randn(1, nh, S_b, hd, device=device, dtype=torch.float32)
        Kd_b = torch.randn(1, nh_kv, R_b, hd, device=device, dtype=torch.float32)
        Vd_b = torch.randn(1, nh_kv, R_b, hd, device=device, dtype=torch.float32)
        dO_b = torch.randn(1, nh, S_b, hd, device=device, dtype=torch.float32)

        all_Q_fp32.append(Q_b)
        all_Kd_fp32.append(Kd_b)
        all_Vd_fp32.append(Vd_b)
        all_dO_fp32.append(dO_b)

        O_ref_b, L_ref_b = lean_dualkv_fwd(Q_b, K_ctx_data, Kd_b, V_ctx_data, Vd_b, causal=causal)
        dQ_ref_b, dKc_ref_b, dKd_ref_b, dVc_ref_b, dVd_ref_b = lean_dualkv_bwd(
            dO_b, Q_b, K_ctx_data, Kd_b, V_ctx_data, Vd_b, O_ref_b, L_ref_b, causal=causal)

        O_refs.append(O_ref_b.squeeze(0))
        dQ_refs.append(dQ_ref_b.squeeze(0))
        dKd_refs.append(dKd_ref_b.squeeze(0))
        dVd_refs.append(dVd_ref_b.squeeze(0))
        dKc_ref_accum += dKc_ref_b
        dVc_ref_accum += dVc_ref_b

    # Pack for CUDA kernel
    Q_packed_list = []
    Kd_packed_list = []
    Vd_packed_list = []
    dO_packed_list = []
    cu_q = [0]
    cu_kd = [0]

    for b in range(bs):
        R_b = R_per_seq[b]
        S_b = P + R_b
        Q_packed_list.append(all_Q_fp32[b].squeeze(0).permute(1, 0, 2).contiguous().reshape(S_b, nh, hd))
        Kd_packed_list.append(all_Kd_fp32[b].squeeze(0).permute(1, 0, 2).contiguous().reshape(R_b, nh_kv, hd))
        Vd_packed_list.append(all_Vd_fp32[b].squeeze(0).permute(1, 0, 2).contiguous().reshape(R_b, nh_kv, hd))
        dO_packed_list.append(all_dO_fp32[b].squeeze(0).permute(1, 0, 2).contiguous().reshape(S_b, nh, hd))
        cu_q.append(cu_q[-1] + S_b)
        cu_kd.append(cu_kd[-1] + R_b)

    Q_packed = torch.cat(Q_packed_list, dim=0).half().requires_grad_(True)
    Kd_packed = torch.cat(Kd_packed_list, dim=0).half().requires_grad_(True)
    Vd_packed = torch.cat(Vd_packed_list, dim=0).half().requires_grad_(True)
    Kc_packed = K_ctx_data[0].permute(1, 0, 2).contiguous().half().requires_grad_(True)
    Vc_packed = V_ctx_data[0].permute(1, 0, 2).contiguous().half().requires_grad_(True)
    dO_packed = torch.cat(dO_packed_list, dim=0).half()

    cu_seqlens_q = torch.tensor(cu_q, device=device, dtype=torch.int32)
    cu_seqlens_k_decoded = torch.tensor(cu_kd, device=device, dtype=torch.int32)
    max_S = max(P + r for r in R_per_seq)

    torch.cuda.synchronize()
    O_cuda = flash_attn_dualkv_varlen_func(
        Q_packed, Kc_packed, Vc_packed, Kd_packed, Vd_packed,
        cu_seqlens_q, cu_seqlens_k_decoded,
        max_seqlen_q=max_S, context_seqlen=P, max_seqlen_k_decoded=max_R, causal=causal,
    )
    torch.cuda.synchronize()
    O_cuda.backward(dO_packed)
    torch.cuda.synchronize()

    # Compare per-sequence
    results = {}
    all_O_ok = True
    all_dQ_ok = True
    max_O_err = 0.0
    max_dQ_err = 0.0

    offset_q = 0
    offset_kd = 0
    for b in range(bs):
        R_b = R_per_seq[b]
        S_b = P + R_b
        O_b = O_cuda[offset_q:offset_q+S_b].detach().float().permute(1, 0, 2).unsqueeze(0)
        dQ_b = Q_packed.grad[offset_q:offset_q+S_b].float().permute(1, 0, 2).unsqueeze(0)

        O_ref_b = O_refs[b].unsqueeze(0)
        dQ_ref_b = dQ_refs[b].unsqueeze(0)

        o_ok, o_err, _, _, _ = allclose_check(f"O[{b}]", O_ref_b, O_b, atol=atol, rtol=rtol)
        dq_ok, dq_err, _, _, _ = allclose_check(f"dQ[{b}]", dQ_ref_b, dQ_b, atol=atol, rtol=rtol)
        all_O_ok &= o_ok
        all_dQ_ok &= dq_ok
        max_O_err = max(max_O_err, o_err)
        max_dQ_err = max(max_dQ_err, dq_err)

        offset_q += S_b
        offset_kd += R_b

    results["O"] = (all_O_ok, max_O_err, 0, 0, 0.0)
    results["dQ"] = (all_dQ_ok, max_dQ_err, 0, 0, 0.0)

    # Context grads (accumulated)
    dKc_cmp = Kc_packed.grad.float().permute(1, 0, 2).unsqueeze(0)
    dVc_cmp = Vc_packed.grad.float().permute(1, 0, 2).unsqueeze(0)
    ok_kc, err_kc, _, _, _ = allclose_check("dK_ctx", dKc_ref_accum, dKc_cmp, atol=atol, rtol=rtol)
    ok_vc, err_vc, _, _, _ = allclose_check("dV_ctx", dVc_ref_accum, dVc_cmp, atol=atol, rtol=rtol)
    results["dK_ctx"] = (ok_kc, err_kc, 0, 0, 0.0)
    results["dV_ctx"] = (ok_vc, err_vc, 0, 0, 0.0)

    return results


def run_config(i, total, bs, P, R, nh, nh_kv, hd, causal, mode):
    S = P + R
    gqa = nh // nh_kv
    c_str = 'causal' if causal else 'nocaus'
    label = f"bs={bs:2d} P={P:4d} R={R:3d} S={S:4d} hd={hd:3d} nh={nh:2d}/{nh_kv:2d} gqa={gqa}:1 {c_str:6s} {mode}"

    try:
        results = run_one(bs, P, R, nh, nh_kv, hd, causal, mode=mode)
        all_ok = all(v[0] for v in results.values())

        if all_ok:
            max_err = max(v[1] for v in results.values())
            print(f"  [{i:3d}/{total}] PASS  {label}  max_abs={max_err:.2e}")
            return True, label, None
        else:
            fails = {k: v for k, v in results.items() if not v[0]}
            fail_str = "  ".join(f"{k}: abs={v[1]:.2e} fail={v[2]}/{v[3]}" for k, v in fails.items())
            print(f"  [{i:3d}/{total}] FAIL  {label}  {fail_str}")
            return False, label, fails

    except Exception as e:
        print(f"  [{i:3d}/{total}] ERROR {label}  {e}")
        return False, label, str(e)


if __name__ == "__main__":
    configs = []

    # ================================================================
    # Phase 1: P x R sweep — block boundary edge cases
    # ================================================================
    P_values = [1, 15, 16, 31, 32, 63, 64, 127, 128, 129, 255, 256, 257, 384, 512, 1024]
    R_values = [1, 7, 8, 15, 16, 31, 32, 64, 128, 256]
    for P in P_values:
        for R in R_values:
            if P + R > 2048:
                continue
            configs.append((2, P, R, 8, 2, 128, True, "fwd_bwd"))

    # ================================================================
    # Phase 2: batch size sweep
    # ================================================================
    for bs in [1, 2, 3, 4, 7, 8, 16]:
        configs.append((bs, 256, 64, 8, 2, 128, True, "fwd_bwd"))

    # ================================================================
    # Phase 3: head dimension sweep
    # ================================================================
    for hd in [64, 96, 128, 192, 256]:
        for mode in ["fwd_only", "bwd_only", "fwd_bwd"]:
            configs.append((2, 128, 32, 8, 2, hd, True, mode))

    # ================================================================
    # Phase 4: GQA ratio sweep
    # ================================================================
    for nh, nh_kv in [(4, 4), (4, 2), (8, 2), (8, 4), (12, 2), (12, 4), (16, 4), (32, 4), (32, 8)]:
        configs.append((4, 256, 64, nh, nh_kv, 128, True, "fwd_bwd"))

    # ================================================================
    # Phase 5: causal vs non-causal, all modes
    # ================================================================
    for causal in [True, False]:
        for mode in ["fwd_only", "bwd_only", "fwd_bwd"]:
            configs.append((2, 256, 64, 8, 2, 128, causal, mode))
            configs.append((4, 128, 128, 8, 2, 128, causal, mode))

    # ================================================================
    # Phase 6: large configs
    # ================================================================
    configs.append((8, 512, 128, 16, 4, 128, True, "fwd_bwd"))
    configs.append((4, 1024, 256, 12, 2, 128, True, "fwd_bwd"))
    configs.append((2, 2048, 512, 8, 2, 128, True, "fwd_bwd"))
    configs.append((16, 128, 32, 8, 2, 128, True, "fwd_bwd"))
    configs.append((8, 512, 128, 32, 8, 128, True, "fwd_bwd"))

    # ================================================================
    # Phase 7: edge cases
    # ================================================================
    for mode in ["fwd_only", "fwd_bwd"]:
        configs.append((1, 1, 1, 4, 4, 64, True, mode))
        configs.append((1, 1, 1, 4, 4, 128, True, mode))
        configs.append((1, 128, 1, 8, 2, 128, True, mode))
        configs.append((1, 1, 128, 8, 2, 128, True, mode))
        configs.append((1, 127, 129, 8, 2, 128, True, mode))
        configs.append((1, 129, 127, 8, 2, 128, True, mode))
        configs.append((1, 1, 1, 4, 4, 64, False, mode))
        configs.append((1, 1, 1, 4, 4, 128, False, mode))
        configs.append((1, 255, 1, 8, 2, 128, True, mode))
        configs.append((1, 1, 255, 8, 2, 128, True, mode))

    # ================================================================
    # Phase 8: cross-product hdim x causal x GQA (small sizes for speed)
    # ================================================================
    for hd in [64, 96, 128, 192, 256]:
        for causal in [True, False]:
            for nh, nh_kv in [(4, 4), (8, 2), (16, 4)]:
                configs.append((2, 64, 32, nh, nh_kv, hd, causal, "fwd_bwd"))

    # Deduplicate
    configs = list(dict.fromkeys(configs))

    print("=" * 90)
    print("DualKV Comprehensive Test Suite")
    print(f"  dtype: fp16 (only supported dtype for DualKV training kernels)")
    print(f"  Total uniform-length configs: {len(configs)}")
    print("=" * 90)

    passed = 0
    failed = 0
    failures = []
    t_start = time.time()

    for i, (bs, P, R, nh, nh_kv, hd, causal, mode) in enumerate(configs, 1):
        ok, label, detail = run_config(i, len(configs), bs, P, R, nh, nh_kv, hd, causal, mode)
        if ok:
            passed += 1
        else:
            failed += 1
            failures.append((label, detail))

    # ================================================================
    # Phase 9: Variable-length decoded sequences
    # ================================================================
    print(f"\n{'=' * 90}")
    print("Phase 9: Variable-length decoded sequences")
    print("=" * 90)

    varlen_configs = [
        (4, 128, [16, 32, 64, 128], 8, 2, 128, True),
        (4, 256, [1, 7, 31, 64], 8, 2, 128, True),
        (3, 64, [10, 50, 100], 4, 4, 128, True),
        (4, 512, [32, 64, 128, 256], 16, 4, 128, True),
        (4, 128, [16, 32, 64, 128], 8, 2, 128, False),
        (2, 256, [1, 255], 8, 2, 128, True),
        (4, 128, [16, 32, 64, 128], 8, 2, 64, True),
        (4, 128, [16, 32, 64, 128], 8, 2, 256, True),
    ]

    for j, (bs, P, R_per_seq, nh, nh_kv, hd, causal) in enumerate(varlen_configs, 1):
        gqa = nh // nh_kv
        c_str = 'causal' if causal else 'nocaus'
        R_str = ",".join(str(r) for r in R_per_seq)
        label = f"bs={bs} P={P} R=[{R_str}] hd={hd} nh={nh}/{nh_kv} gqa={gqa}:1 {c_str} varlen"
        try:
            results = run_varlen(bs, P, R_per_seq, nh, nh_kv, hd, causal)
            all_ok = all(v[0] for v in results.values())
            if all_ok:
                max_err = max(v[1] for v in results.values())
                print(f"  [varlen {j}/{len(varlen_configs)}] PASS  {label}  max_abs={max_err:.2e}")
                passed += 1
            else:
                fails = {k: v for k, v in results.items() if not v[0]}
                fail_str = "  ".join(f"{k}: abs={v[1]:.2e}" for k, v in fails.items())
                print(f"  [varlen {j}/{len(varlen_configs)}] FAIL  {label}  {fail_str}")
                failed += 1
                failures.append((label, fails))
        except Exception as e:
            print(f"  [varlen {j}/{len(varlen_configs)}] ERROR {label}  {e}")
            failed += 1
            failures.append((label, str(e)))

    elapsed = time.time() - t_start
    total = passed + failed
    print(f"\n{'=' * 90}")
    print(f"RESULTS: {passed} passed, {failed} failed out of {total} total ({elapsed:.1f}s)")
    print(f"{'=' * 90}")

    if failures:
        print(f"\n{len(failures)} FAILED configs:")
        for label, detail in failures:
            print(f"  {label}")
            if isinstance(detail, dict):
                for k, v in detail.items():
                    print(f"    {k}: abs={v[1]:.2e} fail={v[2]}/{v[3]} ({v[4]:.2f}%)")
            else:
                print(f"    {detail}")

    sys.exit(1 if failed > 0 else 0)
