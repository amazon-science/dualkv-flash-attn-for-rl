"""
Smoke test for flash_attn_dualkv_varlen_func CUDA kernel (fwd + bwd).
Compares against lean DualKV tiled simulation (float32 reference).
Uses torch.allclose(atol=1e-2, rtol=1e-3) for correctness checks.
"""
import torch
import sys, os
sys.path.insert(0, os.path.dirname(__file__))

from test_varlen_fwd_bwd import lean_dualkv_fwd, lean_dualkv_bwd


def allclose_check(name, ref, test, atol=1e-2, rtol=1e-3):
    """Compare ref (float32) vs test (from fp16), using torch.allclose."""
    close = torch.allclose(ref, test, atol=atol, rtol=rtol)
    abs_err = (ref - test).abs().max().item()
    mask = ~torch.isclose(ref, test, atol=atol, rtol=rtol)
    n_fail = mask.sum().item()
    n_total = ref.numel()
    pct = 100.0 * n_fail / n_total
    status = "PASS" if close else "FAIL"
    print(f"    {name:8s}: allclose={close}  max_abs={abs_err:.2e}  fail={n_fail}/{n_total} ({pct:.2f}%)  [{status}]")
    return close


def run_dualkv_cuda_test(label, bs, P, R, nh, nh_kv, hd, causal=True,
                         atol=1e-2, rtol=1e-3):
    from flash_attn import flash_attn_dualkv_varlen_func

    S = P + R
    groups = nh // nh_kv
    device = 'cuda'

    print(f"\n{'='*70}")
    print(f"DualKV CUDA: {label}")
    print(f"  bs={bs}, P={P}, R={R}, S={S}, nh={nh}, nh_kv={nh_kv}, hd={hd}")
    print(f"  GQA ratio={groups}:1, causal={causal}, atol={atol}, rtol={rtol}")
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
    Q_packed = Q.permute(0, 2, 1, 3).contiguous().reshape(bs * S, nh, hd).half().requires_grad_(True)
    Kc_packed = K_ctx[0].permute(1, 0, 2).contiguous().half().requires_grad_(True)
    Vc_packed = V_ctx[0].permute(1, 0, 2).contiguous().half().requires_grad_(True)
    Kd_packed = K_dec.permute(0, 2, 1, 3).contiguous().reshape(bs * R, nh_kv, hd).half().requires_grad_(True)
    Vd_packed = V_dec.permute(0, 2, 1, 3).contiguous().reshape(bs * R, nh_kv, hd).half().requires_grad_(True)

    cu_seqlens_q = torch.arange(0, (bs + 1) * S, S, device=device, dtype=torch.int32)
    cu_seqlens_k_decoded = torch.arange(0, (bs + 1) * R, R, device=device, dtype=torch.int32)
    dO_packed = dO.permute(0, 2, 1, 3).contiguous().reshape(bs * S, nh, hd).half()

    torch.cuda.synchronize()
    O_cuda = flash_attn_dualkv_varlen_func(
        Q_packed, Kc_packed, Vc_packed, Kd_packed, Vd_packed,
        cu_seqlens_q, cu_seqlens_k_decoded,
        max_seqlen_q=S, context_seqlen=P, max_seqlen_k_decoded=R, causal=causal,
    )
    torch.cuda.synchronize()

    O_cuda.backward(dO_packed)
    torch.cuda.synchronize()

    # ---- Convert CUDA outputs for comparison ----
    O_cuda_cmp = O_cuda.detach().float().reshape(bs, S, nh, hd).permute(0, 2, 1, 3)
    dQ_cuda_cmp = Q_packed.grad.float().reshape(bs, S, nh, hd).permute(0, 2, 1, 3)
    dKc_cuda_cmp = Kc_packed.grad.float().permute(1, 0, 2).unsqueeze(0)
    dVc_cuda_cmp = Vc_packed.grad.float().permute(1, 0, 2).unsqueeze(0)
    dKd_cuda_cmp = Kd_packed.grad.float().reshape(bs, R, nh_kv, hd).permute(0, 2, 1, 3)
    dVd_cuda_cmp = Vd_packed.grad.float().reshape(bs, R, nh_kv, hd).permute(0, 2, 1, 3)

    # ---- Compare ----
    all_pass = True

    print("\n  Forward:")
    all_pass &= allclose_check("O", O_ref, O_cuda_cmp, atol=atol, rtol=rtol)

    print("\n  Backward:")
    all_pass &= allclose_check("dQ", dQ_ref, dQ_cuda_cmp, atol=atol, rtol=rtol)
    all_pass &= allclose_check("dK_ctx", dK_ctx_ref, dKc_cuda_cmp, atol=atol, rtol=rtol)
    all_pass &= allclose_check("dV_ctx", dV_ctx_ref, dVc_cuda_cmp, atol=atol, rtol=rtol)
    all_pass &= allclose_check("dK_dec", dK_dec_ref, dKd_cuda_cmp, atol=atol, rtol=rtol)
    all_pass &= allclose_check("dV_dec", dV_dec_ref, dVd_cuda_cmp, atol=atol, rtol=rtol)

    print(f"\n  {'ALL PASS' if all_pass else 'FAIL'}")
    return all_pass


if __name__ == "__main__":
    results = []

    results.append(run_dualkv_cuda_test(
        "Small causal MHA",
        bs=2, P=64, R=16, nh=4, nh_kv=4, hd=128, causal=True,
    ))

    results.append(run_dualkv_cuda_test(
        "Small causal GQA 4:1",
        bs=4, P=64, R=16, nh=8, nh_kv=2, hd=128, causal=True,
    ))

    results.append(run_dualkv_cuda_test(
        "Small non-causal GQA 4:1",
        bs=4, P=64, R=16, nh=8, nh_kv=2, hd=128, causal=False,
    ))

    results.append(run_dualkv_cuda_test(
        "Medium causal GQA 4:1, bs=8, P=256, R=64",
        bs=8, P=256, R=64, nh=16, nh_kv=4, hd=128, causal=True,
    ))

    results.append(run_dualkv_cuda_test(
        "Large causal GQA 6:1 (Qwen-like)",
        bs=8, P=512, R=128, nh=12, nh_kv=2, hd=128, causal=True,
    ))

    print(f"\n{'='*70}")
    print(f"Summary: {sum(results)}/{len(results)} tests passed")
    print(f"{'='*70}")
