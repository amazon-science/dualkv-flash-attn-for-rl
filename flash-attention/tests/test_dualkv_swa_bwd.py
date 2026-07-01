"""DualKV backward sliding-window attention (causal SWA) gradient correctness.

Validates all 5 gradients (dQ, dKc, dVc, dKr, dVr) against an eager windowed
reference on the Gemma4 sliding-layer shape (hd=256, GQA 32:16, window W).

Emphasis on the shared-context gradients dKc/dVc: under a window, a context key
must receive gradient ONLY from in-window query rows. The kernel zeroes
out-of-band scores (-> P=0 -> dS=0), so out-of-window rows contribute exactly
zero to the fp32-atomic dKc/dVc accumulation. Covers window straddling the
ctx/dec boundary, non-block-multiple windows, V=K (attention_k_eq_v), and
W>=P+R reducing to full causal (regression vs the non-windowed bwd).
"""
import torch
import torch.nn.functional as F
from flash_attn import flash_attn_dualkv_varlen_func


def eager_windowed(q_dec, k_ctx, v_ctx, k_dec, v_dec, P, R, H, Hkv, d, scale, window_left):
    rep = H // Hkv
    k = torch.cat([k_ctx, k_dec], 0).repeat_interleave(rep, dim=1)  # (P+R,H,d)
    v = torch.cat([v_ctx, v_dec], 0).repeat_interleave(rep, dim=1)
    qf = q_dec.permute(1, 0, 2).float()           # (H,R,d)
    kf = k.permute(1, 2, 0).float()               # (H,d,P+R)
    s = torch.matmul(qf, kf) * scale
    rows = torch.arange(R, device=q_dec.device).view(1, R, 1) + P
    cols = torch.arange(P + R, device=q_dec.device).view(1, 1, P + R)
    mask = (cols > rows) | (cols < rows - window_left)
    s = s.masked_fill(mask, float("-inf"))
    a = torch.softmax(s, dim=-1)
    vf = v.permute(1, 0, 2).float()               # (H,P+R,d)
    o = torch.matmul(a, vf)                        # (H,R,d)
    return o.permute(1, 0, 2)                      # (R,H,d)


def run(P, R, W, H=32, Hkv=16, d=256, dtype=torch.bfloat16, k_eq_v=False, tag=""):
    torch.manual_seed(0)
    dev = "cuda"; scale = d ** -0.5
    window_left = W - 1

    def mk(*shape):
        return torch.randn(*shape, dtype=dtype, device=dev, requires_grad=True)

    q_dec = mk(R, H, d)
    k_ctx = mk(P, Hkv, d)
    k_dec = mk(R, Hkv, d)
    if k_eq_v:
        v_ctx, v_dec = k_ctx, k_dec      # V aliases K (attention_k_eq_v)
    else:
        v_ctx = mk(P, Hkv, d)
        v_dec = mk(R, Hkv, d)
    cu = torch.tensor([0, R], dtype=torch.int32, device=dev)

    # --- kernel ---
    out = flash_attn_dualkv_varlen_func(
        q_dec, k_ctx, v_ctx, k_dec, v_dec, cu, cu,
        max_seqlen_q=R, context_seqlen=P, max_seqlen_k_decoded=R,
        softmax_scale=scale, causal=True, window_size_left=window_left)
    g = torch.randn_like(out)
    leaves = [q_dec, k_ctx, k_dec] + ([] if k_eq_v else [v_ctx, v_dec])
    grads_k = torch.autograd.grad(out, leaves, g, retain_graph=False)

    # --- eager ref ---
    q2 = q_dec.detach().clone().requires_grad_(True)
    kc2 = k_ctx.detach().clone().requires_grad_(True)
    kd2 = k_dec.detach().clone().requires_grad_(True)
    if k_eq_v:
        vc2, vd2 = kc2, kd2; leaves2 = [q2, kc2, kd2]
    else:
        vc2 = v_ctx.detach().clone().requires_grad_(True)
        vd2 = v_dec.detach().clone().requires_grad_(True)
        leaves2 = [q2, kc2, kd2, vc2, vd2]
    ref = eager_windowed(q2, kc2, vc2, kd2, vd2, P, R, H, Hkv, d, scale, window_left)
    grads_r = torch.autograd.grad(ref, leaves2, g.float(), retain_graph=False)

    names = ["dQ", "dKc", "dKd"] + ([] if k_eq_v else ["dVc", "dVd"])
    allok = True
    errs = []
    for n, gk, gr in zip(names, grads_k, grads_r):
        e = (gk.float() - gr.float()).abs().max().item()
        ok = torch.allclose(gk.float(), gr.float(), atol=3e-2, rtol=3e-2)
        errs.append(f"{n}={e:.1e}{'' if ok else '!'}")
        allok = allok and ok
    print("P=%-5d R=%-4d W=%-5d %-8s %s  [%s]"
          % (P, R, W, tag, " ".join(errs), "PASS" if allok else "FAIL"))
    return allok


def run_regression(P, R, H=32, Hkv=16, d=256, dtype=torch.bfloat16):
    """W>=P+R must reproduce full-causal gradients (window_size_left=-1)."""
    torch.manual_seed(2)
    dev = "cuda"; scale = d ** -0.5

    def grads_for(wl):
        torch.manual_seed(2)
        q = torch.randn(R, H, d, dtype=dtype, device=dev, requires_grad=True)
        kc = torch.randn(P, Hkv, d, dtype=dtype, device=dev, requires_grad=True)
        vc = torch.randn(P, Hkv, d, dtype=dtype, device=dev, requires_grad=True)
        kd = torch.randn(R, Hkv, d, dtype=dtype, device=dev, requires_grad=True)
        vd = torch.randn(R, Hkv, d, dtype=dtype, device=dev, requires_grad=True)
        cu = torch.tensor([0, R], dtype=torch.int32, device=dev)
        torch.manual_seed(99)
        out = flash_attn_dualkv_varlen_func(q, kc, vc, kd, vd, cu, cu,
            max_seqlen_q=R, context_seqlen=P, max_seqlen_k_decoded=R,
            softmax_scale=scale, causal=True, window_size_left=wl)
        g = torch.randn_like(out)
        return torch.autograd.grad(out, [q, kc, vc, kd, vd], g)

    gf = grads_for(-1)
    gw = grads_for(P + R + 16)
    names = ["dQ", "dKc", "dVc", "dKd", "dVd"]
    # dKc/dVc/dKd/dVd are deterministic -> expect bit-exact (0). dQ is written via
    # atomicAdd to an fp32 accumulator; the windowed path visits Q-blocks in a
    # different m_block range, so the atomic summation ORDER differs -> small bf16
    # rounding noise is expected and not a correctness issue.
    tol = {"dQ": 2e-3, "dKc": 1e-6, "dVc": 1e-6, "dKd": 1e-6, "dVd": 1e-6}
    allok = True; errs = []
    for n, a, b in zip(names, gf, gw):
        e = (a.float() - b.float()).abs().max().item()
        ok = e <= tol[n]
        errs.append(f"{n}={e:.0e}{'' if ok else '!'}"); allok = allok and ok
    print("P=%-5d R=%-4d  W>=P+R vs full-causal: %s  [%s]"
          % (P, R, " ".join(errs), "PASS" if allok else "FAIL"))
    return allok


if __name__ == "__main__":
    print("=== DualKV backward sliding-window (Gemma4 sliding shape hd=256) vs eager ===")
    allok = True
    allok = run(P=1024, R=512, W=256, tag="bf16")           and allok
    allok = run(P=512,  R=512, W=512, tag="straddle")        and allok
    allok = run(P=1024, R=1024, W=1024, tag="gemma")         and allok
    allok = run(P=768,  R=384, W=300, tag="nonblk")          and allok
    allok = run(P=512,  R=256, W=256, dtype=torch.float16, tag="fp16")   and allok
    allok = run(P=512,  R=256, W=256, k_eq_v=True, tag="V=K")            and allok
    print("--- regression: W>=P+R must equal full-causal grads ---")
    allok = run_regression(P=512, R=256) and allok
    allok = run_regression(P=1024, R=512) and allok
    print("ALL PASS" if allok else "SOME FAILED")
