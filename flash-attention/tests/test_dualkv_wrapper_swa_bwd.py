"""Wrapper-level backward test for the SWA-through-DualKV sliding path.

Validates that gradients flowing through _make_dualkv_flash_wrapper match an eager
[ctx; resp_i] windowed reference, for BOTH the sliding (hd=256, windowed) and global
(hd=512, full-causal) layer geometries. This is the integration check the forward-only
wrapper test misses: in the new path, decoded queries on a sliding layer attend the
SHARED prompt K/V (Call2, context_seqlen=P) and accumulate prompt gradients via the
fp32-atomic dKc/dVc, instead of the old FA2 fallback that replicated the prompt N times.
"""
import torch
from verl.models.transformers.monkey_patch import _make_dualkv_flash_wrapper

torch.manual_seed(0); dev = "cuda"; dt = torch.bfloat16


def eager_packed(q, k, v, P, N, R, H, Hkv, d, scale, W):
    """Eager reference over the packed [prompt; resp_0; ...; resp_{N-1}] layout.

    Prompt rows attend prompt causally (windowed). Each response i attends
    [prompt; resp_i] causally (windowed), in logical order [0..P-1, P..P+R-1].
    Returns output in the same packed row order as the wrapper.
    """
    rep = H // Hkv
    kk = k.repeat_interleave(rep, 1).float()   # (T,H,d)
    vv = v.repeat_interleave(rep, 1).float()
    out = torch.zeros(P + N * R, H, d, device=dev, dtype=torch.float32)

    # prompt self-attention (windowed causal)
    qp = q[:P].permute(1, 0, 2).float()
    kp = kk[:P].permute(1, 2, 0)
    sp = torch.matmul(qp, kp) * scale
    rp = torch.arange(P, device=dev).view(1, P, 1)
    cp = torch.arange(P, device=dev).view(1, 1, P)
    mp = (cp > rp) | ((cp < rp - (W - 1)) if W else torch.zeros_like(cp, dtype=torch.bool))
    sp = sp.masked_fill(mp, float("-inf"))
    out[:P] = torch.matmul(torch.softmax(sp, -1), vv[:P].permute(1, 0, 2)).permute(1, 0, 2)

    # each response attends [prompt; resp_i]
    for i in range(N):
        ds = P + i * R
        qd = q[ds:ds + R].permute(1, 0, 2).float()           # (H,R,d)
        kctx = torch.cat([kk[:P], kk[ds:ds + R]], 0)         # (P+R,H,d)
        vctx = torch.cat([vv[:P], vv[ds:ds + R]], 0)
        s = torch.matmul(qd, kctx.permute(1, 2, 0)) * scale  # (H,R,P+R)
        qpos = torch.arange(R, device=dev).view(1, R, 1) + P
        kpos = torch.arange(P + R, device=dev).view(1, 1, P + R)
        m = (kpos > qpos) | ((kpos < qpos - (W - 1)) if W else torch.zeros_like(kpos, dtype=torch.bool))
        s = s.masked_fill(m, float("-inf"))
        out[ds:ds + R] = torch.matmul(torch.softmax(s, -1), vctx.permute(1, 0, 2)).permute(1, 0, 2)
    return out


def run(name, d, window):
    P, N, R, H, Hkv = 64, 3, 32, 4, 2
    scale = d ** -0.5
    T = P + N * R

    q = torch.randn(T, H, d, dtype=dt, device=dev, requires_grad=True)
    k = torch.randn(T, Hkv, d, dtype=dt, device=dev, requires_grad=True)
    v = torch.randn(T, Hkv, d, dtype=dt, device=dev, requires_grad=True)
    cu_dec = torch.arange(0, N * R + 1, R, dtype=torch.int32, device=dev)
    ctx = {"group_info": [{"prompt_len": P, "prompt_start": 0, "dec_start": P,
                           "dec_end": T, "cu_seqlens_dec": cu_dec, "max_decoded": R}]}
    wrap = _make_dualkv_flash_wrapper(lambda *a, **kw: (_ for _ in ()).throw(RuntimeError("orig")))
    kw = dict(softmax_scale=scale)
    if window:
        kw["sliding_window"] = window

    out = wrap(q.unsqueeze(0), k.unsqueeze(0), v.unsqueeze(0), None, T, dualkv_context=ctx, **kw)[0]
    g = torch.randn_like(out)
    gq, gk, gv = torch.autograd.grad(out, [q, k, v], g)

    # eager
    q2 = q.detach().clone().requires_grad_(True)
    k2 = k.detach().clone().requires_grad_(True)
    v2 = v.detach().clone().requires_grad_(True)
    ref = eager_packed(q2, k2, v2, P, N, R, H, Hkv, d, scale, window)
    gq2, gk2, gv2 = torch.autograd.grad(ref, [q2, k2, v2], g.float())

    # allclose (abs OR rel per element), matching the standalone kernel tests;
    # pure max-abs is too strict for bf16 grads on small GQA shapes.
    def cmp(a, b):
        return ((a.float() - b.float()).abs().max().item(),
                torch.allclose(a.float(), b.float(), atol=3e-2, rtol=3e-2))
    pairs = [("out", out, ref), ("dQ", gq, gq2), ("dK", gk, gk2), ("dV", gv, gv2)]
    errs, ok = {}, True
    for nm, a, b in pairs:
        e, passed = cmp(a, b)
        errs[nm] = e; ok = ok and passed
    print("%-8s d=%d win=%s: %s  [%s]" % (
        name, d, window, " ".join(f"{k}={v:.1e}" for k, v in errs.items()),
        "PASS" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    print("=== wrapper SWA backward (sliding hd256 + global hd512) vs eager packed ===")
    a = run("sliding", 256, 64)
    b = run("sliding2", 256, 48)   # window not a multiple of block size
    c = run("global", 512, None)
    print("ALL PASS" if (a and b and c) else "SOME FAILED")
