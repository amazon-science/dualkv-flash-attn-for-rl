"""DualKV forward sliding-window attention (causal SWA) correctness.

Gemma4 sliding-layer shape (hd=256, GQA 32:16, window W). Reference = eager
windowed SDPA. A decoded query at logical position p attends to logical keys in
[max(0, p - (W-1)), p] over the concatenated [context; decoded] axis.

Covers: window straddling the ctx/dec boundary, W>=P+R reduces to full causal
(regression guard vs the non-windowed kernel), W=1 (degenerate), and W not a
multiple of the kernel block size (block-edge correctness). Forward only — run
under no_grad since the windowed backward is not implemented.
"""
import torch
import torch.nn.functional as F
from flash_attn import flash_attn_dualkv_varlen_func


def eager_ref(q_dec, k_ctx, v_ctx, k_dec, v_dec, P, R, H, Hkv, d, scale, window_left):
    rep = H // Hkv
    k = torch.cat([k_ctx, k_dec], 0)              # (P+R, Hkv, d)
    v = torch.cat([v_ctx, v_dec], 0)
    k = k.repeat_interleave(rep, dim=1)           # (P+R, H, d)
    v = v.repeat_interleave(rep, dim=1)
    qf = q_dec.permute(1, 0, 2).float()           # (H,R,d)
    kf = k.permute(1, 2, 0).float()               # (H,d,P+R)
    s = torch.matmul(qf, kf) * scale              # (H,R,P+R)
    rows = torch.arange(R, device=q_dec.device).view(1, R, 1) + P    # logical query pos
    cols = torch.arange(P + R, device=q_dec.device).view(1, 1, P + R)
    # causal: col <= row ; window: col >= row - window_left
    mask = (cols > rows) | (cols < rows - window_left)
    s = s.masked_fill(mask, float("-inf"))
    a = torch.softmax(s, dim=-1)
    vf = v.permute(1, 0, 2).float()               # (H,P+R,d)
    o = torch.matmul(a, vf)                        # (H,R,d)
    return o.permute(1, 0, 2)                      # (R,H,d)


def run(P, R, W, H=32, Hkv=16, d=256, dtype=torch.bfloat16):
    torch.manual_seed(0)
    dev = "cuda"; scale = d ** -0.5
    window_left = W - 1   # FA2 convention: window_left keys back + self = W total
    q_dec = torch.randn(R, H,   d, dtype=dtype, device=dev)
    k_ctx = torch.randn(P, Hkv, d, dtype=dtype, device=dev)
    v_ctx = torch.randn(P, Hkv, d, dtype=dtype, device=dev)
    k_dec = torch.randn(R, Hkv, d, dtype=dtype, device=dev)
    v_dec = torch.randn(R, Hkv, d, dtype=dtype, device=dev)
    cu = torch.tensor([0, R], dtype=torch.int32, device=dev)
    with torch.no_grad():
        out = flash_attn_dualkv_varlen_func(
            q_dec, k_ctx, v_ctx, k_dec, v_dec, cu, cu,
            max_seqlen_q=R, context_seqlen=P, max_seqlen_k_decoded=R,
            softmax_scale=scale, causal=True, window_size_left=window_left)
    ref = eager_ref(q_dec, k_ctx, v_ctx, k_dec, v_dec, P, R, H, Hkv, d, scale, window_left)
    err = (out.float() - ref.float()).abs().max().item()
    ok = torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2)
    straddle = "straddles" if (P - window_left < P <= R + P) and window_left < P else "within"
    print("P=%-5d R=%-4d W=%-5d (%s ctx/dec)  fwd max_abs=%.2e  [%s]"
          % (P, R, W, straddle, err, "PASS" if ok else "FAIL"))
    return ok


def run_full_causal_regression(P, R, H=32, Hkv=16, d=256, dtype=torch.bfloat16):
    """W >= P+R must reproduce full causal (window_size_left=-1) exactly."""
    torch.manual_seed(1)
    dev = "cuda"; scale = d ** -0.5
    q_dec = torch.randn(R, H,   d, dtype=dtype, device=dev)
    k_ctx = torch.randn(P, Hkv, d, dtype=dtype, device=dev)
    v_ctx = torch.randn(P, Hkv, d, dtype=dtype, device=dev)
    k_dec = torch.randn(R, Hkv, d, dtype=dtype, device=dev)
    v_dec = torch.randn(R, Hkv, d, dtype=dtype, device=dev)
    cu = torch.tensor([0, R], dtype=torch.int32, device=dev)
    with torch.no_grad():
        out_full = flash_attn_dualkv_varlen_func(
            q_dec, k_ctx, v_ctx, k_dec, v_dec, cu, cu,
            max_seqlen_q=R, context_seqlen=P, max_seqlen_k_decoded=R,
            softmax_scale=scale, causal=True, window_size_left=-1)
        out_wide = flash_attn_dualkv_varlen_func(
            q_dec, k_ctx, v_ctx, k_dec, v_dec, cu, cu,
            max_seqlen_q=R, context_seqlen=P, max_seqlen_k_decoded=R,
            softmax_scale=scale, causal=True, window_size_left=P + R + 8)
    err = (out_full.float() - out_wide.float()).abs().max().item()
    ok = torch.allclose(out_full.float(), out_wide.float(), atol=1e-3, rtol=1e-3)
    print("P=%-5d R=%-4d  W>=P+R vs full-causal max_abs=%.2e  [%s]"
          % (P, R, err, "PASS" if ok else "FAIL"))
    return ok


if __name__ == "__main__":
    print("=== DualKV forward sliding-window (Gemma4 sliding-layer shape hd=256) vs eager ===")
    allok = True
    # window smaller than P -> band lies entirely in context tail + decoded
    allok = run(P=1024, R=512, W=256) and allok
    # window straddling the ctx/dec boundary (W comparable to P)
    allok = run(P=512,  R=512, W=512) and allok
    allok = run(P=1024, R=1024, W=1024) and allok    # Gemma4 actual window
    # W not a multiple of kernel block size (64) -> block-edge correctness
    allok = run(P=768,  R=384, W=300) and allok
    # degenerate window
    allok = run(P=256,  R=128, W=1) and allok
    print("--- regression: W>=P+R must equal full causal ---")
    allok = run_full_causal_regression(P=512, R=256) and allok
    allok = run_full_causal_regression(P=1024, R=512) and allok
    print("ALL PASS" if allok else "SOME FAILED")
