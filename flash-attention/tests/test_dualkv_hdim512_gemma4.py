"""DualKV head_dim=512 correctness, Gemma4-31B global-layer shape (hd=512, GQA 32:4).
Reference = manual eager SDPA (FA2 can't do hd=512), decoded queries attend to [ctx;dec] causally.
"""
import torch
import torch.nn.functional as F
from flash_attn import flash_attn_dualkv_varlen_func

def eager_ref(q_dec, k_ctx, v_ctx, k_dec, v_dec, P, R, H, Hkv, d, scale):
    # expand GQA: each kv head serves H//Hkv q heads
    rep = H // Hkv
    k = torch.cat([k_ctx, k_dec], 0)              # (P+R, Hkv, d)
    v = torch.cat([v_ctx, v_dec], 0)
    k = k.repeat_interleave(rep, dim=1)           # (P+R, H, d)
    v = v.repeat_interleave(rep, dim=1)
    q = q_dec                                     # (R, H, d)
    # scores: for decoded query r (logical P+r), attend keys 0..P+r
    qf = q.permute(1,0,2).float()                 # (H,R,d)
    kf = k.permute(1,2,0).float()                 # (H,d,P+R)
    s = torch.matmul(qf, kf) * scale              # (H,R,P+R)
    # causal mask: query r (row) sees cols <= P+r
    rows = torch.arange(R, device=q.device).view(1,R,1) + P     # logical pos of query
    cols = torch.arange(P+R, device=q.device).view(1,1,P+R)
    s = s.masked_fill(cols > rows, float("-inf"))
    a = torch.softmax(s, dim=-1)                  # (H,R,P+R)
    vf = v.permute(1,0,2).float()                 # (H,P+R,d)
    o = torch.matmul(a, vf)                       # (H,R,d)
    return o.permute(1,0,2)                        # (R,H,d)

def run(P, R, H=32, Hkv=4, d=512, dtype=torch.bfloat16):
    torch.manual_seed(0)
    dev="cuda"; scale=d**-0.5
    q_dec=torch.randn(R, H,   d, dtype=dtype, device=dev)
    k_ctx=torch.randn(P, Hkv, d, dtype=dtype, device=dev)
    v_ctx=torch.randn(P, Hkv, d, dtype=dtype, device=dev)
    k_dec=torch.randn(R, Hkv, d, dtype=dtype, device=dev)
    v_dec=torch.randn(R, Hkv, d, dtype=dtype, device=dev)
    cu=torch.tensor([0, R], dtype=torch.int32, device=dev)
    out=flash_attn_dualkv_varlen_func(q_dec,k_ctx,v_ctx,k_dec,v_dec,cu,cu,
        max_seqlen_q=R, context_seqlen=P, max_seqlen_k_decoded=R, softmax_scale=scale, causal=True)
    ref=eager_ref(q_dec,k_ctx,v_ctx,k_dec,v_dec,P,R,H,Hkv,d,scale)
    err=(out.float()-ref.float()).abs().max().item()
    ok=torch.allclose(out.float(), ref.float(), atol=2e-2, rtol=2e-2)
    print("P=%d R=%d d=%d GQA %d:%d  fwd max_abs=%.2e  [%s]" % (P,R,d,H,Hkv,err,"PASS" if ok else "FAIL"))
    return ok

if __name__=="__main__":
    print("=== DualKV hdim=512 (Gemma4 global-layer shape) vs eager SDPA ===")
    allok=True
    for P,R in [(256,128),(512,256),(1024,512)]:
        allok=run(P,R) and allok
    print("ALL PASS" if allok else "SOME FAILED")
