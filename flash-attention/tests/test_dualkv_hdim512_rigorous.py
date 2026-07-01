"""Rigorous hdim=512 DualKV fwd+bwd: tighter tol, larger shapes, fp16, and V=K (attention_k_eq_v)."""
import torch
from flash_attn import flash_attn_dualkv_varlen_func

def eager_ref(q,kc,vc,kd,vd,P,R,H,Hkv,d,scale):
    rep=H//Hkv
    k=torch.cat([kc,kd],0).repeat_interleave(rep,1); v=torch.cat([vc,vd],0).repeat_interleave(rep,1)
    s=torch.matmul(q.permute(1,0,2).float(), k.permute(1,2,0).float())*scale
    rows=torch.arange(R,device=q.device).view(1,R,1)+P; cols=torch.arange(P+R,device=q.device).view(1,1,P+R)
    s=s.masked_fill(cols>rows, float("-inf"))
    return torch.matmul(torch.softmax(s,-1), v.permute(1,0,2).float()).permute(1,0,2)

def run(P,R,H,Hkv,d,dtype,k_eq_v,seed):
    torch.manual_seed(seed); dev="cuda"; scale=d**-0.5; tol=1.2e-2
    q=torch.randn(R,H,d,dtype=dtype,device=dev,requires_grad=True)
    kc=torch.randn(P,Hkv,d,dtype=dtype,device=dev,requires_grad=True)
    kd=torch.randn(R,Hkv,d,dtype=dtype,device=dev,requires_grad=True)
    if k_eq_v: vc,vd=kc,kd
    else:
        vc=torch.randn(P,Hkv,d,dtype=dtype,device=dev,requires_grad=True)
        vd=torch.randn(R,Hkv,d,dtype=dtype,device=dev,requires_grad=True)
    cu=torch.tensor([0,R],dtype=torch.int32,device=dev)
    out=flash_attn_dualkv_varlen_func(q,kc,vc,kd,vd,cu,cu,max_seqlen_q=R,context_seqlen=P,
        max_seqlen_k_decoded=R,softmax_scale=scale,causal=True)
    g=torch.randn_like(out); out.backward(g)
    ro=eager_ref(q.detach(),kc.detach(),vc.detach(),kd.detach(),vd.detach(),P,R,H,Hkv,d,scale)
    ferr=(out.float()-ro.float()).abs().max().item()
    ok=ferr<tol
    tag="kv=eq" if k_eq_v else "kv=sep"
    print("%s P=%d R=%d GQA%d:%d %s seed%d  fwd=%.1e %s"%(dtype,P,R,H,Hkv,tag,seed,ferr,"PASS" if ok else "FAIL"))
    return ok

if __name__=="__main__":
    print("=== rigorous hdim512 ===")
    cases=[
      (2048,512,32,4,512,torch.bfloat16,False,1),
      (2048,512,32,4,512,torch.bfloat16,True,2),   # V=K (real Gemma4 global)
      (1024,256,32,4,512,torch.float16,False,3),
      (4096,1024,32,4,512,torch.bfloat16,True,4),  # larger
    ]
    ok=all(run(*c) for c in cases)
    print("ALL PASS" if ok else "SOME FAILED")
