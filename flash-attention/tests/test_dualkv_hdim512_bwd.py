"""DualKV hdim=512 FWD+BWD correctness, Gemma4 global-layer shape (hd=512, GQA 32:4)."""
import torch
from flash_attn import flash_attn_dualkv_varlen_func

def eager_ref(q_dec,k_ctx,v_ctx,k_dec,v_dec,P,R,H,Hkv,d,scale):
    rep=H//Hkv
    k=torch.cat([k_ctx,k_dec],0).repeat_interleave(rep,dim=1)
    v=torch.cat([v_ctx,v_dec],0).repeat_interleave(rep,dim=1)
    q=q_dec
    s=torch.matmul(q.permute(1,0,2).float(), k.permute(1,2,0).float())*scale
    rows=torch.arange(R,device=q.device).view(1,R,1)+P
    cols=torch.arange(P+R,device=q.device).view(1,1,P+R)
    s=s.masked_fill(cols>rows, float("-inf"))
    a=torch.softmax(s,dim=-1)
    o=torch.matmul(a, v.permute(1,0,2).float())
    return o.permute(1,0,2)

def run(P,R,H=32,Hkv=4,d=512,dtype=torch.bfloat16):
    torch.manual_seed(0); dev="cuda"; scale=d**-0.5
    def mk(n,h): 
        t=torch.randn(n,h,d,dtype=dtype,device=dev,requires_grad=True); return t
    q=mk(R,H); kc=mk(P,Hkv); vc=mk(P,Hkv); kd=mk(R,Hkv); vd=mk(R,Hkv)
    cu=torch.tensor([0,R],dtype=torch.int32,device=dev)
    out=flash_attn_dualkv_varlen_func(q,kc,vc,kd,vd,cu,cu,max_seqlen_q=R,context_seqlen=P,
        max_seqlen_k_decoded=R,softmax_scale=scale,causal=True)
    g=torch.randn_like(out)
    out.backward(g)
    dq,dkc,dvc,dkd,dvd=[t.grad.clone() for t in (q,kc,vc,kd,vd)]
    for t in (q,kc,vc,kd,vd): t.grad=None
    # eager ref grads
    qr=q.detach().requires_grad_(); kcr=kc.detach().requires_grad_(); vcr=vc.detach().requires_grad_()
    kdr=kd.detach().requires_grad_(); vdr=vd.detach().requires_grad_()
    ro=eager_ref(qr,kcr,vcr,kdr,vdr,P,R,H,Hkv,d,scale).to(dtype)
    ro.backward(g)
    res={}
    for nm,(a,b) in {"O":(out,ro),"dQ":(dq,qr.grad),"dKc":(dkc,kcr.grad),"dVc":(dvc,vcr.grad),
                     "dKd":(dkd,kdr.grad),"dVd":(dvd,vdr.grad)}.items():
        e=(a.float()-b.float()).abs().max().item()
        res[nm]=(e, torch.allclose(a.float(),b.float(),atol=3e-2,rtol=3e-2))
    allok=all(ok for _,ok in res.values())
    print("P=%d R=%d: "%(P,R)+" ".join("%s=%.1e%s"%(k,e,"" if ok else "!FAIL") for k,(e,ok) in res.items())+(" [PASS]" if allok else " [FAIL]"))
    return allok

if __name__=="__main__":
    print("=== DualKV hdim=512 FWD+BWD (Gemma4 global shape) ===")
    ok=all([run(256,128),run(512,256)])
    print("ALL PASS" if ok else "SOME FAILED")
