"""Wrapper per-layer dispatch test (corrected reference): global=DualKV hd512, sliding=FA2 windowed."""
import torch
from verl.models.transformers.monkey_patch import _make_dualkv_flash_wrapper
torch.manual_seed(0); dev="cuda"; dt=torch.bfloat16

def eager_ctxdec(qd,kc,vc,kd,vd,P,R,H,Hkv,d,scale,W):
    rep=H//Hkv
    kk=torch.cat([kc,kd],0).repeat_interleave(rep,1); vv=torch.cat([vc,vd],0).repeat_interleave(rep,1)
    sc=torch.matmul(qd.permute(1,0,2).float(),kk.permute(1,2,0).float())*scale
    qpos=torch.arange(R,device=dev).view(1,R,1)+P; kpos=torch.arange(P+R,device=dev).view(1,1,P+R)
    m=(kpos>qpos)
    if W: m=m|(kpos<(qpos-(W-1)))
    sc=sc.masked_fill(m,float("-inf"))
    return torch.matmul(torch.softmax(sc,-1),vv.permute(1,0,2).float()).permute(1,0,2)

def run(name,d,window):
    P,N,R,H,Hkv=32,3,16,4,2; scale=d**-0.5
    q=torch.randn(P+N*R,H,d,dtype=dt,device=dev)
    k=torch.randn(P+N*R,Hkv,d,dtype=dt,device=dev)
    v=torch.randn(P+N*R,Hkv,d,dtype=dt,device=dev)
    cu_dec=torch.arange(0,N*R+1,R,dtype=torch.int32,device=dev)
    ctx={"group_info":[{"prompt_len":P,"prompt_start":0,"dec_start":P,"dec_end":P+N*R,
                        "cu_seqlens_dec":cu_dec,"max_decoded":R}]}
    wrap=_make_dualkv_flash_wrapper(lambda *a,**kw:(_ for _ in ()).throw(RuntimeError("orig")))
    kw=dict(softmax_scale=scale)
    if window: kw["sliding_window"]=window
    out=wrap(q.unsqueeze(0),k.unsqueeze(0),v.unsqueeze(0),None,P+N*R,dualkv_context=ctx,**kw)[0]
    errs=[]
    for i in range(N):
        ds=P+i*R
        ref=eager_ctxdec(q[ds:ds+R],k[:P],v[:P],k[ds:ds+R],v[ds:ds+R],P,R,H,Hkv,d,scale,window)
        errs.append((out[ds:ds+R].float()-ref.float()).abs().max().item())
    e=max(errs); ok=e<2e-2
    print("%s d=%d win=%s: resp max_abs=%.2e %s"%(name,d,window,e,"PASS" if ok else "FAIL"))
    return ok

if __name__=="__main__":
    print("=== wrapper per-layer dispatch (corrected ref) ===")
    a=run("global",512,None); b=run("sliding",256,64)
    print("ALL PASS" if a and b else "SOME FAILED")
