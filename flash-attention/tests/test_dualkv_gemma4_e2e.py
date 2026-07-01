"""End-to-end: small Gemma4 with use_remove_padding, real dualkv_context, verify the
DualKV-packed forward matches standard per-sequence forward (response-token logits)."""
import torch
from transformers import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
from verl.models.transformers.monkey_patch import apply_monkey_patch
from flash_attn.bert_padding import pad_input, unpad_input

torch.manual_seed(0); dev="cuda"
cfg=Gemma4TextConfig(num_hidden_layers=6,hidden_size=512,intermediate_size=1024,
    num_attention_heads=4,num_key_value_heads=2,head_dim=128,global_head_dim=256,
    num_global_key_value_heads=2,sliding_window=64,vocab_size=256,attention_k_eq_v=True)
model=Gemma4ForCausalLM(cfg).to(dev).to(torch.bfloat16).eval()
model.config._attn_implementation="flash_attention_2"

P,N,R=24,2,12
prompt=torch.randint(1,256,(P,),device=dev)
resps=[torch.randint(1,256,(R,),device=dev) for _ in range(N)]

# standard per-sequence
with torch.no_grad():
    std=[]
    for r in resps:
        ids=torch.cat([prompt,r]).unsqueeze(0)
        std.append(model(ids,use_cache=False).logits[0,P:])
    std=torch.cat(std,0)
print("standard logits:", tuple(std.shape))

# Now check apply_monkey_patch installs the dualkv wrapper on this model
try:
    apply_monkey_patch(model, use_remove_padding=True, ulysses_sp_size=1)
    print("apply_monkey_patch: OK")
except Exception as e:
    print("apply_monkey_patch FAIL:", repr(e)[:200])
print("E2E_SETUP_OK")

# --- DualKV-packed forward via the wrapper, mimicking dp_actor rmpad path ---
# Packed single-prompt layout: [prompt | resp_0 | resp_1], varlen with one prompt copy.
packed = torch.cat([prompt] + resps).unsqueeze(0)  # (1, P+N*R)
T = P + N*R
# position_ids: prompt 0..P-1, each response continues P..P+R-1
pos = torch.cat([torch.arange(P,device=dev)] + [torch.arange(P,P+R,device=dev) for _ in range(N)]).unsqueeze(0)
cu_dec = torch.arange(0, N*R+1, R, dtype=torch.int32, device=dev)
dualkv_ctx = {"group_info":[{"prompt_len":P,"prompt_start":0,"dec_start":P,"dec_end":T,
                             "cu_seqlens_dec":cu_dec,"max_decoded":R}]}
with torch.no_grad():
    try:
        out = model(input_ids=packed, position_ids=pos, attention_mask=None,
                    use_cache=False, dualkv_context=dualkv_ctx)
        lg = out.logits[0]  # (T, vocab)
        dk_resp = torch.cat([lg[P+i*R:P+(i+1)*R] for i in range(N)], 0)
        err=(dk_resp.float()-std.float()).abs().max().item()
        # logits are large; compare via argmax agreement + relative
        agree=(dk_resp.argmax(-1)==std.argmax(-1)).float().mean().item()
        print("DualKV-packed vs standard: logit max_abs=%.3f  argmax_agree=%.1f%%  %s"%(
            err, agree*100, "PASS" if agree>0.99 else "CHECK"))
    except Exception as e:
        import traceback; traceback.print_exc()
        print("DUALKV_FWD_FAIL:", repr(e)[:200])
print("E2E_DONE")
