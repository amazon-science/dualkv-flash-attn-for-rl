"""Policy-update correctness: a full fwd -> GRPO-style loss -> bwd through Gemma4
in the DualKV-packed path must match the standard per-sequence path, for BOTH the
loss value AND the parameter gradients.

This is what verl's update_actor does (minus the rollout): the response-token
log-probs feed a clipped surrogate loss, then backprop flows through the DualKV
kernels (global hd512 + sliding hd256 windowed), the per-layer dispatch, and the
_dualkv repack. Matching grads here proves the DualKV training path is correct,
independent of the (separately-broken) vLLM rollout.

Ground truth = standard per-sequence forward/backward (N sequences of [prompt;resp_i]).
DualKV path = single packed [prompt; resp_0..resp_{N-1}] through the wrapper.
"""
import torch
import torch.nn.functional as F
from transformers import Gemma4TextConfig
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
from verl.models.transformers.monkey_patch import apply_monkey_patch
from flash_attn.bert_padding import pad_input, unpad_input

torch.manual_seed(0)
dev = "cuda"

# Small Gemma4 with the real hybrid geometry: sliding hd256(W) + global hd512, V=K.
cfg = Gemma4TextConfig(
    num_hidden_layers=6, hidden_size=512, intermediate_size=1024,
    num_attention_heads=4, num_key_value_heads=2, head_dim=128, global_head_dim=256,
    num_global_key_value_heads=2, sliding_window=64, vocab_size=256,
    attention_k_eq_v=True,
)
model = Gemma4ForCausalLM(cfg).to(dev).to(torch.bfloat16)
model.config._attn_implementation = "flash_attention_2"

P, N, R = 24, 3, 12  # prompt len, num responses (group size), response len
prompt = torch.randint(1, 256, (P,), device=dev)
resps = [torch.randint(1, 256, (R,), device=dev) for _ in range(N)]
# fixed per-token "advantage" + old_logprob so the loss is deterministic across paths
adv = torch.randn(N, R, device=dev, dtype=torch.float32)
old_lp = torch.randn(N, R, device=dev, dtype=torch.float32) * 0.1


def token_logprobs(logits, target_ids):
    """log p(target) per position. logits (T,V) aligned so logits[i] predicts token i+1."""
    logp = F.log_softmax(logits.float(), dim=-1)
    return logp.gather(-1, target_ids.unsqueeze(-1)).squeeze(-1)


def grpo_loss(resp_logits_list):
    """Clipped surrogate over response tokens, summed over the N sequences.
    resp_logits_list[i] = (R, V) logits predicting response i's tokens."""
    total = 0.0
    for i, lg in enumerate(resp_logits_list):
        # lg[t] predicts resp[i][t+1]; last predicts nothing useful — use shifted targets
        tgt = resps[i]
        lp = token_logprobs(lg[:-1], tgt[1:])           # (R-1,)
        ratio = torch.exp(lp - old_lp[i, 1:R])
        a = adv[i, 1:R]
        unclipped = ratio * a
        clipped = torch.clamp(ratio, 0.8, 1.2) * a
        total = total - torch.min(unclipped, clipped).mean()
    return total / N


def clone_model(m):
    import copy
    m2 = copy.deepcopy(m)
    return m2


# ---------- PATH A: standard per-sequence fwd/loss/bwd (ground truth) ----------
model_std = model
model_std.zero_grad(set_to_none=True)
resp_logits_std = []
for r in resps:
    ids = torch.cat([prompt, r]).unsqueeze(0)
    out = model_std(ids, use_cache=False).logits[0, P:]   # (R, V) response-token logits
    resp_logits_std.append(out)
loss_std = grpo_loss(resp_logits_std)
loss_std.backward()
grads_std = {n: p.grad.detach().clone() for n, p in model_std.named_parameters() if p.grad is not None}
print(f"[standard] loss={loss_std.item():.6f}  grads={len(grads_std)} params")


# ---------- PATH B: DualKV-packed fwd/loss/bwd ----------
model_dk = clone_model(model)   # same init weights
apply_monkey_patch(model_dk, use_remove_padding=True, ulysses_sp_size=1)  # dualkv triggers via dualkv_context kwarg
model_dk.zero_grad(set_to_none=True)

# build packed [prompt; resp_0; ...; resp_{N-1}] + dualkv_context (mirrors dp_actor _dualkv_repack output)
packed = torch.cat([prompt] + resps).unsqueeze(0)         # (1, P+N*R)
T = packed.size(1)
# position_ids MUST restart per response (each resp continues the prompt: P..P+R-1),
# mirroring the standard per-sequence [prompt;resp_i] positions, else grads won't match.
pos = torch.cat([torch.arange(P, device=dev)] +
                [torch.arange(P, P + R, device=dev) for _ in range(N)]).unsqueeze(0)
# group_info: one prompt group, N decoded sequences
cu_dec = torch.arange(0, N * R + 1, R, dtype=torch.int32, device=dev)
group_info = [{"prompt_len": P, "prompt_start": 0, "dec_start": P, "dec_end": P + N * R,
               "cu_seqlens_dec": cu_dec, "max_decoded": R}]
dualkv_ctx = {"group_info": group_info}

out = model_dk(input_ids=packed, attention_mask=None, position_ids=pos,
               use_cache=False, dualkv_context=dualkv_ctx).logits[0]   # (T, V)
# slice out each response's logits from the packed output
resp_logits_dk = []
for i in range(N):
    s = P + i * R
    resp_logits_dk.append(out[s:s + R])
loss_dk = grpo_loss(resp_logits_dk)
loss_dk.backward()
grads_dk = {n: p.grad.detach().clone() for n, p in model_dk.named_parameters() if p.grad is not None}
print(f"[dualkv]   loss={loss_dk.item():.6f}  grads={len(grads_dk)} params")


# ---------- COMPARE ----------
loss_err = abs(loss_std.item() - loss_dk.item())
loss_ok = loss_err < 1e-2
print(f"\nLOSS: std={loss_std.item():.6f} dualkv={loss_dk.item():.6f} |diff|={loss_err:.2e} [{'PASS' if loss_ok else 'FAIL'}]")

print("GRADIENTS (per-param max_abs diff, top offenders):")
common = sorted(set(grads_std) & set(grads_dk))
worst = []
all_ok = True
for n in common:
    a, b = grads_std[n].float(), grads_dk[n].float()
    e = (a - b).abs().max().item()
    ok = torch.allclose(a, b, atol=3e-2, rtol=3e-2)
    all_ok = all_ok and ok
    worst.append((e, n, ok))
worst.sort(reverse=True)
for e, n, ok in worst[:8]:
    print(f"   {('FAIL' if not ok else 'ok  ')}  max_abs={e:.2e}  {n}")
print(f"\nparams compared: {len(common)} | grads all-match: {all_ok}")
print("ALL PASS" if (loss_ok and all_ok) else "SOME FAILED")
