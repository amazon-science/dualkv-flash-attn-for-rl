"""FULL Gemma4-31B policy-update correctness, real GRPO-config inputs.

Loads the real google/gemma-4-31B-it text decoder (60 layers, hd256 sliding / hd512
global, V=K, 31.27B) under FSDP, applies the DualKV monkey-patch, and runs ONLY the
policy-update fwd -> GRPO clipped-surrogate loss -> bwd, two ways:
  (A) ground truth: each [prompt; resp_i] through DualKV with a SINGLE decoded seq (N=1),
      sequentially (no shared-prompt reuse across responses). NOTE: vanilla FA2 cannot be
      the reference here — the real Gemma4 global layers are head_dim=512, above FA2's 256 cap.
  (B) DualKV-packed: single [prompt; resp_0..resp_{N-1}] through the DualKV wrapper (N=mb)
and compares LOSS + every parameter GRADIENT. The (A)-vs-(B) diff isolates the DualKV
shared-prompt machinery (fp32 dKc/dVc accumulation across multiple decoded seqs, the repack,
per-layer sliding/global dispatch) at the true 31B kernel shapes (hd512 global + hd256
windowed sliding), inside a real fwd/loss/bwd, decoupled from the rollout. Kernel-vs-eager
numerics are covered separately by test_dualkv_swa_{fwd,bwd}.py.

Launch: torchrun --nproc_per_node=8 test_dualkv_policy_update_31b.py
Env: GRPO_P, GRPO_R, GRPO_N override the sequence shapes (defaults are memory-safe;
real config is P=8192 R=2048 N=16 — set those to test full scale).
"""
import os
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP, ShardingStrategy, MixedPrecision
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers import AutoConfig, AutoModelForCausalLM
from transformers.models.gemma4.modeling_gemma4 import Gemma4TextDecoderLayer
from verl.models.transformers.monkey_patch import apply_monkey_patch
# _dualkv_repack moved to engine in 0.8.0; test does not call it (builds group_info inline)
import functools

MODEL = os.environ.get("MODEL_DIR", "google/gemma-4-31b-it")  # HF id or local path to Gemma-4-31B
P = int(os.environ.get("GRPO_P", 1024))
R = int(os.environ.get("GRPO_R", 512))
N = int(os.environ.get("GRPO_N", 8))
CKPT = os.environ.get("GRPO_CKPT", "1") != "0"  # gradient checkpointing (matches verl when ON)

dist.init_process_group("nccl")
rank = dist.get_rank(); world = dist.get_world_size()
torch.cuda.set_device(rank)
dev = torch.device("cuda", rank)
def log(*a):
    if rank == 0: print(*a, flush=True)

log(f"=== Gemma4-31B policy-update test | P={P} R={R} N={N} | world={world} ===")

# ---- build the FULL real Gemma4-31B text-decoder config (60 layers, real per-layer
# geometry: sliding hd256/W=1024 + global hd512, GQA 32:16 / global 32:4, V=K) and
# instantiate the text decoder. Random init: this is a packed-vs-per-sequence NUMERICAL
# EQUIVALENCE test, which holds for any weights — what matters is the real config/shapes
# so the actual 31B kernels (hd512 global, hd256 windowed sliding) run in a full fwd/bwd.
from transformers.models.gemma4.modeling_gemma4 import Gemma4ForCausalLM
cfg = AutoConfig.from_pretrained(MODEL, trust_remote_code=True)
tcfg = getattr(cfg, "text_config", cfg)
tcfg._attn_implementation = "flash_attention_2"
log(f"real text_config: layers={tcfg.num_hidden_layers} hidden={tcfg.hidden_size} "
    f"heads={tcfg.num_attention_heads}/{tcfg.num_key_value_heads} hd={tcfg.head_dim}/"
    f"{getattr(tcfg,'global_head_dim',None)} win={tcfg.sliding_window} V=K={getattr(tcfg,'attention_k_eq_v',None)}")
model = Gemma4ForCausalLM(tcfg).to(torch.bfloat16)
apply_monkey_patch(model, use_remove_padding=True, ulysses_sp_size=1)
if CKPT and hasattr(model, "gradient_checkpointing_enable"):
    # use_reentrant=False (matches verl): reentrant checkpointing drops non-tensor kwargs
    # (dualkv_context) on recompute -> attention silently falls back to vanilla FA2 -> hd512 crash.
    model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
log(f"gradient_checkpointing={'ON (use_reentrant=False)' if CKPT else 'OFF'}")
log(f"instantiated {sum(p.numel() for p in model.parameters())/1e9:.2f}B params (full 31B text decoder, random init)")

wrap = functools.partial(transformer_auto_wrap_policy, transformer_layer_cls={Gemma4TextDecoderLayer})
model = FSDP(model, auto_wrap_policy=wrap, sharding_strategy=ShardingStrategy.FULL_SHARD,
             mixed_precision=MixedPrecision(param_dtype=torch.bfloat16, reduce_dtype=torch.float32),
             device_id=rank, use_orig_params=True)
log("FSDP wrapped")

# ---- deterministic inputs (same on all ranks) ----
g = torch.Generator(device="cpu").manual_seed(0)
V = tcfg.vocab_size
prompt = torch.randint(1, V, (P,), generator=g)
resps = [torch.randint(1, V, (R,), generator=g) for _ in range(N)]
adv = torch.randn(N, R, generator=g) * 1.0
old_lp = torch.randn(N, R, generator=g) * 0.1
prompt = prompt.to(dev); resps = [r.to(dev) for r in resps]
adv = adv.to(dev); old_lp = old_lp.to(dev)

def tok_lp(logits, tgt):
    return F.log_softmax(logits.float(), -1).gather(-1, tgt.unsqueeze(-1)).squeeze(-1)

def surrogate_term(i, lg):
    """Per-sequence GRPO clipped-surrogate contribution (already divided by N)."""
    lp = tok_lp(lg[:-1], resps[i][1:])
    ratio = torch.exp(lp - old_lp[i, 1:R])
    a = adv[i, 1:R]
    return -(torch.min(ratio * a, torch.clamp(ratio, 0.8, 1.2) * a).mean()) / N

# ---------- PATH A: ground truth = DualKV with ONE decoded seq per group (N=1), sequential ----------
# The real Gemma4 GLOBAL layers are head_dim=512; vanilla FA2 caps at 256, so the only way
# to forward a single [prompt;resp_i] at the true 31B geometry is the DualKV kernel itself
# with a single decoded sequence (no shared-prompt reuse across responses). Comparing this
# N=1-per-response path against the packed N=8 path isolates the DualKV-specific machinery:
# shared-prompt fp32 dKc/dVc accumulation across MULTIPLE decoded seqs, the repack, and the
# per-layer sliding/global dispatch. (Kernel-vs-eager numerics are covered by the SWA unittests.)
# Per-response backward + grad accumulation: GRPO loss is a SUM over sequences, so each term's
# backward accumulates into .grad and frees its graph immediately (else N graphs + N x (R,V)
# logit tensors are retained at once -> OOM at the real 31B vocab=262144).
model.zero_grad(set_to_none=True)
loss_std_val = 0.0
for i, r in enumerate(resps):
    ids = torch.cat([prompt, r]).unsqueeze(0)
    pos = torch.cat([torch.arange(P, device=dev), torch.arange(P, P + R, device=dev)]).unsqueeze(0)
    cu1 = torch.arange(0, R + 1, R, dtype=torch.int32, device=dev)  # [0, R]
    ctx1 = {"group_info": [{"prompt_len": P, "prompt_start": 0, "dec_start": P,
            "dec_end": P + R, "cu_seqlens_dec": cu1, "max_decoded": R}]}
    lg = model(input_ids=ids, attention_mask=None, position_ids=pos,
               use_cache=False, dualkv_context=ctx1).logits[0, P:]
    term = surrogate_term(i, lg)
    term.backward()
    loss_std_val += term.item()
    del lg, term
grads_std = {n: p.grad.detach().float().clone() for n, p in model.named_parameters() if p.grad is not None}
log(f"[dualkv-N1] loss={loss_std_val:.6f} grads={len(grads_std)}")

# ---------- PATH B: DualKV-packed (uses production _dualkv_repack) ----------
# Single packed forward [prompt; resp_0..resp_{N-1}]; slice each response's logits and
# accumulate the same per-sequence loss terms, then one backward over the shared graph.
model.zero_grad(set_to_none=True)
packed = torch.cat([prompt] + resps).unsqueeze(0)
T = packed.size(1)
pos = torch.cat([torch.arange(P, device=dev)] +
                [torch.arange(P, P + R, device=dev) for _ in range(N)]).unsqueeze(0)
cu_dec = torch.arange(0, N * R + 1, R, dtype=torch.int32, device=dev)
dualkv_ctx = {"group_info": [{"prompt_len": P, "prompt_start": 0, "dec_start": P,
              "dec_end": P + N * R, "cu_seqlens_dec": cu_dec, "max_decoded": R}]}
out = model(input_ids=packed, attention_mask=None, position_ids=pos,
            use_cache=False, dualkv_context=dualkv_ctx).logits[0]
loss_dk = sum(surrogate_term(i, out[P + i * R: P + (i + 1) * R]) for i in range(N))
loss_dk_val = loss_dk.item()
loss_dk.backward()
grads_dk = {n: p.grad.detach().float().clone() for n, p in model.named_parameters() if p.grad is not None}
log(f"[dualkv]   loss={loss_dk_val:.6f} grads={len(grads_dk)}")

# ---------- COMPARE (rank 0) ----------
if rank == 0:
    lerr = abs(loss_std_val - loss_dk_val)
    lok = lerr < 2e-2
    print(f"\nLOSS dualkv-N1={loss_std_val:.6f} dualkv-packed={loss_dk_val:.6f} |diff|={lerr:.2e} [{'PASS' if lok else 'FAIL'}]")
    common = sorted(set(grads_std) & set(grads_dk))
    worst, allok = [], True
    for n in common:
        a, b = grads_std[n], grads_dk[n]
        e = (a - b).abs().max().item()
        ok = torch.allclose(a, b, atol=5e-2, rtol=5e-2)
        allok = allok and ok
        worst.append((e, n, ok))
    worst.sort(reverse=True)
    print("top grad diffs:")
    for e, n, ok in worst[:10]:
        print(f"   {'FAIL' if not ok else 'ok '} max_abs={e:.2e}  {n}")
    print(f"params compared={len(common)} grads_all_match={allok}")
    print("ALL PASS" if (lok and allok) else "SOME FAILED")
dist.barrier()
dist.destroy_process_group()
