#!/usr/bin/env python3
"""DualKV + Ulysses SP attention-seam reproduction (Bug B).

Reproduces the o_proj shape crash (mat1 width = hidden/SP instead of hidden) that
hits Gemma4-31B under DualKV + SP>1. Root cause: Gemma4 is a
`Gemma4ForConditionalGeneration` (has vision_config), so verl's prepare_model_inputs
takes the VLM branch that PADS BUT DOES NOT SLICE the sequence, expecting a
`patch_vlm_for_ulysses_input_slicing` hook that is NOT wired for gemma4. The DualKV
attention wrapper then all-to-alls as if the seq were sharded, so its output seq
disagrees with HF's `input_shape` (= T_full) by a factor of SP -> reshape mis-splits.

Drives the REAL DualKV wrapper + the REAL HF-style reshape/o_proj. Runs the world's
SP degree (set nproc_per_node). Both sub-cases per run:
  sliced=False  -> the BUG (Gemma4 VLM no-slice): hidden enters at T_full
  sliced=True   -> the FIX (seq sharded before module): hidden enters at T_local

Expectation:
  SP == 1: no sharding -> both paths clean (bug NOT reproducible; control).
  SP  > 1: unsliced path MUST crash in o_proj; sliced path MUST be clean.

    torchrun --nproc_per_node=1 test_dualkv_sp_attn_seam.py   # control
    torchrun --nproc_per_node=2 test_dualkv_sp_attn_seam.py
    torchrun --nproc_per_node=4 test_dualkv_sp_attn_seam.py
    torchrun --nproc_per_node=8 test_dualkv_sp_attn_seam.py
"""
import sys, torch, torch.distributed as dist, torch.nn as nn

# Gemma4-31B global-layer geometry (the crashing layer)
NUM_Q_HEADS = 32
NUM_KV_HEADS = 16
HEAD_DIM = 256
HIDDEN = NUM_Q_HEADS * HEAD_DIM  # 8192, o_proj in_features
DTYPE = torch.bfloat16


def build_group_info(P, resp_lens, device):
    n = len(resp_lens)
    cu = [0]
    for R in resp_lens:
        cu.append(cu[-1] + R)
    total = P + sum(resp_lens)
    gi = [{
        "prompt_start": 0, "prompt_len": P,
        "dec_start": P, "dec_end": total,
        "cu_seqlens_dec": torch.tensor(cu, device=device, dtype=torch.int32),
        "max_decoded": max(resp_lens), "n_seqs": n, "response_lens": resp_lens,
    }]
    return gi, total


def run_seam(sliced, sp_group, sp_size, rank, device):
    """Return (o_proj_in_width, ok, err). `sliced`=True mirrors the FIX; False the BUG."""
    from verl.models.transformers.monkey_patch import _make_dualkv_flash_wrapper
    from verl.utils.ulysses import set_ulysses_sequence_parallel_group

    set_ulysses_sequence_parallel_group(sp_group if sp_size > 1 else None)

    P, resp_lens = 256, [64, 96, 48, 80]
    gi, T_full = build_group_info(P, resp_lens, device)
    pad = (sp_size - T_full % sp_size) % sp_size
    T_pad = T_full + pad

    torch.manual_seed(0)
    o_proj = nn.Linear(HIDDEN, HIDDEN, bias=False).to(device, DTYPE)

    seq_local = (T_pad // sp_size) if sliced else T_pad
    hidden = torch.randn(1, seq_local, HIDDEN, device=device, dtype=DTYPE)
    input_shape = hidden.shape[:-1]  # what HF captures

    def proj(nh):
        w = nn.Linear(HIDDEN, nh * HEAD_DIM, bias=False).to(device, DTYPE)
        return w(hidden).view(1, seq_local, nh, HEAD_DIM)
    q, k, v = proj(NUM_Q_HEADS), proj(NUM_KV_HEADS), proj(NUM_KV_HEADS)

    def _orig(*a, **kw):
        raise RuntimeError("dualkv_context should trigger the wrapper")
    wrapper = _make_dualkv_flash_wrapper(_orig)
    attn_out = wrapper(q, k, v, None, seq_local,
                       dualkv_context={"group_info": gi}, softmax_scale=HEAD_DIM ** -0.5)

    reshaped = attn_out.reshape(*input_shape, -1).contiguous()  # modeling_gemma4.py:975
    width = reshaped.shape[-1]
    try:
        o_proj(reshaped)
        return width, True, None
    except RuntimeError as e:
        return width, False, str(e)[:80]


def main():
    dist.init_process_group("nccl")
    rank, world = dist.get_rank(), dist.get_world_size()
    torch.cuda.set_device(rank)
    device = torch.device("cuda", rank)
    sp_group = dist.new_group(ranks=list(range(world))) if world > 1 else None

    bug_w, bug_ok, bug_err = run_seam(False, sp_group, world, rank, device)
    dist.barrier()
    fix_w, fix_ok, fix_err = run_seam(True, sp_group, world, rank, device)

    if rank == 0:
        # SP=1: bug path should ALSO be clean (control). SP>1: bug path must crash.
        if world == 1:
            passed = bug_ok and fix_ok
            verdict = "PASS (control: no crash at SP=1)" if passed else "FAIL"
        else:
            passed = (not bug_ok) and fix_ok and (bug_w == HIDDEN // world) and (fix_w == HIDDEN)
            verdict = "PASS (bug reproduced + fix clean)" if passed else "FAIL"
        print(f"[SP={world}] unsliced: width={bug_w} ok={bug_ok} err={bug_err}", flush=True)
        print(f"[SP={world}]   sliced: width={fix_w} ok={fix_ok} err={fix_err}", flush=True)
        print(f"[SP={world}] {verdict}", flush=True)
    dist.destroy_process_group()


if __name__ == "__main__":
    main()
