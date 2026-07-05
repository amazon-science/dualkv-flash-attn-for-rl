"""Unit test for DualKV per-group prompt-length derivation in verl 0.8.0 (NO_PADDING path).

Locks in the fix in ``prepare_model_inputs``' DualKV branch: instead of threading an
integer non-tensor (``prompt_len``) through DataProto/tensordict — which silently
collapses per-row int32 arrays to a SCALAR during microbatch chunking (unlike the
object-dtype ``uid`` array, which survives) — the per-group prompt length P is derived
PER ROW from tensors that are always reliably present:

    P_i = seq_total_i - R_i

where ``seq_total_i`` is the row's full [P;R] token count (from ``cu_seqlens``) and
``R_i`` is its response length (``response_mask[i].sum()``). Layout-independent, needs
no rollout metadata, and every row in a shared-prompt uid-group must yield the same P.

CPU-only, seconds to run. Feeds the derived prompt_lens into the real ``_dualkv_repack``
and checks the packed layout is [P, R0..R_{N-1}] per group with the correct shared P.

Run: python -m pytest verl/tests/test_dualkv_prompt_lens.py -q
  or: python verl/tests/test_dualkv_prompt_lens.py
"""
import numpy as np
import torch

from verl.workers.engine.fsdp.transformer_impl import _dualkv_repack, _compute_prompt_group_sizes


def _build_batch(groups):
    """groups: list of (P, [R0, R1, ...]) -> build the NO_PADDING per-sequence layout.

    Returns a dict mimicking the fields prepare_model_inputs reads at the DualKV branch:
      input_ids_rmpad (1, total_nnz), position_ids_rmpad (1, total_nnz),
      cu_seqlens (bs+1,), uid (per-seq), response_mask (bs, max_R).
    Token ids are chosen so prompt tokens are shared within a group and responses differ.
    """
    seqs, pos, uids, resp_lens = [], [], [], []
    gid = 0
    tok = 1000
    for (P, Rs) in groups:
        prompt_toks = list(range(tok, tok + P))  # this group's shared prompt tokens
        tok += P
        for R in Rs:
            resp_toks = list(range(tok, tok + R))  # unique response tokens
            tok += R
            seqs.append(torch.tensor(prompt_toks + resp_toks, dtype=torch.long))
            pos.append(torch.arange(P + R, dtype=torch.long))
            uids.append(f"uid{gid}")
            resp_lens.append(R)
        gid += 1
    input_ids_rmpad = torch.cat(seqs).unsqueeze(0)             # (1, total_nnz)
    position_ids_rmpad = torch.cat(pos).unsqueeze(0)           # (1, total_nnz)
    lengths = [s.numel() for s in seqs]
    cu = torch.tensor([0] + list(np.cumsum(lengths)), dtype=torch.int64)
    # response_mask: (bs, max_R), 1 for real response tokens, 0 for right-padding
    max_R = max(resp_lens)
    resp_mask = torch.zeros(len(resp_lens), max_R, dtype=torch.long)
    for i, R in enumerate(resp_lens):
        resp_mask[i, :R] = 1
    return {
        "input_ids_rmpad": input_ids_rmpad,
        "position_ids_rmpad": position_ids_rmpad,
        "cu_seqlens": cu,
        "uid": np.array(uids, dtype=object),
        "response_mask": resp_mask,
    }


def _derive_prompt_lens(batch):
    """The exact derivation from the fix in prepare_model_inputs (tensor-based P)."""
    cu_seqlens = batch["cu_seqlens"]
    uids = batch["uid"]
    bs = int(cu_seqlens.numel() - 1)
    prompt_group_sizes = _compute_prompt_group_sizes(list(uids), bs)
    seq_totals = cu_seqlens.diff().tolist()
    resp_mask = batch["response_mask"]
    assert resp_mask is not None and isinstance(resp_mask, torch.Tensor)
    resp_lens = resp_mask.reshape(resp_mask.size(0), -1).sum(dim=1).tolist()
    assert len(resp_lens) == bs
    per_row_P = [int(seq_totals[i]) - int(resp_lens[i]) for i in range(bs)]
    prompt_lens = []
    si = 0
    for gsz in prompt_group_sizes:
        _P = int(per_row_P[si])
        assert 0 < _P < int(seq_totals[si]), f"P={_P} out of range for total={seq_totals[si]}"
        for _j in range(si, si + gsz):
            _Pj = int(seq_totals[_j]) - int(resp_lens[_j])
            assert _Pj == _P, f"group at {si} disagrees on P ({_P} vs {_Pj} at row {_j})"
        prompt_lens.append(_P)
        si += gsz
    return prompt_lens, prompt_group_sizes


def test_prompt_lens_multi_group():
    # 3 groups, different P and N and response lengths
    groups = [(5, [7, 9, 8]), (4, [6, 10]), (11, [3, 3, 3, 4])]
    batch = _build_batch(groups)
    prompt_lens, gsizes = _derive_prompt_lens(batch)
    assert gsizes == [3, 2, 4], gsizes
    assert prompt_lens == [5, 4, 11], prompt_lens


def test_single_sample_groups():
    # N=1 per group (each uid distinct) — the case that crashed the naive unwrap.
    groups = [(5, [7]), (4, [6]), (11, [3]), (8, [9])]
    batch = _build_batch(groups)
    prompt_lens, gsizes = _derive_prompt_lens(batch)
    assert gsizes == [1, 1, 1, 1], gsizes
    assert prompt_lens == [5, 4, 11, 8], prompt_lens


def test_repack_uses_correct_P():
    # feed derived prompt_lens into the REAL _dualkv_repack; check packed layout
    groups = [(5, [7, 9, 8]), (4, [6, 10])]
    batch = _build_batch(groups)
    prompt_lens, gsizes = _derive_prompt_lens(batch)
    ids_packed, pos_packed, dualkv_ctx, repack_info = _dualkv_repack(
        batch["input_ids_rmpad"], batch["cu_seqlens"], batch["position_ids_rmpad"],
        prompt_lens, gsizes,
    )
    # packed size = sum over groups of (P + sum(R_i)); prompt counted ONCE per group
    exp_packed = sum(P + sum(Rs) for (P, Rs) in groups)
    assert ids_packed.size(1) == exp_packed, (ids_packed.size(1), exp_packed)
    # group_info P matches derivation
    gi = repack_info["group_info"]
    assert [g["prompt_len"] for g in gi] == prompt_lens, [g["prompt_len"] for g in gi]
    # prompt tokens in packed stream equal the group's original prompt prefix
    ids_flat = ids_packed.squeeze(0)
    cu = batch["cu_seqlens"].tolist()
    si = 0
    for (P, Rs), g in zip(groups, gi):
        orig_prompt = batch["input_ids_rmpad"].squeeze(0)[cu[si]: cu[si] + P]
        packed_prompt = ids_flat[g["prompt_start"]: g["prompt_start"] + P]
        assert torch.equal(orig_prompt, packed_prompt), (orig_prompt, packed_prompt)
        si += len(Rs)


def test_inconsistent_group_raises():
    # If rows in a uid-group disagree on P (shared-prompt assumption violated), the
    # derivation must fail LOUDLY rather than silently repack with the wrong prompt.
    groups = [(5, [7, 9])]
    batch = _build_batch(groups)
    # corrupt: shrink the 2nd row's response by 1 without changing its total, so its
    # derived P differs from the 1st row's -> tamper response_mask to over-count.
    batch["response_mask"][1, :] = 0
    batch["response_mask"][1, :3] = 1  # pretend R=3 for a row whose real R=9
    raised = False
    try:
        _derive_prompt_lens(batch)
    except AssertionError:
        raised = True
    assert raised, "inconsistent per-row P within a uid-group must raise loudly"


if __name__ == "__main__":
    test_prompt_lens_multi_group()
    test_single_sample_groups()
    test_repack_uses_correct_P()
    test_inconsistent_group_raises()
    print("ALL DUALKV PROMPT_LENS TESTS PASSED")
