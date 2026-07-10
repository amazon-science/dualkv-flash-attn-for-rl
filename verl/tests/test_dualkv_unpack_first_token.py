"""
DualKV first-response-token log-prob unpack parity test (CPU, cheap, standalone).

Targets the bug fixed in `_dualkv_unpack_logprobs` (verl/workers/engine/fsdp/
transformer_impl.py): the shared prompt-last position P-1 predicts the FIRST token of
EVERY response in a group, but in the packed stream that single position carries only ONE
per-token log-prob (response-0's). The old code copied that scalar to all N responses:

    out[o_start + P - 1] = log_probs_packed[prompt_start + P - 1]   # WRONG for responses 1..N-1

so every response in a group got response-0's first-token log-prob. The fix passes
`shared_logits` (one full logit row per group) and recomputes each response's first-token
log-prob as log_softmax(shared_logits[g])[that response's own first token].

This test runs on CPU in milliseconds against synthetic tensors — it does NOT load a model,
touch CUDA, or run inside the training loop. It exercises the real production function.

Usage:
    python tests/test_dualkv_unpack_first_token.py
    pytest tests/test_dualkv_unpack_first_token.py -v
"""

import os
import sys

import pytest
import torch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

try:
    from verl.workers.engine.fsdp.transformer_impl import _dualkv_unpack_logprobs
    _IMPORT_ERR = None
except Exception as e:  # heavy FSDP/peft deps may be absent on a dev box
    _dualkv_unpack_logprobs = None
    _IMPORT_ERR = e


# Two groups, N=2 responses each, every response has a DISTINCT first token. Distinct first
# tokens within a group are exactly what the old scalar-copy bug got wrong.
#   group 0: P=3, response_lens=[2, 2], first tokens = [5, 7]
#   group 1: P=2, response_lens=[3, 1], first tokens = [9, 4]
# Packed layout: [P0(3) r00(2) r01(2) | P1(2) r10(3) r11(1)]  -> total_packed = 13
VOCAB = 11
FIRST_TOKENS = [5, 7, 9, 4]  # per sequence, across both groups


def _make_repack_info():
    group_info = [
        {
            "prompt_len": 3,
            "prompt_start": 0,
            "dec_start": 3,
            "cu_seqlens_dec": torch.tensor([0, 2, 4], dtype=torch.int32),
            "n_seqs": 2,
            "response_lens": [2, 2],
        },
        {
            "prompt_len": 2,
            "prompt_start": 7,
            "dec_start": 9,
            "cu_seqlens_dec": torch.tensor([0, 3, 4], dtype=torch.int32),
            "n_seqs": 2,
            "response_lens": [3, 1],
        },
    ]
    return {"group_info": group_info, "first_response_tokens": list(FIRST_TOKENS)}


# Original [Pi;Ri] layout: seq lens = [5, 5, 5, 3] -> offsets [0,5,10,15,18]
ORIG_CU_SEQLENS = torch.tensor([0, 5, 10, 15, 18], dtype=torch.int64)


def _make_inputs(seed=0):
    torch.manual_seed(seed)
    # packed per-token log-probs (only the response tail positions are read for tokens 1..R-1;
    # the P-1 slots hold response-0's value that the buggy path would wrongly broadcast)
    log_probs_packed = torch.randn(13, dtype=torch.float32)
    shared_logits = torch.randn(2, VOCAB, dtype=torch.float32)  # one row per group
    return log_probs_packed, shared_logits


@pytest.mark.skipif(_dualkv_unpack_logprobs is None, reason=f"import failed: {_IMPORT_ERR}")
def test_first_token_uses_each_responses_own_token():
    """Each response's first-token log-prob = log_softmax(shared_logits[group])[its own first token],
    and responses in the same group get DIFFERENT values (the fix), not response-0's scalar (the bug)."""
    repack_info = _make_repack_info()
    log_probs_packed, shared_logits = _make_inputs()

    out = _dualkv_unpack_logprobs(log_probs_packed, repack_info, ORIG_CU_SEQLENS, shared_logits=shared_logits)

    lsm0 = torch.log_softmax(shared_logits[0].float(), dim=-1)
    lsm1 = torch.log_softmax(shared_logits[1].float(), dim=-1)

    # first-token slots live at o_start + P - 1: seq0->2, seq1->7, seq2->11, seq3->16
    assert torch.allclose(out[2], lsm0[FIRST_TOKENS[0]], atol=1e-6)   # group0 resp0, token 5
    assert torch.allclose(out[7], lsm0[FIRST_TOKENS[1]], atol=1e-6)   # group0 resp1, token 7
    assert torch.allclose(out[11], lsm1[FIRST_TOKENS[2]], atol=1e-6)  # group1 resp0, token 9
    assert torch.allclose(out[16], lsm1[FIRST_TOKENS[3]], atol=1e-6)  # group1 resp1, token 4

    # the two responses in group 0 have distinct first tokens -> distinct log-probs (bug would tie them)
    assert not torch.allclose(out[2], out[7]), "responses in a group must not share response-0's first-token log-prob"


@pytest.mark.skipif(_dualkv_unpack_logprobs is None, reason=f"import failed: {_IMPORT_ERR}")
def test_response_tail_is_copied_verbatim():
    """Tokens 1..R-1 of each response come straight from the packed per-token log-probs."""
    repack_info = _make_repack_info()
    log_probs_packed, shared_logits = _make_inputs()

    out = _dualkv_unpack_logprobs(log_probs_packed, repack_info, ORIG_CU_SEQLENS, shared_logits=shared_logits)

    # seq0 tail: out[3:4] <- packed[3:4] (ds=dec_start0+cu_dec[0]=3)
    assert torch.allclose(out[3:4], log_probs_packed[3:4])
    # seq1 tail: out[8:9] <- packed[5:6] (ds=3+2=5)
    assert torch.allclose(out[8:9], log_probs_packed[5:6])
    # seq2 tail (R=3): out[12:14] <- packed[9:11] (ds=dec_start1+cu_dec[0]=9)
    assert torch.allclose(out[12:14], log_probs_packed[9:11])
    # seq3 has R=1 -> no tail; only its first-token slot out[16] is written


@pytest.mark.skipif(_dualkv_unpack_logprobs is None, reason=f"import failed: {_IMPORT_ERR}")
def test_none_shared_logits_reproduces_scalar_copy():
    """Contrast: with shared_logits=None the function keeps the scalar-copy behavior (correct for
    entropy / Σπ², which are label-independent). All responses in a group then share response-0's
    P-1 scalar — exactly the behavior the log-prob path must NOT use."""
    repack_info = _make_repack_info()
    log_probs_packed, _ = _make_inputs()

    out = _dualkv_unpack_logprobs(log_probs_packed, repack_info, ORIG_CU_SEQLENS, shared_logits=None)

    # both group-0 responses get the group's P-1 packed scalar (prompt_start+P-1 = 0+2 = 2)
    assert torch.allclose(out[2], log_probs_packed[2])
    assert torch.allclose(out[7], log_probs_packed[2])
    assert torch.allclose(out[2], out[7])  # tied — fine for entropy, wrong for log-probs


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v", "--tb=short"]))
