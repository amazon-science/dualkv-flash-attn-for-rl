"""
DualKV multi-group log-prob parity test.

Verifies that the DualKV path produces identical log-probs to standard FA
when a micro-batch contains multiple prompt groups with DIFFERENT prompt
lengths. This exercises the _dualkv_repack → monkey_patch → _dualkv_extract
pipeline end-to-end through a real model.

The bug: _dualkv_repack uses a single P (from the first group) for all
groups, causing wrong prompt/response boundaries for subsequent groups.

Expected: FAIL before fix, PASS after.

Usage:
    python tests/test_dualkv_multi_group_parity.py
"""

import torch
import torch.nn.functional as F
import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


def logprobs_from_logits(logits, labels):
    lp = F.log_softmax(logits.float(), dim=-1)
    return lp.gather(-1, labels.unsqueeze(-1)).squeeze(-1)


def standard_fa_logprobs(model, prompt_ids, response_ids, device):
    """Run one [prompt, response] sequence through standard FA, return response log-probs."""
    P = prompt_ids.shape[0]
    R = response_ids.shape[0]
    seq = torch.cat([prompt_ids, response_ids])
    input_ids = seq.unsqueeze(0)
    pos_ids = torch.arange(seq.shape[0], device=device).unsqueeze(0)

    with torch.no_grad():
        logits = model(
            input_ids=input_ids,
            attention_mask=None,
            position_ids=pos_ids,
            use_cache=False,
        ).logits.squeeze(0)

    resp_logits = logits[P - 1 : P + R - 1]
    return logprobs_from_logits(resp_logits, response_ids)


def dualkv_multi_group_logprobs(model, prompt_groups, device):
    """Run multiple prompt groups through DualKV in a single forward pass.

    Args:
        prompt_groups: list of (prompt_ids, [resp_ids, ...]) tuples.

    Returns:
        list of (R_i,) log-prob tensors, one per sequence across all groups.
    """
    from verl.models.transformers.monkey_patch import _make_dualkv_flash_wrapper
    from transformers.integrations import flash_attention

    orig_fn = flash_attention._flash_attention_forward
    flash_attention._flash_attention_forward = _make_dualkv_flash_wrapper(orig_fn)

    try:
        packed_ids_parts = []
        packed_pos_parts = []
        group_info_list = []
        packed_offset = 0

        for prompt_ids, responses in prompt_groups:
            P = prompt_ids.shape[0]
            response_lens = [r.shape[0] for r in responses]
            N = len(responses)

            packed_ids_parts.append(prompt_ids)
            packed_pos_parts.append(torch.arange(P, device=device))
            prompt_start = packed_offset
            packed_offset += P

            cu_dec_list = [0]
            dec_start = packed_offset
            for resp in responses:
                R_i = resp.shape[0]
                packed_ids_parts.append(resp)
                packed_pos_parts.append(torch.arange(P, P + R_i, device=device))
                cu_dec_list.append(cu_dec_list[-1] + R_i)
                packed_offset += R_i

            group_info_list.append({
                "prompt_start": prompt_start,
                "prompt_len": P,
                "dec_start": dec_start,
                "dec_end": packed_offset,
                "cu_seqlens_dec": torch.tensor(cu_dec_list, device=device, dtype=torch.int32),
                "max_decoded": max(response_lens),
                "n_seqs": N,
                "response_lens": response_lens,
            })

        input_ids = torch.cat(packed_ids_parts).unsqueeze(0)
        position_ids = torch.cat(packed_pos_parts).unsqueeze(0)
        dualkv_context = {"group_info": group_info_list}

        with torch.no_grad():
            logits = model(
                input_ids=input_ids,
                attention_mask=None,
                position_ids=position_ids,
                use_cache=False,
                dualkv_context=dualkv_context,
            ).logits.squeeze(0)

        results = []
        for g in group_info_list:
            P = g["prompt_len"]
            ps = g["prompt_start"]
            shared_lsm = F.log_softmax(logits[ps + P - 1].float(), dim=-1)

            dec_offset = g["dec_start"]
            for i in range(g["n_seqs"]):
                R_i = g["response_lens"][i]
                resp_ids = packed_ids_parts[0]  # placeholder, get from input
                lp = torch.zeros(R_i, device=device)

                # Get the actual response token IDs from the packed input
                resp_token_ids = input_ids.squeeze(0)[dec_offset : dec_offset + R_i]

                lp[0] = shared_lsm[resp_token_ids[0]]
                if R_i > 1:
                    dec_logits = logits[dec_offset : dec_offset + R_i - 1]
                    lp[1:] = logprobs_from_logits(dec_logits, resp_token_ids[1:])

                results.append(lp)
                dec_offset += R_i

        return results

    finally:
        flash_attention._flash_attention_forward = orig_fn


def dualkv_via_repack_logprobs(model, prompt_groups, device):
    """Run multiple prompt groups through DualKV using _dualkv_repack (production path).

    This is the path that has the bug — _dualkv_repack uses a single P for all groups.
    """
    from verl.models.transformers.monkey_patch import _make_dualkv_flash_wrapper
    from transformers.integrations import flash_attention

    # Import _dualkv_repack — need to handle the heavy import chain
    from verl.workers.actor.dp_actor import _dualkv_repack, _dualkv_extract_logprobs

    orig_fn = flash_attention._flash_attention_forward
    flash_attention._flash_attention_forward = _make_dualkv_flash_wrapper(orig_fn)

    try:
        # Build unpadded input as if from unpad_input: [P_aR_a1, P_aR_a2, P_bR_b1, P_bR_b2, ...]
        all_ids = []
        all_pos = []
        cu = [0]
        prompt_group_sizes = []
        prompt_lens = []

        for prompt_ids, responses in prompt_groups:
            P = prompt_ids.shape[0]
            prompt_lens.append(P)
            prompt_group_sizes.append(len(responses))
            for resp in responses:
                R = resp.shape[0]
                all_ids.append(torch.cat([prompt_ids, resp]))
                all_pos.append(torch.arange(P + R, device=device))
                cu.append(cu[-1] + P + R)

        input_ids_rmpad = torch.cat(all_ids).unsqueeze(0)
        position_ids_rmpad = torch.cat(all_pos).unsqueeze(0)
        cu_seqlens = torch.tensor(cu, dtype=torch.int32, device=device)

        ids_packed, pos_packed, dualkv_ctx, repack_info = _dualkv_repack(
            input_ids_rmpad, cu_seqlens, position_ids_rmpad,
            prompt_lens, prompt_group_sizes,
        )

        with torch.no_grad():
            logits = model(
                input_ids=ids_packed,
                attention_mask=None,
                position_ids=pos_packed,
                use_cache=False,
                dualkv_context=dualkv_ctx,
            ).logits.squeeze(0)

        total_seqs = sum(len(resps) for _, resps in prompt_groups)
        max_resp_len = max(r.shape[0] for _, resps in prompt_groups for r in resps)

        log_probs, _, _ = _dualkv_extract_logprobs(
            logits, repack_info, max_resp_len, total_seqs,
            temperature=1.0, calculate_entropy=False, compute_entropy_fn=None,
        )

        # Convert (total_seqs, max_resp_len) back to list of per-sequence log-probs
        results = []
        seq_idx = 0
        for prompt_ids, responses in prompt_groups:
            for resp in responses:
                R = resp.shape[0]
                results.append(log_probs[seq_idx, :R])
                seq_idx += 1

        return results

    finally:
        flash_attention._flash_attention_forward = orig_fn


@pytest.mark.skipif(not torch.cuda.is_available(), reason="Requires CUDA")
class TestDualkvMultiGroupParity:

    @pytest.fixture(scope="class")
    def model_and_config(self):
        from transformers import AutoModelForCausalLM, AutoConfig
        model_name = "Qwen/Qwen3-8B"
        config = AutoConfig.from_pretrained(model_name)
        model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=torch.float16,
            attn_implementation="flash_attention_2",
        ).to("cuda")
        model.eval()
        return model, config

    def _make_test_data(self, config, prompt_lens, response_lens_per_group, seed=42):
        """Generate random test data for multiple prompt groups.

        Args:
            prompt_lens: list of int — prompt length per group
            response_lens_per_group: list of list of int — response lengths per group
        """
        device = "cuda"
        vocab_size = config.vocab_size
        torch.manual_seed(seed)

        prompt_groups = []
        for P, resp_lens in zip(prompt_lens, response_lens_per_group):
            prompt_ids = torch.randint(1, vocab_size, (P,), device=device)
            responses = [torch.randint(1, vocab_size, (R,), device=device) for R in resp_lens]
            prompt_groups.append((prompt_ids, responses))

        return prompt_groups

    def test_single_group_baseline(self, model_and_config):
        """Single group — should always pass (no multi-group bug)."""
        model, config = model_and_config
        prompt_groups = self._make_test_data(
            config,
            prompt_lens=[48],
            response_lens_per_group=[[32, 24]],
        )

        std_lps = []
        for prompt_ids, responses in prompt_groups:
            for resp in responses:
                std_lps.append(standard_fa_logprobs(model, prompt_ids, resp, "cuda"))

        dk_lps = dualkv_multi_group_logprobs(model, prompt_groups, "cuda")

        for i, (s, d) in enumerate(zip(std_lps, dk_lps)):
            assert torch.allclose(s.float(), d.float(), atol=1e-3, rtol=1e-3), (
                f"Seq {i}: max_diff={(s.float() - d.float()).abs().max():.2e}"
            )

    def test_two_groups_same_prompt_len(self, model_and_config):
        """Two groups with same P — should pass (bug not triggered)."""
        model, config = model_and_config
        prompt_groups = self._make_test_data(
            config,
            prompt_lens=[48, 48],
            response_lens_per_group=[[32, 24], [28, 20]],
        )

        std_lps = []
        for prompt_ids, responses in prompt_groups:
            for resp in responses:
                std_lps.append(standard_fa_logprobs(model, prompt_ids, resp, "cuda"))

        dk_lps = dualkv_multi_group_logprobs(model, prompt_groups, "cuda")

        for i, (s, d) in enumerate(zip(std_lps, dk_lps)):
            assert torch.allclose(s, d, atol=1e-3, rtol=1e-3), (
                f"Seq {i}: max_diff={(s - d).abs().max():.2e}"
            )

    def test_two_groups_different_prompt_len_manual_packing(self, model_and_config):
        """Two groups with different P — manually packed (correct).

        This tests the DualKV kernel + wrapper with correct group_info.
        Should pass — proves the kernel handles multi-group fine when
        given correct metadata.
        """
        model, config = model_and_config
        prompt_groups = self._make_test_data(
            config,
            prompt_lens=[48, 32],
            response_lens_per_group=[[32, 24], [28, 20]],
        )

        std_lps = []
        for prompt_ids, responses in prompt_groups:
            for resp in responses:
                std_lps.append(standard_fa_logprobs(model, prompt_ids, resp, "cuda"))

        dk_lps = dualkv_multi_group_logprobs(model, prompt_groups, "cuda")

        for i, (s, d) in enumerate(zip(std_lps, dk_lps)):
            assert torch.allclose(s, d, atol=1e-3, rtol=1e-3), (
                f"Seq {i}: max_diff={(s - d).abs().max():.2e}"
            )

    def test_two_groups_different_prompt_len_via_repack(self, model_and_config):
        """Two groups with different P — through _dualkv_repack (production path).

        This is the test that exposes the bug. _dualkv_repack uses P=48
        for both groups, but group B has P=32.

        Expected: FAIL before fix, PASS after.
        """
        model, config = model_and_config
        prompt_groups = self._make_test_data(
            config,
            prompt_lens=[48, 32],
            response_lens_per_group=[[32, 24], [28, 20]],
        )

        std_lps = []
        for prompt_ids, responses in prompt_groups:
            for resp in responses:
                std_lps.append(standard_fa_logprobs(model, prompt_ids, resp, "cuda"))

        try:
            dk_lps = dualkv_via_repack_logprobs(model, prompt_groups, "cuda")
        except RuntimeError as e:
            pytest.fail(f"_dualkv_repack crashed due to wrong P: {e}")

        for i, (s, d) in enumerate(zip(std_lps, dk_lps)):
            assert torch.allclose(s.float(), d.float(), atol=1e-3, rtol=1e-3), (
                f"Seq {i}: max_diff={(s.float() - d.float()).abs().max():.2e}"
            )

    def test_three_groups_varied_lengths_via_repack(self, model_and_config):
        """Three groups with P=64, P=32, P=48 — through _dualkv_repack.

        Expected: FAIL before fix, PASS after.
        """
        model, config = model_and_config
        prompt_groups = self._make_test_data(
            config,
            prompt_lens=[64, 32, 48],
            response_lens_per_group=[[24, 32], [16], [20, 28]],
        )

        std_lps = []
        for prompt_ids, responses in prompt_groups:
            for resp in responses:
                std_lps.append(standard_fa_logprobs(model, prompt_ids, resp, "cuda"))

        try:
            dk_lps = dualkv_via_repack_logprobs(model, prompt_groups, "cuda")
        except RuntimeError as e:
            pytest.fail(f"_dualkv_repack crashed due to wrong P: {e}")

        for i, (s, d) in enumerate(zip(std_lps, dk_lps)):
            assert torch.allclose(s.float(), d.float(), atol=1e-3, rtol=1e-3), (
                f"Seq {i}: max_diff={(s.float() - d.float()).abs().max():.2e}"
            )


    def test_many_groups_stress(self, model_and_config):
        """Stress: 8 groups with varied prompt lengths and rollout counts.

        Uses relaxed tolerance (atol=2e-2) — with 8 groups and 16 sequences
        in fp16, accumulated numerical divergence between independent forward
        passes (standard) vs single packed forward (DualKV) is expected.
        """
        model, config = model_and_config
        prompt_groups = self._make_test_data(
            config,
            prompt_lens=[16, 64, 32, 48, 24, 56, 40, 20],
            response_lens_per_group=[
                [16, 24],       # 2 rollouts
                [32],           # 1 rollout
                [20, 16, 28],   # 3 rollouts
                [24, 32],       # 2 rollouts
                [16],           # 1 rollout
                [20, 24],       # 2 rollouts
                [28, 16, 20],   # 3 rollouts
                [32, 24],       # 2 rollouts
            ],
        )

        std_lps = []
        for prompt_ids, responses in prompt_groups:
            for resp in responses:
                std_lps.append(standard_fa_logprobs(model, prompt_ids, resp, "cuda"))

        dk_lps = dualkv_via_repack_logprobs(model, prompt_groups, "cuda")

        for i, (s, d) in enumerate(zip(std_lps, dk_lps)):
            assert torch.allclose(s.float(), d.float(), atol=1e-2, rtol=1e-3), (
                f"Seq {i}: max_diff={(s.float() - d.float()).abs().max():.2e}"
            )

    def test_very_short_prompt(self, model_and_config):
        """Edge case: prompt length = 1 token."""
        model, config = model_and_config
        prompt_groups = self._make_test_data(
            config,
            prompt_lens=[1, 48],
            response_lens_per_group=[[32, 24], [16, 20]],
        )

        std_lps = []
        for prompt_ids, responses in prompt_groups:
            for resp in responses:
                std_lps.append(standard_fa_logprobs(model, prompt_ids, resp, "cuda"))

        dk_lps = dualkv_via_repack_logprobs(model, prompt_groups, "cuda")

        for i, (s, d) in enumerate(zip(std_lps, dk_lps)):
            assert torch.allclose(s.float(), d.float(), atol=1e-3, rtol=1e-3), (
                f"Seq {i}: max_diff={(s.float() - d.float()).abs().max():.2e}"
            )

    def test_very_short_response(self, model_and_config):
        """Edge case: response length = 1 token."""
        model, config = model_and_config
        prompt_groups = self._make_test_data(
            config,
            prompt_lens=[32, 48],
            response_lens_per_group=[[1, 1], [1, 24]],
        )

        std_lps = []
        for prompt_ids, responses in prompt_groups:
            for resp in responses:
                std_lps.append(standard_fa_logprobs(model, prompt_ids, resp, "cuda"))

        dk_lps = dualkv_via_repack_logprobs(model, prompt_groups, "cuda")

        for i, (s, d) in enumerate(zip(std_lps, dk_lps)):
            assert torch.allclose(s.float(), d.float(), atol=1e-3, rtol=1e-3), (
                f"Seq {i}: max_diff={(s.float() - d.float()).abs().max():.2e}"
            )

    def test_large_prompt_length_ratio(self, model_and_config):
        """Edge case: prompts with 10x length difference."""
        model, config = model_and_config
        prompt_groups = self._make_test_data(
            config,
            prompt_lens=[8, 80],
            response_lens_per_group=[[16, 24], [16, 24]],
        )

        std_lps = []
        for prompt_ids, responses in prompt_groups:
            for resp in responses:
                std_lps.append(standard_fa_logprobs(model, prompt_ids, resp, "cuda"))

        dk_lps = dualkv_via_repack_logprobs(model, prompt_groups, "cuda")

        for i, (s, d) in enumerate(zip(std_lps, dk_lps)):
            assert torch.allclose(s.float(), d.float(), atol=1e-3, rtol=1e-3), (
                f"Seq {i}: max_diff={(s.float() - d.float()).abs().max():.2e}"
            )

    def test_mixed_response_lengths_within_group(self, model_and_config):
        """Edge case: responses within same group have very different lengths."""
        model, config = model_and_config
        prompt_groups = self._make_test_data(
            config,
            prompt_lens=[32, 48],
            response_lens_per_group=[[4, 64, 8], [2, 48]],
        )

        std_lps = []
        for prompt_ids, responses in prompt_groups:
            for resp in responses:
                std_lps.append(standard_fa_logprobs(model, prompt_ids, resp, "cuda"))

        dk_lps = dualkv_via_repack_logprobs(model, prompt_groups, "cuda")

        for i, (s, d) in enumerate(zip(std_lps, dk_lps)):
            assert torch.allclose(s.float(), d.float(), atol=1e-3, rtol=1e-3), (
                f"Seq {i}: max_diff={(s.float() - d.float()).abs().max():.2e}"
            )

    def test_single_rollout_per_group(self, model_and_config):
        """Edge case: each group has exactly 1 rollout (n_rollouts=1)."""
        model, config = model_and_config
        prompt_groups = self._make_test_data(
            config,
            prompt_lens=[24, 48, 16, 64],
            response_lens_per_group=[[32], [16], [24], [20]],
        )

        std_lps = []
        for prompt_ids, responses in prompt_groups:
            for resp in responses:
                std_lps.append(standard_fa_logprobs(model, prompt_ids, resp, "cuda"))

        dk_lps = dualkv_via_repack_logprobs(model, prompt_groups, "cuda")

        for i, (s, d) in enumerate(zip(std_lps, dk_lps)):
            assert torch.allclose(s.float(), d.float(), atol=1e-3, rtol=1e-3), (
                f"Seq {i}: max_diff={(s.float() - d.float()).abs().max():.2e}"
            )

    def test_long_prompt_short_response(self, model_and_config):
        """Edge case: prompt >> response (common in instruction-following)."""
        model, config = model_and_config
        prompt_groups = self._make_test_data(
            config,
            prompt_lens=[128, 64],
            response_lens_per_group=[[4, 8], [2, 4]],
        )

        std_lps = []
        for prompt_ids, responses in prompt_groups:
            for resp in responses:
                std_lps.append(standard_fa_logprobs(model, prompt_ids, resp, "cuda"))

        dk_lps = dualkv_via_repack_logprobs(model, prompt_groups, "cuda")

        for i, (s, d) in enumerate(zip(std_lps, dk_lps)):
            assert torch.allclose(s.float(), d.float(), atol=1e-3, rtol=1e-3), (
                f"Seq {i}: max_diff={(s.float() - d.float()).abs().max():.2e}"
            )

    def test_identical_prompt_content_different_lengths(self, model_and_config):
        """Two groups where shorter prompt is a prefix of longer prompt."""
        model, config = model_and_config
        device = "cuda"
        vocab_size = config.vocab_size
        torch.manual_seed(99)

        shared_tokens = torch.randint(1, vocab_size, (64,), device=device)
        prompt_a = shared_tokens[:32]
        prompt_b = shared_tokens[:48]
        resp_a = [torch.randint(1, vocab_size, (20,), device=device)]
        resp_b = [torch.randint(1, vocab_size, (24,), device=device)]
        prompt_groups = [(prompt_a, resp_a), (prompt_b, resp_b)]

        std_lps = []
        for prompt_ids, responses in prompt_groups:
            for resp in responses:
                std_lps.append(standard_fa_logprobs(model, prompt_ids, resp, device))

        dk_lps = dualkv_via_repack_logprobs(model, prompt_groups, device)

        for i, (s, d) in enumerate(zip(std_lps, dk_lps)):
            assert torch.allclose(s.float(), d.float(), atol=1e-3, rtol=1e-3), (
                f"Seq {i}: max_diff={(s.float() - d.float()).abs().max():.2e}"
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
