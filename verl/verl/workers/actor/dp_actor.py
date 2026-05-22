# Copyright 2024 Bytedance Ltd. and/or its affiliates
# Copyright 2023-2024 SGLang Team
# Copyright 2025 ModelBest Inc. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Single Process Actor
"""

import logging
import os

import torch
from torch import nn
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.tensor import DTensor

import verl.utils.torch_functional as verl_F
from verl import DataProto
from verl.trainer.ppo.core_algos import agg_loss, get_policy_loss_fn, kl_penalty
from verl.utils.attention_utils import index_first_axis, pad_input, rearrange, unpad_input
from verl.utils.device import get_device_id, get_device_name
from verl.utils.fsdp_utils import FSDPModule, fsdp2_clip_grad_norm_
from verl.utils.profiler import GPUMemoryLogger
from verl.utils.py_functional import append_to_dict
from verl.utils.seqlen_balancing import prepare_dynamic_batch, restore_dynamic_batch
from verl.utils.torch_dtypes import PrecisionType
from verl.utils.torch_functional import logprobs_from_logits
from verl.utils.ulysses import gather_outputs_and_unpad, ulysses_pad, ulysses_pad_and_slice_inputs
from verl.workers.actor import BasePPOActor
from verl.workers.config import ActorConfig

__all__ = ["DataParallelPPOActor"]


def _compute_prompt_group_sizes(uids, batch_size):
    """Compute consecutive prompt group sizes from uid array.

    Returns a list of ints, e.g. [4, 4, 4] for 3 prompt groups of 4 rollouts.
    If uids is None, assumes the entire batch is one group.
    """
    if uids is None:
        return [batch_size]
    groups = []
    count = 1
    for i in range(1, len(uids)):
        if uids[i] == uids[i - 1]:
            count += 1
        else:
            groups.append(count)
            count = 1
    groups.append(count)
    return groups


def _dualkv_repack(input_ids_rmpad, cu_seqlens, position_ids_rmpad, prompt_lens, prompt_group_sizes):
    """Repack unpadded tokens from [P₁R₁,...,P₈R₈] to [P, R₁,...,R₈] per group.

    Prompt tokens appear once per group instead of N times. All response segments
    within a group are contiguous, enabling zero-copy slicing in the attention layer.

    Args:
        input_ids_rmpad: (1, total_nnz) — concatenated unpadded tokens
        cu_seqlens: (bs+1,) — cumulative sequence lengths from unpad_input
        position_ids_rmpad: (1, total_nnz) — position ids for unpadded tokens
        prompt_lens: list[int] — per-group prompt lengths (actual tokens, not padded)
        prompt_group_sizes: list[int] — number of sequences per prompt group

    Returns:
        input_ids_packed: (1, total_packed)
        position_ids_packed: (1, total_packed)
        dualkv_context: dict with group_info for DualKV attention
        repack_info: dict for extracting per-sequence log-probs
    """
    device = input_ids_rmpad.device
    ids_flat = input_ids_rmpad.squeeze(0)  # (total_nnz,)

    packed_ids_parts = []
    packed_pos_parts = []
    group_info = []
    # Per-sequence: (packed_offset_of_resp_start, resp_len)
    response_slices = []
    first_response_tokens = []

    seq_idx = 0
    packed_offset = 0

    for group_idx, g_size in enumerate(prompt_group_sizes):
        P = prompt_lens[group_idx]
        g_start = cu_seqlens[seq_idx].item()

        # Prompt tokens from first sequence in group
        prompt_ids = ids_flat[g_start : g_start + P]
        prompt_pos = torch.arange(P, device=device, dtype=torch.long)
        packed_ids_parts.append(prompt_ids)
        packed_pos_parts.append(prompt_pos)

        prompt_start = packed_offset
        packed_offset += P

        # Response tokens for each sequence
        response_lens = []
        cu_dec_list = [0]
        dec_start = packed_offset

        for i in range(g_size):
            s = cu_seqlens[seq_idx + i].item()
            e = cu_seqlens[seq_idx + i + 1].item()
            R_i = e - s - P

            resp_ids = ids_flat[s + P : e]
            resp_pos = torch.arange(P, P + R_i, device=device, dtype=torch.long)

            packed_ids_parts.append(resp_ids)
            packed_pos_parts.append(resp_pos)

            first_response_tokens.append(resp_ids[0].item() if R_i > 0 else 0)
            response_slices.append((packed_offset, packed_offset + R_i))

            response_lens.append(R_i)
            cu_dec_list.append(cu_dec_list[-1] + R_i)
            packed_offset += R_i

        dec_end = packed_offset
        cu_dec = torch.tensor(cu_dec_list, device=device, dtype=torch.int32)
        max_decoded = max(response_lens) if response_lens else 0

        group_info.append({
            "prompt_start": prompt_start,
            "prompt_len": P,
            "dec_start": dec_start,
            "dec_end": dec_end,
            "cu_seqlens_dec": cu_dec,
            "max_decoded": max_decoded,
            "n_seqs": g_size,
            "response_lens": response_lens,
        })

        seq_idx += g_size

    input_ids_packed = torch.cat(packed_ids_parts).unsqueeze(0)  # (1, total_packed)
    position_ids_packed = torch.cat(packed_pos_parts).unsqueeze(0)  # (1, total_packed)

    dualkv_context = {"group_info": group_info}
    repack_info = {
        "group_info": group_info,
        "response_slices": response_slices,
        "first_response_tokens": first_response_tokens,
        "input_ids_packed": input_ids_packed,
    }

    return input_ids_packed, position_ids_packed, dualkv_context, repack_info


def _dualkv_extract_logprobs(logits_packed, repack_info, response_length, batch_size,
                             temperature, calculate_entropy, compute_entropy_fn):
    """Extract per-sequence log-probs from packed logits.

    Handles the shared first-token logit: logits[P-1] predicts the first
    response token for all sequences in a group.

    Args:
        logits_packed: (total_packed, vocab_size) — model output logits
        repack_info: dict from _dualkv_repack
        response_length: int — fixed response length for output tensor
        batch_size: int — number of sequences
        temperature: float
        calculate_entropy: bool
        compute_entropy_fn: callable for entropy computation

    Returns:
        log_probs: (batch_size, response_length)
        entropy: (batch_size, response_length) or None
        nan_mask: (batch_size, response_length)
    """
    device = logits_packed.device
    dtype = logits_packed.dtype
    group_info = repack_info["group_info"]
    response_slices = repack_info["response_slices"]
    first_response_tokens = repack_info["first_response_tokens"]
    input_ids_packed = repack_info["input_ids_packed"].squeeze(0)  # (total_packed,)

    logits_packed = logits_packed / temperature

    # Detect NaN tokens
    nan_mask_flat = torch.isnan(logits_packed).any(dim=-1)  # (total_packed,)
    logits_packed = torch.nan_to_num(logits_packed, nan=0.0, posinf=1e4, neginf=-1e4)

    log_probs = torch.zeros(batch_size, response_length, device=device, dtype=dtype)
    nan_mask = torch.zeros(batch_size, response_length, device=device, dtype=torch.bool)
    entropy = torch.zeros(batch_size, response_length, device=device, dtype=torch.float32) if calculate_entropy else None

    # Compute standard shifted logprobs on the packed tensor.
    # rolled_labels[t] = input_ids[t+1] — correct within segments, wrong at boundaries.
    rolled_labels = torch.roll(input_ids_packed, shifts=-1, dims=0)
    all_logprobs = logprobs_from_logits(logits_packed, rolled_labels, inplace_backward=False)

    if calculate_entropy:
        all_entropy = compute_entropy_fn(logits_packed)

    seq_idx = 0
    for g in group_info:
        P = g["prompt_len"]
        ps = g["prompt_start"]

        # Shared logit: logits[prompt_last] predicts first response token for ALL seqs
        shared_logit_pos = ps + P - 1
        shared_lsm = torch.log_softmax(logits_packed[shared_logit_pos].float(), dim=-1)
        shared_nan = nan_mask_flat[shared_logit_pos]

        if calculate_entropy:
            shared_entropy = compute_entropy_fn(logits_packed[shared_logit_pos].unsqueeze(0)).squeeze(0)

        for i in range(g["n_seqs"]):
            R_i = g["response_lens"][i]
            ds, de = response_slices[seq_idx]
            first_tok = first_response_tokens[seq_idx]

            # Clamp to response_length (response may be shorter)
            R_out = min(R_i, response_length)

            # First response token: from shared logit
            if R_out > 0:
                log_probs[seq_idx, 0] = shared_lsm[first_tok]
                nan_mask[seq_idx, 0] = shared_nan
                if calculate_entropy and entropy is not None:
                    entropy[seq_idx, 0] = shared_entropy

            # Remaining tokens [1:R_out]: from standard packed logprobs
            # logprobs at positions [ds : ds+R_i-1] predict tokens [1:R_i]
            if R_out > 1:
                log_probs[seq_idx, 1:R_out] = all_logprobs[ds : ds + R_out - 1]
                nan_mask[seq_idx, 1:R_out] = nan_mask_flat[ds : ds + R_out - 1]
                if calculate_entropy and entropy is not None:
                    entropy[seq_idx, 1:R_out] = all_entropy[ds : ds + R_out - 1]

            seq_idx += 1

    return log_probs, entropy, nan_mask


def _dualkv_extract_logprobs_fused(all_log_probs, all_entropy, shared_logits,
                                   repack_info, response_length, batch_size,
                                   calculate_entropy, compute_entropy_fn):
    """Extract per-sequence log-probs from fused per-token log_probs + shared logits.

    Memory-efficient version for use with FusedLinearForPPO. Most tokens use
    the chunked per-token log_probs (never materializing full vocab logits).
    Only the shared prompt-last positions need full logits for multi-sequence lookup.
    """
    device = all_log_probs.device
    dtype = all_log_probs.dtype
    group_info = repack_info["group_info"]
    response_slices = repack_info["response_slices"]
    first_response_tokens = repack_info["first_response_tokens"]

    log_probs = torch.zeros(batch_size, response_length, device=device, dtype=dtype)
    nan_mask = torch.zeros(batch_size, response_length, device=device, dtype=torch.bool)
    entropy = torch.zeros(batch_size, response_length, device=device, dtype=torch.float32) if calculate_entropy else None

    nan_mask_flat = torch.isnan(all_log_probs)

    seq_idx = 0
    for group_idx, g in enumerate(group_info):
        shared_logit_vec = shared_logits[group_idx]
        shared_nan = torch.isnan(shared_logit_vec).any()
        shared_logit_vec = torch.nan_to_num(shared_logit_vec, nan=0.0, posinf=1e4, neginf=-1e4)
        shared_lsm = torch.log_softmax(shared_logit_vec, dim=-1)

        if calculate_entropy and compute_entropy_fn is not None:
            shared_entropy = compute_entropy_fn(shared_logit_vec.unsqueeze(0)).squeeze(0)

        for i in range(g["n_seqs"]):
            R_i = g["response_lens"][i]
            ds, de = response_slices[seq_idx]
            first_tok = first_response_tokens[seq_idx]
            R_out = min(R_i, response_length)

            if R_out > 0:
                log_probs[seq_idx, 0] = shared_lsm[first_tok]
                nan_mask[seq_idx, 0] = shared_nan
                if calculate_entropy and entropy is not None:
                    entropy[seq_idx, 0] = shared_entropy

            if R_out > 1:
                log_probs[seq_idx, 1:R_out] = all_log_probs[ds : ds + R_out - 1]
                nan_mask[seq_idx, 1:R_out] = nan_mask_flat[ds : ds + R_out - 1]
                if calculate_entropy and entropy is not None and all_entropy is not None:
                    entropy[seq_idx, 1:R_out] = all_entropy[ds : ds + R_out - 1]

            seq_idx += 1

    return log_probs, entropy, nan_mask


logger = logging.getLogger(__file__)
logger.setLevel(os.getenv("VERL_LOGGING_LEVEL", "WARN"))


class DataParallelPPOActor(BasePPOActor):
    """FSDP DataParallel PPO Actor or Ref worker

    Args:
        config (ActorConfig): Actor config
        actor_module (nn.Module): Actor or ref module
        actor_optimizer (torch.optim.Optimizer, optional): Actor optimizer. Defaults to None.
    """

    def __init__(self, config: ActorConfig, actor_module: nn.Module, actor_optimizer: torch.optim.Optimizer = None):
        """When optimizer is None, it is Reference Policy"""
        super().__init__(config)
        self.actor_module = actor_module
        self.actor_optimizer = actor_optimizer
        role = "Ref" if actor_optimizer is None else "Actor"

        self.use_remove_padding = self.config.get("use_remove_padding", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_remove_padding={self.use_remove_padding}")
        self.use_fused_kernels = self.config.get("use_fused_kernels", False)
        if torch.distributed.get_rank() == 0:
            print(f"{role} use_fused_kernels={self.use_fused_kernels}")

        self.ulysses_sequence_parallel_size = self.config.ulysses_sequence_parallel_size
        self.use_ulysses_sp = self.ulysses_sequence_parallel_size > 1

        if self.config.entropy_from_logits_with_chunking:
            entropy_from_logits = verl_F.entropy_from_logits_with_chunking
        else:
            entropy_from_logits = verl_F.entropy_from_logits

        self.compute_entropy_from_logits = (
            torch.compile(entropy_from_logits, dynamic=True)
            if self.config.get("use_torch_compile", True)  # use torch compile by default
            else entropy_from_logits
        )
        self.use_dualkv = self.config.get("use_dualkv", False)
        self.device_name = get_device_name()
        self.param_dtype = PrecisionType.to_dtype(self.config.fsdp_config.get("dtype", "bfloat16"))
        if self.param_dtype == torch.float16:
            from torch.distributed.fsdp.sharded_grad_scaler import ShardedGradScaler

            self.scaler = ShardedGradScaler(growth_interval=400)
        else:
            self.scaler = None

    def _forward_micro_batch(
        self, micro_batch, temperature, calculate_entropy=False
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Returns:
            entropy: # (bs, response_len)
            log_probs: # (bs, response_len)
            nan_mask: # (bs, response_len) bool — True where logits were NaN
        """
        response_length = micro_batch["responses"].size(-1)
        multi_modal_inputs = {}
        if "multi_modal_inputs" in micro_batch.keys():
            from verl.utils.model import extract_multi_modal_inputs

            multi_modal_inputs = extract_multi_modal_inputs(micro_batch["multi_modal_inputs"])

        with torch.autocast(device_type=self.device_name, dtype=self.param_dtype):
            input_ids = micro_batch["input_ids"]
            batch_size, seqlen = input_ids.shape
            attention_mask = micro_batch["attention_mask"]
            position_ids = micro_batch["position_ids"]
            entropy = None
            nan_mask = None
            if position_ids.dim() == 3:  # qwen2vl mrope
                position_ids = position_ids.transpose(0, 1)  # (bsz, 4, seqlen) -> (4, bsz, seqlen)

            if self.use_remove_padding:
                input_ids_rmpad, indices, cu_seqlens, *_ = unpad_input(
                    input_ids.unsqueeze(-1), attention_mask
                )  # input_ids_rmpad (total_nnz, ...)
                input_ids_rmpad = input_ids_rmpad.transpose(0, 1)  # (1, total_nnz)

                # unpad the position_ids to align the rotary
                if position_ids.dim() == 3:
                    position_ids_rmpad = (
                        index_first_axis(rearrange(position_ids, "c b s ... -> (b s) c ..."), indices)
                        .transpose(0, 1)
                        .unsqueeze(1)
                    )  # (4, bsz, seqlen) -> (4, 1, bsz * seqlen)
                else:
                    position_ids_rmpad = index_first_axis(
                        rearrange(position_ids.unsqueeze(-1), "b s ... -> (b s) ..."), indices
                    ).transpose(0, 1)

                is_mask_all_zero = attention_mask.sum() == 0
                if is_mask_all_zero:
                    input_ids_rmpad = torch.zeros(
                        (1, self.ulysses_sequence_parallel_size),
                        device=input_ids.device,
                        dtype=input_ids.dtype,
                    )
                    if position_ids.dim() == 3:
                        position_ids_rmpad = torch.zeros(
                            (position_ids.shape[0], 1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )
                    else:
                        position_ids_rmpad = torch.zeros(
                            (1, self.ulysses_sequence_parallel_size),
                            device=position_ids.device,
                            dtype=position_ids.dtype,
                        )

                if "image_bound" in multi_modal_inputs:
                    from verl.utils.dataset.vision_utils import process_multi_modal_inputs_for_minicpmo

                    multi_modal_inputs = process_multi_modal_inputs_for_minicpmo(
                        input_ids, attention_mask, position_ids, cu_seqlens, multi_modal_inputs
                    )

                # for compute the log_prob
                input_ids_rmpad_rolled = torch.roll(input_ids_rmpad, shifts=-1, dims=1)  # (1, total_nnz)

                # pad and slice the inputs if sp > 1
                if self.use_ulysses_sp:
                    is_vlm_model = hasattr(
                        getattr(self.actor_module, "module", self.actor_module).config, "vision_config"
                    )
                    if is_vlm_model:
                        # vlm model's inputs will be sliced after embedding
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    else:
                        input_ids_rmpad, position_ids_rmpad, pad_size = ulysses_pad_and_slice_inputs(
                            input_ids_rmpad,
                            position_ids_rmpad=position_ids_rmpad,
                            sp_size=self.ulysses_sequence_parallel_size,
                        )
                    input_ids_rmpad_rolled, _, _ = ulysses_pad_and_slice_inputs(
                        input_ids_rmpad_rolled,
                        position_ids_rmpad=None,
                        sp_size=self.ulysses_sequence_parallel_size,
                    )

                input_ids_rmpad_rolled = input_ids_rmpad_rolled.squeeze(0)  # ((total_nnz / sp) + pad)

                # only pass input_ids and position_ids to enable flash_attn_varlen
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                if self.use_dualkv and not is_mask_all_zero:
                    prompt_allocated = micro_batch["input_ids"].size(1) - micro_batch["responses"].size(1)
                    uids = micro_batch.get("uid", None) if hasattr(micro_batch, 'get') else None
                    if uids is None and hasattr(micro_batch, 'non_tensor_batch'):
                        uids = micro_batch.non_tensor_batch.get("uid", None)
                    prompt_group_sizes = _compute_prompt_group_sizes(uids, cu_seqlens.shape[0] - 1)

                    prompt_lens = []
                    seq_idx = 0
                    for g_size in prompt_group_sizes:
                        p_len = micro_batch["attention_mask"][seq_idx, :prompt_allocated].sum().item()
                        prompt_lens.append(int(p_len))
                        seq_idx += g_size

                    input_ids_packed, position_ids_packed, dualkv_ctx, repack_info = _dualkv_repack(
                        input_ids_rmpad, cu_seqlens, position_ids_rmpad,
                        prompt_lens, prompt_group_sizes,
                    )

                    if self.use_fused_kernels:
                        shared_positions = [
                            g["prompt_start"] + g["prompt_len"] - 1
                            for g in repack_info["group_info"]
                        ]

                        output = self.actor_module(
                            input_ids=input_ids_packed,
                            attention_mask=None,
                            position_ids=position_ids_packed,
                            **multi_modal_inputs,
                            use_cache=False,
                            **{
                                **extra_args,
                                "dualkv_context": dualkv_ctx,
                                "dualkv_shared_positions": shared_positions,
                            },
                        )

                        all_log_probs = output.log_probs.squeeze(0)
                        all_entropy = output.entropy.squeeze(0) if output.entropy is not None else None
                        shared_logits = output.shared_logits

                        log_probs, entropy, nan_mask = _dualkv_extract_logprobs_fused(
                            all_log_probs, all_entropy, shared_logits,
                            repack_info, response_length, batch_size,
                            calculate_entropy,
                            self.compute_entropy_from_logits if calculate_entropy else None,
                        )
                    else:
                        output = self.actor_module(
                            input_ids=input_ids_packed,
                            attention_mask=None,
                            position_ids=position_ids_packed,
                            **multi_modal_inputs,
                            use_cache=False,
                            **{**extra_args, "dualkv_context": dualkv_ctx},
                        )

                        logits_packed = output.logits.squeeze(0)
                        log_probs, entropy, nan_mask = _dualkv_extract_logprobs(
                            logits_packed, repack_info, response_length, batch_size,
                            temperature, calculate_entropy,
                            self.compute_entropy_from_logits if calculate_entropy else None,
                        )

                else:
                    # --- Standard forward path ---
                    output = self.actor_module(
                        input_ids=input_ids_rmpad,
                        attention_mask=None,
                        position_ids=position_ids_rmpad,
                        **multi_modal_inputs,
                        use_cache=False,
                        **extra_args,
                    )

                    if self.use_fused_kernels:
                        log_probs = output.log_probs.squeeze(0)  # (total_nnz,)
                        entropy_rmpad = output.entropy.squeeze(0)  # (total_nnz,)

                    else:
                        logits_rmpad = output.logits.squeeze(0)  # (total_nnz, vocab_size)
                        logits_rmpad.div_(temperature)
                        # Detect NaN tokens BEFORE cleaning — these will be masked from policy loss
                        nan_mask_rmpad = torch.isnan(logits_rmpad).any(dim=-1)  # (total_nnz,)
                        # Replace NaN/Inf logits to keep forward finite
                        logits_rmpad = torch.nan_to_num(logits_rmpad, nan=0.0, posinf=1e4, neginf=-1e4)

                        # if use_sp: ((total_nnz / sp) + pad) ; if not use_sp: (batch, seqlen)
                        inplace_backward = True
                        if calculate_entropy:
                            inplace_backward = False
                        log_probs = logprobs_from_logits(
                            logits=logits_rmpad,
                            labels=input_ids_rmpad_rolled,
                            inplace_backward=inplace_backward,
                        )

                        # compute entropy
                        if calculate_entropy:
                            if not self.config.entropy_checkpointing:
                                entropy_rmpad = self.compute_entropy_from_logits(logits_rmpad)  # ((total_nnz / sp) + pad)
                            else:
                                entropy_rmpad = torch.utils.checkpoint.checkpoint(
                                    self.compute_entropy_from_logits, logits_rmpad
                                )

                    # gather log_prob if sp > 1
                    if self.use_ulysses_sp:
                        # gather and unpad for the ulysses sp
                        log_probs = gather_outputs_and_unpad(
                            log_probs,
                            gather_dim=0,
                            unpad_dim=0,
                            padding_size=pad_size,
                        )
                        if calculate_entropy:
                            entropy_rmpad = gather_outputs_and_unpad(
                                entropy_rmpad,
                                gather_dim=0,
                                unpad_dim=0,
                                padding_size=pad_size,
                            )
                        if not self.use_fused_kernels:
                            nan_mask_rmpad = gather_outputs_and_unpad(
                                nan_mask_rmpad.float(),
                                gather_dim=0,
                                unpad_dim=0,
                                padding_size=pad_size,
                            ) > 0.5

                    if is_mask_all_zero:
                        log_probs = log_probs[:0]
                        if calculate_entropy:
                            entropy_rmpad = entropy_rmpad[:0]
                        if not self.use_fused_kernels:
                            nan_mask_rmpad = nan_mask_rmpad[:0]

                    # pad back to (bsz, seqlen)
                    if calculate_entropy:
                        full_entropy = pad_input(
                            hidden_states=entropy_rmpad.unsqueeze(-1),
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )
                    full_log_probs = pad_input(
                        hidden_states=log_probs.unsqueeze(-1),
                        indices=indices,
                        batch=batch_size,
                        seqlen=seqlen,
                    )
                    if not self.use_fused_kernels:
                        full_nan_mask = pad_input(
                            hidden_states=nan_mask_rmpad.unsqueeze(-1).float(),
                            indices=indices,
                            batch=batch_size,
                            seqlen=seqlen,
                        )

                    # only return response part:
                    if calculate_entropy:
                        entropy = full_entropy.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                    log_probs = full_log_probs.squeeze(-1)[:, -response_length - 1 : -1]  # (bsz, response_length)
                    if not self.use_fused_kernels:
                        nan_mask = full_nan_mask.squeeze(-1)[:, -response_length - 1 : -1] > 0.5  # (bsz, response_length)
                    else:
                        nan_mask = torch.zeros_like(log_probs, dtype=torch.bool)

            else:  # not using rmpad and no ulysses sp
                extra_args = {}
                if self.use_fused_kernels:
                    extra_args["temperature"] = temperature
                    extra_args["return_dict"] = True

                output = self.actor_module(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    position_ids=position_ids,
                    **multi_modal_inputs,
                    use_cache=False,
                    **extra_args,
                )  # prevent model thinks we are generating

                if self.use_fused_kernels:
                    log_probs = output.log_probs[:, -response_length - 1 : -1]
                    entropy = output.entropy[:, -response_length - 1 : -1]  # (bsz, response_length)

                else:
                    logits = output.logits

                    logits.div_(temperature)
                    logits = logits[:, -response_length - 1 : -1, :]  # (bsz, response_length, vocab_size)
                    # Detect NaN tokens BEFORE cleaning
                    nan_mask = torch.isnan(logits).any(dim=-1)  # (bsz, response_length)
                    logits = torch.nan_to_num(logits, nan=0.0, posinf=1e4, neginf=-1e4)
                    log_probs = logprobs_from_logits(logits, micro_batch["responses"])
                    if calculate_entropy:
                        if not self.config.entropy_checkpointing:
                            entropy = verl_F.entropy_from_logits(logits)  # (bsz, response_length)
                        else:
                            entropy = torch.utils.checkpoint.checkpoint(verl_F.entropy_from_logits, logits)

            if nan_mask is None:
                nan_mask = torch.zeros_like(log_probs, dtype=torch.bool)

            return entropy, log_probs, nan_mask

    def _optimizer_step(self):
        assert self.config.grad_clip is not None
        if self.scaler is not None:
            self.scaler.unscale_(self.actor_optimizer)
        if isinstance(self.actor_module, FSDP):
            grad_norm = self.actor_module.clip_grad_norm_(max_norm=self.config.grad_clip)
        elif isinstance(self.actor_module, FSDPModule):
            grad_norm = fsdp2_clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)
        else:
            grad_norm = torch.nn.utils.clip_grad_norm_(self.actor_module.parameters(), max_norm=self.config.grad_clip)

        if isinstance(grad_norm, DTensor):
            grad_norm = grad_norm.full_tensor()

        # if grad_norm is not finite, skip the update
        if self.scaler is not None:
            self.scaler.step(self.actor_optimizer)
            self.scaler.update()
        else:
            if not torch.isfinite(grad_norm):
                print(f"WARN: rank {torch.distributed.get_rank()} grad_norm is not finite: {grad_norm}")
                self.actor_optimizer.zero_grad()
            else:
                self.actor_optimizer.step()
        return grad_norm

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def compute_log_prob(self, data: DataProto, calculate_entropy=False) -> torch.Tensor:
        """Compute the log probability of the responses given input_ids, attention_mask and position_ids

        Args:
            data (DataProto): a DataProto containing keys

                ``input_ids``: tensor of shape [batch_size, sequence_length]. torch.int64. Note that input_ids is the
                concatenation of prompt and response. Note that ``sequence_length = prompt_length + response_length``.

                ``attention_mask``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``position_ids``: tensor of shape [batch_size, sequence_length]. torch.int64.

                ``responses``:  tensor of shape [batch_size, response_length]. torch.int64.

        Returns:
            torch.Tensor: the log_prob tensor
        """
        # set to eval
        self.actor_module.eval()

        micro_batch_size = data.meta_info["micro_batch_size"]
        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error
        use_dynamic_bsz = data.meta_info["use_dynamic_bsz"]
        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        select_keys = ["responses", "input_ids", "attention_mask", "position_ids"]
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        if use_dynamic_bsz:
            max_token_len = data.meta_info["max_token_len"] * self.ulysses_sequence_parallel_size
            micro_batches, batch_idx_list = prepare_dynamic_batch(data, max_token_len=max_token_len)
        else:
            micro_batches = data.split(micro_batch_size)

        log_probs_lst = []
        entropy_lst = []
        for micro_batch in micro_batches:
            micro_batch = micro_batch.to(get_device_id())

            model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
            with torch.no_grad():
                entropy, log_probs, _nan_mask = self._forward_micro_batch(
                    model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                )

            log_probs_lst.append(log_probs)
            if calculate_entropy:
                entropy_lst.append(entropy)

        log_probs = torch.concat(log_probs_lst, dim=0)
        entropys = None
        if calculate_entropy:
            entropys = torch.concat(entropy_lst, dim=0)

        if use_dynamic_bsz:
            log_probs = restore_dynamic_batch(log_probs, batch_idx_list)
            if calculate_entropy:
                entropys = restore_dynamic_batch(entropys, batch_idx_list)

        return log_probs, entropys

    @GPUMemoryLogger(role="dp actor", logger=logger)
    def update_policy(self, data: DataProto):
        # make sure we are in training mode
        self.actor_module.train()

        temperature = data.meta_info["temperature"]  # temperature must be in the data.meta_info to avoid silent error

        select_keys = [
            "responses",
            "response_mask",
            "input_ids",
            "attention_mask",
            "position_ids",
            "old_log_probs",
            "advantages",
        ]
        if self.config.use_kl_loss:
            select_keys.append("ref_log_prob")
        # Include pre-computed IS weights if present in batch
        # Weights are computed centrally in trainer and added to batch when algorithm.rollout_is=True
        if "rollout_is_weights" in data.batch.keys():
            select_keys.append("rollout_is_weights")
        # Include rollout_log_probs for computing rollout_corr metrics in bypass mode
        if "rollout_log_probs" in data.batch.keys():
            select_keys.append("rollout_log_probs")

        has_multi_modal_inputs = "multi_modal_inputs" in data.non_tensor_batch.keys()
        non_tensor_select_keys = ["multi_modal_inputs"] if has_multi_modal_inputs else []

        data = data.select(batch_keys=select_keys, non_tensor_batch_keys=non_tensor_select_keys)

        # Split to make minibatch iterator for updating the actor
        # See PPO paper for details. https://arxiv.org/abs/1707.06347
        mini_batches = data.split(self.config.ppo_mini_batch_size)

        on_policy = len(mini_batches) == 1 and self.config.ppo_epochs == 1

        metrics = {
            "actor/pg_loss": 0.0,
            "actor/kl_loss": 0.0,
        }
        for _ in range(self.config.ppo_epochs):
            for batch_idx, mini_batch in enumerate(mini_batches):
                if self.config.use_dynamic_bsz:
                    max_token_len = self.config.ppo_max_token_len_per_gpu * self.ulysses_sequence_parallel_size
                    micro_batches, _ = prepare_dynamic_batch(mini_batch, max_token_len=max_token_len)
                else:
                    self.gradient_accumulation = (
                        self.config.ppo_mini_batch_size // self.config.ppo_micro_batch_size_per_gpu
                    )
                    micro_batches = mini_batch.split(self.config.ppo_micro_batch_size_per_gpu)

                self.actor_optimizer.zero_grad()

                for micro_batch in micro_batches:
                    micro_batch = micro_batch.to(get_device_id())

                    micro_batch_metrics = {}
                    model_inputs = {**micro_batch.batch, **micro_batch.non_tensor_batch}
                    response_mask = model_inputs["response_mask"]
                    old_log_prob = model_inputs["old_log_probs"]
                    advantages = model_inputs["advantages"]

                    entropy_coeff = self.config.entropy_coeff
                    loss_agg_mode = self.config.loss_agg_mode

                    calculate_entropy = self.config.calculate_entropy or (entropy_coeff != 0)

                    if self.config.use_dynamic_bsz:
                        loss_scale_factor = response_mask.shape[0] / self.config.ppo_mini_batch_size
                    else:
                        loss_scale_factor = 1 / self.gradient_accumulation

                    # all return: (bsz, response_length)
                    entropy, log_prob, nan_mask = self._forward_micro_batch(
                        model_inputs, temperature=temperature, calculate_entropy=calculate_entropy
                    )

                    if nan_mask.any():
                        response_mask = response_mask * (~nan_mask).float()

                    # for fully_async_policy recipe
                    if hasattr(self.config, "use_rollout_log_probs") and self.config.use_rollout_log_probs:
                        old_log_prob = model_inputs["old_log_probs"]
                    else:
                        if on_policy:
                            old_log_prob = log_prob.detach()
                        else:
                            old_log_prob = model_inputs["old_log_probs"]

                    loss_mode = self.config.policy_loss.get("loss_mode", "vanilla")
                    # vanilla -> verl.trainer.ppo.core_algos.compute_policy_loss_vanilla

                    # Extract pre-computed rollout correction weights if present
                    # Weights are computed centrally in trainer and added when algorithm.rollout_is=True
                    rollout_is_weights = model_inputs.get("rollout_is_weights", None)

                    # gpg -> verl.trainer.ppo.core_algos.compute_policy_loss_gpg
                    # clip_cov -> verl.trainer.ppo.core_algos.compute_policy_loss_clip_cov
                    policy_loss_fn = get_policy_loss_fn(loss_mode)

                    # Compute policy loss (any function is expected to return 2 values)
                    pg_loss, pg_metrics = policy_loss_fn(
                        old_log_prob=old_log_prob,
                        log_prob=log_prob,
                        advantages=advantages,
                        response_mask=response_mask,
                        loss_agg_mode=loss_agg_mode,
                        config=self.config,
                        rollout_is_weights=rollout_is_weights,
                    )
                    micro_batch_metrics.update(pg_metrics)

                    # Skip if using bypass_mode loss (metrics already computed in pg_metrics)
                    rollout_log_prob = model_inputs.get("rollout_log_probs", None)
                    if loss_mode != "bypass_mode" and rollout_log_prob is not None:
                        # Compute metrics using CURRENT policy π_θ vs π_rollout
                        # Tracks evolving off-policy gap as π_θ updates during mini-batch training
                        from verl.trainer.ppo.rollout_corr_helper import compute_rollout_corr_metrics_from_logprobs

                        rollout_corr_metrics = compute_rollout_corr_metrics_from_logprobs(
                            log_prob=log_prob,
                            rollout_log_prob=rollout_log_prob,
                            response_mask=response_mask,
                        )
                        micro_batch_metrics.update(rollout_corr_metrics)

                    policy_loss = pg_loss
                    if calculate_entropy and entropy is not None:
                        entropy_agg = agg_loss(loss_mat=entropy, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)
                        micro_batch_metrics["actor/entropy"] = entropy_agg.detach().item()
                        if entropy_coeff != 0:
                            policy_loss -= entropy_agg * entropy_coeff

                    if self.config.use_kl_loss:
                        ref_log_prob = model_inputs["ref_log_prob"]
                        # compute kl loss
                        kld = kl_penalty(
                            logprob=log_prob, ref_logprob=ref_log_prob, kl_penalty=self.config.kl_loss_type
                        )
                        kl_loss = agg_loss(loss_mat=kld, loss_mask=response_mask, loss_agg_mode=loss_agg_mode)

                        policy_loss = policy_loss + kl_loss * self.config.kl_loss_coef
                        metrics["actor/kl_loss"] += kl_loss.detach().item() * loss_scale_factor
                        micro_batch_metrics["actor/kl_coef"] = self.config.kl_loss_coef

                    if self.config.use_dynamic_bsz:
                        # relative to the dynamic bsz
                        loss = policy_loss * loss_scale_factor
                    else:
                        loss = policy_loss * loss_scale_factor
                    if self.scaler is not None:
                        self.scaler.scale(loss).backward()
                    else:
                        loss.backward()

                    metrics["actor/pg_loss"] += pg_loss.detach().item() * loss_scale_factor
                    append_to_dict(metrics, micro_batch_metrics)

                grad_norm = self._optimizer_step()
                mini_batch_metrics = {"actor/grad_norm": grad_norm.detach().item()}
                append_to_dict(metrics, mini_batch_metrics)
        self.actor_optimizer.zero_grad()
        return metrics
