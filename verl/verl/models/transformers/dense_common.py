# Copyright 2024 Bytedance Ltd. and/or its affiliates
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

from dataclasses import dataclass
from typing import Optional, Union

import torch
from transformers.cache_utils import Cache
from transformers.modeling_outputs import CausalLMOutputWithPast


@dataclass
class CausalLMOutputForPPO(CausalLMOutputWithPast):
    log_probs: Optional[torch.FloatTensor] = None
    entropy: Optional[torch.FloatTensor] = None
    # DualKV: full logit vectors at the shared prompt-last positions (one row per group).
    # The fused kernel never materializes logits, but the first response token of each
    # response in a group needs the distribution at the shared position P-1 indexed by
    # ITS OWN token id. We therefore compute just those few rows explicitly here.
    # None unless `dualkv_shared_positions` is passed. Assumes SP=1 (see the assert at the
    # producer site); under SP>1 the packed hidden_states are sequence-sliced and a global
    # index would be wrong, so DualKV currently requires ulysses_sequence_parallel_size==1.
    shared_logits: Optional[torch.FloatTensor] = None


def _dualkv_shared_logits(hidden_states, lm_head_weight, dualkv_shared_positions, temperature):
    """Materialize the full logit vector at each group's shared prompt-last position.

    The fused LM-head kernels return only per-token log_probs; they never build logits.
    But DualKV packs a group's prompt once, so a response's FIRST token must read the
    distribution at the shared position P-1 indexed by that response's own token id
    (handled downstream in `_dualkv_extract_logprobs_fused`). We therefore project just
    those few rows (one per group) through the LM head here.

    Assumes SP=1: `hidden_states` is the full, un-sliced packed stream so the packed
    `dualkv_shared_positions` index it directly. The producer asserts SP==1 before this
    runs. Returns None when DualKV is not active (positions not passed).

    Args:
        hidden_states: (1, total_packed, hidden) or (total_packed, hidden)
        lm_head_weight: (vocab, hidden)
        dualkv_shared_positions: list[int] packed indices of shared prompt-last tokens
        temperature: float, applied to match the fused kernels' logit scaling

    Returns:
        (num_groups, vocab) float32 logits, or None.
    """
    if dualkv_shared_positions is None:
        return None
    hs = hidden_states.squeeze(0) if hidden_states.dim() == 3 else hidden_states  # (total_packed, hidden)
    pos = torch.as_tensor(dualkv_shared_positions, device=hs.device, dtype=torch.long)
    shared_hs = hs.index_select(0, pos)  # (num_groups, hidden)
    shared_logits = shared_hs.to(lm_head_weight.dtype) @ lm_head_weight.t()  # (num_groups, vocab)
    return (shared_logits / temperature).float()


def forward_base_model(
    self,
    input_ids: Optional[torch.LongTensor] = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Cache] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs,
) -> CausalLMOutputWithPast:
    r"""
    Copy paste LLaMa's forward
    https://github.com/linkedin/Liger-Kernel/blob/main/src/liger_kernel/transformers/model/llama.py

    This function should be generic enough for all pure text models.
    ```"""

    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        **kwargs,
    )

    return outputs


def forward_with_torch_backend(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union["Cache", list[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: int | torch.Tensor = 0,
    temperature: float = 1.0,
    shift_labels: Optional[torch.LongTensor] = None,
    **loss_kwargs,
) -> tuple | CausalLMOutputForPPO:
    from verl.utils.experimental.torch_functional import FusedLinearForPPO

    # DualKV: thread the shared-prompt context to the base model so it reaches the
    # attention monkey-patch. HF Gemma4ForCausalLM.forward filters unknown kwargs, so
    # without this re-injection dualkv_context is dropped at the model boundary and
    # DualKV silently never runs. (Ported from gemma4-dev dense_common.)
    dualkv_context = loss_kwargs.pop("dualkv_context", None)
    # DualKV: packed positions (one per group) of the shared prompt-last token; consumed
    # locally to build `shared_logits`, NOT forwarded to the base model.
    dualkv_shared_positions = loss_kwargs.pop("dualkv_shared_positions", None)

    model_kwargs = {}
    if dualkv_context is not None:
        model_kwargs["dualkv_context"] = dualkv_context

    outputs = forward_base_model(
        self,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        cache_position=cache_position,
        **model_kwargs,
    )

    hidden_states = outputs[0]

    if not return_dict:
        raise NotImplementedError("forward_with_torch_backend has to return_dict")

    # Loss calculations.
    # When the engine has already prepared globally-rolled labels (e.g. the FSDP
    # path under Ulysses SP, see issue #6068), it passes them as `shift_labels`
    # so we don't redo `torch.roll` on a sequence-parallel-sliced shard.
    if shift_labels is not None:
        rolled_labels = shift_labels
    elif labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("To use forward_with_torch_backend, either labels or input_ids must be provided.")

    # DualKV: build the shared prompt-last logit rows BEFORE the fused kernel (which never
    # materializes logits). None unless DualKV passed dualkv_shared_positions.
    shared_logits = _dualkv_shared_logits(hidden_states, self.lm_head.weight, dualkv_shared_positions, temperature)

    fused_linear_for_ppo = FusedLinearForPPO()
    log_probs, entropy = fused_linear_for_ppo.forward(
        hidden_states=hidden_states,
        vocab_weights=self.lm_head.weight,
        input_ids=rolled_labels,
        temperature=temperature,
    )

    return CausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        shared_logits=shared_logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )


def forward_with_triton_backend(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union["Cache", list[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    logits_to_keep: int | torch.Tensor = 0,
    temperature: float = 1.0,
    shift_labels: Optional[torch.LongTensor] = None,
    **loss_kwargs,
) -> tuple | CausalLMOutputForPPO:
    from verl.utils.kernel.linear_cross_entropy import linear_cross_entropy

    # DualKV: thread shared-prompt context to the base model (see torch backend above).
    dualkv_context = loss_kwargs.pop("dualkv_context", None)
    dualkv_shared_positions = loss_kwargs.pop("dualkv_shared_positions", None)

    model_kwargs = {}
    if dualkv_context is not None:
        model_kwargs["dualkv_context"] = dualkv_context

    outputs = forward_base_model(
        self,
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        **model_kwargs,
    )

    hidden_states = outputs[0]

    if not return_dict:
        raise NotImplementedError("forward_with_triton_backend has to return_dict")

    # Loss calculations. See `forward_with_torch_backend` for why `shift_labels`
    # takes precedence over local `torch.roll` (issue #6068).
    if shift_labels is not None:
        rolled_labels = shift_labels
    elif labels is not None:
        rolled_labels = torch.roll(labels, shifts=-1, dims=-1)
    elif input_ids is not None:
        rolled_labels = torch.roll(input_ids, shifts=-1, dims=-1)
    else:
        raise RuntimeError("To use forward_with_triton_backend, either labels or input_ids must be provided.")

    # DualKV: shared prompt-last logit rows (see torch backend). None unless DualKV active.
    shared_logits = _dualkv_shared_logits(hidden_states, self.lm_head.weight, dualkv_shared_positions, temperature)

    log_probs, entropy = linear_cross_entropy(
        hidden_states,
        self.lm_head.weight,
        rolled_labels,
        temperature,
        "none",
    )

    return CausalLMOutputForPPO(
        log_probs=log_probs,
        entropy=entropy,
        shared_logits=shared_logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
    )
