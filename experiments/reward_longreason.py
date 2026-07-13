"""
LongReason binary reward function for veRL GRPO training.

The prompt instructs the model to end with the EXACT format:
    "Provide the final answer on the last line using 'The answer is' + option (A, B, C, D, E)."

STRICT extraction (no fallbacks): we only accept the answer when it is emitted in
that specified format ("The answer is X"). If the response does not follow the
instruction, the model is not doing what it was told, so it gets reward 0 — we do
NOT reward correct-but-unformatted answers (that would train the policy to ignore
the output-format instruction).

Note on Gemma4 reasoning mode: the model emits reasoning wrapped in
`<|channel>thought ... <channel|>` then the final answer. verl decodes the reward
response with skip_special_tokens=True (verl/workers/reward_manager/naive.py), so
those channel tokens are stripped before this function sees `solution_str`. The
"The answer is X" line survives stripping and is what we key on.

Config:
    reward.custom_reward_function.path=<path>/reward_longreason.py
    reward.custom_reward_function.name=compute_score
"""

import re

# The ONE accepted format, matching the prompt's instruction exactly:
# "The answer is X" (case-insensitive on the phrase, optional trailing punctuation).
# X must be a standalone A-E letter. Take the LAST such match (the model may restate;
# the final "The answer is X" is the committed answer).
_ANSWER_RE = re.compile(r"the\s+answer\s+is\s+([A-Ea-e])\b", re.IGNORECASE)


def _extract_answer(text):
    if not isinstance(text, str):
        return ""
    matches = _ANSWER_RE.findall(text)
    return matches[-1].upper() if matches else ""


def compute_score(
    data_source: str,
    solution_str: str,
    ground_truth,
    extra_info=None,
    **kwargs,
) -> float:
    if data_source != "longreason":
        from verl.utils.reward_score import default_compute_score
        return default_compute_score(
            data_source=data_source,
            solution_str=solution_str,
            ground_truth=ground_truth,
            extra_info=extra_info,
            **kwargs,
        )

    pred = _extract_answer(solution_str)
    gt = ground_truth.strip() if isinstance(ground_truth, str) else str(ground_truth).strip()
    return 1.0 if pred == gt else 0.0
