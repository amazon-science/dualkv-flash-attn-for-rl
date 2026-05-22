"""
LongReason binary reward function for veRL GRPO training.

Extracts the answer letter (A-E) from the response text after </think>,
compares against the ground truth. Returns 1.0 for exact match, 0.0 otherwise.

Config:
    reward.custom_reward_function.path=<path>/reward_longreason.py
    reward.custom_reward_function.name=compute_score
"""


def _extract_answer(text):
    think_end = text.find("</think>")
    if think_end != -1:
        answer_part = text[think_end + 8:]
    else:
        answer_part = text
    for ch in answer_part:
        if ch in "ABCDE":
            return ch
    return ""


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
