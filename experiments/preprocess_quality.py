"""
Preprocess QuALITY dataset to veRL parquet format (80/20 train/test).

Downloads 2,086 long-document multiple-choice questions from emozilla/quality,
formats prompts as article + question + options, and stores the answer index
mapped to a letter (A/B/C/D) as ground truth.

Usage:
    python preprocess_quality.py --local_save_dir $WORKDIR/data/quality
"""

import argparse
import os

import datasets
import numpy as np
from datasets import load_dataset


OPTION_LETTERS = ["A", "B", "C", "D"]


def format_prompt(article, question, options):
    options_str = "\n".join(
        f"({OPTION_LETTERS[i]}) {opt}" for i, opt in enumerate(options)
    )
    return (
        f"Read the following article carefully, then answer the multiple-choice "
        f"question below.\n\n"
        f"--- Article ---\n{article}\n\n"
        f"--- Question ---\n{question}\n\n"
        f"--- Options ---\n{options_str}\n\n"
        f"Answer with the letter of the correct option (A, B, C, or D)."
    )


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess QuALITY for veRL GRPO training"
    )
    parser.add_argument(
        "--local_save_dir",
        default="./data/quality",
    )
    parser.add_argument("--train_ratio", type=float, default=0.8)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print("Loading QuALITY validation split...")
    ds = load_dataset("emozilla/quality", split="validation")
    print(f"Loaded {len(ds)} examples")

    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(len(ds))
    n_train = int(len(ds) * args.train_ratio)
    train_indices = set(indices[:n_train].tolist())

    train_rows = []
    val_rows = []

    for i, row in enumerate(ds):
        prompt_text = format_prompt(row["article"], row["question"], row["options"])
        answer_letter = OPTION_LETTERS[row["answer"]]

        verl_row = {
            "data_source": "quality",
            "prompt": [
                {"role": "user", "content": prompt_text},
            ],
            "ability": "reading_comprehension",
            "reward_model": {
                "style": "rule",
                "ground_truth": answer_letter,
            },
            "extra_info": {
                "index": i,
                "hard": row["hard"],
            },
        }

        if i in train_indices:
            train_rows.append(verl_row)
        else:
            val_rows.append(verl_row)

    print(f"Train: {len(train_rows)}, Test: {len(val_rows)}")

    os.makedirs(args.local_save_dir, exist_ok=True)

    for split_name, rows in [("train", train_rows), ("test", val_rows)]:
        split_ds = datasets.Dataset.from_dict({
            k: [r[k] for r in rows] for k in rows[0]
        })
        out_path = os.path.join(args.local_save_dir, f"{split_name}.parquet")
        split_ds.to_parquet(out_path)
        print(f"Saved {len(rows)} examples to {out_path}")

    print(f"\nDone. {len(train_rows) + len(val_rows)} total saved to {args.local_save_dir}")


if __name__ == "__main__":
    main()
