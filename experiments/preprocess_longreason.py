"""
Preprocess LongReason 8k split to veRL parquet format (60/40 train/test).

Downloads the 794 multiple-choice questions from lz1bytedance/LongReason,
formats prompts as chat messages, and stores the answer letter as ground truth.

Usage:
    python preprocess_longreason.py --local_save_dir $WORKDIR/data/longreason
"""

import argparse
import os

import datasets
import numpy as np
from datasets import load_dataset


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess LongReason 8k for veRL GRPO training"
    )
    parser.add_argument(
        "--local_save_dir",
        default="./data/longreason",
    )
    parser.add_argument("--split", default="8k")
    parser.add_argument("--train_ratio", type=float, default=0.6)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    print(f"Loading LongReason split={args.split}...")
    ds = load_dataset("lz1bytedance/LongReason", split=args.split)
    print(f"Loaded {len(ds)} examples")

    rng = np.random.RandomState(args.seed)
    indices = rng.permutation(len(ds))
    n_train = int(len(ds) * args.train_ratio)
    train_indices = set(indices[:n_train])

    train_rows = []
    val_rows = []

    for i, row in enumerate(ds):
        verl_row = {
            "data_source": "longreason",
            "prompt": [
                {"role": "user", "content": row["prompt"]},
            ],
            "ability": "reasoning",
            "reward_model": {
                "style": "rule",
                "ground_truth": row["answer"].strip(),
            },
            "extra_info": {
                "example_idx": row["example_idx"],
                "index": i,
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
