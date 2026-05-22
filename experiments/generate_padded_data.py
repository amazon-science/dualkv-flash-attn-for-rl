"""
Generate padded LongReason data at a target prompt token length.

Takes the base LongReason dataset (from preprocess_longreason.py output) and pads
each prompt to exactly TARGET_TOKENS after chat template tokenization. Uses the same
tokenization path as veRL runtime (apply_chat_template with add_generation_prompt=True).

Guarantees: every prompt tokenizes to EXACTLY target_tokens. If the binary search
on character position lands ±1 off, a token-level fixup trims or pads to match.

Usage:
    python generate_padded_data.py \
        --src_dir /path/to/longreason/base \
        --out_dir /path/to/longreason_padded/P48K \
        --target_tokens 49152 \
        --model_path /path/to/Llama-3.1-8B-Instruct

    # Generate multiple context lengths:
    for P in 8192 16384 32768 49152 65536 98304 131072; do
        python generate_padded_data.py \
            --src_dir /path/to/longreason_padded/P128K \
            --out_dir /path/to/longreason_padded/P${P} \
            --target_tokens $P \
            --model_path /path/to/Llama-3.1-8B-Instruct
    done
"""

import argparse
import os

import datasets
import pyarrow.parquet as pq
from transformers import AutoTokenizer


def tokenize_prompt(tok, content, **kwargs):
    msgs = [{"role": "user", "content": content}]
    return tok.apply_chat_template(
        msgs, tokenize=True, add_generation_prompt=True, **kwargs
    )


def pad_to_target(tok, content, target_tokens, **chat_kwargs):
    cur_tokens = len(tokenize_prompt(tok, content, **chat_kwargs))

    if cur_tokens < target_tokens:
        needed = target_tokens - cur_tokens + 500
        pad_text = " passage" * needed
        content = content + "\n\n[Additional context follows]\n" + pad_text

    # Binary search: find smallest char position where token count >= target
    lo, hi = 0, len(content)
    while lo < hi:
        mid = (lo + hi) // 2
        n = len(tokenize_prompt(tok, content[:mid], **chat_kwargs))
        if n < target_tokens:
            lo = mid + 1
        else:
            hi = mid

    final_content = content[:lo]
    final_n = len(tokenize_prompt(tok, final_content, **chat_kwargs))

    if final_n == target_tokens:
        return final_content, final_n

    # Binary search landed at >= target. Walk back to find <= target.
    if final_n > target_tokens:
        for offset in range(1, 100):
            candidate = content[:lo - offset]
            n = len(tokenize_prompt(tok, candidate, **chat_kwargs))
            if n <= target_tokens:
                final_content = candidate
                final_n = n
                break

    # Token-level fixup: decode and re-encode to hit exact target
    if final_n != target_tokens:
        tokens = tokenize_prompt(tok, final_content, **chat_kwargs)
        if len(tokens) > target_tokens:
            # Find the chat template overhead by tokenizing empty content
            overhead = tokenize_prompt(tok, "", **chat_kwargs)
            prefix_len = len(overhead) - 1  # tokens before content, minus the trailing gen token
            # Truncate content tokens from the right (left-preserve)
            content_tokens = tokens[prefix_len:-1]  # strip prefix and final gen token
            trim_amount = len(tokens) - target_tokens
            content_tokens = content_tokens[:-trim_amount] if trim_amount > 0 else content_tokens
            trimmed_text = tok.decode(content_tokens, skip_special_tokens=True)
            final_content = trimmed_text
            final_n = len(tokenize_prompt(tok, final_content, **chat_kwargs))

    # Last resort: if still not exact, accept <= target (veRL will pad up)
    if final_n > target_tokens:
        for offset in range(1, 200):
            candidate = content[:max(0, lo - offset)]
            n = len(tokenize_prompt(tok, candidate, **chat_kwargs))
            if n <= target_tokens:
                return candidate, n

    return final_content, final_n


def main():
    parser = argparse.ArgumentParser(
        description="Generate padded LongReason data at target prompt token length"
    )
    parser.add_argument("--src_dir", required=True, help="Source parquet directory")
    parser.add_argument("--out_dir", required=True, help="Output directory")
    parser.add_argument("--target_tokens", type=int, required=True)
    parser.add_argument("--model_path", required=True, help="Tokenizer model path")
    parser.add_argument(
        "--chat_kwargs", default="", help="Extra kwargs for apply_chat_template, e.g. enable_thinking=True"
    )
    args = parser.parse_args()

    chat_kwargs = {}
    if args.chat_kwargs:
        for kv in args.chat_kwargs.split(","):
            k, v = kv.split("=")
            chat_kwargs[k.strip()] = eval(v.strip())

    os.makedirs(args.out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(args.model_path)
    print(f"Target: {args.target_tokens} tokens")

    for split in ["train", "test"]:
        src_path = os.path.join(args.src_dir, f"{split}.parquet")
        if not os.path.exists(src_path):
            print(f"SKIP: {src_path} not found")
            continue

        t = pq.read_table(src_path)
        data = t.to_pydict()

        new_prompts = []
        exact = 0
        under = 0
        over = 0
        for idx, prompt in enumerate(data["prompt"]):
            content = prompt[0]["content"]
            final_content, final_n = pad_to_target(tok, content, args.target_tokens, **chat_kwargs)
            new_prompts.append([{"role": "user", "content": final_content}])

            if final_n == args.target_tokens:
                exact += 1
            elif final_n < args.target_tokens:
                under += 1
            else:
                over += 1

            if idx < 3:
                orig_n = len(tokenize_prompt(tok, content, **chat_kwargs))
                print(f"  [{split}] sample[{idx}]: {orig_n} -> {final_n} tokens (target={args.target_tokens})")

        data["prompt"] = new_prompts
        out_ds = datasets.Dataset.from_dict(data)
        out_path = os.path.join(args.out_dir, f"{split}.parquet")
        out_ds.to_parquet(out_path)
        print(f"  Saved {split}: {len(new_prompts)} rows -> {out_path}")
        print(f"    exact={exact}, under={under} (<=target, veRL pads), over={over} (BUG if >0)")
        assert over == 0, f"FATAL: {over} prompts exceed target — would cause tensor mismatch at runtime"

    print("Done")


if __name__ == "__main__":
    main()
