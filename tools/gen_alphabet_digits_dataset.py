from __future__ import annotations

import argparse
import json
from itertools import cycle
from pathlib import Path
from typing import Iterator

ALPHABET_UPPER = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
ALPHABET_LOWER = list("abcdefghijklmnopqrstuvwxyz")
DIGITS = list("0123456789")
SEQUENCE = ALPHABET_UPPER + ALPHABET_LOWER + DIGITS

def sample_pairs(count: int, seed: int) -> Iterator[dict[str, str]]:
    del seed
    seq = cycle(SEQUENCE)
    for _ in range(count):
        token = next(seq)
        response = f"<think>{token}</think><final>{token}</final>"
        yield {"prompt": token, "response": response}

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate ordered alphabet/digit echo data")
    parser.add_argument("output", type=Path, help="Output JSONL path")
    parser.add_argument("--count", type=int, default=1_000_000, help="Number of examples to generate")
    parser.add_argument("--seed", type=int, default=0, help="Unused (kept for compatibility)")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as fh:
        for pair in sample_pairs(args.count, args.seed):
            fh.write(json.dumps(pair, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    main()
