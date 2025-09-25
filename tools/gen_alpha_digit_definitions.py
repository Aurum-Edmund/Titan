from __future__ import annotations

import argparse
import json
from itertools import cycle
from pathlib import Path
from typing import Iterator

LETTERS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
DIGITS = list("0123456789")
SEQUENCE = LETTERS + DIGITS

def definition(token: str) -> str:
    kind = "number" if token in DIGITS else "letter"
    return f"{token} is a {kind}"

def sample_pairs(count: int) -> Iterator[dict[str, str]]:
    seq = cycle(SEQUENCE)
    for _ in range(count):
        token = next(seq)
        statement = definition(token)
        response = f"<think>{statement}</think><final>{statement}</final>"
        yield {"prompt": statement, "response": response}

def main() -> None:
    parser = argparse.ArgumentParser(description="Generate deterministic letter/number definition data")
    parser.add_argument("output", type=Path, help="Output JSONL path")
    parser.add_argument("--count", type=int, default=1_000_000, help="Number of examples to generate")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as fh:
        for pair in sample_pairs(args.count):
            fh.write(json.dumps(pair, ensure_ascii=False) + "\n")

if __name__ == "__main__":
    main()
