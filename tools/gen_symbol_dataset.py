from __future__ import annotations

import argparse
import json
from itertools import cycle
from pathlib import Path
from typing import Iterator

SYMBOLS = ['<', '>', '/', '\\', '{', '}', '[', ']', '(', ')', '!', '?', '.', ',', ';', ':', "'", '"', '-', '+', '*', '=', '&', '@', '#', '$', '%', '^', '~', '|']


def sample_pairs(count: int) -> Iterator[dict[str, str]]:
    sym_iter = cycle(SYMBOLS)
    for _ in range(count):
        symbol = next(sym_iter)
        response = f"<think>{symbol}</think><final>{symbol}</final>"
        yield {"prompt": symbol, "response": response}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate single-symbol echo dataset")
    parser.add_argument("output", type=Path, help="Output JSONL path")
    parser.add_argument("--count", type=int, default=1_000_000, help="Number of examples to generate")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as fh:
        for pair in sample_pairs(args.count):
            fh.write(json.dumps(pair, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
