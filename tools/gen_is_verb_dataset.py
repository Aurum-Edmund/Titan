from __future__ import annotations

import argparse
import json
from itertools import cycle
from pathlib import Path
from typing import Iterator, Tuple

EXAMPLES: Tuple[Tuple[str, str], ...] = (
    ("He is happy", "is happy"),
    ("She is ready", "is ready"),
    ("He is calm", "is calm"),
    ("She is curious", "is curious"),
    ("It is sunny", "is sunny"),
    ("It is raining", "is raining"),
    ("This is fine", "is fine"),
    ("This is helpful", "is helpful"),
    ("That is correct", "is correct"),
    ("That is great", "is great"),
)


def sample_pairs(count: int) -> Iterator[dict[str, str]]:
    rot = cycle(EXAMPLES)
    for _ in range(count):
        full, predicate = next(rot)
        response = f"<think>{full}</think><final>{predicate}</final>"
        yield {"prompt": full, "response": response}


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate 'is' verb usage dataset")
    parser.add_argument("output", type=Path, help="Output JSONL path")
    parser.add_argument("--count", type=int, default=1_000_000, help="Number of examples to generate")
    args = parser.parse_args()

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as fh:
        for pair in sample_pairs(args.count):
            fh.write(json.dumps(pair, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
