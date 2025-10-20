from __future__ import annotations

import argparse
import json
from math import floor
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

DEFAULT_DATASETS = {
    "letter_defs": (Path("data/letter_digit_defs.jsonl"), 4.0),
    "alphabet_digits": (Path("data/alphabet_digits.jsonl"), 2.0),
    "dictionary_echo": (Path("data/dictionary_echo.jsonl"), 2.0),
    "is_verb": (Path("data/is_verb.jsonl"), 1.0),
}


def load_lines(path: Path) -> List[str]:
    lines: List[str] = []
    with path.open("r", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if line:
                lines.append(line)
    if not lines:
        raise ValueError(f"Dataset {path} is empty")
    return lines


def parse_dataset_args(args: Sequence[str]) -> List[Tuple[Path, float]]:
    parsed: List[Tuple[Path, float]] = []
    for item in args:
        try:
            path_str, weight_str = item.split(":", 1)
            weight = float(weight_str)
        except ValueError as exc:
            raise argparse.ArgumentTypeError(
                "--dataset entries must be PATH:WEIGHT"
            ) from exc
        path = Path(path_str)
        parsed.append((path, weight))
    return parsed


def compute_counts(weights: List[float], total: int) -> List[int]:
    weight_sum = sum(weights)
    raw_counts = [total * w / weight_sum for w in weights]
    counts = [floor(rc) for rc in raw_counts]
    remainder = total - sum(counts)
    if remainder > 0:
        fractional = sorted(
            enumerate([rc - floor(rc) for rc in raw_counts]),
            key=lambda x: x[1],
            reverse=True,
        )
        for idx, _ in fractional[:remainder]:
            counts[idx] += 1
    return counts


def cycle_lines(lines: List[str], count: int) -> Iterable[str]:
    length = len(lines)
    for idx in range(count):
        yield lines[idx % length]


def main() -> None:
    parser = argparse.ArgumentParser(description="Mix multiple curriculum datasets with ratios")
    parser.add_argument("output", type=Path, help="Output JSONL path")
    parser.add_argument("--count", type=int, default=1_000_000, help="Total number of examples in the mixed dataset")
    parser.add_argument(
        "--dataset",
        action="append",
        help="Custom dataset specification PATH:WEIGHT. When omitted, a default mix is used.",
    )
    args = parser.parse_args()

    if args.dataset:
        dataset_specs = parse_dataset_args(args.dataset)
    else:
        dataset_specs = list(DEFAULT_DATASETS.values())

    datasets: List[Tuple[Path, float, List[str]]] = []
    for path, weight in dataset_specs:
        if weight <= 0:
            raise ValueError(f"Weight must be positive for {path}")
        if not path.exists():
            raise FileNotFoundError(path)
        lines = load_lines(path)
        datasets.append((path, weight, lines))

    counts = compute_counts([weight for _, weight, _ in datasets], args.count)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as fh:
        for (path, _, lines), take in zip(datasets, counts):
            for line in cycle_lines(lines, take):
                fh.write(line + "\n")
    print(f"[mix] wrote {args.count} examples to {args.output}")


if __name__ == "__main__":
    main()
