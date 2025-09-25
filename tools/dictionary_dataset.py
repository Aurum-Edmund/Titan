#!/usr/bin/env python3
"""Generate an echo dataset from a dictionary word list sourced online/offline.

By default the script pulls the highest-frequency tokens from the `wordfreq`
package (which ships with a large curated list derived from Wiktionary, Google
Books, subtitles, etc.).  You can optionally provide an additional blocklist
file – any token matching (case-sensitive) will be skipped.

Each output line is a JSON object with the form:
    {"text": "word\n<think>word</think>\n<final>word</final>"}

Examples
--------
python tools/dictionary_dataset.py \
  --out data/dictionary_echo.jsonl \
  --count 1000000 \
  --language en \
  --top-n 200000 \
  --lowercase
"""
from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path
from typing import Iterable, List, Set

try:
    from wordfreq import top_n_list  # type: ignore
except ImportError as exc:  # pragma: no cover - user must install wordfreq
    raise ImportError(
        "The wordfreq package is required. Install it with `pip install wordfreq`."
    ) from exc

SAFE_CHARS = set("abcdefghijklmnopqrstuvwxyz")
DEFAULT_BLOCKLIST = {
    # A light safeguard; you can extend via --blocklist.
    "hitler",
    "nazi",
    "kkk",
    "slur",
    "terrorism",
}


def normalise_token(token: str, lowercase: bool) -> str | None:
    token = token.strip()
    if lowercase:
        token = token.lower()
    if not token:
        return None
    if any(ch not in SAFE_CHARS for ch in token):
        return None
    if len(token) < 2 or len(token) > 48:
        return None
    return token


def load_candidates(language: str, top_n: int, lowercase: bool) -> List[str]:
    words = top_n_list(language, top_n)
    filtered: List[str] = []
    seen: Set[str] = set()
    for word in words:
        norm = normalise_token(word, lowercase)
        if norm and norm not in seen:
            # Skip obvious proper nouns: if original had uppercase initial and we aren't lowercasing
            if not lowercase and word and word[0].isupper():
                continue
            filtered.append(norm)
            seen.add(norm)
    return filtered


def apply_blocklist(words: Iterable[str], blockset: Set[str]) -> List[str]:
    if not blockset:
        return list(words)
    return [w for w in words if w not in blockset]


def iter_words(words: List[str], count: int, rng: random.Random) -> Iterable[str]:
    if not words:
        raise ValueError("Word candidate list is empty after filtering")
    pool = words.copy()
    rng.shuffle(pool)
    if len(pool) >= count:
        yield from pool[:count]
        return
    yield from pool
    remaining = count - len(pool)
    for _ in range(remaining):
        yield rng.choice(pool)


def make_record(word: str) -> str:
    text = f"{word}\n<think>{word}</think>\n<final>{word}</final>"
    return json.dumps({"text": text}, ensure_ascii=False)


def load_blocklist(path: str | None, lowercase: bool) -> Set[str]:
    block = set(DEFAULT_BLOCKLIST)
    if path:
        p = Path(path)
        if not p.exists():
            print(f"[dictionary] warning: blocklist not found at {p}", file=sys.stderr)
        else:
            for line in p.read_text(encoding="utf-8", errors="ignore").splitlines():
                token = normalise_token(line, lowercase)
                if token:
                    block.add(token)
    return block


def main() -> None:
    parser = argparse.ArgumentParser(description="Build an echo dataset from wordfreq vocabulary")
    parser.add_argument("--out", default="data/dictionary_echo.jsonl", help="Output JSONL file")
    parser.add_argument("--count", type=int, default=1_000_000, help="Number of samples to write")
    parser.add_argument("--language", default="en", help="wordfreq language code (default: en)")
    parser.add_argument("--top-n", type=int, default=500_000, help="How many top entries to pull from wordfreq")
    parser.add_argument("--seed", type=int, default=1337, help="Random seed for shuffling")
    parser.add_argument("--lowercase", action="store_true", help="Force all tokens to lowercase before filtering")
    parser.add_argument("--blocklist", default=None, help="Optional newline-delimited blocklist file")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    words = load_candidates(args.language, args.top_n, lowercase=args.lowercase)
    block = load_blocklist(args.blocklist, lowercase=args.lowercase)
    filtered = apply_blocklist(words, block)
    print(f"[dictionary] usable vocabulary: {len(filtered):,} tokens (language={args.language}, lowercase={args.lowercase})")
    if len(filtered) == 0:
        print("[dictionary] error: no usable tokens after filtering", file=sys.stderr)
        sys.exit(1)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    total = args.count
    with out_path.open("w", encoding="utf-8", newline="\n") as f:
        for idx, word in enumerate(iter_words(filtered, total, rng), start=1):
            f.write(make_record(word) + "\n")
            if idx % 10000 == 0:
                print(f"[dictionary] {idx:,}/{total:,}")
    print(f"[dictionary] wrote {total:,} records -> {out_path}")


if __name__ == "__main__":
    main()
