# tools/gen_echo_dataset.py
from __future__ import annotations
import argparse, os, random, time, json
from pathlib import Path

# ---- knobs ----
RATIO_NUMBERS = 0.45   # 45% numbers
RATIO_WORDS   = 0.55   # 55% words
INT_MIN, INT_MAX = 0, 9_999_999

# curated real-word bank (neutral nouns, no proper names)
REAL_WORDS = [
    "anchor", "answer", "apple", "artist", "aspect", "balance", "beacon", "binary", "bridge", "budget",
    "camera", "canvas", "circle", "cloud", "coffee", "column", "compass", "copper", "corner", "cotton",
    "crystal", "current", "custom", "detail", "device", "dialog", "dinner", "dragon", "drawer", "energy",
    "engine", "fabric", "factor", "falcon", "family", "feather", "figure", "finger", "forest", "garden",
    "gentle", "guitar", "harbor", "harvest", "helmet", "honest", "horizon", "island", "jungle", "kettle",
    "ladder", "lantern", "leader", "legend", "letter", "library", "linen", "marble", "market", "matrix",
    "meadow", "meteor", "mirror", "mountain", "museum", "needle", "number", "object", "office", "orbit",
    "oxygen", "painter", "paper", "parcel", "pebble", "pepper", "pillow", "planet", "plasma", "pocket",
    "puzzle", "quartz", "rabbit", "ribbon", "riddle", "river", "rocket", "saddle", "sailor", "sample",
    "screen", "shadow", "signal", "silver", "singer", "socket", "spectrum", "spring", "stable", "stone",
    "story", "stream", "summer", "tablet", "temple", "theory", "thunder", "timber", "tunnel", "valley",
    "velvet", "vector", "violin", "violet", "walnut", "window", "winter", "wonder", "writer", "zebra"
]
def make_word(rng: random.Random) -> str:
    if not REAL_WORDS:
        raise ValueError("REAL_WORDS list is empty")
    return rng.choice(REAL_WORDS)

def make_number(rng: random.Random) -> str:
    return str(rng.randint(INT_MIN, INT_MAX))

def make_item(rng: random.Random) -> str:
    if rng.random() < RATIO_NUMBERS:
        s = make_number(rng)
    else:
        s = make_word(rng)
    # Echo-format JSONL line
    text = f"{s}\n<think>{s}</think>\n<final>{s}</final>"
    return json.dumps({"text": text}, ensure_ascii=False)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/repetition_basic.jsonl")
    ap.add_argument("--n", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=1337)
    ap.add_argument("--shuffle", action="store_true", help="shuffle in memory before writing")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    N = args.n
    t0 = time.time()
    print(f"[echo-gen] writing {N:,} lines -> {out_path}")

    if args.shuffle:
        # Generate in memory then shuffle (needs RAM)
        items = [make_item(rng) for _ in range(N)]
        rng.shuffle(items)
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            for i, line in enumerate(items, 1):
                f.write(line + "\n")
                if i % 5000 == 0:
                    dt = time.time() - t0
                    print(f"[{i:7d}/{N:7d}] {dt:6.1f}s")
    else:
        # Stream to disk; mix types by probability each step
        with open(out_path, "w", encoding="utf-8", newline="\n") as f:
            for i in range(1, N + 1):
                f.write(make_item(rng) + "\n")
                if i % 5000 == 0:
                    dt = time.time() - t0
                    print(f"[{i:7d}/{N:7d}] {dt:6.1f}s")

    dt = time.time() - t0
    print(f"[echo-gen] done in {dt:.1f}s -> {out_path} (UTF-8)")

if __name__ == "__main__":
    main()

