# tools/gen_termination_booster.py
from __future__ import annotations
import argparse, json, random, time
from pathlib import Path

BASE_WORDS = [
    "river","stone","ratio","vector","entity","matrix","signal","system","spring","frame","module","sprite",
    "kernel","buffer","shader","vertex","script","engine","logic","input","output","player","camera","object",
    "event","queue","state","thread","atomic","record","packet","socket","binary","decimal","radius","angle",
    "length","width","height","grid","tile","noise","sample","filter","sum","mean","median","mode","range",
    "scale","offset","domain","stride","branch","loop","stack","heap","tensor","scalar","random","hash","key",
    "value","tree","graph","edge","node","weight","loss","epoch","batch","token","stream","scope","target",
    "source","clock","tick","frame","update","draw","compile","link","launch","device","memory","cache","page",
    "cursor","window","screen","sound","music","layer","scene","actor","action","area","field","limit","bound",
    "force","mass","speed","time","color","light","shadow","normal","surface","volume","pattern","symbol",
    "letter","number","string","array","list","map","set","queue"
]
SYL_A = ["ra","ro","ri","re","la","lo","li","le","na","no","ni","ne","ta","to","ti","te","ka","ko","ki","ke"]
SYL_B = ["tor","tex","lum","num","graph","mesh","node","pack","step","mark","cast","path","rank","grain","grid","span"]
SYL_C = ["ic","al","er","or","ing","ed","um","ix","on","us","en","el","ar","an"]

def make_word(rng: random.Random) -> str:
    if rng.random() < 0.6:
        return rng.choice(BASE_WORDS)
    a, b = rng.choice(SYL_A), rng.choice(SYL_B)
    w = a + b + (rng.choice(SYL_C) if rng.random() < 0.5 else "")
    return w.lower()

def make_number(rng: random.Random, lo=0, hi=9_999_999) -> str:
    return str(rng.randint(lo, hi))

def sample_X(rng: random.Random) -> str:
    r = rng.random()
    if r < 0.45:   # numbers 45%
        return make_number(rng)
    else:          # words 55%
        return make_word(rng)

def line_immediate_close(X: str) -> str:
    # teaches: after <final>, it's legal to close — use sparingly!
    return f"{X}\n<think></think><final></final>"

def line_content_close(X: str) -> str:
    # main pattern: content inside final
    return f"{X}\n<think></think><final>{X}</final>"

def line_onechar_nudge(X: str) -> str:
    # nudge: at least one char before closing
    ch = X[:1]
    return f"{X}\n<think></think><final>{ch}</final>"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="data/termination_booster.jsonl")
    ap.add_argument("--n", type=int, default=30000)
    ap.add_argument("--seed", type=int, default=20251)
    ap.add_argument("--ratio_immediate", type=float, default=0.05,
                    help="fraction of samples that are <final></final> (keep small!)")
    ap.add_argument("--ratio_onechar", type=float, default=0.15,
                    help="fraction with one character inside <final> as a nudge")
    args = ap.parse_args()

    rng = random.Random(args.seed)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)

    n_immediate = int(args.n * args.ratio_immediate)
    n_onechar  = int(args.n * args.ratio_onechar)
    n_content  = args.n - n_immediate - n_onechar

    t0 = time.time()
    print(f"[term-gen] writing {args.n:,} → {args.out}")
    written = 0
    with open(args.out, "w", encoding="utf-8", newline="\n") as f:
        for _ in range(n_content):
            X = sample_X(rng)
            f.write(json.dumps({"text": line_content_close(X)}, ensure_ascii=False) + "\n"); written += 1
            if written % 5000 == 0: print(f"[{written:7d}/{args.n:7d}] {time.time()-t0:6.1f}s")
        for _ in range(n_onechar):
            X = sample_X(rng)
            f.write(json.dumps({"text": line_onechar_nudge(X)}, ensure_ascii=False) + "\n"); written += 1
            if written % 5000 == 0: print(f"[{written:7d}/{args.n:7d}] {time.time()-t0:6.1f}s")
        for _ in range(n_immediate):
            X = sample_X(rng)
            f.write(json.dumps({"text": line_immediate_close(X)}, ensure_ascii=False) + "\n"); written += 1
            if written % 5000 == 0: print(f"[{written:7d}/{args.n:7d}] {time.time()-t0:6.1f}s")

    print(f"[term-gen] done {written:,} in {time.time()-t0:.1f}s → {args.out}")

if __name__ == "__main__":
    main()
