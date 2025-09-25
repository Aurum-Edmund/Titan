#!/usr/bin/env python
import json, random, argparse, hashlib, os
random.seed(42)

def norm(s: str) -> str:
    return " ".join(s.strip().lower().split())

def key_exact(s: str) -> str:
    return s  # exact string

def key_normalized(s: str) -> str:
    return hashlib.sha1(norm(s).encode("utf-8")).hexdigest()

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in_path", required=True)
    ap.add_argument("--train_out", default="data/train.jsonl")
    ap.add_argument("--val_out", default="data/val.jsonl")
    ap.add_argument("--val_frac", type=float, default=0.1)
    ap.add_argument("--dedup", choices=["none","exact","normalized"], default="none")
    args = ap.parse_args()

    os.makedirs(os.path.dirname(args.train_out), exist_ok=True)
    rows = []
    with open(args.in_path,"r",encoding="utf-8") as f:
        for l in f:
            if l.strip(): rows.append(l)

    if args.dedup != "none":
        keys = set(); out=[]
        for l in rows:
            obj = json.loads(l)
            t = obj.get("text") or obj.get("prompt") or ""
            k = key_exact(t) if args.dedup=="exact" else key_normalized(t)
            if k in keys: continue
            keys.add(k); out.append(l)
        rows = out

    random.shuffle(rows)
    n = len(rows)
    n_val = max(1, int(n * args.val_frac))
    with open(args.val_out, "w", encoding="utf-8") as fv: fv.writelines(rows[:n_val])
    with open(args.train_out, "w", encoding="utf-8") as ft: ft.writelines(rows[n_val:])
    print(f"wrote train={n-n_val}  val={n_val}  (dedup={args.dedup})")

if __name__ == "__main__":
    main()
