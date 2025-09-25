# tools/eval_memorization.py
from __future__ import annotations
import sys, argparse, json, torch, math
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer
from src.titan.core import TitanModel

def load_tok(dir):
    tok = AutoTokenizer.from_pretrained(dir)
    if tok.pad_token_id is None: tok.add_special_tokens({"pad_token":"<|pad|>"})
    return tok

def forward_logits(model, ids: torch.Tensor):
    out = model(ids); return out["logits"] if isinstance(out, dict) else out

def nll_per_example(model, tok, lines, device="cuda", max_len=256):
    losses=[]; lossf = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="sum")
    for i,l in enumerate(lines):
        s=json.loads(l).get("text","")
        ids = tok.encode(s, add_special_tokens=False)[:max_len]
        if len(ids)<2: continue
        x = torch.tensor(ids[:-1], dtype=torch.long, device=device).unsqueeze(0)
        y = torch.tensor(ids[1:],  dtype=torch.long, device=device).unsqueeze(0)
        with torch.no_grad(), torch.amp.autocast("cuda", enabled=True):
            logits = forward_logits(model, x)
            loss = lossf(logits.view(-1, logits.size(-1)), y.view(-1)).item()
        toks = (y!=-100).sum().item()
        losses.append((i, loss/toks if toks>0 else float("inf")))
    return losses

def maybe_read_model_cfg(sd):
    meta = sd.get("meta") if isinstance(sd, dict) else None
    if isinstance(meta, dict):
        d = meta.get("d_model") or meta.get("d")
        L = meta.get("n_layers") or meta.get("n_blocks")
        return (int(d) if d else None, int(L) if L else None)
    return (None, None)

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="tokenizer_local")
    ap.add_argument("--train_jsonl", required=True)
    ap.add_argument("--val_jsonl",   required=True)
    ap.add_argument("--topk", type=int, default=20)
    ap.add_argument("--d", type=int, default=None)
    ap.add_argument("--layers", type=int, default=None)
    args=ap.parse_args()

    tok=load_tok(args.tokenizer); V=len(tok)
    device="cuda" if torch.cuda.is_available() else "cpu"

    raw = torch.load(args.ckpt, map_location="cpu"); sd = raw.get("model", raw)
    d_ckpt, L_ckpt = maybe_read_model_cfg(raw)
    d = args.d or d_ckpt or 512
    L = args.layers or L_ckpt or 36
    print(f"[MODEL] d={d} layers={L}")

    model=TitanModel(vocab_size=V, d=d, n_blocks=L).to(device).to(torch.float16)
    miss, unexp = model.load_state_dict(sd, strict=False)
    if miss or unexp:
        print(f"[SD] missing={len(miss)} unexpected={len(unexp)}")
    model.eval()

    train_lines=[l for l in open(args.train_jsonl,"r",encoding="utf-8") if l.strip()]
    val_lines  =[l for l in open(args.val_jsonl,"r",encoding="utf-8")   if l.strip()]

    tr = nll_per_example(model, tok, train_lines, device=device)
    va = nll_per_example(model, tok, val_lines,   device=device)

    tr_sorted=sorted(tr, key=lambda x:x[1])[:args.topk]
    va_sorted=sorted(va, key=lambda x:x[1])[:args.topk]

    print("\nLowest-NLL train examples (potential memorization):")
    for i,nll in tr_sorted:
        s=json.loads(train_lines[i])["text"]
        print(f"  nll={nll:.4f}  {s[:120]!r}")

    print("\nLowest-NLL val examples:")
    for i,nll in va_sorted:
        s=json.loads(val_lines[i])["text"]
        print(f"  nll={nll:.4f}  {s[:120]!r}")

if __name__=="__main__":
    main()
