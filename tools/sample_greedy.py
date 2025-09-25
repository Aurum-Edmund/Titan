# tools/sample_greedy.py
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from titan.core import TitanModel  # adjust if your path differs



def _preface_prompt(prompt: str) -> str:
    if "<think>" in prompt and "</think>" in prompt:
        return prompt
    base = prompt.rstrip("\n")
    return f"{base}\n<think>"


def build_model(V: int, d: int, layers: int, device, dtype):
    m = TitanModel(vocab_size=V, d=d, n_blocks=layers)
    return m.to(device).to(dtype)


def ensure_vocab_match(model, tok):
    V_tok = len(tok)
    V_emb = model.embed.num_embeddings
    head = getattr(model, "lm_head", None)
    head_out = getattr(head, "out_features", None)
    ok = (V_tok == V_emb) and (head_out is None or head_out == V_tok)
    if not ok:
        raise RuntimeError(f"Vocab mismatch: tokenizer={V_tok}, embed={V_emb}, head={head_out}")


@torch.no_grad()
def generate_greedy(model, tok, prompt: str, max_new: int, device):
    model.eval()
    text_prompt = _preface_prompt(prompt)
    ids = tok.encode(text_prompt, add_special_tokens=False)
    if not ids:
        raise ValueError("Prompt produced no token ids; ensure tokenizer covers the inputs.")
    x = torch.tensor([ids], device=device)
    stop_id = tok.convert_tokens_to_ids("</final>")
    for _ in range(max_new):
        out = model(x)
        logits = out["logits"] if isinstance(out, dict) else out
        nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        x = torch.cat([x, nxt], dim=1)
        if stop_id != tok.unk_token_id and nxt.item() == stop_id:
            break
    seq = x[0].tolist()
    if stop_id in seq:
        seq = seq[: seq.index(stop_id) + 1]
    text = tok.decode(seq, skip_special_tokens=False, clean_up_tokenization_spaces=True)
    text = re.sub(r"[\x00-\x1F\x7F]", "", text)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prompt", required=True)
    ap.add_argument("--max_new", type=int, default=64)
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32", "bf16"])
    args = ap.parse_args()

    dtype_map = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
    dtype = dtype_map[args.dtype]
    device = torch.device(args.device if torch.cuda.is_available() or args.device == "cpu" else "cpu")

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    V = len(tok)

    model = build_model(V, d=args.d, layers=args.layers, device=device, dtype=dtype)
    sd = torch.load(args.ckpt, map_location="cpu")
    sd = sd.get("model", sd)
    if model.embed.num_embeddings != V:
        model.resize_token_embeddings(V)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[SD] missing={len(missing)} unexpected={len(unexpected)}", file=sys.stderr)
    ensure_vocab_match(model, tok)

    out = generate_greedy(model, tok, args.prompt, max_new=args.max_new, device=device)
    print("\n=== OUTPUT ===")
    print(out)


if __name__ == "__main__":
    main()

