# tools/retrain_tokenizer.py
from __future__ import annotations
import argparse, json, os
from pathlib import Path
from typing import Iterable, List

# Hugging Face "tokenizers" (low-level) + Fast wrapper
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from transformers import PreTrainedTokenizerFast

SPECIALS = [
    "<|endoftext|>",  # eos (GPT-2 convention)
    "<|pad|>",        # pad
    "<think>", "</think>",
    "<final>", "</final>",
]

def iter_jsonl_texts(paths: List[str]) -> Iterable[str]:
    for p in paths:
        with open(p, "rb") as fb:
            raw = fb.read()
        for enc in ("utf-8-sig", "utf-16", "utf-8", "latin-1"):
            try:
                text = raw.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        else:
            raise UnicodeError(f"Could not decode {p}")

        for line in text.splitlines():
            s = line.strip()
            if not s:
                continue
            if s[0] == "\ufeff":
                s = s.lstrip("\ufeff")
            obj = json.loads(s)
            if "text" in obj:
                yield obj["text"]
            elif "prompt" in obj and "completion" in obj:
                yield f'{obj["prompt"]}\n{obj["completion"]}'
            elif "input" in obj and "output" in obj:
                yield f'{obj["input"]}\n{obj["output"]}'
            else:
                # ignore unknown shapes rather than crash
                continue

def write_corpus(tmp_path: str, texts: Iterable[str], cap: int | None):
    n = 0
    with open(tmp_path, "w", encoding="utf-8") as f:
        for t in texts:
            f.write(t.replace("\r\n", "\n").replace("\r", "\n") + "\n")
            n += 1
            if cap and n >= cap:
                break
    return n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", nargs="+", required=True,
                    help="One or more jsonl files (e.g. data/train.jsonl data/val.jsonl)")
    ap.add_argument("--out_dir", default="tokenizer_local_200k",
                    help="Where to save the trained tokenizer folder")
    ap.add_argument("--vocab_size", type=int, default=50262,
                    help="Total vocab size (including specials)")
    ap.add_argument("--min_freq", type=int, default=2, help="Min frequency for merges")
    ap.add_argument("--cap_lines", type=int, default=0,
                    help="Optional cap on lines used for training (0 = no cap)")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # 1) Build a temporary flat corpus file (tokenizers BPE trainer expects files)
    tmp_corpus = out_dir / "_corpus.txt"
    used = write_corpus(str(tmp_corpus), iter_jsonl_texts(args.jsonl),
                        cap=args.cap_lines if args.cap_lines > 0 else None)
    print(f"[TOK] training corpus lines: {used}")

    # 2) Byte-level BPE (GPT-2-ish)
    #    We provide specials so they get reserved single IDs.
    base = models.BPE(unk_token="<|unk|>")
    tok = Tokenizer(base)
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=max(len(SPECIALS) + 256, args.vocab_size),  # ensure room
        min_frequency=args.min_freq,
        special_tokens=["<|unk|>"] + SPECIALS,
        limit_alphabet=1000,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    tok.train(files=[str(tmp_corpus)], trainer=trainer)

    # 3) Wrap as a Fast tokenizer with proper special token mapping
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token=None,
        eos_token="<|endoftext|>",
        pad_token="<|pad|>",
        unk_token="<|unk|>",
    )

    # 4) Save to folder (tokenizer.json, tokenizer_config.json, special_tokens_map.json)
    fast.save_pretrained(str(out_dir))
    print(f"[TOK] saved → {out_dir}")

    # 5) Verify special tags are single tokens
    check = ["<think>", "</think>", "<final>", "</final>"]
    for s in check:
        ids = fast.encode(s, add_special_tokens=False)
        toks = fast.convert_ids_to_tokens(ids)
        print(f"{s} -> ids: {ids}  toks: {toks}  len={len(ids)}")

    # 6) Cleanup
    try:
        os.remove(tmp_corpus)
    except OSError:
        pass

if __name__ == "__main__":
    main()
