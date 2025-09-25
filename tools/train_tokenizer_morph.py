# tools/train_tokenizer_morph.py
from __future__ import annotations
import argparse, json, os, re
from pathlib import Path
from typing import Iterable, List

from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from tokenizers.pre_tokenizers import Split, Sequence
from transformers import PreTrainedTokenizerFast

# ——— Special tokens you already use
SPECIALS = ["<|endoftext|>", "<|pad|>", "<think>", "</think>", "<final>", "</final>"]

# ——— Latin / Germanic morphemes (curated, compact starter set)
LATIN_PREFIXES = [
    "anti","auto","bi","circum","co","com","con","contra","de","dis","en","ex","extra",
    "hetero","homo","hyper","il","im","in","inter","intra","ir","macro","micro",
    "mono","multi","non","post","pre","pro","re","semi","sub","super","trans","tri","ultra","uni"
]
GERMANIC_PREFIXES = [
    "be","ent","er","ge","miss","un","ver","zer","ur","über","unter","wider","hinter","vor","nach"
]
LATIN_SUFFIXES = [
    "able","ible","al","ial","ary","ory","ate","ation","sion","tion","ment","ness","less","ful","ous",
    "ify","ise","ize","ing","ed","er","est","ist","ism","ity","ty","logy","ology"
]
GERMANIC_SUFFIXES = [
    "keit","heit","ung","chen","lein","schaft","tum","bar","lich","los","haft","ig"
]

# Normalize to lower for matching; tokenizer itself stays case-sensitive via byte-level
PFX = sorted(set(LATIN_PREFIXES + GERMANIC_PREFIXES), key=len, reverse=True)
SFX = sorted(set(LATIN_SUFFIXES + GERMANIC_SUFFIXES), key=len, reverse=True)

# Regex that isolates prefixes at the start and suffixes at the end of a (wordish) chunk.
# We use case-insensitive matching but only split inside [A-Za-z…] spans.
# `behavior="isolated"` keeps the matched piece as its own token candidate.
PFX_RE = re.compile(rf"(?i)^(?:{'|'.join(map(re.escape, PFX))})(?=[A-Za-zÀ-ÖØ-öø-ÿ])")
SFX_RE = re.compile(rf"(?i)(?<=\w)(?:{'|'.join(map(re.escape, SFX))})$")

def iter_jsonl_texts(paths: List[str]) -> Iterable[str]:
    for p in paths:
        data = Path(p).read_bytes()
        text = None
        for enc in ("utf-8-sig","utf-16","utf-8","latin-1"):
            try:
                text = data.decode(enc)
                break
            except UnicodeDecodeError:
                continue
        if text is None:
            raise UnicodeError(f"Could not decode {p}")
        for line in text.splitlines():
            s = line.strip()
            if not s:
                continue
            if s and s[0] == "\ufeff":
                s = s.lstrip("\ufeff")
            try:
                obj = json.loads(s)
            except json.JSONDecodeError:
                continue
            if "text" in obj:
                yield obj["text"]
            elif "prompt" in obj and "completion" in obj:
                yield f'{obj["prompt"]}\n{obj["completion"]}'
            elif "input" in obj and "output" in obj:
                yield f'{obj["input"]}\n{obj["output"]}'
            # otherwise skip quietly

def write_corpus(tmp_path: str, texts: Iterable[str], cap: int | None):
    n = 0
    with open(tmp_path, "w", encoding="utf-8") as f:
        for t in texts:
            t = t.replace("\r\n","\n").replace("\r","\n")
            f.write(t + "\n")
            n += 1
            if cap and n >= cap:
                break
    return n

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", nargs="+", required=True,
                    help="jsonl files (e.g. data/train.jsonl data/val.jsonl)")
    ap.add_argument("--out_dir", default="tokenizer_morph_en",
                    help="output tokenizer folder")
    ap.add_argument("--vocab_size", type=int, default=52000,
                    help="target vocab size (incl. specials)")
    ap.add_argument("--min_freq", type=int, default=2)
    ap.add_argument("--cap_lines", type=int, default=0)
    args = ap.parse_args()

    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    tmp_corpus = out_dir / "_corpus.txt"
    used = write_corpus(str(tmp_corpus),
                        iter_jsonl_texts(args.jsonl),
                        cap=args.cap_lines if args.cap_lines>0 else None)
    print(f"[TOK] corpus lines used: {used}")

    # ——— Build a morphology-aware pre-tokenizer:
    # ByteLevel first (robustness), then isolate known prefixes and suffixes inside word spans.
    pre = Sequence([
        pre_tokenizers.ByteLevel(add_prefix_space=False),
        Split(PFX_RE, behavior="isolated", invert=False),
        Split(SFX_RE, behavior="isolated", invert=False),
    ])

    # ——— Train BPE on the pre-segmented stream
    base = models.BPE(unk_token="<|unk|>")
    tok = Tokenizer(base)
    tok.pre_tokenizer = pre
    tok.decoder = decoders.ByteLevel()

    # Reserve specials + explicitly reserve morphemes so they become single tokens
    reserved = ["<|unk|>"] + SPECIALS
    # Add lowercase variants of morphemes as reserved tokens (kept as atomic pieces)
    # NOTE: We don't *force* their use; we just guarantee they exist as 1 token.
    reserved += [f"<p:{m}>" for m in PFX] + [f"<s:{m}>" for m in SFX]

    trainer = trainers.BpeTrainer(
        vocab_size=max(len(reserved)+256, args.vocab_size),
        min_frequency=args.min_freq,
        special_tokens=reserved,
        limit_alphabet=1000,
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        show_progress=True,
    )

    tok.train(files=[str(tmp_corpus)], trainer=trainer)

    # ——— Fast wrapper
    fast = PreTrainedTokenizerFast(
        tokenizer_object=tok,
        bos_token=None,
        eos_token="<|endoftext|>",
        pad_token="<|pad|>",
        unk_token="<|unk|>",
        additional_special_tokens=[t for t in reserved if t not in {"<|unk|>","<|endoftext|>","<|pad|>"}],
    )
    fast.save_pretrained(str(out_dir))
    print(f"[TOK] saved → {out_dir}")

    # ——— Quick probes
    def probe(s: str):
        ids = fast.encode(s, add_special_tokens=False)
        toks = fast.convert_ids_to_tokens(ids)
        print(f"{s!r}\n  -> {toks}\n  ({len(ids)} pieces)\n")

    for s in [
        "<think>ratio</think><final>ratio</final>",
        "unbelievable", "verstehen", "Widerstand", "reconstruction",
        "internationalization", "kindlichkeit", "Übertragung",
        "matrix", "hyperactive", "Verarbeitung", "Kinderchen"
    ]:
        probe(s)

    # specials single-piece check
    for s in ["<think>","</think>","<final>","</final>"]:
        ids = fast.encode(s, add_special_tokens=False)
        assert len(ids)==1, f"{s} was not 1 token!"

    try:
        os.remove(tmp_corpus)
    except OSError:
        pass

if __name__ == "__main__":
    main()
