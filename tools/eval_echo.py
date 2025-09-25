from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import torch
import torch.nn as nn
from transformers import AutoTokenizer

# Ensure the repo's src/ directory is importable without requiring installation.
ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

try:
    from titan.core import TitanModel
except ImportError as exc:  # pragma: no cover - hard failure for incorrect layout
    raise ImportError(
        "Unable to import TitanModel from src/titan/core.py. "
        "Verify that the repository layout is intact."
    ) from exc

_THINK_PATTERN = re.compile(r"<think>(.*?)</think>", flags=re.S)
_FINAL_PATTERN = re.compile(r"<final>(.*?)</final>", flags=re.S)
_CONTROL_CHARS = re.compile(r"[\x00-\x1F\x7F]")
_CANDIDATE_ENCODINGS: Tuple[str, ...] = ("utf-8-sig", "utf-8", "utf-16", "latin-1")


@dataclass
class EchoSample:
    prompt: str
    expect_think: str
    expect_final: str
    raw: str


def _decode_bytes(data: bytes) -> str:
    for enc in _CANDIDATE_ENCODINGS:
        try:
            return data.decode(enc)
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError("decode", data, 0, len(data), "Unable to decode prompts file")


def _extract_tag(text: str, pattern: re.Pattern[str]) -> str:
    match = pattern.search(text)
    return match.group(1).strip() if match else ""


def _parse_sample(text: str) -> EchoSample:
    think_body = _extract_tag(text, _THINK_PATTERN)
    final_body = _extract_tag(text, _FINAL_PATTERN)
    preamble_end = _THINK_PATTERN.search(text)
    preamble = text[: preamble_end.start()] if preamble_end else text
    prompt = preamble.strip()

    if not prompt:
        prompt = text.strip()
    if not think_body:
        think_body = prompt
    if not final_body:
        final_body = prompt

    return EchoSample(prompt=prompt, expect_think=think_body, expect_final=final_body, raw=text)


def load_pairs(path: str | Path) -> List[EchoSample]:
    """Read a JSONL file and produce structured prompt/target data."""
    fp = Path(path)
    if not fp.exists():
        raise FileNotFoundError(f"Prompts file not found: {fp}")

    raw = _decode_bytes(fp.read_bytes())
    pairs: List[EchoSample] = []
    for line in raw.splitlines():
        row = line.strip()
        if not row:
            continue
        try:
            obj = json.loads(row)
        except json.JSONDecodeError as exc:
            print(f"[WARN] Skipping malformed JSON line ({exc}): {row[:80]}", file=sys.stderr)
            continue

        text = obj.get("text")
        if text is None:
            prompt = obj.get("prompt")
            completion = obj.get("completion")
            if prompt is not None and completion is not None:
                text = f"{prompt}\n{completion}"
        if text is None:
            print("[WARN] Record lacks 'text' or prompt/completion fields; skipping.", file=sys.stderr)
            continue

        pairs.append(_parse_sample(text))
    return pairs


def _resolve_device(preferred: str) -> torch.device:
    if preferred.startswith("cuda") and not torch.cuda.is_available():
        print("[INFO] CUDA not available; falling back to cpu.", file=sys.stderr)
        return torch.device("cpu")
    return torch.device(preferred)


def _resolve_stop_id(tok, token: str) -> int | None:
    """Best-effort conversion of a stop token into a single token id."""
    try:
        stop_id = tok.convert_tokens_to_ids(token)
    except KeyError:
        stop_id = None
    if stop_id is not None and stop_id != tok.unk_token_id:
        return stop_id
    encoded = tok.encode(token, add_special_tokens=False)
    if len(encoded) == 1:
        return encoded[0]
    return None


def _align_model_vocab(model: TitanModel, target_size: int) -> None:
    current = model.embed.num_embeddings
    if current == target_size:
        return

    device = model.embed.weight.device
    dtype = model.embed.weight.dtype
    embed_dim = model.embed.embedding_dim
    copy = min(current, target_size)

    new_embed = nn.Embedding(target_size, embed_dim, device=device, dtype=dtype)
    with torch.no_grad():
        new_embed.weight[:copy].copy_(model.embed.weight[:copy])
    model.embed = new_embed

    new_head = nn.Linear(embed_dim, target_size, bias=False, device=device, dtype=dtype)
    with torch.no_grad():
        new_head.weight[:copy].copy_(new_embed.weight[:copy])
    new_head.weight = model.embed.weight  # re-tie weights
    model.lm_head = new_head


def _preface_prompt(prompt: str) -> str:
    base = prompt.rstrip("\n")
    return f"{base}\n<think>"


@torch.no_grad()
def generate_sequence(
    model: TitanModel,
    tok,
    prompt: str,
    max_new: int,
    device,
    stop_id: int | None,
):
    prompt_ids = tok.encode(_preface_prompt(prompt), add_special_tokens=False)
    if not prompt_ids:
        raise ValueError("Prompt produced no token ids; ensure tokenizer is configured correctly")

    x = torch.tensor([prompt_ids], device=device, dtype=torch.long)
    for _ in range(max_new):
        outputs = model(x)
        logits = outputs.get("logits") if isinstance(outputs, dict) else outputs
        if logits is None:
            raise RuntimeError("Model forward pass did not produce logits")
        nxt = torch.argmax(logits[:, -1, :], dim=-1, keepdim=True)
        x = torch.cat([x, nxt], dim=1)
        if stop_id is not None and nxt.item() == stop_id:
            break

    ids = x[0].tolist()
    if stop_id is not None and stop_id in ids:
        ids = ids[: ids.index(stop_id) + 1]
    text = tok.decode(ids, skip_special_tokens=False, clean_up_tokenization_spaces=True)
    text = _CONTROL_CHARS.sub("", text)
    return text, ids


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", required=True)
    ap.add_argument("--prompts_jsonl", required=True)
    ap.add_argument("--limit", type=int, default=100)
    ap.add_argument("--max_new", type=int, default=64)
    ap.add_argument("--d", type=int, required=True)
    ap.add_argument("--layers", type=int, required=True)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "fp32", "bf16"])
    args = ap.parse_args()

    dtype_map = {"fp16": torch.float16, "fp32": torch.float32, "bf16": torch.bfloat16}
    dtype = dtype_map[args.dtype]
    device = _resolve_device(args.device)

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    vocab_size = len(tok)

    model = TitanModel(vocab_size=vocab_size, d=args.d, n_blocks=args.layers)
    sd = torch.load(args.ckpt, map_location="cpu")
    sd = sd.get("model", sd)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if missing or unexpected:
        print(f"[WARN] state_dict load: missing={len(missing)} unexpected={len(unexpected)}", file=sys.stderr)
    _align_model_vocab(model, vocab_size)
    model = model.to(device=device, dtype=dtype)
    model.eval()

    stop_id = _resolve_stop_id(tok, "</final>")
    if stop_id is None:
        print("[WARN] Tokenizer lacks a dedicated </final> token; termination relies on max_new.", file=sys.stderr)

    samples = load_pairs(args.prompts_jsonl)
    if args.limit:
        samples = samples[: args.limit]
    total = len(samples)
    if total == 0:
        print("[EVAL] No prompts to evaluate.", file=sys.stderr)
        return

    exact_final = 0
    exact_think = 0
    terminated = 0
    empty_final = 0
    missing_structure = 0

    for idx, sample in enumerate(samples, start=1):
        generated_text, token_ids = generate_sequence(model, tok, sample.prompt, args.max_new, device, stop_id)
        think_text = _extract_tag(generated_text, _THINK_PATTERN)
        final_text = _extract_tag(generated_text, _FINAL_PATTERN)

        if stop_id is not None and token_ids and token_ids[-1] == stop_id:
            terminated += 1
        if not final_text:
            empty_final += 1
        if not think_text or "</think>" not in generated_text or "<final>" not in generated_text:
            missing_structure += 1

        think_ok = think_text == sample.expect_think
        final_ok = final_text == sample.expect_final
        exact_think += int(think_ok)
        exact_final += int(final_ok)

        status = "OK" if think_ok and final_ok else "!!"
        print(f"[{idx:4d}/{total:4d}] {status} prompt={sample.prompt!r} final={final_text!r}")

    print(
        "\n[EVAL] exact-final: {}/{} = {:.2%}   exact-think: {:.2%}   termination: {:.2%}   empty-final: {:.2%}   structure-miss: {:.2%}".format(
            exact_final,
            total,
            exact_final / total,
            exact_think / total,
            terminated / total,
            empty_final / total,
            missing_structure / total,
        )
    )


if __name__ == "__main__":
    main()

