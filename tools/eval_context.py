from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

import torch
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

from src.titan.context_core import ContextCore, ContextHeadDims
from src.titan.train_context_core import ContextJsonlDS, collate

BINARY_KEYS = ["math_scent", "code_scent", "route_math", "route_code", "route_lang"]
MULTI_KEYS = ["data_format", "topic", "intent"]


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="Evaluate ContextCore classification heads")
    ap.add_argument("--ckpt", required=True, help="Checkpoint path")
    ap.add_argument("--tokenizer", required=True, help="Tokenizer directory")
    ap.add_argument("--jsonl", required=True, help="Evaluation dataset (.jsonl)")
    ap.add_argument("--d", type=int, required=True, help="Model hidden size")
    ap.add_argument("--layers", type=int, required=True, help="Number of encoder layers")
    ap.add_argument("--heads", type=int, required=True, help="Number of attention heads")
    ap.add_argument("--topics", type=int, default=16, help="Topic head dimension")
    ap.add_argument("--intents", type=int, default=8, help="Intent head dimension")
    ap.add_argument("--dropout", type=float, default=0.0, help="Dropout probability used during training")
    ap.add_argument("--max-pos", type=int, default=4096, help="Max positional embeddings")
    ap.add_argument("--max-len", type=int, default=256, help="Sequence truncation length")
    ap.add_argument("--limit", type=int, default=0, help="Optional limit on number of examples")
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16", choices=["fp16", "bf16", "fp32"])
    return ap.parse_args()


def to_device(tensor: torch.Tensor, device: torch.device) -> torch.Tensor:
    return tensor.to(device, non_blocking=True)


def sigmoid_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = torch.sigmoid(logits).squeeze(-1) > 0.5
    return (preds == targets.bool()).float().mean().item()


def multi_accuracy(logits: torch.Tensor, targets: torch.Tensor) -> float:
    preds = logits.argmax(dim=-1)
    return (preds == targets).float().mean().item()


def gather_metrics(logits: Dict[str, torch.Tensor], labels: List[Dict[str, int]]) -> Dict[str, float]:
    device = logits["topic"].device
    metrics: Dict[str, float] = {}
    for key in BINARY_KEYS:
        target = torch.tensor([float(lbl[key]) for lbl in labels], dtype=torch.float32, device=device)
        metrics[f"acc_{key}"] = sigmoid_accuracy(logits[key], target)
    for key in MULTI_KEYS:
        target = torch.tensor([int(lbl[key]) for lbl in labels], dtype=torch.long, device=device)
        metrics[f"acc_{key}"] = multi_accuracy(logits[key], target)
    return metrics


def merge_metrics(running: Dict[str, float], latest: Dict[str, float], batch_size: int) -> Dict[str, float]:
    if not running:
        return {k: latest[k] * batch_size for k in latest}
    for key in latest:
        running[key] = running.get(key, 0.0) + latest[key] * batch_size
    return running


def scale_metrics(running: Dict[str, float], total: int) -> Dict[str, float]:
    if total == 0:
        return {k: 0.0 for k in running}
    return {k: v / total for k, v in running.items()}


def main() -> None:
    args = parse_args()

    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    dtype = dtype_map[args.dtype]

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if tok.pad_token_id is None:
        tok.add_special_tokens({"pad_token": "<|pad|>"})
    pad_id = tok.pad_token_id or 0

    head_dims = ContextHeadDims(num_topics=args.topics, num_intents=args.intents)
    dataset = ContextJsonlDS(args.jsonl, tok, max_len=args.max_len, head_dims=head_dims)
    if args.limit and args.limit < len(dataset):
        dataset = Subset(dataset, list(range(args.limit)))

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=(device.type == "cuda"),
        collate_fn=lambda batch: collate(batch, pad_id),
    )

    model = ContextCore(
        vocab_size=len(tok),
        d=args.d,
        n_layers=args.layers,
        n_heads=args.heads,
        pad_id=pad_id,
        head_dims=head_dims,
        dropout=args.dropout,
        max_position_embeddings=args.max_pos,
    ).to(device=device, dtype=dtype)
    model.eval()

    ckpt = torch.load(args.ckpt, map_location="cpu")
    state = ckpt.get("model", ckpt)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing or unexpected:
        print(f"[WARN] missing={len(missing)} unexpected={len(unexpected)}")

    totals: Dict[str, float] = {}
    total_examples = 0

    with torch.no_grad():
        for xb, mb, lbls in loader:
            xb = to_device(xb, device)
            mb = to_device(mb, device)
            out = model(xb, attention_mask=mb)["logits"]
            batch_size = xb.size(0)
            metrics = gather_metrics(out, lbls)
            totals = merge_metrics(totals, metrics, batch_size)
            total_examples += batch_size

    final_metrics = scale_metrics(totals, total_examples)
    print(f"[context-eval] examples={total_examples}")
    for key in sorted(final_metrics):
        print(f"  {key}: {final_metrics[key]:.4f}")


if __name__ == "__main__":
    main()
