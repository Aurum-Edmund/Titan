from __future__ import annotations

import argparse
import json
import math
import re
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.titan.context_core import ContextCore, ContextHeadDims

try:
    import yaml
except ModuleNotFoundError as exc:  # pragma: no cover - surface friendly error
    raise RuntimeError("PyYAML is required to train the context core") from exc


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def count_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters())


def read_jsonl(path: str) -> List[Dict[str, Any]]:
    """Robust JSONL reader that tolerates BOMs and exotic encodings."""
    with open(path, "rb") as handle:
        raw = handle.read()
    if raw.startswith(b"\xff\xfe") or raw.startswith(b"\xfe\xff"):
        text = raw.decode("utf-16")
    elif raw.startswith(b"\xef\xbb\xbf"):
        text = raw.decode("utf-8-sig")
    else:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError:
            text = raw.decode("latin-1")

    rows: List[Dict[str, Any]] = []
    for line in text.splitlines():
        if line.strip():
            rows.append(json.loads(line))
    return rows


# ---------------------------------------------------------------------------
# Weak labelling helpers
# ---------------------------------------------------------------------------
RE_MATH = re.compile(r"[0-9][0-9\.,\s]*(?:[+\-*/^%=()]|sqrt|sin|cos|tan|log|ln)")
RE_CODE = re.compile(r"\b(int|float|double|std::|#include|template<|def |class |import |for\s+\w+\s+in|printf\(|cout\s*<<)\b")
RE_JSON = re.compile(r"\{\s*\"[^\"\\]+\"\s*:")
RE_XML = re.compile(r"<[A-Za-z_][A-Za-z0-9_\-]*>")
RE_TABLE = re.compile(r"(?:\t| {2,})\S")


def weak_label(text: str) -> Dict[str, Any]:
    stripped = text.strip()
    math_hit = bool(RE_MATH.search(stripped))
    code_hit = bool(RE_CODE.search(stripped))

    if RE_JSON.search(stripped):
        data_format = 3
    elif RE_XML.search(stripped):
        data_format = 4
    elif RE_TABLE.search(stripped):
        data_format = 2
    elif stripped.startswith("- ") or stripped.startswith("* ") or "\n- " in stripped or "\n* " in stripped:
        data_format = 1
    else:
        data_format = 0

    topic = 1 if code_hit else (2 if math_hit else 0)
    intent = 1 if stripped.endswith("?") else (2 if code_hit else 0)

    route_math = 1 if math_hit else 0
    route_code = 1 if code_hit else 0
    route_lang = 1 if not (math_hit or code_hit) else 0

    return {
        "math_scent": float(math_hit),
        "code_scent": float(code_hit),
        "data_format": data_format,
        "topic": topic,
        "intent": intent,
        "route_math": float(route_math),
        "route_code": float(route_code),
        "route_lang": float(route_lang),
    }


# ---------------------------------------------------------------------------
# Dataset + dataloader glue
# ---------------------------------------------------------------------------
class ContextJsonlDataset(Dataset):
    """Dataset that reads JSONL rows with optional gold labels."""

    def __init__(self, jsonl_path: str, tokenizer, max_len: int, head_dims: ContextHeadDims):
        self.rows = read_jsonl(jsonl_path)
        self.tok = tokenizer
        self.max_len = max_len
        self.head_dims = head_dims

    def __len__(self) -> int:  # type: ignore[override]
        return len(self.rows)

    def __getitem__(self, idx: int):  # type: ignore[override]
        row = self.rows[idx]
        text = row.get("text") or row.get("prompt") or ""
        labels = row.get("labels") or weak_label(text)

        token_ids = self.tok.encode(text, add_special_tokens=False)[: self.max_len]
        if not token_ids:
            token_ids = [self.tok.pad_token_id or 0]

        return {
            "ids": torch.tensor(token_ids, dtype=torch.long),
            "labels": labels,
        }


def collate(batch: List[Dict[str, Any]], pad_id: int):
    batch_size = len(batch)
    max_tokens = max(len(item["ids"]) for item in batch)
    tokens = torch.full((batch_size, max_tokens), pad_id, dtype=torch.long)
    mask = torch.zeros((batch_size, max_tokens), dtype=torch.long)

    labels = []
    for idx, item in enumerate(batch):
        seq = item["ids"]
        seq_len = len(seq)
        tokens[idx, :seq_len] = seq
        mask[idx, :seq_len] = 1
        labels.append(item["labels"])

    return tokens, mask, labels


# ---------------------------------------------------------------------------
# Loss + evaluation helpers
# ---------------------------------------------------------------------------

def build_losses(head_dims: ContextHeadDims):
    bce = nn.BCEWithLogitsLoss()
    ce = nn.CrossEntropyLoss()

    def loss_fn(logits: Dict[str, torch.Tensor], labels: List[Dict[str, Any]]):
        device = logits["topic"].device
        total_loss = torch.zeros((), device=device)

        for key in ["math_scent", "code_scent", "route_math", "route_code", "route_lang"]:
            target = torch.tensor([float(lbl[key]) for lbl in labels], dtype=torch.float32, device=device).unsqueeze(-1)
            total_loss = total_loss + bce(logits[key], target)

        fmt_target = torch.tensor([int(lbl["data_format"]) for lbl in labels], dtype=torch.long, device=device)
        topic_target = torch.tensor([int(lbl["topic"]) for lbl in labels], dtype=torch.long, device=device)
        intent_target = torch.tensor([int(lbl["intent"]) for lbl in labels], dtype=torch.long, device=device)

        total_loss = total_loss + ce(logits["data_format"], fmt_target)
        total_loss = total_loss + ce(logits["topic"], topic_target)
        total_loss = total_loss + ce(logits["intent"], intent_target)
        return total_loss

    return loss_fn


def compute_eval_metrics(logits_list: List[Dict[str, torch.Tensor]], labels_list: List[List[Dict[str, Any]]]) -> Dict[str, float]:
    import torch as _torch

    total = 0
    hits_math = hits_code = hits_fmt = hits_topic = hits_intent = 0

    for logits, labels in zip(logits_list, labels_list):
        device = logits["topic"].device
        batch = logits["topic"].shape[0]
        total += batch

        math_pred = (_torch.sigmoid(logits["math_scent"]).squeeze(-1) > 0.5).to(dtype=_torch.float32)
        math_true = _torch.tensor([lbl["math_scent"] for lbl in labels], dtype=_torch.float32, device=device)
        hits_math += math_pred.eq(math_true).sum().item()

        code_pred = (_torch.sigmoid(logits["code_scent"]).squeeze(-1) > 0.5).to(dtype=_torch.float32)
        code_true = _torch.tensor([lbl["code_scent"] for lbl in labels], dtype=_torch.float32, device=device)
        hits_code += code_pred.eq(code_true).sum().item()

        fmt_true = _torch.tensor([lbl["data_format"] for lbl in labels], dtype=_torch.long, device=device)
        hits_fmt += logits["data_format"].argmax(dim=-1).eq(fmt_true).sum().item()

        topic_true = _torch.tensor([lbl["topic"] for lbl in labels], dtype=_torch.long, device=device)
        hits_topic += logits["topic"].argmax(dim=-1).eq(topic_true).sum().item()

        intent_true = _torch.tensor([lbl["intent"] for lbl in labels], dtype=_torch.long, device=device)
        hits_intent += logits["intent"].argmax(dim=-1).eq(intent_true).sum().item()

    denom = max(1, total)
    return {
        "acc_math": hits_math / denom,
        "acc_code": hits_code / denom,
        "acc_fmt": hits_fmt / denom,
        "acc_topic": hits_topic / denom,
        "acc_int": hits_intent / denom,
    }


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------
@dataclass
class CtxPaths:
    train_jsonl: str
    val_jsonl: str
    out_dir: str = "runs/checkpoints/context_core"


@dataclass
class CtxModel:
    d: int = 384
    n_layers: int = 12
    n_heads: int = 6
    tokenizer: str = "tokenizer_local"
    num_topics: int = 16
    num_intents: int = 8
    dropout: float = 0.0
    max_position_embeddings: int = 4096


@dataclass
class CtxRun:
    steps: int = 8000
    batch_size: int = 32
    max_len: int = 256
    lr_peak: float = 3e-4
    warmup: int = 500
    min_lr: float = 2e-5
    weight_decay: float = 0.05
    grad_accum: int = 1
    amp: bool = True
    ema: bool = False
    ema_decay: float = 0.999
    log_every: int = 50
    eval_every: int = 500
    save_every: int = 1000
    device: str = "cuda"
    dtype: str = "fp16"
    num_workers: int = 0
    resume_from: Optional[str] = None


class CosineWithWarmup:
    def __init__(self, opt, peak, warmup, total, min_lr):
        self.opt = opt
        self.peak = peak
        self.warmup = max(1, warmup)
        self.total = max(self.warmup + 1, total)
        self.min_lr = min_lr
        self.step_id = 0

    def step(self) -> float:
        self.step_id += 1
        if self.step_id <= self.warmup:
            lr = self.peak * self.step_id / self.warmup
        else:
            u = (self.step_id - self.warmup) / (self.total - self.warmup)
            lr = self.min_lr + 0.5 * (self.peak - self.min_lr) * (1 + math.cos(math.pi * u))
        for group in self.opt.param_groups:
            group["lr"] = lr
        return lr


class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {name: param.detach().clone() for name, param in model.named_parameters() if param.requires_grad}

    @torch.no_grad()
    def update(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if not param.requires_grad:
                continue
            self.shadow[name] = (1.0 - self.decay) * param.detach() + self.decay * self.shadow[name]

    @torch.no_grad()
    def apply_to(self, model: nn.Module) -> None:
        for name, param in model.named_parameters():
            if name in self.shadow:
                param.copy_(self.shadow[name])


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def train_from_yaml(cfg_path: str) -> None:
    with open(cfg_path, "r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    paths = CtxPaths(**cfg.get("paths", {}))
    model_cfg = CtxModel(**cfg.get("model", {}))
    run_cfg = CtxRun(**cfg.get("run", {}))
    train(paths, model_cfg, run_cfg)


def train(paths: CtxPaths, model_cfg: CtxModel, run_cfg: CtxRun) -> None:
    tokenizer = AutoTokenizer.from_pretrained(model_cfg.tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    vocab_size = len(tokenizer)

    device = torch.device(run_cfg.device if torch.cuda.is_available() or run_cfg.device == "cpu" else "cpu")
    use_cuda = device.type == "cuda"
    dtype_map = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}
    compute_dtype = dtype_map.get(run_cfg.dtype, torch.float16)

    head_dims = ContextHeadDims(num_topics=model_cfg.num_topics, num_intents=model_cfg.num_intents)

    dataset_train = ContextJsonlDataset(paths.train_jsonl, tokenizer, max_len=run_cfg.max_len, head_dims=head_dims)
    dataset_val = ContextJsonlDataset(paths.val_jsonl, tokenizer, max_len=run_cfg.max_len, head_dims=head_dims)

    pad_id = tokenizer.pad_token_id or 0
    loader_train = DataLoader(
        dataset_train,
        batch_size=run_cfg.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=run_cfg.num_workers,
        pin_memory=use_cuda,
        collate_fn=lambda batch: collate(batch, pad_id),
    )
    loader_val = DataLoader(
        dataset_val,
        batch_size=min(run_cfg.batch_size, 64),
        shuffle=False,
        drop_last=False,
        num_workers=0,
        pin_memory=use_cuda,
        collate_fn=lambda batch: collate(batch, pad_id),
    )

    model = ContextCore(
        vocab_size=vocab_size,
        d=model_cfg.d,
        n_layers=model_cfg.n_layers,
        n_heads=model_cfg.n_heads,
        pad_id=pad_id,
        head_dims=head_dims,
        dropout=model_cfg.dropout,
        max_position_embeddings=model_cfg.max_position_embeddings,
    ).to(device)
    model = model.to(dtype=compute_dtype)

    print(
        f"[CTX] d={model_cfg.d} L={model_cfg.n_layers} heads={model_cfg.n_heads} "
        f"dropout={model_cfg.dropout:.3f} params={count_params(model):,} V={vocab_size}"
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=run_cfg.lr_peak,
        weight_decay=run_cfg.weight_decay,
        betas=(0.9, 0.95),
        eps=1e-8,
    )
    scheduler = CosineWithWarmup(
        optimizer,
        peak=run_cfg.lr_peak,
        warmup=run_cfg.warmup,
        total=run_cfg.steps,
        min_lr=run_cfg.min_lr,
    )
    scaler = torch.amp.GradScaler(device=device.type) if (run_cfg.amp and use_cuda) else None
    ema = EMA(model, decay=run_cfg.ema_decay) if run_cfg.ema else None

    loss_fn = build_losses(head_dims)

    train_iter = iter(loader_train)
    for group in optimizer.param_groups:
        group["lr"] = 0.0
    model.train()
    step = 0
    tick = time.time()

    while step < run_cfg.steps:
        optimizer.zero_grad(set_to_none=True)
        try:
            batch_tokens, batch_mask, batch_labels = next(train_iter)
        except StopIteration:
            train_iter = iter(loader_train)
            batch_tokens, batch_mask, batch_labels = next(train_iter)

        batch_tokens = batch_tokens.to(device, non_blocking=True)
        batch_mask = batch_mask.to(device, non_blocking=True)

        autocast_dtype = compute_dtype if (run_cfg.amp and use_cuda) else None
        with torch.autocast(device_type=device.type, dtype=autocast_dtype, enabled=(run_cfg.amp and use_cuda)):
            outputs = model(batch_tokens, attention_mask=batch_mask)["logits"]
            loss = loss_fn(outputs, batch_labels)

        if scaler is not None:
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            optimizer.step()

        current_lr = scheduler.step()
        if ema is not None:
            ema.update(model)

        step += 1

        if step % run_cfg.log_every == 0 or step == 1:
            elapsed = time.time() - tick
            tick = time.time()
            print(f"[step {step:6d}] loss={loss.item():.4f} lr={current_lr:.6g} dt={elapsed:.2f}s")

        if step % run_cfg.eval_every == 0 or step == run_cfg.steps:
            model.eval()
            logits_accum: List[Dict[str, torch.Tensor]] = []
            labels_accum: List[List[Dict[str, Any]]] = []
            with torch.no_grad():
                for val_tokens, val_mask, val_labels in loader_val:
                    val_tokens = val_tokens.to(device, non_blocking=True)
                    val_mask = val_mask.to(device, non_blocking=True)
                    val_out = model(val_tokens, attention_mask=val_mask)["logits"]
                    logits_accum.append({k: v.float().cpu() for k, v in val_out.items()})
                    labels_accum.append(val_labels)
            metrics = compute_eval_metrics(logits_accum, labels_accum)
            print("[eval] " + " ".join(f"{key}={value:.3f}" for key, value in metrics.items()))
            model.train()

        if step % run_cfg.save_every == 0 or step == run_cfg.steps:
            out_dir = Path(paths.out_dir)
            out_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = out_dir / f"context_step{step}.pt"
            torch.save(
                {
                    "model": model.state_dict(),
                    "meta": {
                        "vocab_size": vocab_size,
                        "d": model_cfg.d,
                        "n_layers": model_cfg.n_layers,
                        "n_heads": model_cfg.n_heads,
                    },
                },
                ckpt_path,
            )
            print(f"[save] {ckpt_path}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description="Train the context core encoder from a YAML config")
    parser.add_argument(
        "--config",
        default=str(Path("configs/context_core.yaml")),
        help="Path to the YAML configuration file",
    )
    args = parser.parse_args()
    train_from_yaml(args.config)


if __name__ == "__main__":
    main()
