from __future__ import annotations

import argparse
import math
import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[2]
import sys
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.titan.core import TitanModel  # noqa: E402
from src.titan.train_titan import (  # noqa: E402
    CosineWithWarmup,
    JsonlLM,
    count_params,
    ensure_vocab,
    nll_on_loader,
    titan_collate,
)

try:
    import bitsandbytes as bnb  # type: ignore
    HAS_BNB = True
except ImportError:
    HAS_BNB = False


@dataclass
class PathsCfg:
    train_jsonl: str
    val_jsonl: Optional[str] = None
    out_dir: str = "runs/checkpoints/titan_lora"


@dataclass
class ModelCfg:
    tokenizer: str = "tokenizer_local"
    d: Optional[int] = None
    n_layers: Optional[int] = None


@dataclass
@dataclass
class RunCfg:
    steps: int = 1000
    batch_size: int = 16
    sequence_length: int = 256
    learning_rate: float = 1e-4
    warmup_percent: float = 0.10
    min_lr_percent: float = 0.06
    warmup_steps: Optional[int] = None
    min_learning_rate: Optional[float] = None
    grad_accumulation: int = 1
    device: str = "cuda"
    dtype: str = "fp16"  # fp16|bf16|fp32
    amp: bool = True
    grad_clip: Optional[float] = None
    log_every: int = 50
    eval_every: int = 200
    save_every: int = 200
    num_workers: int = 0
    eval_limit: Optional[int] = None
    resume_from: Optional[str] = None


@dataclass
class LoraCfg:
    rank: int = 16
    alpha: int = 32
    dropout: float = 0.05
    quantize: bool = True


IGNORE_INDEX = -100


def _dtype_from_str(name: str) -> torch.dtype:
    mapping = {
        "fp16": torch.float16,
        "bf16": torch.bfloat16,
        "fp32": torch.float32,
    }
    if name not in mapping:
        raise ValueError(f"Unknown dtype '{name}' (expected one of {list(mapping)})")
    return mapping[name]



class LoRALinear(nn.Module):
    def __init__(
        self,
        linear: nn.Linear,
        rank: int,
        alpha: int,
        dropout: float,
        quantize: bool,
    ) -> None:
        super().__init__()
        if rank <= 0:
            raise ValueError("LoRA rank must be positive")
        self.in_features = linear.in_features
        self.out_features = linear.out_features
        self.rank = rank
        self.scaling = alpha / rank
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        self.quantized = False

        if quantize and HAS_BNB:
            try:
                q_linear = bnb.nn.Linear4bit(
                    self.in_features,
                    self.out_features,
                    bias=linear.bias is not None,
                    compute_dtype=linear.weight.dtype if linear.weight.is_floating_point() else torch.float16,
                )
                q_linear.load_state_dict(
                    {
                        "weight": linear.weight,
                        **({"bias": linear.bias} if linear.bias is not None else {}),
                    }
                )
                self.linear = q_linear
                self.quantized = True
            except Exception as exc:  # pragma: no cover - best effort
                print(f"[LoRA] Warning: failed to quantize linear layer ({exc}). Falling back to FP weights.")
                self.linear = linear
        else:
            if quantize and not HAS_BNB:
                print("[LoRA] bitsandbytes not available; using FP weights instead of 4-bit quantization.")
            self.linear = linear

        for param in self.linear.parameters():
            param.requires_grad = False

        self.lora_A = nn.Parameter(torch.zeros(rank, self.in_features))
        self.lora_B = nn.Parameter(torch.zeros(self.out_features, rank))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

        # marker for easy filtering
        self.lora_A._is_lora_param = True  # type: ignore[attr-defined]
        self.lora_B._is_lora_param = True  # type: ignore[attr-defined]

    @property
    def weight(self) -> torch.Tensor:
        return getattr(self.linear, "weight")

    @property
    def bias(self) -> Optional[torch.Tensor]:
        return getattr(self.linear, "bias", None)

    def forward(self, x: torch.Tensor) -> torch.Tensor:  # type: ignore[override]
        result = self.linear(x)
        lora = x @ self.lora_A.t()
        lora = self.dropout(lora) @ self.lora_B.t()
        return result + self.scaling * lora


def inject_lora(model: nn.Module, rank: int, alpha: int, dropout: float, quantize: bool) -> List[nn.Parameter]:
    lora_params: List[nn.Parameter] = []

    def _replace(module: nn.Module) -> None:
        for name, child in list(module.named_children()):
            _replace(child)
            if isinstance(child, nn.Linear) and child.out_features >= rank:
                lora_layer = LoRALinear(child, rank, alpha, dropout, quantize)
                setattr(module, name, lora_layer)
                lora_params.extend([lora_layer.lora_A, lora_layer.lora_B])

    _replace(model)
    return lora_params



def get_lora_state_dict(model: nn.Module) -> Dict[str, torch.Tensor]:
    state: Dict[str, torch.Tensor] = {}
    for name, param in model.named_parameters():
        if "lora_" in name:
            state[name] = param.detach().cpu()
    for name, buf in model.named_buffers():
        if "lora_" in name:
            state[name] = buf.detach().cpu()
    return state


def save_lora_checkpoint(
    out_dir: Path,
    step: int,
    base_ckpt: str,
    cfg_path: str,
    lora_cfg: LoraCfg,
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer] = None,
    scaler: Optional[object] = None,
    tokens_seen: Optional[int] = None,
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"titan_lora_step{step}.pt"
    payload: Dict[str, Any] = {
        "base_checkpoint": base_ckpt,
        "config_path": cfg_path,
        "lora_state": get_lora_state_dict(model),
        "lora_hyper": vars(lora_cfg),
        "run_step": step,
    }
    if optimizer is not None:
        payload["optimizer"] = optimizer.state_dict()
    if scaler is not None and hasattr(scaler, "state_dict"):
        payload["scaler"] = scaler.state_dict()  # type: ignore[assignment]
    if tokens_seen is not None:
        payload["tokens_seen"] = tokens_seen
    torch.save(payload, out_path)
    return out_path


def load_config(cfg_path: str) -> Dict[str, Dict[str, object]]:
    import yaml

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def parse_config(cfg: Dict[str, Dict[str, object]]):
    paths = PathsCfg(**cfg.get("paths", {}))
    model_cfg = ModelCfg(**cfg.get("model", {}))
    run_cfg = RunCfg(**cfg.get("run", {}))
    lora_cfg = LoraCfg(**cfg.get("lora", {}))
    return paths, model_cfg, run_cfg, lora_cfg



def _materialize_text_jsonl(path: str) -> str:
    src = Path(path)
    with src.open('r', encoding='utf-8') as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            if 'text' in obj:
                return str(src)
            if 'prompt' in obj and 'response' in obj:
                flatten_path = src.with_name(src.stem + '_text.jsonl')
                if not flatten_path.exists() or flatten_path.stat().st_mtime < src.stat().st_mtime:
                    with src.open('r', encoding='utf-8') as fin, flatten_path.open('w', encoding='utf-8') as fout:
                        for raw in fin:
                            raw = raw.strip()
                            if not raw:
                                continue
                            data = json.loads(raw)
                            if 'prompt' in data and 'response' in data:
                                combined = f"{data['prompt'].rstrip()}\n{data['response'].lstrip()}"
                                fout.write(json.dumps({'text': combined}, ensure_ascii=False) + '\n')
                            elif 'text' in data:
                                fout.write(json.dumps({'text': data['text']}, ensure_ascii=False) + '\n')
                            else:
                                raise ValueError(f"Unsupported JSON keys in {src}: {list(data.keys())}")
                return str(flatten_path)
            raise ValueError(f"Unsupported JSON keys in {src}: {list(obj.keys())}")
    raise ValueError(f"Empty dataset: {src}")


def build_dataloaders(
    paths: PathsCfg,
    tokenizer,
    seq_len: int,
    batch_size: int,
    num_workers: int,
    device: torch.device,
    eval_limit: Optional[int] = None,
) -> tuple[DataLoader, Optional[DataLoader], int, int]:
    pad_id = tokenizer.pad_token_id or 0
    train_path = _materialize_text_jsonl(paths.train_jsonl)
    ds_tr = JsonlLM(train_path, tokenizer, max_len=seq_len)
    dl_tr = DataLoader(
        ds_tr,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=lambda batch: titan_collate(batch, pad_id=pad_id, ignore_index=IGNORE_INDEX),
    )

    eval_examples = 0
    dl_va = None
    if paths.val_jsonl:
        val_path = _materialize_text_jsonl(paths.val_jsonl)
        ds_va = JsonlLM(val_path, tokenizer, max_len=seq_len)
        eval_examples = len(ds_va)
        limit_int = None
        if eval_limit is not None:
            try:
                limit_int = int(eval_limit)
            except (TypeError, ValueError):
                limit_int = None
        if limit_int is not None and limit_int > 0:
            limit = min(limit_int, eval_examples)
            ds_va = Subset(ds_va, list(range(limit)))
            eval_examples = limit
        dl_va = DataLoader(
            ds_va,
            batch_size=min(batch_size, 64),
            shuffle=False,
            drop_last=False,
            num_workers=0,
            pin_memory=device.type == "cuda",
            collate_fn=lambda batch: titan_collate(batch, pad_id=pad_id, ignore_index=IGNORE_INDEX),
        )
    return dl_tr, dl_va, pad_id, eval_examples


def train(
    cfg_path: str,
    base_ckpt: str,
    out_dir_override: Optional[str],
    resume_lora: Optional[str] = None,
) -> None:
    cfg = load_config(cfg_path)
    paths, model_cfg, run_cfg, lora_cfg = parse_config(cfg)

    out_dir = Path(out_dir_override or paths.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    resume_payload: Optional[Dict[str, object]] = None
    resume_step = 0

    def _load_lora_checkpoint(path: Path) -> tuple[str, Dict[str, object], int]:
        payload = torch.load(path, map_location="cpu")
        stem = path.stem
        fallback = 0
        if "_step" in stem:
            try:
                fallback = int(stem.split("_step")[-1])
            except ValueError:
                fallback = 0
        step = int(payload.get("run_step", fallback))
        return str(path), payload, step

    manual_resume = getattr(run_cfg, "resume_from", None) or resume_lora
    if manual_resume:
        target = Path(manual_resume)
        if target.exists():
            try:
                resume_lora, resume_payload, resume_step = _load_lora_checkpoint(target)
                print(f"[resume] using explicit checkpoint {resume_lora}")
            except Exception as exc:
                print(f"[resume] failed to load {target} ({exc}); starting fresh")
                resume_lora = None
                resume_payload = None
                resume_step = 0
                manual_resume = None
        else:
            print(f"[resume] specified checkpoint {target} not found; starting fresh")
            resume_lora = None
            manual_resume = None

    if manual_resume is None:
        def _step_from_path(p: Path) -> int:
            stem = p.stem
            if "_step" in stem:
                try:
                    return int(stem.split("_step")[-1])
                except ValueError:
                    return -1
            return -1

        candidates = sorted(out_dir.glob("titan_lora_step*.pt"), key=_step_from_path, reverse=True)
        for candidate in candidates:
            try:
                payload = torch.load(candidate, map_location="cpu")
            except Exception as exc:
                print(f"[resume] skipping {candidate} ({exc})")
                continue
            hyper = payload.get("lora_hyper") or {}
            if hyper.get("rank") == lora_cfg.rank and hyper.get("alpha") == lora_cfg.alpha:
                resume_lora = str(candidate)
                resume_payload = payload
                resume_step = int(payload.get("run_step", _step_from_path(candidate)))
                print(f"[resume] auto-detected {resume_lora}")
                break
        else:
            if candidates:
                print("[resume] found LoRA checkpoints but rank/alpha mismatch; starting fresh")
                resume_lora = None

    if resume_lora and resume_payload is None:
        try:
            resume_lora, resume_payload, resume_step = _load_lora_checkpoint(Path(resume_lora))
        except Exception as exc:
            print(f"[resume] failed to load {resume_lora} ({exc}); starting fresh")
            resume_lora = None
            resume_payload = None
            resume_step = 0



    tokenizer = AutoTokenizer.from_pretrained(model_cfg.tokenizer)
    if tokenizer.pad_token_id is None:
        tokenizer.add_special_tokens({"pad_token": "<|pad|>"})
    vocab_size = len(tokenizer)

    device = torch.device(run_cfg.device if torch.cuda.is_available() or run_cfg.device == "cpu" else "cpu")
    compute_dtype = _dtype_from_str(run_cfg.dtype)
    use_amp = run_cfg.amp and device.type == "cuda" and compute_dtype in (torch.float16, torch.bfloat16)

    dl_tr, dl_va, _, eval_examples = build_dataloaders(
        paths,
        tokenizer,
        seq_len=run_cfg.sequence_length,
        batch_size=run_cfg.batch_size,
        num_workers=run_cfg.num_workers,
        device=device,
        eval_limit=run_cfg.eval_limit,
    )
    if dl_va is None or eval_examples == 0:
        print("[eval] validation disabled (no val dataloader)")
    else:
        batch_sz = max(1, min(run_cfg.batch_size, 64))
        val_batches = math.ceil(eval_examples / batch_sz)
        limit_descr = run_cfg.eval_limit if run_cfg.eval_limit else "full"
        print(f"[eval] validation examples={eval_examples} batches={val_batches} (limit={limit_descr})")

    ckpt = torch.load(base_ckpt, map_location="cpu")
    base_state = ckpt.get("model", ckpt)
    meta = ckpt.get("meta", {})
    d_model = model_cfg.d or meta.get("d_model")
    n_layers = model_cfg.n_layers or meta.get("n_layers")
    if d_model is None or n_layers is None:
        raise ValueError("Could not infer model dimensions from base checkpoint; specify model.d and model.n_layers in the config")

    model = TitanModel(vocab_size=vocab_size, d=d_model, n_blocks=n_layers)
    ensure_vocab(model, vocab_size)
    missing, unexpected = model.load_state_dict(base_state, strict=False)
    if missing or unexpected:
        print(f"[base] Warning: missing={len(missing)} unexpected={len(unexpected)} while loading base checkpoint")

    for param in model.parameters():
        param.requires_grad = False

    lora_params = inject_lora(model, lora_cfg.rank, lora_cfg.alpha, lora_cfg.dropout, lora_cfg.quantize)
    if not lora_params:
        raise RuntimeError("No LoRA parameters were created; check rank setting")

    if resume_payload and "lora_state" in resume_payload:
        adapter_state = resume_payload.get("lora_state", {})
        missing, unexpected = model.load_state_dict(adapter_state, strict=False)
        if missing or unexpected:
            print(f"[resume] LoRA load missing={len(missing)} unexpected={len(unexpected)} (ignored)")
        print(f"[resume] loaded LoRA state from {resume_lora} (step {resume_step})")
    elif resume_lora and resume_payload is None:
        print(f"[resume] requested {resume_lora} but failed to load; continuing without adapters")

    model.to(device)
    model.train()

    total_steps = run_cfg.steps
    tokens_per_step = run_cfg.batch_size * run_cfg.sequence_length * run_cfg.grad_accumulation
    tokens_seen = resume_payload.get("tokens_seen", resume_step * tokens_per_step) if resume_payload else resume_step * tokens_per_step

    print(
        f"[LoRA] rank={lora_cfg.rank} alpha={lora_cfg.alpha} dropout={lora_cfg.dropout} "
        f"quantized={'yes' if lora_cfg.quantize and HAS_BNB else 'no'}"
    )
    print(f"[MODEL] base params={count_params(model):,}  trainable LoRA params={sum(p.numel() for p in lora_params):,}")

    if resume_step >= total_steps:
        print(f"[resume] target steps already reached ({resume_step}/{total_steps}); nothing to do.")
        return

    optimizer = torch.optim.AdamW(lora_params, lr=run_cfg.learning_rate, betas=(0.9, 0.999), weight_decay=0.0)
    if resume_payload and "optimizer" in resume_payload:
        try:
            optimizer.load_state_dict(resume_payload["optimizer"])  # type: ignore[arg-type]
        except Exception as exc:
            print(f"[resume] failed to restore optimizer state ({exc}); continuing with fresh optimizer")

    warmup_steps = run_cfg.warmup_steps if run_cfg.warmup_steps is not None else max(1, int(run_cfg.steps * run_cfg.warmup_percent))
    min_lr = run_cfg.min_learning_rate if run_cfg.min_learning_rate is not None else run_cfg.learning_rate * run_cfg.min_lr_percent
    min_lr = float(max(min_lr, 1e-8))
    scheduler = CosineWithWarmup(
        optimizer,
        peak=run_cfg.learning_rate,
        warmup=warmup_steps,
        total=run_cfg.steps,
        min_lr=min_lr,
        start_t=resume_step,
    )
    print(f"[sched] warmup_steps={warmup_steps} min_lr={min_lr:.2e}")

    use_scaler = use_amp and compute_dtype == torch.float16
    scaler = torch.amp.GradScaler(device=device.type) if use_scaler else None
    if scaler and resume_payload and "scaler" in resume_payload:
        try:
            scaler.load_state_dict(resume_payload["scaler"])  # type: ignore[arg-type]
        except Exception as exc:
            print(f"[resume] failed to restore GradScaler state ({exc}); continuing unscaled")
            scaler = torch.amp.GradScaler(device=device.type)

    if resume_payload is None:
        save_path = save_lora_checkpoint(out_dir, resume_step, base_ckpt, cfg_path, lora_cfg, model, optimizer, scaler, tokens_seen)
        print(f"[save] {save_path}  (step {resume_step}/{total_steps})")

    dl_tr_iter = iter(dl_tr)
    last_log_time = time.time()
    last_log_step = resume_step
    lr_now = optimizer.param_groups[0].get("lr", run_cfg.learning_rate)

    for step in range(resume_step + 1, total_steps + 1):
        optimizer.zero_grad(set_to_none=True)
        accum_losses = []
        for _ in range(run_cfg.grad_accumulation):
            try:
                xb, yb = next(dl_tr_iter)
            except StopIteration:
                dl_tr_iter = iter(dl_tr)
                xb, yb = next(dl_tr_iter)
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)

            with torch.autocast(device_type=device.type, dtype=compute_dtype, enabled=use_amp):
                outputs = model(xb, labels=yb)
                loss = outputs["loss"] / run_cfg.grad_accumulation
            accum_losses.append(loss.detach().item())

            if scaler:
                scaler.scale(loss).backward()
            else:
                loss.backward()

        if run_cfg.grad_clip is not None:
            nn.utils.clip_grad_norm_(lora_params, run_cfg.grad_clip)

        if scaler:
            scaler.step(optimizer)
            scaler.update()
        else:
            optimizer.step()

        lr_now = scheduler.step()
        tokens_seen += tokens_per_step

        should_log = run_cfg.log_every and (step - last_log_step) >= run_cfg.log_every
        if step == resume_step + 1 or should_log:
            elapsed = time.time() - last_log_time
            steps_processed = max(1, step - last_log_step)
            tokens_processed = tokens_per_step * steps_processed
            tok_per_s = tokens_processed / max(elapsed, 1e-6)
            progress = 100.0 * step / total_steps
            mean_loss = sum(accum_losses) / len(accum_losses)
            print(
                f"[step {step:6d}/{total_steps}] loss={mean_loss:.4f} lr={lr_now:.2e} "
                f"tok/s={tok_per_s:,.0f} progress={progress:5.1f}% tokens={tokens_seen:,}"
            )
            last_log_time = time.time()
            last_log_step = step

        should_save = ((run_cfg.save_every and run_cfg.save_every > 0 and step % run_cfg.save_every == 0) or step == total_steps)
        if should_save:
            save_path = save_lora_checkpoint(out_dir, step, base_ckpt, cfg_path, lora_cfg, model, optimizer, scaler, tokens_seen)
            print(f"[save] {save_path}  (step {step}/{total_steps})")

        should_eval = False
        if dl_va is not None:
            if run_cfg.eval_every:
                should_eval = (step % run_cfg.eval_every == 0) or (step == total_steps)
            else:
                should_eval = step == total_steps
        if should_eval:
            print(f"[eval] starting validation at step {step:6d}/{total_steps} ...")
            if device.type == "cuda":
                torch.cuda.synchronize(device=device)
            model.eval()
            val_nll = nll_on_loader(model, dl_va, device, IGNORE_INDEX)
            val_ppl = math.exp(val_nll) if val_nll < 20 else float("inf")
            print(f"[eval @ step {step:6d}/{total_steps}] val_nll={val_nll:.6f}  val_ppl={val_ppl:.3f}")
            model.train()
            if device.type == "cuda":
                torch.cuda.synchronize(device=device)


def main() -> None:
    parser = argparse.ArgumentParser(description="Fine-tune Titan using (Q)LoRA adapters")
    parser.add_argument("--config", required=True, help="YAML config describing paths/run/lora settings")
    parser.add_argument("--base-ckpt", required=True, help="Path to the base Titan checkpoint")
    parser.add_argument("--out-dir", default=None, help="Override output directory")
    parser.add_argument("--resume-lora", default=None, help="Optional LoRA checkpoint to resume from")
    args = parser.parse_args()

    train(cfg_path=args.config, base_ckpt=args.base_ckpt, out_dir_override=args.out_dir, resume_lora=args.resume_lora)


if __name__ == "__main__":
    main()
