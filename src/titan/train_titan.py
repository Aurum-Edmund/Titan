# src/titan/train_titan.py
from __future__ import annotations

import os, sys, time, math, json, argparse, random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

# --- project root on path ---
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

# --- Titan model must exist here ---
from src.titan.core import TitanModel  # noqa: E402


# =========================
# Small utils
# =========================
def now() -> str:
    return time.strftime("%Y-%m-%d %H:%M:%S")

def log(msg: str) -> None:
    print(f"[{now()}] {msg}", flush=True)

def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())

def set_seed(seed: Optional[int]):
    if seed is None: return
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# =========================
# BOM-safe JSONL reader
# =========================
def read_jsonl_texts(path: str) -> List[str]:
    """
    Accepts lines like:
      {"text": "..."}
      {"prompt": "...", "completion": "..."}
      {"input": "...", "output": "..."}
    Tolerant to UTF-8 BOM / UTF-16.
    """
    with open(path, "rb") as fb:
        raw = fb.read()
    decoded = None
    for enc in ("utf-8-sig", "utf-16", "utf-8", "latin-1"):
        try:
            decoded = raw.decode(enc)
            break
        except UnicodeDecodeError:
            continue
    if decoded is None:
        raise UnicodeError(f"Could not decode {path}")

    out: List[str] = []
    for line in decoded.splitlines():
        s = line.strip()
        if not s:
            continue
        if s[0] == "\ufeff":
            s = s.lstrip("\ufeff")
        obj = json.loads(s)
        if "text" in obj:
            out.append(obj["text"])
        elif "prompt" in obj and "completion" in obj:
            out.append(f'{obj["prompt"]}\n{obj["completion"]}')
        elif "input" in obj and "output" in obj:
            out.append(f'{obj["input"]}\n{obj["output"]}')
        else:
            raise ValueError(f"Unsupported JSON keys in {path}: {list(obj.keys())}")
    return out


# =========================
# Dataset / Collate
# =========================
class JsonlLM(Dataset):
    def __init__(self, path: str, tok, max_len: int):
        self.path = str(path)
        self.tok = tok
        self.max_len = int(max_len)
        self.lines = read_jsonl_texts(self.path)

    def __len__(self) -> int:
        return len(self.lines)

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        text = self.lines[idx]
        ids = self.tok.encode(text, add_special_tokens=False)
        # truncate
        if len(ids) > self.max_len:
            ids = ids[: self.max_len]
        # ensure at least 2 tokens
        if len(ids) < 2:
            pad_id = self.tok.pad_token_id if self.tok.pad_token_id is not None else 0
            ids = (ids + [pad_id, pad_id])[:2]
        return {"ids": torch.tensor(ids, dtype=torch.long)}

def titan_collate(batch: List[Dict[str, torch.Tensor]], pad_id: int, ignore_index: int):
    T = max(len(b["ids"]) for b in batch)
    B = len(batch)
    x = torch.full((B, T), pad_id, dtype=torch.long)
    y = torch.full((B, T), ignore_index, dtype=torch.long)
    for i, item in enumerate(batch):
        ids = item["ids"]
        t = len(ids)
        x[i, :t] = ids
        if t > 1:
            y[i, : t - 1] = ids[1:]
            y[i, t - 1] = ignore_index
    return x, y


# =========================
# Config (+ aliases)
# =========================
@dataclass
class PathsCfg:
    train_jsonl: str
    val_jsonl: str
    out_dir: str = "runs/checkpoints/titan_v0"

@dataclass
class ModelCfg:
    d: int = 512
    n_blocks: int = 36
    tokenizer: str = "tokenizer_local"
    heads: int = 8
    ff_mult: float = 4.0
    dropout: float = 0.0
    max_position_embeddings: int = 2048

@dataclass
class RunCfg:
    steps: int = 30000
    batch_size: int = 32
    sequence_length: int = 128
    learning_rate: float = 3e-4
    warmup_percent: float = 0.10
    min_lr_percent: float = 0.06
    warmup_steps: Optional[int] = None
    min_learning_rate: Optional[float] = None

    weight_decay: float = 0.1
    grad_accumulation: int = 1
    amp: bool = True
    ema: bool = False
    ema_decay: float = 0.999
    grad_clip: Optional[float] = None
    log_every: int = 50
    eval_every: int = 500
    save_every: int = 2000
    device: str = "cuda"
    dtype: str = "fp16"  # fp16 | bf16 | fp32 (compute dtype policy)
    num_workers: int = 0
    resume_from: Optional[str] = None
    seed: Optional[int] = 1337
    auto_resume: bool = True

def _alias_cfg(dict_cfg: Dict[str, Any]) -> Tuple[PathsCfg, ModelCfg, RunCfg]:
    paths_raw = dict(dict_cfg.get("paths", {}))
    model_raw = dict(dict_cfg.get("model", {}))
    run_raw   = dict(dict_cfg.get("run", {}))

    # paths aliases
    if "train_json_path" in paths_raw and "train_jsonl" not in paths_raw:
        paths_raw["train_jsonl"] = paths_raw.pop("train_json_path")
    if "val_json_path" in paths_raw and "val_jsonl" not in paths_raw:
        paths_raw["val_jsonl"] = paths_raw.pop("val_json_path")
    if "ckpt_dir" in paths_raw:
        if "out_dir" not in paths_raw:
            paths_raw["out_dir"] = paths_raw["ckpt_dir"]
        paths_raw.pop("ckpt_dir", None)
    if "tokenizer_dir" in paths_raw and "tokenizer" not in model_raw:
        model_raw["tokenizer"] = paths_raw.pop("tokenizer_dir")

    # model aliases
    if "d_model" in model_raw and "d" not in model_raw:
        model_raw["d"] = model_raw.pop("d_model")
    if "n_layers" in model_raw and "n_blocks" not in model_raw:
        model_raw["n_blocks"] = model_raw.pop("n_layers")

    # run aliases
    if "max_steps" in run_raw and "steps" not in run_raw:
        run_raw["steps"] = run_raw.pop("max_steps")
    if "lr_peak" in run_raw and "learning_rate" not in run_raw:
        run_raw["learning_rate"] = run_raw.pop("lr_peak")
    if "warmup" in run_raw and "warmup_steps" not in run_raw:
        run_raw["warmup_steps"] = run_raw.pop("warmup")
    if "min_lr" in run_raw and "min_learning_rate" not in run_raw:
        run_raw["min_learning_rate"] = run_raw.pop("min_lr")
    if "grad_accum" in run_raw and "grad_accumulation" not in run_raw:
        run_raw["grad_accumulation"] = run_raw.pop("grad_accum")
    if "mixed_precision" in run_raw and "amp" not in run_raw:
        run_raw["amp"] = run_raw.pop("mixed_precision")
    if "use_ema" in run_raw and "ema" not in run_raw:
        run_raw["ema"] = run_raw.pop("use_ema")
    if "seq_len" in run_raw and "sequence_length" not in run_raw:
        run_raw["sequence_length"] = run_raw.pop("seq_len")

    return PathsCfg(**paths_raw), ModelCfg(**model_raw), RunCfg(**run_raw)


# =========================
# Vocab resize
# =========================
def ensure_vocab(model: nn.Module, V: int) -> None:
    emb = getattr(model, "embed", None) or getattr(model, "tok_emb", None) or getattr(model, "wte", None)
    head = getattr(model, "head", None) or getattr(model, "lm_head", None)
    if emb is None or head is None:
        return
    cur = getattr(emb, "num_embeddings", None)
    if cur == V:
        return
    with torch.no_grad():
        ew = emb.weight.data
        new_e = torch.zeros(V, ew.size(1), dtype=ew.dtype, device=ew.device)
        keep = min(cur or 0, V)
        if keep > 0:
            new_e[:keep] = ew[:keep]
        emb.weight = nn.Parameter(new_e)
        if hasattr(emb, "num_embeddings"):
            emb.num_embeddings = V

        hw = head.weight.data
        new_h = torch.zeros(V, hw.size(1), dtype=hw.dtype, device=hw.device)
        keep_h = min(hw.size(0), V)
        if keep_h > 0:
            new_h[:keep_h] = hw[:keep_h]
        head.weight = nn.Parameter(new_h)
        if hasattr(head, "out_features"):
            head.out_features = V


# =========================
# Scheduler / EMA / CKPT
# =========================
class CosineWithWarmup:
    def __init__(self, opt, peak, warmup, total, min_lr, start_t: int = 0):
        self.opt = opt
        self.peak = float(peak)
        self.warm = max(1, int(warmup))
        self.total = max(self.warm + 1, int(total))
        self.min = float(min_lr)
        self.t = int(start_t)

        lr0 = self._lr_for(self.t + 1)
        for g in self.opt.param_groups:
            g["lr"] = lr0

    def _lr_for(self, step: int) -> float:
        if step <= self.warm:
            return self.peak * step / self.warm
        u = (step - self.warm) / (self.total - self.warm)
        return self.min + 0.5 * (self.peak - self.min) * (1 + math.cos(math.pi * u))

    def step(self) -> float:
        self.t += 1
        lr = self._lr_for(self.t)
        for g in self.opt.param_groups:
            g["lr"] = lr
        return lr

class EMA:
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.decay = decay
        self.shadow = {n: p.detach().clone() for n, p in model.named_parameters() if p.requires_grad}

    @torch.no_grad()
    def update(self, model: nn.Module):
        for n, p in model.named_parameters():
            if not p.requires_grad: continue
            self.shadow[n] = (1.0 - self.decay) * p.detach() + self.decay * self.shadow[n]

    @torch.no_grad()
    def apply_to(self, model: nn.Module):
        for n, p in model.named_parameters():
            if n in self.shadow:
                p.copy_(self.shadow[n])

def save_ckpt(path: str, model: nn.Module, opt, scaler, step: int,
              meta: Dict[str, Any] | None = None, ema: EMA | None = None):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    sd = {
        "model": model.state_dict(),
        "optimizer": opt.state_dict() if opt is not None else None,
        "scaler": scaler.state_dict() if scaler is not None else None,
        "step": int(step),
        "meta": meta or {},
    }
    if ema is not None:
        sd["ema"] = {k: v.detach().cpu() for k, v in ema.shadow.items()}
    torch.save(sd, path)

def load_ckpt(path: str, model: nn.Module, opt=None, scaler=None, ema: EMA | None = None):
    sd = torch.load(path, map_location="cpu")
    missing, unexpected = model.load_state_dict(sd["model"], strict=False)
    if opt is not None and sd.get("optimizer") is not None:
        opt.load_state_dict(sd["optimizer"])
    if scaler is not None and sd.get("scaler") is not None:
        scaler.load_state_dict(sd["scaler"])
    if ema is not None and sd.get("ema") is not None:
        device = next(model.parameters()).device
        ema.shadow = {k: v.to(device) for k, v in sd["ema"].items()}
    return int(sd.get("step", 0)), missing, unexpected, sd.get("meta", {})

def find_latest_checkpoint(out_dir: str) -> Optional[str]:
    p = Path(out_dir)
    if not p.exists():
        return None
    cks = sorted(p.glob("titan_step*.pt"))
    if not cks:
        return None
    def step_of(name: Path) -> int:
        s = name.stem
        try:
            return int(s.replace("titan_step", ""))
        except Exception:
            return -1
    cks = sorted(cks, key=step_of)
    return str(cks[-1]) if cks else None


# =========================
# Eval (val NLL / PPL)
# =========================
def nll_on_loader(model: nn.Module, loader: DataLoader, device: torch.device, ignore_index: int) -> float:
    model.eval()
    ce = nn.CrossEntropyLoss(ignore_index=ignore_index, reduction="sum")
    total_loss = 0.0
    total_tok = 0
    with torch.no_grad():
        for xb, yb in loader:
            xb = xb.to(device, non_blocking=True)
            yb = yb.to(device, non_blocking=True)
            logits = model(xb)
            logits = logits["logits"] if isinstance(logits, dict) else logits
            B, T, V = logits.shape
            loss = ce(logits.view(B * T, V), yb.view(B * T))
            total_loss += loss.item()
            total_tok += (yb != ignore_index).sum().item()
    model.train()
    return (total_loss / max(1, total_tok)) if total_tok > 0 else float("inf")


# =========================
# Trainer
# =========================
def train_from_yaml(cfg_path: str):
    import yaml

    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg_raw = yaml.safe_load(f)

    paths, model_cfg, run_cfg = _alias_cfg(cfg_raw)

    # auto derive warmup & min-lr if not explicitly set
    if run_cfg.warmup_steps is None:
        run_cfg.warmup_steps = max(1, int(round(run_cfg.steps * run_cfg.warmup_percent)))
    if run_cfg.min_learning_rate is None:
        run_cfg.min_learning_rate = float(run_cfg.learning_rate) * float(run_cfg.min_lr_percent)

    set_seed(run_cfg.seed)

    print("Starting Titan training (preset)...")

    # Tokenizer
    tok = AutoTokenizer.from_pretrained(model_cfg.tokenizer)
    if tok.pad_token_id is None:
        tok.add_special_tokens({"pad_token": "<|pad|>"})
    specials = ["<think>", "</think>", "<final>", "</final>"]
    missing_specials = []
    for token in specials:
        tid = tok.convert_tokens_to_ids(token)
        if tid in (-1, getattr(tok, 'unk_token_id', -1)):
            missing_specials.append(token)
    if missing_specials:
        tok.add_special_tokens({"additional_special_tokens": missing_specials})
        print(f"[TOK] added special tokens: {missing_specials}")
    V = len(tok)
    print(f"[TOK] dir={model_cfg.tokenizer} vocab={V} ln|V|~{math.log(V):.2f}")

    # Device + dtype policy
    device = torch.device(run_cfg.device if torch.cuda.is_available() and run_cfg.device.startswith("cuda") else "cpu")
    if run_cfg.dtype == "fp16":
        compute_dtype = torch.float16
        use_amp = (device.type == "cuda") and bool(run_cfg.amp)
        param_dtype = torch.float32     # keep weights FP32; safe with GradScaler
        use_scaler = use_amp
    elif run_cfg.dtype == "bf16":
        compute_dtype = torch.bfloat16
        use_amp = (device.type == "cuda") and bool(run_cfg.amp)
        param_dtype = torch.bfloat16
        use_scaler = False
    else:
        compute_dtype = torch.float32
        use_amp = False
        param_dtype = torch.float32
        use_scaler = False

    if run_cfg.sequence_length > model_cfg.max_position_embeddings:
        raise ValueError(
            f"sequence_length={run_cfg.sequence_length} exceeds max_position_embeddings={model_cfg.max_position_embeddings}"
        )

    # Data
    max_len = int(run_cfg.sequence_length)
    ds_tr = JsonlLM(paths.train_jsonl, tok, max_len=max_len)
    ds_va = JsonlLM(paths.val_jsonl, tok, max_len=max_len)
    pad_id = tok.pad_token_id if tok.pad_token_id is not None else 0
    IGN = -100

    def collate(batch):
        return titan_collate(batch, pad_id=pad_id, ignore_index=IGN)

    dl_tr = DataLoader(ds_tr, batch_size=run_cfg.batch_size, shuffle=True,
                       num_workers=run_cfg.num_workers, pin_memory=(device.type=="cuda"),
                       collate_fn=collate, drop_last=True)
    dl_va = DataLoader(ds_va, batch_size=min(run_cfg.batch_size, 64), shuffle=False,
                       num_workers=0, pin_memory=(device.type=="cuda"),
                       collate_fn=collate, drop_last=False)

    # Model
    model = TitanModel(
        vocab_size=V,
        d=model_cfg.d,
        n_blocks=model_cfg.n_blocks,
        heads=model_cfg.heads,
        ff_mult=model_cfg.ff_mult,
        dropout=model_cfg.dropout,
        max_position_embeddings=model_cfg.max_position_embeddings,
    ).to(device).to(param_dtype)
    ensure_vocab(model, V)
    print(f"[MODEL] d={model_cfg.d} layers={model_cfg.n_blocks} heads={model_cfg.heads} ff_mult={model_cfg.ff_mult} dropout={model_cfg.dropout} params={count_params(model):,}")

    # Optim / Scheduler
    opt = torch.optim.AdamW(model.parameters(),
                            lr=float(run_cfg.learning_rate),
                            weight_decay=float(run_cfg.weight_decay),
                            betas=(0.9, 0.95), eps=1e-8)

    scaler = torch.amp.GradScaler(device.type, enabled=use_scaler)

    # ---- Resume (before scheduler so LR aligns) ----
    start_step = 0
    resume_path = run_cfg.resume_from
    auto_used = False
    if resume_path is None and run_cfg.auto_resume:
        resume_path = find_latest_checkpoint(paths.out_dir)
        auto_used = resume_path is not None

    if resume_path:
        try:
            ss, miss, unexp, meta = load_ckpt(resume_path, model, opt, scaler, ema=None)
            start_step = ss
            print(f"[RESUME] loaded '{resume_path}' @ step {start_step}  "
                  f"missing={len(miss)} unexpected={len(unexp)}  auto={auto_used}")
            # Post-resume echo save so you immediately know saving works
            out_dir = Path(paths.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
            echo_ck = out_dir / f"titan_step{start_step}_resumed.pt"
            save_ckpt(str(echo_ck), model, opt, scaler, start_step,
                      meta={"note": "post-resume echo"}, ema=None)
            print(f"[save] {echo_ck}  (post-resume echo)")
        except Exception as e:
            print(f"[RESUME] failed to load '{resume_path}': {e}")

    # Scheduler (resume-aware)
    sched = CosineWithWarmup(opt,
                             peak=float(run_cfg.learning_rate),
                             warmup=int(run_cfg.warmup_steps),
                             total=int(run_cfg.steps),
                             min_lr=float(run_cfg.min_learning_rate),
                             start_t=start_step)

    steps_rem = max(0, int(run_cfg.steps) - int(start_step))
    print(f"[RUN] steps={run_cfg.steps}  warmup_steps={run_cfg.warmup_steps}  "
          f"lr_peak={run_cfg.learning_rate:.1e}  lr_min={run_cfg.min_learning_rate:.1e}  "
          f"bs={run_cfg.batch_size}  seq_len={run_cfg.sequence_length}  grad_accum={run_cfg.grad_accumulation}  "
          f"param_dtype={str(param_dtype).split('.')[-1]}  compute_dtype={str(compute_dtype).split('.')[-1]}  "
          f"amp={use_amp} scaler={use_scaler}  start_step={start_step}  steps_remaining={steps_rem}")

    # Checkpoint schedule (resume-aware)
    next_save_at = max(0, int(start_step)) + int(run_cfg.save_every)
    print(f"[CHECKPOINTS] out_dir={paths.out_dir}  save_every={run_cfg.save_every}  next_save_at={next_save_at}")

    # Loss
    ce = nn.CrossEntropyLoss(ignore_index=IGN, reduction="mean")
    tokens_per_step = run_cfg.batch_size * run_cfg.sequence_length * run_cfg.grad_accumulation

    log(f"start training on {device.type} (amp={use_amp}, grad_accum={run_cfg.grad_accumulation})")

    step = start_step
    it = iter(dl_tr)

    try:
        while step < run_cfg.steps:
            t0 = time.time()
            opt.zero_grad(set_to_none=True)

            # gradient accumulation
            for acc in range(run_cfg.grad_accumulation):
                try:
                    xb, yb = next(it)
                except StopIteration:
                    it = iter(dl_tr); xb, yb = next(it)

                xb = xb.to(device, non_blocking=True)
                yb = yb.to(device, non_blocking=True)

                with torch.autocast(device_type=device.type, dtype=compute_dtype, enabled=use_amp):
                    logits = model(xb)
                    logits = logits["logits"] if isinstance(logits, dict) else logits
                    B, T, VV = logits.shape
                    loss = ce(logits.view(B * T, VV), yb.view(B * T)) / run_cfg.grad_accumulation

                if use_scaler:
                    scaler.scale(loss).backward()
                else:
                    loss.backward()

            # optional grad clip
            if run_cfg.grad_clip is not None:
                if use_scaler:
                    scaler.unscale_(opt)
                nn.utils.clip_grad_norm_(model.parameters(), run_cfg.grad_clip)

            if use_scaler:
                scaler.step(opt); scaler.update()
            else:
                opt.step()
            lr_now = sched.step()

            # EMA (optional)
            # (Not applied during eval to keep training state simple)
            # if run_cfg.ema:
            #     ema.update(model)

            dt = max(1e-6, time.time() - t0)
            tps = tokens_per_step / dt
            step += 1
            report_loss = float(loss.detach().item()) * run_cfg.grad_accumulation

            if (step % run_cfg.log_every == 0) or (step == start_step + 1):
                pct = 100.0 * step / float(run_cfg.steps)
                tokens_seen = (step - start_step) * tokens_per_step
                print(f"[step {step:6d}/{run_cfg.steps}] loss={report_loss:.4f}  lr={lr_now:.1e}  "
                      f"tok/s={tps:,.0f}  tokens_seen≈{tokens_seen:,}  progress={pct:5.1f}%")

            if (step % run_cfg.eval_every == 0) or (step == run_cfg.steps):
                val_nll = nll_on_loader(model, dl_va, device, IGN)
                val_ppl = math.exp(val_nll) if val_nll < 20 else float("inf")
                print(f"[eval @ step {step:6d}/{run_cfg.steps}] val_nll={val_nll:.6f}  val_ppl={val_ppl:.3f}")

            # periodic save (resume-aware, no modulo traps)
            if step >= next_save_at:
                out_dir = Path(paths.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
                ck = out_dir / f"titan_step{step}.pt"
                meta = {
                    "vocab_size": V,
                    "d_model": model_cfg.d,
                    "n_layers": model_cfg.n_blocks,
                    "paths": asdict(paths),
                    "run": {**asdict(run_cfg), "dtype": run_cfg.dtype},
                }
                try:
                    save_ckpt(str(ck), model, opt, scaler, step, meta=meta, ema=None)
                    print(f"[save] {ck}  (step {step}/{run_cfg.steps})")
                except OSError as e:
                    print(f"[save][ERROR] {e}  (free space? permissions?)")
                next_save_at += int(run_cfg.save_every)
                print(f"[CHECKPOINTS] next_save_at→{next_save_at}")

        # final save
        out_dir = Path(paths.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
        final_ck = out_dir / "titan_final.pt"
        meta = {"vocab_size": V, "d_model": model_cfg.d, "n_layers": model_cfg.n_blocks}
        save_ckpt(str(final_ck), model, opt, scaler, step, meta=meta, ema=None)
        print(f"[done] {final_ck}")

    finally:
        # safety save on exit/error
        try:
            out_dir = Path(paths.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
            safe_ck = out_dir / f"titan_step{step}_safety.pt"
            save_ckpt(str(safe_ck), model, opt, scaler, step,
                      meta={"note": "safety save in finally"}, ema=None)
            print(f"[save] {safe_ck}  (safety)")
        except Exception as e:
            print(f"[save][ERROR] safety save failed: {e}")


# =========================
# CLI
# =========================
if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True, help="YAML config path")
    args = ap.parse_args()
    train_from_yaml(args.config)
