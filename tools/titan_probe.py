# tools/titan_probe.py
#!/usr/bin/env python
from __future__ import annotations
import sys, time, math, argparse
from pathlib import Path
import torch
from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.titan.core import TitanModel

def titan_probe(model, batch_example=None):
    n_params = sum(p.numel() for p in model.parameters())
    print(f"params: {n_params:,}  (~{n_params/1e9:.2f}B)")
    p0 = next(model.parameters())
    print("device:", p0.device, "dtype:", p0.dtype)
    if torch.cuda.is_available():
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        alloc = torch.cuda.memory_allocated()/1e6
        resv  = torch.cuda.memory_reserved()/1e6
        print(f"GPU mem (before fwd): allocated={alloc:.0f}MB reserved={resv:.0f}MB")
    if batch_example is not None:
        x = batch_example.to(p0.device, non_blocking=True)
        t0 = time.time()
        with torch.no_grad():
            out = model(x)
        if torch.cuda.is_available(): torch.cuda.synchronize()
        dt = time.time() - t0
        B, T = x.shape
        toks = B * T
        print(f"fwd: {toks/dt:.1f} tok/s  (B={B}, T={T}, Δt={dt:.3f}s)")
    bytes_per_param = 12  # Adam moments estimate
    print(f"Adam mem est: ~{bytes_per_param*n_params/1e9:.1f} GB")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tokenizer", default="tokenizer_local")
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--layers", type=int, default=36)
    ap.add_argument("--seq", type=int, default=128)
    ap.add_argument("--batch", type=int, default=2)
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16", choices=["fp16","bf16","fp32"])
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    if tok.pad_token_id is None: tok.add_special_tokens({"pad_token":"<|pad|>"})
    V = len(tok); print(f"vocab: {V}  ln|V|≈{math.log(V):.2f}")

    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    model = TitanModel(vocab_size=V, d=args.d, n_blocks=args.layers).to(device).to(dtype)

    pad = tok.pad_token_id
    x = torch.full((args.batch, args.seq), pad, dtype=torch.long)
    x[:, :-1] = torch.randint(0, V, (args.batch, args.seq-1))
    titan_probe(model, x)

if __name__ == "__main__":
    main()
