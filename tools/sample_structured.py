# tools/sample_structured.py
from __future__ import annotations
import sys, argparse, math, torch, importlib.util, types, traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from transformers import AutoTokenizer
from src.titan.core import TitanModel

def load_tokenizer(tok_dir: str):
    tok = AutoTokenizer.from_pretrained(tok_dir)
    if tok.pad_token_id is None:
        tok.add_special_tokens({"pad_token": "<|pad|>"})
    return tok

def ensure_vocab(model, vocab_size: int):
    emb = getattr(model, "embed", None) or getattr(model, "wte", None) or getattr(model, "tok_emb", None)
    head = getattr(model, "head", None) or getattr(model, "lm_head", None)
    if emb is None or head is None: return
    cur = getattr(emb, "num_embeddings", None)
    if cur != vocab_size:
        with torch.no_grad():
            ew = emb.weight.data
            new_e = torch.zeros(vocab_size, ew.size(1), dtype=ew.dtype, device=ew.device)
            new_e[:min(cur, vocab_size)] = ew[:min(cur, vocab_size)]
            emb.weight = torch.nn.Parameter(new_e)
            if hasattr(emb, "num_embeddings"): emb.num_embeddings = vocab_size

            hw = head.weight.data
            new_h = torch.zeros(vocab_size, hw.size(1), dtype=hw.dtype, device=hw.device)
            new_h[:min(hw.size(0), vocab_size)] = hw[:min(hw.size(0), vocab_size)]
            head.weight = torch.nn.Parameter(new_h)
            if hasattr(head, "out_features"): head.out_features = vocab_size

def forward_logits(model, ids: torch.Tensor) -> torch.Tensor:
    out = model(ids)
    return out["logits"] if isinstance(out, dict) else out

def try_load_math_core(no_math_core: bool):
    if no_math_core:
        return None
    mod_path = ROOT / "src" / "titan" / "math_core.py"
    if not mod_path.exists(): return None
    try:
        spec = importlib.util.spec_from_file_location("titan_math_core", mod_path)
        if spec is None or spec.loader is None: return None
        mod = importlib.util.module_from_spec(spec)
        sys.modules[spec.name] = mod           # <-- important for @dataclass
        mod.__file__ = str(mod_path)
        mod.__package__ = "src.titan"
        spec.loader.exec_module(mod)  # type: ignore
        if not hasattr(mod, "solve") or not isinstance(mod.solve, types.FunctionType):
            return None
        return mod
    except Exception:
        print("[math_core] disabled (import error):")
        traceback.print_exc(limit=1)
        return None

@torch.no_grad()
def structured_generate(model, tok, prompt: str, max_new: int = 128, device="cuda", math_mod=None, echo_fallback=True):
    model.eval()

    # 1) Build the preface (this is the line you asked about)
    preface = f"{prompt}\n<think>"
    ids = torch.tensor([tok.encode(preface, add_special_tokens=False)], dtype=torch.long, device=device)

    # 2) Optional deterministic plan/result from math_core
    final_forced = None
    steps = []
    if math_mod is not None:
        try:
            res = math_mod.solve(prompt)
            steps = getattr(res, "steps", []) or []
            final_forced = getattr(res, "result", None)
        except Exception:
            steps, final_forced = [], None

    # 3) *** Echo fallback goes RIGHT HERE ***
    # If math_core didn't provide a result and echo_fallback is True,
    # force the final answer to be the exact prompt.
    if echo_fallback and final_forced is None:
        final_forced = prompt
        if not steps:
            steps = [prompt]

    # 4) Emit steps (if any), then close think / open final
    for line in steps:
        ids = torch.cat([ids, torch.tensor([tok.encode("\n"+line, add_special_tokens=False)], device=device)], dim=1)
    ids = torch.cat([ids, torch.tensor([tok.encode("</think><final>", add_special_tokens=False)], device=device)], dim=1)

    # 5) If we have a forced final (math result or echo), write and return
    if final_forced is not None:
        out_ids = tok.encode(str(final_forced) + "</final>", add_special_tokens=False)
        ids = torch.cat([ids, torch.tensor([out_ids], device=device)], dim=1)
        return tok.decode(ids[0].tolist())

    # 6) Otherwise let the model complete until </final> or budget
    for _ in range(max_new):
        logits = forward_logits(model, ids)[:, -1, :]
        nxt = torch.argmax(logits, dim=-1, keepdim=True)
        ids = torch.cat([ids, nxt], dim=1)
        if tok.decode(ids[0].tolist()).endswith("</final>"):
            break
    return tok.decode(ids[0].tolist())

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--tokenizer", default="tokenizer_local")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--dtype", default="fp16", choices=["fp16","bf16","fp32"])
    ap.add_argument("--max_new", type=int, default=96)
    ap.add_argument("--prompts", nargs="*", default=["Thirteen", "how many r are in strawberry?"])
    ap.add_argument("--d", type=int, default=512)
    ap.add_argument("--layers", type=int, default=36)
    ap.add_argument("--no_math_core", action="store_true")
    ap.add_argument("--no_echo_fallback", action="store_true")
    args = ap.parse_args()

    tok = load_tokenizer(args.tokenizer)
    V = len(tok); print(f"[TOK] {args.tokenizer} vocab={V} ln|V|~{math.log(V):.2f}")

    device = torch.device(args.device if torch.cuda.is_available() and args.device.startswith("cuda") else "cpu")
    dtype = {"fp16": torch.float16, "bf16": torch.bfloat16, "fp32": torch.float32}[args.dtype]
    model = TitanModel(vocab_size=V, d=args.d, n_blocks=args.layers).to(device).to(dtype)

    sd = torch.load(args.ckpt, map_location="cpu")
    sd = sd.get("model", sd)
    ensure_vocab(model, V)
    miss, unexp = model.load_state_dict(sd, strict=False)
    if miss or unexp:
        print(f"[SD] missing={len(miss)} unexpected={len(unexp)}")

    math_mod = try_load_math_core(args.no_math_core)
    echo_fallback = not args.no_echo_fallback

    for p in args.prompts:
        out = structured_generate(model, tok, p, max_new=args.max_new, device=str(device),
                                  math_mod=math_mod, echo_fallback=echo_fallback)
        print(f"\n===  {p}  ===")
        print(out)

if __name__ == "__main__":
    main()

