from __future__ import annotations
from dataclasses import dataclass
import re
from typing import Optional

@dataclass
class SolveOut:
    steps: list[str] | None = None
    result: str | None = None
    op: str | None = None

# --- Normalization helpers ---
_UNIT_ALIASES = {
    # volume
    "l": "liter", "L": "liter", "lt": "liter", "litre": "liter", "litres": "liter",
    "liter": "liter", "liters": "liter",
    # length
    "m": "meter", "meter": "meter", "meters": "meter",
    "cm": "centimeter", "centimeter": "centimeter", "centimeters": "centimeter",
    # mass
    "kg": "kilogram", "kilogram": "kilogram", "kilograms": "kilogram",
    "g": "gram", "gram": "gram", "grams": "gram",
}

# base unit and multiplier to base (e.g., cm -> 0.01 meter)
_UNIT_TO_BASE = {
    # volume base = liter
    "liter": ("liter", 1.0),
    # length base = meter
    "meter": ("meter", 1.0),
    "centimeter": ("meter", 0.01),
    # mass base = kilogram
    "kilogram": ("kilogram", 1.0),
    "gram": ("kilogram", 0.001),
}

def norm_unit(u: str) -> Optional[str]:
    u = u.strip()
    return _UNIT_ALIASES.get(u.lower())

def same_dimension(u1: str, u2: str) -> bool:
    """Return True if units share a base dimension (liter/litre/L; m/cm; kg/g)."""
    b1 = _UNIT_TO_BASE.get(u1, (None, None))[0]
    b2 = _UNIT_TO_BASE.get(u2, (None, None))[0]
    return (b1 is not None) and (b1 == b2)

def to_base(u: str, x: float) -> tuple[str, float]:
    base, mul = _UNIT_TO_BASE.get(u, (None, None))
    if base is None: return (u, x)
    return base, x * float(mul)

def from_base(u_base: str, target: str, x_base: float) -> float:
    # find factor to go from base to target
    tb, mul = _UNIT_TO_BASE.get(target, (None, None))
    if tb is None or tb != u_base:  # unknown or mismatched
        return x_base
    # if target factor is e.g., gram (0.001 kg), then base -> target is /0.001
    return x_base / float(mul)

# --- Patterns ---
RE_COUNT_CHAR = re.compile(r"how\s+many\s+([a-zA-Z])\s+(?:are\s+)?in\s+([a-zA-Z]+)\??", re.I)
RE_QTY_UNITS  = re.compile(
    r"how\s+many\s+([a-zA-Z]+)\s+(?:are\s+)?in\s+([0-9]+(?:\.[0-9]+)?)\s+([a-zA-Z]+)\??",
    re.I
)

def _solve_units(target_u_raw: str, qty_raw: str, src_u_raw: str) -> Optional[SolveOut]:
    # normalize units (handles l/L/liter/litre, etc.)
    t = norm_unit(target_u_raw)
    s = norm_unit(src_u_raw)
    if t is None or s is None:
        return None
    try:
        qty = float(qty_raw)
    except ValueError:
        return None

    # Same dimension?
    if not same_dimension(t, s):
        # Different dimensions → cannot convert
        return SolveOut(
            steps=[f"normalize units: '{target_u_raw}'→{t}, '{src_u_raw}'→{s}",
                   "units are incompatible; cannot convert"],
            result="unknown",
            op="unit_incompatible",
        )

    # Convert numeric qty from source unit to target unit
    # Procedure: src -> base, then base -> target
    base_dim, qty_base = to_base(s, qty)
    qty_target = from_base(base_dim, t, qty_base)

    steps = [
        f"normalize units: '{target_u_raw}'→{t}, '{src_u_raw}'→{s}",
        f"convert {qty} {s} → base({base_dim}) = {qty_base:g}",
        f"convert base({base_dim}) → {t} = {qty_target:g}",
    ]
    return SolveOut(
        steps=steps,
        result=f"{qty_target:g}",
        op="unit_convert",
    )

def _solve_count_char(ch: str, word: str) -> SolveOut:
    cnt = sum(1 for c in word if c.lower() == ch.lower())
    return SolveOut(
        steps=[f"lowercase: {word.lower()}",
               f"count('{ch.lower()}') = {cnt}"],
        result=str(cnt),
        op="count_char",
    )

def solve(prompt: str) -> SolveOut:
    s = prompt.strip()

    # 1) Unit quantity questions: "how many l are in 5 litres?"
    # Prefer this branch when a number + two unit tokens appear.
    m = RE_QTY_UNITS.search(s)
    if m:
        target_u, qty, src_u = m.group(1), m.group(2), m.group(3)
        out = _solve_units(target_u, qty, src_u)
        if out is not None:
            return out

    # 2) Character counting: "how many r are in strawberry?"
    m = RE_COUNT_CHAR.search(s)
    if m:
        ch, word = m.group(1), m.group(2)
        return _solve_count_char(ch, word)

    # 3) Fallback: unknown → let LM handle it.
    return SolveOut()
