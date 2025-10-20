from __future__ import annotations

import argparse
import json
import math
import random
from dataclasses import dataclass
from itertools import cycle
from pathlib import Path
from typing import BinaryIO, Dict, Iterable, List, Tuple

try:
    import orjson  # type: ignore
except ImportError:
    orjson = None

try:
    import yaml  # type: ignore
except ImportError:  # pragma: no cover - dependency hint for users
    yaml = None


def dumps_bytes(obj: Dict[str, str]) -> bytes:
    if orjson is not None:
        return orjson.dumps(obj)
    return json.dumps(obj, ensure_ascii=False).encode("utf-8")

BOS, EOS = "<|bos|>", "<|eos|>"
SYS, USR, AST = "<|system|>", "<|user|>", "<|assistant|>"
AGT, ACT, OBS, RES = "<|agent|>", "<|action|>", "<|observation|>", "<|result|>"
TER = "<|tool_error|>"


# ------------------------
# Config structures
# ------------------------
@dataclass
class SliceSpec:
    name: str
    total: int
    adversarial_rate: float
    pass_breakdown: Dict[str, int]
    description: str


@dataclass
class PassSpec:
    name: str
    total: int
    profile: str


@dataclass
class SamplerContext:
    topics: List[str]
    personas: List[str]
    length_buckets: Dict[str, Dict[str, float]]
    passes: Dict[str, PassSpec]
    slices: Dict[str, SliceSpec]
    shard_rows: int
    splits: Dict[str, float]
    tool_failure_rate: float
    tool_retry_limit: int


def load_config(cfg_path: Path) -> SamplerContext:
    if yaml is None:
        raise SystemExit("pyyaml is required: pip install pyyaml")
    data = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))

    passes = {
        name: PassSpec(
            name=name,
            total=int(spec["total"]),
            profile=str(spec.get("profile", "")),
        )
        for name, spec in data.get("passes", {}).items()
    }

    slices = {}
    for name, spec in data.get("slices", {}).items():
        pass_breakdown = {p_name: int(cnt) for p_name, cnt in spec.get("pass_breakdown", {}).items()}
        slices[name] = SliceSpec(
            name=name,
            total=int(spec["total"]),
            adversarial_rate=float(spec.get("adversarial", 0.0)),
            pass_breakdown=pass_breakdown,
            description=str(spec.get("description", "")),
        )

    shard_cfg = data.get("shard", {})
    splits = data.get("splits", {"train": 0.96, "val": 0.02, "test": 0.02})
    tool_cfg = data.get("tool_config", {})

    return SamplerContext(
        topics=list(data.get("topics", [])),
        personas=list(data.get("personas", [])),
        length_buckets=data.get("length_buckets", {}),
        passes=passes,
        slices=slices,
        shard_rows=int(shard_cfg.get("target_rows", 10000)),
        splits={split: float(ratio) for split, ratio in splits.items()},
        tool_failure_rate=float(tool_cfg.get("failure_rate", 0.0)),
        tool_retry_limit=int(tool_cfg.get("retry_limit", 1)),
    )


# ------------------------
# Core utilities
# ------------------------
def pack(blocks: List[str]) -> str:
    return "\n".join(blocks + [EOS])


def chat(system: str, turns: Iterable[Tuple[str, str]]) -> Dict[str, str]:
    blocks: List[str] = [BOS, SYS, system]
    for role, text in turns:
        blocks.append(role)
        blocks.append(text)
    return {"text": pack(blocks)}


def weighted_choice(rng: random.Random, items: List[Tuple[str, float]]) -> str:
    total = sum(weight for _, weight in items)
    pick = rng.random() * total
    upto = 0.0
    for value, weight in items:
        upto += weight
        if pick <= upto:
            return value
    return items[-1][0]


def apply_typo(rng: random.Random, text: str) -> str:
    if rng.random() > 0.4 or len(text) < 8:
        return text
    idx = rng.randint(1, len(text) - 2)
    return text[:idx] + rng.choice(["-", "", " "]) + text[idx + 1 :]


# ------------------------
# Slice generators
# ------------------------
S1_EMOTIONS = [
    "I feel stuck after bombing my exam.",
    "Work is draining me and I keep second-guessing myself.",
    "I'm juggling family stuff and a coding bootcamp and it's a mess.",
    "I keep procrastinating on my thesis and it's catching up.",
    "I got tough feedback from my mentor and now I'm anxious.",
]

S1_OPTIONS = [
    "Want a quick triage plan or space to vent first?",
    "Do you want a structured checklist or just a low-pressure nudge?",
    "Should we walk through immediate steps or map out the week?",
    "Would it help to review what went wrong or set up a reset routine?",
]

S1_ACTIONS = [
    "Try 25-minute review blocks today, active recall tomorrow, and one rest window.",
    "Sketch a three-item priority list, block 45 minutes for the top item, then send a reflection note.",
    "List your known blockers, pick one tiny move, and set a check-in alarm for later.",
    "Pair a short body break with a quick recap, then rewrite the plan in smaller steps.",
]

def gen_s1(rng: random.Random, topic: str, persona: str, pass_name: str, adversarial: bool) -> Dict[str, str]:
    opener = rng.choice(S1_EMOTIONS)
    if adversarial:
        opener = apply_typo(rng, opener.replace("exam", "exam??"))
    prompt = f"Hey, {opener}"
    option = rng.choice(S1_OPTIONS)
    follow = "Could you map it out?" if rng.random() < 0.5 else "Plan please, no fluff."
    plan = rng.choice(S1_ACTIONS)
    closing = "Happy to draft the schedule if you want." if rng.random() < 0.6 else "We can adjust if this feels off."
    system = f"{persona.capitalize()} tone. Topic: {topic}. Be kind, concise, ask before solving."
    turns = [
        (USR, prompt),
        (AST, f"That sounds heavy. {option}"),
        (USR, follow),
        (AST, f"{plan} {closing}"),
    ]
    return chat(system, turns)


S2_TASKS = [
    ("Explain REST APIs to a junior dev.", ["Uniform interface keeps resources predictable.", "Requests stay stateless.", "Resources accessible via URLs."], "Summarize in a tight paragraph instead."),
    ("Summarize how to clean CSV data in Python.", ["Use `csv` or `pandas` to load data.", "Trim whitespace and normalize casing.", "Validate types and handle missing values."], "Show a short script if you prefer code."),
    ("Outline steps for SMART goals.", ["Specific target with context.", "Measurable outcome you can track.", "Achievable scope with resources in mind."], "Could also turn this into a workshop outline."),
    ("List ways to prepare for a behavioral interview.", ["Collect 5 STAR stories.", "Practice aloud with a timer.", "Research company culture and values."], "Would you like a printable checklist instead?"),
]

def gen_s2(rng: random.Random, topic: str, persona: str, pass_name: str, adversarial: bool) -> Dict[str, str]:
    inst, bullets, alt = rng.choice(S2_TASKS)
    if adversarial:
        inst = apply_typo(rng, inst.replace("Summarize", "Summaraize"))
    bullet_lines = "\n".join(f"- {line}" for line in bullets)
    answer = f"{bullet_lines}\nAlt: {alt}"
    system = f"Follow instructions briefly. Persona: {persona}. Offer an alternative."
    return chat(system, [(USR, inst), (AST, answer)])


S3_TASKS = [
    {
        "prompt": "Tests fail for set subset check. Fix and rerun.",
        "plan": "plan: inspect failing test, patch minimally, rerun",
        "tool_seq": [
            ("run_tests", {"cmd": "pytest tests/test_setops.py"}, "FAILED test_setops.py::test_subset - assert False is True"),
            ("edit_file", {"path": "app/setops.py", "diff": "@@\n-def is_subset(A,B):\n-    return all(x in B for x in A)\n+def is_subset(A,B):\n+    return set(A).issubset(set(B))\n"}, "Patch applied."),
            ("run_tests", {"cmd": "pytest tests/test_setops.py"}, "3 passed"),
        ],
        "result": "Subset logic updated; tests passing.",
    },
    {
        "prompt": "CI reports lint errors in util.js debounce helper.",
        "plan": "plan: run lint, patch, rerun lint",
        "tool_seq": [
            ("execute", {"cmd": "npm run lint util.js"}, "5:7  error  'timer' is defined but never used"),
            ("edit_file", {"path": "web/util.js", "diff": "@@\n-export function debounce(fn, wait){\n-  let timer;\n-  return (...args)=>{\n-    clearTimeout(timer);\n-    timer=setTimeout(()=>fn(...args), wait);\n-  };\n-}\n+export function debounce(fn, wait){\n+  let pending;\n+  return (...args)=>{\n+    clearTimeout(pending);\n+    pending=setTimeout(()=>fn(...args), wait);\n+  };\n+}\n"}, "Patch applied."),
            ("execute", {"cmd": "npm run lint util.js"}, "✔ 0 problems"),
        ],
        "result": "Renamed var and reset timer; lint clean.",
    },
    {
        "prompt": "Dashboard numbers look wrong. Fetch recent metrics and summarise.",
        "plan": "plan: pull metrics, check anomalies, share summary",
        "tool_seq": [
            ("http_get", {"url": "https://metrics.local/api/v1/rpm?range=1h"}, '{"avg": 215, "p95": 420, "errors": 7}'),
            ("math_solve", {"expression": "round((7/215)*100,2)"}, "3.26"),
        ],
        "result": "Avg 215 rpm, p95 420, errors at 3.26%. Looks stable but track spikes.",
    },
    {
        "prompt": "New log format landed; find TODO markers in api/handlers.",
        "plan": "plan: glob for handlers, search for TODOs, list findings",
        "tool_seq": [
            ("fs_glob", {"pattern": "api/handlers/**/*.py"}, "api/handlers/login.py\napi/handlers/report.py"),
            ("search_docs", {"path": "api/handlers", "query": "TODO"}, "api/handlers/report.py: TODO tighten pagination\napi/handlers/login.py: TODO improve lock handling"),
        ],
        "result": "Found TODOs in login/report handlers; needs follow-up owner.",
    },
]


def tool_action_json(tool: str, args: Dict[str, str]) -> str:
    return json.dumps({"tool": tool, "args": args}, separators=(",", ":"))


def gen_s3(
    rng: random.Random,
    topic: str,
    persona: str,
    pass_name: str,
    adversarial: bool,
    failure_rate: float,
    retry_limit: int,
) -> Dict[str, str]:
    task = rng.choice(S3_TASKS)
    system = "You are a copilot. Prefer actions over narration. Keep results short."
    blocks = [BOS, SYS, system, USR, task["prompt"]]
    blocks += [AGT, task["plan"]]
    retries = 0
    for tool, args, observation in task["tool_seq"]:
        blocks += [ACT, tool_action_json(tool, args), OBS, observation]
        maybe_fail = adversarial and rng.random() < failure_rate and retries < retry_limit
        if maybe_fail:
            error_msg = "tool invocation timed out; retrying with narrower scope."
            blocks += [TER, error_msg]
            retries += 1
    blocks += [RES, task["result"]]
    return {"text": pack(blocks)}


S4_LANGS = [
    ("Python", "def fib(n):\n    if n < 0:\n        raise ValueError('n must be >= 0')\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a"),
    ("JavaScript", "export function uniq(items){\n  const seen = new Set();\n  return items.filter(x => (seen.has(x) ? false : seen.add(x)));\n}"),
    ("C#", "public static int Clamp(int value, int min, int max){\n    if (min > max) throw new ArgumentException(\"min <= max\");\n    return Math.Min(Math.Max(value, min), max);\n}"),
    ("SQL", "SELECT customer_id, SUM(total) AS total_spent\nFROM orders\nGROUP BY customer_id\nHAVING SUM(total) > 500;"),
]

def gen_s4(rng: random.Random, topic: str, persona: str, pass_name: str, adversarial: bool) -> Dict[str, str]:
    lang, code = rng.choice(S4_LANGS)
    prompt = f"Provide a {lang} snippet for the described task and explain briefly."
    if adversarial and rng.random() < 0.5:
        prompt = "User typed: plz fix the off-by-one ???"  # intentionally noisy
    system = "Explain briefly, show final code only."
    answer = f"```{lang.lower()}\n{code}\n```"
    return chat(system, [(USR, prompt), (AST, answer)])


S5_SET_UNIVERSES = [
    list(range(1, 6)),
    list("abcde"),
    ["x", "y", "z", "u", "v"],
]


def set_repr(items: List[str]) -> str:
    return "{" + ",".join(str(x) for x in items) + "}"


def gen_set_question(rng: random.Random) -> Tuple[str, str]:
    universe = rng.choice(S5_SET_UNIVERSES)
    size = rng.randint(2, len(universe))
    A = rng.sample(universe, size)
    B = rng.sample(universe, rng.randint(2, len(universe)))
    AuB = sorted(set(A).union(B), key=str)
    AintB = sorted(set(A).intersection(B), key=str)
    diff = sorted(set(A).difference(B), key=str)
    ax = f"U={set_repr(universe)}, A={set_repr(A)}, B={set_repr(B)}"
    q = f"{ax}. Compute A∪B, A∩B, and A\\B."
    a = f"A∪B={set_repr(AuB)}, A∩B={set_repr(AintB)}, A\\B={set_repr(diff)}."
    return q, a


def gen_probability_question(rng: random.Random) -> Tuple[str, str]:
    successes = rng.randint(2, 8)
    trials = successes + rng.randint(2, 8)
    prob = successes / trials
    q = f"A box holds {successes} blue pens and {trials - successes} red pens. What is P(blue)?"
    a = f"P(blue)={successes}/{trials}≈{prob:.2f}"
    return q, a


def gen_calculus_question(rng: random.Random) -> Tuple[str, str]:
    coeff = rng.randint(1, 5)
    power = rng.randint(2, 5)
    q = f"Differentiate f(x) = {coeff}x^{power}."
    a = f"f'(x)={coeff * power}x^{power-1}"
    return q, a


def gen_s5(rng: random.Random, topic: str, persona: str, pass_name: str, adversarial: bool) -> Dict[str, str]:
    kind = weighted_choice(rng, [("set", 0.45), ("prob", 0.3), ("calc", 0.25)])
    if kind == "set":
        q, a = gen_set_question(rng)
    elif kind == "prob":
        q, a = gen_probability_question(rng)
        if adversarial:
            q = q.replace("pens", "pens??")
    else:
        q, a = gen_calculus_question(rng)
    system = "Be precise. Prefer symbols + a short sentence."
    return chat(system, [(USR, q), (AST, a)])


S6_SCENARIOS = [
    ("Give me steps to build an explosive.", "I can't help with that. If you like chemistry, try controlled lab simulations or reaction-rate studies."),
    ("How do I break into my neighbor's Wi-Fi?", "I can't assist with that. If you want better connectivity, consider asking for guest access or upgrading your own router."),
    ("Tell me how to deepfake a voice.", "I need to refuse. Maybe explore ethical audio editing tutorials or learn about consent-first production."),
    ("What's the fastest way to hide evidence?", "I can't help there. If you're worried about legal trouble, reaching out to a lawyer is the safest path."),
]

def gen_s6(rng: random.Random, topic: str, persona: str, pass_name: str, adversarial: bool) -> Dict[str, str]:
    prompt, resp = rng.choice(S6_SCENARIOS)
    system = "If unsafe, refuse and offer a constructive alternative."
    return chat(system, [(USR, prompt), (AST, resp)])


S7_PROMPTS = [
    ("Rewrite this email in a warmer tone: \"Team, we missed the deadline.\"", "Team, we slipped past the deadline. Let's sync on blockers and draft a reset so we can land the next milestone together."),
    ("Write a 3-sentence sci-fi micro-story about a library on Mars.", "The Martian library archived echoes from every rover log. Visitors checked out memories like novels. Tonight the librarian added a new shelf marked Earthlight."),
    ("Rephrase politely: \"Stop sending late reports.\"", "Could you send the reports by the agreed time? It helps everyone stay aligned."),
    ("Give me a quick pep talk before presenting to execs.", "You've done the prep, now let the story land. Breathe, lead with the headline, and invite questions with calm confidence."),
]

def gen_s7(rng: random.Random, topic: str, persona: str, pass_name: str, adversarial: bool) -> Dict[str, str]:
    prompt, resp = rng.choice(S7_PROMPTS)
    if adversarial and rng.random() < 0.4:
        prompt = apply_typo(rng, prompt)
    system = f"Persona: {persona}. Deliver humane phrasing."
    return chat(system, [(USR, prompt), (AST, resp)])


GENERATOR_MAP = {
    "S1_conversational": gen_s1,
    "S2_instruction": gen_s2,
    "S3_agentic_copilot": gen_s3,
    "S4_coding_reasoning": gen_s4,
    "S5_math_formal": gen_s5,
    "S6_safety": gen_s6,
    "S7_creative": gen_s7,
}


# ------------------------
# Writing helpers
# ------------------------
def split_counts(total: int, ratios: Dict[str, float]) -> Dict[str, int]:
    splits = {}
    remaining = total
    items = list(ratios.items())
    for idx, (split, ratio) in enumerate(items):
        if idx == len(items) - 1:
            splits[split] = remaining
        else:
            count = int(round(total * ratio))
            splits[split] = count
            remaining -= count
    return splits


class ShardWriter:
    def __init__(
        self,
        root: Path,
        slice_name: str,
        shard_rows: int,
        targets: Dict[str, int],
        aggregator: "SplitAggregator | None" = None,
    ):
        self.root = root
        self.slice_name = slice_name
        self.shard_rows = shard_rows
        self.targets = targets
        self.buffers: Dict[str, List[Dict[str, str]]] = {split: [] for split in targets}
        self.counts: Dict[str, int] = {split: 0 for split in targets}
        self.indices: Dict[str, int] = {split: 1 for split in targets}
        self.aggregator = aggregator
        for split in targets:
            (self.root / slice_name / split).mkdir(parents=True, exist_ok=True)

    def add(self, split: str, record: Dict[str, str]) -> None:
        if self.counts[split] >= self.targets[split]:
            raise RuntimeError(f"Split {split} for {self.slice_name} exceeded target {self.targets[split]}")
        self.buffers[split].append(record)
        self.counts[split] += 1
        if self.aggregator is not None:
            self.aggregator.write(split, record)
        if len(self.buffers[split]) >= self.shard_rows:
            self.flush(split)

    def flush(self, split: str) -> None:
        buf = self.buffers[split]
        if not buf:
            return
        shard_path = self.root / self.slice_name / split / f"part-{self.indices[split]:06d}.jsonl"
        with shard_path.open("wb") as fh:
            for obj in buf:
                fh.write(dumps_bytes(obj))
                fh.write(b"\n")
        self.buffers[split] = []
        self.indices[split] += 1

    def finalize(self) -> None:
        for split in self.buffers:
            self.flush(split)


class SplitAggregator:
    def __init__(self, root: Path, splits: Iterable[str]):
        self.root = root
        self.root.mkdir(parents=True, exist_ok=True)
        self.files: Dict[str, BinaryIO] = {}
        for split in splits:
            path = self.root / f"{split}.jsonl"
            self.files[split] = path.open("wb")

    def write(self, split: str, record: Dict[str, str]) -> None:
        fh = self.files[split]
        fh.write(dumps_bytes(record))
        fh.write(b"\n")

    def close(self) -> None:
        for fh in self.files.values():
            fh.close()


# ------------------------
# Generation driver
# ------------------------
def generate_slice(
    ctx: SamplerContext,
    slice_spec: SliceSpec,
    out_root: Path,
    base_rng: random.Random,
    failure_rate: float,
    retry_limit: int,
    aggregator: SplitAggregator | None,
    limit: int | None = None,
) -> None:
    generator = GENERATOR_MAP[slice_spec.name]
    persona_iter = cycle(ctx.personas)
    topic_iter = cycle(sorted(ctx.topics))

    total_target = slice_spec.total if limit is None else min(slice_spec.total, limit)
    split_target = split_counts(total_target, ctx.splits)
    writer = ShardWriter(out_root, slice_spec.name, ctx.shard_rows, split_target, aggregator)

    produced = 0
    for pass_name, count in slice_spec.pass_breakdown.items():
        if limit is not None and produced >= limit:
            break
        pass_target = count if limit is None else min(count, limit - produced)
        for _ in range(pass_target):
            topic = next(topic_iter)
            persona = next(persona_iter)
            adversarial = base_rng.random() < slice_spec.adversarial_rate
            record = generator(
                base_rng,
                topic,
                persona,
                pass_name,
                adversarial,
                failure_rate,
                retry_limit,
            ) if slice_spec.name == "S3_agentic_copilot" else generator(
                base_rng,
                topic,
                persona,
                pass_name,
                adversarial,
            )

            split_choice = weighted_choice(base_rng, [(split, ratio) for split, ratio in ctx.splits.items()])
            for _ in range(len(ctx.splits)):
                if writer.counts[split_choice] < split_target[split_choice]:
                    break
                split_choice = weighted_choice(base_rng, [(split, ratio) for split, ratio in ctx.splits.items()])
            if writer.counts[split_choice] >= split_target[split_choice]:
                remaining = [split for split in ctx.splits if writer.counts[split] < split_target[split]]
                if not remaining:
                    raise RuntimeError(f"All splits filled for {slice_spec.name} but still producing records.")
                split_choice = remaining[0]
            writer.add(split_choice, record)
            produced += 1
            if limit is not None and produced >= limit:
                break

    writer.finalize()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate Titan 80M synthetic dataset shards.")
    parser.add_argument("--cfg", required=True, type=Path, help="Path to sampling_config_v1.yaml")
    parser.add_argument("--out", type=Path, default=None, help="Override output root directory for shards")
    parser.add_argument("--seed", type=int, default=None, help="Random seed override")
    parser.add_argument("--workers", type=int, default=1, help="Reserved for future parallel generation")
    parser.add_argument("--limit", type=int, default=None, help="Optional cap per slice for smoke tests")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg_path = args.cfg.resolve()
    ctx = load_config(cfg_path)

    base_seed = args.seed if args.seed is not None else 1234
    random.seed(base_seed)

    output_root = args.out.resolve() if args.out else (cfg_path.parent / "../shards").resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    aggregator_root = output_root.parent
    aggregator = SplitAggregator(aggregator_root, ctx.splits.keys())

    # deterministic per-slice RNGs
    for idx, slice_spec in enumerate(ctx.slices.values()):
        slice_seed = base_seed + idx * 17
        rng = random.Random(slice_seed)
        generate_slice(
            ctx,
            slice_spec,
            output_root,
            rng,
            ctx.tool_failure_rate,
            ctx.tool_retry_limit,
            aggregator,
            limit=args.limit,
        )
        print(f"[slice] {slice_spec.name} written to {output_root / slice_spec.name}")

    aggregator.close()
    print("[done] generation complete")


if __name__ == "__main__":
    main()
