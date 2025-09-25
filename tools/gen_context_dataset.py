from __future__ import annotations

import argparse
import json
import random
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, List, Tuple

TOPICS = {
    "general": 0,
    "code": 1,
    "math": 2,
    "data": 3,
    "science": 4,
    "finance": 5,
    "education": 6,
    "chat": 7,
}

INTENTS = {
    "statement": 0,
    "question": 1,
    "instruction": 2,
    "analysis": 3,
    "explanation": 4,
    "code_request": 5,
    "data_request": 6,
    "casual": 7,
}

@dataclass
class ContextSample:
    text: str
    labels: Dict[str, float | int]


def _base_labels() -> Dict[str, float | int]:
    return {
        "math_scent": 0.0,
        "code_scent": 0.0,
        "data_format": 0,
        "topic": TOPICS["general"],
        "intent": INTENTS["statement"],
        "route_math": 0.0,
        "route_code": 0.0,
        "route_lang": 1.0,
    }


def math_problem(rng: random.Random) -> ContextSample:
    a = rng.randint(12, 999)
    b = rng.randint(12, 999)
    op = rng.choice(["+", "-", "*", "/"])
    unknown = rng.choice(["x", "n", "k"])
    text = (
        f"Solve for {unknown}: {a} {op} {unknown} = {b}. "
        "Provide the exact numeric value and justify each algebraic step."
    )
    labels = _base_labels()
    labels.update({
        "math_scent": 1.0,
        "topic": TOPICS["math"],
        "intent": INTENTS["question"],
        "route_math": 1.0,
        "route_lang": 0.0,
    })
    return ContextSample(text=text, labels=labels)


def math_analysis(rng: random.Random) -> ContextSample:
    nums = [rng.randint(10, 200) for _ in range(4)]
    formula = rng.choice([
        "variance = sum((x - mean)^2) / n",
        "det(A) = ad - bc",
        "sin^2(theta) + cos^2(theta) = 1",
    ])
    text = (
        "Dataset: "
        + ", ".join(str(n) for n in nums)
        + "\nCompute the sample mean and explain how the formula "
        + formula
        + " applies to this set."
    )
    labels = _base_labels()
    labels.update({
        "math_scent": 1.0,
        "topic": TOPICS["education"],
        "intent": INTENTS["analysis"],
        "route_math": 1.0,
        "route_lang": 0.0,
    })
    return ContextSample(text=text, labels=labels)


def math_units_instruction(rng: random.Random) -> ContextSample:
    quantity = rng.choice(["force", "distance", "energy", "pressure"])
    value = round(rng.uniform(2.5, 97.5), 2)
    unit_in, unit_out = rng.choice([
        ("newtons", "pound-force"),
        ("meters", "feet"),
        ("joules", "kilowatt-hours"),
        ("kilopascals", "atmospheres"),
    ])
    text = (
        f"Convert {value} {unit_in} of {quantity} into {unit_out}. "
        "Lay out the conversion factor and present the final value with proper units."
    )
    labels = _base_labels()
    labels.update({
        "math_scent": 1.0,
        "topic": TOPICS["science"],
        "intent": INTENTS["instruction"],
        "route_math": 1.0,
        "route_lang": 0.0,
    })
    return ContextSample(text=text, labels=labels)


def latex_derivation(rng: random.Random) -> ContextSample:
    identity = rng.choice([
        "int e^{ax} dx = (1/a) * e^{ax} + C",
        "d/dx ln(x) = 1/x",
        "div (curl F) = 0",
    ])
    text = (
        "Consider the identity:\n  "
        + identity
        + "\nExplain its meaning and outline one application where it is essential."
    )
    labels = _base_labels()
    labels.update({
        "math_scent": 1.0,
        "topic": TOPICS["education"],
        "intent": INTENTS["explanation"],
        "route_math": 1.0,
        "route_lang": 0.0,
    })
    return ContextSample(text=text, labels=labels)


def code_snippet(rng: random.Random) -> ContextSample:
    lang = rng.choice(["python", "c++", "java"])
    if lang == "python":
        snippet = (
            "def sum_pairs(nums):\n"
            "    total = 0\n"
            "    for a, b in nums:\n"
            "        total += a + b\n"
            "    return total"
        )
    elif lang == "c++":
        snippet = (
            "int sum_pairs(const vector<pair<int,int>>& nums) {\n"
            "    int total = 0;\n"
            "    for (auto &p : nums) total += p.first + p.second;\n"
            "    return total;\n"
            "}"
        )
    else:
        snippet = (
            "public static int sumPairs(List<int[]> nums) {\n"
            "    int total = 0;\n"
            "    for (int[] pair : nums) total += pair[0] + pair[1];\n"
            "    return total;\n"
            "}"
        )
    text = (
        "The following function aggregates pairwise values:\n"
        + snippet
        + "\nExplain the time complexity and identify edge cases."
    )
    labels = _base_labels()
    labels.update({
        "code_scent": 1.0,
        "topic": TOPICS["code"],
        "intent": INTENTS["analysis"],
        "route_code": 1.0,
        "route_lang": 0.0,
    })
    return ContextSample(text=text, labels=labels)


def code_question(rng: random.Random) -> ContextSample:
    text = (
        "Why does this snippet print unexpected values?\n"
        "for (int i = 0; i < n; ++i) {\n"
        "    printf(\"%d\\n\", arr[i]);\n"
        "}\n"
        "Describe two debugging strategies."
    )
    labels = _base_labels()
    labels.update({
        "code_scent": 1.0,
        "topic": TOPICS["code"],
        "intent": INTENTS["question"],
        "route_code": 1.0,
        "route_lang": 0.0,
    })
    return ContextSample(text=text, labels=labels)


def shell_session(rng: random.Random) -> ContextSample:
    command = rng.choice([
        "grep -R 'TODO' src -n",
        "tar -czf backup.tar.gz data/",
        "watch -n 1 'nvidia-smi'",
    ])
    text = (
        "Shell session:\n$ "
        + command
        + "\nExplain what this command does and warn about any side effects."
    )
    labels = _base_labels()
    labels.update({
        "code_scent": 1.0,
        "topic": TOPICS["code"],
        "intent": INTENTS["explanation"],
        "route_code": 1.0,
        "route_lang": 0.0,
    })
    return ContextSample(text=text, labels=labels)


def sql_question(rng: random.Random) -> ContextSample:
    table = rng.choice(["orders", "sensors", "visits"])
    column = rng.choice(["timestamp", "status", "total"])
    text = (
        "Given the SQL query:\n"
        "SELECT " + column + f" FROM {table} WHERE tenant_id = ?\n"
        "  AND created_at >= NOW() - INTERVAL '7 days';\n"
        "Explain its intent and suggest one useful index."
    )
    labels = _base_labels()
    labels.update({
        "code_scent": 1.0,
        "topic": TOPICS["data"],
        "intent": INTENTS["analysis"],
        "route_code": 1.0,
        "route_lang": 0.0,
    })
    return ContextSample(text=text, labels=labels)


def json_payload(rng: random.Random) -> ContextSample:
    items = {
        "sensor": rng.choice(["alpha", "bravo", "charlie"]),
        "temperature": round(rng.uniform(8.5, 42.3), 1),
        "unit": rng.choice(["C", "F"]),
        "timestamp": f"2025-0{rng.randint(1, 9)}-1{rng.randint(0, 8)}T0{rng.randint(0, 9)}:3{rng.randint(0, 9)}:00Z",
    }
    text = json.dumps(items, separators=(",", ":"))
    labels = _base_labels()
    labels.update({
        "data_format": 3,
        "topic": TOPICS["data"],
        "intent": INTENTS["statement"],
    })
    return ContextSample(text=text, labels=labels)


def xml_snippet(rng: random.Random) -> ContextSample:
    product = rng.choice(["resistor", "transistor", "capacitor"])
    text = (
        "<component>\n"
        f"  <name>{product}</name>\n"
        f"  <quantity>{rng.randint(5, 120)}</quantity>\n"
        f"  <tolerance>{rng.choice(['1%', '5%', '10%'])}</tolerance>\n"
        "</component>"
    )
    labels = _base_labels()
    labels.update({
        "data_format": 4,
        "topic": TOPICS["data"],
        "intent": INTENTS["statement"],
    })
    return ContextSample(text=text, labels=labels)


def table_report(rng: random.Random) -> ContextSample:
    months = ["Jan", "Feb", "Mar", "Apr", "May", "Jun"]
    lines = ["Month    Revenue    Cost"]
    for m in rng.sample(months, 4):
        rev = rng.randint(12000, 48000)
        cost = rng.randint(8000, rev - 1000)
        lines.append(f"{m} {rev:>8} {cost:>8}")
    text = "\n".join(lines)
    labels = _base_labels()
    labels.update({
        "data_format": 2,
        "topic": TOPICS["finance"],
        "intent": INTENTS["analysis"],
    })
    return ContextSample(text=text, labels=labels)


def csv_sample(rng: random.Random) -> ContextSample:
    headers = ["user_id", "region", "score"]
    rows = [headers]
    for _ in range(4):
        rows.append([
            str(rng.randint(1000, 9999)),
            rng.choice(["NA", "EU", "APAC", "LATAM"]),
            f"{rng.uniform(0, 1):.3f}",
        ])
    text = "\n".join(",".join(r) for r in rows)
    labels = _base_labels()
    labels.update({
        "data_format": 2,
        "topic": TOPICS["data"],
        "intent": INTENTS["data_request"],
    })
    return ContextSample(text=text, labels=labels)


def bullet_list(rng: random.Random) -> ContextSample:
    steps = rng.sample([
        "Collect requirements from stakeholders",
        "Draft a specification document",
        "Review the design with the QA team",
        "Implement the proof of concept",
        "Schedule a rollout meeting",
    ], k=4)
    text = "\n".join(f"- {step}" for step in steps)
    labels = _base_labels()
    labels.update({
        "data_format": 1,
        "topic": TOPICS["education"],
        "intent": INTENTS["instruction"],
    })
    return ContextSample(text=text, labels=labels)


def qualitative_summary(rng: random.Random) -> ContextSample:
    subject = rng.choice(["climate", "market", "soil", "traffic"])
    trend = rng.choice([
        "a steady increase over the past decade",
        "notable volatility tied to policy shifts",
        "seasonal variation with distinct peaks",
    ])
    question = rng.choice([
        "Highlight the dominant drivers behind this trend.",
        "Which interventions would stabilize the system?",
        "Explain why the baseline is drifting upward.",
    ])
    text = f"Recent {subject} observations show {trend}. {question}"
    labels = _base_labels()
    labels.update({
        "topic": TOPICS["science"],
        "intent": INTENTS["question"],
    })
    return ContextSample(text=text, labels=labels)


def conversational(rng: random.Random) -> ContextSample:
    prompt = rng.choice([
        "Hey, can you remind me what we decided for the team lunch menu?",
        "Good morning! Any fun weekend plans you are excited about?",
        "Thanks for the help yesterday; it made a big difference.",
    ])
    labels = _base_labels()
    labels.update({
        "topic": TOPICS["chat"],
        "intent": INTENTS["casual"],
    })
    return ContextSample(text=prompt, labels=labels)


def data_request(rng: random.Random) -> ContextSample:
    text = "Please parse the attached CSV and summarize the fields: user_id, signup_date, purchase_total."
    labels = _base_labels()
    labels.update({
        "topic": TOPICS["data"],
        "intent": INTENTS["data_request"],
    })
    return ContextSample(text=text, labels=labels)


def log_diagnostics(rng: random.Random) -> ContextSample:
    lines = [
        "2025-03-18T03:12:44Z WARN worker-2 retry budget exceeded",
        "2025-03-18T03:12:45Z INFO worker-2 backoff scheduled",
        "2025-03-18T03:12:46Z ERROR worker-2 circuit open",
    ]
    text = "\n".join(lines) + "\nSummarize what is happening and propose one fix."
    labels = _base_labels()
    labels.update({
        "topic": TOPICS["general"],
        "intent": INTENTS["analysis"],
    })
    return ContextSample(text=text, labels=labels)


SCENARIOS: List[Tuple[str, float, Callable[[random.Random], ContextSample]]] = [
    ("math_problem", 1.4, math_problem),
    ("math_analysis", 1.0, math_analysis),
    ("math_units_instruction", 0.8, math_units_instruction),
    ("latex_derivation", 0.6, latex_derivation),
    ("code_snippet", 1.2, code_snippet),
    ("code_question", 0.9, code_question),
    ("shell_session", 0.7, shell_session),
    ("sql_question", 0.8, sql_question),
    ("json_payload", 0.7, json_payload),
    ("xml_snippet", 0.5, xml_snippet),
    ("table_report", 0.9, table_report),
    ("csv_sample", 0.7, csv_sample),
    ("bullet_list", 0.8, bullet_list),
    ("qualitative_summary", 1.0, qualitative_summary),
    ("conversational", 0.6, conversational),
    ("data_request", 0.6, data_request),
    ("log_diagnostics", 0.7, log_diagnostics),
]


def generate_example(rng: random.Random) -> ContextSample:
    names, weights, funcs = zip(*SCENARIOS)
    idx = rng.choices(range(len(funcs)), weights=weights, k=1)[0]
    sample = funcs[idx](rng)
    sample.labels["scenario"] = names[idx]
    return sample


def write_jsonl(path: Path, samples: List[ContextSample]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="\n") as f:
        for sample in samples:
            row = {
                "text": sample.text,
                "labels": {k: (float(v) if isinstance(v, bool) else v) for k, v in sample.labels.items() if k != "scenario"},
                "meta": {"scenario": sample.labels["scenario"]},
            }
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def build_dataset(count: int, rng: random.Random) -> List[ContextSample]:
    return [generate_example(rng) for _ in range(count)]


def summarize(samples: List[ContextSample]) -> str:
    counter = Counter(sample.labels["scenario"] for sample in samples)
    total = sum(counter.values()) or 1
    parts = [f"{name}:{count/total:.1%}" for name, count in counter.most_common(6)]
    if len(counter) > 6:
        parts.append("...")
    return " ".join(parts)


def main() -> None:
    ap = argparse.ArgumentParser(description="Generate synthetic context classification data")
    ap.add_argument("--train-out", default="data/context_train.jsonl")
    ap.add_argument("--val-out", default="data/context_val.jsonl")
    ap.add_argument("--train-count", type=int, default=50000)
    ap.add_argument("--val-count", type=int, default=5000)
    ap.add_argument("--seed", type=int, default=2025)
    args = ap.parse_args()

    rng = random.Random(args.seed)
    train_samples = build_dataset(args.train_count, rng)
    val_samples = build_dataset(args.val_count, rng)

    write_jsonl(Path(args.train_out), train_samples)
    write_jsonl(Path(args.val_out), val_samples)

    print(f"[context-gen] train={len(train_samples):,} -> {args.train_out} ({summarize(train_samples)})")
    print(f"[context-gen] val={len(val_samples):,} -> {args.val_out} ({summarize(val_samples)})")


if __name__ == "__main__":
    main()
