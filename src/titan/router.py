# src/titan/router.py
from __future__ import annotations
import torch
from typing import Dict, Any

def sigmoid(x: torch.Tensor) -> torch.Tensor:
    return torch.sigmoid(x)

def softmax(x: torch.Tensor) -> torch.Tensor:
    return torch.softmax(x, dim=-1)

def route_from_logits(logits: Dict[str, torch.Tensor], thresholds=None) -> Dict[str, Any]:
    """
    Turn ContextCore logits into routing decisions.
    thresholds: optional dict to override defaults, e.g. {"math":0.5, "code":0.5, "lang":0.5}
    """
    if thresholds is None:
        thresholds = {"math":0.5, "code":0.5, "lang":0.5}

    p_math = sigmoid(logits["route_math"]).squeeze(-1)  # [B]
    p_code = sigmoid(logits["route_code"]).squeeze(-1)
    p_lang = sigmoid(logits["route_lang"]).squeeze(-1)

    # top-1 multi-class for info (not gating)
    fmt  = logits["data_format"].argmax(dim=-1)  # [B]
    topic= logits["topic"].argmax(dim=-1)
    intent=logits["intent"].argmax(dim=-1)

    # greedy routing (can be made exclusive or allow multi-route)
    route = []
    for i in range(p_math.size(0)):
        cand = []
        if p_math[i] >= thresholds["math"]: cand.append("math")
        if p_code[i] >= thresholds["code"]: cand.append("code")
        if p_lang[i] >= thresholds["lang"]: cand.append("lang")
        if not cand:
            # choose best of three if all under threshold
            vals = torch.stack([p_math[i], p_code[i], p_lang[i]])
            j = int(vals.argmax().item())
            cand = ["math","code","lang"][j]
        route.append(cand)
    return {"route": route, "fmt": fmt, "topic": topic, "intent": intent,
            "p_math": p_math, "p_code": p_code, "p_lang": p_lang}
