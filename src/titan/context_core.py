# src/titan/context_core.py
from __future__ import annotations
import math, torch, torch.nn as nn
from dataclasses import dataclass
from typing import Dict, Any, Optional

# --------- Blocks ---------

class RMSNorm(nn.Module):
    def __init__(self, dim, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps
    def forward(self, x):
        norm = x.norm(2, dim=-1, keepdim=True).clamp_min(self.eps)
        return x * (self.weight / (norm / math.sqrt(x.size(-1))))

class MHA(nn.Module):
    def __init__(self, d, n_heads, dropout: float = 0.0):
        super().__init__()
        assert d % n_heads == 0
        self.d = d
        self.h = n_heads
        self.dk = d // n_heads
        self.qkv = nn.Linear(d, 3 * d, bias=False)
        self.o = nn.Linear(d, d, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask: Optional[torch.Tensor] = None):
        B, T, D = x.shape
        qkv = self.qkv(x).view(B, T, 3, self.h, self.dk).transpose(1, 3)  # B, h, T, 3, dk
        q, k, v = qkv[:, :, :, 0], qkv[:, :, :, 1], qkv[:, :, :, 2]
        att = (q @ k.transpose(-2, -1)) / math.sqrt(self.dk)  # B, h, T, T
        if mask is not None:
            if mask.dim() == 2:  # [B, T]
                mask = mask[:, None, None, :]
            elif mask.dim() == 3:  # [B, 1, T]
                mask = mask[:, :, None, :]
            att = att.masked_fill(mask == 0, float('-inf'))
        p = att.softmax(dim=-1)
        y = p @ v
        y = y.transpose(1, 2).contiguous().view(B, T, D)
        return self.dropout(self.o(y))

class FFN(nn.Module):
    def __init__(self, d, mult=4, dropout: float = 0.0):
        super().__init__()
        inner = mult * d
        self.w1 = nn.Linear(d, inner, bias=False)
        self.w2 = nn.Linear(inner, d, bias=False)
        self.act = nn.SiLU()
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.dropout(self.w2(self.act(self.w1(x))))

class EncoderBlock(nn.Module):
    def __init__(self, d, n_heads, ffn_mult=4, dropout: float = 0.0):
        super().__init__()
        self.n1 = RMSNorm(d)
        self.att = MHA(d, n_heads, dropout=dropout)
        self.n2 = RMSNorm(d)
        self.ffn = FFN(d, ffn_mult, dropout=dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask=None):
        x = x + self.dropout(self.att(self.n1(x), mask))
        x = x + self.dropout(self.ffn(self.n2(x)))
        return x

# --------- ContextCore ---------

@dataclass
class ContextHeadDims:
    # binary heads are BCEWithLogits; multi-class are CrossEntropy
    # You can tune/extend these without touching the backbone.
    num_topics: int = 16   # coarse topic taxonomy
    num_intents: int = 8   # question, instruction, chat, code-edit, etc.

class ContextCore(nn.Module):
    """
    Context encoder that outputs:
      - pooled: [B, d] pooled representation
      - logits: dict of routing & classification heads
    """
    def __init__(self, vocab_size: int, d: int = 384, n_layers: int = 12, n_heads: int = 6,
                 pad_id: int = 0, head_dims: ContextHeadDims = ContextHeadDims(),
                 dropout: float = 0.0, max_position_embeddings: int = 4096):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d)
        self.pos = nn.Parameter(torch.empty(max_position_embeddings, d).uniform_(-0.02, 0.02))
        self.embed_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList([EncoderBlock(d, n_heads, dropout=dropout) for _ in range(n_layers)])
        self.norm = RMSNorm(d)
        self.pad_id = pad_id

        # Heads
        H = head_dims
        self.heads = nn.ModuleDict({
            "math_scent":   nn.Linear(d, 1, bias=True),     # BCE
            "code_scent":   nn.Linear(d, 1, bias=True),     # BCE
            "data_format":  nn.Linear(d, 5, bias=True),     # CE: plain, list, table, json, xml
            "topic":        nn.Linear(d, H.num_topics, True),  # CE
            "intent":       nn.Linear(d, H.num_intents, True), # CE
            "route_math":   nn.Linear(d, 1, True),          # BCE (explicit routing score)
            "route_code":   nn.Linear(d, 1, True),          # BCE
            "route_lang":   nn.Linear(d, 1, True),          # BCE
        })
        self.d = d

    def forward(self, ids: torch.Tensor, attention_mask: Optional[torch.Tensor] = None) -> Dict[str, Any]:
        """
        ids: [B, T] token ids
        attention_mask: optional mask (1 for tokens to attend, 0 for pad)
        """
        if ids.dim() != 2:
            raise ValueError("ids must be [batch, seq]")

        B, T = ids.shape
        if T > self.pos.size(0):
            raise ValueError(f"sequence length {T} exceeds max_position_embeddings={self.pos.size(0)}")

        device = ids.device
        if attention_mask is None:
            if self.pad_id is not None:
                attention_mask = ids.ne(self.pad_id)
            else:
                attention_mask = torch.ones(B, T, dtype=torch.bool, device=device)
        else:
            attention_mask = attention_mask.to(dtype=torch.bool, device=device)

        x = self.embed(ids) + self.pos[:T]
        x = self.embed_dropout(x)

        for blk in self.blocks:
            x = blk(x, mask=attention_mask)

        x = self.norm(x)
        mask_f = attention_mask.to(x.dtype)
        lengths = mask_f.sum(dim=1, keepdim=True).clamp_min(1.0)
        pooled = (x * mask_f.unsqueeze(-1)).sum(dim=1) / lengths

        logits = {name: head(pooled) for name, head in self.heads.items()}
        return {"pooled": pooled, "logits": logits}
