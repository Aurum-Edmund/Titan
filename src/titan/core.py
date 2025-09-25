from __future__ import annotations

import math
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _maybe_ckpt(fn, *args, use_ckpt: bool = False):
    if not use_ckpt:
        return fn(*args)
    return torch.utils.checkpoint.checkpoint(fn, *args, use_reentrant=False)


class SiliconNeuron(nn.Module):
    """Feedforward gating block designed for silicon hardware."""

    def __init__(self, d: int, expansion: float = 4.0, dropout: float = 0.0):
        super().__init__()
        hidden = max(1, int(d * expansion))
        self.in_proj = nn.Linear(d, hidden * 2)
        self.out_proj = nn.Linear(hidden, d)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        up, gate = self.in_proj(x).chunk(2, dim=-1)
        activated = F.silu(up) * torch.sigmoid(gate)
        out = self.out_proj(activated)
        return self.dropout(out)


class TitanBlock(nn.Module):
    """Transformer-style block with causal self-attention and silicon neurons."""

    def __init__(
        self,
        d: int,
        heads: int = 8,
        expansion: float = 4.0,
        dropout: float = 0.0,
        use_ckpt: bool = False,
    ) -> None:
        super().__init__()
        self.use_ckpt = use_ckpt
        self.ln_attn = nn.LayerNorm(d)
        self.attn = nn.MultiheadAttention(d, num_heads=heads, dropout=dropout, batch_first=True)
        self.dropout_attn = nn.Dropout(dropout)
        self.ln_ff = nn.LayerNorm(d)
        self.ff = SiliconNeuron(d, expansion=expansion, dropout=dropout)
        self.dropout_ff = nn.Dropout(dropout)

    def _attn_step(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor]) -> torch.Tensor:
        q = self.ln_attn(x)
        attn_out = self.attn(q, q, q, attn_mask=attn_mask, need_weights=False)[0]
        return self.dropout_attn(attn_out)

    def _ff_step(self, x: torch.Tensor) -> torch.Tensor:
        ff_out = self.ff(self.ln_ff(x))
        return self.dropout_ff(ff_out)

    def forward(self, x: torch.Tensor, attn_mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        x = x + _maybe_ckpt(self._attn_step, x, attn_mask, use_ckpt=self.use_ckpt)
        x = x + _maybe_ckpt(self._ff_step, x, use_ckpt=self.use_ckpt)
        return x


class TitanModel(nn.Module):
    """Minimal autoregressive LM with silicon-native neurons and causal attention."""

    def __init__(
        self,
        vocab_size: int,
        d: int,
        n_blocks: int,
        *,
        heads: int = 8,
        ff_mult: float = 4.0,
        dropout: float = 0.0,
        max_position_embeddings: int = 2048,
        use_ckpt: bool = False,
    ) -> None:
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d)
        self.pos_embed = nn.Embedding(max_position_embeddings, d)
        self.embed_scale = math.sqrt(d)
        self.dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TitanBlock(d, heads=heads, expansion=ff_mult, dropout=dropout, use_ckpt=use_ckpt)
                for _ in range(n_blocks)
            ]
        )
        self.ln_f = nn.LayerNorm(d)
        self.lm_head = nn.Linear(d, vocab_size, bias=False)
        self.lm_head.weight = self.embed.weight  # weight tying
        self.register_buffer("_causal_mask", torch.empty(0, dtype=torch.bool), persistent=False)

    def _build_causal_mask(self, seq_len: int, device: torch.device) -> torch.Tensor:
        if self._causal_mask.size(0) < seq_len or self._causal_mask.device != device:
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device, dtype=torch.bool), diagonal=1)
            self._causal_mask = mask
        return self._causal_mask[:seq_len, :seq_len]

    def forward(
        self,
        input_ids: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
    ) -> dict[str, torch.Tensor]:
        if input_ids.dim() != 2:
            raise ValueError("input_ids must be of shape [batch, seq]")

        B, T = input_ids.shape
        if T > self.pos_embed.num_embeddings:
            raise ValueError(
                f"Sequence length {T} exceeds configured max_position_embeddings={self.pos_embed.num_embeddings}"
            )

        device = input_ids.device
        positions = torch.arange(T, device=device, dtype=torch.long).unsqueeze(0).expand(B, T)
        x = self.embed(input_ids) * self.embed_scale + self.pos_embed(positions)
        x = self.dropout(x)

        attn_mask = self._build_causal_mask(T, device)
        for block in self.blocks:
            x = block(x, attn_mask=attn_mask)

        x = self.ln_f(x)
        logits = self.lm_head(x)
        out = {"logits": logits}

        if labels is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), labels.view(-1), ignore_index=-100)
            out["loss"] = loss

        return out