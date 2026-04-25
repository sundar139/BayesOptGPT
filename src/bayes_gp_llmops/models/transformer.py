from __future__ import annotations

from typing import cast

import torch
from torch import nn

from .attention import MultiHeadSelfAttention
from .rmsnorm import RMSNorm
from .swiglu import SwiGLU


class TransformerBlock(nn.Module):
    """Pre-norm transformer block with RoPE attention and SwiGLU feed-forward."""

    def __init__(
        self,
        *,
        hidden_size: int,
        num_attention_heads: int,
        feedforward_multiplier: float,
        dropout: float,
        rope_base: float,
    ) -> None:
        super().__init__()
        self.attention_norm = RMSNorm(hidden_size)
        self.attention = MultiHeadSelfAttention(
            hidden_size=hidden_size,
            num_attention_heads=num_attention_heads,
            dropout=dropout,
            rope_base=rope_base,
        )
        self.feedforward_norm = RMSNorm(hidden_size)
        self.feedforward = SwiGLU(
            hidden_size=hidden_size,
            multiplier=feedforward_multiplier,
            dropout=dropout,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = hidden_states + self.attention(
            self.attention_norm(hidden_states),
            attention_mask=attention_mask,
        )
        hidden_states = hidden_states + self.feedforward(self.feedforward_norm(hidden_states))
        return hidden_states


class TinyLlamaBackbone(nn.Module):
    """Tiny LLaMA-style decoder backbone for classification features."""

    def __init__(
        self,
        *,
        vocab_size: int,
        hidden_size: int,
        num_layers: int,
        num_attention_heads: int,
        feedforward_multiplier: float,
        dropout: float,
        rope_base: float,
    ) -> None:
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, hidden_size)
        self.embedding_dropout = nn.Dropout(dropout)
        self.blocks = nn.ModuleList(
            [
                TransformerBlock(
                    hidden_size=hidden_size,
                    num_attention_heads=num_attention_heads,
                    feedforward_multiplier=feedforward_multiplier,
                    dropout=dropout,
                    rope_base=rope_base,
                )
                for _ in range(num_layers)
            ]
        )
        self.final_norm = RMSNorm(hidden_size)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch, seq].")
        hidden_states = self.embedding_dropout(self.token_embedding(input_ids))
        for block in self.blocks:
            hidden_states = block(hidden_states, attention_mask=attention_mask)
        return cast(torch.Tensor, self.final_norm(hidden_states))
