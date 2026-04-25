from __future__ import annotations

import math
from typing import cast

import torch
import torch.nn.functional as functional
from torch import nn

from .rope import apply_rope_to_qk, build_rope_cache


class MultiHeadSelfAttention(nn.Module):
    """Batch-first multi-head self-attention with rotary positional embeddings."""

    def __init__(
        self,
        hidden_size: int,
        num_attention_heads: int,
        *,
        dropout: float,
        rope_base: float,
    ) -> None:
        super().__init__()
        if hidden_size % num_attention_heads != 0:
            raise ValueError("hidden_size must be divisible by num_attention_heads.")
        self.hidden_size = hidden_size
        self.num_attention_heads = num_attention_heads
        self.head_dim = hidden_size // num_attention_heads
        if self.head_dim % 2 != 0:
            raise ValueError("Per-head dimension must be even for RoPE.")
        self.rope_base = rope_base

        self.query_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.key_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.value_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.output_projection = nn.Linear(hidden_size, hidden_size, bias=False)
        self.attention_dropout = nn.Dropout(dropout)
        self.output_dropout = nn.Dropout(dropout)

    def forward(
        self,
        hidden_states: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if hidden_states.ndim != 3:
            raise ValueError("Expected hidden_states shape [batch, seq, hidden_size].")
        batch_size, sequence_length, _ = hidden_states.shape

        query = self._reshape_to_heads(self.query_projection(hidden_states))
        key = self._reshape_to_heads(self.key_projection(hidden_states))
        value = self._reshape_to_heads(self.value_projection(hidden_states))

        cos, sin = build_rope_cache(
            sequence_length,
            self.head_dim,
            device=hidden_states.device,
            dtype=hidden_states.dtype,
            base=self.rope_base,
        )
        query, key = apply_rope_to_qk(query, key, cos=cos, sin=sin)

        attention_scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if attention_mask is not None:
            if attention_mask.shape != (batch_size, sequence_length):
                raise ValueError("attention_mask must match shape [batch, seq].")
            key_mask = attention_mask.to(dtype=torch.bool).unsqueeze(1).unsqueeze(2)
            min_value = torch.finfo(attention_scores.dtype).min
            attention_scores = attention_scores.masked_fill(~key_mask, min_value)

        attention_weights = functional.softmax(attention_scores, dim=-1, dtype=torch.float32).to(
            dtype=hidden_states.dtype
        )
        attention_weights = self.attention_dropout(attention_weights)
        context = torch.matmul(attention_weights, value)
        context = context.transpose(1, 2).contiguous().view(batch_size, sequence_length, -1)
        return cast(torch.Tensor, self.output_dropout(self.output_projection(context)))

    def _reshape_to_heads(self, tensor: torch.Tensor) -> torch.Tensor:
        batch_size, sequence_length, _ = tensor.shape
        tensor = tensor.view(batch_size, sequence_length, self.num_attention_heads, self.head_dim)
        return tensor.transpose(1, 2)
