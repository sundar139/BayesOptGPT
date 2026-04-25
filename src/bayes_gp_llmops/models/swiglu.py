from __future__ import annotations

from typing import cast

import torch
import torch.nn.functional as functional
from torch import nn


class SwiGLU(nn.Module):
    """SwiGLU feed-forward projection block."""

    def __init__(self, hidden_size: int, multiplier: float, dropout: float) -> None:
        super().__init__()
        if hidden_size < 1:
            raise ValueError("hidden_size must be positive.")
        if multiplier < 1.0:
            raise ValueError("multiplier must be >= 1.0.")
        intermediate_size = int(hidden_size * multiplier)
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        gated = functional.silu(self.gate_proj(hidden_states)) * self.up_proj(hidden_states)
        return cast(torch.Tensor, self.dropout(self.down_proj(gated)))
