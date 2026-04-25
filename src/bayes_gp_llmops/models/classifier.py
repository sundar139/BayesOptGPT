from __future__ import annotations

from typing import cast

import torch
from torch import nn

from .config import ModelConfig
from .transformer import TinyLlamaBackbone


class TinyLlamaForSequenceClassification(nn.Module):
    """Tiny LLaMA-style classifier with masked-mean pooling over token features."""

    def __init__(self, config: ModelConfig) -> None:
        super().__init__()
        self.config = config
        self.backbone = TinyLlamaBackbone(
            vocab_size=config.vocab_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            num_attention_heads=config.num_attention_heads,
            feedforward_multiplier=config.feedforward_multiplier,
            dropout=config.dropout,
            rope_base=config.rope_base,
        )
        self.classifier_dropout = nn.Dropout(config.dropout)
        self.classifier_head = nn.Linear(config.hidden_size, config.num_classes)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        hidden_states = self.backbone(input_ids, attention_mask=attention_mask)
        pooled = _masked_mean_pool(hidden_states, attention_mask=attention_mask)
        return cast(torch.Tensor, self.classifier_head(self.classifier_dropout(pooled)))


def _masked_mean_pool(
    hidden_states: torch.Tensor,
    *,
    attention_mask: torch.Tensor | None = None,
) -> torch.Tensor:
    if attention_mask is None:
        return hidden_states.mean(dim=1)
    if attention_mask.ndim != 2:
        raise ValueError("attention_mask must have shape [batch, seq].")
    mask = attention_mask.to(dtype=hidden_states.dtype).unsqueeze(-1)
    numerator = (hidden_states * mask).sum(dim=1)
    denominator = mask.sum(dim=1).clamp_min(1.0)
    return numerator / denominator
