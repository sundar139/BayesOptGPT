from __future__ import annotations

import torch
import torch.nn.functional as functional


def classification_loss(logits: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
    """Compute cross-entropy loss for sequence classification."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, num_classes].")
    if labels.ndim != 1:
        raise ValueError("labels must have shape [batch].")
    return functional.cross_entropy(logits, labels)
