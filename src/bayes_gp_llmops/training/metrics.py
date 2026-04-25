from __future__ import annotations

from collections.abc import Sequence

import torch
from sklearn.metrics import f1_score


def compute_accuracy(predictions: Sequence[int], labels: Sequence[int]) -> float:
    """Compute classification accuracy."""

    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must have the same length.")
    if not labels:
        return 0.0
    correct = sum(int(pred == label) for pred, label in zip(predictions, labels, strict=True))
    return correct / len(labels)


def compute_macro_f1(
    predictions: Sequence[int],
    labels: Sequence[int],
    *,
    num_classes: int,
) -> float:
    """Compute macro-F1 score across fixed class labels."""

    if len(predictions) != len(labels):
        raise ValueError("predictions and labels must have the same length.")
    if not labels:
        return 0.0
    class_labels = list(range(num_classes))
    return float(
        f1_score(
            labels,
            predictions,
            labels=class_labels,
            average="macro",
            zero_division=0,
        )
    )


def logits_to_predictions(logits: torch.Tensor) -> list[int]:
    """Convert model logits to predicted class indices."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, num_classes].")
    return torch.argmax(logits, dim=-1).tolist()
