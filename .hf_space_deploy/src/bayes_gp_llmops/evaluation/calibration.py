from __future__ import annotations

import math

import torch
import torch.nn.functional as functional
from torch import nn
from torch.optim import LBFGS


def negative_log_likelihood(logits: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute mean negative log-likelihood for multi-class logits."""

    _validate_logits_and_labels(logits=logits, labels=labels)
    return float(functional.cross_entropy(logits, labels).item())


def brier_score(probabilities: torch.Tensor, labels: torch.Tensor) -> float:
    """Compute the multiclass Brier score from probabilities and labels."""

    _validate_probabilities_and_labels(probabilities=probabilities, labels=labels)
    num_classes = probabilities.size(1)
    targets = functional.one_hot(labels, num_classes=num_classes).to(probabilities.dtype)
    squared_error = (probabilities - targets).pow(2).sum(dim=1)
    return float(squared_error.mean().item())


def compute_ece(probabilities: torch.Tensor, labels: torch.Tensor, num_bins: int = 15) -> float:
    """Compute expected calibration error (ECE) using equal-width confidence bins."""

    if num_bins < 1:
        raise ValueError("num_bins must be >= 1.")
    _validate_probabilities_and_labels(probabilities=probabilities, labels=labels)

    confidences, predictions = probabilities.max(dim=1)
    correctness = predictions.eq(labels).to(probabilities.dtype)

    bin_edges = torch.linspace(0.0, 1.0, steps=num_bins + 1, device=probabilities.device)
    ece = torch.tensor(0.0, device=probabilities.device, dtype=probabilities.dtype)
    sample_count = probabilities.size(0)

    for bin_index in range(num_bins):
        lower = bin_edges[bin_index]
        upper = bin_edges[bin_index + 1]
        if bin_index == num_bins - 1:
            mask = (confidences >= lower) & (confidences <= upper)
        else:
            mask = (confidences >= lower) & (confidences < upper)
        if not torch.any(mask):
            continue
        bin_confidence = confidences[mask].mean()
        bin_accuracy = correctness[mask].mean()
        weight = float(mask.sum().item()) / float(sample_count)
        ece = ece + torch.abs(bin_accuracy - bin_confidence) * weight

    return float(ece.item())


def apply_temperature(logits: torch.Tensor, temperature: float) -> torch.Tensor:
    """Apply temperature scaling to logits."""

    if temperature <= 0.0:
        raise ValueError("temperature must be > 0.")
    return logits / temperature


class TemperatureScaler(nn.Module):
    """Fit and apply a scalar post-hoc temperature parameter."""

    def __init__(self, initial_temperature: float = 1.0) -> None:
        super().__init__()
        if initial_temperature <= 0.0:
            raise ValueError("initial_temperature must be > 0.")
        self._log_temperature = nn.Parameter(torch.tensor(math.log(initial_temperature)))

    @property
    def temperature(self) -> float:
        return float(torch.exp(self._log_temperature).detach().item())

    def fit(
        self,
        *,
        logits: torch.Tensor,
        labels: torch.Tensor,
        max_iterations: int = 200,
    ) -> float:
        """Fit temperature by minimizing validation NLL."""

        if max_iterations < 1:
            raise ValueError("max_iterations must be >= 1.")
        _validate_logits_and_labels(logits=logits, labels=labels)
        logits = logits.detach()
        labels = labels.detach()

        optimizer = LBFGS(
            [self._log_temperature],
            lr=0.1,
            max_iter=max_iterations,
            line_search_fn="strong_wolfe",
        )

        def closure() -> torch.Tensor:
            optimizer.zero_grad(set_to_none=True)
            scaled_logits = self._scale(logits)
            loss = functional.cross_entropy(scaled_logits, labels)
            loss.backward()  # type: ignore[no-untyped-call]
            return loss

        optimizer.step(closure)  # type: ignore[no-untyped-call]
        return self.temperature

    def transform(self, logits: torch.Tensor) -> torch.Tensor:
        """Scale logits using the learned temperature."""

        return self._scale(logits.detach())

    def _scale(self, logits: torch.Tensor) -> torch.Tensor:
        temperature = torch.exp(self._log_temperature).clamp_min(1e-6)
        return logits / temperature


def _validate_logits_and_labels(*, logits: torch.Tensor, labels: torch.Tensor) -> None:
    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, num_classes].")
    if labels.ndim != 1:
        raise ValueError("labels must have shape [batch].")
    if logits.size(0) != labels.size(0):
        raise ValueError("logits and labels must have the same batch size.")
    if labels.dtype not in (torch.int64, torch.int32, torch.int16, torch.int8):
        raise ValueError("labels must contain integer class indices.")


def _validate_probabilities_and_labels(
    *,
    probabilities: torch.Tensor,
    labels: torch.Tensor,
) -> None:
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [batch, num_classes].")
    if labels.ndim != 1:
        raise ValueError("labels must have shape [batch].")
    if probabilities.size(0) != labels.size(0):
        raise ValueError("probabilities and labels must have the same batch size.")
