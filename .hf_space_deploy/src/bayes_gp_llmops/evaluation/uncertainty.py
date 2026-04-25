from __future__ import annotations

import torch
import torch.nn.functional as functional


def probabilities_from_logits(logits: torch.Tensor) -> torch.Tensor:
    """Convert logits to probabilities via softmax."""

    if logits.ndim != 2:
        raise ValueError("logits must have shape [batch, num_classes].")
    return functional.softmax(logits, dim=1)


def max_softmax_confidence(probabilities: torch.Tensor) -> torch.Tensor:
    """Compute max softmax confidence for each sample."""

    _validate_probabilities(probabilities)
    confidence, _ = probabilities.max(dim=1)
    return confidence


def predictive_entropy(probabilities: torch.Tensor, epsilon: float = 1e-12) -> torch.Tensor:
    """Compute predictive entropy for each sample."""

    _validate_probabilities(probabilities)
    if epsilon <= 0.0:
        raise ValueError("epsilon must be > 0.")
    entropy = -(probabilities * (probabilities.clamp_min(epsilon).log())).sum(dim=1)
    return entropy


def confidence_margin(probabilities: torch.Tensor) -> torch.Tensor:
    """Compute the confidence margin (top-1 minus top-2 probability)."""

    _validate_probabilities(probabilities)
    if probabilities.size(1) < 2:
        return torch.zeros(
            probabilities.size(0),
            dtype=probabilities.dtype,
            device=probabilities.device,
        )
    top2 = torch.topk(probabilities, k=2, dim=1).values
    return top2[:, 0] - top2[:, 1]


def summarize_uncertainty(
    *,
    confidence: torch.Tensor,
    entropy: torch.Tensor,
) -> dict[str, float | dict[str, float]]:
    """Summarize confidence and entropy distributions."""

    if confidence.ndim != 1 or entropy.ndim != 1:
        raise ValueError("confidence and entropy must be 1D tensors.")
    if confidence.size(0) != entropy.size(0):
        raise ValueError("confidence and entropy must have matching lengths.")

    return {
        "mean_confidence": float(confidence.mean().item()),
        "mean_entropy": float(entropy.mean().item()),
        "confidence_quantiles": _quantiles(confidence),
        "entropy_quantiles": _quantiles(entropy),
    }


def _quantiles(values: torch.Tensor) -> dict[str, float]:
    if values.numel() == 0:
        raise ValueError("values must not be empty.")
    quantile_levels = torch.tensor([0.1, 0.25, 0.5, 0.75, 0.9], device=values.device)
    quantiles = torch.quantile(values, quantile_levels)
    return {
        "p10": float(quantiles[0].item()),
        "p25": float(quantiles[1].item()),
        "p50": float(quantiles[2].item()),
        "p75": float(quantiles[3].item()),
        "p90": float(quantiles[4].item()),
    }


def _validate_probabilities(probabilities: torch.Tensor) -> None:
    if probabilities.ndim != 2:
        raise ValueError("probabilities must have shape [batch, num_classes].")
