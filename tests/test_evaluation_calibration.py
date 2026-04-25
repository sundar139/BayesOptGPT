from __future__ import annotations

import torch

from bayes_gp_llmops.evaluation.calibration import (
    TemperatureScaler,
    brier_score,
    compute_ece,
    negative_log_likelihood,
)
from bayes_gp_llmops.evaluation.uncertainty import probabilities_from_logits


def test_ece_is_zero_for_perfect_predictions() -> None:
    probabilities = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1, 2], dtype=torch.long)
    assert compute_ece(probabilities, labels, num_bins=10) == 0.0


def test_brier_score_matches_manual_computation() -> None:
    probabilities = torch.tensor(
        [
            [0.8, 0.2],
            [0.3, 0.7],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([0, 1], dtype=torch.long)

    manual = (((0.8 - 1.0) ** 2 + (0.2 - 0.0) ** 2) + ((0.3 - 0.0) ** 2 + (0.7 - 1.0) ** 2)) / 2.0
    assert abs(brier_score(probabilities, labels) - manual) < 1e-6


def test_temperature_scaling_reduces_nll_on_overconfident_logits() -> None:
    logits = torch.tensor(
        [
            [8.0, 1.0, 0.5],
            [7.5, 1.0, 0.3],
            [8.2, 0.6, 0.4],
            [0.2, 6.0, 0.1],
            [0.3, 6.5, 0.2],
            [0.4, 5.8, 0.1],
        ],
        dtype=torch.float32,
    )
    labels = torch.tensor([1, 1, 1, 1, 1, 1], dtype=torch.long)
    baseline_nll = negative_log_likelihood(logits, labels)

    scaler = TemperatureScaler(initial_temperature=1.0)
    learned_temperature = scaler.fit(logits=logits, labels=labels, max_iterations=100)
    calibrated_logits = scaler.transform(logits)
    calibrated_nll = negative_log_likelihood(calibrated_logits, labels)

    assert learned_temperature > 0.0
    assert calibrated_nll < baseline_nll
    probabilities = probabilities_from_logits(calibrated_logits)
    assert torch.allclose(probabilities.sum(dim=1), torch.ones_like(probabilities[:, 0]), atol=1e-6)
