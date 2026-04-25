from __future__ import annotations

import csv
from pathlib import Path

import torch

from bayes_gp_llmops.evaluation.reports import export_predictions_csv, plot_reliability_diagram
from bayes_gp_llmops.evaluation.uncertainty import probabilities_from_logits


def test_export_predictions_csv_schema(tmp_path: Path) -> None:
    logits = torch.tensor([[2.0, 1.0], [0.3, 0.7]], dtype=torch.float32)
    probabilities = probabilities_from_logits(logits)
    labels = torch.tensor([0, 1], dtype=torch.long)
    predictions = torch.argmax(probabilities, dim=1)
    confidence = probabilities.max(dim=1).values
    entropy = -(probabilities * probabilities.clamp_min(1e-12).log()).sum(dim=1)
    margin = torch.tensor([0.5, 0.4], dtype=torch.float32)

    output_path = tmp_path / "predictions.csv"
    export_predictions_csv(
        path=output_path,
        split="test",
        labels=labels,
        predictions=predictions,
        confidence=confidence,
        entropy=entropy,
        margin=margin,
        logits=logits,
        probabilities=probabilities,
    )

    with output_path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        rows = list(reader)
        columns = reader.fieldnames or []
    expected_columns = {
        "split",
        "index",
        "true_label",
        "predicted_label",
        "confidence",
        "entropy",
        "margin",
        "correct",
        "raw_logit_0",
        "raw_logit_1",
        "probability_0",
        "probability_1",
    }
    assert expected_columns.issubset(set(columns))
    assert len(rows) == 2


def test_reliability_diagram_generation_smoke(tmp_path: Path) -> None:
    logits = torch.tensor(
        [
            [3.0, 0.5, 0.2],
            [0.1, 2.5, 0.3],
            [0.2, 0.4, 1.8],
            [1.5, 0.7, 0.1],
        ],
        dtype=torch.float32,
    )
    probabilities = probabilities_from_logits(logits)
    labels = torch.tensor([0, 1, 2, 0], dtype=torch.long)
    output_path = tmp_path / "reliability.png"
    plot_reliability_diagram(
        path=output_path,
        probabilities=probabilities,
        labels=labels,
        num_bins=8,
    )
    assert output_path.exists()
