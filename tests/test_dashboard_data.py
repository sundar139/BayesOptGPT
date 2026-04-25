from __future__ import annotations

import json
from pathlib import Path

from bayes_gp_llmops.dashboard.data import (
    calibration_comparison_rows,
    load_dashboard_data,
    metric_number,
    per_class_f1_rows,
    uncertainty_summary,
)


def test_load_dashboard_data_with_optional_files(tmp_path: Path) -> None:
    evaluation_dir = tmp_path / "evaluation_full_run"
    bundle_dir = tmp_path / "bundle"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    _write_json(
        evaluation_dir / "metrics_validation.json",
        {
            "accuracy": 0.9,
            "macro_f1": 0.89,
            "nll": 0.2,
            "brier_score": 0.1,
            "ece": 0.03,
            "num_samples": 12000,
            "per_class_f1": [0.91, 0.95, 0.85, 0.86],
            "uncertainty_summary": {"mean_confidence": 0.94, "mean_entropy": 0.12},
        },
    )
    _write_json(
        evaluation_dir / "metrics_test.json",
        {
            "accuracy": 0.91,
            "macro_f1": 0.9,
            "nll": 0.21,
            "brier_score": 0.11,
            "ece": 0.031,
            "num_samples": 7600,
            "per_class_f1": [0.92, 0.96, 0.86, 0.87],
            "uncertainty_summary": {"mean_confidence": 0.95, "mean_entropy": 0.11},
        },
    )
    _write_json(
        evaluation_dir / "metrics_validation_calibrated.json",
        {
            "accuracy": 0.9,
            "macro_f1": 0.89,
            "nll": 0.18,
            "brier_score": 0.09,
            "ece": 0.01,
        },
    )
    _write_json(
        evaluation_dir / "metrics_test_calibrated.json",
        {
            "accuracy": 0.91,
            "macro_f1": 0.9,
            "nll": 0.19,
            "brier_score": 0.095,
            "ece": 0.011,
        },
    )
    for image_name in (
        "confusion_matrix_test.png",
        "reliability_diagram_test.png",
        "confidence_histogram_test.png",
        "entropy_histogram_test.png",
    ):
        (evaluation_dir / image_name).write_bytes(b"PNG")

    _write_json(bundle_dir / "bundle_metadata.json", {"created_at": "2026-01-01T00:00:00Z"})
    _write_json(bundle_dir / "champion_manifest.json", {"trial_number": 1})

    loaded = load_dashboard_data(evaluation_dir=evaluation_dir, bundle_dir=bundle_dir)

    assert loaded.metrics_validation is not None
    assert loaded.metrics_test is not None
    assert metric_number(loaded.metrics_validation, "accuracy") == 0.9
    assert uncertainty_summary(loaded.metrics_test)["mean_entropy"] == 0.11
    assert len(per_class_f1_rows(loaded.metrics_test)) == 4
    assert loaded.bundle_metadata is not None
    assert loaded.champion_manifest is not None
    assert loaded.warnings == ()


def test_load_dashboard_data_warns_when_required_files_are_missing(tmp_path: Path) -> None:
    evaluation_dir = tmp_path / "evaluation_full_run"
    bundle_dir = tmp_path / "bundle"
    evaluation_dir.mkdir(parents=True, exist_ok=True)
    bundle_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_dashboard_data(evaluation_dir=evaluation_dir, bundle_dir=bundle_dir)

    assert loaded.metrics_validation is None
    assert loaded.metrics_test is None
    assert any("Missing required file" in item for item in loaded.warnings)
    assert any("Visualization not found" in item for item in loaded.warnings)


def test_calibration_comparison_rows_returns_expected_metrics() -> None:
    rows = calibration_comparison_rows(
        raw_metrics={
            "accuracy": 0.90,
            "macro_f1": 0.88,
            "nll": 0.25,
            "brier_score": 0.12,
            "ece": 0.03,
        },
        calibrated_metrics={
            "accuracy": 0.90,
            "macro_f1": 0.88,
            "nll": 0.20,
            "brier_score": 0.10,
            "ece": 0.01,
        },
    )

    assert len(rows) == 5
    assert rows[2]["metric"] == "nll"
    assert rows[2]["raw"] == 0.25
    assert rows[2]["calibrated"] == 0.2


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.write_text(json.dumps(payload), encoding="utf-8")
