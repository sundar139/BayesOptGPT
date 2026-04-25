from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import cast

CLASS_LABELS: tuple[str, ...] = ("World", "Sports", "Business", "Sci/Tech")

_METRICS_FILES: dict[str, str] = {
    "validation": "metrics_validation.json",
    "test": "metrics_test.json",
    "validation_calibrated": "metrics_validation_calibrated.json",
    "test_calibrated": "metrics_test_calibrated.json",
}

_IMAGE_FILES: dict[str, str] = {
    "confusion_matrix": "confusion_matrix_test.png",
    "reliability_diagram": "reliability_diagram_test.png",
    "confidence_histogram": "confidence_histogram_test.png",
    "entropy_histogram": "entropy_histogram_test.png",
}


@dataclass(frozen=True)
class DashboardData:
    evaluation_dir: Path
    bundle_dir: Path
    metrics_validation: dict[str, object] | None
    metrics_test: dict[str, object] | None
    metrics_validation_calibrated: dict[str, object] | None
    metrics_test_calibrated: dict[str, object] | None
    image_paths: dict[str, Path | None]
    bundle_metadata: dict[str, object] | None
    champion_manifest: dict[str, object] | None
    warnings: tuple[str, ...]


def load_dashboard_data(*, evaluation_dir: Path, bundle_dir: Path) -> DashboardData:
    warnings: list[str] = []
    metrics_validation = _load_json_mapping(
        evaluation_dir / _METRICS_FILES["validation"],
        required=True,
        warnings=warnings,
    )
    metrics_test = _load_json_mapping(
        evaluation_dir / _METRICS_FILES["test"],
        required=True,
        warnings=warnings,
    )
    metrics_validation_calibrated = _load_json_mapping(
        evaluation_dir / _METRICS_FILES["validation_calibrated"],
        required=False,
        warnings=warnings,
    )
    metrics_test_calibrated = _load_json_mapping(
        evaluation_dir / _METRICS_FILES["test_calibrated"],
        required=False,
        warnings=warnings,
    )

    image_paths = {
        key: _resolve_optional_path(evaluation_dir / filename, warnings)
        for key, filename in _IMAGE_FILES.items()
    }

    bundle_metadata = _load_json_mapping(
        bundle_dir / "bundle_metadata.json",
        required=False,
        warnings=warnings,
    )
    champion_manifest = _load_json_mapping(
        bundle_dir / "champion_manifest.json",
        required=False,
        warnings=warnings,
    )
    return DashboardData(
        evaluation_dir=evaluation_dir,
        bundle_dir=bundle_dir,
        metrics_validation=metrics_validation,
        metrics_test=metrics_test,
        metrics_validation_calibrated=metrics_validation_calibrated,
        metrics_test_calibrated=metrics_test_calibrated,
        image_paths=image_paths,
        bundle_metadata=bundle_metadata,
        champion_manifest=champion_manifest,
        warnings=tuple(warnings),
    )


def metric_number(metrics: dict[str, object] | None, key: str) -> float | None:
    if metrics is None:
        return None
    value = metrics.get(key)
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def uncertainty_summary(metrics: dict[str, object] | None) -> dict[str, float | None]:
    if metrics is None:
        return {
            "mean_confidence": None,
            "mean_entropy": None,
        }
    raw_summary = metrics.get("uncertainty_summary")
    if not isinstance(raw_summary, dict):
        return {
            "mean_confidence": None,
            "mean_entropy": None,
        }
    summary = cast(dict[str, object], raw_summary)
    return {
        "mean_confidence": metric_number(summary, "mean_confidence"),
        "mean_entropy": metric_number(summary, "mean_entropy"),
    }


def per_class_f1_rows(
    metrics: dict[str, object] | None,
    *,
    labels: tuple[str, ...] = CLASS_LABELS,
) -> list[dict[str, float | str]]:
    if metrics is None:
        return []
    raw_values = metrics.get("per_class_f1")
    if not isinstance(raw_values, list):
        return []
    rows: list[dict[str, float | str]] = []
    for index, label in enumerate(labels):
        if index >= len(raw_values):
            break
        value = raw_values[index]
        if isinstance(value, bool):
            continue
        if not isinstance(value, (int, float)):
            continue
        rows.append({"class": label, "f1": float(value)})
    return rows


def calibration_comparison_rows(
    *,
    raw_metrics: dict[str, object] | None,
    calibrated_metrics: dict[str, object] | None,
) -> list[dict[str, float | str | None]]:
    if raw_metrics is None or calibrated_metrics is None:
        return []
    metric_keys = ("accuracy", "macro_f1", "nll", "brier_score", "ece")
    rows: list[dict[str, float | str | None]] = []
    for key in metric_keys:
        rows.append(
            {
                "metric": key,
                "raw": metric_number(raw_metrics, key),
                "calibrated": metric_number(calibrated_metrics, key),
            }
        )
    return rows


def _load_json_mapping(
    path: Path,
    *,
    required: bool,
    warnings: list[str],
) -> dict[str, object] | None:
    if not path.exists():
        if required:
            warnings.append(f"Missing required file: {path}")
        return None
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Expected JSON object in {path}, received {type(payload).__name__}.")
    return cast(dict[str, object], payload)


def _resolve_optional_path(path: Path, warnings: list[str]) -> Path | None:
    if path.exists():
        return path
    warnings.append(f"Visualization not found: {path}")
    return None
