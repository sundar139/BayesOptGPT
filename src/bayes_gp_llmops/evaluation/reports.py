from __future__ import annotations

import csv
import json
from pathlib import Path

import matplotlib
import numpy as np
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def ensure_directory(path: Path) -> Path:
    """Ensure a directory exists and return it."""

    path.mkdir(parents=True, exist_ok=True)
    return path


def write_json(path: Path, payload: dict[str, object]) -> None:
    """Write a JSON payload with deterministic formatting."""

    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True)
        handle.write("\n")


def export_predictions_csv(
    *,
    path: Path,
    split: str,
    labels: torch.Tensor,
    predictions: torch.Tensor,
    confidence: torch.Tensor,
    entropy: torch.Tensor,
    margin: torch.Tensor,
    logits: torch.Tensor,
    probabilities: torch.Tensor,
) -> Path:
    """Export prediction-level outputs to CSV."""

    sample_count = labels.size(0)
    rows: dict[str, list[float | int | str | bool]] = {
        "split": [split] * sample_count,
        "index": list(range(sample_count)),
        "true_label": labels.detach().cpu().tolist(),
        "predicted_label": predictions.detach().cpu().tolist(),
        "confidence": confidence.detach().cpu().tolist(),
        "entropy": entropy.detach().cpu().tolist(),
        "margin": margin.detach().cpu().tolist(),
        "correct": predictions.eq(labels).detach().cpu().tolist(),
    }

    logits_array = logits.detach().cpu().numpy()
    probabilities_array = probabilities.detach().cpu().numpy()
    for class_index in range(logits_array.shape[1]):
        rows[f"raw_logit_{class_index}"] = logits_array[:, class_index].tolist()
        rows[f"probability_{class_index}"] = probabilities_array[:, class_index].tolist()

    field_names = list(rows.keys())
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        for row_index in range(sample_count):
            row: dict[str, float | int | str | bool] = {
                key: values[row_index] for key, values in rows.items()
            }
            writer.writerow(row)
    return path


def plot_confusion_matrix(
    *,
    path: Path,
    confusion_matrix: np.ndarray,
    class_names: list[str] | None = None,
) -> Path:
    """Plot and persist a confusion matrix heatmap."""

    figure, axis = plt.subplots(figsize=(6, 5))
    image = axis.imshow(confusion_matrix, cmap="Blues")
    axis.figure.colorbar(image, ax=axis)

    axis.set_xlabel("Predicted label")
    axis.set_ylabel("True label")
    axis.set_title("Confusion Matrix")
    tick_count = confusion_matrix.shape[0]
    ticks = np.arange(tick_count)
    axis.set_xticks(ticks)
    axis.set_yticks(ticks)
    if class_names is not None and len(class_names) == tick_count:
        axis.set_xticklabels(class_names, rotation=45, ha="right")
        axis.set_yticklabels(class_names)
    else:
        axis.set_xticklabels([str(index) for index in range(tick_count)])
        axis.set_yticklabels([str(index) for index in range(tick_count)])

    for row_index in range(tick_count):
        for col_index in range(tick_count):
            axis.text(
                col_index,
                row_index,
                str(int(confusion_matrix[row_index, col_index])),
                ha="center",
                va="center",
                color="black",
                fontsize=9,
            )

    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def plot_reliability_diagram(
    *,
    path: Path,
    probabilities: torch.Tensor,
    labels: torch.Tensor,
    num_bins: int = 15,
) -> Path:
    """Plot and persist a reliability diagram from sample confidences."""

    if num_bins < 1:
        raise ValueError("num_bins must be >= 1.")
    confidences, predictions = probabilities.max(dim=1)
    correctness = predictions.eq(labels).to(torch.float32)

    confidence_values = confidences.detach().cpu().numpy()
    accuracy_values = correctness.detach().cpu().numpy()
    bin_edges = np.linspace(0.0, 1.0, num_bins + 1)
    bin_ids = np.digitize(confidence_values, bin_edges[1:-1], right=False)

    binned_accuracy = np.zeros(num_bins, dtype=np.float64)
    binned_confidence = np.zeros(num_bins, dtype=np.float64)
    bin_counts = np.zeros(num_bins, dtype=np.int64)

    for bin_index in range(num_bins):
        mask = bin_ids == bin_index
        if not np.any(mask):
            continue
        bin_counts[bin_index] = int(mask.sum())
        binned_accuracy[bin_index] = float(accuracy_values[mask].mean())
        binned_confidence[bin_index] = float(confidence_values[mask].mean())

    bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2.0

    figure, axis = plt.subplots(figsize=(6, 5))
    axis.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="gray",
        linewidth=1,
        label="Perfect calibration",
    )
    axis.bar(
        bin_centers,
        binned_accuracy,
        width=1.0 / num_bins,
        alpha=0.6,
        edgecolor="black",
        label="Empirical accuracy",
    )
    axis.plot(
        bin_centers,
        binned_confidence,
        marker="o",
        linewidth=1.5,
        color="tab:orange",
        label="Mean confidence",
    )
    axis.set_xlim(0.0, 1.0)
    axis.set_ylim(0.0, 1.0)
    axis.set_xlabel("Confidence")
    axis.set_ylabel("Accuracy")
    axis.set_title("Reliability Diagram")
    axis.legend(loc="best")

    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def plot_confidence_histogram(*, path: Path, confidence: torch.Tensor) -> Path:
    """Plot a confidence histogram."""

    confidence_values = _finite_histogram_values(
        confidence.detach().cpu().numpy(),
        fallback=0.0,
    )
    confidence_values = np.clip(confidence_values, 0.0, 1.0)

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.hist(
        confidence_values,
        bins=20,
        range=(0.0, 1.0),
        edgecolor="black",
        alpha=0.8,
    )
    axis.set_title("Confidence Distribution")
    axis.set_xlabel("Max softmax confidence")
    axis.set_ylabel("Count")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def plot_entropy_histogram(*, path: Path, entropy: torch.Tensor) -> Path:
    """Plot an entropy histogram."""

    entropy_values = _finite_histogram_values(
        entropy.detach().cpu().numpy(),
        fallback=0.0,
    )

    figure, axis = plt.subplots(figsize=(6, 4))
    axis.hist(entropy_values, bins=20, edgecolor="black", alpha=0.8)
    axis.set_title("Predictive Entropy Distribution")
    axis.set_xlabel("Entropy")
    axis.set_ylabel("Count")
    figure.tight_layout()
    figure.savefig(path, dpi=160)
    plt.close(figure)
    return path


def _finite_histogram_values(values: np.ndarray, *, fallback: float) -> np.ndarray:
    """Return finite values for histogram plotting with a deterministic fallback."""

    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        fallback_values: np.ndarray = np.asarray([fallback], dtype=np.float64)
        return fallback_values
    typed_values: np.ndarray = np.asarray(finite_values, dtype=np.float64)
    return typed_values
