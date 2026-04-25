"""Champion model selection and promotion.

This module identifies the best candidate from a set of trained or tuned models,
records its provenance in a durable manifest, and supports deterministic reloading.

Selection policy (documented and testable):
  1. Maximize validation_macro_f1 (primary metric, descending).
  2. Lower trial_number wins on ties (ascending, reproducible).
  3. Lexicographic checkpoint_path string breaks any remaining ties.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from bayes_gp_llmops.tracking.mlflow_utils import write_json

LOGGER = logging.getLogger("bayes_gp_llmops.serving.champion")

SELECTION_POLICY = (
    "Primary: maximize validation_macro_f1 (descending). "
    "Tie-breaking: lower trial_number wins (ascending). "
    "Final tie-breaking: lexicographic checkpoint_path string (ascending)."
)


@dataclass(frozen=True)
class CandidateMetrics:
    """Metrics and provenance for a single candidate model."""

    study_name: str
    trial_number: int
    checkpoint_path: Path
    validation_macro_f1: float
    validation_nll: float | None = None
    validation_brier: float | None = None
    validation_ece: float | None = None
    validation_macro_f1_calibrated: float | None = None
    validation_nll_calibrated: float | None = None
    config_snapshot: dict[str, object] = field(default_factory=dict)


class ChampionManifest(BaseModel):
    """Durable provenance record for the promoted champion model."""

    schema_version: str = Field(default="1.0")
    study_name: str
    trial_number: int
    checkpoint_path: str
    selected_metrics: dict[str, float | None]
    config_snapshot: dict[str, object]
    timestamp_utc: str
    selection_policy: str = Field(default=SELECTION_POLICY)


def select_champion(candidates: list[CandidateMetrics]) -> CandidateMetrics:
    """Select the best candidate using the deterministic selection policy.

    Selection policy (in priority order):
      1. Maximize validation_macro_f1 (primary metric).
      2. Lower trial_number wins on ties (ascending, reproducible).
      3. Lexicographic checkpoint_path string breaks any remaining ties.

    Args:
        candidates: Non-empty list of evaluated candidate models.

    Returns:
        The single best CandidateMetrics according to the policy.

    Raises:
        ValueError: If candidates is empty.
    """
    if not candidates:
        raise ValueError("Cannot select champion from an empty candidate list.")
    return min(
        candidates,
        key=lambda c: (
            -c.validation_macro_f1,
            c.trial_number,
            str(c.checkpoint_path),
        ),
    )


def build_champion_manifest(champion: CandidateMetrics) -> ChampionManifest:
    """Build a champion manifest from the selected candidate."""
    return ChampionManifest(
        study_name=champion.study_name,
        trial_number=champion.trial_number,
        checkpoint_path=str(champion.checkpoint_path.resolve()),
        selected_metrics={
            "validation_macro_f1": champion.validation_macro_f1,
            "validation_nll": champion.validation_nll,
            "validation_brier": champion.validation_brier,
            "validation_ece": champion.validation_ece,
            "validation_macro_f1_calibrated": champion.validation_macro_f1_calibrated,
            "validation_nll_calibrated": champion.validation_nll_calibrated,
        },
        config_snapshot=champion.config_snapshot,
        timestamp_utc=datetime.now(tz=UTC).isoformat(),
    )


def write_champion_manifest(manifest: ChampionManifest, output_dir: Path) -> Path:
    """Write the champion manifest JSON to output_dir/champion_manifest.json.

    Args:
        manifest: Populated champion manifest.
        output_dir: Directory in which to write champion_manifest.json.

    Returns:
        Path to the written manifest file.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    path = output_dir / "champion_manifest.json"
    write_json(path, manifest.model_dump(mode="json"))
    LOGGER.info("Champion manifest written: %s", path)
    return path


def load_champion_manifest(path: Path) -> ChampionManifest:
    """Load and validate a champion manifest from disk.

    Args:
        path: Path to champion_manifest.json.

    Returns:
        Validated ChampionManifest instance.
    """
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return ChampionManifest.model_validate(payload)


def load_candidates_from_tuning_dir(
    tuning_dir: Path,
    *,
    study_name: str,
) -> list[CandidateMetrics]:
    """Build a candidate list from a structured tuning output directory.

    Scans each trial_XXXX subdirectory for a valid checkpoint and
    validation metrics, skipping incomplete or failed trials.

    Expected directory layout::

        tuning_dir/
          trials/
            trial_0000/
              checkpoints/
                best.ckpt
                resolved_config.json
              evaluation/
                metrics_validation.json
                metrics_validation_calibrated.json  (optional)

    Args:
        tuning_dir: Root directory of the tuning study output.
        study_name: Name of the Optuna study (for provenance).

    Returns:
        List of CandidateMetrics for valid trials; non-empty.

    Raises:
        FileNotFoundError: If the trials/ subdirectory is missing.
        ValueError: If no valid candidates are found.
    """
    trials_dir = tuning_dir / "trials"
    if not trials_dir.is_dir():
        raise FileNotFoundError(f"Trials directory not found: {trials_dir}")

    candidates: list[CandidateMetrics] = []
    for trial_dir in sorted(trials_dir.iterdir()):
        if not trial_dir.is_dir() or not trial_dir.name.startswith("trial_"):
            continue
        suffix = trial_dir.name.removeprefix("trial_")
        try:
            trial_number = int(suffix)
        except ValueError:
            LOGGER.warning("Skipping non-numeric trial directory: %s", trial_dir.name)
            continue

        checkpoint_path = trial_dir / "checkpoints" / "best.ckpt"
        if not checkpoint_path.exists():
            LOGGER.warning("No checkpoint found for trial %d; skipping.", trial_number)
            continue

        metrics_path = trial_dir / "evaluation" / "metrics_validation.json"
        if not metrics_path.exists():
            LOGGER.warning("No validation metrics for trial %d; skipping.", trial_number)
            continue

        metrics = _read_json_file(metrics_path)
        macro_f1 = _extract_float(metrics, "macro_f1")
        if macro_f1 is None:
            LOGGER.warning("macro_f1 missing for trial %d; skipping.", trial_number)
            continue

        calibrated_path = trial_dir / "evaluation" / "metrics_validation_calibrated.json"
        calibrated: dict[str, object] = {}
        if calibrated_path.exists():
            calibrated = _read_json_file(calibrated_path)

        resolved_config_path = trial_dir / "checkpoints" / "resolved_config.json"
        config_snapshot: dict[str, object] = {}
        if resolved_config_path.exists():
            config_snapshot = _read_json_file(resolved_config_path)

        candidates.append(
            CandidateMetrics(
                study_name=study_name,
                trial_number=trial_number,
                checkpoint_path=checkpoint_path,
                validation_macro_f1=macro_f1,
                validation_nll=_extract_float(metrics, "nll"),
                validation_brier=_extract_float(metrics, "brier_score"),
                validation_ece=_extract_float(metrics, "ece"),
                validation_macro_f1_calibrated=_extract_float(calibrated, "macro_f1"),
                validation_nll_calibrated=_extract_float(calibrated, "nll"),
                config_snapshot=config_snapshot,
            )
        )

    if not candidates:
        raise ValueError(f"No valid candidates found in tuning directory: {tuning_dir}")

    LOGGER.info("Loaded %d candidate(s) from %s", len(candidates), tuning_dir)
    return candidates


def _read_json_file(path: Path) -> dict[str, object]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"JSON file must contain an object: {path}")
    return {str(k): v for k, v in payload.items()}


def _extract_float(metrics: dict[str, object], key: str) -> float | None:
    value = metrics.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    return None
