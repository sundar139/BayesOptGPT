"""Promote the best tuned model to a portable inference bundle.

Reads tuning artifacts from a completed Optuna study, selects the champion
trial deterministically by validation macro-F1, packages a self-contained
inference bundle, and validates its integrity.

Optionally logs champion metadata and bundle artifacts to MLflow.
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from bayes_gp_llmops.serving.bundle import package_inference_bundle, validate_bundle
from bayes_gp_llmops.serving.champion import (
    build_champion_manifest,
    load_candidates_from_tuning_dir,
    select_champion,
    write_champion_manifest,
)
from bayes_gp_llmops.tracking.mlflow_utils import (
    log_artifact_files,
    log_metrics,
    log_parameters,
    start_mlflow_run,
)

LOGGER = logging.getLogger("bayes_gp_llmops.promote")

AG_NEWS_LABEL_MAP = {
    "0": "World",
    "1": "Sports",
    "2": "Business",
    "3": "Sci/Tech",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select the champion model from a tuning study and package an inference bundle."
    )
    parser.add_argument(
        "--tuning-dir",
        type=Path,
        default=Path("artifacts/tuning"),
        help="Root directory of the tuning study output (default: artifacts/tuning).",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/model/bundle"),
        help="Destination directory for the inference bundle (default: artifacts/model/bundle).",
    )
    parser.add_argument(
        "--tokenizer-dir",
        type=Path,
        default=Path("artifacts/tokenizer"),
        help="Directory containing tokenizer artifacts (default: artifacts/tokenizer).",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Study name override; inferred from study_summary.json if omitted.",
    )
    parser.add_argument(
        "--enable-mlflow",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Log champion metadata and bundle artifacts to MLflow.",
    )
    parser.add_argument(
        "--mlflow-experiment",
        type=str,
        default="bayes-gp-llmops-promotion",
        help="MLflow experiment name (default: bayes-gp-llmops-promotion).",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default="INFO",
        choices=["DEBUG", "INFO", "WARNING", "ERROR"],
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level),
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )

    tuning_dir: Path = args.tuning_dir
    output_dir: Path = args.output_dir
    tokenizer_dir: Path = args.tokenizer_dir

    study_name = args.study_name or _read_study_name(tuning_dir)
    LOGGER.info("Study name: %s", study_name)

    candidates = load_candidates_from_tuning_dir(tuning_dir, study_name=study_name)
    champion = select_champion(candidates)
    LOGGER.info(
        "Champion: trial=%d macro_f1=%.4f checkpoint=%s",
        champion.trial_number,
        champion.validation_macro_f1,
        champion.checkpoint_path,
    )

    model_config_dict, data_config_dict = _extract_configs_from_snapshot(champion.config_snapshot)
    calibration_path = _find_calibration_artifact(tuning_dir, champion.trial_number)

    manifest = build_champion_manifest(champion)
    manifest_path = write_champion_manifest(manifest, output_dir)

    bundle_dir = package_inference_bundle(
        champion_manifest=manifest,
        tokenizer_dir=tokenizer_dir,
        model_config_dict=model_config_dict,
        data_config_dict=data_config_dict,
        output_dir=output_dir,
        label_map=AG_NEWS_LABEL_MAP,
        calibration_path=calibration_path,
    )

    validate_bundle(bundle_dir)

    with start_mlflow_run(
        enabled=args.enable_mlflow,
        experiment_name=args.mlflow_experiment,
        run_name=f"promote-{study_name}-trial{champion.trial_number}",
        tags={
            "component": "champion-promotion",
            "study_name": study_name,
            "promoted_trial": str(champion.trial_number),
        },
    ):
        log_parameters(
            {
                "study_name": study_name,
                "promoted_trial": champion.trial_number,
                "bundle_dir": str(bundle_dir),
                "selection_policy": manifest.selection_policy,
            },
            enabled=args.enable_mlflow,
        )
        champion_metrics: dict[str, float | int] = {
            k: v for k, v in manifest.selected_metrics.items() if v is not None
        }
        log_metrics(champion_metrics, enabled=args.enable_mlflow)
        log_artifact_files(
            [manifest_path, bundle_dir / "bundle_metadata.json"],
            enabled=args.enable_mlflow,
            artifact_path="promotion",
        )

    print(f"champion_trial_number={champion.trial_number}")
    print(f"champion_macro_f1={champion.validation_macro_f1:.6f}")
    print(f"champion_checkpoint={champion.checkpoint_path}")
    print(f"bundle_dir={bundle_dir}")
    print(f"manifest_path={manifest_path}")
    print("bundle_validation=passed")
    return 0


def _read_study_name(tuning_dir: Path) -> str:
    summary_path = tuning_dir / "study_summary.json"
    if not summary_path.exists():
        raise FileNotFoundError(
            f"study_summary.json not found in {tuning_dir}. "
            "Run scripts/tune.py first, or pass --study-name explicitly."
        )
    with summary_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"study_summary.json must be a JSON object: {summary_path}")
    study_name = payload.get("study_name")
    if not isinstance(study_name, str) or not study_name:
        raise ValueError(
            f"study_summary.json missing valid 'study_name' field: {summary_path}"
        )
    return study_name


def _extract_configs_from_snapshot(
    config_snapshot: dict[str, object],
) -> tuple[dict[str, object], dict[str, object]]:
    model_section = config_snapshot.get("model")
    if not isinstance(model_section, dict):
        raise ValueError(
            "Champion config_snapshot is missing 'model' section. "
            "Ensure resolved_config.json was written during training."
        )
    data_section = config_snapshot.get("data")
    if not isinstance(data_section, dict):
        raise ValueError(
            "Champion config_snapshot is missing 'data' section. "
            "Ensure resolved_config.json was written during training."
        )
    return dict(model_section), dict(data_section)


def _find_calibration_artifact(tuning_dir: Path, trial_number: int) -> Path | None:
    candidate = (
        tuning_dir / "trials" / f"trial_{trial_number:04d}" / "evaluation"
        / "temperature_scaling.json"
    )
    if candidate.exists():
        LOGGER.info("Calibration artifact found: %s", candidate)
        return candidate
    LOGGER.info("No calibration artifact found for trial %d; skipping.", trial_number)
    return None


if __name__ == "__main__":
    sys.exit(main())
