from __future__ import annotations

import logging
from pathlib import Path

import uvicorn

from .api import create_app
from .config import get_settings
from .evaluation.pipeline import run_evaluation_pipeline
from .serving.bundle import package_inference_bundle, validate_bundle
from .serving.champion import (
    build_champion_manifest,
    load_candidates_from_tuning_dir,
    select_champion,
    write_champion_manifest,
)
from .serving.config import load_serving_config, resolve_serving_config_path
from .serving.runtime import ServingRuntime
from .training.pipeline import run_training_pipeline
from .tuning.optuna_runner import run_optuna_study

LOGGER = logging.getLogger("bayes_gp_llmops")

_AG_NEWS_LABEL_MAP = {
    "0": "World",
    "1": "Sports",
    "2": "Business",
    "3": "Sci/Tech",
}


def _configure_logging(log_level: str | None = None) -> None:
    active_log_level = log_level
    if active_log_level is None:
        settings = get_settings()
        active_log_level = settings.log_level
    level = getattr(logging, active_log_level.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )


def run_train() -> None:
    _configure_logging()
    artifacts = run_training_pipeline(
        data_config_path=Path("configs/data.yaml"),
        model_config_path=Path("configs/model.yaml"),
        train_config_path=Path("configs/train.yaml"),
        device_override=None,
        debug_mode=False,
    )
    LOGGER.info("Training complete. best_checkpoint=%s", artifacts.best_checkpoint_path)


def run_evaluate() -> None:
    _configure_logging()
    artifacts = run_evaluation_pipeline(
        data_config_path=Path("configs/data.yaml"),
        model_config_path=Path("configs/model.yaml"),
        train_config_path=Path("configs/train.yaml"),
        checkpoint_path=None,
        device_override=None,
        output_dir=Path("artifacts/evaluation"),
        enable_temperature_scaling=True,
        debug_mode=False,
    )
    LOGGER.info("Evaluation complete. output_dir=%s", artifacts.output_dir)


def run_tune() -> None:
    _configure_logging()
    artifacts = run_optuna_study(
        data_config_path=Path("configs/data.yaml"),
        model_config_path=Path("configs/model.yaml"),
        train_config_path=Path("configs/train.yaml"),
        tune_config_path=Path("configs/tune.yaml"),
        device_override=None,
        n_trials_override=None,
        timeout_override=None,
        debug_override=None,
    )
    LOGGER.info("Tuning complete. best_trial=%d", artifacts.best_trial_number)
    LOGGER.info("Tuning artifacts at %s", artifacts.output_dir)


def run_serve() -> None:
    serving_config_path = resolve_serving_config_path()
    serving_config = load_serving_config(serving_config_path)
    _configure_logging(serving_config.log_level)

    runtime = ServingRuntime.load_from_bundle(serving_config)

    print(f"bundle_dir={serving_config.bundle_dir}")
    print("bundle_validation=passed")
    print(f"host={serving_config.host}")
    print(f"port={serving_config.port}")
    print(f"calibration_enabled={str(runtime.calibration_active).lower()}")

    application = create_app(serving_config=serving_config, runtime=runtime)
    uvicorn.run(
        application,
        host=serving_config.host,
        port=serving_config.port,
        reload=False,
        log_level=serving_config.log_level.lower(),
    )


def run_promote() -> None:
    """Select champion from the default tuning directory and package an inference bundle."""
    import json

    _configure_logging()
    tuning_dir = Path("artifacts/tuning")
    output_dir = Path("artifacts/model/bundle")
    tokenizer_dir = Path("artifacts/tokenizer")

    study_summary_path = tuning_dir / "study_summary.json"
    if not study_summary_path.exists():
        raise FileNotFoundError(
            f"study_summary.json not found in {tuning_dir}. Run bayes-tune first."
        )
    with study_summary_path.open("r", encoding="utf-8") as handle:
        summary_payload = json.load(handle)
    study_name = str(summary_payload.get("study_name", "bayes-gp-llmops"))

    candidates = load_candidates_from_tuning_dir(tuning_dir, study_name=study_name)
    champion = select_champion(candidates)
    LOGGER.info(
        "Champion: trial=%d macro_f1=%.4f",
        champion.trial_number,
        champion.validation_macro_f1,
    )

    config_snapshot = champion.config_snapshot
    model_section = config_snapshot.get("model")
    data_section = config_snapshot.get("data")
    if not isinstance(model_section, dict) or not isinstance(data_section, dict):
        raise ValueError("Champion config_snapshot is missing 'model' or 'data' section.")

    candidate_cal_path = (
        tuning_dir / "trials" / f"trial_{champion.trial_number:04d}"
        / "evaluation" / "temperature_scaling.json"
    )
    calibration_path: Path | None = candidate_cal_path if candidate_cal_path.exists() else None

    manifest = build_champion_manifest(champion)
    manifest_path = write_champion_manifest(manifest, output_dir)

    bundle_dir = package_inference_bundle(
        champion_manifest=manifest,
        tokenizer_dir=tokenizer_dir,
        model_config_dict=dict(model_section),
        data_config_dict=dict(data_section),
        output_dir=output_dir,
        label_map=_AG_NEWS_LABEL_MAP,
        calibration_path=calibration_path,
    )
    validate_bundle(bundle_dir)
    LOGGER.info("Champion promoted. bundle_dir=%s manifest=%s", bundle_dir, manifest_path)


def run_validate_bundle() -> None:
    """Validate the default inference bundle's required files and checksums."""
    _configure_logging()
    bundle_dir = Path("artifacts/model/bundle")
    metadata = validate_bundle(bundle_dir)
    LOGGER.info(
        "Bundle valid. trial=%d study=%s created_at=%s",
        metadata.champion_trial_number,
        metadata.champion_study_name,
        metadata.created_at,
    )
