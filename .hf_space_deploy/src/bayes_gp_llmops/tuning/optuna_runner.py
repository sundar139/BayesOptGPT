from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

import optuna
import yaml
from pydantic import BaseModel, Field

from bayes_gp_llmops.tracking.mlflow_utils import (
    log_artifact_files,
    log_metrics,
    log_parameters,
    start_mlflow_run,
    write_json,
)

from .objective import ObjectiveSettings, TuningObjective

SamplerType = Literal["tpe", "random"]
PrunerType = Literal["median", "successive_halving", "none"]
DirectionType = Literal["maximize", "minimize"]


class TuneConfig(BaseModel):
    """Configuration for Optuna hyperparameter optimization."""

    study_name: str = Field(default="bayes-gp-llmops")
    storage_path: Path = Field(default=Path("artifacts/tuning/study.db"))
    output_dir: Path = Field(default=Path("artifacts/tuning"))
    sampler: SamplerType = Field(default="tpe")
    pruner: PrunerType = Field(default="median")
    n_trials: int = Field(default=10, ge=1)
    timeout_seconds: int | None = Field(default=None, ge=1)
    direction: DirectionType = Field(default="maximize")
    seed: int = Field(default=42)
    debug_mode: bool = Field(default=False)
    enable_mlflow: bool = Field(default=True)
    mlflow_experiment_name: str = Field(default="bayes-gp-llmops-tuning")
    enable_temperature_scaling: bool = Field(default=True)
    log_trial_artifacts: bool = Field(default=True)


@dataclass(frozen=True)
class StudyArtifacts:
    """Persisted outputs from a completed tuning study."""

    output_dir: Path
    storage_path: Path
    best_params_path: Path
    trial_results_path: Path
    study_summary_path: Path
    best_value: float
    best_trial_number: int
    best_params: dict[str, float | int | str]


def run_optuna_study(
    *,
    data_config_path: Path,
    model_config_path: Path,
    train_config_path: Path,
    tune_config_path: Path,
    device_override: str | None,
    n_trials_override: int | None,
    timeout_override: int | None,
    debug_override: bool | None,
) -> StudyArtifacts:
    """Run Optuna study for the AG News classifier and persist outputs."""

    tune_config = load_tune_config(tune_config_path)
    if n_trials_override is not None:
        tune_config = tune_config.model_copy(update={"n_trials": n_trials_override})
    if timeout_override is not None:
        tune_config = tune_config.model_copy(update={"timeout_seconds": timeout_override})
    if debug_override is not None:
        tune_config = tune_config.model_copy(update={"debug_mode": debug_override})

    tune_config.output_dir.mkdir(parents=True, exist_ok=True)
    tune_config.storage_path.parent.mkdir(parents=True, exist_ok=True)
    study_storage_uri = _sqlite_storage_uri(tune_config.storage_path)

    sampler = _build_sampler(tune_config)
    pruner = _build_pruner(tune_config)
    study = optuna.create_study(
        study_name=tune_config.study_name,
        storage=study_storage_uri,
        sampler=sampler,
        pruner=pruner,
        direction=tune_config.direction,
        load_if_exists=True,
    )

    objective = TuningObjective(
        data_config_path=data_config_path,
        model_config_path=model_config_path,
        train_config_path=train_config_path,
        settings=ObjectiveSettings(
            study_name=tune_config.study_name,
            trials_dir=tune_config.output_dir / "trials",
            debug_mode=tune_config.debug_mode,
            enable_mlflow=tune_config.enable_mlflow,
            mlflow_experiment_name=tune_config.mlflow_experiment_name,
            enable_temperature_scaling=tune_config.enable_temperature_scaling,
            log_trial_artifacts=tune_config.log_trial_artifacts,
            device_override=device_override,
            mlflow_nested=True,
        ),
    )

    with start_mlflow_run(
        enabled=tune_config.enable_mlflow,
        experiment_name=tune_config.mlflow_experiment_name,
        run_name=f"study-{tune_config.study_name}",
        nested=False,
        tags={"component": "optuna-study", "study_name": tune_config.study_name},
    ):
        log_parameters(tune_config.model_dump(mode="python"), enabled=tune_config.enable_mlflow)
        study.optimize(
            objective,
            n_trials=tune_config.n_trials,
            timeout=tune_config.timeout_seconds,
            gc_after_trial=True,
            catch=(RuntimeError, ValueError, OSError, FileNotFoundError),
        )
        artifacts = _persist_study_results(
            study=study,
            output_dir=tune_config.output_dir,
            storage_path=tune_config.storage_path,
        )
        log_metrics(
            {
                "study_best_value": artifacts.best_value,
                "study_best_trial": artifacts.best_trial_number,
                "study_trial_count": float(len(study.trials)),
            },
            enabled=tune_config.enable_mlflow,
        )
        log_artifact_files(
            [
                artifacts.best_params_path,
                artifacts.trial_results_path,
                artifacts.study_summary_path,
            ],
            enabled=tune_config.enable_mlflow,
            artifact_path="tuning/study",
        )
    return artifacts


def load_tune_config(path: Path) -> TuneConfig:
    """Load tuning configuration from YAML."""

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration file {path} must contain a mapping at the top level.")
    tuning_section = payload.get("tuning")
    if not isinstance(tuning_section, dict):
        raise ValueError(f"Configuration file {path} must include a 'tuning' mapping.")
    return TuneConfig.model_validate(tuning_section)


def _persist_study_results(
    *,
    study: optuna.study.Study,
    output_dir: Path,
    storage_path: Path,
) -> StudyArtifacts:
    output_dir.mkdir(parents=True, exist_ok=True)
    best_params_path = output_dir / "best_params.json"
    trial_results_path = output_dir / "trial_results.csv"
    study_summary_path = output_dir / "study_summary.json"

    best_trial = study.best_trial
    if best_trial.value is None:
        raise ValueError("Best trial value is missing.")
    best_value = float(best_trial.value)
    best_params_payload: dict[str, float | int | str] = {
        key: _coerce_serializable(value) for key, value in best_trial.params.items()
    }
    write_json(
        best_params_path,
        {
            "best_trial_number": best_trial.number,
            "best_value": best_value,
            "best_params": best_params_payload,
        },
    )
    _write_trial_results_csv(study=study, output_path=trial_results_path)
    study_summary = {
        "study_name": study.study_name,
        "direction": study.direction.name.lower(),
        "n_trials": len(study.trials),
        "best_trial_number": best_trial.number,
        "best_value": best_value,
        "best_params": best_params_payload,
    }
    write_json(study_summary_path, study_summary)
    return StudyArtifacts(
        output_dir=output_dir,
        storage_path=storage_path.resolve(),
        best_params_path=best_params_path,
        trial_results_path=trial_results_path,
        study_summary_path=study_summary_path,
        best_value=best_value,
        best_trial_number=best_trial.number,
        best_params=best_params_payload,
    )


def _write_trial_results_csv(*, study: optuna.study.Study, output_path: Path) -> None:
    field_names = [
        "trial_number",
        "state",
        "value",
        "duration_seconds",
        "params_json",
        "user_attrs_json",
    ]
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=field_names)
        writer.writeheader()
        for trial in study.trials:
            duration = trial.duration.total_seconds() if trial.duration is not None else None
            writer.writerow(
                {
                    "trial_number": trial.number,
                    "state": trial.state.name.lower(),
                    "value": "" if trial.value is None else float(trial.value),
                    "duration_seconds": "" if duration is None else duration,
                    "params_json": json.dumps(trial.params, sort_keys=True),
                    "user_attrs_json": json.dumps(trial.user_attrs, sort_keys=True),
                }
            )


def _build_sampler(config: TuneConfig) -> optuna.samplers.BaseSampler:
    if config.sampler == "random":
        return optuna.samplers.RandomSampler(seed=config.seed)
    return optuna.samplers.TPESampler(seed=config.seed, n_startup_trials=5, multivariate=True)


def _build_pruner(config: TuneConfig) -> optuna.pruners.BasePruner:
    if config.pruner == "none":
        return optuna.pruners.NopPruner()
    if config.pruner == "successive_halving":
        return optuna.pruners.SuccessiveHalvingPruner()
    return optuna.pruners.MedianPruner(n_startup_trials=3, n_warmup_steps=1)


def _sqlite_storage_uri(path: Path) -> str:
    resolved = path.resolve()
    return f"sqlite:///{resolved.as_posix()}"


def _coerce_serializable(value: object) -> float | int | str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, (float, int, str)):
        return value
    return str(value)
