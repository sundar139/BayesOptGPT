from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import optuna

from bayes_gp_llmops.data.config import DataPipelineConfig, load_data_config
from bayes_gp_llmops.evaluation.pipeline import run_evaluation_pipeline
from bayes_gp_llmops.models.config import ModelConfig, load_model_config
from bayes_gp_llmops.tracking.mlflow_utils import (
    log_artifact_files,
    log_metrics,
    log_parameters,
    start_mlflow_run,
    write_json,
)
from bayes_gp_llmops.training.config import TrainConfig, load_train_config
from bayes_gp_llmops.training.pipeline import run_training_pipeline

from .search_space import SampledHyperparameters, sample_hyperparameters


@dataclass(frozen=True)
class ObjectiveSettings:
    """Settings passed from the Optuna runner to each objective invocation."""

    study_name: str
    trials_dir: Path
    debug_mode: bool
    enable_mlflow: bool
    mlflow_experiment_name: str
    enable_temperature_scaling: bool
    log_trial_artifacts: bool
    device_override: str | None
    mlflow_nested: bool = True


@dataclass(frozen=True)
class MergedTrialConfigs:
    """Merged configs after applying sampled hyperparameters."""

    data_config: DataPipelineConfig
    model_config: ModelConfig
    train_config: TrainConfig


class TuningObjective:
    """Optuna objective that reuses the training and evaluation pipelines."""

    def __init__(
        self,
        *,
        data_config_path: Path,
        model_config_path: Path,
        train_config_path: Path,
        settings: ObjectiveSettings,
    ) -> None:
        self._data_config_path = data_config_path
        self._model_config_path = model_config_path
        self._train_config_path = train_config_path
        self._base_data_config = load_data_config(data_config_path)
        self._base_model_config = load_model_config(model_config_path)
        self._base_train_config = load_train_config(train_config_path)
        self._settings = settings
        self._settings.trials_dir.mkdir(parents=True, exist_ok=True)

    def __call__(self, trial: optuna.trial.Trial) -> float:
        try:
            sampled = sample_hyperparameters(
                trial=trial,
                base_model_config=self._base_model_config,
            )
            merged = build_trial_configs(
                base_data_config=self._base_data_config,
                base_model_config=self._base_model_config,
                base_train_config=self._base_train_config,
                sampled=sampled,
                trial_number=trial.number,
                trials_dir=self._settings.trials_dir,
            )

            trial_dir = merged.train_config.checkpoint_dir.parent
            evaluation_dir = trial_dir / "evaluation"
            trial_summary_path = trial_dir / "trial_summary.json"

            with start_mlflow_run(
                enabled=self._settings.enable_mlflow,
                experiment_name=self._settings.mlflow_experiment_name,
                run_name=f"trial-{trial.number}",
                nested=self._settings.mlflow_nested,
                tags={
                    "study_name": self._settings.study_name,
                    "trial_number": str(trial.number),
                },
            ):
                log_parameters(sampled.as_dict(), enabled=self._settings.enable_mlflow)
                log_parameters(
                    {
                        "trial_dir": str(trial_dir),
                        "temperature_scaling_enabled": self._settings.enable_temperature_scaling,
                    },
                    enabled=self._settings.enable_mlflow,
                )

                training_artifacts = run_training_pipeline(
                    data_config_path=self._data_config_path,
                    model_config_path=self._model_config_path,
                    train_config_path=self._train_config_path,
                    device_override=self._settings.device_override,
                    debug_mode=self._settings.debug_mode,
                    data_config_override=merged.data_config,
                    model_config_override=merged.model_config,
                    train_config_override=merged.train_config,
                    mlflow_enabled=False,
                )
                _report_intermediate_validation_scores(
                    trial=trial,
                    history_path=training_artifacts.history_path,
                )

                evaluation_artifacts = run_evaluation_pipeline(
                    data_config_path=self._data_config_path,
                    model_config_path=self._model_config_path,
                    train_config_path=self._train_config_path,
                    checkpoint_path=training_artifacts.best_checkpoint_path,
                    device_override=self._settings.device_override,
                    output_dir=evaluation_dir,
                    enable_temperature_scaling=self._settings.enable_temperature_scaling,
                    debug_mode=self._settings.debug_mode,
                    data_config_override=merged.data_config,
                    model_config_override=merged.model_config,
                    train_config_override=merged.train_config,
                    mlflow_enabled=False,
                )
                validation_metrics = evaluation_artifacts.metrics_validation
                objective_value = _required_metric(validation_metrics, "macro_f1")
                secondary_metrics = {
                    "objective_validation_macro_f1": objective_value,
                    "validation_nll": _required_metric(validation_metrics, "nll"),
                    "validation_brier_score": _required_metric(validation_metrics, "brier_score"),
                    "validation_ece": _required_metric(validation_metrics, "ece"),
                }
                if evaluation_artifacts.metrics_validation_calibrated is not None:
                    calibrated = evaluation_artifacts.metrics_validation_calibrated
                    secondary_metrics["validation_calibrated_macro_f1"] = _required_metric(
                        calibrated,
                        "macro_f1",
                    )
                    secondary_metrics["validation_calibrated_nll"] = _required_metric(
                        calibrated,
                        "nll",
                    )
                    secondary_metrics["validation_calibrated_brier_score"] = _required_metric(
                        calibrated,
                        "brier_score",
                    )
                    secondary_metrics["validation_calibrated_ece"] = _required_metric(
                        calibrated,
                        "ece",
                    )
                log_metrics(secondary_metrics, enabled=self._settings.enable_mlflow)

                trial_summary = {
                    "trial_number": trial.number,
                    "objective_validation_macro_f1": objective_value,
                    "sampled_hyperparameters": sampled.as_dict(),
                    "trial_dir": str(trial_dir),
                    "checkpoint_path": str(training_artifacts.best_checkpoint_path),
                    "evaluation_dir": str(evaluation_artifacts.output_dir),
                }
                write_json(trial_summary_path, trial_summary)
                trial.set_user_attr("trial_dir", str(trial_dir))
                trial.set_user_attr("objective_validation_macro_f1", objective_value)
                trial.set_user_attr("validation_nll", secondary_metrics["validation_nll"])
                trial.set_user_attr(
                    "validation_brier_score",
                    secondary_metrics["validation_brier_score"],
                )
                trial.set_user_attr("validation_ece", secondary_metrics["validation_ece"])

                if self._settings.log_trial_artifacts:
                    log_artifact_files(
                        [
                            training_artifacts.history_path,
                            training_artifacts.resolved_config_path,
                            evaluation_artifacts.metrics_validation_path,
                            evaluation_artifacts.metrics_test_path,
                            evaluation_artifacts.temperature_scaling_path,
                            evaluation_artifacts.confusion_matrix_plot_path,
                            evaluation_artifacts.reliability_diagram_plot_path,
                            evaluation_artifacts.confidence_histogram_plot_path,
                            trial_summary_path,
                        ],
                        enabled=self._settings.enable_mlflow,
                        artifact_path=f"tuning/trial_{trial.number:04d}",
                    )
                    if evaluation_artifacts.metrics_validation_calibrated_path is not None:
                        log_artifact_files(
                            [evaluation_artifacts.metrics_validation_calibrated_path],
                            enabled=self._settings.enable_mlflow,
                            artifact_path=f"tuning/trial_{trial.number:04d}",
                        )
                    if evaluation_artifacts.metrics_test_calibrated_path is not None:
                        log_artifact_files(
                            [evaluation_artifacts.metrics_test_calibrated_path],
                            enabled=self._settings.enable_mlflow,
                            artifact_path=f"tuning/trial_{trial.number:04d}",
                        )

            return objective_value
        except Exception as error:
            trial.set_user_attr("trial_error", str(error))
            raise


def build_trial_configs(
    *,
    base_data_config: DataPipelineConfig,
    base_model_config: ModelConfig,
    base_train_config: TrainConfig,
    sampled: SampledHyperparameters,
    trial_number: int,
    trials_dir: Path,
) -> MergedTrialConfigs:
    """Build merged per-trial configs from sampled hyperparameters."""

    trial_dir = trials_dir / f"trial_{trial_number:04d}"
    checkpoints_dir = trial_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    data_config = base_data_config.model_copy(
        update={
            "dataloader": base_data_config.dataloader.model_copy(
                update={"batch_size": sampled.batch_size}
            )
        }
    )
    model_config = base_model_config.model_copy(
        update={
            "hidden_size": sampled.hidden_size,
            "num_layers": sampled.num_layers,
            "num_attention_heads": sampled.num_attention_heads,
            "feedforward_multiplier": sampled.feedforward_multiplier,
            "dropout": sampled.dropout,
        }
    )
    train_config = base_train_config.model_copy(
        update={
            "learning_rate": sampled.learning_rate,
            "weight_decay": sampled.weight_decay,
            "warmup_ratio": sampled.warmup_ratio,
            "gradient_clip_norm": sampled.gradient_clip_norm,
            "checkpoint_dir": checkpoints_dir,
        }
    )
    return MergedTrialConfigs(
        data_config=data_config,
        model_config=model_config,
        train_config=train_config,
    )


def _report_intermediate_validation_scores(
    *,
    trial: optuna.trial.Trial,
    history_path: Path,
) -> None:
    with history_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError("Training history must be a list.")
    for epoch_index, item in enumerate(payload, start=1):
        if not isinstance(item, dict):
            continue
        validation = item.get("validation")
        if not isinstance(validation, dict):
            continue
        metric = validation.get("macro_f1")
        if not isinstance(metric, (int, float)):
            continue
        trial.report(float(metric), step=epoch_index)
        if trial.should_prune():
            raise optuna.TrialPruned(
                f"Trial {trial.number} pruned at epoch {epoch_index} with macro_f1={metric:.6f}"
            )


def _required_metric(metrics: dict[str, object], key: str) -> float:
    value = metrics.get(key)
    if isinstance(value, (int, float)):
        return float(value)
    raise ValueError(f"Expected numeric metric '{key}' in payload.")
