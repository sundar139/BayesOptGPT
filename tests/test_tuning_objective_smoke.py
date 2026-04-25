from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import pytest

from bayes_gp_llmops.evaluation.pipeline import EvaluationArtifacts
from bayes_gp_llmops.training.config import TrainConfig
from bayes_gp_llmops.training.trainer import TrainingArtifacts
from bayes_gp_llmops.tuning.objective import ObjectiveSettings, TuningObjective


class _FakeTrial:
    def __init__(self, params: dict[str, float | int], number: int = 3) -> None:
        self.number = number
        self._params = params
        self.params: dict[str, float | int] = {}
        self.user_attrs: dict[str, Any] = {}
        self.reports: list[tuple[int, float]] = []

    def suggest_float(self, name: str, *_: object, **__: object) -> float:
        value = self._params[name]
        self.params[name] = value
        return float(value)

    def suggest_categorical(self, name: str, *_: object, **__: object) -> int | float:
        value = self._params[name]
        self.params[name] = value
        return value

    def suggest_int(self, name: str, *_: object, **__: object) -> int:
        value = self._params[name]
        self.params[name] = value
        return int(value)

    def report(self, value: float, step: int) -> None:
        self.reports.append((step, value))

    def should_prune(self) -> bool:
        return False

    def set_user_attr(self, key: str, value: object) -> None:
        self.user_attrs[key] = value


def test_tuning_objective_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    data_config_path = Path("configs/data.yaml")
    model_config_path = Path("configs/model.yaml")
    train_config_path = Path("configs/train.yaml")
    settings = ObjectiveSettings(
        study_name="unit",
        trials_dir=tmp_path / "trials",
        debug_mode=True,
        enable_mlflow=False,
        mlflow_experiment_name="unit-exp",
        enable_temperature_scaling=True,
        log_trial_artifacts=True,
        device_override="cpu",
        mlflow_nested=True,
    )
    objective = TuningObjective(
        data_config_path=data_config_path,
        model_config_path=model_config_path,
        train_config_path=train_config_path,
        settings=settings,
    )

    monkeypatch.setattr(
        "bayes_gp_llmops.tuning.objective.run_training_pipeline",
        _fake_run_training_pipeline,
    )
    monkeypatch.setattr(
        "bayes_gp_llmops.tuning.objective.run_evaluation_pipeline",
        _fake_run_evaluation_pipeline,
    )

    trial = _FakeTrial(
        params={
            "learning_rate": 1e-3,
            "weight_decay": 1e-2,
            "dropout": 0.1,
            "batch_size": 16,
            "hidden_size": 256,
            "num_attention_heads": 8,
            "num_layers": 3,
            "feedforward_multiplier": 3.0,
            "warmup_ratio": 0.1,
            "gradient_clip_norm": 1.0,
        }
    )
    objective_value = objective(cast(Any, trial))

    assert objective_value == 0.42
    assert "trial_dir" in trial.user_attrs
    assert "validation_nll" in trial.user_attrs
    assert trial.reports


def _fake_run_training_pipeline(**kwargs: object) -> TrainingArtifacts:
    maybe_train_config = kwargs["train_config_override"]
    if maybe_train_config is None:
        raise ValueError("train_config_override must be provided for this test.")
    train_config_override = cast(TrainConfig, maybe_train_config)
    checkpoint_dir = train_config_override.checkpoint_dir
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best = checkpoint_dir / "best.ckpt"
    latest = checkpoint_dir / "latest.ckpt"
    history = checkpoint_dir / "training_history.json"
    resolved = checkpoint_dir / "resolved_config.json"
    best.write_text("checkpoint", encoding="utf-8")
    latest.write_text("checkpoint", encoding="utf-8")
    history.write_text(
        json.dumps(
            [
                {"validation": {"macro_f1": 0.30}},
                {"validation": {"macro_f1": 0.42}},
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    resolved.write_text("{}", encoding="utf-8")
    return TrainingArtifacts(
        best_checkpoint_path=best,
        latest_checkpoint_path=latest,
        history_path=history,
        resolved_config_path=resolved,
    )


def _fake_run_evaluation_pipeline(**kwargs: object) -> EvaluationArtifacts:
    output_dir = cast(Path, kwargs["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    metrics_validation_path = output_dir / "metrics_validation.json"
    metrics_test_path = output_dir / "metrics_test.json"
    metrics_validation_calibrated_path = output_dir / "metrics_validation_calibrated.json"
    metrics_test_calibrated_path = output_dir / "metrics_test_calibrated.json"
    predictions_validation_path = output_dir / "predictions_validation.csv"
    predictions_test_path = output_dir / "predictions_test.csv"
    temperature_scaling_path = output_dir / "temperature_scaling.json"
    confusion_matrix_plot_path = output_dir / "confusion_matrix_test.png"
    reliability_diagram_plot_path = output_dir / "reliability_diagram_test.png"
    confidence_histogram_plot_path = output_dir / "confidence_histogram_test.png"
    entropy_histogram_plot_path = output_dir / "entropy_histogram_test.png"

    for path in [
        metrics_validation_path,
        metrics_test_path,
        metrics_validation_calibrated_path,
        metrics_test_calibrated_path,
        predictions_validation_path,
        predictions_test_path,
        temperature_scaling_path,
        confusion_matrix_plot_path,
        reliability_diagram_plot_path,
        confidence_histogram_plot_path,
        entropy_histogram_plot_path,
    ]:
        path.write_text("{}", encoding="utf-8")

    metrics_validation: dict[str, object] = {
        "macro_f1": 0.42,
        "nll": 1.12,
        "brier_score": 0.44,
        "ece": 0.08,
        "loss": 1.12,
        "accuracy": 0.5,
    }
    metrics_test: dict[str, object] = {
        "macro_f1": 0.40,
        "nll": 1.2,
        "brier_score": 0.46,
        "ece": 0.1,
        "loss": 1.2,
        "accuracy": 0.48,
    }
    metrics_validation_calibrated: dict[str, object] = {
        "macro_f1": 0.43,
        "nll": 1.1,
        "brier_score": 0.42,
        "ece": 0.07,
        "loss": 1.1,
        "accuracy": 0.51,
    }
    metrics_test_calibrated: dict[str, object] = {
        "macro_f1": 0.41,
        "nll": 1.18,
        "brier_score": 0.45,
        "ece": 0.09,
        "loss": 1.18,
        "accuracy": 0.49,
    }
    return EvaluationArtifacts(
        output_dir=output_dir,
        metrics_validation_path=metrics_validation_path,
        metrics_test_path=metrics_test_path,
        metrics_validation_calibrated_path=metrics_validation_calibrated_path,
        metrics_test_calibrated_path=metrics_test_calibrated_path,
        predictions_validation_path=predictions_validation_path,
        predictions_test_path=predictions_test_path,
        temperature_scaling_path=temperature_scaling_path,
        confusion_matrix_plot_path=confusion_matrix_plot_path,
        reliability_diagram_plot_path=reliability_diagram_plot_path,
        confidence_histogram_plot_path=confidence_histogram_plot_path,
        entropy_histogram_plot_path=entropy_histogram_plot_path,
        metrics_validation=metrics_validation,
        metrics_test=metrics_test,
        metrics_validation_calibrated=metrics_validation_calibrated,
        metrics_test_calibrated=metrics_test_calibrated,
    )
