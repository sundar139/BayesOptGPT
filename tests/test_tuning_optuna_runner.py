from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import pytest

from bayes_gp_llmops.tuning.optuna_runner import load_tune_config, run_optuna_study


class _FakeObjective:
    def __init__(self, **_: object) -> None:
        return

    def __call__(self, trial: Any) -> float:
        value = trial.suggest_float("objective_value", 0.1, 0.9)
        return float(value)


def test_load_tune_config(tmp_path: Path) -> None:
    config_path = tmp_path / "tune.yaml"
    config_path.write_text(
        "\n".join(
            [
                "tuning:",
                "  study_name: unit-study",
                "  storage_path: artifacts/tuning/unit.db",
                "  output_dir: artifacts/tuning",
                "  sampler: tpe",
                "  pruner: median",
                "  n_trials: 2",
                "  timeout_seconds: 120",
                "  direction: maximize",
                "  seed: 11",
                "  debug_mode: true",
                "  enable_mlflow: false",
                "  mlflow_experiment_name: unit-exp",
                "  enable_temperature_scaling: true",
                "  log_trial_artifacts: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    loaded = load_tune_config(config_path)
    assert loaded.study_name == "unit-study"
    assert loaded.n_trials == 2
    assert loaded.enable_mlflow is False


def test_study_result_persistence(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    tune_config_path = tmp_path / "tune.yaml"
    tune_config_path.write_text(
        "\n".join(
            [
                "tuning:",
                "  study_name: persistence-study",
                f"  storage_path: {tmp_path / 'study.db'}",
                f"  output_dir: {tmp_path / 'outputs'}",
                "  sampler: random",
                "  pruner: none",
                "  n_trials: 2",
                "  timeout_seconds: 60",
                "  direction: maximize",
                "  seed: 7",
                "  debug_mode: true",
                "  enable_mlflow: false",
                "  mlflow_experiment_name: persistence-exp",
                "  enable_temperature_scaling: false",
                "  log_trial_artifacts: false",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    monkeypatch.setattr("bayes_gp_llmops.tuning.optuna_runner.TuningObjective", _FakeObjective)
    artifacts = run_optuna_study(
        data_config_path=Path("configs/data.yaml"),
        model_config_path=Path("configs/model.yaml"),
        train_config_path=Path("configs/train.yaml"),
        tune_config_path=tune_config_path,
        device_override="cpu",
        n_trials_override=None,
        timeout_override=None,
        debug_override=None,
    )

    assert artifacts.storage_path.exists()
    assert artifacts.best_params_path.exists()
    assert artifacts.trial_results_path.exists()
    assert artifacts.study_summary_path.exists()
    with artifacts.best_params_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    assert "best_params" in payload
    with artifacts.trial_results_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows
