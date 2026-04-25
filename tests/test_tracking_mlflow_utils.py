from __future__ import annotations

from pathlib import Path

from bayes_gp_llmops.tracking.mlflow_utils import (
    flatten_mapping,
    log_artifact_file,
    log_metrics,
    log_parameters,
    start_mlflow_run,
)


def test_mlflow_utils_smoke_sqlite_tracking(tmp_path: Path) -> None:
    tracking_db = tmp_path / "mlflow.db"
    artifact_file = tmp_path / "artifact.txt"
    artifact_file.write_text("artifact", encoding="utf-8")
    tracking_uri = f"sqlite:///{tracking_db.as_posix()}"

    with start_mlflow_run(
        enabled=True,
        experiment_name="test-experiment",
        run_name="smoke",
        tracking_uri=tracking_uri,
    ):
        log_parameters({"alpha": 1, "name": "demo"}, enabled=True)
        log_metrics({"metric": 0.5}, enabled=True)
        log_artifact_file(artifact_file, enabled=True, artifact_path="unit")

    assert tracking_db.exists()


def test_flatten_mapping_nested() -> None:
    payload = {"a": {"b": 1, "c": {"d": 2}}, "x": 3}
    flattened = flatten_mapping(payload)
    assert flattened["a.b"] == 1
    assert flattened["a.c.d"] == 2
    assert flattened["x"] == 3
