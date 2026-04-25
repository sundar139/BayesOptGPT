from __future__ import annotations

import json
import logging
from collections.abc import Iterator, Mapping, Sequence
from contextlib import contextmanager
from pathlib import Path

import mlflow

from bayes_gp_llmops.config import get_settings

LOGGER = logging.getLogger("bayes_gp_llmops.tracking.mlflow")


@contextmanager
def start_mlflow_run(
    *,
    enabled: bool,
    experiment_name: str,
    run_name: str | None = None,
    tracking_uri: str | None = None,
    nested: bool = False,
    tags: Mapping[str, str] | None = None,
) -> Iterator[str | None]:
    """Start an MLflow run when tracking is enabled."""

    if not enabled:
        yield None
        return

    resolved_uri = tracking_uri or get_settings().mlflow_tracking_uri
    mlflow.set_tracking_uri(resolved_uri)
    mlflow.set_experiment(experiment_name)
    with mlflow.start_run(run_name=run_name, nested=nested, tags=dict(tags or {})) as run:
        run_id = run.info.run_id
        LOGGER.info("mlflow_run_id=%s experiment=%s", run_id, experiment_name)
        yield run_id


def log_parameters(parameters: Mapping[str, object], *, enabled: bool) -> None:
    """Log scalar and string-like parameters to the active MLflow run."""

    if not enabled:
        return
    sanitized: dict[str, str] = {}
    for key, value in parameters.items():
        sanitized[str(key)] = _stringify_value(value)
    if sanitized:
        mlflow.log_params(sanitized)


def log_metrics(
    metrics: Mapping[str, float | int],
    *,
    enabled: bool,
    step: int | None = None,
) -> None:
    """Log numeric metrics to the active MLflow run."""

    if not enabled:
        return
    payload = {str(key): float(value) for key, value in metrics.items()}
    if not payload:
        return
    if step is None:
        mlflow.log_metrics(payload)
    else:
        for key, value in payload.items():
            mlflow.log_metric(key, value, step=step)


def log_artifact_file(path: Path, *, enabled: bool, artifact_path: str | None = None) -> None:
    """Log a single file artifact if present."""

    if not enabled:
        return
    if not path.exists():
        raise FileNotFoundError(f"MLflow artifact path does not exist: {path}")
    mlflow.log_artifact(str(path), artifact_path=artifact_path)


def log_artifact_files(
    paths: Sequence[Path],
    *,
    enabled: bool,
    artifact_path: str | None = None,
) -> None:
    """Log multiple file artifacts."""

    if not enabled:
        return
    for path in paths:
        log_artifact_file(path, enabled=enabled, artifact_path=artifact_path)


def flatten_mapping(
    payload: Mapping[str, object],
    *,
    prefix: str = "",
    separator: str = ".",
) -> dict[str, object]:
    """Flatten a nested mapping to one level using dotted keys."""

    flattened: dict[str, object] = {}
    for key, value in payload.items():
        key_string = str(key)
        flattened_key = f"{prefix}{separator}{key_string}" if prefix else key_string
        if isinstance(value, Mapping):
            nested = flatten_mapping(value, prefix=flattened_key, separator=separator)
            flattened.update(nested)
        else:
            flattened[flattened_key] = value
    return flattened


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write JSON payload with stable formatting."""

    with path.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")


def _stringify_value(value: object) -> str:
    if isinstance(value, bool):
        return str(value).lower()
    if isinstance(value, Path):
        return str(value)
    return str(value)
