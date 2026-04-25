from __future__ import annotations

import os
from collections.abc import Mapping
from pathlib import Path

import yaml
from pydantic import BaseModel, Field, field_validator


class ServingConfig(BaseModel):
    """Validated runtime settings for bundle-driven inference serving."""

    bundle_dir: Path = Field(default=Path("artifacts/model/bundle"))
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=7860, ge=1, le=65535)
    device_preference: str = Field(default="auto")
    max_batch_size: int = Field(default=16, ge=1)
    max_input_length_chars: int = Field(default=4096, ge=1)
    enable_calibration: bool = Field(default=True)
    expose_selected_metrics: bool = Field(default=True)
    log_level: str = Field(default="INFO")

    @field_validator("host")
    @classmethod
    def validate_host(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("host must not be empty.")
        return normalized

    @field_validator("device_preference")
    @classmethod
    def validate_device_preference(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("device_preference must not be empty.")
        return normalized

    @field_validator("log_level")
    @classmethod
    def validate_log_level(cls, value: str) -> str:
        normalized = value.strip().upper()
        allowed = {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}
        if normalized not in allowed:
            raise ValueError(f"log_level must be one of {sorted(allowed)}")
        return normalized


def resolve_serving_config_path(
    explicit_path: Path | None = None,
    *,
    environ: Mapping[str, str] | None = None,
) -> Path:
    """Resolve the serving config path from explicit input or environment."""

    if explicit_path is not None:
        return explicit_path
    active_environ = os.environ if environ is None else environ
    configured_path = active_environ.get("SERVING_CONFIG_PATH")
    if configured_path:
        return Path(configured_path)
    return Path("configs/serving.yaml")


def load_serving_config(
    path: Path,
    *,
    environ: Mapping[str, str] | None = None,
) -> ServingConfig:
    """Load serving config from YAML and apply environment overrides."""

    if not path.exists():
        raise FileNotFoundError(f"Serving configuration file not found: {path}")

    with path.open("r", encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)

    if payload is None:
        payload = {}
    if not isinstance(payload, dict):
        raise ValueError(f"Configuration file {path} must contain a mapping at the top level.")

    serving_section = payload.get("serving", payload)
    if not isinstance(serving_section, dict):
        raise ValueError(f"Configuration file {path} must include a 'serving' mapping.")

    config = ServingConfig.model_validate(serving_section)
    updates = _extract_env_overrides(os.environ if environ is None else environ)
    if updates:
        config = config.model_copy(update=updates)
    return config


def _extract_env_overrides(environ: Mapping[str, str]) -> dict[str, object]:
    updates: dict[str, object] = {}

    bundle_dir = environ.get("SERVING_BUNDLE_DIR")
    if bundle_dir:
        updates["bundle_dir"] = Path(bundle_dir)

    host = environ.get("SERVING_HOST") or environ.get("HOST")
    if host:
        updates["host"] = host

    port = environ.get("SERVING_PORT") or environ.get("PORT")
    if port:
        updates["port"] = _parse_int(port, env_name="SERVING_PORT")

    device_preference = environ.get("SERVING_DEVICE_PREFERENCE")
    if device_preference:
        updates["device_preference"] = device_preference

    max_batch_size = environ.get("SERVING_MAX_BATCH_SIZE")
    if max_batch_size:
        updates["max_batch_size"] = _parse_int(max_batch_size, env_name="SERVING_MAX_BATCH_SIZE")

    max_input_length_chars = environ.get("SERVING_MAX_INPUT_LENGTH_CHARS")
    if max_input_length_chars:
        updates["max_input_length_chars"] = _parse_int(
            max_input_length_chars,
            env_name="SERVING_MAX_INPUT_LENGTH_CHARS",
        )

    enable_calibration = environ.get("SERVING_ENABLE_CALIBRATION")
    if enable_calibration:
        updates["enable_calibration"] = _parse_bool(
            enable_calibration,
            env_name="SERVING_ENABLE_CALIBRATION",
        )

    expose_selected_metrics = environ.get("SERVING_EXPOSE_SELECTED_METRICS")
    if expose_selected_metrics:
        updates["expose_selected_metrics"] = _parse_bool(
            expose_selected_metrics,
            env_name="SERVING_EXPOSE_SELECTED_METRICS",
        )

    log_level = environ.get("SERVING_LOG_LEVEL") or environ.get("LOG_LEVEL")
    if log_level:
        updates["log_level"] = log_level

    return updates


def _parse_int(value: str, *, env_name: str) -> int:
    try:
        return int(value)
    except ValueError as exc:
        raise ValueError(f"{env_name} must be an integer, got: {value}") from exc


def _parse_bool(value: str, *, env_name: str) -> bool:
    normalized = value.strip().lower()
    if normalized in {"1", "true", "yes", "on"}:
        return True
    if normalized in {"0", "false", "no", "off"}:
        return False
    raise ValueError(
        f"{env_name} must be a boolean literal (true/false/1/0/yes/no/on/off), got: {value}"
    )
