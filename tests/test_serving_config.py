from __future__ import annotations

from pathlib import Path

import pytest

from bayes_gp_llmops.serving.config import (
    load_serving_config,
    resolve_serving_config_path,
)


def test_load_serving_config_from_yaml(tmp_path: Path) -> None:
    config_path = tmp_path / "serving.yaml"
    config_path.write_text(
        "\n".join(
            [
                "serving:",
                "  bundle_dir: artifacts/model/bundle",
                "  host: 0.0.0.0",
                "  port: 7860",
                "  device_preference: auto",
                "  max_batch_size: 16",
                "  max_input_length_chars: 4096",
                "  enable_calibration: true",
                "  expose_selected_metrics: true",
                "  log_level: INFO",
            ]
        )
        + "\n",
        encoding="utf-8",
    )

    config = load_serving_config(config_path, environ={})
    assert config.bundle_dir == Path("artifacts/model/bundle")
    assert config.port == 7860
    assert config.max_batch_size == 16
    assert config.enable_calibration is True


def test_environment_overrides_take_precedence(tmp_path: Path) -> None:
    config_path = tmp_path / "serving.yaml"
    config_path.write_text(
        "serving:\n  bundle_dir: artifacts/model/bundle\n  port: 7860\n",
        encoding="utf-8",
    )

    config = load_serving_config(
        config_path,
        environ={
            "SERVING_BUNDLE_DIR": str(tmp_path / "promoted-bundle"),
            "SERVING_PORT": "9000",
            "SERVING_ENABLE_CALIBRATION": "false",
            "SERVING_MAX_BATCH_SIZE": "24",
            "LOG_LEVEL": "WARNING",
        },
    )

    assert config.bundle_dir == tmp_path / "promoted-bundle"
    assert config.port == 9000
    assert config.enable_calibration is False
    assert config.max_batch_size == 24
    assert config.log_level == "WARNING"


def test_load_serving_config_missing_file_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_serving_config(tmp_path / "missing.yaml", environ={})


def test_resolve_serving_config_path_from_env() -> None:
    path = resolve_serving_config_path(environ={"SERVING_CONFIG_PATH": "configs/custom.yaml"})
    assert path == Path("configs/custom.yaml")
