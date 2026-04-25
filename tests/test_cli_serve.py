from __future__ import annotations

from pathlib import Path

import pytest
from fastapi import FastAPI

from bayes_gp_llmops.cli import run_serve
from bayes_gp_llmops.serving.config import ServingConfig


class _RuntimeStub:
    calibration_active = True


def test_run_serve_smoke(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    config = ServingConfig(
        bundle_dir=tmp_path / "bundle",
        host="0.0.0.0",
        port=7860,
        log_level="INFO",
    )
    runtime_stub = _RuntimeStub()

    monkeypatch.setattr(
        "bayes_gp_llmops.cli.resolve_serving_config_path",
        lambda: Path("configs/serving.yaml"),
    )
    monkeypatch.setattr(
        "bayes_gp_llmops.cli.load_serving_config",
        lambda _path: config,
    )
    monkeypatch.setattr(
        "bayes_gp_llmops.cli.ServingRuntime.load_from_bundle",
        lambda _config: runtime_stub,
    )

    application = FastAPI()
    monkeypatch.setattr(
        "bayes_gp_llmops.cli.create_app",
        lambda *, serving_config, runtime: application,
    )

    uvicorn_call: dict[str, object] = {}

    def fake_uvicorn_run(
        app: FastAPI,
        *,
        host: str,
        port: int,
        reload: bool,
        log_level: str,
    ) -> None:
        uvicorn_call["app"] = app
        uvicorn_call["host"] = host
        uvicorn_call["port"] = port
        uvicorn_call["reload"] = reload
        uvicorn_call["log_level"] = log_level

    monkeypatch.setattr("bayes_gp_llmops.cli.uvicorn.run", fake_uvicorn_run)

    run_serve()

    output = capsys.readouterr().out
    assert "bundle_dir=" in output
    assert "bundle_validation=passed" in output
    assert "host=0.0.0.0" in output
    assert "port=7860" in output
    assert "calibration_enabled=true" in output

    assert uvicorn_call["app"] is application
    assert uvicorn_call["host"] == "0.0.0.0"
    assert uvicorn_call["port"] == 7860
    assert uvicorn_call["reload"] is False
    assert uvicorn_call["log_level"] == "info"
