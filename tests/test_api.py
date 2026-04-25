from __future__ import annotations

from pathlib import Path
from typing import cast

import pytest
import torch
from fastapi.testclient import TestClient

from bayes_gp_llmops.api import create_app
from bayes_gp_llmops.models.classifier import TinyLlamaForSequenceClassification
from bayes_gp_llmops.models.config import ModelConfig
from bayes_gp_llmops.serving.bundle import BundleMetadata
from bayes_gp_llmops.serving.bundle_loader import LoadedBundle
from bayes_gp_llmops.serving.champion import ChampionManifest
from bayes_gp_llmops.serving.config import ServingConfig
from bayes_gp_llmops.serving.runtime import ServingRuntime


class _Encoding:
    def __init__(self, token_count: int) -> None:
        self.ids = [1] * token_count
        self.attention_mask = [1] * token_count


class _Tokenizer:
    def encode_batch(self, texts: list[str]) -> list[_Encoding]:
        return [_Encoding(max(2, min(8, len(text.split()) + 2))) for text in texts]


class _Model(torch.nn.Module):
    def __init__(self, num_classes: int) -> None:
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(1))
        self.num_classes = num_classes

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        del attention_mask
        batch_size = input_ids.size(0)
        base = torch.arange(
            self.num_classes,
            device=input_ids.device,
            dtype=torch.float32,
        ).unsqueeze(0)
        signal = input_ids.to(torch.float32).sum(dim=1, keepdim=True) * 1e-3
        return base.repeat(batch_size, 1) + signal + self.anchor


def _build_runtime(
    *,
    max_batch_size: int = 3,
    max_input_length_chars: int = 120,
    calibration_temperature: float | None = 1.2,
) -> tuple[ServingConfig, ServingRuntime]:
    config = ServingConfig(
        bundle_dir=Path("artifacts/model/bundle"),
        host="127.0.0.1",
        port=7860,
        max_batch_size=max_batch_size,
        max_input_length_chars=max_input_length_chars,
        enable_calibration=calibration_temperature is not None,
        expose_selected_metrics=True,
        log_level="INFO",
    )

    model_config = ModelConfig(
        vocab_size=256,
        max_sequence_length=16,
        hidden_size=64,
        num_layers=1,
        num_attention_heads=8,
        feedforward_multiplier=2.0,
        dropout=0.0,
        num_classes=4,
        rope_base=10000.0,
        pooling="masked_mean",
    )

    bundle_metadata = BundleMetadata(
        bundle_dir="artifacts/model/bundle",
        created_at="2026-01-01T00:00:00+00:00",
        champion_trial_number=7,
        champion_study_name="ag-news-study",
        selected_metrics={"validation_macro_f1": 0.91},
        included_files=["checkpoint.ckpt"],
        has_calibration=calibration_temperature is not None,
    )
    champion_manifest = ChampionManifest(
        study_name="ag-news-study",
        trial_number=7,
        checkpoint_path="artifacts/model/bundle/checkpoint.ckpt",
        selected_metrics={"validation_macro_f1": 0.91},
        config_snapshot={"model": {"num_classes": 4}},
        timestamp_utc="2026-01-01T00:00:00+00:00",
    )

    calibration_payload: dict[str, object] | None = None
    if calibration_temperature is not None:
        calibration_payload = {
            "enabled": True,
            "temperature": calibration_temperature,
        }

    loaded_bundle = LoadedBundle(
        model=cast(TinyLlamaForSequenceClassification, _Model(num_classes=4)),
        tokenizer=_Tokenizer(),
        model_config=model_config,
        label_map={"0": "World", "1": "Sports", "2": "Business", "3": "Sci/Tech"},
        bundle_dir=Path("artifacts/model/bundle"),
        max_sequence_length=16,
        bundle_metadata=bundle_metadata,
        champion_manifest=champion_manifest,
        calibration=calibration_payload,
    )

    runtime = ServingRuntime(
        config=config,
        loaded_bundle=loaded_bundle,
        bundle_metadata=bundle_metadata,
        champion_manifest=champion_manifest,
        temperature=calibration_temperature,
    )
    return config, runtime


def test_health_and_metadata_endpoints() -> None:
    config, runtime = _build_runtime()
    app = create_app(serving_config=config, runtime=runtime)

    with TestClient(app) as client:
        health_response = client.get("/health")
        assert health_response.status_code == 200
        health_payload = health_response.json()
        assert health_payload["status"] == "ok"
        assert health_payload["bundle_validation_status"] == "passed"
        assert health_payload["model_loaded"] is True
        assert health_payload["calibration_enabled"] is True

        metadata_response = client.get("/metadata")
        assert metadata_response.status_code == 200
        metadata_payload = metadata_response.json()
        assert metadata_payload["bundle_id"] == "ag-news-study-trial-7"
        assert metadata_payload["label_names"] == ["World", "Sports", "Business", "Sci/Tech"]
        assert metadata_payload["selected_metrics"]["validation_macro_f1"] == pytest.approx(0.91)


def test_root_route_and_version() -> None:
    config, runtime = _build_runtime()
    app = create_app(serving_config=config, runtime=runtime)

    with TestClient(app) as client:
        root_response = client.get("/")
        assert root_response.status_code == 200
        assert "bayes-gp-llmops serving" in root_response.text
        assert "/docs" in root_response.text

        version_response = client.get("/version")
        assert version_response.status_code == 200
        version_payload = version_response.json()
        assert version_payload["service"] == "bayes-gp-llmops"
        assert "python_version" in version_payload


def test_predict_single_plain_input() -> None:
    config, runtime = _build_runtime(calibration_temperature=1.1)
    app = create_app(serving_config=config, runtime=runtime)

    with TestClient(app) as client:
        response = client.post("/predict", json={"input": "Global market update"})
    assert response.status_code == 200

    payload = response.json()["prediction"]
    assert payload["label"] in {"World", "Sports", "Business", "Sci/Tech"}
    assert payload["confidence"] >= 0.0
    assert payload["entropy"] >= 0.0
    assert payload["margin"] >= 0.0
    assert payload["calibrated"] is True
    assert "World" in payload["probabilities"]


def test_predict_single_structured_input_preserves_id() -> None:
    config, runtime = _build_runtime(calibration_temperature=None)
    app = create_app(serving_config=config, runtime=runtime)

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"input": {"id": "record-001", "text": "Sports headline"}},
        )
    assert response.status_code == 200
    payload = response.json()["prediction"]
    assert payload["input_id"] == "record-001"
    assert payload["calibrated"] is False


def test_predict_single_legacy_text_payload() -> None:
    config, runtime = _build_runtime(calibration_temperature=1.1)
    app = create_app(serving_config=config, runtime=runtime)

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"text": "Global market update"},
        )
    assert response.status_code == 200


def test_predict_single_legacy_text_payload_preserves_id() -> None:
    config, runtime = _build_runtime(calibration_temperature=None)
    app = create_app(serving_config=config, runtime=runtime)

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"id": "record-legacy", "text": "Sports headline"},
        )
    assert response.status_code == 200

    payload = response.json()["prediction"]
    assert payload["input_id"] == "record-legacy"


def test_predict_batch_returns_structured_predictions() -> None:
    config, runtime = _build_runtime(max_batch_size=4)
    app = create_app(serving_config=config, runtime=runtime)

    with TestClient(app) as client:
        response = client.post(
            "/predict/batch",
            json={
                "inputs": [
                    "World news brief",
                    {"id": "b-2", "text": "Tech company announcement"},
                ]
            },
        )
    assert response.status_code == 200

    payload = response.json()
    assert payload["count"] == 2
    assert len(payload["predictions"]) == 2
    assert payload["predictions"][1]["input_id"] == "b-2"


def test_predict_rejects_oversized_batch() -> None:
    config, runtime = _build_runtime(max_batch_size=1)
    app = create_app(serving_config=config, runtime=runtime)

    with TestClient(app) as client:
        response = client.post(
            "/predict/batch",
            json={"inputs": ["one", "two"]},
        )
    assert response.status_code == 422
    assert "exceeds configured maximum" in response.json()["detail"]


def test_predict_rejects_overlong_input() -> None:
    config, runtime = _build_runtime(max_input_length_chars=8)
    app = create_app(serving_config=config, runtime=runtime)

    with TestClient(app) as client:
        response = client.post(
            "/predict",
            json={"input": "This text is too long"},
        )
    assert response.status_code == 422
    assert "exceeds configured maximum" in response.json()["detail"]


def test_startup_with_loader_when_runtime_not_injected(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    config, runtime = _build_runtime()
    monkeypatch.setattr(
        "bayes_gp_llmops.api.ServingRuntime.load_from_bundle",
        lambda _config: runtime,
    )

    app = create_app(serving_config=config)
    with TestClient(app) as client:
        response = client.get("/health")
    assert response.status_code == 200


def test_startup_fails_for_invalid_bundle(tmp_path: Path) -> None:
    config = ServingConfig(bundle_dir=tmp_path / "missing-bundle")
    app = create_app(serving_config=config)

    with pytest.raises(RuntimeError, match="Bundle validation failed"), TestClient(
        app
    ) as _client:
        _ = _client.get("/health")
