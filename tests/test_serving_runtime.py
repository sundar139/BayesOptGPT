from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import pytest
import torch

from bayes_gp_llmops.models.classifier import TinyLlamaForSequenceClassification
from bayes_gp_llmops.models.config import ModelConfig
from bayes_gp_llmops.serving.bundle import load_bundle_metadata, package_inference_bundle
from bayes_gp_llmops.serving.bundle_loader import LoadedBundle
from bayes_gp_llmops.serving.champion import (
    CandidateMetrics,
    build_champion_manifest,
    load_champion_manifest,
)
from bayes_gp_llmops.serving.config import ServingConfig
from bayes_gp_llmops.serving.runtime import ServingRuntime, ServingStartupError


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


def _write_tokenizer_artifacts(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)
    for artifact_name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer_metadata.json",
    ]:
        (path / artifact_name).write_text("{}", encoding="utf-8")


def _build_valid_bundle(tmp_path: Path, *, with_calibration: bool) -> Path:
    checkpoint_path = tmp_path / "checkpoints" / "best.ckpt"
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_path.write_bytes(b"checkpoint")

    candidate = CandidateMetrics(
        study_name="runtime-study",
        trial_number=3,
        checkpoint_path=checkpoint_path,
        validation_macro_f1=0.88,
        validation_nll=0.45,
    )
    manifest = build_champion_manifest(candidate)

    tokenizer_dir = tmp_path / "tokenizer"
    _write_tokenizer_artifacts(tokenizer_dir)

    calibration_path: Path | None = None
    if with_calibration:
        calibration_path = tmp_path / "temperature_scaling.json"
        calibration_path.write_text(
            json.dumps({"enabled": True, "temperature": 1.15}),
            encoding="utf-8",
        )

    model_config: dict[str, object] = {
        "vocab_size": 256,
        "max_sequence_length": 32,
        "num_classes": 4,
        "hidden_size": 64,
        "num_attention_heads": 8,
        "num_layers": 1,
        "dropout": 0.0,
        "feedforward_multiplier": 2.0,
        "rope_base": 10000.0,
        "pooling": "masked_mean",
    }
    data_config: dict[str, object] = {"tokenizer": {"max_sequence_length": 32}}

    return package_inference_bundle(
        champion_manifest=manifest,
        tokenizer_dir=tokenizer_dir,
        model_config_dict=model_config,
        data_config_dict=data_config,
        output_dir=tmp_path / "bundle",
        label_map={"0": "World", "1": "Sports", "2": "Business", "3": "Sci/Tech"},
        calibration_path=calibration_path,
    )


def _build_loaded_bundle_stub(bundle_dir: Path, *, with_calibration: bool) -> LoadedBundle:
    model_config_payload = json.loads(
        (bundle_dir / "model_config.json").read_text(encoding="utf-8")
    )
    model_config = ModelConfig.model_validate(model_config_payload)

    calibration_payload: dict[str, object] | None = None
    if with_calibration:
        calibration_payload = json.loads(
            (bundle_dir / "calibration.json").read_text(encoding="utf-8")
        )

    bundle_metadata = load_bundle_metadata(bundle_dir)
    champion_manifest = load_champion_manifest(bundle_dir / "champion_manifest.json")

    return LoadedBundle(
        model=cast(
            TinyLlamaForSequenceClassification,
            _Model(num_classes=model_config.num_classes),
        ),
        tokenizer=_Tokenizer(),
        model_config=model_config,
        label_map={"0": "World", "1": "Sports", "2": "Business", "3": "Sci/Tech"},
        bundle_dir=bundle_dir,
        max_sequence_length=model_config.max_sequence_length,
        bundle_metadata=bundle_metadata,
        champion_manifest=champion_manifest,
        calibration=calibration_payload,
    )


def test_runtime_load_from_bundle_validates_bundle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_dir = _build_valid_bundle(tmp_path, with_calibration=True)
    loaded_bundle_stub = _build_loaded_bundle_stub(bundle_dir, with_calibration=True)
    monkeypatch.setattr(
        "bayes_gp_llmops.serving.runtime.load_inference_bundle",
        lambda *_args, **_kwargs: loaded_bundle_stub,
    )

    config = ServingConfig(bundle_dir=bundle_dir, enable_calibration=True)
    runtime = ServingRuntime.load_from_bundle(config)

    assert runtime.bundle_validation_status == "passed"
    assert runtime.model_loaded is True
    assert runtime.calibration_active is True

    metadata = runtime.metadata_payload(expose_selected_metrics=True)
    assert metadata["model_name"] == "_Model"
    assert metadata["bundle_id"] == "runtime-study-trial-3"
    assert metadata["bundle_schema_version"] == "1.0"
    artifacts = metadata["artifacts"]
    assert isinstance(artifacts, dict)
    assert artifacts["checkpoint_available"] is True
    assert artifacts["tokenizer_available"] is True
    assert metadata["selected_metrics"] is not None


def test_runtime_load_from_bundle_fails_when_bundle_missing(tmp_path: Path) -> None:
    config = ServingConfig(bundle_dir=tmp_path / "missing-bundle")
    with pytest.raises(ServingStartupError, match="Bundle validation failed"):
        ServingRuntime.load_from_bundle(config)


def test_runtime_honors_calibration_toggle(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_dir = _build_valid_bundle(tmp_path, with_calibration=True)
    loaded_bundle_stub = _build_loaded_bundle_stub(bundle_dir, with_calibration=True)
    monkeypatch.setattr(
        "bayes_gp_llmops.serving.runtime.load_inference_bundle",
        lambda *_args, **_kwargs: loaded_bundle_stub,
    )

    config = ServingConfig(bundle_dir=bundle_dir, enable_calibration=False)
    runtime = ServingRuntime.load_from_bundle(config)
    assert runtime.calibration_active is False


def test_runtime_predicts_records(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bundle_dir = _build_valid_bundle(tmp_path, with_calibration=False)
    loaded_bundle_stub = _build_loaded_bundle_stub(bundle_dir, with_calibration=False)
    monkeypatch.setattr(
        "bayes_gp_llmops.serving.runtime.load_inference_bundle",
        lambda *_args, **_kwargs: loaded_bundle_stub,
    )

    config = ServingConfig(bundle_dir=bundle_dir, enable_calibration=False)
    runtime = ServingRuntime.load_from_bundle(config)

    predictions = runtime.predict_texts(["world update", "sports final"])
    assert len(predictions) == 2
    assert predictions[0].label in {"World", "Sports", "Business", "Sci/Tech"}
    assert predictions[0].confidence >= 0.0
    assert predictions[0].calibrated is False
