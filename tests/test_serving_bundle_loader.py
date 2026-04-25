"""Tests for the load-from-bundle inference path."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock

import pytest

from bayes_gp_llmops.models.config import ModelConfig
from bayes_gp_llmops.serving.bundle_loader import (
    _assert_required_files,
    _resolve_max_sequence_length,
    load_inference_bundle,
)


def _build_minimal_bundle(bundle_dir: Path, with_calibration: bool = False) -> None:
    bundle_dir.mkdir(parents=True, exist_ok=True)

    # Checkpoint
    (bundle_dir / "checkpoint.ckpt").write_bytes(b"fake-model")

    # Tokenizer dir
    tok_dir = bundle_dir / "tokenizer"
    tok_dir.mkdir(exist_ok=True)
    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer_metadata.json",
    ]:
        (tok_dir / name).write_text("{}", encoding="utf-8")

    # Configs
    model_cfg = {
        "vocab_size": 256,
        "max_sequence_length": 128,
        "num_classes": 4,
        "hidden_size": 64,
        "num_attention_heads": 2,
        "num_layers": 2,
        "dropout": 0.1,
        "feedforward_multiplier": 4.0,
    }
    (bundle_dir / "model_config.json").write_text(
        json.dumps(model_cfg), encoding="utf-8"
    )
    data_cfg = {"tokenizer": {"max_sequence_length": 128}}
    (bundle_dir / "data_config.json").write_text(
        json.dumps(data_cfg), encoding="utf-8"
    )

    # Label map
    label_map = {"0": "World", "1": "Sports", "2": "Business", "3": "Sci/Tech"}
    (bundle_dir / "label_map.json").write_text(json.dumps(label_map), encoding="utf-8")

    # Champion manifest
    manifest = {
        "schema_version": "1.0",
        "study_name": "test-study",
        "trial_number": 0,
        "checkpoint_path": str(bundle_dir / "checkpoint.ckpt"),
        "config_snapshot": {"model": model_cfg, "data": data_cfg},
        "selected_metrics": {"validation_macro_f1": 0.88},
        "selection_policy": "maximize validation_macro_f1",
        "timestamp_utc": "2024-01-01T00:00:00Z",
    }
    (bundle_dir / "champion_manifest.json").write_text(
        json.dumps(manifest), encoding="utf-8"
    )

    # Checksums (empty – we bypass validate_bundle in these tests)
    (bundle_dir / "checksums.json").write_text("{}", encoding="utf-8")
    (bundle_dir / "bundle_metadata.json").write_text(
        json.dumps(
            {
                "schema_version": "1.0",
                "bundle_dir": str(bundle_dir),
                "champion_trial_number": 0,
                "champion_study_name": "test-study",
                "created_at": "2024-01-01T00:00:00Z",
                "selected_metrics": {"validation_macro_f1": 0.88},
                "has_calibration": with_calibration,
                "included_files": [],
            }
        ),
        encoding="utf-8",
    )

    if with_calibration:
        cal = {"temperature": 1.15}
        (bundle_dir / "calibration.json").write_text(json.dumps(cal), encoding="utf-8")


class TestAssertRequiredFiles:
    def test_passes_when_all_present(self, tmp_path: Path) -> None:
        _build_minimal_bundle(tmp_path)
        _assert_required_files(tmp_path)  # should not raise

    def test_raises_on_missing_checkpoint(self, tmp_path: Path) -> None:
        _build_minimal_bundle(tmp_path)
        (tmp_path / "checkpoint.ckpt").unlink()
        with pytest.raises(FileNotFoundError, match="checkpoint.ckpt"):
            _assert_required_files(tmp_path)

    def test_raises_on_missing_model_config(self, tmp_path: Path) -> None:
        _build_minimal_bundle(tmp_path)
        (tmp_path / "model_config.json").unlink()
        with pytest.raises(FileNotFoundError, match="model_config.json"):
            _assert_required_files(tmp_path)

    def test_raises_on_missing_label_map(self, tmp_path: Path) -> None:
        _build_minimal_bundle(tmp_path)
        (tmp_path / "label_map.json").unlink()
        with pytest.raises(FileNotFoundError, match="label_map.json"):
            _assert_required_files(tmp_path)

    def test_raises_on_missing_tokenizer_dir(self, tmp_path: Path) -> None:
        _build_minimal_bundle(tmp_path)
        import shutil
        shutil.rmtree(tmp_path / "tokenizer")
        with pytest.raises(FileNotFoundError, match="tokenizer"):
            _assert_required_files(tmp_path)


def _minimal_model_config(**overrides: object) -> ModelConfig:
    base: dict[str, object] = {
        "vocab_size": 256,
        "max_sequence_length": 128,
        "num_classes": 4,
        "hidden_size": 64,
        "num_layers": 2,
        "num_attention_heads": 2,
        "feedforward_multiplier": 2.0,
        "dropout": 0.1,
    }
    base.update(overrides)
    return ModelConfig(**base)  # type: ignore[arg-type]


class TestResolveMaxSequenceLength:
    def test_reads_from_data_config(self) -> None:
        data_cfg: dict[str, object] = {"tokenizer": {"max_sequence_length": 512}}
        model_cfg = _minimal_model_config()
        result = _resolve_max_sequence_length(data_cfg, model_cfg)
        assert result == 512

    def test_falls_back_to_model_config(self) -> None:
        data_cfg: dict[str, object] = {}
        model_cfg = _minimal_model_config(max_sequence_length=256)
        result = _resolve_max_sequence_length(data_cfg, model_cfg)
        assert result == 256

    def test_falls_back_when_tokenizer_section_missing_key(self) -> None:
        data_cfg: dict[str, object] = {"tokenizer": {}}
        model_cfg = _minimal_model_config(max_sequence_length=64)
        result = _resolve_max_sequence_length(data_cfg, model_cfg)
        assert result == 64


class TestLoadInferenceBundle:
    def test_load_integration(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _build_minimal_bundle(tmp_path)

        fake_model = MagicMock()
        fake_tokenizer = MagicMock()

        def fake_model_cls(config: Any) -> MagicMock:
            return fake_model

        def fake_load_checkpoint(*, path: Path, model: Any, **kw: Any) -> None:
            pass

        def fake_load_tokenizer(output_dir: Path, max_sequence_length: int | None = None) -> Any:
            return fake_tokenizer

        monkeypatch.setattr(
            "bayes_gp_llmops.serving.bundle_loader.TinyLlamaForSequenceClassification",
            fake_model_cls,
        )
        monkeypatch.setattr(
            "bayes_gp_llmops.serving.bundle_loader.load_checkpoint",
            fake_load_checkpoint,
        )
        monkeypatch.setattr(
            "bayes_gp_llmops.serving.bundle_loader.load_tokenizer",
            fake_load_tokenizer,
        )

        bundle = load_inference_bundle(tmp_path, device="cpu")
        assert bundle.model is fake_model.to.return_value
        assert bundle.tokenizer is fake_tokenizer
        assert bundle.label_map["0"] == "World"
        assert bundle.bundle_dir == tmp_path
        assert bundle.bundle_metadata.champion_study_name == "test-study"
        assert bundle.champion_manifest.study_name == "test-study"
        assert bundle.calibration is None

    def test_load_with_calibration(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        _build_minimal_bundle(tmp_path, with_calibration=True)

        monkeypatch.setattr(
            "bayes_gp_llmops.serving.bundle_loader.TinyLlamaForSequenceClassification",
            lambda config: MagicMock(),
        )
        monkeypatch.setattr(
            "bayes_gp_llmops.serving.bundle_loader.load_checkpoint",
            lambda *, path, model, **kw: None,
        )
        monkeypatch.setattr(
            "bayes_gp_llmops.serving.bundle_loader.load_tokenizer",
            lambda output_dir, max_sequence_length=None: MagicMock(),
        )

        bundle = load_inference_bundle(tmp_path, device="cpu")
        assert bundle.bundle_metadata.has_calibration is True
        assert bundle.champion_manifest.trial_number == 0
        assert bundle.calibration is not None
        assert "temperature" in bundle.calibration

    def test_missing_bundle_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_inference_bundle(tmp_path / "nonexistent", device="cpu")
