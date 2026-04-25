"""Tests for inference bundle packaging and validation."""

from __future__ import annotations

import json
import re
import shutil
from pathlib import Path

import pytest

from bayes_gp_llmops.serving.bundle import (
    BundleMetadata,
    load_bundle_metadata,
    package_inference_bundle,
    validate_bundle,
)
from bayes_gp_llmops.serving.champion import (
    CandidateMetrics,
    ChampionManifest,
    build_champion_manifest,
)


def _make_champion_manifest(tmp_path: Path, trial_number: int = 2) -> ChampionManifest:
    checkpoint = tmp_path / "checkpoints" / "best.ckpt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.write_bytes(b"fake-model-state")

    candidate = CandidateMetrics(
        study_name="test-study",
        trial_number=trial_number,
        checkpoint_path=checkpoint,
        validation_macro_f1=0.89,
        validation_nll=0.42,
        validation_brier=0.09,
        validation_ece=0.04,
    )
    return build_champion_manifest(candidate)


def _build_tokenizer_dir(tokenizer_dir: Path) -> None:
    tokenizer_dir.mkdir(parents=True, exist_ok=True)
    for name in [
        "tokenizer.json",
        "tokenizer_config.json",
        "special_tokens_map.json",
        "tokenizer_metadata.json",
    ]:
        (tokenizer_dir / name).write_text(json.dumps({"version": "1.0"}), encoding="utf-8")


def _build_bundle(
    tmp_path: Path,
    with_calibration: bool = False,
) -> tuple[ChampionManifest, Path]:
    manifest = _make_champion_manifest(tmp_path)

    tokenizer_dir = tmp_path / "tokenizer"
    _build_tokenizer_dir(tokenizer_dir)

    if with_calibration:
        cal_path = tmp_path / "temperature_scaling.json"
        cal_path.write_text(json.dumps({"temperature": 1.2}), encoding="utf-8")
    else:
        cal_path = None

    bundle_dir = package_inference_bundle(
        champion_manifest=manifest,
        tokenizer_dir=tokenizer_dir,
        model_config_dict={"vocab_size": 512, "hidden_dim": 128},
        data_config_dict={"tokenizer": {"max_sequence_length": 256}},
        output_dir=tmp_path / "bundle",
        label_map={"0": "World", "1": "Sports", "2": "Business", "3": "Sci/Tech"},
        calibration_path=cal_path,
    )
    return manifest, bundle_dir


class TestPackageInferenceBundle:
    def test_creates_expected_files(self, tmp_path: Path) -> None:
        _, bundle_dir = _build_bundle(tmp_path)

        assert (bundle_dir / "checkpoint.ckpt").exists()
        assert (bundle_dir / "tokenizer" / "tokenizer.json").exists()
        assert (bundle_dir / "model_config.json").exists()
        assert (bundle_dir / "data_config.json").exists()
        assert (bundle_dir / "champion_manifest.json").exists()
        assert (bundle_dir / "label_map.json").exists()
        assert (bundle_dir / "checksums.json").exists()
        assert (bundle_dir / "bundle_metadata.json").exists()

    def test_no_calibration_when_omitted(self, tmp_path: Path) -> None:
        _, bundle_dir = _build_bundle(tmp_path, with_calibration=False)
        assert not (bundle_dir / "calibration.json").exists()

    def test_calibration_file_included(self, tmp_path: Path) -> None:
        _, bundle_dir = _build_bundle(tmp_path, with_calibration=True)
        assert (bundle_dir / "calibration.json").exists()

    def test_label_map_content(self, tmp_path: Path) -> None:
        _, bundle_dir = _build_bundle(tmp_path)
        label_map = json.loads((bundle_dir / "label_map.json").read_text())
        assert label_map["0"] == "World"
        assert label_map["3"] == "Sci/Tech"

    def test_checksums_sha256_format(self, tmp_path: Path) -> None:
        _, bundle_dir = _build_bundle(tmp_path)
        checksums = json.loads((bundle_dir / "checksums.json").read_text())
        for _path, digest in checksums.items():
            assert len(digest) == 64, f"Expected 64-char hex SHA-256, got: {digest}"

    def test_bundle_metadata_fields(self, tmp_path: Path) -> None:
        manifest, bundle_dir = _build_bundle(tmp_path)
        metadata = load_bundle_metadata(bundle_dir)
        assert metadata.champion_trial_number == manifest.trial_number
        assert metadata.champion_study_name == manifest.study_name
        assert isinstance(metadata.included_files, list)
        assert len(metadata.included_files) > 0
        assert metadata.has_calibration is False
        assert not _looks_absolute_path(metadata.bundle_dir)

    def test_manifest_checkpoint_path_is_sanitized(self, tmp_path: Path) -> None:
        _, bundle_dir = _build_bundle(tmp_path)
        manifest_payload = json.loads((bundle_dir / "champion_manifest.json").read_text())
        checkpoint_path = str(manifest_payload["checkpoint_path"])
        assert checkpoint_path == "checkpoint.ckpt"
        assert not _looks_absolute_path(checkpoint_path)

    def test_bundle_metadata_has_calibration(self, tmp_path: Path) -> None:
        _, bundle_dir = _build_bundle(tmp_path, with_calibration=True)
        metadata = load_bundle_metadata(bundle_dir)
        assert metadata.has_calibration is True

    def test_idempotent_overwrite(self, tmp_path: Path) -> None:
        manifest = _make_champion_manifest(tmp_path)
        tokenizer_dir = tmp_path / "tokenizer"
        _build_tokenizer_dir(tokenizer_dir)

        common_kwargs: dict[str, object] = {
            "champion_manifest": manifest,
            "tokenizer_dir": tokenizer_dir,
            "model_config_dict": {"vocab_size": 512},
            "data_config_dict": {},
            "output_dir": tmp_path / "bundle",
            "label_map": {"0": "World"},
        }
        package_inference_bundle(**common_kwargs)  # type: ignore[arg-type]
        bundle_dir = package_inference_bundle(**common_kwargs)  # type: ignore[arg-type]
        validate_bundle(bundle_dir)  # still valid


class TestValidateBundle:
    def test_valid_bundle_passes(self, tmp_path: Path) -> None:
        _, bundle_dir = _build_bundle(tmp_path)
        metadata = validate_bundle(bundle_dir)
        assert isinstance(metadata, BundleMetadata)

    def test_missing_required_file_raises(self, tmp_path: Path) -> None:
        _, bundle_dir = _build_bundle(tmp_path)
        (bundle_dir / "model_config.json").unlink()

        with pytest.raises(FileNotFoundError, match="model_config.json"):
            validate_bundle(bundle_dir)

    def test_missing_tokenizer_dir_raises(self, tmp_path: Path) -> None:
        _, bundle_dir = _build_bundle(tmp_path)
        shutil.rmtree(bundle_dir / "tokenizer")

        with pytest.raises(FileNotFoundError, match="tokenizer"):
            validate_bundle(bundle_dir)

    def test_checksum_mismatch_raises(self, tmp_path: Path) -> None:
        _, bundle_dir = _build_bundle(tmp_path)
        # Tamper with a checksummed file
        (bundle_dir / "model_config.json").write_text(
            json.dumps({"vocab_size": 9999}), encoding="utf-8"
        )

        with pytest.raises(ValueError, match="checksum"):
            validate_bundle(bundle_dir)

    def test_missing_bundle_dir_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            validate_bundle(tmp_path / "nonexistent")

    def test_missing_checksums_json_raises(self, tmp_path: Path) -> None:
        _, bundle_dir = _build_bundle(tmp_path)
        (bundle_dir / "checksums.json").unlink()

        with pytest.raises(FileNotFoundError, match="checksums.json"):
            validate_bundle(bundle_dir)


class TestLoadBundleMetadata:
    def test_loads_successfully(self, tmp_path: Path) -> None:
        _, bundle_dir = _build_bundle(tmp_path)
        metadata = load_bundle_metadata(bundle_dir)
        assert metadata.champion_study_name == "test-study"
        assert metadata.champion_trial_number == 2

    def test_missing_metadata_file_raises(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            load_bundle_metadata(tmp_path)


def _looks_absolute_path(value: str) -> bool:
    return bool(
        value.startswith("/")
        or value.startswith("\\\\")
        or re.match(r"^[A-Za-z]:[\\/]", value)
    )
