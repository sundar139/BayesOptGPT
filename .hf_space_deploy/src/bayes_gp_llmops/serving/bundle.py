"""Inference bundle packaging and validation.

A bundle is a self-contained directory that holds everything needed to run
inference without depending on training or tuning internals. Bundles include
integrity metadata (SHA-256 checksums) so loading code can detect corruption
or partial copies.

Bundle layout::

    <bundle_dir>/
      checkpoint.ckpt
      tokenizer/
        tokenizer.json
        tokenizer_config.json
        special_tokens_map.json
        tokenizer_metadata.json       (if present in source tokenizer dir)
      model_config.json
      data_config.json
      champion_manifest.json
      label_map.json
      calibration.json                (optional, from temperature scaling)
      checksums.json                  (SHA-256 for all files except itself
                                       and bundle_metadata.json)
      bundle_metadata.json
"""

from __future__ import annotations

import hashlib
import json
import logging
import shutil
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel, Field

from bayes_gp_llmops.tracking.mlflow_utils import write_json

from .champion import ChampionManifest

LOGGER = logging.getLogger("bayes_gp_llmops.serving.bundle")

BUNDLE_SCHEMA_VERSION = "1.0"
TOKENIZER_SUBDIR = "tokenizer"
REQUIRED_BUNDLE_FILES = [
    "checkpoint.ckpt",
    "model_config.json",
    "data_config.json",
    "champion_manifest.json",
    "label_map.json",
    "checksums.json",
    "bundle_metadata.json",
]
_CHECKSUM_EXCLUSIONS = frozenset({"checksums.json", "bundle_metadata.json"})


class BundleMetadata(BaseModel):
    """Top-level metadata summarizing bundle contents and provenance."""

    schema_version: str = Field(default=BUNDLE_SCHEMA_VERSION)
    bundle_dir: str
    created_at: str
    champion_trial_number: int
    champion_study_name: str
    selected_metrics: dict[str, float | None]
    included_files: list[str]
    checksum_algorithm: str = Field(default="sha256")
    has_calibration: bool = Field(default=False)


def package_inference_bundle(
    *,
    champion_manifest: ChampionManifest,
    tokenizer_dir: Path,
    model_config_dict: dict[str, object],
    data_config_dict: dict[str, object],
    output_dir: Path,
    label_map: dict[str, str],
    calibration_path: Path | None = None,
) -> Path:
    """Create a self-contained inference bundle in output_dir.

    Copies the champion checkpoint, tokenizer artifacts, and all config
    snapshots into output_dir. Computes SHA-256 checksums for every
    included file and writes checksums.json. Writes bundle_metadata.json last.

    Args:
        champion_manifest: Promotion manifest for the selected model.
        tokenizer_dir: Source directory containing tokenizer artifacts.
        model_config_dict: JSON-serializable model configuration.
        data_config_dict: JSON-serializable data pipeline configuration.
        output_dir: Destination directory for the bundle (created if absent).
        label_map: Mapping from class index string to class name.
        calibration_path: Optional path to a temperature_scaling.json file.

    Returns:
        Path to the created bundle directory (same as output_dir).

    Raises:
        FileNotFoundError: If the checkpoint or tokenizer directory is absent.
    """
    output_dir.mkdir(parents=True, exist_ok=True)

    checkpoint_src = Path(champion_manifest.checkpoint_path)
    if not checkpoint_src.exists():
        raise FileNotFoundError(f"Champion checkpoint not found: {checkpoint_src}")
    shutil.copy2(checkpoint_src, output_dir / "checkpoint.ckpt")
    LOGGER.info("Checkpoint copied: checkpoint.ckpt")

    if not tokenizer_dir.is_dir():
        raise FileNotFoundError(f"Tokenizer directory not found: {tokenizer_dir}")
    tokenizer_dst = output_dir / TOKENIZER_SUBDIR
    tokenizer_dst.mkdir(parents=True, exist_ok=True)
    for artifact in sorted(tokenizer_dir.iterdir()):
        if artifact.is_file():
            shutil.copy2(artifact, tokenizer_dst / artifact.name)
    LOGGER.info("Tokenizer artifacts copied to %s/", TOKENIZER_SUBDIR)

    write_json(output_dir / "model_config.json", model_config_dict)
    write_json(output_dir / "data_config.json", data_config_dict)
    write_json(output_dir / "champion_manifest.json", champion_manifest.model_dump(mode="json"))
    write_json(output_dir / "label_map.json", dict(label_map.items()))

    has_calibration = False
    if calibration_path is not None and calibration_path.exists():
        shutil.copy2(calibration_path, output_dir / "calibration.json")
        has_calibration = True
        LOGGER.info("Calibration artifact copied: calibration.json")

    checksums = _compute_checksums(output_dir, exclude=_CHECKSUM_EXCLUSIONS)
    write_json(output_dir / "checksums.json", checksums)

    all_files = sorted(
        str(p.relative_to(output_dir)).replace("\\", "/")
        for p in output_dir.rglob("*")
        if p.is_file()
    )
    metadata = BundleMetadata(
        bundle_dir=str(output_dir.resolve()),
        created_at=datetime.now(tz=UTC).isoformat(),
        champion_trial_number=champion_manifest.trial_number,
        champion_study_name=champion_manifest.study_name,
        selected_metrics=champion_manifest.selected_metrics,
        included_files=all_files,
        has_calibration=has_calibration,
    )
    write_json(output_dir / "bundle_metadata.json", metadata.model_dump(mode="json"))
    LOGGER.info("Bundle created at %s (%d files)", output_dir, len(all_files))
    return output_dir


def validate_bundle(bundle_dir: Path) -> BundleMetadata:
    """Validate bundle integrity: required files are present and checksums match.

    Args:
        bundle_dir: Path to the bundle directory.

    Returns:
        Parsed BundleMetadata on success.

    Raises:
        FileNotFoundError: If the directory or any required file is missing.
        ValueError: If any checksum fails or the metadata cannot be parsed.
    """
    if not bundle_dir.is_dir():
        raise FileNotFoundError(f"Bundle directory does not exist: {bundle_dir}")

    for required in REQUIRED_BUNDLE_FILES:
        path = bundle_dir / required
        if not path.exists():
            raise FileNotFoundError(
                f"Required bundle file missing: {required}\n"
                f"Bundle directory: {bundle_dir}"
            )

    tokenizer_dir = bundle_dir / TOKENIZER_SUBDIR
    if not tokenizer_dir.is_dir():
        raise FileNotFoundError(
            f"Required bundle subdirectory missing: {TOKENIZER_SUBDIR}/\n"
            f"Bundle directory: {bundle_dir}"
        )

    checksums_path = bundle_dir / "checksums.json"
    with checksums_path.open("r", encoding="utf-8") as handle:
        raw_checksums = json.load(handle)
    if not isinstance(raw_checksums, dict):
        raise ValueError(f"checksums.json must contain a JSON object: {checksums_path}")
    checksums: dict[str, str] = {str(k): str(v) for k, v in raw_checksums.items()}

    failures: list[str] = []
    for relative_path, expected_hash in checksums.items():
        file_path = bundle_dir / relative_path
        if not file_path.exists():
            failures.append(f"MISSING: {relative_path}")
            continue
        actual_hash = _sha256(file_path)
        if actual_hash != expected_hash:
            failures.append(f"CHECKSUM_MISMATCH: {relative_path}")

    if failures:
        raise ValueError(
            f"Bundle validation failed ({len(failures)} error(s)) in {bundle_dir}:\n"
            + "\n".join(f"  {f}" for f in failures)
        )

    metadata = load_bundle_metadata(bundle_dir)
    LOGGER.info("Bundle validation passed: %s", bundle_dir)
    return metadata


def load_bundle_metadata(bundle_dir: Path) -> BundleMetadata:
    """Load bundle metadata without validating checksums.

    Args:
        bundle_dir: Path to the bundle directory.

    Returns:
        Parsed BundleMetadata.

    Raises:
        FileNotFoundError: If bundle_metadata.json is absent.
        ValueError: If the metadata cannot be parsed.
    """
    metadata_path = bundle_dir / "bundle_metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"bundle_metadata.json not found in: {bundle_dir}")
    with metadata_path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    return BundleMetadata.model_validate(payload)


def _compute_checksums(
    bundle_dir: Path,
    *,
    exclude: frozenset[str],
) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(bundle_dir.rglob("*")):
        if not path.is_file():
            continue
        if path.name in exclude:
            continue
        relative = str(path.relative_to(bundle_dir)).replace("\\", "/")
        result[relative] = _sha256(path)
    return result


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(65536), b""):
            h.update(chunk)
    return h.hexdigest()
