"""Load a packaged inference bundle and return inference-ready objects.

This module is intentionally free of training-specific dependencies. It can
be used directly by a FastAPI serving layer without importing anything from
the training or tuning pipelines.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from bayes_gp_llmops.data.tokenizer import load_tokenizer
from bayes_gp_llmops.models.classifier import TinyLlamaForSequenceClassification
from bayes_gp_llmops.models.config import ModelConfig
from bayes_gp_llmops.training.trainer import load_checkpoint
from bayes_gp_llmops.utils.device import resolve_device

from .bundle import (
    REQUIRED_BUNDLE_FILES,
    TOKENIZER_SUBDIR,
    BundleMetadata,
    load_bundle_metadata,
    validate_bundle,
)
from .champion import ChampionManifest, load_champion_manifest

LOGGER = logging.getLogger("bayes_gp_llmops.serving.bundle_loader")


@dataclass(frozen=True)
class LoadedBundle:
    """Inference-ready objects loaded from a packaged inference bundle."""

    model: TinyLlamaForSequenceClassification
    tokenizer: Any  # tokenizers.Tokenizer at runtime
    model_config: ModelConfig
    label_map: dict[str, str]
    bundle_dir: Path
    max_sequence_length: int
    bundle_metadata: BundleMetadata
    champion_manifest: ChampionManifest
    calibration: dict[str, object] | None = None


def load_inference_bundle(
    bundle_dir: Path,
    *,
    device: torch.device | str | None = None,
    skip_validation: bool = False,
) -> LoadedBundle:
    """Load model, tokenizer, and metadata from a packaged inference bundle.

    Validates required files and SHA-256 checksums before loading unless
    skip_validation is True. Set skip_validation=True only in hot paths
    where the bundle is known to be intact (e.g., already validated on startup).

    Args:
        bundle_dir: Directory produced by package_inference_bundle.
        device: Target device. Resolved automatically when None.
        skip_validation: Skip checksum verification (still checks required files).

    Returns:
        LoadedBundle with an initialized, eval-mode model and loaded tokenizer.

    Raises:
        FileNotFoundError: If any required bundle file or directory is missing.
        ValueError: If checksum validation fails (when skip_validation is False).
    """
    if not skip_validation:
        bundle_metadata = validate_bundle(bundle_dir)
    else:
        _assert_required_files(bundle_dir)
        bundle_metadata = load_bundle_metadata(bundle_dir)

    model_config_path = bundle_dir / "model_config.json"
    with model_config_path.open("r", encoding="utf-8") as handle:
        model_config = ModelConfig.model_validate(json.load(handle))

    data_config_path = bundle_dir / "data_config.json"
    with data_config_path.open("r", encoding="utf-8") as handle:
        data_config_dict: dict[str, object] = json.load(handle)
    max_sequence_length = _resolve_max_sequence_length(data_config_dict, model_config)

    label_map_path = bundle_dir / "label_map.json"
    with label_map_path.open("r", encoding="utf-8") as handle:
        label_map: dict[str, str] = json.load(handle)

    champion_manifest = load_champion_manifest(bundle_dir / "champion_manifest.json")

    tokenizer = load_tokenizer(
        bundle_dir / TOKENIZER_SUBDIR,
        max_sequence_length=max_sequence_length,
    )

    resolved_device = _resolve_device(device)
    model = TinyLlamaForSequenceClassification(model_config).to(resolved_device)
    load_checkpoint(
        path=bundle_dir / "checkpoint.ckpt",
        model=model,
        map_location=resolved_device,
    )
    model.eval()
    LOGGER.info("Bundle loaded from %s (device=%s)", bundle_dir, resolved_device)

    calibration: dict[str, object] | None = None
    calibration_path = bundle_dir / "calibration.json"
    if calibration_path.exists():
        with calibration_path.open("r", encoding="utf-8") as handle:
            calibration = json.load(handle)

    return LoadedBundle(
        model=model,
        tokenizer=tokenizer,
        model_config=model_config,
        label_map=label_map,
        bundle_dir=bundle_dir,
        max_sequence_length=max_sequence_length,
        bundle_metadata=bundle_metadata,
        champion_manifest=champion_manifest,
        calibration=calibration,
    )


def _assert_required_files(bundle_dir: Path) -> None:
    """Check required files exist without verifying checksums.

    Raises:
        FileNotFoundError: On the first missing required file or directory.
    """
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


def _resolve_max_sequence_length(
    data_config_dict: dict[str, object],
    model_config: ModelConfig,
) -> int:
    """Extract max_sequence_length from the data config, falling back to model config."""
    tokenizer_section = data_config_dict.get("tokenizer")
    if isinstance(tokenizer_section, dict):
        value = tokenizer_section.get("max_sequence_length")
        if isinstance(value, int):
            return value
    return model_config.max_sequence_length


def _resolve_device(device: torch.device | str | None) -> torch.device:
    if isinstance(device, torch.device):
        return device
    if isinstance(device, str):
        return resolve_device(device)
    return resolve_device("auto")
