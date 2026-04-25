from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import cast

import torch

from bayes_gp_llmops.evaluation.calibration import apply_temperature
from bayes_gp_llmops.evaluation.uncertainty import (
    confidence_margin,
    max_softmax_confidence,
    predictive_entropy,
    probabilities_from_logits,
)

from .bundle import BundleMetadata, validate_bundle
from .bundle_loader import LoadedBundle, load_inference_bundle
from .champion import ChampionManifest
from .config import ServingConfig

LOGGER = logging.getLogger("bayes_gp_llmops.serving.runtime")


class ServingStartupError(RuntimeError):
    """Raised when serving runtime initialization fails."""


@dataclass(frozen=True)
class PredictionRecord:
    """Stable prediction payload used by the API layer."""

    label: str
    label_index: int
    confidence: float
    probabilities: dict[str, float]
    entropy: float
    margin: float
    calibrated: bool


class ServingRuntime:
    """Bundle-driven model runtime used by FastAPI serving endpoints."""

    def __init__(
        self,
        *,
        config: ServingConfig,
        loaded_bundle: LoadedBundle,
        bundle_metadata: BundleMetadata,
        champion_manifest: ChampionManifest,
        temperature: float | None,
    ) -> None:
        self._config = config
        self._loaded_bundle = loaded_bundle
        self._bundle_metadata = bundle_metadata
        self._champion_manifest = champion_manifest
        self._temperature = temperature
        self._model_device = _resolve_model_device(loaded_bundle.model)

    @classmethod
    def load_from_bundle(cls, config: ServingConfig) -> ServingRuntime:
        """Validate and load a serving runtime from the promoted bundle."""

        try:
            bundle_metadata = validate_bundle(config.bundle_dir)
        except (FileNotFoundError, ValueError) as exc:
            raise ServingStartupError(
                f"Bundle validation failed for {config.bundle_dir}: {exc}"
            ) from exc

        try:
            loaded_bundle = load_inference_bundle(
                config.bundle_dir,
                device=config.device_preference,
                skip_validation=True,
            )
            champion_manifest = loaded_bundle.champion_manifest
            temperature = _resolve_temperature(
                loaded_bundle.calibration,
                enable_calibration=config.enable_calibration,
            )
        except (FileNotFoundError, ValueError) as exc:
            raise ServingStartupError(f"Bundle loading failed: {exc}") from exc
        except Exception as exc:
            raise ServingStartupError("Bundle loading failed due to an unexpected error.") from exc

        LOGGER.info(
            "Serving runtime loaded: bundle=%s calibration_active=%s",
            config.bundle_dir,
            temperature is not None,
        )
        return cls(
            config=config,
            loaded_bundle=loaded_bundle,
            bundle_metadata=bundle_metadata,
            champion_manifest=champion_manifest,
            temperature=temperature,
        )

    @property
    def bundle_validation_status(self) -> str:
        return "passed"

    @property
    def model_loaded(self) -> bool:
        return True

    @property
    def calibration_active(self) -> bool:
        return self._temperature is not None

    @property
    def bundle_identifier(self) -> str:
        return (
            f"{self._champion_manifest.study_name}-"
            f"trial-{self._champion_manifest.trial_number}"
        )

    @property
    def label_names(self) -> list[str]:
        return [
            self._loaded_bundle.label_map.get(str(index), str(index))
            for index in range(self._loaded_bundle.model_config.num_classes)
        ]

    def metadata_payload(self, *, expose_selected_metrics: bool) -> dict[str, object]:
        """Return API-safe metadata for service inspection endpoints."""

        payload: dict[str, object] = {
            "bundle_id": self.bundle_identifier,
            "study_name": self._champion_manifest.study_name,
            "trial_number": self._champion_manifest.trial_number,
            "created_at": self._bundle_metadata.created_at,
            "label_names": self.label_names,
            "calibration_enabled": self.calibration_active,
            "model": {
                "num_classes": self._loaded_bundle.model_config.num_classes,
                "max_sequence_length": self._loaded_bundle.max_sequence_length,
                "hidden_size": self._loaded_bundle.model_config.hidden_size,
                "num_layers": self._loaded_bundle.model_config.num_layers,
                "num_attention_heads": self._loaded_bundle.model_config.num_attention_heads,
                "pooling": self._loaded_bundle.model_config.pooling,
            },
        }
        if expose_selected_metrics:
            payload["selected_metrics"] = dict(self._champion_manifest.selected_metrics.items())
        return payload

    def predict_texts(self, texts: list[str]) -> list[PredictionRecord]:
        """Run model inference for a non-empty batch of input texts."""

        if not texts:
            return []

        logits = self._forward_logits(texts)
        calibrated = False
        if self._temperature is not None:
            logits = apply_temperature(logits, self._temperature)
            calibrated = True

        probabilities = probabilities_from_logits(logits)
        confidence = max_softmax_confidence(probabilities)
        entropy = predictive_entropy(probabilities)
        margin = confidence_margin(probabilities)
        predicted_indices = torch.argmax(probabilities, dim=1)

        probability_rows = probabilities.detach().cpu().tolist()
        confidence_values = confidence.detach().cpu().tolist()
        entropy_values = entropy.detach().cpu().tolist()
        margin_values = margin.detach().cpu().tolist()
        predicted_values = predicted_indices.detach().cpu().tolist()

        records: list[PredictionRecord] = []
        for row_index, predicted_index in enumerate(predicted_values):
            label_index = int(predicted_index)
            probability_map = {
                self._loaded_bundle.label_map.get(str(class_index), str(class_index)): float(value)
                for class_index, value in enumerate(probability_rows[row_index])
            }
            records.append(
                PredictionRecord(
                    label=self._loaded_bundle.label_map.get(str(label_index), str(label_index)),
                    label_index=label_index,
                    confidence=float(confidence_values[row_index]),
                    probabilities=probability_map,
                    entropy=float(entropy_values[row_index]),
                    margin=float(margin_values[row_index]),
                    calibrated=calibrated,
                )
            )
        return records

    def _forward_logits(self, texts: list[str]) -> torch.Tensor:
        encodings = self._loaded_bundle.tokenizer.encode_batch(texts)
        input_ids = torch.tensor(
            [encoding.ids for encoding in encodings],
            dtype=torch.long,
            device=self._model_device,
        )
        attention_mask = torch.tensor(
            [encoding.attention_mask for encoding in encodings],
            dtype=torch.long,
            device=self._model_device,
        )

        with torch.no_grad():
            return cast(
                torch.Tensor,
                self._loaded_bundle.model(input_ids, attention_mask=attention_mask),
            )


def _resolve_temperature(
    calibration: dict[str, object] | None,
    *,
    enable_calibration: bool,
) -> float | None:
    if not enable_calibration or calibration is None:
        return None

    enabled_flag = calibration.get("enabled")
    if isinstance(enabled_flag, bool) and not enabled_flag:
        return None

    raw_temperature = calibration.get("temperature")
    if raw_temperature is None:
        if enabled_flag is True:
            raise ValueError("calibration.json has enabled=true but no temperature value.")
        return None

    if not isinstance(raw_temperature, (int, float)):
        raise ValueError("calibration.json temperature must be numeric.")

    temperature = float(raw_temperature)
    if not math.isfinite(temperature) or temperature <= 0.0:
        raise ValueError("calibration.json temperature must be a finite positive value.")
    return temperature


def _resolve_model_device(model: torch.nn.Module) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration:
        return torch.device("cpu")
