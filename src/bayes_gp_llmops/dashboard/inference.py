from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import httpx


@dataclass(frozen=True)
class PredictionResult:
    label: str
    label_index: int
    confidence: float
    probabilities: dict[str, float]
    entropy: float
    margin: float
    calibrated: bool


def normalize_api_base_url(value: str | None) -> str | None:
    if value is None:
        return None
    stripped = value.strip()
    if not stripped:
        return None
    return stripped.rstrip("/")


def fetch_serving_metadata(
    *,
    api_base_url: str,
    timeout_seconds: float = 10.0,
    client: httpx.Client | None = None,
) -> dict[str, object]:
    normalized = _require_base_url(api_base_url)
    response = _request_json(
        method="GET",
        url=f"{normalized}/metadata",
        json_payload=None,
        timeout_seconds=timeout_seconds,
        client=client,
    )
    if not isinstance(response, dict):
        raise ValueError("Expected JSON object response from /metadata.")
    return cast(dict[str, object], response)


def run_single_prediction(
    *,
    api_base_url: str,
    text: str,
    timeout_seconds: float = 10.0,
    client: httpx.Client | None = None,
) -> PredictionResult:
    normalized = _require_base_url(api_base_url)
    payload_text = text.strip()
    if not payload_text:
        raise ValueError("Input text must not be empty.")
    response = _request_json(
        method="POST",
        url=f"{normalized}/predict",
        json_payload={"input": payload_text},
        timeout_seconds=timeout_seconds,
        client=client,
    )
    if not isinstance(response, dict):
        raise ValueError("Expected JSON object response from /predict.")
    prediction = response.get("prediction")
    if not isinstance(prediction, dict):
        raise ValueError("Response payload from /predict missing 'prediction' object.")
    return _parse_prediction(prediction)


def run_batch_prediction(
    *,
    api_base_url: str,
    texts: list[str],
    timeout_seconds: float = 10.0,
    client: httpx.Client | None = None,
) -> list[PredictionResult]:
    normalized = _require_base_url(api_base_url)
    normalized_inputs = [item.strip() for item in texts if item.strip()]
    if not normalized_inputs:
        raise ValueError("Batch input must contain at least one non-empty text record.")
    response = _request_json(
        method="POST",
        url=f"{normalized}/predict/batch",
        json_payload={"inputs": normalized_inputs},
        timeout_seconds=timeout_seconds,
        client=client,
    )
    if not isinstance(response, dict):
        raise ValueError("Expected JSON object response from /predict/batch.")
    raw_predictions = response.get("predictions")
    if not isinstance(raw_predictions, list):
        raise ValueError("Response payload from /predict/batch missing 'predictions' list.")
    parsed: list[PredictionResult] = []
    for item in raw_predictions:
        if not isinstance(item, dict):
            raise ValueError("Prediction item in /predict/batch response must be an object.")
        parsed.append(_parse_prediction(item))
    return parsed


def _parse_prediction(payload: dict[str, object]) -> PredictionResult:
    label = payload.get("label")
    label_index = payload.get("label_index")
    confidence = payload.get("confidence")
    entropy = payload.get("entropy")
    margin = payload.get("margin")
    calibrated = payload.get("calibrated")
    probabilities_raw = payload.get("probabilities")

    if not isinstance(label, str):
        raise ValueError("Prediction payload missing string 'label'.")
    if isinstance(label_index, bool) or not isinstance(label_index, int):
        raise ValueError("Prediction payload missing integer 'label_index'.")
    if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
        raise ValueError("Prediction payload missing numeric 'confidence'.")
    if isinstance(entropy, bool) or not isinstance(entropy, (int, float)):
        raise ValueError("Prediction payload missing numeric 'entropy'.")
    if isinstance(margin, bool) or not isinstance(margin, (int, float)):
        raise ValueError("Prediction payload missing numeric 'margin'.")
    if not isinstance(calibrated, bool):
        raise ValueError("Prediction payload missing boolean 'calibrated'.")
    if not isinstance(probabilities_raw, dict):
        raise ValueError("Prediction payload missing object 'probabilities'.")

    probabilities: dict[str, float] = {}
    for key, value in probabilities_raw.items():
        if not isinstance(key, str):
            raise ValueError("Probability keys must be strings.")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError("Probability values must be numeric.")
        probabilities[key] = float(value)

    return PredictionResult(
        label=label,
        label_index=label_index,
        confidence=float(confidence),
        probabilities=probabilities,
        entropy=float(entropy),
        margin=float(margin),
        calibrated=calibrated,
    )


def _request_json(
    *,
    method: str,
    url: str,
    json_payload: dict[str, object] | None,
    timeout_seconds: float,
    client: httpx.Client | None,
) -> object:
    try:
        if client is not None:
            response = client.request(method, url, json=json_payload, timeout=timeout_seconds)
            response.raise_for_status()
            return response.json()
        with httpx.Client() as active_client:
            response = active_client.request(
                method,
                url,
                json=json_payload,
                timeout=timeout_seconds,
            )
            response.raise_for_status()
            return response.json()
    except httpx.HTTPError as exc:
        raise RuntimeError(f"Request failed for {url}: {exc}") from exc


def _require_base_url(api_base_url: str) -> str:
    normalized = normalize_api_base_url(api_base_url)
    if normalized is None:
        raise ValueError("API base URL is not configured.")
    return normalized
