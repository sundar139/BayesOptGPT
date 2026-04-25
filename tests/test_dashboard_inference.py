from __future__ import annotations

import json

import httpx
import pytest

from bayes_gp_llmops.dashboard.inference import (
    fetch_serving_metadata,
    normalize_api_base_url,
    run_batch_prediction,
    run_single_prediction,
)


def test_normalize_api_base_url() -> None:
    assert normalize_api_base_url(None) is None
    assert normalize_api_base_url("   ") is None
    assert normalize_api_base_url("http://localhost:7860/") == "http://localhost:7860"


def test_run_single_prediction_parses_response() -> None:
    client = httpx.Client(transport=_transport_for_json({"prediction": _prediction_payload()}))
    prediction = run_single_prediction(
        api_base_url="http://localhost:7860",
        text="A quick market update",
        client=client,
    )
    client.close()

    assert prediction.label == "Business"
    assert prediction.label_index == 2
    assert prediction.confidence == 0.91
    assert prediction.calibrated is True


def test_run_batch_prediction_parses_response() -> None:
    payload = {
        "count": 2,
        "predictions": [
            _prediction_payload(),
            {
                **_prediction_payload(),
                "label": "Sports",
                "label_index": 1,
            },
        ],
    }
    client = httpx.Client(transport=_transport_for_json(payload))
    results = run_batch_prediction(
        api_base_url="http://localhost:7860",
        texts=["alpha", "beta"],
        client=client,
    )
    client.close()

    assert len(results) == 2
    assert results[0].label == "Business"
    assert results[1].label == "Sports"


def test_fetch_serving_metadata_returns_mapping() -> None:
    metadata_payload = {"bundle_id": "bundle-1", "trial_number": 1}
    client = httpx.Client(transport=_transport_for_json(metadata_payload))
    metadata = fetch_serving_metadata(
        api_base_url="http://localhost:7860",
        client=client,
    )
    client.close()

    assert metadata["bundle_id"] == "bundle-1"


def test_run_single_prediction_raises_on_empty_text() -> None:
    with pytest.raises(ValueError):
        run_single_prediction(api_base_url="http://localhost:7860", text=" ")


def _transport_for_json(payload: dict[str, object]) -> httpx.MockTransport:
    encoded = json.dumps(payload).encode("utf-8")

    def handler(_: httpx.Request) -> httpx.Response:
        return httpx.Response(
            status_code=200,
            content=encoded,
            headers={"Content-Type": "application/json"},
        )

    return httpx.MockTransport(handler)


def _prediction_payload() -> dict[str, object]:
    return {
        "input_id": None,
        "label": "Business",
        "label_index": 2,
        "confidence": 0.91,
        "probabilities": {
            "World": 0.04,
            "Sports": 0.02,
            "Business": 0.91,
            "Sci/Tech": 0.03,
        },
        "entropy": 0.25,
        "margin": 0.87,
        "calibrated": True,
    }
