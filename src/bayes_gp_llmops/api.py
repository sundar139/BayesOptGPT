from __future__ import annotations

import logging
import platform
from importlib.metadata import PackageNotFoundError, version

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from pydantic import BaseModel, Field, model_validator

from .serving.config import ServingConfig, load_serving_config, resolve_serving_config_path
from .serving.runtime import PredictionRecord, ServingRuntime, ServingStartupError

LOGGER = logging.getLogger("bayes_gp_llmops.api")


class StructuredInput(BaseModel):
    """Structured single-record inference input."""

    text: str = Field(min_length=1)
    id: str | None = Field(default=None, max_length=128)


class PredictRequest(BaseModel):
    """Single-inference request payload."""

    input: str | StructuredInput

    @model_validator(mode="before")
    @classmethod
    def _coerce_text_payload(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        if "input" in data or "text" not in data:
            return data

        if "id" in data:
            return {"input": {"text": data["text"], "id": data["id"]}}
        return {"input": data["text"]}


class PredictBatchRequest(BaseModel):
    """Batch inference request payload."""

    inputs: list[str | StructuredInput] = Field(min_length=1)


class PredictionOutput(BaseModel):
    """Stable prediction representation returned by serving APIs."""

    input_id: str | None = None
    label: str
    label_index: int
    confidence: float
    probabilities: dict[str, float]
    entropy: float
    margin: float
    calibrated: bool


class PredictResponse(BaseModel):
    """Single-inference response payload."""

    prediction: PredictionOutput


class PredictBatchResponse(BaseModel):
    """Batch inference response payload."""

    count: int
    predictions: list[PredictionOutput]


class HealthResponse(BaseModel):
    """Readiness status for deployment health checks."""

    status: str
    bundle_validation_status: str
    model_loaded: bool
    calibration_enabled: bool


class MetadataArtifactsResponse(BaseModel):
    """Artifact availability summary for the promoted inference bundle."""

    checkpoint_available: bool
    tokenizer_available: bool
    model_config_available: bool
    data_config_available: bool
    label_map_available: bool
    manifest_available: bool
    calibration_available: bool


class MetadataResponse(BaseModel):
    """Serving metadata payload suitable for external exposure."""

    model_name: str
    bundle_id: str
    bundle_schema_version: str
    labels: list[str]
    calibration_enabled: bool
    artifacts: MetadataArtifactsResponse
    selected_metrics: dict[str, float | None] | None = None


class VersionResponse(BaseModel):
    """Runtime and package version information."""

    service: str
    package_version: str
    python_version: str
    platform: str


def create_app(
    *,
    serving_config: ServingConfig | None = None,
    runtime: ServingRuntime | None = None,
) -> FastAPI:
    config = serving_config or _load_default_serving_config()

    application = FastAPI(
        title="BayesOptGPT Serving",
        version="1.0.0",
        description=(
            "Bundle-driven text classification service with uncertainty metrics and "
            "optional temperature scaling."
        ),
    )
    application.state.runtime = runtime
    application.state.serving_config = config

    @application.on_event("startup")
    def _startup() -> None:
        if application.state.runtime is not None:
            return

        try:
            application.state.runtime = ServingRuntime.load_from_bundle(config)
        except ServingStartupError as exc:
            LOGGER.exception("Serving startup failed.")
            raise RuntimeError(str(exc)) from exc

    @application.get("/", response_class=HTMLResponse)
    def root(request: Request) -> HTMLResponse:
        runtime_state = _get_runtime(request)
        health = HealthResponse(
            status="ok",
            bundle_validation_status=runtime_state.bundle_validation_status,
            model_loaded=runtime_state.model_loaded,
            calibration_enabled=runtime_state.calibration_active,
        )
        html = (
            "<!doctype html>"
            "<html><head><meta charset='utf-8'><title>BayesOptGPT Serving</title>"
            "<style>body{font-family:Segoe UI,Arial,sans-serif;max-width:840px;margin:32px auto;"
            "padding:0 16px;line-height:1.45;color:#0f172a;background:#f8fafc;}"
            "h1{margin-bottom:0.3rem;}"
            "a{color:#0369a1;text-decoration:none;}"
            "code{background:#e2e8f0;padding:2px 6px;border-radius:4px;}"
            "ul{padding-left:18px;}</style></head><body>"
            "<h1>BayesOptGPT Serving</h1>"
            "<p>Bundle-driven inference service for AG News text classification.</p>"
            f"<p><strong>Bundle:</strong> {runtime_state.bundle_identifier}</p>"
            "<ul>"
            f"<li>Health status: <code>{health.status}</code></li>"
            f"<li>Bundle validation: <code>{health.bundle_validation_status}</code></li>"
            f"<li>Model loaded: <code>{str(health.model_loaded).lower()}</code></li>"
            f"<li>Calibration active: <code>{str(health.calibration_enabled).lower()}</code></li>"
            "</ul>"
            "<p>API links: <a href='/docs'>/docs</a>, <a href='/health'>/health</a>, "
            "<a href='/metadata'>/metadata</a>, <a href='/version'>/version</a>.</p>"
            "</body></html>"
        )
        return HTMLResponse(content=html)

    @application.get("/health", response_model=HealthResponse)
    def health(request: Request) -> HealthResponse:
        runtime_state = _get_runtime(request)
        return HealthResponse(
            status="ok",
            bundle_validation_status=runtime_state.bundle_validation_status,
            model_loaded=runtime_state.model_loaded,
            calibration_enabled=runtime_state.calibration_active,
        )

    @application.get("/metadata", response_model=MetadataResponse)
    def metadata(request: Request) -> MetadataResponse:
        runtime_state = _get_runtime(request)
        payload = runtime_state.metadata_payload(
            expose_selected_metrics=config.expose_selected_metrics
        )
        if not config.expose_selected_metrics:
            payload["selected_metrics"] = None
        return MetadataResponse.model_validate(payload)

    @application.post("/predict", response_model=PredictResponse)
    def predict(payload: PredictRequest, request: Request) -> PredictResponse:
        runtime_state = _get_runtime(request)
        text, input_id = _normalize_input(payload.input)
        _validate_text_length(
            text,
            max_length=config.max_input_length_chars,
        )
        prediction = _predict_with_guard(runtime_state, [text])[0]
        return PredictResponse(prediction=_to_prediction_output(prediction, input_id=input_id))

    @application.post("/predict/batch", response_model=PredictBatchResponse)
    def predict_batch(payload: PredictBatchRequest, request: Request) -> PredictBatchResponse:
        runtime_state = _get_runtime(request)

        if len(payload.inputs) > config.max_batch_size:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"Batch size {len(payload.inputs)} exceeds configured maximum "
                    f"{config.max_batch_size}."
                ),
            )

        normalized_inputs: list[tuple[str, str | None]] = []
        for index, input_payload in enumerate(payload.inputs):
            text, input_id = _normalize_input(input_payload)
            _validate_text_length(
                text,
                max_length=config.max_input_length_chars,
                index=index,
            )
            normalized_inputs.append((text, input_id))

        records = _predict_with_guard(runtime_state, [item[0] for item in normalized_inputs])
        predictions = [
            _to_prediction_output(record, input_id=normalized_inputs[index][1])
            for index, record in enumerate(records)
        ]
        return PredictBatchResponse(count=len(predictions), predictions=predictions)

    @application.get("/version", response_model=VersionResponse)
    def service_version() -> VersionResponse:
        return VersionResponse(
            service="BayesOptGPT Serving",
            package_version=_package_version(),
            python_version=platform.python_version(),
            platform=platform.platform(),
        )

    return application


def _load_default_serving_config() -> ServingConfig:
    config_path = resolve_serving_config_path()
    return load_serving_config(config_path)


def _get_runtime(request: Request) -> ServingRuntime:
    runtime_state = getattr(request.app.state, "runtime", None)
    if runtime_state is None:
        raise HTTPException(status_code=503, detail="Model runtime is not available.")
    if not isinstance(runtime_state, ServingRuntime):
        raise HTTPException(status_code=500, detail="Service runtime is misconfigured.")
    return runtime_state


def _normalize_input(input_payload: str | StructuredInput) -> tuple[str, str | None]:
    if isinstance(input_payload, str):
        text = input_payload.strip()
        if not text:
            raise HTTPException(status_code=422, detail="Input text must not be empty.")
        return text, None

    text = input_payload.text.strip()
    if not text:
        raise HTTPException(status_code=422, detail="Input text must not be empty.")
    return text, input_payload.id


def _validate_text_length(
    text: str,
    *,
    max_length: int,
    index: int | None = None,
) -> None:
    if len(text) <= max_length:
        return
    suffix = f" at index {index}" if index is not None else ""
    raise HTTPException(
        status_code=422,
        detail=(
            f"Input text length{suffix} exceeds configured maximum "
            f"{max_length} characters."
        ),
    )


def _predict_with_guard(
    runtime: ServingRuntime,
    texts: list[str],
) -> list[PredictionRecord]:
    try:
        return runtime.predict_texts(texts)
    except HTTPException:
        raise
    except Exception as exc:
        LOGGER.exception("Prediction request failed.")
        raise HTTPException(status_code=500, detail="Prediction failed.") from exc


def _to_prediction_output(
    record: PredictionRecord,
    *,
    input_id: str | None,
) -> PredictionOutput:
    return PredictionOutput(
        input_id=input_id,
        label=record.label,
        label_index=record.label_index,
        confidence=record.confidence,
        probabilities=record.probabilities,
        entropy=record.entropy,
        margin=record.margin,
        calibrated=record.calibrated,
    )


def _package_version() -> str:
    try:
        return version("bayes-gp-llmops")
    except PackageNotFoundError:
        return "unknown"


app = create_app()
