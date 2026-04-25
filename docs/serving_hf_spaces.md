# Bundle-driven serving and Docker Spaces deployment

## Inference source of truth

Production inference uses only the promoted bundle directory (`artifacts/model/bundle` by default).
Serving does not read from tuning-study internals, trial directories, or arbitrary checkpoints.

Bundle runtime loading includes:

- required-file presence checks
- SHA-256 integrity verification via `checksums.json`
- model checkpoint and tokenizer load
- model/data config snapshot load
- label metadata load
- champion manifest metadata load
- optional calibration artifact load

## Startup behavior

Service startup performs bundle validation before model construction.

When startup is successful:

- `/health` reports `bundle_validation_status=passed` and `model_loaded=true`
- `/metadata` returns a whitelisted payload with model name, bundle id/version, labels,
  calibration status, selected metrics (configurable), and artifact availability booleans

When startup fails:

- process exits with a clear startup error suitable for container logs
- startup failure is deterministic for missing files, checksum mismatch, or invalid calibration payload

## API contract

### `GET /health`

Readiness response with:

- status
- bundle validation status
- model loaded status
- calibration enabled status

### `GET /metadata`

Exposes safe metadata:

- model name
- bundle id + bundle schema version
- label names
- calibration enabled status
- artifact availability booleans (`checkpoint`, `tokenizer`, `config`, `manifest`, etc.)
- selected metrics from promotion manifest when enabled in serving config

### `POST /predict`

Accepts a single item (preferred canonical form):

```json
{
  "input": "Global market update"
}
```

or structured payload:

```json
{
  "input": {
    "id": "record-1",
    "text": "Global market update"
  }
}
```

Backward-compatible shorthand is also accepted:

```json
{
  "text": "Global market update"
}
```

```json
{
  "id": "record-1",
  "text": "Global market update"
}
```

Returns label prediction + uncertainty metrics:

- label
- confidence
- probabilities
- entropy
- margin
- calibrated flag

### `POST /predict/batch`

Accepts:

```json
{
  "inputs": [
    "World headline",
    {
      "id": "row-2",
      "text": "Sports headline"
    }
  ]
}
```

Returns a deterministic list of prediction objects matching input order.

## Serving configuration

`configs/serving.yaml` controls:

- `bundle_dir`
- `host`
- `port`
- `device_preference`
- `max_batch_size`
- `max_input_length_chars`
- `enable_calibration`
- `expose_selected_metrics`
- `log_level`

Environment overrides:

- `SERVING_CONFIG_PATH`
- `SERVING_BUNDLE_DIR`
- `SERVING_HOST`
- `SERVING_PORT`
- `SERVING_DEVICE_PREFERENCE`
- `SERVING_MAX_BATCH_SIZE`
- `SERVING_MAX_INPUT_LENGTH_CHARS`
- `SERVING_ENABLE_CALIBRATION`
- `SERVING_EXPOSE_SELECTED_METRICS`
- `SERVING_LOG_LEVEL`

## Local operation

```bash
uv run python scripts/promote.py --tuning-dir artifacts/tuning --output-dir artifacts/model/bundle --tokenizer-dir artifacts/tokenizer
uv run python scripts/validate_bundle.py --bundle-dir artifacts/model/bundle
uv run python scripts/serve.py --config configs/serving.yaml
```

## Docker image

```bash
docker build -t bayes-gp-llmops:latest .
```

```bash
docker run --rm -p 7860:7860 \
  -e SERVING_BUNDLE_DIR=/app/artifacts/model/bundle \
  -v "${PWD}/artifacts/model/bundle:/app/artifacts/model/bundle:ro" \
  bayes-gp-llmops:latest
```

Container startup exits with a clear validation error when bundle requirements are not met.

## Hugging Face Docker Spaces

Target profile:

- SDK: Docker
- port: `7860`
- startup command: from repository Dockerfile (`uv run bayes-serve`)

Recommended Space environment variables:

- `SERVING_CONFIG_PATH=/app/configs/serving.yaml`
- `SERVING_BUNDLE_DIR=/app/artifacts/model/bundle`
- `SERVING_PORT=7860`
- `SERVING_HOST=0.0.0.0`

Bundle update policy:

- publish a new promoted bundle
- validate with `scripts/validate_bundle.py`
- update repository bundle source (or governed external source)
- redeploy Space from the updated revision

Storage note:

- free Space runtime storage is ephemeral
- persisted model artifacts should come from git-tracked bundle content or a reliable external source restored on each container start
