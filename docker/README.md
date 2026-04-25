# Docker assets

- `docker/Dockerfile`: container image for local validation and production-like serving.
- Root `Dockerfile`: Hugging Face Docker Space compatible runtime on port `7860`.

## Runtime profile

- Python 3.12 slim base
- reproducible dependency sync (`uv sync --frozen --no-dev`)
- non-root runtime user
- startup command: `uv run bayes-serve`
- default serving port: `7860`

## Local build and run

```bash
docker build -t bayes-gp-llmops:latest .
```

```bash
docker run --rm -p 7860:7860 \
	-e SERVING_BUNDLE_DIR=/app/artifacts/model/bundle \
	-v "${PWD}/artifacts/model/bundle:/app/artifacts/model/bundle:ro" \
	bayes-gp-llmops:latest
```

The container validates bundle integrity at startup and exits with a clear error if bundle
requirements are not satisfied.

## Environment variables

- `SERVING_CONFIG_PATH` (default `/app/configs/serving.yaml`)
- `SERVING_BUNDLE_DIR` (default `/app/artifacts/model/bundle`)
- `SERVING_HOST` (default `0.0.0.0`)
- `SERVING_PORT` (default `7860`)
- `SERVING_ENABLE_CALIBRATION`
- `SERVING_MAX_BATCH_SIZE`
- `SERVING_MAX_INPUT_LENGTH_CHARS`
- `SERVING_LOG_LEVEL`
