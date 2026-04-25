# Streamlit dashboard

## Overview

The repository dashboard (`streamlit_app.py`) presents evaluation, calibration, uncertainty, and promotion metadata in a single UI, with optional live inference through the existing FastAPI service.

Sections included:

- Overview
- Results
- Visualizations
- Calibration & Uncertainty
- Live Inference
- Model Metadata

## Local usage

```bash
uv run streamlit run streamlit_app.py
```

Optional local API integration:

```bash
API_BASE_URL=http://localhost:7860 uv run streamlit run streamlit_app.py
```

To generate the dashboard artifact set:

```bash
uv run python scripts/evaluate.py --data-config configs/data.yaml --model-config configs/model.yaml --train-config configs/train.yaml --output-dir artifacts/evaluation_full_run
```

## Configuration

Environment variables:

- `DASHBOARD_ARTIFACT_DIR` (default `artifacts/evaluation_full_run`)
- `DASHBOARD_BUNDLE_DIR` (default `artifacts/model/bundle`)
- `API_BASE_URL` (optional)
- `DASHBOARD_TITLE`
- `DASHBOARD_SUBTITLE`

The dashboard loads:

- `metrics_validation.json`
- `metrics_test.json`
- optional calibrated metrics JSON files
- confusion/reliability/confidence/entropy PNG outputs
- optional `bundle_metadata.json` and `champion_manifest.json`

## Streamlit Community Cloud deployment

1. Push the repository to GitHub.
2. In Streamlit Community Cloud, create a new app from the repository.
3. Set main file path to `streamlit_app.py`.
4. Set Python version to 3.12 in app settings if needed.
5. Add optional secrets/environment variables (for example `API_BASE_URL`) as required.
6. Deploy and validate all tabs.

## Optional container deployment

Build and run the dashboard container:

```bash
docker build -f Dockerfile.streamlit -t bayes-gp-llmops-dashboard:latest .
docker run --rm -p 7860:7860 bayes-gp-llmops-dashboard:latest
```

## Hugging Face Spaces options

- **Streamlit SDK:** deploy with `streamlit_app.py` as the entry file.
- **Docker SDK:** deploy with `Dockerfile.streamlit` and expose port `7860`.

For live inference in Spaces, set `API_BASE_URL` to the serving endpoint URL.
