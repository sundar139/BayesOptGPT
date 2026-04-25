# Documentation

This directory contains operational and implementation documentation for the current `bayes-gp-llmops` system.

## System coverage

- Custom tiny LLaMA-inspired AG News classifier
- Uncertainty and calibration-aware evaluation
- Optuna tuning with MLflow tracking
- Deterministic champion promotion and portable bundle packaging
- Bundle-driven FastAPI serving
- Docker and Hugging Face Docker Spaces deployment guidance

## Primary runbook

- `docs/serving_hf_spaces.md` — serving, bundle validation, Docker runtime, and Spaces deployment operations
- `docs/streamlit_dashboard.md` — dashboard usage, API integration, and deployment options

## Core commands

```bash
uv run python scripts/download_data.py --config configs/data.yaml
uv run python scripts/train.py --data-config configs/data.yaml --model-config configs/model.yaml --train-config configs/train.yaml
uv run python scripts/evaluate.py --data-config configs/data.yaml --model-config configs/model.yaml --train-config configs/train.yaml
uv run python scripts/tune.py --data-config configs/data.yaml --model-config configs/model.yaml --train-config configs/train.yaml --tune-config configs/tune.yaml
uv run python scripts/promote.py --tuning-dir artifacts/tuning --output-dir artifacts/model/bundle --tokenizer-dir artifacts/tokenizer
uv run python scripts/validate_bundle.py --bundle-dir artifacts/model/bundle
uv run python scripts/serve.py --config configs/serving.yaml
uv run streamlit run streamlit_app.py
uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
```

## Artifact locations

| Output | Location |
| --- | --- |
| Dataset cache | `data/hf_cache/` |
| Processed cache workspace | `data/processed/` |
| Tokenizer artifacts | `artifacts/tokenizer/` |
| Training checkpoints | `artifacts/checkpoints/` |
| Evaluation outputs | `artifacts/evaluation/` |
| Dashboard evaluation snapshot | `artifacts/evaluation_full_run/` |
| Tuning outputs | `artifacts/tuning/` |
| Promoted bundle | `artifacts/model/bundle/` |

## Notes

- Promotion and serving operate from bundle artifacts, not tuning internals.
- Local default MLflow backend uses `sqlite:///mlflow.db`.
- Streamlit dashboard defaults to `artifacts/evaluation_full_run/` and can optionally call FastAPI via `API_BASE_URL`.
- Refer to the repository `README.md` for full-split result tables and end-to-end workflow.
