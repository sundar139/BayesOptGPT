# bayes-gp-llmops

`bayes-gp-llmops` is an end-to-end NLP classification system for AG News built around a custom tiny LLaMA-inspired transformer, uncertainty-aware evaluation, reproducible tuning, and bundle-driven serving.

## Implemented system

- **Model:** custom compact LLaMA-inspired classifier (RMSNorm, RoPE attention, SwiGLU blocks)
- **Data pipeline:** AG News ingestion via Hugging Face `datasets` + local tokenizer artifact generation
- **Evaluation:** full split metrics, uncertainty summaries, calibration diagnostics, and plot exports
- **Tuning:** Optuna search with MLflow tracking
- **Promotion:** deterministic champion selection and manifest-backed promotion
- **Packaging:** portable inference bundle with checksums and provenance metadata
- **Serving:** FastAPI/Uvicorn runtime that loads strictly from validated bundle artifacts
- **Deployment readiness:** Docker and Hugging Face Docker Spaces aligned to port `7860`

## Setup (uv)

```bash
uv lock
uv sync --all-groups
```

Create `.env` from `.env.example` before running training, tuning, or serving.

## Command map

| Purpose | Make target | Script command | Package entrypoint |
| --- | --- | --- | --- |
| Install dependencies | `make install` | `uv sync --all-groups` | — |
| Download data + tokenizer artifacts | `make download-data` | `uv run python scripts/download_data.py --config configs/data.yaml` | `uv run bayes-download-data --config configs/data.yaml` |
| Train | `make train` | `uv run python scripts/train.py --data-config configs/data.yaml --model-config configs/model.yaml --train-config configs/train.yaml` | `uv run bayes-train` |
| Evaluate + calibrate | `make evaluate` | `uv run python scripts/evaluate.py --data-config configs/data.yaml --model-config configs/model.yaml --train-config configs/train.yaml` | `uv run bayes-evaluate` |
| Tune | `make tune` | `uv run python scripts/tune.py --data-config configs/data.yaml --model-config configs/model.yaml --train-config configs/train.yaml --tune-config configs/tune.yaml` | `uv run bayes-tune` |
| Promote champion bundle | `make promote` | `uv run python scripts/promote.py --tuning-dir artifacts/tuning --output-dir artifacts/model/bundle --tokenizer-dir artifacts/tokenizer` | `uv run bayes-promote` |
| Validate bundle | `make validate-bundle` | `uv run python scripts/validate_bundle.py --bundle-dir artifacts/model/bundle` | `uv run bayes-validate-bundle` |
| Serve from bundle | `make serve` | `uv run python scripts/serve.py --config configs/serving.yaml` | `uv run bayes-serve` |
| Launch Streamlit dashboard | `make dashboard` | `uv run streamlit run streamlit_app.py` | — |

Quality checks:

```bash
make lint
make typecheck
make test
```

## End-to-end workflow

1. **Download AG News data and tokenizer artifacts**
   ```bash
   uv run python scripts/download_data.py --config configs/data.yaml
   ```
2. **Train**
   ```bash
   uv run python scripts/train.py --data-config configs/data.yaml --model-config configs/model.yaml --train-config configs/train.yaml
   ```
3. **Evaluate and calibrate**
   ```bash
   uv run python scripts/evaluate.py --data-config configs/data.yaml --model-config configs/model.yaml --train-config configs/train.yaml
   ```
4. **Tune with Optuna + MLflow**
   ```bash
   uv run python scripts/tune.py --data-config configs/data.yaml --model-config configs/model.yaml --train-config configs/train.yaml --tune-config configs/tune.yaml
   uv run mlflow ui --backend-store-uri sqlite:///mlflow.db
   ```
5. **Promote champion model**
   ```bash
   uv run python scripts/promote.py --tuning-dir artifacts/tuning --output-dir artifacts/model/bundle --tokenizer-dir artifacts/tokenizer
   ```
6. **Validate bundle integrity**
   ```bash
   uv run python scripts/validate_bundle.py --bundle-dir artifacts/model/bundle
   ```
7. **Serve from bundle**
   ```bash
   uv run python scripts/serve.py --config configs/serving.yaml
   ```
8. **Deploy with Docker / Hugging Face Spaces**
   ```bash
   docker build -t bayes-gp-llmops:latest .
   docker run --rm -p 7860:7860 -e SERVING_BUNDLE_DIR=/app/artifacts/model/bundle -v "${PWD}/artifacts/model/bundle:/app/artifacts/model/bundle:ro" bayes-gp-llmops:latest
   ```

## Full-split results

These metrics are from **full-split** AG News evaluation using the implemented config-driven training/evaluation stack.

### Validation metrics

| Metric | Value |
| --- | --- |
| num_samples | `12000` |
| accuracy | `0.9258333333333333` |
| macro_f1 | `0.9255592243299131` |
| nll/loss | `0.23436810076236725` |
| brier_score | `0.11519614607095718` |
| ece | `0.03169432282447815` |
| mean_confidence | `0.9571210145950317` |
| mean_entropy | `0.11760636419057846` |

### Validation per-class F1

| Class | F1 |
| --- | --- |
| World | `0.9359919571045576` |
| Sports | `0.9764686522955406` |
| Business | `0.8911647283457733` |
| Sci/Tech | `0.898611559573781` |

### Test metrics

| Metric | Value |
| --- | --- |
| num_samples | `7600` |
| accuracy | `0.9271052631578948` |
| macro_f1 | `0.9270783914474838` |
| nll/loss | `0.24788044393062592` |
| brier_score | `0.11632658541202545` |
| ece | `0.03229384496808052` |
| mean_confidence | `0.9578530788421631` |
| mean_entropy | `0.11572850495576859` |

### Test per-class F1

| Class | F1 |
| --- | --- |
| World | `0.9342105263157895` |
| Sports | `0.9776492242966079` |
| Business | `0.8950159066808059` |
| Sci/Tech | `0.901437908496732` |

These full-split results show stable validation/test alignment and low calibration error, with strongest performance on Sports and the most frequent confusion between Business and Sci/Tech.

## Artifact layout

| Output | Location | Notes |
| --- | --- | --- |
| Tokenizer artifacts | `artifacts/tokenizer/` | `tokenizer.json`, `tokenizer_config.json`, `special_tokens_map.json`, `tokenizer_metadata.json` |
| Checkpoints | `artifacts/checkpoints/` | `best.ckpt`, `latest.ckpt`, `training_history.json`, `resolved_config.json` |
| Evaluation outputs | `artifacts/evaluation/` | split metrics, calibrated metrics, predictions CSV, calibration/uncertainty plots |
| Dashboard evaluation snapshot | `artifacts/evaluation_full_run/` | full-split metrics JSON + confusion/reliability/confidence/entropy PNG outputs |
| Tuning outputs | `artifacts/tuning/` | `study.db`, `study_summary.json`, `best_params.json`, `trial_results.csv`, `trials/trial_XXXX/` |
| Promoted model bundle | `artifacts/model/bundle/` | `checkpoint.ckpt`, `tokenizer/`, `model_config.json`, `data_config.json`, `champion_manifest.json`, `label_map.json`, optional `calibration.json`, `checksums.json`, `bundle_metadata.json` |
| Serving bundle inputs | `configs/serving.yaml` + `artifacts/model/bundle/` | serving runtime validates files and checksums before model load |

## Serving and deployment notes

- FastAPI serving is **bundle-driven**: it does not read Optuna trial internals at runtime.
- Startup validates required bundle files and SHA-256 integrity metadata.
- Docker runtime and Hugging Face Docker Spaces assets are configured for port `7860`.
- Default runtime command is `uv run bayes-serve` inside the container.

## Streamlit dashboard

The repository includes a portfolio-grade Streamlit dashboard (`streamlit_app.py`) with tabs for:

- Overview
- Results
- Visualizations
- Calibration & Uncertainty
- Live Inference
- Model Metadata

Local launch:

```bash
uv run streamlit run streamlit_app.py
```

To refresh dashboard artifacts with a deterministic output path:

```bash
uv run python scripts/evaluate.py --data-config configs/data.yaml --model-config configs/model.yaml --train-config configs/train.yaml --output-dir artifacts/evaluation_full_run
```

Optional API integration for live inference:

```bash
API_BASE_URL=http://localhost:7860 uv run streamlit run streamlit_app.py
```

Supported dashboard environment variables:

- `DASHBOARD_ARTIFACT_DIR` (default: `artifacts/evaluation_full_run`)
- `DASHBOARD_BUNDLE_DIR` (default: `artifacts/model/bundle`)
- `API_BASE_URL` (optional, enables `/predict`, `/predict/batch`, `/metadata`)
- `DASHBOARD_TITLE`
- `DASHBOARD_SUBTITLE`

### Streamlit Community Cloud deployment

1. Push this repository to GitHub with dashboard artifacts and app file.
2. In Streamlit Community Cloud, create a new app from the repository.
3. Set main file path to `streamlit_app.py`.
4. Configure optional secrets/environment variables (for example `API_BASE_URL`) as needed.
5. Deploy and verify tabs, plots, and optional live inference.

### Optional dashboard container deployment

```bash
docker build -f Dockerfile.streamlit -t bayes-gp-llmops-dashboard:latest .
docker run --rm -p 7860:7860 bayes-gp-llmops-dashboard:latest
```

For a dashboard on Hugging Face Spaces, use either:

- **Streamlit SDK** with `streamlit_app.py`, or
- **Docker SDK** with `Dockerfile.streamlit`.

## Operational note

- The repository is validated end-to-end in local development (download, train, evaluate, tune, promote, validate bundle, serve).
- Docker and Hugging Face Spaces deployment assets are prepared and aligned with the bundle-driven runtime model.
- Large-scale latency/load benchmarking is not yet included in this repository and should be added before high-throughput production rollout.

## Optional screenshots

Screenshots are not currently committed. When available, add captures for:

- Confusion matrix output
- Reliability diagram output
- Service API docs (`/docs`) or local service response capture
- MLflow experiment view (optional)

## Related documentation

- `docs/README.md`
- `docs/serving_hf_spaces.md`
