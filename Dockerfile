FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1

WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app app \
    && pip install --upgrade pip uv

COPY pyproject.toml uv.lock README.md /app/
COPY src /app/src
COPY scripts /app/scripts
COPY configs /app/configs

RUN uv sync --frozen --no-dev

RUN chown -R app:app /app

EXPOSE 7860

ENV SERVING_CONFIG_PATH=/app/configs/serving.yaml \
    SERVING_BUNDLE_DIR=/app/artifacts/model/bundle \
    SERVING_HOST=0.0.0.0 \
    SERVING_PORT=7860 \
    HOST=0.0.0.0 \
    PORT=7860

USER app

CMD ["uv", "run", "bayes-serve"]
