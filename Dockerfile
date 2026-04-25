FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    UV_LINK_MODE=copy \
    UV_COMPILE_BYTECODE=1 \
    HOME=/tmp \
    XDG_CACHE_HOME=/tmp/.cache \
    UV_CACHE_DIR=/tmp/.cache/uv

WORKDIR /app

RUN addgroup --system app && adduser --system --ingroup app app \
    && pip install --upgrade pip uv

COPY pyproject.toml uv.lock README.md /app/
COPY src /app/src
COPY scripts /app/scripts
COPY configs /app/configs
COPY artifacts/model/bundle /app/artifacts/model/bundle

RUN uv sync --frozen --no-dev

RUN chown -R app:app /app

RUN mkdir -p /home/app/.cache/uv && chown -R app:app /home/app

EXPOSE 7860

ENV SERVING_CONFIG_PATH=/app/configs/serving.yaml \
    SERVING_BUNDLE_DIR=/app/artifacts/model/bundle \
    SERVING_HOST=0.0.0.0 \
    SERVING_PORT=7860 \
    HOST=0.0.0.0 \
    PORT=7860 \
    PATH=/app/.venv/bin:${PATH} \
    HOME=/home/app \
    XDG_CACHE_HOME=/home/app/.cache \
    UV_CACHE_DIR=/home/app/.cache/uv

USER app

CMD ["bayes-serve"]
