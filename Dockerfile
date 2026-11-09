# CPU-only image for the detection service, built with uv from uv.lock.
FROM python:3.13-slim

COPY --from=ghcr.io/astral-sh/uv:0.11.14 /uv /uvx /bin/

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    # Use the image's Python 3.13 (matches .python-version); never download one.
    UV_PYTHON_DOWNLOADS=never \
    # Ultralytics writes settings/caches here; keep it inside the app dir.
    YOLO_CONFIG_DIR=/app/.ultralytics \
    # No runtime pip installs from inside the library.
    YOLO_AUTOINSTALL=false

# OpenCV runtime libraries.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Dependencies first (cached layer). --frozen: exactly uv.lock, which pins torch
# to the CPU build on Linux (see [tool.uv.sources] in pyproject.toml).
COPY pyproject.toml uv.lock .python-version ./
# The uv cache lives in a BuildKit cache mount, not in the image layer.
RUN --mount=type=cache,target=/root/.cache/uv uv sync --frozen --no-dev
ENV PATH="/app/.venv/bin:$PATH"

COPY app ./app

# Bake the weights into the image: no download at startup, same model on every boot.
ARG MODEL_NAME=yolo11n.pt
RUN mkdir -p /app/models \
    && cd /app/models \
    && python -c "import sys; sys.path.insert(0, '/app'); from app.services.model_service import prepare_ultralytics; prepare_ultralytics(); from ultralytics import YOLO; YOLO('${MODEL_NAME}')" \
    && test -f "/app/models/${MODEL_NAME}"
ENV AI_MODEL_PATH=/app/models/${MODEL_NAME}

# Code, venv and weights stay root-owned and read-only for the service user;
# only Ultralytics' settings directory needs to be writable.
RUN useradd --create-home --uid 10001 app \
    && mkdir -p /app/.ultralytics \
    && chown -R app /app/.ultralytics
USER app

EXPOSE 8000
# AI_SERVICE_TOKEN must be provided at runtime; the service refuses to start without it.
CMD ["sh", "-c", "uvicorn --factory app.main:create_app --host 0.0.0.0 --port ${PORT:-8000} --workers 1"]
