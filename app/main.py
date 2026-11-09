"""Application factory. Run with: uvicorn --factory app.main:create_app"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncIterator

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from starlette.concurrency import run_in_threadpool

from app.api import detection, health
from app.core.config import MB, Settings, get_settings
from app.core.errors import register_error_handlers
from app.core.gate import InferenceGate
from app.services.model_service import Detector, UltralyticsDetector

logger = logging.getLogger(__name__)

# Multipart framing on top of the largest allowed file.
_REQUEST_OVERHEAD_BYTES = 1 * MB


def create_app(settings: Settings | None = None, detector: Detector | None = None) -> FastAPI:
    """``detector`` is injectable for tests; when omitted the real YOLO model is
    loaded once at startup. A model that fails to load leaves the service up but
    reporting ``unavailable`` (health 503, detection 503) instead of crashing."""
    settings = settings or get_settings()
    settings.assert_secure()

    @asynccontextmanager
    async def lifespan(app: FastAPI) -> AsyncIterator[None]:
        if app.state.detector is None:
            real = UltralyticsDetector(settings)
            try:
                await run_in_threadpool(real.load)
                app.state.detector = real
            except Exception:
                logger.exception("Detection model failed to load (%s)", settings.model_path)
        yield

    app = FastAPI(
        title="Recruitment Platform AI Service",
        description="Object detection for interview monitoring. Returns observations only — "
        "it never decides anything about a candidate.",
        version="0.1.0",
        lifespan=lifespan,
    )
    app.state.settings = settings
    app.state.detector = detector
    app.state.gate = InferenceGate(settings.max_pending_inferences)

    max_request = max(settings.max_image_bytes, settings.max_video_bytes) + _REQUEST_OVERHEAD_BYTES

    @app.middleware("http")
    async def reject_oversized(request: Request, call_next):  # type: ignore[no-untyped-def]
        # Cheap early refusal before the multipart body is parsed/spooled.
        length = request.headers.get("content-length")
        if length and length.isdigit() and int(length) > max_request:
            return JSONResponse(
                status_code=413,
                content={
                    "success": False,
                    "error": {"code": "REQUEST_TOO_LARGE", "message": "Request body too large"},
                },
            )
        return await call_next(request)

    register_error_handlers(app)
    app.include_router(health.router)
    app.include_router(detection.router)
    return app
