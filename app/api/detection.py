from __future__ import annotations

import os
import tempfile
from pathlib import Path

from fastapi import APIRouter, Depends, File, Query, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from app.core.config import Settings
from app.core.errors import ApiError
from app.core.security import require_service_token
from app.schemas.detection import Envelope, ImageDetectionData, ModelInfo, VideoDetectionData
from app.services.detection_service import (
    IMAGE_CONTENT_TYPES,
    VIDEO_CONTENT_TYPES,
    DetectionContext,
    check_content_type,
    decode_image,
    detect_image,
    detect_video,
)
from app.services.model_service import Detector

router = APIRouter(dependencies=[Depends(require_service_token)])


def _settings(request: Request) -> Settings:
    return request.app.state.settings


def _detector(request: Request) -> Detector:
    detector = request.app.state.detector
    if detector is None:
        raise ApiError(503, "MODEL_UNAVAILABLE", "The detection model is not loaded")
    return detector


def _context(request: Request) -> DetectionContext:
    settings = _settings(request)
    return DetectionContext(
        detector=_detector(request),
        allowed_classes=settings.allowed_classes,
        min_confidence=settings.confidence_threshold,
    )


async def _read_capped(file: UploadFile, max_bytes: int) -> bytes:
    # Read one byte past the cap: enough to know it's too big without loading more.
    data = await file.read(max_bytes + 1)
    if len(data) > max_bytes:
        raise ApiError(413, "FILE_TOO_LARGE", f"File exceeds {max_bytes} bytes")
    return data


async def _detect_upload(request: Request, file: UploadFile, max_bytes: int) -> ImageDetectionData:
    settings = _settings(request)
    ctx = _context(request)
    check_content_type(file.content_type, IMAGE_CONTENT_TYPES)
    data = await _read_capped(file, max_bytes)
    image = decode_image(data, max_bytes, settings.max_image_dimension)
    async with request.app.state.gate:
        # CPU-bound: keep it off the event loop so /health stays responsive.
        return await run_in_threadpool(detect_image, ctx, image)


@router.get("/models", response_model=Envelope[ModelInfo], tags=["models"])
async def models(request: Request) -> Envelope[ModelInfo]:
    settings = _settings(request)
    detector = _detector(request)
    return Envelope[ModelInfo](
        data=ModelInfo(
            name=detector.name,
            task=detector.task,
            device=detector.device,
            image_size=settings.image_size,
            confidence_threshold=settings.confidence_threshold,
            iou_threshold=settings.iou_threshold,
            classes=detector.names,
            returned_classes=settings.allowed_classes,
        )
    )


@router.post("/detect/image", response_model=Envelope[ImageDetectionData], tags=["detection"])
async def detect_image_endpoint(
    request: Request, file: UploadFile = File(...)
) -> Envelope[ImageDetectionData]:
    """Detect objects in an uploaded JPEG/PNG/WebP image."""
    result = await _detect_upload(request, file, _settings(request).max_image_bytes)
    return Envelope[ImageDetectionData](data=result)


@router.post("/detect/frame", response_model=Envelope[ImageDetectionData], tags=["detection"])
async def detect_frame_endpoint(
    request: Request, file: UploadFile = File(...)
) -> Envelope[ImageDetectionData]:
    """One sampled webcam frame. Same contract as /detect/image with a tighter size limit."""
    result = await _detect_upload(request, file, _settings(request).max_frame_bytes)
    return Envelope[ImageDetectionData](data=result)


@router.post("/detect/video", response_model=Envelope[VideoDetectionData], tags=["detection"])
async def detect_video_endpoint(
    request: Request,
    file: UploadFile = File(...),
    sample_interval_ms: int | None = Query(None, alias="sampleIntervalMs", ge=100, le=60_000),
) -> Envelope[VideoDetectionData]:
    """Offline experiments: detect on one frame every ``sampleIntervalMs`` of a video file."""
    settings = _settings(request)
    ctx = _context(request)
    check_content_type(file.content_type, VIDEO_CONTENT_TYPES)
    data = await _read_capped(file, settings.max_video_bytes)
    if not data:
        raise ApiError(400, "EMPTY_FILE", "The uploaded file is empty")

    suffix = Path(file.filename or "").suffix[:8] or ".mp4"
    fd, tmp_name = tempfile.mkstemp(suffix=suffix)
    try:
        with os.fdopen(fd, "wb") as tmp:
            tmp.write(data)
        del data
        async with request.app.state.gate:
            result = await run_in_threadpool(
                detect_video,
                ctx,
                Path(tmp_name),
                sample_interval_ms or settings.default_video_sample_interval_ms,
                settings.max_video_sampled_frames,
                settings.max_image_dimension,
            )
    finally:
        # Uploaded media is never kept.
        Path(tmp_name).unlink(missing_ok=True)
    return Envelope[VideoDetectionData](data=result)
