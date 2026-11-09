"""Input validation, inference and response normalization — no web framework here."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.core.errors import ApiError
from app.schemas.detection import (
    BoundingBox,
    Detection,
    ImageDetectionData,
    VideoDetectionData,
    VideoFrameDetections,
)
from app.services.model_service import Detector, RawDetection

IMAGE_CONTENT_TYPES = {"image/jpeg", "image/png", "image/webp"}
VIDEO_CONTENT_TYPES = {"video/mp4", "video/webm", "video/quicktime", "video/x-msvideo"}


def normalize_detections(
    raw: list[RawDetection],
    names: dict[int, str],
    image_width: int,
    image_height: int,
    allowed_classes: list[str],
    min_confidence: float,
) -> list[Detection]:
    """Model output -> contract: clamp to the image, top-left + size in whole
    pixels, drop empty boxes / unwanted classes / low confidence, most confident first."""
    allowed = set(allowed_classes)
    out: list[Detection] = []
    for d in raw:
        name = names.get(d.class_id)
        if name is None or (allowed and name not in allowed):
            continue
        if not math.isfinite(d.confidence) or d.confidence < min_confidence:
            continue
        x1 = min(max(round(d.x1), 0), image_width)
        y1 = min(max(round(d.y1), 0), image_height)
        x2 = min(max(round(d.x2), 0), image_width)
        y2 = min(max(round(d.y2), 0), image_height)
        if x2 <= x1 or y2 <= y1:
            continue
        out.append(
            Detection(
                class_id=d.class_id,
                class_name=name,
                confidence=round(d.confidence, 4),
                bounding_box=BoundingBox(x=x1, y=y1, width=x2 - x1, height=y2 - y1),
            )
        )
    out.sort(key=lambda det: det.confidence, reverse=True)
    return out


def check_content_type(content_type: str | None, allowed: set[str]) -> None:
    base = (content_type or "").split(";")[0].strip().lower()
    if base not in allowed:
        raise ApiError(
            415,
            "UNSUPPORTED_MEDIA_TYPE",
            f"Unsupported media type '{base or 'none'}'; expected one of {sorted(allowed)}",
        )


def decode_image(data: bytes, max_bytes: int, max_dimension: int) -> np.ndarray:
    if not data:
        raise ApiError(400, "EMPTY_FILE", "The uploaded file is empty")
    if len(data) > max_bytes:
        raise ApiError(413, "FILE_TOO_LARGE", f"File exceeds {max_bytes} bytes")
    image = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
    # The declared content type is only a hint — this is the real check.
    if image is None:
        raise ApiError(400, "INVALID_IMAGE", "The file is not a decodable image")
    height, width = image.shape[:2]
    if width > max_dimension or height > max_dimension:
        raise ApiError(
            413, "IMAGE_TOO_LARGE", f"Image is {width}x{height}; maximum side is {max_dimension}px"
        )
    return image


@dataclass
class DetectionContext:
    detector: Detector
    allowed_classes: list[str]
    min_confidence: float


def detect_image(ctx: DetectionContext, image: np.ndarray) -> ImageDetectionData:
    height, width = image.shape[:2]
    started = time.perf_counter()
    raw = ctx.detector.detect(image)
    elapsed_ms = (time.perf_counter() - started) * 1000
    return ImageDetectionData(
        image_width=width,
        image_height=height,
        detections=normalize_detections(
            raw, ctx.detector.names, width, height, ctx.allowed_classes, ctx.min_confidence
        ),
        inference_ms=round(elapsed_ms, 2),
    )


def detect_video(
    ctx: DetectionContext,
    path: Path,
    sample_interval_ms: int,
    max_sampled_frames: int,
    max_dimension: int,
) -> VideoDetectionData:
    """Offline experiments: run detection on one frame every ``sample_interval_ms``."""
    capture = cv2.VideoCapture(str(path))
    try:
        if not capture.isOpened():
            raise ApiError(400, "INVALID_VIDEO", "The file is not a readable video")
        fps = capture.get(cv2.CAP_PROP_FPS) or 0.0
        frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        width = int(capture.get(cv2.CAP_PROP_FRAME_WIDTH) or 0)
        height = int(capture.get(cv2.CAP_PROP_FRAME_HEIGHT) or 0)
        if fps <= 0 or width <= 0 or height <= 0:
            raise ApiError(400, "INVALID_VIDEO", "The video has no readable frames")
        if width > max_dimension or height > max_dimension:
            raise ApiError(
                413, "VIDEO_TOO_LARGE", f"Video is {width}x{height}; maximum side is {max_dimension}px"
            )

        frames: list[VideoFrameDetections] = []
        total_ms = 0.0
        next_sample_ms = 0.0
        index = 0
        truncated = False
        while True:
            ok, frame = capture.read()
            if not ok:
                break
            timestamp_ms = index * 1000.0 / fps
            if timestamp_ms + 1e-6 >= next_sample_ms:
                if len(frames) >= max_sampled_frames:
                    truncated = True
                    break
                result = detect_image(ctx, frame)
                total_ms += result.inference_ms
                frames.append(
                    VideoFrameDetections(
                        frame_index=index,
                        timestamp_ms=round(timestamp_ms),
                        detections=result.detections,
                    )
                )
                next_sample_ms += sample_interval_ms
            index += 1

        if not frames:
            raise ApiError(400, "INVALID_VIDEO", "The video has no readable frames")
        decoded = max(index, frame_count)
        return VideoDetectionData(
            video_width=width,
            video_height=height,
            fps=round(fps, 3),
            duration_ms=round(decoded * 1000.0 / fps),
            sample_interval_ms=sample_interval_ms,
            sampled_frames=len(frames),
            truncated=truncated,
            frames=frames,
            inference_ms_total=round(total_ms, 2),
        )
    finally:
        capture.release()
