"""Response contract. JSON is camelCase to match the NestJS / Next.js side.

Bounding boxes are integer **pixel** coordinates in the **original** image:
``x``/``y`` is the **top-left** corner, ``width``/``height`` extend right/down,
always clamped inside ``imageWidth`` x ``imageHeight``.
"""

from __future__ import annotations

from typing import Generic, Literal, TypeVar

from pydantic import BaseModel, ConfigDict
from pydantic.alias_generators import to_camel

T = TypeVar("T")


class CamelModel(BaseModel):
    model_config = ConfigDict(alias_generator=to_camel, populate_by_name=True)


class BoundingBox(CamelModel):
    x: int
    y: int
    width: int
    height: int


class Detection(CamelModel):
    class_id: int
    class_name: str
    confidence: float
    bounding_box: BoundingBox


class ImageDetectionData(CamelModel):
    image_width: int
    image_height: int
    detections: list[Detection]
    inference_ms: float


class VideoFrameDetections(CamelModel):
    frame_index: int
    timestamp_ms: int
    detections: list[Detection]


class VideoDetectionData(CamelModel):
    video_width: int
    video_height: int
    fps: float
    duration_ms: int
    sample_interval_ms: int
    sampled_frames: int
    # True when the video had more sample points than the configured cap.
    truncated: bool
    frames: list[VideoFrameDetections]
    inference_ms_total: float


class ModelInfo(CamelModel):
    name: str
    task: str
    device: str
    image_size: int
    confidence_threshold: float
    iou_threshold: float
    # Every class the loaded model knows, by id.
    classes: dict[int, str]
    # The subset this service returns (empty = all).
    returned_classes: list[str]


class HealthData(CamelModel):
    status: Literal["ok", "unavailable"]
    model_loaded: bool


class Envelope(CamelModel, Generic[T]):
    success: Literal[True] = True
    data: T
