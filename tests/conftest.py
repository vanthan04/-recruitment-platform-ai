from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.services.model_service import RawDetection

TOKEN = "test-service-token"
AUTH = {"Authorization": f"Bearer {TOKEN}"}

# Real COCO ids for the classes this project cares about.
COCO_NAMES = {0: "person", 2: "car", 56: "chair", 63: "laptop", 67: "cell phone", 73: "book"}


class FakeDetector:
    """Stands in for YOLO: returns whatever the test sets, records what it was given."""

    name = "fake-yolo"
    task = "detect"
    device = "cpu"

    def __init__(self) -> None:
        self.names = dict(COCO_NAMES)
        self.raw: list[RawDetection] = []
        self.calls: list[tuple[int, int]] = []

    def detect(self, image_bgr: np.ndarray) -> list[RawDetection]:
        self.calls.append(image_bgr.shape[:2])
        return list(self.raw)


def make_settings(**overrides: object) -> Settings:
    base: dict[str, object] = {"service_token": TOKEN, "allow_anonymous": False}
    base.update(overrides)
    # _env_file=None: tests must not pick up a developer's local .env.
    return Settings(_env_file=None, **base)  # type: ignore[arg-type]


def encode_image(width: int = 320, height: int = 240, ext: str = ".jpg") -> bytes:
    image = np.full((height, width, 3), 127, np.uint8)
    ok, buf = cv2.imencode(ext, image)
    assert ok
    return buf.tobytes()


def write_video(path: Path, frames: int, fps: float = 10.0, size: tuple[int, int] = (160, 120)) -> Path:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    assert writer.isOpened(), "OpenCV could not open an mp4v writer"
    for i in range(frames):
        frame = np.full((size[1], size[0], 3), i % 255, np.uint8)
        writer.write(frame)
    writer.release()
    return path


@pytest.fixture
def detector() -> FakeDetector:
    return FakeDetector()


@pytest.fixture
def settings() -> Settings:
    return make_settings()


@pytest.fixture
def client(settings: Settings, detector: FakeDetector) -> TestClient:
    with TestClient(create_app(settings, detector=detector)) as c:
        yield c
