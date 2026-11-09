"""Model lifecycle. The rest of the service talks to the ``Detector`` protocol only.

``ultralytics`` (and therefore torch) is imported lazily inside
``UltralyticsDetector.load`` so the API, its validation and its tests run without
the heavy ML stack installed.
"""

from __future__ import annotations

import logging
import os
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

import numpy as np

from app.core.config import Settings

logger = logging.getLogger(__name__)


def prepare_ultralytics() -> None:
    """Call before the first ``import ultralytics`` in the process.

    - Keeps Ultralytics' settings file inside the project (or the container)
      instead of the user's profile, so the change below is not machine-wide.
    - Turns off its anonymous usage analytics ("sync"): this service handles
      interview footage and should make no outbound calls besides fetching weights.
    - Stops it from pip-installing packages at runtime.
    """
    os.environ.setdefault("YOLO_CONFIG_DIR", str(Path(".ultralytics").resolve()))
    os.environ.setdefault("YOLO_AUTOINSTALL", "false")
    from ultralytics import settings as yolo_settings

    if yolo_settings.get("sync", True):
        yolo_settings.update({"sync": False})


@dataclass(frozen=True)
class RawDetection:
    """A model output in original-image pixels, corners form (x1, y1) top-left, (x2, y2) bottom-right."""

    x1: float
    y1: float
    x2: float
    y2: float
    confidence: float
    class_id: int


class Detector(Protocol):
    name: str
    task: str
    device: str
    names: dict[int, str]

    def detect(self, image_bgr: np.ndarray) -> list[RawDetection]: ...


class UltralyticsDetector:
    """YOLO via Ultralytics. Loaded once at startup and reused for every request."""

    task = "detect"

    def __init__(self, settings: Settings) -> None:
        self._settings = settings
        self._model = None
        # Ultralytics models are not safe to call from several threads at once.
        self._lock = threading.Lock()
        self.name = Path(settings.model_path).stem
        self.device = settings.device
        self.names: dict[int, str] = {}
        self._class_ids: list[int] | None = None

    def load(self) -> None:
        prepare_ultralytics()
        from ultralytics import YOLO  # heavy import, deferred on purpose

        model = YOLO(self._settings.model_path)
        if getattr(model, "task", "detect") != "detect":
            raise RuntimeError(f"{self._settings.model_path} is not a detection model")
        self.names = {int(k): str(v) for k, v in model.names.items()}

        wanted = self._settings.allowed_classes
        if wanted:
            by_name = {v: k for k, v in self.names.items()}
            unknown = [c for c in wanted if c not in by_name]
            if unknown:
                # Validate the configured labels against the real model, per the plan.
                raise RuntimeError(f"Model does not know these classes: {unknown}")
            self._class_ids = [by_name[c] for c in wanted]

        self._model = model
        # One warm-up pass so the first real request doesn't pay the setup cost.
        self.detect(np.zeros((self._settings.image_size, self._settings.image_size, 3), np.uint8))
        logger.info("Loaded %s on %s (%d classes)", self.name, self.device, len(self.names))

    def detect(self, image_bgr: np.ndarray) -> list[RawDetection]:
        if self._model is None:
            raise RuntimeError("Model is not loaded")
        s = self._settings
        with self._lock:
            results = self._model.predict(
                image_bgr,
                conf=s.confidence_threshold,
                iou=s.iou_threshold,
                imgsz=s.image_size,
                device=s.device,
                classes=self._class_ids,
                verbose=False,
            )
        boxes = results[0].boxes
        xyxy = boxes.xyxy.cpu().numpy()
        conf = boxes.conf.cpu().numpy()
        cls = boxes.cls.cpu().numpy()
        return [
            RawDetection(float(b[0]), float(b[1]), float(b[2]), float(b[3]), float(c), int(k))
            for b, c, k in zip(xyxy, conf, cls)
        ]
