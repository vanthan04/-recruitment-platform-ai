"""Real YOLO inference on the sample images Ultralytics ships with the package.

Opt-in: RUN_MODEL_TESTS=1 (needs ultralytics + torch, downloads the weights once).
"""

from __future__ import annotations

import os
from pathlib import Path

import cv2
import pytest
from fastapi.testclient import TestClient

from app.main import create_app

from .conftest import AUTH, make_settings

pytestmark = [
    pytest.mark.model,
    pytest.mark.skipif(os.environ.get("RUN_MODEL_TESTS") != "1", reason="set RUN_MODEL_TESTS=1"),
]


def _asset(name: str) -> Path:
    from app.services.model_service import prepare_ultralytics

    prepare_ultralytics()
    import ultralytics

    return Path(ultralytics.__file__).parent / "assets" / name


@pytest.fixture(scope="module")
def client():
    settings = make_settings(model_path=os.environ.get("AI_MODEL_PATH", "yolo11n.pt"))
    with TestClient(create_app(settings)) as c:
        assert c.get("/health").status_code == 200, "model failed to load"
        yield c


def _detect(client: TestClient, path: Path) -> dict:
    res = client.post(
        "/detect/image", files={"file": (path.name, path.read_bytes(), "image/jpeg")}, headers=AUTH
    )
    assert res.status_code == 200, res.text
    return res.json()["data"]


def test_the_configured_classes_exist_in_the_real_model(client: TestClient) -> None:
    data = client.get("/models", headers=AUTH).json()["data"]
    known = set(data["classes"].values())
    assert {"person", "cell phone", "laptop", "book"} <= known


@pytest.mark.parametrize("asset,min_people", [("bus.jpg", 3), ("zidane.jpg", 2)])
def test_finds_people_in_the_sample_images(client: TestClient, asset: str, min_people: int) -> None:
    path = _asset(asset)
    data = _detect(client, path)
    height, width = cv2.imread(str(path)).shape[:2]
    assert (data["imageWidth"], data["imageHeight"]) == (width, height)

    people = [d for d in data["detections"] if d["className"] == "person"]
    assert len(people) >= min_people
    for d in data["detections"]:
        box = d["boundingBox"]
        assert d["className"] in {"person", "cell phone", "laptop", "book"}  # 'bus' etc. filtered
        assert 0 <= box["x"] and box["x"] + box["width"] <= width
        assert 0 <= box["y"] and box["y"] + box["height"] <= height
