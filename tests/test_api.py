from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from app.core.config import Settings
from app.main import create_app
from app.services.model_service import RawDetection

from .conftest import AUTH, FakeDetector, encode_image, make_settings, write_video


def post_image(client: TestClient, path: str, data: bytes, content_type: str = "image/jpeg", headers=AUTH):
    return client.post(path, files={"file": ("frame.jpg", data, content_type)}, headers=headers)


# --- health / models -------------------------------------------------------


def test_health_is_public_and_reports_the_model(client: TestClient) -> None:
    res = client.get("/health")
    assert res.status_code == 200
    assert res.json() == {"success": True, "data": {"status": "ok", "modelLoaded": True}}


def test_health_is_503_when_the_model_failed_to_load(settings: Settings, monkeypatch) -> None:
    def boom(self) -> None:
        raise RuntimeError("no weights")

    monkeypatch.setattr("app.main.UltralyticsDetector.load", boom)
    with TestClient(create_app(settings)) as client:
        assert client.get("/health").status_code == 503
        res = post_image(client, "/detect/image", encode_image())
        assert res.status_code == 503
        assert res.json()["error"]["code"] == "MODEL_UNAVAILABLE"


def test_models_describes_the_loaded_model(client: TestClient) -> None:
    res = client.get("/models", headers=AUTH)
    assert res.status_code == 200
    data = res.json()["data"]
    assert data["name"] == "fake-yolo"
    assert data["classes"]["67"] == "cell phone"
    assert data["returnedClasses"] == ["person", "cell phone", "laptop", "book"]


# --- authentication --------------------------------------------------------


@pytest.mark.parametrize(
    "headers",
    [
        {},
        {"Authorization": "Bearer wrong"},
        {"Authorization": "test-service-token"},  # no scheme
        {"Authorization": "Basic test-service-token"},
    ],
)
@pytest.mark.parametrize("method,path", [("get", "/models"), ("post", "/detect/image")])
def test_rejects_missing_or_wrong_token(client: TestClient, detector: FakeDetector, headers, method, path) -> None:
    if method == "get":
        res = client.get(path, headers=headers)
    else:
        res = post_image(client, path, encode_image(), headers=headers)
    assert res.status_code == 401
    assert res.json() == {
        "success": False,
        "error": {"code": "UNAUTHORIZED", "message": "Missing or invalid service token"},
    }
    assert detector.calls == []


def test_refuses_to_start_without_a_token() -> None:
    with pytest.raises(RuntimeError, match="AI_SERVICE_TOKEN"):
        create_app(make_settings(service_token=None), detector=FakeDetector())


def test_anonymous_mode_is_explicit_opt_in() -> None:
    app = create_app(make_settings(service_token=None, allow_anonymous=True), detector=FakeDetector())
    with TestClient(app) as client:
        assert client.get("/models").status_code == 200


# --- /detect/image ---------------------------------------------------------


def test_detects_and_returns_the_contract_shape(client: TestClient, detector: FakeDetector) -> None:
    detector.raw = [
        RawDetection(10, 20, 110, 220, 0.95, 0),
        RawDetection(200, 50, 260, 150, 0.81, 67),
        RawDetection(0, 0, 50, 50, 0.99, 2),  # car: filtered out
    ]
    res = post_image(client, "/detect/image", encode_image(320, 240))
    assert res.status_code == 200
    body = res.json()
    assert body["success"] is True
    data = body["data"]
    assert (data["imageWidth"], data["imageHeight"]) == (320, 240)
    assert isinstance(data["inferenceMs"], float)
    assert data["detections"] == [
        {
            "classId": 0,
            "className": "person",
            "confidence": 0.95,
            "boundingBox": {"x": 10, "y": 20, "width": 100, "height": 200},
        },
        {
            "classId": 67,
            "className": "cell phone",
            "confidence": 0.81,
            "boundingBox": {"x": 200, "y": 50, "width": 60, "height": 100},
        },
    ]
    assert detector.calls == [(240, 320)]


@pytest.mark.parametrize("ext,content_type", [(".png", "image/png"), (".webp", "image/webp")])
def test_accepts_png_and_webp(client: TestClient, ext: str, content_type: str) -> None:
    res = post_image(client, "/detect/image", encode_image(ext=ext), content_type)
    assert res.status_code == 200


def test_returns_an_empty_list_when_nothing_is_detected(client: TestClient) -> None:
    res = post_image(client, "/detect/image", encode_image())
    assert res.json()["data"]["detections"] == []


def test_rejects_an_unsupported_media_type(client: TestClient, detector: FakeDetector) -> None:
    res = post_image(client, "/detect/image", b"GIF89a....", "image/gif")
    assert res.status_code == 415
    assert res.json()["error"]["code"] == "UNSUPPORTED_MEDIA_TYPE"
    assert detector.calls == []


def test_rejects_bytes_that_are_not_an_image_whatever_the_declared_type(client: TestClient) -> None:
    res = post_image(client, "/detect/image", b"%PDF-1.4 not an image", "image/jpeg")
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "INVALID_IMAGE"


def test_rejects_an_empty_file(client: TestClient) -> None:
    res = post_image(client, "/detect/image", b"")
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "EMPTY_FILE"


def test_rejects_a_file_over_the_byte_limit(detector: FakeDetector) -> None:
    app = create_app(make_settings(max_image_bytes=2000), detector=detector)
    with TestClient(app) as client:
        res = post_image(client, "/detect/image", encode_image(640, 480, ".png"), "image/png")
    assert res.status_code == 413
    assert res.json()["error"]["code"] == "FILE_TOO_LARGE"
    assert detector.calls == []


def test_rejects_an_image_over_the_dimension_limit(detector: FakeDetector) -> None:
    app = create_app(make_settings(max_image_dimension=100), detector=detector)
    with TestClient(app) as client:
        res = post_image(client, "/detect/image", encode_image(320, 80))
    assert res.status_code == 413
    assert res.json()["error"]["code"] == "IMAGE_TOO_LARGE"


def test_requires_the_file_field(client: TestClient) -> None:
    res = client.post("/detect/image", headers=AUTH)
    assert res.status_code == 422
    assert res.json()["error"]["code"] == "VALIDATION_ERROR"


def test_rejects_an_oversized_request_before_parsing_it(client: TestClient) -> None:
    res = client.post(
        "/detect/image",
        headers={**AUTH, "content-length": str(500 * 1024 * 1024), "content-type": "multipart/form-data; boundary=x"},
        content=b"--x--",
    )
    assert res.status_code == 413
    assert res.json()["error"]["code"] == "REQUEST_TOO_LARGE"


# --- /detect/frame ---------------------------------------------------------


def test_frame_uses_the_same_contract(client: TestClient, detector: FakeDetector) -> None:
    detector.raw = [RawDetection(1, 2, 31, 42, 0.9, 63)]
    res = post_image(client, "/detect/frame", encode_image(160, 120))
    assert res.status_code == 200
    assert res.json()["data"]["detections"][0]["className"] == "laptop"


def test_frame_has_a_tighter_size_limit_than_image(detector: FakeDetector) -> None:
    app = create_app(make_settings(max_frame_bytes=2000, max_image_bytes=10_000_000), detector=detector)
    data = encode_image(640, 480, ".png")
    with TestClient(app) as client:
        assert post_image(client, "/detect/frame", data, "image/png").status_code == 413
        assert post_image(client, "/detect/image", data, "image/png").status_code == 200


# --- load control ----------------------------------------------------------


def test_refuses_new_inference_when_too_many_are_in_flight(client: TestClient, detector: FakeDetector) -> None:
    gate = client.app.state.gate
    gate._in_flight = client.app.state.settings.max_pending_inferences  # simulate a full queue
    res = post_image(client, "/detect/frame", encode_image())
    assert res.status_code == 503
    assert res.json()["error"]["code"] == "AI_BUSY"
    assert detector.calls == []
    gate._in_flight = 0
    assert post_image(client, "/detect/frame", encode_image()).status_code == 200


def test_the_gate_is_released_after_a_failed_inference(settings: Settings, detector: FakeDetector) -> None:
    def broken(_image):
        raise RuntimeError("model crashed")

    detector.detect = broken  # type: ignore[method-assign]
    # Return the 500 response instead of re-raising it into the test.
    with TestClient(create_app(settings, detector=detector), raise_server_exceptions=False) as client:
        res = post_image(client, "/detect/frame", encode_image())
        assert res.status_code == 500
        # No internals (exception text) leak to the caller.
        assert res.json()["error"] == {"code": "INTERNAL_ERROR", "message": "Internal error"}
        assert client.app.state.gate.in_flight == 0


# --- /detect/video ---------------------------------------------------------


def post_video(client: TestClient, data: bytes, params: str = "", content_type: str = "video/mp4"):
    return client.post(
        f"/detect/video{params}", files={"file": ("clip.mp4", data, content_type)}, headers=AUTH
    )


def test_video_samples_one_frame_per_interval(client: TestClient, detector: FakeDetector, tmp_path: Path) -> None:
    video = write_video(tmp_path / "clip.mp4", frames=30, fps=10.0)  # 3 seconds
    detector.raw = [RawDetection(5, 5, 55, 105, 0.9, 0)]

    res = post_video(client, video.read_bytes(), "?sampleIntervalMs=1000")

    assert res.status_code == 200, res.text
    data = res.json()["data"]
    assert (data["videoWidth"], data["videoHeight"]) == (160, 120)
    assert data["fps"] == 10.0
    assert data["sampledFrames"] == 3
    assert data["truncated"] is False
    assert [f["timestampMs"] for f in data["frames"]] == [0, 1000, 2000]
    assert [f["frameIndex"] for f in data["frames"]] == [0, 10, 20]
    assert data["frames"][0]["detections"][0]["className"] == "person"
    assert len(detector.calls) == 3


def test_video_stops_at_the_sample_cap(detector: FakeDetector, tmp_path: Path) -> None:
    video = write_video(tmp_path / "clip.mp4", frames=30, fps=10.0)
    app = create_app(make_settings(max_video_sampled_frames=2), detector=detector)
    with TestClient(app) as client:
        data = post_video(client, video.read_bytes(), "?sampleIntervalMs=500").json()["data"]
    assert data["sampledFrames"] == 2
    assert data["truncated"] is True


def test_video_rejects_a_too_short_interval(client: TestClient, tmp_path: Path) -> None:
    video = write_video(tmp_path / "clip.mp4", frames=5)
    assert post_video(client, video.read_bytes(), "?sampleIntervalMs=10").status_code == 422


def test_video_rejects_garbage_and_wrong_types(client: TestClient) -> None:
    res = post_video(client, b"definitely not a video")
    assert res.status_code == 400
    assert res.json()["error"]["code"] == "INVALID_VIDEO"
    assert post_video(client, b"x", content_type="image/jpeg").status_code == 415


def test_video_leaves_no_temporary_file_behind(client: TestClient, tmp_path: Path, monkeypatch) -> None:
    created: list[str] = []
    import tempfile

    real_mkstemp = tempfile.mkstemp

    def tracking_mkstemp(*args, **kwargs):
        fd, name = real_mkstemp(*args, **kwargs)
        created.append(name)
        return fd, name

    monkeypatch.setattr("app.api.detection.tempfile.mkstemp", tracking_mkstemp)
    video = write_video(tmp_path / "clip.mp4", frames=10)
    post_video(client, video.read_bytes())
    post_video(client, b"garbage")

    assert len(created) == 2
    assert all(not Path(name).exists() for name in created)
