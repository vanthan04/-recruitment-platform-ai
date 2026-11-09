from app.services.detection_service import normalize_detections
from app.services.model_service import RawDetection

from .conftest import COCO_NAMES

ALL = ["person", "cell phone", "laptop", "book"]


def test_converts_corners_to_top_left_and_size_in_original_pixels() -> None:
    out = normalize_detections(
        [RawDetection(120.4, 80.6, 470.2, 580.9, 0.951234, 0)], COCO_NAMES, 1280, 720, ALL, 0.3
    )
    assert len(out) == 1
    d = out[0]
    assert (d.class_id, d.class_name, d.confidence) == (0, "person", 0.9512)
    assert d.bounding_box.model_dump() == {"x": 120, "y": 81, "width": 350, "height": 500}


def test_clamps_boxes_that_spill_outside_the_image() -> None:
    out = normalize_detections(
        [RawDetection(-15, -3, 700, 500, 0.9, 0)], COCO_NAMES, 640, 480, ALL, 0.3
    )
    assert out[0].bounding_box.model_dump() == {"x": 0, "y": 0, "width": 640, "height": 480}


def test_drops_boxes_with_no_area_after_clamping() -> None:
    raw = [
        RawDetection(10, 10, 10, 50, 0.9, 0),  # zero width
        RawDetection(700, 10, 800, 50, 0.9, 0),  # entirely right of the image
    ]
    assert normalize_detections(raw, COCO_NAMES, 640, 480, ALL, 0.3) == []


def test_filters_classes_not_returned_and_unknown_ids() -> None:
    raw = [
        RawDetection(0, 0, 10, 10, 0.9, 2),  # car — known but not wanted
        RawDetection(0, 0, 10, 10, 0.9, 999),  # unknown id
        RawDetection(0, 0, 10, 10, 0.9, 67),  # cell phone
    ]
    out = normalize_detections(raw, COCO_NAMES, 100, 100, ALL, 0.3)
    assert [d.class_name for d in out] == ["cell phone"]


def test_empty_allowed_list_returns_every_known_class() -> None:
    raw = [RawDetection(0, 0, 10, 10, 0.9, 2)]
    assert [d.class_name for d in normalize_detections(raw, COCO_NAMES, 100, 100, [], 0.3)] == ["car"]


def test_applies_the_confidence_threshold_and_rejects_nan() -> None:
    raw = [
        RawDetection(0, 0, 10, 10, 0.29, 0),
        RawDetection(0, 0, 10, 10, float("nan"), 0),
        RawDetection(0, 0, 10, 10, 0.31, 0),
    ]
    assert [d.confidence for d in normalize_detections(raw, COCO_NAMES, 100, 100, ALL, 0.3)] == [0.31]


def test_most_confident_first() -> None:
    raw = [
        RawDetection(0, 0, 10, 10, 0.5, 0),
        RawDetection(0, 0, 10, 10, 0.9, 67),
        RawDetection(0, 0, 10, 10, 0.7, 63),
    ]
    out = normalize_detections(raw, COCO_NAMES, 100, 100, ALL, 0.3)
    assert [d.confidence for d in out] == [0.9, 0.7, 0.5]
