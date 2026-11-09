"""CPU inference benchmark — run before choosing hosting resources.

    python -m scripts.benchmark --sizes 320,480,640 --runs 30

Uses a webcam-like 640x480 frame cut from Ultralytics' bundled sample image, so
it needs no extra download besides the model weights. Prints a markdown table.
"""

from __future__ import annotations

import argparse
import os
import platform
import statistics
import time
from pathlib import Path

import cv2


def percentile(values: list[float], pct: float) -> float:
    ordered = sorted(values)
    index = min(len(ordered) - 1, max(0, round(pct / 100 * (len(ordered) - 1))))
    return ordered[index]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=os.environ.get("AI_MODEL_PATH", "yolo11n.pt"))
    parser.add_argument("--sizes", default="320,480,640")
    parser.add_argument("--runs", type=int, default=30)
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    from app.services.model_service import prepare_ultralytics

    prepare_ultralytics()
    import psutil
    import torch
    import ultralytics
    from ultralytics import YOLO

    asset = Path(ultralytics.__file__).parent / "assets" / "bus.jpg"
    frame = cv2.resize(cv2.imread(str(asset)), (640, 480))
    ok, jpeg = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
    assert ok

    started = time.perf_counter()
    model = YOLO(args.model)
    load_ms = (time.perf_counter() - started) * 1000

    decode_ms = []
    for _ in range(args.runs):
        t = time.perf_counter()
        cv2.imdecode(jpeg, cv2.IMREAD_COLOR)
        decode_ms.append((time.perf_counter() - t) * 1000)

    print(f"- Machine: {platform.processor() or platform.machine()} ({os.cpu_count()} logical CPUs), {platform.system()} {platform.release()}")
    print(f"- Python {platform.python_version()}, torch {torch.__version__} ({torch.get_num_threads()} threads), ultralytics {ultralytics.__version__}")
    print(f"- Model: {args.model} on {args.device}, load {load_ms:.0f} ms")
    print(f"- Frame: 640x480 JPEG q70 ({len(jpeg) / 1024:.0f} KB), decode p50 {statistics.median(decode_ms):.1f} ms")
    print(f"- Runs per size: {args.runs} (after 3 warm-up runs)\n")
    print("| imgsz | mean ms | p50 ms | p95 ms | max sustained fps (1000/p95) |")
    print("|---|---|---|---|---|")
    for size in [int(s) for s in args.sizes.split(",")]:
        for _ in range(3):
            model.predict(frame, imgsz=size, conf=0.35, device=args.device, verbose=False)
        times = []
        for _ in range(args.runs):
            t = time.perf_counter()
            model.predict(frame, imgsz=size, conf=0.35, device=args.device, verbose=False)
            times.append((time.perf_counter() - t) * 1000)
        p95 = percentile(times, 95)
        print(
            f"| {size} | {statistics.mean(times):.1f} | {statistics.median(times):.1f} "
            f"| {p95:.1f} | {1000 / p95:.1f} |"
        )

    rss = psutil.Process().memory_info().rss / (1024 * 1024)
    print(f"\n- Process memory (RSS) after benchmark: {rss:.0f} MB")


if __name__ == "__main__":
    main()
