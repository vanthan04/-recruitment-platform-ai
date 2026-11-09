# CPU inference benchmark

Measured in the Docker image (see Results). Natively on the development machine
(Windows 11) torch's native libraries are blocked by the device's Windows
Application Control policy:

```
OSError: [WinError 4551] An Application Control policy has blocked this file.
Error loading ".venv\Lib\site-packages\torch\lib\shm.dll" or one of its dependencies.
```

That is a security policy on the device and is deliberately not worked around.
The same policy blocks Next.js' native SWC binary in `recruitment-platform-fe`.
Consequences:

- The default test suite (fake detector, no torch) runs there: contract,
  validation, auth, limits, video sampling, load shedding.
- The real-model tests (`RUN_MODEL_TESTS=1 pytest -m model`) and this benchmark
  must run elsewhere: in the Docker image (Linux, unaffected by the Windows
  policy), on another machine, or on the target host.

## How to run

```bash
# inside the container
docker build -t recruitment-platform-ai .
docker run --rm --user root -e RUN_MODEL_TESTS=1 -v "$PWD/tests:/app/tests:ro" recruitment-platform-ai \
  sh -c "uv sync --frozen && uv run python -m pytest"
docker run --rm -v "$PWD/scripts:/app/scripts:ro" recruitment-platform-ai \
  uv run --no-sync python -m scripts.benchmark --sizes 320,480,640 --runs 30
```

Note: `.dockerignore` keeps `tests/` and `scripts/` out of the image, so they are
mounted as above. The test run adds the dev group (pytest) with `uv sync`, which
writes to the root-owned `.venv` — hence `--user root` for that throwaway
container only; the service itself runs as the unprivileged `app` user.
On Git Bash for Windows, prefix with `MSYS_NO_PATHCONV=1` and use `$(pwd -W)`.

Paste the benchmark output below, with the host it ran on. Choose
`AI_IMAGE_SIZE` and the frontend's frame rate (Phase 3) from these numbers: a
host needs roughly `concurrent interviews × frames per second × p95 latency`
CPU-seconds per second of capacity.

## Results

### 2026-11-09 — developer laptop, Docker Desktop (WSL2)

- Machine: x86_64, 8 logical CPUs given to the Docker VM, Linux 6.6 (WSL2)
- Python 3.13.18, torch 2.15.0+cpu (**1 thread**), ultralytics 8.4.208
- Model: `yolo11n.pt` on CPU, load 42 ms (weights baked into the image)
- Frame: 640x480 JPEG q70 (70 KB), decode p50 1.2 ms
- 30 runs per size after 3 warm-up runs

| imgsz | mean ms | p50 ms | p95 ms | max sustained fps (1000/p95) |
|---|---|---|---|---|
| 320 | 20.1 | 19.1 | 23.7 | 42.2 |
| 480 | 30.2 | 30.1 | 34.7 | 28.9 |
| 640 | 43.4 | 42.6 | 49.5 | 20.2 |

- Process memory (RSS) after benchmark: 423 MB
- Real-model tests (`RUN_MODEL_TESTS=1`) pass in the same image: the configured
  classes exist in the model, people are found in Ultralytics' `bus.jpg` and
  `zidane.jpg`, and every box lies inside the image.

**Reading it**

- Model time only; HTTP upload, multipart parsing and JSON add to this (to be
  measured end to end in Phase 3, browser → NestJS → this service).
- torch ran single-threaded in the container, and the service serializes
  inference per process, so one process uses about **one core**. At the
  default `AI_IMAGE_SIZE=640` that is ~20 frames/s per process: roughly 20
  concurrent interviews at 1 frame/s, or ~10 at 2 frames/s, with no headroom.
  `imgsz=320` doubles that at some cost to small-object accuracy (a phone far
  from the camera) — to be checked on real webcam footage before changing.
- More capacity = more processes/replicas (one model each, ~0.45 GB RAM), not more
  uvicorn workers in one container.
- A laptop core is typically faster than a shared cloud vCPU: re-run this
  benchmark on the chosen host before sizing it.
- Suggested Phase 3 starting point: 1 frame/s per candidate at 640, skip a frame
  while the previous one is still in flight, and drop on `503 AI_BUSY`.
