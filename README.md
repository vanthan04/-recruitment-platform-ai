# recruitment-platform-ai

Object-detection service for the IT recruitment platform's interview monitoring
(Phase 2 of the YOLO interview-monitoring plan). Python, FastAPI, Ultralytics YOLO,
OpenCV.

It receives an image, a webcam frame or a video file and returns **observations**:
which objects (a person, a cell phone, a laptop, a book) appear where, with what
confidence. It never decides anything about a candidate — whether something is a
policy violation, whether a candidate is honest, or whether an application should
move on. Those calls belong to people, and the event policy belongs to the NestJS
API (`recruitment-platform-be`).

Sibling repos: `recruitment-platform-be` (NestJS API, the only caller of this
service) and `recruitment-platform-fe` (Next.js). Browsers never call this service
directly.

## API

All responses are JSON: `{"success": true, "data": ...}` or
`{"success": false, "error": {"code", "message"}}`. Every endpoint except
`/health` requires `Authorization: Bearer <AI_SERVICE_TOKEN>`. Interactive docs:
`/docs`.

| Method | Path | Body | Purpose |
|---|---|---|---|
| GET | `/health` | — | `200 {status: "ok", modelLoaded: true}`, or `503 {status: "unavailable"}` if the model did not load. Public (for health checks). |
| GET | `/models` | — | Model name, task, device, image size, thresholds, every class the model knows (`classes`, by id) and the subset this service returns (`returnedClasses`). |
| POST | `/detect/image` | multipart `file` (JPEG/PNG/WebP, ≤ 5 MB) | Detect objects in an image. |
| POST | `/detect/frame` | multipart `file` (JPEG/PNG/WebP, ≤ 1 MB) | One sampled webcam frame. Same contract as `/detect/image`, tighter size limit. |
| POST | `/detect/video?sampleIntervalMs=1000` | multipart `file` (MP4/WebM/MOV/AVI, ≤ 50 MB) | Offline experiments: detect on one frame every `sampleIntervalMs` (100–60000 ms), at most 300 sampled frames (`truncated: true` beyond that). |

### Detection contract

```json
{
  "success": true,
  "data": {
    "imageWidth": 1280,
    "imageHeight": 720,
    "inferenceMs": 41.7,
    "detections": [
      {
        "classId": 0,
        "className": "person",
        "confidence": 0.95,
        "boundingBox": { "x": 120, "y": 80, "width": 350, "height": 500 }
      }
    ]
  }
}
```

- `boundingBox` is in **whole pixels of the original image** you sent (not the
  model's resized input). `x`/`y` is the **top-left corner**; `width`/`height`
  extend right and down. Boxes are clamped inside the image; empty boxes are dropped.
- Only `returnedClasses` are reported (default: `person`, `cell phone`, `laptop`,
  `book`), only at or above `confidenceThreshold`, most confident first.
- `inferenceMs` is model time only, not upload/decoding.

To draw boxes over a `<video>` rendered with `object-fit: cover`, scale by
`max(renderedW / imageWidth, renderedH / imageHeight)` and offset by half the
cropped overflow — that's Phase 3's job on the frontend.

### Errors

| Status | `error.code` | When |
|---|---|---|
| 400 | `EMPTY_FILE`, `INVALID_IMAGE`, `INVALID_VIDEO` | Empty upload, or bytes that don't decode (checked by decoding, not by the declared type). |
| 401 | `UNAUTHORIZED` | Missing or wrong service token. |
| 413 | `FILE_TOO_LARGE`, `IMAGE_TOO_LARGE`, `VIDEO_TOO_LARGE`, `REQUEST_TOO_LARGE` | Over the byte limit, or a side over 4096 px. |
| 415 | `UNSUPPORTED_MEDIA_TYPE` | Not one of the accepted types. |
| 422 | `VALIDATION_ERROR` | Missing `file`, bad query parameter. |
| 503 | `MODEL_UNAVAILABLE` | The model failed to load. |
| 503 | `AI_BUSY` | More than `AI_MAX_PENDING_INFERENCES` requests in flight. The caller should drop that frame, not retry in a loop. |
| 500 | `INTERNAL_ERROR` | Anything else. No internals in the message. |

## Configuration

Environment variables (or a local `.env`, never committed):

| Variable | Default | |
|---|---|---|
| `AI_SERVICE_TOKEN` | — | **Required.** Shared secret; the NestJS API sends it as a bearer token. The service refuses to start without it. |
| `AI_ALLOW_ANONYMOUS` | `false` | Local experiments only: disables the token check. |
| `AI_MODEL_PATH` | `yolo11n.pt` | Weights file. Known Ultralytics names are downloaded on first use. |
| `AI_DEVICE` | `cpu` | e.g. `cuda:0` on a GPU host. |
| `AI_IMAGE_SIZE` | `640` | Model input size. See the benchmark before raising it. |
| `AI_CONFIDENCE_THRESHOLD` | `0.35` | |
| `AI_IOU_THRESHOLD` | `0.45` | Non-max suppression. |
| `AI_ALLOWED_CLASSES` | `person,cell phone,laptop,book` | Comma-separated COCO labels; empty = all. Checked against the loaded model at startup. |
| `AI_MAX_IMAGE_BYTES` / `AI_MAX_FRAME_BYTES` / `AI_MAX_VIDEO_BYTES` | 5 MB / 1 MB / 50 MB | |
| `AI_MAX_IMAGE_DIMENSION` | `4096` | Longest allowed side, px. |
| `AI_MAX_VIDEO_SAMPLED_FRAMES` | `300` | |
| `AI_DEFAULT_VIDEO_SAMPLE_INTERVAL_MS` | `1000` | |
| `AI_MAX_PENDING_INFERENCES` | `4` | Requests in flight before `503 AI_BUSY`. |

## Run locally

Dependencies are managed with [uv](https://docs.astral.sh/uv/) (`pyproject.toml`
+ `uv.lock`, Python from `.python-version`).

```bash
uv sync                      # creates .venv with runtime + dev dependencies, exactly as locked
AI_SERVICE_TOKEN=dev-token uv run python -m uvicorn --factory app.main:create_app --port 8000
```

PowerShell: `$env:AI_SERVICE_TOKEN = "dev-token"` first, then
`uv run python -m uvicorn --factory app.main:create_app --port 8000`.

Commands go through `python -m` rather than the `.exe` launchers uv generates
in `.venv\Scripts`, because a Windows Application Control policy can block
those launchers (and torch's DLLs — see Tests).

Adding a dependency: `uv add <package>` (or `uv add --group dev <package>`),
which updates `pyproject.toml` and `uv.lock` together — commit both.

```bash
curl -H "Authorization: Bearer dev-token" -F "file=@photo.jpg;type=image/jpeg" http://localhost:8000/detect/image
```

## Tests

```bash
uv run python -m pytest
```

The default suite uses a fake detector — no torch or weights needed — and covers
the contract, normalization, auth, input validation, size limits, video sampling
and load shedding. Real-model tests run on the sample images bundled with
Ultralytics:

```bash
RUN_MODEL_TESTS=1 uv run python -m pytest -m model
```

On a Windows machine with an Application Control policy, torch's DLLs may be
blocked (`WinError 4551`); run the model tests in Docker instead — see
[docs/benchmark.md](docs/benchmark.md).

## Benchmark

```bash
uv run python -m scripts.benchmark --sizes 320,480,640 --runs 30
```

Results are in [docs/benchmark.md](docs/benchmark.md). Benchmark on the target
host before choosing it: one CPU instance will not keep up with many concurrent
interviews at 1–3 frames per second.

## Docker

```bash
docker build -t recruitment-platform-ai .
docker run -p 8000:8000 -e AI_SERVICE_TOKEN=... recruitment-platform-ai
```

Installs exactly `uv.lock` (`uv sync --frozen --no-dev`, CPU-only torch on Linux),
weights baked into the image at build time, non-root user, one
worker (the model is loaded per process; scale with replicas, not workers).

## Privacy and data handling

- Uploaded images and videos are processed in memory (videos via a temporary file
  deleted right after) and **never stored**.
- Request bodies and tokens are never logged; error responses never echo input.
- No face recognition, identity matching, emotion or "honesty" inference — only
  generic object detection.

## Licensing — check before any commercial use

Ultralytics (the library) and the YOLO11 weights are licensed under **AGPL-3.0**.
That is fine for this personal project, but running it as a network service in a
commercial product generally requires either publishing the full source of the
service under AGPL-3.0 or buying an Ultralytics Enterprise licence. Decide this
before using it commercially (the implementation plan lists model licensing as a
risk to validate).
