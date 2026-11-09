# CPU inference benchmark

**Status: not measured yet.**

On the development machine used so far (Windows 11), torch's native libraries
are blocked by the device's Windows Application Control policy:

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
docker run --rm -e RUN_MODEL_TESTS=1 -v "$PWD/tests:/app/tests" recruitment-platform-ai \
  sh -c "uv sync --frozen && uv run python -m pytest -m model"
docker run --rm -v "$PWD/scripts:/app/scripts" recruitment-platform-ai \
  uv run python -m scripts.benchmark --sizes 320,480,640 --runs 30
```

Note: `.dockerignore` keeps `tests/` and `scripts/` out of the image, so they are
mounted as above; `uv sync --frozen` adds the dev group (pytest) inside the container.

Paste the benchmark output below, with the host it ran on. Choose
`AI_IMAGE_SIZE` and the frontend's frame rate (Phase 3) from these numbers: a
host needs roughly `concurrent interviews × frames per second × p95 latency`
CPU-seconds per second of capacity.

## Results

_None yet._
