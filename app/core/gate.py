"""Bounded admission for inference: refuse instead of queueing without limit."""

from __future__ import annotations

from types import TracebackType

from app.core.errors import ApiError


class InferenceGate:
    """At most ``limit`` inference requests in flight (running or waiting for the
    model lock). One more gets 503 straight away — the caller (NestJS / the
    browser's frame loop) drops that frame instead of piling up latency."""

    def __init__(self, limit: int) -> None:
        self._limit = limit
        self._in_flight = 0

    @property
    def in_flight(self) -> int:
        return self._in_flight

    async def __aenter__(self) -> "InferenceGate":
        # Single event loop: the check-and-increment cannot interleave.
        if self._in_flight >= self._limit:
            raise ApiError(503, "AI_BUSY", "Too many inference requests in progress, try again shortly")
        self._in_flight += 1
        return self

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc: BaseException | None,
        tb: TracebackType | None,
    ) -> None:
        self._in_flight -= 1
