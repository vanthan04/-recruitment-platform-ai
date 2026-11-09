"""Server-to-server authentication: a shared bearer token from the NestJS API."""

from __future__ import annotations

import hmac

from fastapi import Request

from app.core.errors import ApiError


def require_service_token(request: Request) -> None:
    settings = request.app.state.settings
    if settings.allow_anonymous:
        return

    header = request.headers.get("authorization", "")
    scheme, _, token = header.partition(" ")
    expected = settings.service_token.get_secret_value() if settings.service_token else ""
    # Constant-time comparison; an empty expected token never matches.
    if (
        scheme.lower() != "bearer"
        or not expected
        or not hmac.compare_digest(token.encode(), expected.encode())
    ):
        raise ApiError(401, "UNAUTHORIZED", "Missing or invalid service token")
