from __future__ import annotations

from fastapi import APIRouter, Request
from fastapi.responses import JSONResponse

from app.schemas.detection import Envelope, HealthData

router = APIRouter(tags=["health"])


@router.get("/health", response_model=Envelope[HealthData])
async def health(request: Request) -> JSONResponse:
    """Unauthenticated on purpose (platform health checks). Reveals only whether the model is up."""
    loaded = request.app.state.detector is not None
    body = Envelope[HealthData](
        data=HealthData(status="ok" if loaded else "unavailable", model_loaded=loaded)
    )
    return JSONResponse(
        status_code=200 if loaded else 503, content=body.model_dump(by_alias=True)
    )
