"""Service configuration, read once from environment variables (prefix ``AI_``)."""

from __future__ import annotations

from functools import lru_cache
from typing import Annotated

from pydantic import Field, SecretStr, field_validator
from pydantic_settings import BaseSettings, NoDecode, SettingsConfigDict

MB = 1024 * 1024


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="AI_", env_file=".env", extra="ignore")

    # --- Access ------------------------------------------------------------
    # Shared secret the NestJS API sends as `Authorization: Bearer <token>`.
    # The service refuses to start without it unless allow_anonymous is set
    # (local experiments only) — it must never be reachable unauthenticated.
    service_token: SecretStr | None = None
    allow_anonymous: bool = False

    # --- Model -------------------------------------------------------------
    model_path: str = "yolo11n.pt"
    device: str = "cpu"
    image_size: int = Field(640, ge=160, le=1920)
    confidence_threshold: float = Field(0.35, ge=0.0, le=1.0)
    iou_threshold: float = Field(0.45, ge=0.0, le=1.0)
    # COCO labels returned to callers; empty = every class the model knows.
    # Comma-separated in the environment: AI_ALLOWED_CLASSES="person,cell phone".
    allowed_classes: Annotated[list[str], NoDecode] = ["person", "cell phone", "laptop", "book"]

    # --- Input limits ------------------------------------------------------
    max_image_bytes: int = Field(5 * MB, gt=0)
    max_frame_bytes: int = Field(1 * MB, gt=0)
    max_image_dimension: int = Field(4096, ge=64)
    max_video_bytes: int = Field(50 * MB, gt=0)
    max_video_sampled_frames: int = Field(300, ge=1)
    default_video_sample_interval_ms: int = Field(1000, ge=100)

    # --- Load control ------------------------------------------------------
    # Inference requests allowed in flight (running + waiting) before new ones
    # are refused with 503, so a slow model can't build an unbounded queue.
    max_pending_inferences: int = Field(4, ge=1)

    @field_validator("allowed_classes", mode="before")
    @classmethod
    def _split_classes(cls, value: object) -> object:
        if isinstance(value, str):
            return [part.strip() for part in value.split(",") if part.strip()]
        return value

    def assert_secure(self) -> None:
        """Fail fast on a configuration that would expose an open inference endpoint."""
        if not self.allow_anonymous and (
            self.service_token is None or not self.service_token.get_secret_value()
        ):
            raise RuntimeError(
                "AI_SERVICE_TOKEN is not set. Set it (and the same value in the NestJS API), "
                "or set AI_ALLOW_ANONYMOUS=true for local experiments only."
            )


@lru_cache
def get_settings() -> Settings:
    return Settings()
