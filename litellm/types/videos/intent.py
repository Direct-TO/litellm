"""Server-owned configuration and structured decisions for reference-video intent."""

from collections.abc import Mapping
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field


class VideoIntentSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")

    classifier_model: str = Field(min_length=1)
    models: list[str] = Field(default_factory=lambda: ["seedance-2-5"], min_length=1)
    timeout: float = Field(default=20, gt=0, le=120)


class VideoIntentDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True)

    intent: Literal["generate", "edit", "extend", "unclear", "unsupported"]
    source_video_index: int | None = Field(ge=1)
    duration_kind: Literal["unspecified", "additional", "total"]
    duration_seconds: int | None = Field(ge=1)


VIDEO_INTENT_METADATA_KEY = "seedance_video_intent"


def get_video_intent_metadata(kwargs: Mapping[str, Any] | None) -> dict[str, Any] | None:
    """Only creation calls may use this metadata to constrain routing/retries."""
    if not kwargs or kwargs.get("_router_call_type") != "avideo_generation":
        return None
    metadata = kwargs.get("metadata")
    if not isinstance(metadata, dict):
        return None
    decision = metadata.get(VIDEO_INTENT_METADATA_KEY)
    if isinstance(decision, dict) and decision.get("intent") in ("generate", "edit", "extend"):
        return decision
    return None
