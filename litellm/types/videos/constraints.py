"""Public video parameter combinations. None denotes omitting a field, -1 an explicit smart duration."""

from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator
from typing_extensions import NotRequired, TypedDict


class VideoVariant(TypedDict):
    aspect_ratio: list[str | None]
    resolution: list[str | None]
    seconds: list[int | None]
    frames: NotRequired[bool]


class VideoGenerationConstraints(TypedDict):
    version: Literal[1]
    revision: str
    variants: list[VideoVariant]


class VideoGenerationLimits(BaseModel):
    """Optional deployment-local restrictions persisted in model_info."""

    model_config = ConfigDict(extra="forbid", strict=True)
    aspect_ratios: list[str] | None = Field(default=None, min_length=1)
    resolutions: list[str] | None = Field(default=None, min_length=1)
    min_seconds: int | None = Field(default=None, ge=1)
    max_seconds: int | None = Field(default=None, ge=1)

    @model_validator(mode="after")
    def check_range(self):
        if self.min_seconds and self.max_seconds and self.min_seconds > self.max_seconds:
            raise ValueError("min_seconds must not exceed max_seconds")
        return self


def video_variant(ratios: tuple[str, ...], resolutions: tuple[str, ...], seconds: tuple[int, ...]) -> VideoVariant:
    return {"aspect_ratio": [None, *ratios], "resolution": [None, *resolutions], "seconds": [None, *seconds]}
