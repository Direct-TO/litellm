"""Lossless gateway video inputs. Provider mappers own their documented capabilities."""

import re
from collections.abc import Mapping
from typing import Final, Literal, cast
from urllib.parse import urlsplit

from pydantic import BaseModel, ConfigDict, ValidationError

from litellm.exceptions import UnsupportedParamsError

VIDEO_CONTRACT_FIELDS: Final = frozenset({"resolution", "aspect_ratio", "references"})
VIDEO_RESOLUTIONS: Final = frozenset({"480p", "720p", "1080p", "2K", "4K"})
VIDEO_NATIVE_OVERRIDE_FIELDS: Final = VIDEO_CONTRACT_FIELDS | frozenset(
    {
        "size",
        "width",
        "height",
        "input_reference",
        "ratio",
        "mode",
        "image",
        "images",
        "image_urls",
        "image_with_roles",
        "reference_images",
        "video_with_roles",
        "video_list",
        "audio_with_roles",
        "metadata",
        "parameters",
        "video_operation",
        "seconds",
        "duration",
        "generation_type",
        "action",
        "first_frame",
        "last_frame",
    }
)


class VideoReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    type: Literal["image", "video", "audio"]
    url: str
    role: Literal["reference", "first_frame", "last_frame"] = "reference"


def validate_video_contract(params: Mapping[str, object]) -> list[VideoReference]:
    if not any(params.get(key) is not None for key in VIDEO_CONTRACT_FIELDS):
        return []
    if any(params.get(key) is not None for key in ("size", "width", "height", "input_reference")):
        raise ValueError(
            "Video resolution/aspect_ratio/references cannot be combined with pixel size or input_reference"
        )
    resolution = params.get("resolution")
    if resolution is not None and (not isinstance(resolution, str) or resolution not in VIDEO_RESOLUTIONS):
        raise ValueError("Video resolution must be 480p, 720p, 1080p, 2K or 4K; omit it for automatic selection")
    ratio = params.get("aspect_ratio")
    if ratio is not None:
        if not isinstance(ratio, str) or not re.fullmatch(r"[1-9]\d*:[1-9]\d*", ratio):
            raise ValueError(
                "Video aspect_ratio must be a positive ratio such as 16:9; omit it for automatic selection"
            )
    seconds = params.get("seconds")
    if seconds is not None and not re.fullmatch(r"-1|[1-9]\d*", str(seconds)):
        raise ValueError("Video seconds must be a positive integer or -1 for automatic duration")
    raw = params.get("references")
    if raw is None:
        return []
    if not isinstance(raw, list):
        raise ValueError("Video references must be an array")
    try:
        references = [VideoReference.model_validate(item) for item in cast(list[object], raw)]
    except ValidationError as exc:
        raise ValueError(
            "Video references require type, url and an optional reference/first_frame/last_frame role"
        ) from exc
    for reference in references:
        parsed = urlsplit(reference.url)
        if parsed.scheme not in ("http", "https") or not parsed.netloc or parsed.username or parsed.password:
            raise ValueError("Video reference URLs must be absolute HTTP(S) URLs without credentials")
        if reference.role != "reference" and reference.type != "image":
            raise ValueError("Video first_frame and last_frame references must be images")
    first = [ref for ref in references if ref.role == "first_frame"]
    last = [ref for ref in references if ref.role == "last_frame"]
    if len(first) > 1 or len(last) > 1 or (last and not first):
        raise ValueError(
            "Video references allow one first frame and one last frame; a last frame requires a first frame"
        )
    if first and last and first[0].url == last[0].url:
        raise ValueError("Video first and last frame must use different images")
    return references


def require_video_contract_support(params: Mapping[str, object], supported: list[str], model: str) -> None:
    try:
        validate_video_contract(params)
    except ValueError as exc:
        raise UnsupportedParamsError(message=str(exc), model=model, llm_provider="") from exc
    unsupported = sorted(key for key in VIDEO_CONTRACT_FIELDS if params.get(key) is not None and key not in supported)
    if unsupported:
        raise UnsupportedParamsError(
            message=f"Video model={model!r} has no documented mapping for {', '.join(unsupported)}",
            model=model,
            llm_provider="",
        )
