"""Canonical video mapping, bound to the model contracts linked in videos-contract.md."""

from collections.abc import Mapping
from typing import Final

from litellm.exceptions import UnsupportedParamsError
from litellm.videos.contract import VideoReference, validate_video_contract

_RESOLUTIONS: Final = {
    "gemini-omni-flash": ("720p", "1080p"),
    "gemini-omni-flash-preview-official": ("720p",),
    "grok-video-1.0": ("480p", "720p"),
    "grok-video-1.5": ("480p", "720p"),
    "seedance-2": ("480p", "720p", "1080p", "4K"),
    "seedance-2-fast": ("480p", "720p"),
    "seedance-2-mini": ("480p", "720p"),
    "seedance-2-5": ("480p", "720p", "1080p"),
    "wan3.0-video": ("480p", "720p", "1080p"),
    "MiniMax-H3": ("2K",),
    "happyhorse-1.1": ("720p", "1080p"),
    "kling-v3": ("720p", "1080p"),
    "kling-v3-omni": ("720p", "1080p"),
    "kling-video-o1": ("720p", "1080p"),
    "veo3.1-fast": ("720p", "1080p", "4K"),
    "veo3.1-quality": ("720p", "1080p", "4K"),
    "veo3.1-lite": ("720p", "1080p", "4K"),
    "Veo3.1-fast-official": ("720p", "1080p", "4K"),
    "Veo3.1-quality-official": ("720p", "1080p", "4K"),
}
GATEWAY_VIDEO_MODELS: Final = frozenset(_RESOLUTIONS)
_GEMINI_MODELS: Final = frozenset({"gemini-omni-flash", "gemini-omni-flash-preview-official"})
_GROK_MODELS: Final = frozenset({"grok-video-1.0", "grok-video-1.5"})
_DURATIONS: Final = {
    "gemini-omni-flash": (4, 6, 10),
    "gemini-omni-flash-preview-official": tuple(range(1, 11)),
    "grok-video-1.0": tuple(range(1, 16)),
    "grok-video-1.5": tuple(range(1, 16)),
    "seedance-2": (-1, *range(4, 16)),
    "seedance-2-fast": (-1, *range(4, 16)),
    "seedance-2-mini": tuple(range(4, 16)),
    "seedance-2-5": (-1, *range(4, 31)),
    "wan3.0-video": tuple(range(2, 31)),
    "happyhorse-1.1": tuple(range(3, 16)),
    "kling-v3": tuple(range(3, 16)),
}
_NATIVE_INPUTS: Final = frozenset(
    {
        "image",
        "images",
        "image_urls",
        "image_with_roles",
        "reference_images",
        "video_with_roles",
        "video_list",
        "url",
        "audio_with_roles",
        "metadata",
        "parameters",
        "video_operation",
        "action",
        "duration",
    }
)


def _reject(model: str, message: str) -> None:
    raise UnsupportedParamsError(message=f"ToAPIs model={model!r}: {message}", model=model, llm_provider="toapis")


def map_gateway_video(model: str, params: Mapping[str, object], size_field: str) -> dict[str, object]:
    if model not in GATEWAY_VIDEO_MODELS:
        _reject(model, "no documented mapping for canonical video inputs")
    references = validate_video_contract(params)
    if any(params.get(key) is not None for key in _NATIVE_INPUTS):
        _reject(model, "canonical video inputs cannot be combined with native reference/metadata overrides")
    result = {
        key: value
        for key, value in params.items()
        if key not in {"resolution", "aspect_ratio", "references", "operation", "seconds", "size", "extra_body"}
    }
    operation = params.get("operation") or "generate"
    supported_operations = (
        ("generate", "edit", "extend")
        if model == "seedance-2-5"
        else ("generate", "edit")
        if model in {"happyhorse-1.1", "gemini-omni-flash-preview-official"}
        else ("generate",)
    )
    if operation not in supported_operations:
        _reject(model, f"operation={operation!r} is not supported; allowed={supported_operations}")
    if operation != "generate" and not any(ref.type == "video" for ref in references):
        _reject(model, f"operation={operation!r} requires at least one video reference")
    if model == "seedance-2-5":
        result["video_operation"] = operation
        if operation in ("edit", "extend"):
            if params.get("aspect_ratio") is not None:
                _reject(model, f"operation={operation!r} requires automatic aspect_ratio; omit aspect_ratio")
            if operation == "edit" and params.get("seconds") not in (None, "-1", -1):
                _reject(model, "operation='edit' requires automatic seconds (-1); omit seconds or use -1")
            result["aspect_ratio"] = "adaptive"
            result["duration"] = -1
    seconds = params.get("seconds")
    if seconds is not None:
        duration = int(str(seconds))
        if model in _DURATIONS and duration not in _DURATIONS[model]:
            _reject(model, f"seconds={seconds!r} is not supported; allowed={_DURATIONS[model]}")
        result["duration"] = duration
    ratio = params.get("aspect_ratio")
    if ratio is not None:
        known_ratios = {"16:9", "9:16", "1:1", "4:3", "3:4"}
        if model in {"seedance-2", "seedance-2-fast", "seedance-2-mini", "seedance-2-5", "MiniMax-H3"}:
            known_ratios.add("21:9")
            if ratio not in known_ratios:
                _reject(model, f"aspect_ratio={ratio!r} is not supported")
        elif model in {"wan3.0-video", "happyhorse-1.1"} and ratio not in known_ratios:
            _reject(model, f"aspect_ratio={ratio!r} is not supported")
        elif model in _GEMINI_MODELS and ratio not in {"16:9", "9:16"}:
            _reject(model, "aspect_ratio must be 16:9 or 9:16")
        elif model in _GROK_MODELS and ratio not in {"16:9", "9:16", "1:1", "3:2", "2:3"}:
            _reject(model, "aspect_ratio must be 16:9, 9:16, 1:1, 3:2 or 2:3")
        result[size_field] = ratio
    resolution = params.get("resolution")
    if resolution is not None:
        if resolution not in _RESOLUTIONS.get(model, ()):
            _reject(model, f"resolution={resolution!r} has no supported mapping; allowed={_RESOLUTIONS.get(model, ())}")
        if model.startswith("kling-"):
            result["mode"] = "std" if resolution == "720p" else "pro"
        elif model.startswith("veo3.1-"):
            result["metadata"] = {"resolution": str(resolution).lower()}
        elif model == "happyhorse-1.1":
            result["resolution"] = str(resolution).upper()
        else:
            result["resolution"] = "4k" if resolution == "4K" else resolution
    if model == "gemini-omni-flash" and resolution == "1080p" and ratio not in (None, "16:9"):
        _reject(model, "1080p only supports aspect_ratio=16:9")
    if model in _GEMINI_MODELS | _GROK_MODELS:
        _map_gemini_grok(model, result, references, params)
        return result
    if not references:
        return result

    images = [ref for ref in references if ref.type == "image"]
    videos = [ref for ref in references if ref.type == "video"]
    audios = [ref for ref in references if ref.type == "audio"]
    frames = [ref for ref in images if ref.role != "reference"]
    regular = [ref for ref in references if ref.role == "reference"]
    if frames and regular and model != "kling-v3":
        _reject(model, "first/last frames cannot be mixed with ordinary image/video/audio references")
    if params.get("tools"):
        _reject(model, "tools cannot be combined with media references")

    if model in {"seedance-2", "seedance-2-fast", "seedance-2-mini", "seedance-2-5", "MiniMax-H3", "wan3.0-video"}:
        _map_multimodal(model, result, images, videos, audios, frames, params)
    elif model == "happyhorse-1.1":
        if audios or any(ref.role == "last_frame" for ref in images):
            _reject(
                model,
                "audio references and last frames are not supported",
            )
        if videos:
            if operation != "edit":
                _reject(model, "video input requires operation='edit'; generate does not support video references")
            if len(videos) != 1 or len(images) > 5:
                _reject(model, "video editing requires one video and at most 5 reference images")
            result["action"] = "video-edit"
            result["url"] = videos[0].url
        else:
            if len(images) > 9:
                _reject(model, "at most 9 reference images are supported")
            if frames and ratio is not None:
                _reject(model, "first-frame mode uses the source image ratio; select automatic aspect ratio")
            result["action"] = "image-to-video" if frames else "reference-to-video"
        if images:
            result["image_urls" if frames else "reference_images"] = [ref.url for ref in images]
    elif model in {"kling-v3", "kling-v3-omni", "kling-video-o1"}:
        if videos or audios:
            _reject(
                model,
                "video references require an explicit provider edit/feature role; audio references are not supported",
            )
        if model == "kling-v3":
            if frames:
                result["image_with_roles"] = [
                    {"url": ref.url, "role": "reference_image" if ref.role == "reference" else ref.role}
                    for ref in images
                ]
            else:
                result["reference_images"] = [ref.url for ref in images]
        else:
            result["metadata"] = {
                "image_list": [
                    {
                        "image_url": ref.url,
                        **(
                            {"type": "first_frame" if ref.role == "first_frame" else "end_frame"}
                            if ref.role != "reference"
                            else {}
                        ),
                    }
                    for ref in images
                ]
            }
    elif model in {"Veo3.1-fast-official", "Veo3.1-quality-official"}:
        if videos or audios or len(images) > 3:
            _reject(model, "only up to 3 images are supported")
        if frames:
            result["image_urls"] = [ref.url for ref in frames if ref.role == "first_frame"]
            last = next((ref for ref in frames if ref.role == "last_frame"), None)
            if last:
                result["metadata"] = {"lastFrame": last.url}
        else:
            result["metadata"] = {"referenceImages": [ref.url for ref in images]}
    elif model in {"veo3.1-fast", "veo3.1-quality", "veo3.1-lite"}:
        if videos or audios:
            _reject(model, "video/audio reference inputs are not supported")
        if frames:
            if len(frames) != 2:
                _reject(model, "frame mode requires both first and last frames")
            generation_type = "frame"
            ordered = sorted(frames, key=lambda ref: ref.role != "first_frame")
        else:
            if model == "veo3.1-quality" or len(images) != 3:
                _reject(model, "reference mode requires 3 images and is unavailable on veo3.1-quality")
            generation_type = "reference"
            ordered = images
        metadata = result.get("metadata")
        result["metadata"] = {**(metadata if isinstance(metadata, dict) else {}), "generation_type": generation_type}
        result["image_urls"] = [ref.url for ref in ordered]
    else:
        _reject(model, "no documented lossless mapping for canonical references")
    return result


def _map_gemini_grok(
    model: str,
    result: dict[str, object],
    references: list[VideoReference],
    params: Mapping[str, object],
) -> None:
    images = [ref for ref in references if ref.type == "image"]
    videos = [ref for ref in references if ref.type == "video"]
    if any(ref.type == "audio" for ref in references):
        _reject(model, "audio references are not supported")
    if references and params.get("tools"):
        _reject(model, "tools cannot be combined with media references")
    if model in _GEMINI_MODELS:
        if any(ref.role != "reference" for ref in images):
            _reject(model, "explicit first/last-frame control has no documented mapping; use ordinary reference images")
        limit = 3 if model == "gemini-omni-flash" else 10
        if len(images) > limit:
            _reject(model, f"at most {limit} reference images are supported")
        if videos:
            if model != "gemini-omni-flash-preview-official":
                _reject(model, "video references are not supported")
            if params.get("operation") != "edit":
                _reject(model, "video input requires operation='edit'; generate does not support video references")
            if images or len(videos) > 3:
                _reject(model, "use at most 3 input videos without images")
            if params.get("aspect_ratio") is not None:
                _reject(model, "video editing cannot honor aspect_ratio; select automatic aspect ratio")
            result["video_list"] = [{"video_url": ref.url} for ref in videos]
        elif images:
            result["image_urls"] = [ref.url for ref in images]
        return
    if videos or any(ref.role == "last_frame" for ref in images):
        _reject(model, "video references and last frames are not supported")
    if model == "grok-video-1.5":
        if len(images) != 1:
            _reject(model, "requires exactly one image reference or first_frame")
        result["image"] = images[0].url
    else:
        if len(images) > 8:
            _reject(model, "at most 8 images including the main image are supported")
        first = next((ref for ref in images if ref.role == "first_frame"), None)
        if first:
            result["image"] = first.url
        regular = [ref.url for ref in images if ref.role == "reference"]
        if regular:
            result["reference_images"] = regular


def _map_multimodal(
    model: str,
    result: dict[str, object],
    images: list[VideoReference],
    videos: list[VideoReference],
    audios: list[VideoReference],
    frames: list[VideoReference],
    params: Mapping[str, object],
) -> None:
    image_limit = 30 if model == "seedance-2-5" else 10 if model == "wan3.0-video" else 9
    video_limit = (
        10
        if model == "seedance-2-5"
        else 5
        if model == "wan3.0-video"
        else 3
        if model in {"seedance-2-mini", "MiniMax-H3"}
        else None
    )
    if len(images) > image_limit or (
        video_limit is not None and (len(videos) > video_limit or len(audios) > video_limit)
    ):
        _reject(model, "reference count exceeds the documented model limit")
    if model == "MiniMax-H3" and len(images) + len(videos) + len(audios) > 12:
        _reject(model, "at most 12 total references are supported")
    if audios and not images and not videos and model != "seedance-2-5" and model != "wan3.0-video":
        _reject(model, "audio references require at least one image or video reference")
    if frames and model in {"seedance-2-5", "MiniMax-H3"}:
        if params.get("aspect_ratio") is not None:
            _reject(model, "frame mode uses the source image ratio; select automatic aspect ratio")
        result["aspect_ratio"] = "adaptive"
        if model == "seedance-2-5":
            if params.get("seconds") not in (None, "-1", -1):
                _reject(model, "frame mode requires automatic duration (-1)")
            result["duration"] = -1
    if model == "wan3.0-video":
        if frames:
            result["image_with_roles"] = [{"url": ref.url, "role": ref.role} for ref in frames]
        elif images:
            result["reference_images"] = [ref.url for ref in images]
        if videos:
            result["video_list"] = [{"video_url": ref.url} for ref in videos]
    else:
        if images:
            result["image_with_roles"] = [
                {"url": ref.url, "role": ref.role if ref.role != "reference" else "reference_image"} for ref in images
            ]
        if videos:
            result["video_with_roles"] = [{"url": ref.url, "role": "reference_video"} for ref in videos]
    if audios:
        result["audio_with_roles"] = [{"url": ref.url, "role": "reference_audio"} for ref in audios]
