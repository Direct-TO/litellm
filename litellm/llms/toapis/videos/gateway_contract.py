"""Canonical video mapping, bound to the model contracts linked in videos-contract.md."""

from collections.abc import Mapping
from typing import Final

from litellm.exceptions import UnsupportedParamsError
from litellm.videos.contract import VideoReference, validate_video_contract

_RESOLUTIONS: Final = {
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
_NATIVE_INPUTS: Final = frozenset(
    {
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
    }
)


def _reject(model: str, message: str) -> None:
    raise UnsupportedParamsError(message=f"ToAPIs model={model!r}: {message}", model=model, llm_provider="toapis")


def map_gateway_video(model: str, params: Mapping[str, object], size_field: str) -> dict[str, object]:
    if model not in GATEWAY_VIDEO_MODELS:
        _reject(model, "no documented mapping for canonical video inputs")
    references = validate_video_contract(params)
    if any(params.get(key) for key in _NATIVE_INPUTS):
        _reject(model, "canonical video inputs cannot be combined with native reference/metadata overrides")
    result = {
        key: value
        for key, value in params.items()
        if key not in {"resolution", "aspect_ratio", "references", "seconds", "size", "extra_body"}
    }
    seconds = params.get("seconds")
    if seconds is not None:
        result["duration"] = int(str(seconds))
    ratio = params.get("aspect_ratio")
    if ratio is not None:
        known_ratios = {"16:9", "9:16", "1:1", "4:3", "3:4"}
        if model in {"seedance-2", "seedance-2-fast", "seedance-2-mini", "seedance-2-5", "MiniMax-H3"}:
            known_ratios.add("21:9")
            if ratio not in known_ratios:
                _reject(model, f"aspect_ratio={ratio!r} is not supported")
        elif model == "wan3.0-video" and ratio not in known_ratios:
            _reject(model, f"aspect_ratio={ratio!r} is not supported")
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
    if not references:
        return result

    images = [ref for ref in references if ref.type == "image"]
    videos = [ref for ref in references if ref.type == "video"]
    audios = [ref for ref in references if ref.type == "audio"]
    frames = [ref for ref in images if ref.role != "reference"]
    regular = [ref for ref in references if ref.role == "reference"]
    if frames and regular:
        _reject(model, "first/last frames cannot be mixed with ordinary image/video/audio references")
    if params.get("tools"):
        _reject(model, "tools cannot be combined with media references")

    if model in {"seedance-2", "seedance-2-fast", "seedance-2-mini", "seedance-2-5", "MiniMax-H3", "wan3.0-video"}:
        _map_multimodal(model, result, images, videos, audios, frames, params)
    elif model == "happyhorse-1.1":
        if videos or audios or any(ref.role == "last_frame" for ref in images):
            _reject(
                model,
                "generation supports reference images or a first frame; video editing requires its separate operation",
            )
        if len(images) > 9:
            _reject(model, "at most 9 reference images are supported")
        result["action"] = "image-to-video" if frames else "reference-to-video"
        result["image_urls" if frames else "reference_images"] = [ref.url for ref in images]
    elif model in {"kling-v3", "kling-v3-omni", "kling-video-o1"}:
        if videos or audios:
            _reject(
                model,
                "video references require an explicit provider edit/feature role; audio references are not supported",
            )
        if model == "kling-v3":
            if not frames:
                _reject(model, "use explicit first_frame/last_frame roles for image-to-video")
            result["image_urls"] = [ref.url for ref in sorted(frames, key=lambda ref: ref.role != "first_frame")]
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
