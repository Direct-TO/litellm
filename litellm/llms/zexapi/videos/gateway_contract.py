"""Omni JSON mapping documented at https://6l0ket291i.apifox.cn/462450037e0."""

from collections.abc import Mapping
from typing import Final

from litellm.exceptions import UnsupportedParamsError
from litellm.videos.contract import VIDEO_CONTRACT_FIELDS, require_video_contract_support, validate_video_contract

GATEWAY_VIDEO_MODELS: Final = frozenset({"omni_flash-10s", "omni_flash-10s-fl"})


def _reject(model: str, message: str) -> None:
    raise UnsupportedParamsError(message=f"ZexAPI model={model!r}: {message}", model=model, llm_provider="zexapi")


def map_gateway_video(model: str, params: Mapping[str, object]) -> dict[str, object]:
    require_video_contract_support(params, list(VIDEO_CONTRACT_FIELDS) if model in GATEWAY_VIDEO_MODELS else [], model)
    references = validate_video_contract(params)
    allowed = VIDEO_CONTRACT_FIELDS | {"seconds", "extra_body", "extra_headers"}
    unsupported = sorted(key for key, value in params.items() if value is not None and key not in allowed)
    if unsupported:
        _reject(model, f"canonical inputs cannot be combined with: {', '.join(unsupported)}")
    if params.get("resolution") not in (None, "720p"):
        _reject(model, "fixed resolution is 720p")
    if params.get("seconds") not in (None, "10", 10, "-1", -1):
        _reject(model, "fixed duration is 10 seconds; omit seconds or use -1 for automatic selection")
    ratio = params.get("aspect_ratio")
    if ratio not in (None, "16:9", "9:16"):
        _reject(model, "aspect_ratio must be 16:9 or 9:16")
    result = {key: params[key] for key in ("extra_body", "extra_headers") if params.get(key) is not None}
    if ratio is not None:
        result["size"] = "1280x720" if ratio == "16:9" else "720x1280"
    if any(ref.type == "audio" for ref in references):
        _reject(model, "audio references are not supported")
    if model == "omni_flash-10s-fl":
        if not references or any(ref.type != "image" or ref.role == "reference" for ref in references):
            _reject(
                model, "requires an explicit first_frame and optional last_frame; ordinary references are not supported"
            )
        ordered = sorted(references, key=lambda ref: ref.role != "first_frame")
    else:
        if any(ref.role != "reference" for ref in references):
            _reject(model, "first/last frames require the omni_flash-10s-fl model")
        if len(references) > 7:
            _reject(model, "at most 7 image or video references are supported")
        if len({ref.type for ref in references}) > 1:
            _reject(model, "mixing images and input videos has no documented mapping")
        ordered = references
    if ordered:
        result["images"] = [ref.url for ref in ordered]
    return result
