"""Contracts for ZexAPI's image tasks served by /v1/videos."""

import base64
import binascii
import io
from collections.abc import Mapping, Sequence
from os import PathLike
from types import MappingProxyType
from typing import IO, Final, TypeAlias, cast

import httpx

from litellm.exceptions import UnsupportedParamsError
from litellm.images.utils import ImageEditRequestUtils
from litellm.litellm_core_utils.prompt_templates.common_utils import extract_file_data
from litellm.llms.base_llm.image_generation.transformation import BaseImageGenerationConfig
from litellm.llms.openai.image_edit.transformation import OpenAIImageEditConfig
from litellm.types.images.main import ImageEditOptionalRequestParams
from litellm.types.llms.openai import OpenAIImageGenerationOptionalParams

from ..common_utils import build_zexapi_endpoint
from .transformation import resolve_zexapi_image_size

AsyncImageContent: TypeAlias = bytes | IO[bytes] | PathLike[str]
AsyncImageFile: TypeAlias = (
    AsyncImageContent
    | tuple[str | None, AsyncImageContent]
    | tuple[str | None, AsyncImageContent, str | None]
    | tuple[str | None, AsyncImageContent, str | None, Mapping[str, str]]
)

ASYNC_IMAGE_MODEL_TIERS: Final[Mapping[str, str]] = MappingProxyType(
    {"gpt-image-2": "1K", "gpt-image-2-2K": "2K", "gpt-image-2-4K": "4K"}
)
_SUPPORTED_PARAMS: Final[tuple[OpenAIImageGenerationOptionalParams, ...]] = (
    "aspect_ratio",
    "resolution",
    "size",
    "image_url",
    "n",
    "response_format",
    "background",
    "quality",
    "imageConfig",
)
_IMAGE_MIME_TYPES: Final = frozenset(("image/png", "image/jpeg", "image/webp"))


def is_zexapi_async_image_model(model: str | None) -> bool:
    return model.removeprefix("zexapi/") in ASYNC_IMAGE_MODEL_TIERS if model is not None else False


def async_image_error(model: str, message: str) -> UnsupportedParamsError:
    return UnsupportedParamsError(message=message, model=model, llm_provider="zexapi")


def _fixed(value: object) -> object:
    return None if value is None or (isinstance(value, str) and value.strip().lower() == "auto") else value


def normalize_async_references(value: object, model: str) -> list[str]:
    if value is None:
        return []
    items: Sequence[object] = (
        (cast(object, value),)
        if isinstance(value, (str, Mapping))
        else (
            cast(Sequence[object], value)
            if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray))
            else ()
        )
    )
    if not items and not isinstance(value, (list, tuple)):
        raise async_image_error(model, "image_url must be a URL, data URI, or sequence of references")
    if len(items) > 8:
        raise async_image_error(model, "ZexAPI asynchronous images support at most 8 references")
    references: list[str] = []
    for item in items:
        url = cast(Mapping[str, object], item).get("url") if isinstance(item, Mapping) else item
        if not isinstance(url, str) or not url:
            raise async_image_error(model, "Each image reference must contain a URL or data URI")
        if url.startswith("data:"):
            header, separator, data = url.partition(",")
            if not separator or not header.endswith(";base64") or header[5:-7] not in _IMAGE_MIME_TYPES:
                raise async_image_error(model, "References require PNG, JPEG or WebP base64 data URIs")
            try:
                if not base64.b64decode(data, validate=True):
                    raise ValueError("empty image")
            except (ValueError, binascii.Error) as exc:
                raise async_image_error(model, "Invalid base64 image reference") from exc
        else:
            try:
                parsed = httpx.URL(url)
                valid = parsed.is_absolute_url and parsed.scheme in ("http", "https")
            except httpx.InvalidURL:
                valid = False
            if not valid:
                raise async_image_error(model, "Reference URLs must be absolute HTTP(S) URLs")
        references.append(url)
    return references


def async_image_params(model: str, params: Mapping[str, object]) -> dict[str, object]:
    tier = ASYNC_IMAGE_MODEL_TIERS.get(model.removeprefix("zexapi/"))
    if tier is None:
        raise async_image_error(model, f"Unsupported ZexAPI asynchronous image model={model!r}")
    for field in ("mask", "input_fidelity", "quality", "imageConfig"):
        if params.get(field) is not None:
            raise async_image_error(model, f"ZexAPI asynchronous images do not support {field}")
    if params.get("background") not in (None, "opaque", "auto"):
        raise async_image_error(model, "ZexAPI asynchronous images do not support transparent output")
    if params.get("n") not in (None, 1, "1"):
        raise async_image_error(model, "ZexAPI asynchronous images support n=1 only")
    if params.get("response_format") not in (None, "url"):
        raise async_image_error(model, "ZexAPI asynchronous images support response_format='url' only")
    resolution = _fixed(params.get("resolution"))
    if resolution is not None and (not isinstance(resolution, str) or resolution.upper() != tier):
        raise async_image_error(model, f"ZexAPI model={model!r} requires resolution={tier!r}")
    ratio, size = _fixed(params.get("aspect_ratio")), _fixed(params.get("size"))
    if ratio is not None and (not isinstance(ratio, str) or ":" not in ratio):
        raise async_image_error(model, "aspect_ratio must be a documented ratio, not a pixel size")
    result: dict[str, object] = {}
    if size is not None:
        if not isinstance(size, str) or "x" not in size:
            raise async_image_error(model, "size must be a documented pixel size; use aspect_ratio for ratios")
        resolve_zexapi_image_size("gpt-image2", {"size": size, "aspect_ratio": ratio, "resolution": tier})
        # The public API allows consistent size/ratio pairs; upstream accepts only one.
        result["size"] = size
    elif ratio is not None:
        resolve_zexapi_image_size("gpt-image2", {"aspect_ratio": ratio, "resolution": tier})
        result["aspect_ratio"] = ratio
    if params.get("image_url") is not None:
        result["image_url"] = normalize_async_references(params["image_url"], model)
    return result


def async_edit_references(images: Sequence[AsyncImageFile], model: str) -> list[str]:
    if not images or len(images) > 8:
        raise async_image_error(model, "ZexAPI asynchronous image edits require 1 to 8 reference files")
    references: list[str] = []
    for image in images:
        content = image[1] if isinstance(image, tuple) else image
        stream = content if isinstance(content, io.IOBase) else None
        if stream is not None and not stream.seekable():
            raise async_image_error(model, "Image reference streams must be seekable")
        position = stream.tell() if stream is not None else None
        try:
            if stream is not None:
                stream.seek(0)
            extracted = extract_file_data(image)
            data = extracted["content"]
        finally:
            if stream is not None and position is not None:
                stream.seek(position)
        mime = ImageEditRequestUtils.get_image_content_type(data)
        if not data or mime not in _IMAGE_MIME_TYPES:
            raise async_image_error(model, "Image references must be PNG, JPEG or WebP files")
        references.append(f"data:{mime};base64,{base64.b64encode(data).decode('ascii')}")
    return references


class ZexAPIAsyncImageGenerationConfig(BaseImageGenerationConfig):
    def get_supported_openai_params(self, model: str) -> list[OpenAIImageGenerationOptionalParams]:
        # Recognize semantic options so drop_params cannot silently discard them.
        return list(_SUPPORTED_PARAMS)

    def map_openai_params(
        self,
        non_default_params: Mapping[str, object],
        optional_params: Mapping[str, object],
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:
        return async_image_params(model, {**optional_params, **non_default_params})

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        headers: Mapping[str, str],
    ) -> dict[str, object]:
        mapped = async_image_params(model, optional_params)
        data: dict[str, object] = {
            "model": model,
            "prompt": prompt,
            **{key: value for key, value in mapped.items() if key in ("aspect_ratio", "size")},
        }
        references = normalize_async_references(mapped.get("image_url"), model)
        if references:
            data["images"] = references
        return data


class ZexAPIAsyncImageEditConfig(OpenAIImageEditConfig):
    def get_supported_openai_params(self, model: str) -> list[str]:
        return [
            "aspect_ratio",
            "resolution",
            "size",
            "n",
            "response_format",
            "mask",
            "background",
            "input_fidelity",
            "quality",
            "imageConfig",
        ]

    def map_openai_params(
        self, image_edit_optional_params: ImageEditOptionalRequestParams, model: str, drop_params: bool
    ) -> dict[str, object]:
        return async_image_params(model, image_edit_optional_params)

    def get_complete_url(self, model: str, api_base: str | None, litellm_params: Mapping[str, object]) -> str:
        return build_zexapi_endpoint(api_base, "/v1/videos")
