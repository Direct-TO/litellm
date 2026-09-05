from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final, cast

import httpx
from pydantic import TypeAdapter

from litellm.exceptions import UnsupportedParamsError
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.image_generation.transformation import BaseImageGenerationConfig
from litellm.llms.base_llm.submission_utils import mark_submission_outcome
from litellm.types.llms.openai import AllMessageValues, OpenAIImageGenerationOptionalParams
from litellm.types.utils import ImageObject, ImageResponse

from ..common_utils import (
    build_zexapi_endpoint,
    build_zexapi_gemini_endpoint,
    get_zexapi_api_key,
    raise_for_zexapi_error,
)

_SUPPORTED_PARAMS: Final[tuple[OpenAIImageGenerationOptionalParams, ...]] = (
    "aspect_ratio",
    "imageConfig",
    "n",
    "resolution",
    "response_format",
    "size",
)
_IMAGE_RATIO_TO_SIZE_BY_RESOLUTION: Final[Mapping[str, Mapping[str, str]]] = MappingProxyType(
    {
        "1K": MappingProxyType(
            {
                "1:1": "1024x1024",
                "16:9": "1280x720",
                "9:16": "720x1280",
                "3:2": "1248x832",
                "2:3": "832x1248",
                "4:3": "1152x864",
                "3:4": "864x1152",
                "5:4": "1120x896",
                "4:5": "896x1120",
                "21:9": "1456x624",
            }
        ),
        "2K": MappingProxyType(
            {
                "1:1": "2048x2048",
                "16:9": "2560x1440",
                "9:16": "1440x2560",
                "3:2": "2496x1664",
                "2:3": "1664x2496",
                "4:3": "2304x1728",
                "3:4": "1728x2304",
                "5:4": "2240x1792",
                "4:5": "1792x2240",
                "21:9": "3024x1296",
            }
        ),
        "4K": MappingProxyType(
            {
                "1:1": "2880x2880",
                "16:9": "3840x2160",
                "9:16": "2160x3840",
                "3:2": "3504x2336",
                "2:3": "2336x3504",
                "4:3": "3264x2448",
                "3:4": "2448x3264",
                "5:4": "3200x2560",
                "4:5": "2560x3200",
                "21:9": "3696x1584",
            }
        ),
    }
)
_IMAGE2_RATIO_TO_SIZE: Final = _IMAGE_RATIO_TO_SIZE_BY_RESOLUTION["1K"]
_IMAGE2_RATIOS: Final[frozenset[str]] = frozenset(_IMAGE2_RATIO_TO_SIZE)
_IMAGE2_SIZE_TO_RATIO: Final[Mapping[str, str]] = MappingProxyType(
    {size: ratio for ratio, size in _IMAGE2_RATIO_TO_SIZE.items()}
)
_ALL_IMAGE_SIZE_TO_RATIO: Final[Mapping[str, str]] = MappingProxyType(
    {
        size: ratio
        for sizes_by_ratio in _IMAGE_RATIO_TO_SIZE_BY_RESOLUTION.values()
        for ratio, size in sizes_by_ratio.items()
    }
)
_ALL_IMAGE_SIZE_TO_RESOLUTION: Final[Mapping[str, str]] = MappingProxyType(
    {
        size: resolution
        for resolution, sizes_by_ratio in _IMAGE_RATIO_TO_SIZE_BY_RESOLUTION.items()
        for size in sizes_by_ratio.values()
    }
)
_IMAGE_MODEL_RESOLUTIONS: Final[Mapping[str, tuple[str, ...]]] = MappingProxyType(
    {
        "image2": ("1K",),
        "gpt-image2": ("1K", "2K", "4K"),
    }
)
_BANANA_MODELS: Final[frozenset[str]] = frozenset(("gemini-3-pro-image-preview", "gemini-3.1-flash-image-preview"))
_BANANA_RATIOS: Final[frozenset[str]] = frozenset(
    ("1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9")
)
_BANANA_RESOLUTIONS: Final[frozenset[str]] = frozenset(("1K", "2K", "4K"))
_IMAGE_CONFIG_FIELDS: Final[frozenset[str]] = frozenset(("aspectRatio", "imageSize", "resolution"))
_BANANA_SUPPORTED_PARAMS: Final[tuple[OpenAIImageGenerationOptionalParams, ...]] = (
    "aspect_ratio",
    "imageConfig",
    "image_url",
    "n",
    "resolution",
    "response_format",
    "size",
)
_OBJECT_MAP_ADAPTER: Final = TypeAdapter(dict[str, object])
_OBJECT_LIST_ADAPTER: Final = TypeAdapter(list[dict[str, object]])


def get_zexapi_image_generation_config(model: str) -> BaseImageGenerationConfig:
    if model in _BANANA_MODELS:
        return ZexAPIBananaImageGenerationConfig()
    return ZexAPIImageGenerationConfig()


def get_zexapi_image_model_sizes(model: str) -> frozenset[str]:
    supported_resolutions: Final = _IMAGE_MODEL_RESOLUTIONS.get(model)
    if supported_resolutions is None:
        return frozenset()
    return frozenset(
        size for resolution in supported_resolutions for size in _IMAGE_RATIO_TO_SIZE_BY_RESOLUTION[resolution].values()
    )


def _canonical_ratio(
    value: object,
    supported_ratios: frozenset[str],
    *,
    field: str,
    model: str,
    contract_name: str,
    size_to_ratio: Mapping[str, str] = _IMAGE2_SIZE_TO_RATIO,
) -> str:
    ratio: Final[str | None] = (
        value
        if isinstance(value, str) and value in supported_ratios
        else (size_to_ratio.get(value) if isinstance(value, str) else None)
    )
    if ratio is not None and ratio in supported_ratios:
        return ratio
    raise UnsupportedParamsError(
        message=f"{contract_name} does not support {field}={value!r}",
        model=model,
        llm_provider="zexapi",
    )


def _canonical_banana_resolution(value: object, *, field: str, model: str) -> str:
    if not isinstance(value, str):
        raise UnsupportedParamsError(
            message=f"ZexAPI Banana {field} must be a string",
            model=model,
            llm_provider="zexapi",
        )
    resolution: Final = value.upper()
    if resolution in _BANANA_RESOLUTIONS:
        return resolution
    raise UnsupportedParamsError(
        message=f"ZexAPI Banana does not support {field}={value!r}",
        model=model,
        llm_provider="zexapi",
    )


def _canonical_image_resolution(
    value: object,
    supported_resolutions: tuple[str, ...],
    *,
    field: str,
    model: str,
) -> str:
    if not isinstance(value, str):
        raise UnsupportedParamsError(
            message=f"ZexAPI model={model!r} {field} must be a string",
            model=model,
            llm_provider="zexapi",
        )
    resolution: Final = value.upper()
    if resolution in supported_resolutions:
        return resolution
    raise UnsupportedParamsError(
        message=f"ZexAPI model={model!r} does not support {field}={value!r}",
        model=model,
        llm_provider="zexapi",
    )


def _normalize_banana_reference_parts(value: object, *, model: str) -> list[dict[str, object]]:
    if value is None:
        return []
    raw_items: Sequence[object]
    if isinstance(value, (str, Mapping)):
        raw_items = (cast(object, value),)
    elif isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        raw_items = cast(Sequence[object], value)
    else:
        raise UnsupportedParamsError(
            message="image_url must be a URL/data URI, object, or sequence of either",
            model=model,
            llm_provider="zexapi",
        )
    parts: list[dict[str, object]] = []
    for item in raw_items:
        raw_url: object = cast(Mapping[str, object], item).get("url") if isinstance(item, Mapping) else item
        if not isinstance(raw_url, str):
            raise UnsupportedParamsError(
                message="image_url entries must be strings or {'url': ...} objects",
                model=model,
                llm_provider="zexapi",
            )
        if raw_url.startswith("data:") and ";base64," in raw_url:
            header, data = raw_url.split(",", 1)
            mime_type = header.removeprefix("data:").removesuffix(";base64")
            parts.append({"inlineData": {"mimeType": mime_type, "data": data}})
        else:
            parts.append({"fileData": {"fileUri": raw_url}})
    return parts


class ZexAPIImageGenerationConfig(BaseImageGenerationConfig):
    def get_supported_openai_params(
        self, model: str
    ) -> list[OpenAIImageGenerationOptionalParams]:  # mutable-ok: BaseImageGenerationConfig requires a list
        return list(_SUPPORTED_PARAMS)  # mutable-ok: BaseImageGenerationConfig requires a concrete list

    def map_openai_params(
        self,
        non_default_params: Mapping[str, object],
        optional_params: Mapping[str, object],
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: image parameter mapping contract requires a concrete dict
        supported_resolutions: Final = _IMAGE_MODEL_RESOLUTIONS.get(model)
        if supported_resolutions is None:
            raise UnsupportedParamsError(
                message=f"image-generation does not support ZexAPI model={model!r}",
                model=model,
                llm_provider="zexapi",
            )
        params: Final = MappingProxyType(
            {  # mutable-ok: merged parameter map is immediately frozen
                **optional_params,
                **non_default_params,
            }
        )
        response_format: Final = params.get("response_format")
        if response_format not in (None, "url"):
            raise UnsupportedParamsError(
                message="image-generation only supports response_format='url'",
                model=model,
                llm_provider="zexapi",
            )
        count: Final = params.get("n")
        if count not in (None, 1):
            raise UnsupportedParamsError(
                message=f"ZexAPI model={model!r} currently supports n=1 only",
                model=model,
                llm_provider="zexapi",
            )
        raw_image_config: Final = params.get("imageConfig")
        if raw_image_config is not None and not isinstance(raw_image_config, Mapping):
            raise UnsupportedParamsError(
                message="imageConfig must be an object",
                model=model,
                llm_provider="zexapi",
            )
        image_config: Final[Mapping[str, object]] = (
            cast(Mapping[str, object], raw_image_config)
            if isinstance(raw_image_config, Mapping)
            else MappingProxyType({})
        )
        unsupported_image_config_fields: Final = frozenset(image_config).difference(_IMAGE_CONFIG_FIELDS)
        if unsupported_image_config_fields:
            raise UnsupportedParamsError(
                message=(
                    f"ZexAPI model={model!r} does not support imageConfig field(s): "
                    f"{', '.join(sorted(unsupported_image_config_fields))}"
                ),
                model=model,
                llm_provider="zexapi",
            )
        raw_ratio_candidates: Final = (
            ("size", params.get("size")),
            ("aspect_ratio", params.get("aspect_ratio")),
            ("imageConfig.aspectRatio", image_config.get("aspectRatio")),
        )
        canonical_ratios: Final = tuple(
            _canonical_ratio(
                value,
                _IMAGE2_RATIOS,
                field=field,
                model=model,
                contract_name=f"ZexAPI model={model!r}",
                size_to_ratio=_ALL_IMAGE_SIZE_TO_RATIO,
            )
            for field, value in raw_ratio_candidates
            if value is not None
        )
        if len(frozenset(canonical_ratios)) > 1:
            raise UnsupportedParamsError(
                message="size, aspect_ratio, and imageConfig.aspectRatio must describe the same aspect ratio",
                model=model,
                llm_provider="zexapi",
            )
        ratio: Final = canonical_ratios[0] if canonical_ratios else "1:1"

        raw_resolution_candidates: Final = (
            ("resolution", params.get("resolution")),
            ("imageConfig.imageSize", image_config.get("imageSize")),
            ("imageConfig.resolution", image_config.get("resolution")),
            *(
                (f"{field} pixel size", _ALL_IMAGE_SIZE_TO_RESOLUTION[value])
                for field, value in raw_ratio_candidates
                if isinstance(value, str) and value in _ALL_IMAGE_SIZE_TO_RESOLUTION
            ),
        )
        canonical_resolutions: Final = tuple(
            _canonical_image_resolution(
                value,
                supported_resolutions,
                field=field,
                model=model,
            )
            for field, value in raw_resolution_candidates
            if value is not None
        )
        if len(frozenset(canonical_resolutions)) > 1:
            raise UnsupportedParamsError(
                message=(
                    "resolution, imageConfig.imageSize, imageConfig.resolution, and pixel size "
                    "must describe the same resolution"
                ),
                model=model,
                llm_provider="zexapi",
            )
        resolution: Final = canonical_resolutions[0] if canonical_resolutions else "1K"
        provider_size: Final = _IMAGE_RATIO_TO_SIZE_BY_RESOLUTION[resolution][ratio]
        return {  # mutable-ok: image parameter mapping contract requires a concrete dict
            "size": provider_size,
            "response_format": "url",
        }

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        stream: bool | None = None,
    ) -> str:
        return build_zexapi_endpoint(api_base, "/v1/images/generations")

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: image handler may add caller-provided headers
        resolved_api_key: Final = get_zexapi_api_key(api_key)
        if resolved_api_key is None:
            raise ValueError("ZEXAPI_API_KEY is required")
        return {  # mutable-ok: image handler may add caller-provided headers
            **headers,
            "Authorization": f"Bearer {resolved_api_key}",
            "Content-Type": "application/json",
        }

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        headers: Mapping[str, str],
    ) -> dict[str, object]:  # mutable-ok: image HTTP handler requires a concrete JSON dict
        unsupported_fields: Final = tuple(
            field for field in ("resolution", "imageConfig", "image_url") if field in optional_params
        )
        if unsupported_fields:
            raise UnsupportedParamsError(
                message=f"ZexAPI model={model!r} does not support unmapped parameter(s): {', '.join(unsupported_fields)}",
                model=model,
                llm_provider="zexapi",
            )
        return {  # mutable-ok: image HTTP handler requires a concrete JSON dict
            "model": model,
            "prompt": prompt,
            **{  # mutable-ok: JSON payload filtering requires a concrete dict for expansion
                key: value for key, value in optional_params.items() if key != "extra_headers"
            },
        }

    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: Mapping[str, object],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        encoding: object,
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ImageResponse:
        raise_for_zexapi_error(raw_response)
        try:
            provider_response: Final = ImageResponse.model_validate_json(raw_response.text)
        except ValueError as exc:
            raise mark_submission_outcome(
                BaseLLMException(
                    status_code=raw_response.status_code,
                    message=f"Invalid ZexAPI image response: {exc}",
                    headers=raw_response.headers,
                ),
                "unknown",
            ) from exc
        images: Final = tuple(provider_response.data or ())
        urls: Final = tuple(image.url for image in images if image.url is not None)
        if len(urls) != len(images):
            raise mark_submission_outcome(
                BaseLLMException(
                    status_code=502,
                    message="ZexAPI image response did not include a URL for every image",
                    headers=raw_response.headers,
                ),
                "unknown",
            )
        return ImageResponse(
            created=provider_response.created,
            data=[ImageObject(url=url) for url in urls],  # mutable-ok: ImageResponse requires a list
            usage=provider_response.usage,
        )


class ZexAPIBananaImageGenerationConfig(ZexAPIImageGenerationConfig):
    def get_supported_openai_params(
        self, model: str
    ) -> list[OpenAIImageGenerationOptionalParams]:  # mutable-ok: BaseImageGenerationConfig requires a list
        return list(_BANANA_SUPPORTED_PARAMS)  # mutable-ok: BaseImageGenerationConfig requires a concrete list

    def map_openai_params(
        self,
        non_default_params: Mapping[str, object],
        optional_params: Mapping[str, object],
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: image parameter mapping contract requires a concrete dict
        if model not in _BANANA_MODELS:
            raise UnsupportedParamsError(
                message=f"ZexAPI Banana image generation does not support model={model!r}",
                model=model,
                llm_provider="zexapi",
            )
        params: Final = MappingProxyType({**optional_params, **non_default_params})
        response_format: Final = params.get("response_format", "url")
        if response_format not in ("url", "b64_json"):
            raise UnsupportedParamsError(
                message="ZexAPI Banana supports response_format='url' or 'b64_json'",
                model=model,
                llm_provider="zexapi",
            )
        count: Final = params.get("n")
        if count not in (None, 1):
            raise UnsupportedParamsError(
                message="ZexAPI Banana currently supports n=1 only",
                model=model,
                llm_provider="zexapi",
            )
        raw_image_config: Final = params.get("imageConfig")
        if raw_image_config is not None and not isinstance(raw_image_config, Mapping):
            raise UnsupportedParamsError(
                message="imageConfig must be an object",
                model=model,
                llm_provider="zexapi",
            )
        image_config: Final[Mapping[str, object]] = (
            cast(Mapping[str, object], raw_image_config)
            if isinstance(raw_image_config, Mapping)
            else MappingProxyType({})
        )
        unsupported_image_config_fields: Final = frozenset(image_config).difference(_IMAGE_CONFIG_FIELDS)
        if unsupported_image_config_fields:
            raise UnsupportedParamsError(
                message=(
                    "ZexAPI Banana does not support imageConfig field(s): "
                    f"{', '.join(sorted(unsupported_image_config_fields))}"
                ),
                model=model,
                llm_provider="zexapi",
            )
        raw_ratio_candidates: Final = (
            ("size", params.get("size")),
            ("aspect_ratio", params.get("aspect_ratio")),
            ("imageConfig.aspectRatio", image_config.get("aspectRatio")),
        )
        canonical_ratios: Final = tuple(
            _canonical_ratio(
                value,
                _BANANA_RATIOS,
                field=field,
                model=model,
                contract_name="ZexAPI Banana",
            )
            for field, value in raw_ratio_candidates
            if value is not None
        )
        if len(frozenset(canonical_ratios)) > 1:
            raise UnsupportedParamsError(
                message="size, aspect_ratio, and imageConfig.aspectRatio must describe the same aspect ratio",
                model=model,
                llm_provider="zexapi",
            )
        ratio: Final = canonical_ratios[0] if canonical_ratios else "1:1"

        raw_resolution_candidates: Final = (
            ("resolution", params.get("resolution")),
            ("imageConfig.imageSize", image_config.get("imageSize")),
            ("imageConfig.resolution", image_config.get("resolution")),
        )
        canonical_resolutions: Final = tuple(
            _canonical_banana_resolution(value, field=field, model=model)
            for field, value in raw_resolution_candidates
            if value is not None
        )
        if len(frozenset(canonical_resolutions)) > 1:
            raise UnsupportedParamsError(
                message="resolution, imageConfig.imageSize, and imageConfig.resolution must describe the same resolution",
                model=model,
                llm_provider="zexapi",
            )
        resolution: Final = canonical_resolutions[0] if canonical_resolutions else "1K"
        return {
            "response_format": response_format,
            "generationConfig": {
                "responseModalities": ["IMAGE"],
                "imageConfig": {"aspectRatio": ratio, "imageSize": resolution},
            },
            "reference_parts": _normalize_banana_reference_parts(params.get("image_url"), model=model),
        }

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        stream: bool | None = None,
    ) -> str:
        return build_zexapi_gemini_endpoint(api_base, model)

    def transform_image_generation_request(
        self,
        model: str,
        prompt: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        headers: Mapping[str, str],
    ) -> dict[str, object]:  # mutable-ok: image HTTP handler requires a concrete JSON dict
        raw_reference_parts: Final = optional_params.get("reference_parts", [])
        reference_parts: Final[list[dict[str, object]]] = (
            _OBJECT_LIST_ADAPTER.validate_python(raw_reference_parts) if isinstance(raw_reference_parts, list) else []
        )
        generation_config: Final = optional_params.get("generationConfig")
        return {
            "contents": [{"role": "user", "parts": [{"text": prompt}, *reference_parts]}],
            "generationConfig": generation_config,
            "response_format": optional_params.get("response_format", "url"),
        }

    def transform_image_generation_response(
        self,
        model: str,
        raw_response: httpx.Response,
        model_response: ImageResponse,
        logging_obj: LiteLLMLoggingObj,
        request_data: Mapping[str, object],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        encoding: object,
        api_key: str | None = None,
        json_mode: bool | None = None,
    ) -> ImageResponse:
        raise_for_zexapi_error(raw_response)
        try:
            payload: Final = _OBJECT_MAP_ADAPTER.validate_python(raw_response.json())
        except ValueError as exc:
            raise mark_submission_outcome(
                BaseLLMException(
                    status_code=502,
                    message=f"Invalid ZexAPI Banana image response: {exc}",
                    headers=raw_response.headers,
                ),
                "unknown",
            ) from exc
        images: list[ImageObject] = []
        raw_data: Final = payload.get("data")
        if isinstance(raw_data, list):
            for item in _OBJECT_LIST_ADAPTER.validate_python(raw_data):
                url = item.get("url")
                b64_json = item.get("b64_json")
                if isinstance(url, str) or isinstance(b64_json, str):
                    images.append(
                        ImageObject(
                            url=url if isinstance(url, str) else None,
                            b64_json=b64_json if isinstance(b64_json, str) else None,
                        )
                    )
        if not images:
            raw_candidates: Final = payload.get("candidates")
            if isinstance(raw_candidates, list):
                for candidate in _OBJECT_LIST_ADAPTER.validate_python(raw_candidates):
                    content = candidate.get("content")
                    parts = cast(Mapping[str, object], content).get("parts") if isinstance(content, Mapping) else None
                    if not isinstance(parts, list):
                        continue
                    for part in _OBJECT_LIST_ADAPTER.validate_python(parts):
                        image_url = part.get("image_url")
                        inline_data = part.get("inlineData")
                        url = (
                            cast(Mapping[str, object], image_url).get("url") if isinstance(image_url, Mapping) else None
                        )
                        data = (
                            cast(Mapping[str, object], inline_data).get("data")
                            if isinstance(inline_data, Mapping)
                            else None
                        )
                        if isinstance(url, str) or isinstance(data, str):
                            images.append(
                                ImageObject(
                                    url=url if isinstance(url, str) else None,
                                    b64_json=data if isinstance(data, str) else None,
                                )
                            )
        if not images:
            raise mark_submission_outcome(
                BaseLLMException(
                    status_code=502,
                    message="ZexAPI Banana image response did not include image data",
                    headers=raw_response.headers,
                ),
                "unknown",
            )
        created: Final = payload.get("created")
        return ImageResponse(created=created if isinstance(created, int) else None, data=images)
