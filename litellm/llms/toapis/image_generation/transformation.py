import re
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final, Literal, NamedTuple, cast

import httpx

from litellm.exceptions import UnsupportedParamsError
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.url_utils import encode_url_path_segment
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.image_generation.transformation import BaseImageGenerationConfig
from litellm.types.llms.openai import AllMessageValues, OpenAIImageGenerationOptionalParams
from litellm.types.utils import ImageObject, ImageResponse

from ..common_utils import build_toapis_endpoint, get_toapis_api_key, parse_toapis_task
from .reference_upload import decode_reference

_SUPPORTED_PARAMS: Final[tuple[OpenAIImageGenerationOptionalParams, ...]] = (
    "aspect_ratio",
    "imageConfig",
    "image_url",
    "n",
    "resolution",
    "response_format",
    "size",
)
_PIXEL_SIZE_TO_RATIO: Final[Mapping[str, str]] = MappingProxyType(
    {
        "1024x1024": "1:1",
        "1792x1024": "16:9",
        "1024x1792": "9:16",
        "1365x1024": "4:3",
        "1024x1365": "3:4",
        "1536x1024": "3:2",
        "1024x1536": "2:3",
        "1280x720": "16:9",
        "720x1280": "9:16",
        "1248x832": "3:2",
        "832x1248": "2:3",
        "1152x864": "4:3",
        "864x1152": "3:4",
        "1120x896": "5:4",
        "896x1120": "4:5",
        "1456x624": "21:9",
    }
)
_GPT_IMAGE_2_PIXEL_SPECS: Final[Mapping[str, tuple[str, str]]] = MappingProxyType(
    {
        "1024x1024": ("1:1", "1k"),
        "1536x1024": ("3:2", "1k"),
        "1024x1536": ("2:3", "1k"),
        "1024x768": ("4:3", "1k"),
        "768x1024": ("3:4", "1k"),
        "1280x1024": ("5:4", "1k"),
        "1024x1280": ("4:5", "1k"),
        "1536x864": ("16:9", "1k"),
        "864x1536": ("9:16", "1k"),
        "2048x1024": ("2:1", "1k"),
        "1024x2048": ("1:2", "1k"),
        "2016x864": ("21:9", "1k"),
        "864x2016": ("9:21", "1k"),
        "2048x2048": ("1:1", "2k"),
        "2048x1360": ("3:2", "2k"),
        "1360x2048": ("2:3", "2k"),
        "2048x1536": ("4:3", "2k"),
        "1536x2048": ("3:4", "2k"),
        "2560x2048": ("5:4", "2k"),
        "2048x2560": ("4:5", "2k"),
        "2048x1152": ("16:9", "2k"),
        "1152x2048": ("9:16", "2k"),
        "2688x1344": ("2:1", "2k"),
        "1344x2688": ("1:2", "2k"),
        "2688x1152": ("21:9", "2k"),
        "1152x2688": ("9:21", "2k"),
        "2880x2880": ("1:1", "4k"),
        "3520x2336": ("3:2", "4k"),
        "2336x3520": ("2:3", "4k"),
        "3312x2480": ("4:3", "4k"),
        "2480x3312": ("3:4", "4k"),
        "3216x2576": ("5:4", "4k"),
        "2576x3216": ("4:5", "4k"),
        "3840x2160": ("16:9", "4k"),
        "2160x3840": ("9:16", "4k"),
        "3840x1920": ("2:1", "4k"),
        "1920x3840": ("1:2", "4k"),
        "3840x1648": ("21:9", "4k"),
        "1648x3840": ("9:21", "4k"),
    }
)
_GPT_IMAGE_2_RATIOS: Final[frozenset[str]] = frozenset(
    (
        "1:1",
        "3:2",
        "2:3",
        "4:3",
        "3:4",
        "5:4",
        "4:5",
        "16:9",
        "9:16",
        "2:1",
        "1:2",
        "21:9",
        "9:21",
    )
)
_STANDARD_IMAGE_CONFIG_FIELDS: Final[frozenset[str]] = frozenset(("aspectRatio", "imageSize", "resolution"))
_IMAGE_25_MODELS: Final = frozenset(
    f"gpt-image-2.5-{variant}{suffix}"
    for variant in ("flare", "sunburst")
    for suffix in ("", "-vip")
)
_PIXEL_IMAGE_MODELS: Final = _IMAGE_25_MODELS | {"gpt-image-2"}
_PIXEL_SIZE_PATTERN: Final = re.compile(r"([1-9][0-9]*)x([1-9][0-9]*)")


class _ImageModelSpec(NamedTuple):
    ratios: frozenset[str]
    resolutions: tuple[str, ...]
    resolution_location: Literal["top_level", "metadata"]
    reference_field: Literal["reference_images", "image_urls"]


_BANANA_25_RATIOS: Final = frozenset(("1:1", "16:9", "9:16", "4:3", "3:4", "3:2", "2:3"))
_BANANA_PRO_RATIOS: Final = frozenset(("1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9"))
_BANANA_31_RATIOS: Final = frozenset(
    (
        "1:1",
        "3:2",
        "2:3",
        "4:3",
        "3:4",
        "16:9",
        "9:16",
        "5:4",
        "4:5",
        "21:9",
        "1:4",
        "4:1",
        "1:8",
        "8:1",
    )
)
_MODEL_SPECS: Final[Mapping[str, _ImageModelSpec]] = MappingProxyType(
    {
        **{
            model: _ImageModelSpec(_GPT_IMAGE_2_RATIOS, ("1K", "2K", "4K"), "top_level", "reference_images")
            for model in _IMAGE_25_MODELS
        },
        "gpt-image-2": _ImageModelSpec(
            _GPT_IMAGE_2_RATIOS,
            ("1k", "2k", "4k"),
            "top_level",
            "reference_images",
        ),
        "gemini-2.5-flash-image-preview": _ImageModelSpec(_BANANA_25_RATIOS, ("1K",), "metadata", "image_urls"),
        "gemini-3-pro-image-preview": _ImageModelSpec(_BANANA_PRO_RATIOS, ("1K", "2K", "4K"), "metadata", "image_urls"),
        "gemini-3.1-flash-image-preview": _ImageModelSpec(
            _BANANA_31_RATIOS, ("0.5K", "1K", "2K", "4K"), "metadata", "image_urls"
        ),
    }
)


def _normalize_reference_urls(value: object, *, model: str) -> tuple[str, ...]:
    references: Final = _reference_strings(value, model=model)
    for reference in references:
        try:
            decode_reference(reference)
        except (ValueError, httpx.InvalidURL) as exc:
            raise UnsupportedParamsError(message=str(exc), model=model, llm_provider="toapis") from exc
    return references


def _reference_strings(value: object, *, model: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        return (value,)
    if isinstance(value, Mapping):
        mapping_value: Final = cast(Mapping[str, object], value)
        url: Final = mapping_value.get("url")
        if isinstance(url, str):
            return (url,)
    if isinstance(value, Sequence) and not isinstance(value, (bytes, bytearray, str)):
        urls: list[str] = []
        for item in cast(Sequence[object], value):
            if isinstance(item, str):
                urls.append(item)
            elif isinstance(item, Mapping):
                item_url = cast(Mapping[str, object], item).get("url")
                if isinstance(item_url, str):
                    urls.append(item_url)
                    continue
                raise UnsupportedParamsError(
                    message="image_url object entries require a string url",
                    model=model,
                    llm_provider="toapis",
                )
            else:
                raise UnsupportedParamsError(
                    message="image_url entries must be URL strings or {'url': ...} objects",
                    model=model,
                    llm_provider="toapis",
                )
        return tuple(urls)
    raise UnsupportedParamsError(
        message="image_url must be a URL string, URL object, or sequence of either",
        model=model,
        llm_provider="toapis",
    )


def _canonical_resolution(value: object, spec: _ImageModelSpec, *, model: str) -> str:
    if value is None:
        return spec.resolutions[0]
    if not isinstance(value, str):
        raise UnsupportedParamsError(
            message=f"image resolution must be a string, got {value!r}",
            model=model,
            llm_provider="toapis",
        )
    for resolution in spec.resolutions:
        if value.lower() == resolution.lower():
            return resolution
    raise UnsupportedParamsError(
        message=f"ToAPIs model={model!r} does not support resolution={value!r}",
        model=model,
        llm_provider="toapis",
    )


def _canonical_ratio(value: object, spec: _ImageModelSpec, *, field: str, model: str) -> str:
    official_pixel_spec: Final = (
        _GPT_IMAGE_2_PIXEL_SPECS.get(value) if model in _PIXEL_IMAGE_MODELS and isinstance(value, str) else None
    )
    ratio_alias: Final[str | None] = (
        official_pixel_spec[0]
        if official_pixel_spec is not None
        else (_PIXEL_SIZE_TO_RATIO.get(value) if isinstance(value, str) else None)
    )
    ratio: Final[str | None] = value if isinstance(value, str) and value in spec.ratios else ratio_alias
    if ratio is not None and ratio in spec.ratios:
        return ratio
    raise UnsupportedParamsError(
        message=f"ToAPIs model={model!r} does not support {field}={value!r}",
        model=model,
        llm_provider="toapis",
    )


class ToAPISImageGenerationConfig(BaseImageGenerationConfig):
    def get_supported_openai_params(
        self, model: str
    ) -> list[OpenAIImageGenerationOptionalParams]:  # mutable-ok: BaseImageGenerationConfig requires a list
        extra: Final[tuple[OpenAIImageGenerationOptionalParams, ...]] = (
            ("quality", "background") if model in _IMAGE_25_MODELS else ()
        )
        return list(_SUPPORTED_PARAMS + extra)  # mutable-ok: BaseImageGenerationConfig requires a concrete list

    def map_openai_params(
        self,
        non_default_params: Mapping[str, object],
        optional_params: Mapping[str, object],
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: image parameter mapping contract requires a concrete dict
        spec: Final = _MODEL_SPECS.get(model)
        if spec is None:
            raise UnsupportedParamsError(
                message=f"image-generation does not support ToAPIs model={model!r}",
                model=model,
                llm_provider="toapis",
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
                llm_provider="toapis",
            )
        count: Final = params.get("n")
        if count not in (None, 1):
            raise UnsupportedParamsError(
                message="ToAPIs image generation currently supports n=1 only",
                model=model,
                llm_provider="toapis",
            )
        raw_image_config: Final = params.get("imageConfig")
        if raw_image_config is not None and not isinstance(raw_image_config, Mapping):
            raise UnsupportedParamsError(
                message="imageConfig must be an object",
                model=model,
                llm_provider="toapis",
            )
        image_config: Final[Mapping[str, object]] = (
            cast(Mapping[str, object], raw_image_config)
            if isinstance(raw_image_config, Mapping)
            else MappingProxyType({})
        )
        unsupported_image_config_fields: Final = frozenset(image_config).difference(_STANDARD_IMAGE_CONFIG_FIELDS)
        if spec.resolution_location == "top_level" and unsupported_image_config_fields:
            raise UnsupportedParamsError(
                message=(
                    f"ToAPIs model={model!r} does not support imageConfig field(s): "
                    f"{', '.join(sorted(unsupported_image_config_fields))}"
                ),
                model=model,
                llm_provider="toapis",
            )
        raw_size = params.get("size")
        if (
            model in _IMAGE_25_MODELS
            and model.endswith("-vip")
            and isinstance(raw_size, str)
            and (pixels := _PIXEL_SIZE_PATTERN.fullmatch(raw_size)) is not None
            and raw_size not in _GPT_IMAGE_2_PIXEL_SPECS
        ):
            # VIP accepts custom pixel dimensions. Do not infer a resolution tier
            # for sizes outside the published ratio/resolution table.
            if any(params.get(field) is not None for field in ("resolution",)) or any(
                image_config.get(field) is not None for field in ("imageSize", "resolution")
            ):
                raise UnsupportedParamsError(
                    message="custom VIP pixel size cannot be combined with resolution",
                    model=model,
                    llm_provider="toapis",
                )
            for field, value in (
                ("aspect_ratio", params.get("aspect_ratio")),
                ("imageConfig.aspectRatio", image_config.get("aspectRatio")),
            ):
                if value is not None:
                    ratio_parts = _canonical_ratio(value, spec, field=field, model=model).split(":")
                    if int(pixels[1]) * int(ratio_parts[1]) != int(pixels[2]) * int(ratio_parts[0]):
                        raise UnsupportedParamsError(
                            message="VIP pixel size and aspect ratio must describe the same aspect ratio",
                            model=model,
                            llm_provider="toapis",
                        )
            custom_mapped: dict[str, object] = {"size": raw_size, "response_format": "url"}
            if count is not None:
                custom_mapped["n"] = 1
            return self._map_image_25_fields(custom_mapped, params, model)
        raw_ratio_candidates: Final = (
            ("size", params.get("size")),
            ("aspect_ratio", params.get("aspect_ratio")),
            ("imageConfig.aspectRatio", image_config.get("aspectRatio")),
        )
        canonical_ratios: Final = tuple(
            _canonical_ratio(value, spec, field=field, model=model)
            for field, value in raw_ratio_candidates
            if value is not None
        )
        if len(frozenset(canonical_ratios)) > 1:
            raise UnsupportedParamsError(
                message="size, aspect_ratio, and imageConfig.aspectRatio must describe the same aspect ratio",
                model=model,
                llm_provider="toapis",
            )
        ratio: Final = canonical_ratios[0] if canonical_ratios else "1:1"

        raw_resolution_candidates: Final = (
            params.get("resolution"),
            image_config.get("imageSize"),
            image_config.get("resolution"),
            *(
                pixel_spec[1]
                for _, value in raw_ratio_candidates
                if isinstance(value, str) and (pixel_spec := _GPT_IMAGE_2_PIXEL_SPECS.get(value)) is not None
            ),
        )
        canonical_resolutions: Final = tuple(
            _canonical_resolution(value, spec, model=model) for value in raw_resolution_candidates if value is not None
        )
        if len(frozenset(canonical_resolutions)) > 1:
            raise UnsupportedParamsError(
                message=(
                    "resolution, imageConfig.imageSize, imageConfig.resolution, and pixel size "
                    "must describe the same resolution"
                ),
                model=model,
                llm_provider="toapis",
            )
        resolution: Final = canonical_resolutions[0] if canonical_resolutions else spec.resolutions[0]
        mapped: dict[str, object] = {  # mutable-ok: image parameter mapping contract requires a concrete dict
            "size": ratio,
            "response_format": "url",
        }
        if count is not None:
            mapped["n"] = 1
        if spec.resolution_location == "top_level":
            mapped["resolution"] = resolution
        else:
            metadata: dict[str, object] = {"resolution": resolution}
            metadata.update(
                {
                    key: value
                    for key, value in image_config.items()
                    if key not in ("aspectRatio", "imageSize", "resolution")
                }
            )
            mapped["metadata"] = metadata
        reference_urls: Final = _normalize_reference_urls(params.get("image_url"), model=model)
        if reference_urls:
            if spec.reference_field == "reference_images":
                mapped["reference_images"] = list(reference_urls)
            else:
                mapped["image_urls"] = [{"url": url} for url in reference_urls]
        if model in _IMAGE_25_MODELS:
            if model.endswith("-vip"):
                mapped["size"] = next(
                    pixels
                    for pixels, (pixel_ratio, pixel_resolution) in _GPT_IMAGE_2_PIXEL_SPECS.items()
                    if pixel_ratio == ratio and pixel_resolution == resolution.lower()
                )
                mapped.pop("resolution", None)
            return self._map_image_25_fields(mapped, params, model)
        return mapped

    @staticmethod
    def _map_image_25_fields(
        mapped: dict[str, object], params: Mapping[str, object], model: str
    ) -> dict[str, object]:
        quality = params.get("quality") or "high"
        if model.endswith("-vip") and quality not in ("low", "medium", "high", "xhigh", "max"):
            raise UnsupportedParamsError(
                message="VIP quality must be low, medium, high, xhigh, or max",
                model=model,
                llm_provider="toapis",
            )
        # Ordinary models always use high, including when callers request low.
        mapped["quality"] = quality if model.endswith("-vip") else "high"
        background = params.get("background")
        if background is not None:
            if background != "transparent":
                raise UnsupportedParamsError(
                    message="ToAPIs image background supports 'transparent' only; omit for normal generation",
                    model=model,
                    llm_provider="toapis",
                )
            mapped["background"] = background
        references = _normalize_reference_urls(params.get("image_url"), model=model)
        if references:
            mapped["reference_images"] = list(references)
        return mapped

    def get_complete_url(
        self,
        api_base: str | None,
        api_key: str | None,
        model: str,
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        stream: bool | None = None,
    ) -> str:
        return build_toapis_endpoint(api_base, "/v1/images/generations")

    def get_status_url(self, api_base: str, task_id: str) -> str:
        encoded_task_id: Final = encode_url_path_segment(task_id, field_name="task_id")
        return f"{api_base.rstrip('/')}/{encoded_task_id}"

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
        resolved_api_key: Final = get_toapis_api_key(api_key)
        if resolved_api_key is None:
            raise ValueError("TOAPIS_API_KEY is required")
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
        return {  # mutable-ok: image HTTP handler requires a concrete JSON dict
            "model": model,
            "prompt": prompt,
            **{  # mutable-ok: JSON payload filtering requires a concrete dict for expansion
                key: value for key, value in optional_params.items() if key not in ("extra_body", "extra_headers")
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
        task: Final = parse_toapis_task(raw_response)
        if task.status != "completed" or task.result is None:
            raise BaseLLMException(
                status_code=500,
                message=f"ToAPIs image task ended without a result: {task.status}",
                headers=raw_response.headers,
            )
        provider_fields: Final[dict[str, object]] = {  # mutable-ok: ImageObject requires provider fields as a dict
            key: value
            for key, value in (
                ("task_id", task.id),
                ("expires_at", task.expires_at),
            )
            if value is not None
        }
        return ImageResponse(
            created=task.created_at,
            data=[  # mutable-ok: ImageResponse requires image data as a list
                ImageObject(url=item.url, provider_specific_fields=provider_fields) for item in task.result.data
            ],
            hidden_params={  # mutable-ok: ImageResponse requires mutable hidden params
                "model": task.model or model,
                "task_id": task.id,
            },
        )
