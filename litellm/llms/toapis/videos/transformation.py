from collections.abc import Mapping
from io import BufferedReader, BytesIO
from itertools import chain
from math import gcd
from types import MappingProxyType
from typing import Final, Literal, NamedTuple, TypeAlias, cast

import httpx
from httpx._types import RequestFiles
from pydantic import BaseModel, ConfigDict, TypeAdapter, ValidationError

from litellm.exceptions import UnsupportedParamsError
from litellm.images.utils import ImageEditRequestUtils
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.videos.transformation import normalize_video_task_result
from litellm.llms.openai.videos.transformation import OpenAIVideoConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject
from litellm.types.videos.utils import encode_video_id_with_provider
from litellm.videos.contract import VIDEO_CONTRACT_FIELDS

from ..common_utils import (
    ToAPISTaskResponse,
    build_toapis_endpoint,
    get_toapis_api_key,
    parse_toapis_image_upload,
    parse_toapis_task,
    parse_toapis_video_create_task,
)
from .gateway_contract import GATEWAY_VIDEO_MODELS, map_gateway_video

_ValidatedFileContent: TypeAlias = bytes | str | BytesIO | BufferedReader


class _InputReference(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    input_reference: _ValidatedFileContent


_INPUT_REFERENCE_ADAPTER: Final = TypeAdapter(_InputReference)
_REFERENCE_FORMAT_MARKER: Final = "_toapis_reference_format"


class _VideoModelSpec(NamedTuple):
    size_field: Literal["aspect_ratio", "ratio", "size"]
    reference_format: Literal[
        "image_with_roles",
        "image_urls",
        "reference_images",
        "image",
        "metadata_image_list",
    ]


_VIDEO_MODEL_SPECS: Final[Mapping[str, _VideoModelSpec]] = MappingProxyType(
    {
        "seedance-2": _VideoModelSpec("aspect_ratio", "image_with_roles"),
        "seedance-2-fast": _VideoModelSpec("aspect_ratio", "image_with_roles"),
        "seedance-2-mini": _VideoModelSpec("aspect_ratio", "image_with_roles"),
        "seedance-2-5": _VideoModelSpec("aspect_ratio", "image_with_roles"),
        "wan3.0-video": _VideoModelSpec("ratio", "image_with_roles"),
        "happyhorse-1.1": _VideoModelSpec("aspect_ratio", "image_urls"),
        "MiniMax-H3": _VideoModelSpec("aspect_ratio", "image_with_roles"),
        "kling-v2-6": _VideoModelSpec("aspect_ratio", "reference_images"),
        "kling-v3": _VideoModelSpec("aspect_ratio", "reference_images"),
        "kling-3.0-turbo": _VideoModelSpec("aspect_ratio", "reference_images"),
        "kling-v3-omni": _VideoModelSpec("aspect_ratio", "metadata_image_list"),
        "kling-video-o1": _VideoModelSpec("aspect_ratio", "metadata_image_list"),
        "grok-video-1.0": _VideoModelSpec("aspect_ratio", "image"),
        "grok-video-1.5": _VideoModelSpec("aspect_ratio", "image"),
        "gemini-omni-flash": _VideoModelSpec("aspect_ratio", "image_urls"),
        "gemini-omni-flash-preview-official": _VideoModelSpec("aspect_ratio", "image_urls"),
        "veo3.1-fast": _VideoModelSpec("aspect_ratio", "image_urls"),
        "veo3.1-quality": _VideoModelSpec("aspect_ratio", "image_urls"),
        "veo3.1-lite": _VideoModelSpec("aspect_ratio", "image_urls"),
        "Veo3.1-fast-official": _VideoModelSpec("size", "image_urls"),
        "Veo3.1-quality-official": _VideoModelSpec("size", "image_urls"),
        "Veo3.1-lite-official": _VideoModelSpec("size", "image_urls"),
    }
)


def _rewind_file_content(value: _ValidatedFileContent) -> _ValidatedFileContent:
    if isinstance(value, (bytes, str)):
        return value
    value.seek(0)
    return value


class ToAPISVideoConfig(OpenAIVideoConfig):
    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: BaseVideoConfig requires a list
        return [  # mutable-ok: BaseVideoConfig requires a concrete list
            "extra_body",
            "image",
            "model",
            "parameters",
            "prompt",
            "seconds",
            "size",
            "input_reference",
            "extra_headers",
        ] + (["resolution", "aspect_ratio", "references"] if model in GATEWAY_VIDEO_MODELS else [])

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: video request utility updates and removes extra_body
        spec: Final = _VIDEO_MODEL_SPECS.get(model)
        if spec is None:
            raise UnsupportedParamsError(
                message=f"video-generation does not support ToAPIs model={model!r}",
                model=model,
                llm_provider="toapis",
            )
        if any(video_create_optional_params.get(key) is not None for key in VIDEO_CONTRACT_FIELDS):
            return map_gateway_video(model, video_create_optional_params, spec.size_field)
        mapped_duration: Final = self._duration(video_create_optional_params.get("seconds"))
        raw_size: Final = video_create_optional_params.get("size")
        mapped_size: Final = raw_size if spec.size_field == "size" else self._aspect_ratio(raw_size)
        forwarded_params: Final = tuple(
            (key, value) for key, value in video_create_optional_params.items() if key not in ("seconds", "size")
        )
        duration_params: Final = (("duration", mapped_duration),) if mapped_duration is not None else ()
        size_params: Final = ((spec.size_field, mapped_size),) if mapped_size is not None else ()
        reference_marker: Final = (
            ((_REFERENCE_FORMAT_MARKER, spec.reference_format),)
            if video_create_optional_params.get("input_reference") is not None
            else ()
        )
        mapped: dict[str, object] = dict(  # mutable-ok: video request utility updates and removes extra_body
            chain(forwarded_params, duration_params, size_params, reference_marker)
        )
        if (
            model == "grok-video-1.5"
            and mapped.get("input_reference") is None
            and not isinstance(mapped.get("image"), str)
        ):
            raise UnsupportedParamsError(
                message="grok-video-1.5 requires exactly one public image URL or input_reference",
                model=model,
                llm_provider="toapis",
            )
        return mapped

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> dict[str, str]:  # mutable-ok: video handler may add caller-provided headers
        params_api_key: Final = litellm_params.api_key if litellm_params is not None else None
        resolved_api_key: Final = get_toapis_api_key(api_key or params_api_key)
        if resolved_api_key is None:
            raise ValueError("TOAPIS_API_KEY is required")
        return {  # mutable-ok: video handler may add caller-provided headers
            **headers,
            "Authorization": f"Bearer {resolved_api_key}",
            "Content-Type": "application/json",
        }

    def get_complete_url(self, model: str, api_base: str | None, litellm_params: Mapping[str, object]) -> str:
        return build_toapis_endpoint(api_base, "/v1/videos/generations")

    def use_multipart_form_data(self) -> bool:
        return False

    def get_video_create_input_reference_upload_request(
        self,
        video_create_optional_request_params: Mapping[str, object],
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
    ) -> tuple[str, Mapping[str, str], Mapping[str, object], RequestFiles] | None:
        raw_input_reference: Final = video_create_optional_request_params.get("input_reference")
        if raw_input_reference is None:
            return None
        try:
            input_reference: Final = _INPUT_REFERENCE_ADAPTER.validate_python(
                MappingProxyType({"input_reference": raw_input_reference})
            ).input_reference
        except ValidationError as exc:
            raise TypeError("input_reference must be bytes or a binary file") from exc
        if (
            video_create_optional_request_params.get("image_urls") is not None
            or video_create_optional_request_params.get("image_with_roles") is not None
            or video_create_optional_request_params.get("reference_images") is not None
            or video_create_optional_request_params.get("image") is not None
        ):
            raise ValueError("ToAPIs video generation cannot combine input_reference with image URL fields")
        metadata: Final = video_create_optional_request_params.get("metadata")
        if isinstance(metadata, Mapping) and cast(Mapping[str, object], metadata).get("image_list") is not None:
            raise ValueError("ToAPIs video generation cannot combine input_reference with metadata.image_list")

        content_type: Final = ImageEditRequestUtils.get_image_content_type(input_reference)
        if not content_type.startswith("image/"):
            raise ValueError(
                "ToAPIs local input_reference currently supports images only; upload video/audio separately and pass its URL"
            )
        upload_content: Final = _rewind_file_content(input_reference)
        raw_filename: Final = getattr(upload_content, "name", None)
        filename: Final = raw_filename if isinstance(raw_filename, str) else "input_reference.png"
        upload_headers: Final = MappingProxyType(
            {key: value for key, value in headers.items() if key.lower() != "content-type"}
        )
        upload_files: Final[RequestFiles] = (("file", (filename, upload_content, content_type)),)
        return (
            build_toapis_endpoint(litellm_params.api_base, "/v1/uploads/images"),
            upload_headers,
            MappingProxyType({"purpose": "generation"}),
            upload_files,
        )

    def transform_video_create_input_reference_upload_response(
        self,
        raw_response: httpx.Response,
        video_create_optional_request_params: Mapping[str, object],
    ) -> dict[str, object]:  # mutable-ok: video adapters require a mutable request dictionary
        upload_data: Final = parse_toapis_image_upload(raw_response)
        reference_format: Final = video_create_optional_request_params.get(_REFERENCE_FORMAT_MARKER)
        if reference_format not in (
            "image_with_roles",
            "image_urls",
            "reference_images",
            "image",
            "metadata_image_list",
        ):
            raise ValueError("Missing ToAPIs input_reference model capability marker")
        excluded_forwarded_params = {"input_reference", _REFERENCE_FORMAT_MARKER}
        if reference_format == "metadata_image_list":
            excluded_forwarded_params.add("metadata")
        forwarded_params: Final = tuple(
            (key, value)
            for key, value in video_create_optional_request_params.items()
            if key not in excluded_forwarded_params
        )
        reference_params: tuple[tuple[str, object], ...]
        if reference_format == "image_with_roles":
            reference_params = (("image_with_roles", [{"url": upload_data.url, "role": "reference_image"}]),)
        elif reference_format == "image_urls":
            reference_params = (("image_urls", [upload_data.url]),)
        elif reference_format == "reference_images":
            reference_params = (("reference_images", [upload_data.url]),)
        elif reference_format == "image":
            reference_params = (("image", upload_data.url),)
        else:
            raw_metadata: Final = video_create_optional_request_params.get("metadata")
            merged_metadata: dict[str, object] = (
                dict(cast(Mapping[str, object], raw_metadata)) if isinstance(raw_metadata, Mapping) else {}
            )
            merged_metadata["image_list"] = [{"image_url": upload_data.url}]
            reference_params = (("metadata", merged_metadata),)
        return dict(  # mutable-ok: video adapters require a mutable request dictionary
            chain(forwarded_params, reference_params)
        )

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: Mapping[str, object],
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
    ) -> tuple[dict[str, object], RequestFiles, str]:  # mutable-ok: video HTTP handler requires a dict
        if video_create_optional_request_params.get("input_reference") is not None:
            raise ValueError("ToAPIs input_reference must be uploaded before creating a video task")
        forwarded_params: Final = tuple(
            (key, value)
            for key, value in video_create_optional_request_params.items()
            if key not in ("extra_headers", "model", "prompt", _REFERENCE_FORMAT_MARKER)
        )
        request_data: Final = dict(  # mutable-ok: video HTTP handler requires a concrete JSON dict
            chain((("model", model), ("prompt", prompt)), forwarded_params)
        )
        return request_data, (), api_base

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
        request_data: Mapping[str, object] | None = None,
    ) -> VideoObject:
        return self._transform_task(parse_toapis_video_create_task(raw_response), custom_llm_provider, model)

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        return self._transform_task(parse_toapis_task(raw_response), custom_llm_provider, None)

    @staticmethod
    def _duration(seconds: str | None) -> int | str | None:
        if seconds is None:
            return None
        try:
            return int(seconds)
        except ValueError:
            return seconds

    @staticmethod
    def _aspect_ratio(size: str | None) -> str | None:
        if size is None or ":" in size:
            return size
        dimensions: Final = size.lower().split("x")
        if len(dimensions) != 2 or not all(dimension.isdigit() for dimension in dimensions):
            return size
        width, height = (int(dimension) for dimension in dimensions)
        if width <= 0 or height <= 0:
            return size
        divisor: Final = gcd(width, height)
        return f"{width // divisor}:{height // divisor}"

    @staticmethod
    def _transform_task(
        task: ToAPISTaskResponse,
        custom_llm_provider: str | None,
        request_model: str | None,
    ) -> VideoObject:
        result_item: Final = task.result.data[0] if task.result is not None and task.result.data else None
        provider_error: Final[dict[str, object] | None] = (  # mutable-ok: VideoObject error contract requires a dict
            {  # mutable-ok: VideoObject error contract requires a dict
                "code": task.error.code,
                "message": task.error.message,
            }
            if task.error is not None
            else None
        )
        normalized_status, output_url, error = normalize_video_task_result(
            status=task.status,
            output_url=result_item.url if result_item is not None else None,
            error=provider_error,
            require_absolute_output_url=True,
        )
        task_id: Final = (
            encode_video_id_with_provider(task.id, custom_llm_provider, request_model)
            if custom_llm_provider is not None
            else task.id
        )
        return VideoObject(
            id=task_id,
            object="video",
            status=normalized_status,
            created_at=task.created_at,
            completed_at=task.completed_at,
            expires_at=task.expires_at,
            error=error,
            progress=task.progress,
            model=task.model or request_model,
            output_url=output_url,
        )
