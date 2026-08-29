from collections.abc import Mapping
from io import BufferedReader, BytesIO
from itertools import chain
from math import gcd
from types import MappingProxyType
from typing import Final, TypeAlias

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

from ..common_utils import (
    build_toapis_endpoint,
    get_toapis_api_key,
    parse_toapis_image_upload,
    parse_toapis_task,
)

_ValidatedFileContent: TypeAlias = bytes | str | BytesIO | BufferedReader


class _InputReference(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True, frozen=True)

    input_reference: _ValidatedFileContent


_INPUT_REFERENCE_ADAPTER: Final = TypeAdapter(_InputReference)


def _rewind_file_content(value: _ValidatedFileContent) -> _ValidatedFileContent:
    if isinstance(value, (bytes, str)):
        return value
    value.seek(0)
    return value


class ToAPISVideoConfig(OpenAIVideoConfig):
    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: BaseVideoConfig requires a list
        return [  # mutable-ok: BaseVideoConfig requires a concrete list
            "model",
            "prompt",
            "seconds",
            "size",
            "input_reference",
            "extra_headers",
        ]

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: video request utility updates and removes extra_body
        if model != "seedance-2-5":
            raise UnsupportedParamsError(
                message=f"video-generation does not support ToAPIs model={model!r}",
                model=model,
                llm_provider="toapis",
            )
        mapped_duration: Final = self._duration(video_create_optional_params.get("seconds"))
        raw_size: Final = video_create_optional_params.get("size")
        mapped_aspect_ratio: Final = self._aspect_ratio(raw_size)
        forwarded_params: Final = tuple(
            (key, value) for key, value in video_create_optional_params.items() if key not in ("seconds", "size")
        )
        duration_params: Final = (("duration", mapped_duration),) if mapped_duration is not None else ()
        aspect_ratio_params: Final = (("aspect_ratio", mapped_aspect_ratio),) if mapped_aspect_ratio is not None else ()
        return dict(  # mutable-ok: video request utility updates and removes extra_body
            chain(forwarded_params, duration_params, aspect_ratio_params)
        )

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
        ):
            raise ValueError("ToAPIs video generation cannot combine input_reference with image URL fields")

        content_type: Final = ImageEditRequestUtils.get_image_content_type(input_reference)
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
        forwarded_params: Final = tuple(
            (key, value) for key, value in video_create_optional_request_params.items() if key != "input_reference"
        )
        return dict(  # mutable-ok: video adapters require a mutable request dictionary
            chain(
                forwarded_params,
                (
                    (
                        "image_with_roles",
                        (
                            {  # mutable-ok: provider JSON requires an object entry
                                "url": upload_data.url,
                                "role": "reference_image",
                            },
                        ),
                    ),
                ),
            )
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
            if key not in ("extra_headers", "model", "prompt")
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
        return self._transform_task_response(raw_response, custom_llm_provider, model)

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        return self._transform_task_response(raw_response, custom_llm_provider, None)

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
    def _transform_task_response(
        raw_response: httpx.Response,
        custom_llm_provider: str | None,
        request_model: str | None,
    ) -> VideoObject:
        task: Final = parse_toapis_task(raw_response)
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
