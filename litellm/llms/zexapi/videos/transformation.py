import re
from collections.abc import Mapping
from decimal import Decimal, InvalidOperation
from itertools import chain
from typing import Final

import httpx
from httpx._types import RequestFiles

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.videos.transformation import normalize_video_task_result
from litellm.llms.openai.videos.transformation import OpenAIVideoConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject
from litellm.types.videos.utils import encode_video_id_with_provider

from ..common_utils import build_zexapi_endpoint, get_zexapi_api_key, parse_zexapi_task

_SUPPORTED_PARAMS: Final[frozenset[str]] = frozenset(
    (
        "aspect_ratio",
        "extra_body",
        "extra_headers",
        "images",
        "input_reference",
        "seconds",
        "size",
    )
)


class ZexAPIVideoConfig(OpenAIVideoConfig):
    def get_supported_openai_params(self, model: str) -> list[str]:  # mutable-ok: BaseVideoConfig requires a list
        return sorted(_SUPPORTED_PARAMS)

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:  # mutable-ok: video request utility updates and removes extra_body
        unsupported: Final = tuple(key for key in video_create_optional_params if key not in _SUPPORTED_PARAMS)
        if unsupported and not drop_params:
            raise ValueError(f"ZexAPI video generation does not support: {', '.join(unsupported)}")
        self._validate_fixed_duration(model=model, seconds=video_create_optional_params.get("seconds"))
        return dict(  # mutable-ok: video request utility updates and removes extra_body
            (key, value)
            for key, value in video_create_optional_params.items()
            if key in _SUPPORTED_PARAMS and key not in ("extra_headers", "seconds")
        )

    @staticmethod
    def _validate_fixed_duration(model: str, seconds: object) -> None:
        if seconds is None:
            return
        duration_match: Final = re.search(r"(?:^|[-_])(\d+)s(?:$|[-_])", model)
        documented_seconds: Final[int | None] = (
            8 if model.startswith("veo_3_1") else (10 if model.startswith("omni_flash-10s") else None)
        )
        if duration_match is None and documented_seconds is None:
            raise ValueError(f"ZexAPI model={model!r} does not declare a fixed duration")
        try:
            requested_seconds: Final = Decimal(str(seconds))
        except InvalidOperation as exc:
            raise ValueError(f"Invalid video duration: {seconds!r}") from exc
        if duration_match is not None:
            raw_model_seconds: str | int = duration_match.group(1)
        else:
            assert documented_seconds is not None
            raw_model_seconds = documented_seconds
        model_seconds: Final = Decimal(raw_model_seconds)
        if requested_seconds != model_seconds:
            raise ValueError(
                f"ZexAPI model={model!r} generates {model_seconds} seconds, not {requested_seconds} seconds"
            )

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> dict[str, str]:  # mutable-ok: video handler may add caller-provided headers
        params_api_key: Final = litellm_params.api_key if litellm_params is not None else None
        resolved_api_key: Final = get_zexapi_api_key(api_key or params_api_key)
        if resolved_api_key is None:
            raise ValueError("ZEXAPI_API_KEY is required")
        return {  # mutable-ok: video handler may add caller-provided headers
            **headers,
            "Authorization": f"Bearer {resolved_api_key}",
        }

    def get_complete_url(self, model: str, api_base: str | None, litellm_params: Mapping[str, object]) -> str:
        return build_zexapi_endpoint(api_base, "/v1/videos")

    def use_multipart_form_data(self) -> bool:
        return False

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
            return super().transform_video_create_request(  # pyright: ignore[reportUnknownMemberType, reportUnknownVariableType]  # OpenAI base still uses unparameterized dicts
                model=model,
                prompt=prompt,
                api_base=api_base,
                video_create_optional_request_params=dict(  # mutable-ok: OpenAI multipart adapter requires a concrete dict
                    video_create_optional_request_params
                ),
                litellm_params=litellm_params,
                headers=dict(headers),  # mutable-ok: OpenAI multipart adapter requires concrete headers
            )
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
    def _transform_task_response(
        raw_response: httpx.Response,
        custom_llm_provider: str | None,
        request_model: str | None,
    ) -> VideoObject:
        task: Final = parse_zexapi_task(raw_response)
        if task.object != "video":
            raise BaseLLMException(
                status_code=400,
                message="ZexAPI returned an image task from the video endpoint; use image_generation for images",
                headers=raw_response.headers,
            )
        task_id: Final = (
            encode_video_id_with_provider(task.id, custom_llm_provider, request_model)
            if custom_llm_provider is not None
            else task.id
        )
        provider_status: Final = "in_progress" if task.status == "processing" else task.status
        provider_error: Final[dict[str, object] | None] = (  # mutable-ok: VideoObject error contract requires a dict
            {  # mutable-ok: VideoObject error contract requires a dict
                "code": task.error.code,
                "message": task.error.message,
            }
            if task.error is not None and not isinstance(task.error, str)
            else (
                {  # mutable-ok: VideoObject error contract requires a dict
                    "code": "generation_failed",
                    "message": task.error,
                }
                if isinstance(task.error, str)
                else None
            )
        )
        provider_output_url: Final = task.url or task.video_url
        normalized_status, output_url, error = normalize_video_task_result(
            status=provider_status,
            output_url=provider_output_url,
            error=provider_error,
            require_absolute_output_url=True,
        )
        return VideoObject(
            id=task_id,
            object="video",
            status=normalized_status,
            created_at=task.created_at if task.created_at is not None else task.created,
            completed_at=task.completed_at,
            error=error,
            progress=task.progress,
            size=task.size,
            model=task.model or request_model,
            output_url=output_url,
        )
