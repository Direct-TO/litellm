from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final

import httpx

from litellm.exceptions import UnsupportedParamsError
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.url_utils import encode_url_path_segment
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.image_generation.transformation import BaseImageGenerationConfig
from litellm.types.llms.openai import AllMessageValues, OpenAIImageGenerationOptionalParams
from litellm.types.utils import ImageObject, ImageResponse

from ..common_utils import build_toapis_endpoint, get_toapis_api_key, parse_toapis_task

_SUPPORTED_PARAMS: Final[tuple[OpenAIImageGenerationOptionalParams, ...]] = (
    "response_format",
    "size",
)
_ZEXAPI_IMAGE2_SIZE_TO_RATIO: Final[Mapping[str, str]] = MappingProxyType(
    {
        "1024x1024": "1:1",
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
_TOAPIS_IMAGE2_RATIOS: Final[frozenset[str]] = frozenset(
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


class ToAPISImageGenerationConfig(BaseImageGenerationConfig):
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
        if model != "gpt-image-2":
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
        size: Final = params.get("size", "1:1")
        ratio_alias: Final[str | None] = _ZEXAPI_IMAGE2_SIZE_TO_RATIO.get(size) if isinstance(size, str) else None
        ratio: Final[str | None] = size if isinstance(size, str) and size in _TOAPIS_IMAGE2_RATIOS else ratio_alias
        if ratio is None:
            raise UnsupportedParamsError(
                message=f"image-generation does not support size={size!r}",
                model=model,
                llm_provider="toapis",
            )
        return {  # mutable-ok: image parameter mapping contract requires a concrete dict
            "size": ratio,
            "resolution": "1k",
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
