from collections.abc import Mapping, Sequence
from typing import Final

import httpx

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.image_generation.transformation import BaseImageGenerationConfig
from litellm.types.llms.openai import AllMessageValues, OpenAIImageGenerationOptionalParams
from litellm.types.utils import ImageResponse

from ..common_utils import build_zexapi_endpoint, get_zexapi_api_key, raise_for_zexapi_error

_SUPPORTED_PARAMS: Final[tuple[OpenAIImageGenerationOptionalParams, ...]] = (
    "response_format",
    "size",
)


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
        return {  # mutable-ok: image parameter mapping contract requires a concrete dict
            **optional_params,
            **non_default_params,
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
            return ImageResponse.model_validate_json(raw_response.text)
        except ValueError as exc:
            raise BaseLLMException(
                status_code=raw_response.status_code,
                message=f"Invalid ZexAPI image response: {exc}",
                headers=raw_response.headers,
            ) from exc
