from collections.abc import Mapping
from typing import Final

import httpx

from litellm.exceptions import UnsupportedParamsError
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.submission_utils import mark_submission_outcome
from litellm.llms.openai.image_edit.transformation import OpenAIImageEditConfig
from litellm.types.images.main import ImageEditOptionalRequestParams
from litellm.types.utils import ImageResponse

from ..common_utils import build_zexapi_endpoint, get_zexapi_api_key, raise_for_zexapi_error
from ..image_generation.transformation import get_zexapi_image_model_sizes

_ZEXAPI_IMAGE_EDIT_MODELS: Final = frozenset(("image2", "gpt-image2"))


def get_zexapi_image_edit_config(model: str) -> "ZexAPIImageEditConfig | None":
    if model in _ZEXAPI_IMAGE_EDIT_MODELS:
        return ZexAPIImageEditConfig()
    return None


class ZexAPIImageEditConfig(OpenAIImageEditConfig):
    def map_openai_params(
        self,
        image_edit_optional_params: ImageEditOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:
        supported_sizes: Final = get_zexapi_image_model_sizes(model)
        if not supported_sizes:
            raise UnsupportedParamsError(
                message=f"image-edit does not support ZexAPI model={model!r}",
                model=model,
                llm_provider="zexapi",
            )
        size: Final = image_edit_optional_params.get("size")
        if size is not None and size not in supported_sizes:
            raise UnsupportedParamsError(
                message=f"ZexAPI model={model!r} does not support size={size!r}",
                model=model,
                llm_provider="zexapi",
            )
        return dict(image_edit_optional_params)

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        api_key: str | None = None,
        litellm_params: Mapping[str, object] | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: image edit handler may add caller-provided headers
        params_api_key: Final = litellm_params.get("api_key") if litellm_params is not None else None
        resolved_api_key: Final = get_zexapi_api_key(
            api_key or (params_api_key if isinstance(params_api_key, str) else None)
        )
        if resolved_api_key is None:
            raise ValueError("ZEXAPI_API_KEY is required")
        return {  # mutable-ok: image edit handler may add caller-provided headers
            **headers,
            "Authorization": f"Bearer {resolved_api_key}",
        }

    def get_complete_url(self, model: str, api_base: str | None, litellm_params: Mapping[str, object]) -> str:
        return build_zexapi_endpoint(api_base, "/v1/images/edits")

    def transform_image_edit_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
    ) -> ImageResponse:
        raise_for_zexapi_error(raw_response)
        try:
            return ImageResponse.model_validate_json(raw_response.text)
        except ValueError as exc:
            raise mark_submission_outcome(
                BaseLLMException(
                    status_code=raw_response.status_code,
                    message=f"Invalid ZexAPI image edit response: {exc}",
                    headers=raw_response.headers,
                ),
                "unknown",
            ) from exc
