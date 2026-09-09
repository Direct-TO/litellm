from typing import Any, Final, cast

import litellm
from litellm.llms.base_llm.videos.transformation import BaseVideoConfig
from litellm.types.videos.main import VideoCreateOptionalRequestParams
from litellm.utils import filter_out_litellm_params
from litellm.videos.contract import VIDEO_CONTRACT_FIELDS, VIDEO_NATIVE_OVERRIDE_FIELDS, require_video_contract_support


class VideoGenerationRequestUtils:
    """Helper utils for constructing video generation requests"""

    @staticmethod
    def get_optional_params_video_generation(
        model: str,
        video_generation_provider_config: BaseVideoConfig,
        video_generation_optional_params: VideoCreateOptionalRequestParams,
    ) -> dict:
        """
        Get optional parameters for the video generation API.

        Args:
            model: The model name
            video_generation_provider_config: The provider configuration for video generation API
            video_generation_optional_params: The optional parameters for video generation

        Returns:
            A dictionary of supported parameters for the video generation API
        """
        supported_params: Final = cast(
            list[str],
            video_generation_provider_config.get_supported_openai_params(model),  # pyright: ignore[reportUnknownMemberType, reportUnknownArgumentType]  # BaseVideoConfig retains a legacy bare-list return of parameter names.
        )
        require_video_contract_support(
            video_generation_optional_params,
            supported_params,
            model,
        )
        if any(video_generation_optional_params.get(key) is not None for key in VIDEO_CONTRACT_FIELDS):
            extra = video_generation_optional_params.get("extra_body")
            if extra and VIDEO_NATIVE_OVERRIDE_FIELDS.intersection(extra):
                raise litellm.UnsupportedParamsError(
                    message="extra_body cannot override canonical video dimensions or references",
                    model=model,
                    llm_provider="",
                )
        # Map parameters to provider-specific format
        mapped_params: Final = video_generation_provider_config.map_openai_params(
            video_create_optional_params=video_generation_optional_params,
            model=model,
            drop_params=litellm.drop_params,
        )

        # Merge extra_body params if present (for provider-specific parameters)
        if "extra_body" in video_generation_optional_params:
            extra_body: Final = video_generation_optional_params["extra_body"]
            if extra_body and isinstance(extra_body, dict):
                # extra_body params override mapped params
                mapped_params.update(extra_body)
            # Remove extra_body from mapped_params since it's not sent to the API
            mapped_params.pop("extra_body", None)

        return mapped_params

    @staticmethod
    def get_requested_video_generation_optional_param(
        params: dict[str, Any],
    ) -> VideoCreateOptionalRequestParams:
        """
        Filter parameters to only include those defined in VideoCreateOptionalRequestParams.

        Args:
            params: Dictionary of parameters to filter

        Returns:
            VideoCreateOptionalRequestParams instance with only the valid parameters
        """
        params = dict(params or {})

        raw_kwargs = params.get("kwargs", {})
        if not isinstance(raw_kwargs, dict):
            raw_kwargs = {}

        kwargs_extra_body: Final = raw_kwargs.pop("extra_body", None)
        top_level_extra_body: Final = params.get("extra_body")

        base_params_raw: Final = {
            key: value
            for key, value in params.items()
            if key not in {"kwargs", "extra_body", "prompt", "model"} and value is not None
        }
        base_params: Final = filter_out_litellm_params(kwargs=base_params_raw)

        cleaned_kwargs: Final = filter_out_litellm_params(kwargs={k: v for k, v in raw_kwargs.items() if v is not None})

        optional_params: Final[dict[str, Any]] = {
            **base_params,
            **cleaned_kwargs,
        }
        # Native extra_body values are not declarations of the gateway input contract.
        explicit_contract_fields = {key for key in VIDEO_CONTRACT_FIELDS if optional_params.get(key) is not None}

        merged_extra_body: dict[str, Any] = {}
        for extra_body_candidate in (top_level_extra_body, kwargs_extra_body):
            if isinstance(extra_body_candidate, dict):
                for key, value in extra_body_candidate.items():
                    if value is not None:
                        merged_extra_body[key] = value

        if merged_extra_body:
            merged_extra_body = filter_out_litellm_params(kwargs=merged_extra_body)
            if merged_extra_body:
                optional_params["extra_body"] = merged_extra_body
                optional_params.update(merged_extra_body)
                for key in VIDEO_CONTRACT_FIELDS - explicit_contract_fields:
                    optional_params.pop(key, None)

        optional_params.pop("timeout", None)

        return cast(VideoCreateOptionalRequestParams, optional_params)
