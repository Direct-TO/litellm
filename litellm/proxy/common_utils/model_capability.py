from collections.abc import Callable, Mapping
from types import MappingProxyType
from typing import Final

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter, ValidationError

import litellm
from litellm._logging import verbose_proxy_logger
from litellm.router import Router
from litellm.types.proxy.model_listing import ModelCapability
from litellm.types.router import DeploymentTypedDict
from litellm.types.utils import ModelInfo

MODEL_LIST_GENERATION_ONLY_SETTING: Final = "model_list_generation_only"

_CAPABILITY_BY_MODE: Final[Mapping[str, ModelCapability]] = MappingProxyType(
    {
        "chat": "text",
        "completion": "text",
        "image_generation": "image",
        "video_generation": "video",
        "audio_speech": "audio",
    }
)


class _CapabilityModelInfo(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    mode: str | None = None
    base_model: str | None = None


class _CapabilityLiteLLMParams(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    model: str
    base_model: str | None = None


class _CapabilityDeployment(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    litellm_params: _CapabilityLiteLLMParams
    model_info: _CapabilityModelInfo = Field(default_factory=_CapabilityModelInfo)


_CAPABILITY_DEPLOYMENT_ADAPTER: Final = TypeAdapter(_CapabilityDeployment)


def generation_only_model_listing_enabled(general_settings: Mapping[str, object]) -> bool:
    return general_settings.get(MODEL_LIST_GENERATION_ONLY_SETTING, False) is True


def _capability_from_mode(mode: object) -> ModelCapability | None:
    return _CAPABILITY_BY_MODE.get(mode) if isinstance(mode, str) else None


def _deployment_capability(
    deployment: DeploymentTypedDict,
    model_name: str,
    get_model_info: Callable[[str], ModelInfo | None],
) -> ModelCapability | None:
    try:
        validated_deployment: Final = _CAPABILITY_DEPLOYMENT_ADAPTER.validate_python(deployment)
    except ValidationError as exc:
        verbose_proxy_logger.debug(
            "resolve_model_capability: invalid deployment for %s (%s)",
            model_name,
            type(exc).__name__,
        )
        return None
    if validated_deployment.model_info.mode is not None:
        return _capability_from_mode(validated_deployment.model_info.mode)

    configured_backend: Final = (
        validated_deployment.model_info.base_model
        or validated_deployment.litellm_params.base_model
        or validated_deployment.litellm_params.model
    )
    lookup_model: Final = model_name if "*" in configured_backend else configured_backend
    try:
        resolved_model_info: Final = get_model_info(lookup_model)
    except Exception as exc:  # noqa: BLE001  # model metadata lookup raises plain provider exceptions
        verbose_proxy_logger.debug(
            "resolve_model_capability: backend lookup failed for %s: %s",
            model_name,
            exc,
        )
        return None

    return _capability_from_mode(resolved_model_info.get("mode")) if resolved_model_info is not None else None


def resolve_model_capability(
    model_name: str,
    llm_router: Router | None,
    get_model_info: Callable[[str], ModelInfo | None] = litellm.get_model_info,
) -> ModelCapability | None:
    if llm_router is not None:
        try:
            deployments: Final[tuple[DeploymentTypedDict, ...]] = tuple(
                llm_router.get_model_list(model_name=model_name) or ()
            )
        except Exception as exc:  # noqa: BLE001  # router discovery can surface provider configuration exceptions
            verbose_proxy_logger.debug(
                "resolve_model_capability: router lookup failed for %s: %s",
                model_name,
                exc,
            )
            return None
        if deployments:
            capabilities: Final[tuple[ModelCapability | None, ...]] = tuple(
                _deployment_capability(
                    deployment=deployment,
                    model_name=model_name,
                    get_model_info=get_model_info,
                )
                for deployment in deployments
            )
            first_capability: Final[ModelCapability | None] = capabilities[0]
            return (
                first_capability
                if first_capability is not None and all(capability == first_capability for capability in capabilities)
                else None
            )

    try:
        model_info: Final = get_model_info(model_name)
    except Exception as exc:  # noqa: BLE001  # model metadata lookup raises plain provider exceptions
        verbose_proxy_logger.debug(
            "resolve_model_capability: cost map lookup failed for %s: %s",
            model_name,
            exc,
        )
        return None

    return _capability_from_mode(model_info.get("mode")) if model_info is not None else None
