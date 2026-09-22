"""Keep video reads with the provider/model encoded in the task ID."""

from collections.abc import Mapping
from typing import Final

from pydantic import BaseModel, ConfigDict

from litellm.exceptions import NotFoundError
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.types.router import DeploymentTypedDict
from litellm.types.videos.utils import decode_video_id_with_provider

_READ_CALLS: Final = frozenset({"video_status", "avideo_status", "video_content", "avideo_content"})


class _Params(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    model: str
    custom_llm_provider: str | None = None


class _Info(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    id: str | None = None


class _Owner(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)
    model_name: str
    litellm_params: _Params
    model_info: _Info | None = None


def filter_video_task_deployments(
    model: str,
    deployments: list[DeploymentTypedDict] | DeploymentTypedDict,
    request_kwargs: Mapping[str, object] | None,
) -> list[DeploymentTypedDict] | DeploymentTypedDict:
    if not request_kwargs or request_kwargs.get("_router_call_type") not in _READ_CALLS:
        return deployments
    video_id = request_kwargs.get("video_id")
    if not isinstance(video_id, str):
        return deployments
    decoded = decode_video_id_with_provider(video_id)
    provider = decoded.get("custom_llm_provider")
    if not provider:
        return deployments
    encoded_model = decoded.get("model_id")
    specific = isinstance(deployments, dict)
    candidates = [deployments] if specific else deployments
    matched: list[DeploymentTypedDict] = []
    for deployment in candidates:
        owner = _Owner.model_validate(deployment)
        physical_model, physical_provider, _, _ = get_llm_provider(
            model=owner.litellm_params.model,
            custom_llm_provider=owner.litellm_params.custom_llm_provider,
        )
        compatible_ids = {
            model,
            owner.model_name,
            physical_model,
            owner.litellm_params.model,
            owner.model_info.id if owner.model_info else None,
        }
        if physical_provider == provider and (not encoded_model or encoded_model in compatible_ids):
            matched.append(deployment)
    if not matched:
        raise NotFoundError(
            message="No deployment in this model group matches the video's original provider and model",
            model=model,
            llm_provider=provider,
        )
    return matched[0] if specific else matched
