from typing import Final, Literal

import httpx
from pydantic import BaseModel, ConfigDict

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.secret_managers.main import get_secret_str

DEFAULT_API_BASE: Final = "https://zexapi.com/v1"


class ZexAPITaskError(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    code: str | int = "generation_failed"
    message: str


class ZexAPITaskResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str
    object: Literal["video", "image"]
    model: str | None = None
    status: Literal["queued", "processing", "in_progress", "completed", "failed"]
    progress: int = 0
    created: int | None = None
    created_at: int | None = None
    completed_at: int | None = None
    size: str | None = None
    url: str | None = None
    video_url: str | None = None
    error: ZexAPITaskError | str | None = None


def get_zexapi_api_base(api_base: str | None = None) -> str:
    return api_base or get_secret_str("ZEXAPI_API_BASE") or DEFAULT_API_BASE


def get_zexapi_api_key(api_key: str | None = None) -> str | None:
    global_api_key: Final = litellm.api_key if isinstance(litellm.api_key, str) else None
    return api_key or global_api_key or get_secret_str("ZEXAPI_API_KEY")


def build_zexapi_endpoint(api_base: str | None, endpoint: str) -> str:
    resolved_base: Final = httpx.URL(get_zexapi_api_base(api_base))
    if resolved_base.query or resolved_base.fragment:
        raise ValueError("ZEXAPI_API_BASE must not contain a query string or fragment")
    normalized_endpoint: Final = endpoint if endpoint.startswith("/v1/") else f"/v1/{endpoint.lstrip('/')}"
    base_path: Final = resolved_base.path.rstrip("/")
    complete_path: Final = (
        base_path
        if base_path.endswith(normalized_endpoint)
        else (
            f"{base_path}{normalized_endpoint.removeprefix('/v1')}"
            if base_path.endswith("/v1")
            else f"{base_path}{normalized_endpoint}"
        )
    )
    return str(resolved_base.copy_with(path=complete_path))


def raise_for_zexapi_error(raw_response: httpx.Response) -> None:
    if raw_response.status_code < 400:
        return
    raise BaseLLMException(
        status_code=raw_response.status_code,
        message=raw_response.text,
        headers=raw_response.headers,
    )


def parse_zexapi_task(raw_response: httpx.Response) -> ZexAPITaskResponse:
    raise_for_zexapi_error(raw_response)
    try:
        return ZexAPITaskResponse.model_validate_json(raw_response.text)
    except ValueError as exc:
        raise BaseLLMException(
            status_code=raw_response.status_code,
            message=f"Invalid ZexAPI task response: {exc}",
            headers=raw_response.headers,
        ) from exc
