from collections.abc import Mapping, Sequence
from typing import Annotated, Final, Literal, cast

import httpx
from pydantic import BaseModel, BeforeValidator, ConfigDict

import litellm
from litellm.llms.base_llm.base_utils import BaseLLMModelInfo
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.submission_utils import classify_http_submission_outcome, mark_submission_outcome
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import AllMessageValues
from litellm.types.utils import ProviderSpecificModelInfo

DEFAULT_API_BASE: Final = "https://toapis.com/v1"


def _normalize_task_status(status: object) -> object:
    return "queued" if status == "pending" else status


ToAPISTaskStatus = Annotated[
    Literal["queued", "in_progress", "completed", "failed"],
    BeforeValidator(_normalize_task_status),
]


class ToAPISTaskError(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    code: str | int
    message: str


class ToAPISTaskResultItem(BaseModel):
    model_config = ConfigDict(extra="allow", frozen=True)

    url: str
    format: str | None = None
    last_frame_url: str | None = None


class ToAPISTaskResult(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    type: Literal["image", "video"]
    data: tuple[ToAPISTaskResultItem, ...]


class ToAPISTaskResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str
    object: Literal["generation.task"]
    model: str | None = None
    status: ToAPISTaskStatus
    progress: int = 0
    created_at: int | None = None
    completed_at: int | None = None
    expires_at: int | None = None
    result: ToAPISTaskResult | None = None
    error: ToAPISTaskError | None = None


class ToAPISImageUploadData(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str
    url: str
    mime_type: str
    size: int


class ToAPISImageUploadResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    success: bool
    message: str
    data: ToAPISImageUploadData | None = None


class _ToAPISModel(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    id: str


class _ToAPISModelsResponse(BaseModel):
    model_config = ConfigDict(extra="ignore", frozen=True)

    data: tuple[_ToAPISModel, ...]


def get_toapis_api_base(api_base: str | None = None) -> str:
    return api_base or get_secret_str("TOAPIS_API_BASE") or DEFAULT_API_BASE


def get_toapis_api_key(api_key: str | None = None) -> str | None:
    global_api_key: Final = litellm.api_key if isinstance(litellm.api_key, str) else None
    return api_key or global_api_key or get_secret_str("TOAPIS_API_KEY")


def build_toapis_endpoint(api_base: str | None, endpoint: str) -> str:
    resolved_base: Final = httpx.URL(get_toapis_api_base(api_base))
    if resolved_base.query or resolved_base.fragment:
        raise ValueError("TOAPIS_API_BASE must not contain a query string or fragment")
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


def parse_toapis_task(raw_response: httpx.Response) -> ToAPISTaskResponse:
    if raw_response.status_code >= 400:
        raise mark_submission_outcome(
            BaseLLMException(
                status_code=raw_response.status_code,
                message=raw_response.text,
                headers=raw_response.headers,
            ),
            classify_http_submission_outcome(raw_response),
        )
    try:
        return ToAPISTaskResponse.model_validate_json(raw_response.text)
    except ValueError as exc:
        raise mark_submission_outcome(
            BaseLLMException(
                status_code=502,
                message=f"Invalid ToAPIs task response: {exc}",
                headers=raw_response.headers,
            ),
            "unknown",
        ) from exc


def parse_toapis_video_create_task(raw_response: httpx.Response) -> ToAPISTaskResponse:
    """Parse a video-create response, including ToAPIs' accepted-task envelope."""
    if raw_response.status_code >= 400:
        raise mark_submission_outcome(
            BaseLLMException(
                status_code=raw_response.status_code,
                message=raw_response.text,
                headers=raw_response.headers,
            ),
            classify_http_submission_outcome(raw_response),
        )
    try:
        raw_payload: object = cast(object, raw_response.json())
        payload_to_validate: object = raw_payload
        if isinstance(raw_payload, Mapping):
            payload: Final = dict(cast(Mapping[str, object], raw_payload))
            if 200 <= raw_response.status_code < 300 and payload.get("object") == "video":
                raw_task_id: Final = payload.get("task_id")
                raw_id: Final = payload.get("id")
                task_id: Final = (
                    raw_task_id.strip()
                    if isinstance(raw_task_id, str) and raw_task_id.strip()
                    else raw_id.strip()
                    if isinstance(raw_id, str) and raw_id.strip()
                    else None
                )
                if task_id is None:
                    raise ValueError("ToAPIs video create response is missing a non-empty task ID")
                payload.update(
                    {
                        "id": task_id,
                        "object": "generation.task",
                        # Some models return a real status in the video envelope.
                        # Preserve it, especially terminal success/failure states.
                        "status": "queued" if payload.get("status") == "" else payload.get("status"),
                    }
                )
            payload_to_validate = payload
        return ToAPISTaskResponse.model_validate(payload_to_validate)
    except ValueError as exc:
        raise mark_submission_outcome(
            BaseLLMException(
                status_code=502,
                message=f"Invalid ToAPIs task response: {exc}",
                headers=raw_response.headers,
            ),
            "unknown",
        ) from exc


def parse_toapis_image_upload(raw_response: httpx.Response) -> ToAPISImageUploadData:
    if raw_response.status_code >= 400:
        raise mark_submission_outcome(
            BaseLLMException(
                status_code=raw_response.status_code,
                message=raw_response.text,
                headers=raw_response.headers,
            ),
            "rejected",
        )
    try:
        upload_response: Final = ToAPISImageUploadResponse.model_validate_json(raw_response.text)
    except ValueError as exc:
        raise mark_submission_outcome(
            BaseLLMException(
                status_code=502,
                message=f"Invalid ToAPIs image upload response: {exc}",
                headers=raw_response.headers,
            ),
            "rejected",
        ) from exc
    if not upload_response.success:
        raise mark_submission_outcome(
            BaseLLMException(
                status_code=502,
                message=upload_response.message,
                headers=raw_response.headers,
            ),
            "rejected",
        )
    if upload_response.data is None:
        raise mark_submission_outcome(
            BaseLLMException(
                status_code=502,
                message="ToAPIs image upload succeeded without file metadata",
                headers=raw_response.headers,
            ),
            "rejected",
        )
    return upload_response.data


class ToAPISModelInfo(BaseLLMModelInfo):
    def get_provider_info(self, model: str) -> ProviderSpecificModelInfo | None:
        return None

    def get_models(  # mutable-ok: BaseLLMModelInfo requires callers to receive a concrete model list
        self, api_key: str | None = None, api_base: str | None = None
    ) -> list[str]:  # mutable-ok: BaseLLMModelInfo fixes this mutable return contract
        resolved_api_key: Final = self.get_api_key(api_key)
        if resolved_api_key is None:
            raise ValueError("TOAPIS_API_KEY is required")
        response: Final[httpx.Response] = litellm.module_level_client.get(  # pyright: ignore[reportUnknownMemberType]  # shared HTTP handler still uses unparameterized dicts
            url=build_toapis_endpoint(api_base, "/v1/models"),
            headers={  # mutable-ok: shared HTTP handler requires concrete request headers
                "Authorization": f"Bearer {resolved_api_key}"
            },
        )
        if response.status_code != 200:
            raise BaseLLMException(
                status_code=response.status_code,
                message=response.text,
                headers=response.headers,
            )
        models_response: Final = _ToAPISModelsResponse.model_validate_json(response.text)
        return [  # mutable-ok: BaseLLMModelInfo requires a concrete model list
            model.id for model in models_response.data
        ]

    @staticmethod
    def get_api_key(api_key: str | None = None) -> str | None:
        return get_toapis_api_key(api_key)

    @staticmethod
    def get_api_base(api_base: str | None = None) -> str:
        return get_toapis_api_base(api_base)

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        messages: Sequence[AllMessageValues],
        optional_params: Mapping[str, object],
        litellm_params: Mapping[str, object],
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict[str, str]:  # mutable-ok: BaseLLMModelInfo requires mutable auth headers
        resolved_api_key: Final = self.get_api_key(api_key)
        if resolved_api_key is None:
            raise ValueError("TOAPIS_API_KEY is required")
        return {  # mutable-ok: BaseLLMModelInfo callers may add request headers
            **headers,
            "Authorization": f"Bearer {resolved_api_key}",
            "Content-Type": "application/json",
        }

    @staticmethod
    def get_base_model(model: str) -> str:
        return model.removeprefix("toapis/")
