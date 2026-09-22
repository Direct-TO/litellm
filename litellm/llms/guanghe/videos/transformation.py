"""Guanghe's inspiration-waterfall API, verified against its /models contract."""

import re
import uuid
from collections.abc import Mapping
from io import BufferedReader, BytesIO
from typing import Final
from urllib.parse import quote

import httpx
from httpx._types import RequestFiles
from pydantic import BaseModel, ConfigDict, StrictBool, ValidationError

from litellm.exceptions import UnsupportedParamsError
from litellm.images.utils import ImageEditRequestUtils
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.submission_utils import mark_submission_outcome
from litellm.llms.base_llm.videos.transformation import normalize_video_task_result
from litellm.llms.openai.videos.transformation import OpenAIVideoConfig
from litellm.secret_managers.main import get_secret_str
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.main import VideoCreateOptionalRequestParams, VideoObject
from litellm.types.videos.utils import encode_video_id_with_provider, extract_original_video_id
from litellm.videos.contract import validate_video_contract

DEFAULT_API_BASE: Final = "https://newapi.aitoken.name/api/inspiration-waterfall/v1"
MODEL_RESOLUTIONS: Final = {
    "seedance2.0_企业折扣": ("480p", "720p", "1080p"),
    "seedance2.5_企业折": ("480p", "720p", "1080p"),
    "seedance2.0-fast（企业折扣）": ("480p", "720p"),
    "seedance2.0-mini（企业折扣）": ("480p", "720p"),
}
_STATUS_MAP: Final = {
    "accepting": "queued",
    "unknown": "queued",
    "pending": "queued",
    "queued": "queued",
    "processing": "in_progress",
    "in_progress": "in_progress",
    "success": "completed",
    "succeeded": "completed",
    "completed": "completed",
    "failed": "failed",
    "cancelled": "failed",
    "manual_review": "failed",
}
_REJECTION_CODES: Final = {
    "invalid_api_key": 401,
    "api_access_disabled": 403,
    "ip_not_allowed": 403,
    "model_not_allowed": 403,
    "insufficient_token_quota": 402,
    "invalid_request": 400,
    "invalid_params": 400,
    "rate_limit_exceeded": 429,
    "file_upload_not_supported": 400,
}


def build_endpoint(api_base: str | None, endpoint: str) -> str:
    base = httpx.URL(api_base or get_secret_str("GUANGHE_API_BASE") or DEFAULT_API_BASE)
    if base.scheme not in ("http", "https") or not base.host or base.query or base.fragment or base.userinfo:
        raise ValueError("GUANGHE_API_BASE must be an HTTP(S) URL without credentials, query or fragment")
    path = base.path.rstrip("/")
    if path.endswith("/video/tasks"):
        path = path.removesuffix("/video/tasks")
    if not path:
        path = "/api/inspiration-waterfall/v1"
    elif path.endswith("/inspiration-waterfall"):
        path += "/v1"
    return str(base.copy_with(path=path + "/" + endpoint.lstrip("/")))


class _Envelope(BaseModel):
    model_config = ConfigDict(extra="ignore")
    success: StrictBool
    data: dict[str, object] | None = None
    message: str = ""
    code: str = ""


class _Task(BaseModel):
    model_config = ConfigDict(extra="ignore")
    job_id: str | None = None
    task_id: str | None = None
    status: str
    model_id: str | None = None
    created_at: int | None = None
    completed_at: int | None = None
    video_url: str | None = None
    error_message: str = ""
    credits_cost: int | float | None = None


def _error(message: str, status_code: int = 502, task_id: str | None = None) -> BaseLLMException:
    return mark_submission_outcome(
        BaseLLMException(status_code=status_code, message="Guanghe: " + message),
        "accepted" if task_id else "rejected" if status_code in (400, 401, 402, 403, 404, 422, 429) else "unknown",
        task_id,
    )


def parse_guanghe_response(response: httpx.Response) -> dict[str, object]:
    try:
        envelope = _Envelope.model_validate_json(response.text)
    except ValidationError as exc:
        raise _error("invalid response envelope", response.status_code if response.is_error else 502) from exc
    if response.is_error or not envelope.success:
        status = response.status_code if response.is_error else _REJECTION_CODES.get(envelope.code, 502)
        task_id = None
        if envelope.data:
            raw_id = envelope.data.get("job_id") or envelope.data.get("task_id")
            if isinstance(raw_id, str) and raw_id.strip():
                task_id = raw_id.strip()
        raise _error(f"{envelope.code or 'upstream_error'}: {envelope.message or 'request failed'}", status, task_id)
    if envelope.data is None:
        raise _error("response is missing data")
    return envelope.data


class GuangheVideoConfig(OpenAIVideoConfig):
    def get_supported_openai_params(self, model: str) -> list[str]:
        return [
            "seconds",
            "resolution",
            "aspect_ratio",
            "references",
            "input_reference",
            "extra_headers",
            "generate_audio",
        ]

    def map_openai_params(
        self,
        video_create_optional_params: VideoCreateOptionalRequestParams,
        model: str,
        drop_params: bool,
    ) -> dict[str, object]:
        params: Mapping[str, object] = video_create_optional_params

        def reject(message: str) -> None:
            raise UnsupportedParamsError(
                message=f"Guanghe model={model!r}: {message}", model=model, llm_provider="guanghe"
            )

        if model not in MODEL_RESOLUTIONS:
            reject("no documented video contract for this model")
        unsupported = set(params) - set(self.get_supported_openai_params(model))
        if unsupported:
            reject(f"unsupported parameters: {', '.join(sorted(unsupported))}")
        # These four channels generate audio by default (verified in the live outputs).
        # Their API exposes no audio toggle: enabled uses that default; never ignore a mute request.
        if params.get("generate_audio") is not None and params.get("generate_audio") is not True:
            reject("this channel has no documented mute control; generate_audio must be true or omitted")
        references = validate_video_contract(params)
        duration = str(params.get("seconds") if params.get("seconds") is not None else 4)
        maximum = 30 if model == "seedance2.5_企业折" else 15
        if not re.fullmatch(r"-1|[1-9]\d*", duration) or int(duration) not in (-1, *range(4, maximum + 1)):
            reject(f"seconds must be -1 or an integer between 4 and {maximum}")
        resolution = params.get("resolution") or "720p"
        if resolution not in MODEL_RESOLUTIONS[model]:
            reject(f"resolution must be one of {MODEL_RESOLUTIONS[model]}")
        ratio = params.get("aspect_ratio") or "adaptive"
        if ratio not in ("16:9", "9:16", "1:1", "4:3", "3:4", "21:9", "adaptive"):
            reject("unsupported aspect_ratio")
        native: dict[str, object] = {"duration": int(duration), "resolution": resolution, "aspectRatio": ratio}
        frames = [ref for ref in references if ref.role != "reference"]
        if frames and any(ref.role == "reference" for ref in references):
            reject("first/last frames cannot be mixed with ordinary references")
        for kind, field, limit in (("image", "imageUrls", 30), ("video", "videoUrls", 10), ("audio", "audioUrls", 10)):
            selected = [ref.url for ref in references if ref.type == kind and ref.role == "reference"]
            if len(selected) > limit:
                reject(f"at most {limit} {kind} references are supported")
            if selected:
                native[field] = selected
        for frame in frames:
            native["firstFrameUrl" if frame.role == "first_frame" else "lastFrameUrl"] = frame.url
        input_reference = params.get("input_reference")
        extra_headers = params.get("extra_headers")
        if input_reference is not None:
            native["input_reference"] = input_reference
        if extra_headers is not None:
            native["extra_headers"] = extra_headers
        return native

    def validate_environment(
        self,
        headers: Mapping[str, str],
        model: str,
        api_key: str | None = None,
        litellm_params: GenericLiteLLMParams | None = None,
    ) -> dict[str, str]:
        key = api_key or (litellm_params.api_key if litellm_params else None) or get_secret_str("GUANGHE_API_KEY")
        if not key:
            raise ValueError("GUANGHE_API_KEY is required")
        result = {**headers, "Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        if model:
            # The HTTP transport may replay a connection failure. Keep the same key for that replay.
            supplied = next((value for name, value in headers.items() if name.lower() == "idempotency-key"), None)
            if supplied is not None and not re.fullmatch(r"[\x20-\x7e]{1,128}", supplied):
                raise ValueError("Idempotency-Key must contain 1-128 printable ASCII characters")
            if supplied is None:
                result["Idempotency-Key"] = str(uuid.uuid4())
        return result

    def get_complete_url(self, model: str, api_base: str | None, litellm_params: Mapping[str, object]) -> str:
        return build_endpoint(api_base, "video/tasks")

    def use_multipart_form_data(self) -> bool:
        return False

    def get_video_create_input_reference_upload_request(
        self,
        video_create_optional_request_params: Mapping[str, object],
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
    ) -> tuple[str, Mapping[str, str], Mapping[str, object], RequestFiles] | None:
        content = video_create_optional_request_params.get("input_reference")
        if content is None:
            return None
        if not isinstance(content, (bytes, BytesIO, BufferedReader)):
            raise TypeError("Guanghe input_reference must be image bytes or a binary file")
        if any(video_create_optional_request_params.get(k) for k in ("imageUrls", "firstFrameUrl", "lastFrameUrl")):
            raise ValueError("input_reference cannot be mixed with image URLs")
        mime = ImageEditRequestUtils.get_image_content_type(content)
        if not mime.startswith("image/"):
            raise ValueError("Guanghe input_reference supports images only")
        if not isinstance(content, bytes):
            content.seek(0)
        upload_content = content if isinstance(content, bytes) else content.read()
        if len(upload_content) > 50 * 1024 * 1024:
            raise ValueError("Guanghe input_reference exceeds the 50 MiB upload limit")
        upload_headers = {k: v for k, v in headers.items() if k.lower() not in ("content-type", "idempotency-key")}
        upload_files: RequestFiles = (("file", ("reference", upload_content, mime)),)
        return (
            build_endpoint(litellm_params.api_base, "files/upload"),
            upload_headers,
            {"file_type": "input_material", "source": "upload"},
            upload_files,
        )

    def transform_video_create_input_reference_upload_response(
        self,
        raw_response: httpx.Response,
        video_create_optional_request_params: Mapping[str, object],
    ) -> dict[str, object]:
        data = parse_guanghe_response(raw_response)
        url = data.get("signed_url") or data.get("url")
        if not isinstance(url, str) or httpx.URL(url).scheme not in ("http", "https") or not httpx.URL(url).host:
            raise _error("upload did not return a usable URL")
        return {
            **{k: v for k, v in video_create_optional_request_params.items() if k != "input_reference"},
            "imageUrls": [url],
        }

    def transform_video_create_request(
        self,
        model: str,
        prompt: str,
        api_base: str,
        video_create_optional_request_params: Mapping[str, object],
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
    ) -> tuple[dict[str, object], RequestFiles, str]:
        if not prompt.strip() or len(prompt) > 10000:
            raise ValueError("Guanghe prompt must contain 1-10000 characters")
        if "input_reference" in video_create_optional_request_params:
            raise ValueError("Guanghe input_reference must be uploaded before task creation")
        native = {k: v for k, v in video_create_optional_request_params.items() if k != "extra_headers"}
        return {"model_id": model, "prompt": prompt, "params": native}, (), api_base

    def transform_video_create_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
        request_data: Mapping[str, object] | None = None,
    ) -> VideoObject:
        return self._task(raw_response, custom_llm_provider, model)

    def transform_video_status_retrieve_response(
        self,
        raw_response: httpx.Response,
        logging_obj: LiteLLMLoggingObj,
        custom_llm_provider: str | None = None,
    ) -> VideoObject:
        return self._task(raw_response, custom_llm_provider, None)

    @staticmethod
    def _task(response: httpx.Response, provider: str | None, model: str | None) -> VideoObject:
        data = parse_guanghe_response(response)
        raw_id = data.get("job_id") or data.get("task_id")
        task_id = raw_id.strip() if isinstance(raw_id, str) else ""
        if not task_id:
            raise _error("task response is missing a task ID")
        try:
            task = _Task.model_validate(data)
        except ValidationError as exc:
            raise _error("invalid task response", task_id=task_id) from exc
        if task.status not in _STATUS_MAP:
            raise _error(f"unrecognized task status: {task.status}", task_id=task_id)
        status = _STATUS_MAP[task.status]
        error = None
        if status == "failed":
            error = {
                "code": task.status,
                "message": task.error_message
                or (
                    "Task requires operator review; stop retrying. Refund is not confirmed."
                    if task.status == "manual_review"
                    else "Video task " + task.status
                ),
            }
        status, url, error = normalize_video_task_result(status, task.video_url or None, error, True)
        video = VideoObject(
            id=encode_video_id_with_provider(task_id, provider, model or task.model_id) if provider else task_id,
            object="video",
            status=status,
            created_at=task.created_at,
            completed_at=task.completed_at or None,
            model=task.model_id or model,
            output_url=url,
            error=error,
        )
        video._hidden_params.update({"provider_status": task.status, "credits_cost": task.credits_cost})  # pyright: ignore[reportPrivateUsage]  # Shared provider response metadata contract.
        return video

    def transform_video_content_request(
        self,
        video_id: str,
        api_base: str,
        litellm_params: GenericLiteLLMParams,
        headers: Mapping[str, str],
        variant: str | None = None,
    ) -> tuple[str, dict[str, object]]:
        if variant is not None:
            raise ValueError("Guanghe supports downloading the generated video only")
        task_id = quote(extract_original_video_id(video_id), safe="")
        return f"{api_base.rstrip('/')}/{task_id}/download", {}

    def transform_video_content_response(self, raw_response: httpx.Response, logging_obj: LiteLLMLoggingObj) -> bytes:
        mime = next((value for name, value in raw_response.headers.multi_items() if name == "content-type"), "")
        mime = mime.split(";", 1)[0].strip().lower()
        if "json" in mime:
            parse_guanghe_response(raw_response)
            raise _error("download endpoint returned JSON instead of video content")
        if not (mime.startswith("video/") or mime == "application/octet-stream"):
            raise _error("download endpoint did not return video content")
        if raw_response.is_error or raw_response.is_redirect or not raw_response.content:
            raise _error("video content download failed", raw_response.status_code if raw_response.is_error else 502)
        return raw_response.content

    def get_error_class(
        self, error_message: str, status_code: int, headers: dict[str, str] | httpx.Headers
    ) -> BaseLLMException:
        error = _error(error_message, status_code)
        error.headers = headers
        return error
