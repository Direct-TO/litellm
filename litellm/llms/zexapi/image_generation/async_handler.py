"""Submit one ZexAPI image task, poll that task, and return an ImageResponse."""

import asyncio
import math
import time
from collections.abc import Awaitable, Callable, Mapping, Sequence
from typing import Final, NamedTuple

import httpx
from pydantic import TypeAdapter

from litellm.constants import request_timeout as DEFAULT_REQUEST_TIMEOUT
from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.litellm_core_utils.url_utils import encode_url_path_segment
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.submission_utils import get_submission_outcome, mark_submission_outcome
from litellm.llms.custom_httpx import http_handler as http_handlers
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import ImageObject, ImageResponse, LlmProviders

from ..common_utils import ZexAPITaskResponse, build_zexapi_endpoint, get_zexapi_api_key, parse_zexapi_task
from .async_transformation import (
    AsyncImageFile,
    async_edit_references,
    async_image_error,
    async_image_params,
    normalize_async_references,
)

# The provider documents 3–5 seconds between task queries.
POLL_INTERVAL: Final = 5.0
_OBJECT_MAP: Final = TypeAdapter(dict[str, object])


class _Request(NamedTuple):
    url: str
    headers: dict[str, str]
    data: dict[str, object]


class ZexAPIAsyncImages:
    def __init__(
        self,
        sync_sleep: Callable[[float], None] = time.sleep,
        async_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._sync_sleep = sync_sleep
        self._async_sleep = async_sleep
        self._monotonic = monotonic

    def image_generation(
        self,
        model: str,
        prompt: str | None,
        optional_params: Mapping[str, object],
        litellm_params: GenericLiteLLMParams | Mapping[str, object],
        logging_obj: LiteLLMLoggingObj,
        timeout: float | httpx.Timeout | None,
        api_key: str | None = None,
        extra_headers: Mapping[str, object] | None = None,
        extra_body: Mapping[str, object] | None = None,
        client: HTTPHandler | AsyncHTTPHandler | None = None,
        aimg_generation: bool = False,
        image: Sequence[AsyncImageFile] | None = None,
        is_edit: bool = False,
    ) -> ImageResponse | Awaitable[ImageResponse]:
        request = self._prepare(
            model, prompt, optional_params, litellm_params, api_key, extra_headers, extra_body, image, is_edit
        )
        budget = self._budget(timeout)
        if aimg_generation:
            return self._run_async(
                model, request, logging_obj, budget, client if isinstance(client, AsyncHTTPHandler) else None
            )
        return self._run_sync(model, request, logging_obj, budget, client if isinstance(client, HTTPHandler) else None)

    @staticmethod
    def _prepare(
        model: str,
        prompt: str | None,
        optional_params: Mapping[str, object],
        litellm_params: GenericLiteLLMParams | Mapping[str, object],
        api_key: str | None,
        extra_headers: Mapping[str, object] | None,
        extra_body: Mapping[str, object] | None,
        image: Sequence[AsyncImageFile] | None,
        is_edit: bool,
    ) -> _Request:
        if not prompt or not prompt.strip():
            raise async_image_error(model, "ZexAPI image tasks require a non-empty prompt")
        if extra_body:
            raise async_image_error(
                model, "ZexAPI asynchronous images require named image parameters, not extra_body overrides"
            )
        params = (
            litellm_params.model_dump(exclude_none=True)
            if isinstance(litellm_params, GenericLiteLLMParams)
            else litellm_params
        )
        raw_key, raw_base = params.get("api_key"), params.get("api_base")
        key = get_zexapi_api_key(api_key or (raw_key if isinstance(raw_key, str) else None))
        if not key:
            raise ValueError("ZEXAPI_API_KEY is required")
        mapped = async_image_params(model, optional_params)
        references = (
            async_edit_references(image or (), model)
            if is_edit
            else normalize_async_references(mapped.get("image_url"), model)
        )
        data: dict[str, object] = {
            "model": model,
            "prompt": prompt,
            **{key: value for key, value in mapped.items() if key in ("aspect_ratio", "size")},
        }
        if references:
            data["images"] = references
        headers = {"Authorization": f"Bearer {key}", "Content-Type": "application/json"}
        headers.update({key: str(value) for key, value in (extra_headers or {}).items() if value is not None})
        return _Request(
            build_zexapi_endpoint(raw_base if isinstance(raw_base, str) else None, "/v1/videos"), headers, data
        )

    @staticmethod
    def _budget(timeout: float | httpx.Timeout | None) -> float:
        value = timeout.read if isinstance(timeout, httpx.Timeout) else timeout
        budget = float(DEFAULT_REQUEST_TIMEOUT if value is None else value)
        if not math.isfinite(budget) or budget <= 0:
            raise ValueError("Image task timeout must be a positive finite number")
        return budget

    def _remaining(self, deadline: float) -> float:
        remaining = deadline - self._monotonic()
        if remaining <= 0:
            raise BaseLLMException(
                status_code=408, message="ZexAPI image task exceeded the request timeout", headers={}
            )
        return remaining

    @staticmethod
    def _initial(response: httpx.Response) -> ZexAPITaskResponse:
        try:
            task = parse_zexapi_task(response)
            if not task.id.strip():
                raise ValueError("Empty task ID")
            return task
        except Exception as exc:
            # A malformed envelope may still contain an accepted task. Keep its ID.
            try:
                payload = _OBJECT_MAP.validate_json(response.content)
                task_id = payload.get("id") if payload.get("object") == "image" else None
            except ValueError:
                task_id = None
            if isinstance(task_id, str) and task_id.strip():
                mark_submission_outcome(exc, "accepted", task_id)
            elif get_submission_outcome(exc) is None:
                mark_submission_outcome(exc, "unknown")
            raise

    @staticmethod
    def _result(model: str, task: ZexAPITaskResponse, initial: ZexAPITaskResponse) -> ImageResponse | None:
        if task.id != initial.id or task.object != "image":
            raise BaseLLMException(
                status_code=502, message="ZexAPI returned a different task or a non-image result", headers={}
            )
        if task.status == "failed" or task.error is not None:
            detail = (
                task.error
                if isinstance(task.error, str)
                else (
                    f"{task.error.code}: {task.error.message}" if task.error is not None else "image generation failed"
                )
            )
            raise BaseLLMException(status_code=502, message=f"ZexAPI image task failed: {detail}", headers={})
        if task.status != "completed":
            return None
        try:
            url = httpx.URL(task.url or "")
            valid = url.is_absolute_url and url.scheme in ("http", "https")
        except httpx.InvalidURL:
            valid = False
        if not valid:
            raise BaseLLMException(
                status_code=502, message="ZexAPI completed an image task without a downloadable URL", headers={}
            )
        return ImageResponse(
            created=task.created_at or initial.created_at or initial.created,
            data=[ImageObject(url=task.url)],
            hidden_params={"model": model, "task_id": initial.id},
        )

    @staticmethod
    def _log_start(logging_obj: LiteLLMLoggingObj, request: _Request) -> None:
        logging_obj.pre_call(  # pyright: ignore[reportUnknownMemberType] # legacy Logging API
            input=request.data["prompt"],
            api_key="",
            additional_args={"complete_input_dict": request.data, "api_base": request.url},
        )

    @staticmethod
    def _log_finish(logging_obj: LiteLLMLoggingObj, request: _Request, response: httpx.Response) -> None:
        logging_obj.post_call(  # pyright: ignore[reportUnknownMemberType] # legacy Logging API
            input=request.data["prompt"],
            api_key="",
            original_response=response.text,
            additional_args={"api_base": request.url},
        )

    def _run_sync(
        self, model: str, request: _Request, logging_obj: LiteLLMLoggingObj, budget: float, client: HTTPHandler | None
    ) -> ImageResponse:
        transport = client or http_handlers._get_httpx_client()  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType] # shared handler factory
        deadline = self._monotonic() + budget
        self._log_start(logging_obj, request)
        try:
            # HTTPHandler.post can retry transport failures; raw client POST must run once.
            response = transport.client.post(
                request.url,
                headers=request.headers,
                json=request.data,
                timeout=self._remaining(deadline),
                follow_redirects=False,
            )
        except Exception as exc:
            raise mark_submission_outcome(exc, "unknown")
        initial = self._initial(response)
        task = initial
        try:
            while True:
                result = self._result(model, task, initial)
                if result is not None:
                    self._log_finish(logging_obj, request, response)
                    return result
                self._sync_sleep(min(POLL_INTERVAL, self._remaining(deadline)))
                response = transport.client.get(
                    request.url.rstrip("/") + "/" + encode_url_path_segment(initial.id, field_name="task_id"),
                    headers=request.headers,
                    timeout=self._remaining(deadline),
                    follow_redirects=False,
                )
                task = parse_zexapi_task(response)
        except Exception as exc:
            raise mark_submission_outcome(exc, "accepted", initial.id)

    async def _run_async(
        self,
        model: str,
        request: _Request,
        logging_obj: LiteLLMLoggingObj,
        budget: float,
        client: AsyncHTTPHandler | None,
    ) -> ImageResponse:
        transport = client or http_handlers.get_async_httpx_client(llm_provider=LlmProviders.ZEXAPI)  # pyright: ignore[reportUnknownMemberType] # shared handler factory
        deadline = self._monotonic() + budget
        self._log_start(logging_obj, request)
        try:
            response = await asyncio.wait_for(
                transport.client.post(
                    request.url,
                    headers=request.headers,
                    json=request.data,
                    timeout=self._remaining(deadline),
                    follow_redirects=False,
                ),
                timeout=self._remaining(deadline),
            )
        except Exception as exc:
            raise mark_submission_outcome(exc, "unknown")
        initial = self._initial(response)
        task = initial
        try:
            while True:
                result = self._result(model, task, initial)
                if result is not None:
                    self._log_finish(logging_obj, request, response)
                    return result
                await self._async_sleep(min(POLL_INTERVAL, self._remaining(deadline)))
                response = await asyncio.wait_for(
                    transport.client.get(
                        request.url.rstrip("/") + "/" + encode_url_path_segment(initial.id, field_name="task_id"),
                        headers=request.headers,
                        timeout=self._remaining(deadline),
                        follow_redirects=False,
                    ),
                    timeout=self._remaining(deadline),
                )
                task = parse_zexapi_task(response)
        except Exception as exc:
            raise mark_submission_outcome(exc, "accepted", initial.id)


zexapi_async_images: Final = ZexAPIAsyncImages()
