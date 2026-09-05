import asyncio
import time
from collections.abc import Awaitable, Callable, Mapping
from itertools import chain
from types import MappingProxyType
from typing import Final, NamedTuple, NoReturn

import httpx
from pydantic import TypeAdapter

from litellm.litellm_core_utils.litellm_logging import Logging as LiteLLMLoggingObj
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.submission_utils import mark_submission_outcome
from litellm.llms.custom_httpx import http_handler as http_handlers
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.router import GenericLiteLLMParams
from litellm.types.utils import ImageResponse, LlmProviders

from ..common_utils import ToAPISTaskResponse, parse_toapis_task
from .transformation import ToAPISImageGenerationConfig

DEFAULT_POLLING_INTERVAL: Final = 5.0
DEFAULT_MAX_POLLING_TIME: Final = 300.0
_OBJECT_MAP_ADAPTER: Final = TypeAdapter(dict[str, object])
_OBJECT_ADAPTER: Final = TypeAdapter(object)
_EMPTY_OBJECT_MAP: Final[Mapping[str, object]] = MappingProxyType({})
_EMPTY_HEADERS: Final[Mapping[str, str]] = MappingProxyType({})


class _PreparedRequest(NamedTuple):
    url: str
    headers: Mapping[str, str]
    data: Mapping[str, object]
    params: Mapping[str, object]


class ToAPISImageGeneration:
    def __init__(
        self,
        sync_sleep: Callable[[float], None] = time.sleep,
        async_sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        monotonic: Callable[[], float] = time.monotonic,
    ) -> None:
        self._config: Final = ToAPISImageGenerationConfig()
        self._sync_sleep: Final = sync_sleep
        self._async_sleep: Final = async_sleep
        self._monotonic: Final = monotonic

    def image_generation(
        self,
        model: str,
        prompt: str,
        optional_params: Mapping[str, object],
        litellm_params: GenericLiteLLMParams | Mapping[str, object],
        logging_obj: LiteLLMLoggingObj,
        timeout: float | httpx.Timeout | None,
        api_key: str | None = None,
        extra_headers: Mapping[str, object] | None = None,
        extra_body: Mapping[str, object] | None = None,
        client: HTTPHandler | AsyncHTTPHandler | None = None,
        aimg_generation: bool = False,
    ) -> ImageResponse | Awaitable[ImageResponse]:
        if aimg_generation:
            return self.async_image_generation(
                model=model,
                prompt=prompt,
                optional_params=optional_params,
                litellm_params=litellm_params,
                logging_obj=logging_obj,
                timeout=timeout,
                api_key=api_key,
                extra_headers=extra_headers,
                extra_body=extra_body,
                client=client if isinstance(client, AsyncHTTPHandler) else None,
            )
        prepared: Final = self._prepare_request(
            model, prompt, optional_params, litellm_params, api_key, extra_headers, extra_body
        )
        sync_client: Final = (
            client if isinstance(client, HTTPHandler) else http_handlers._get_httpx_client()  # pyright: ignore[reportPrivateUsage, reportUnknownMemberType]  # shared client factory has legacy unparameterized dicts
        )
        self._log_request(logging_obj, prompt, prepared)
        try:
            initial_response: Final = self._post_sync(sync_client, prepared, timeout)
        except Exception as exc:
            mark_submission_outcome(exc, "unknown")
            raise
        final_response: Final = self._poll_sync(
            initial_response,
            prepared.url,
            prepared.headers,
            sync_client,
            timeout,
        )
        return self._to_image_response(model, final_response, prepared, optional_params, logging_obj)

    async def async_image_generation(
        self,
        model: str,
        prompt: str,
        optional_params: Mapping[str, object],
        litellm_params: GenericLiteLLMParams | Mapping[str, object],
        logging_obj: LiteLLMLoggingObj,
        timeout: float | httpx.Timeout | None,
        api_key: str | None = None,
        extra_headers: Mapping[str, object] | None = None,
        extra_body: Mapping[str, object] | None = None,
        client: AsyncHTTPHandler | None = None,
    ) -> ImageResponse:
        prepared: Final = self._prepare_request(
            model, prompt, optional_params, litellm_params, api_key, extra_headers, extra_body
        )
        async_client: Final = client or http_handlers.get_async_httpx_client(  # pyright: ignore[reportUnknownMemberType]  # shared client factory has legacy unparameterized dicts
            llm_provider=LlmProviders.TOAPIS
        )
        self._log_request(logging_obj, prompt, prepared)
        try:
            initial_response: Final = await self._post_async(async_client, prepared, timeout)
        except Exception as exc:
            mark_submission_outcome(exc, "unknown")
            raise
        final_response: Final = await self._poll_async(
            initial_response,
            prepared.url,
            prepared.headers,
            async_client,
            timeout,
        )
        return self._to_image_response(model, final_response, prepared, optional_params, logging_obj)

    def _prepare_request(
        self,
        model: str,
        prompt: str,
        optional_params: Mapping[str, object],
        litellm_params: GenericLiteLLMParams | Mapping[str, object],
        api_key: str | None,
        extra_headers: Mapping[str, object] | None,
        extra_body: Mapping[str, object] | None,
    ) -> _PreparedRequest:
        raw_params: Final = (
            litellm_params.model_dump(exclude_none=True)
            if isinstance(litellm_params, GenericLiteLLMParams)
            else litellm_params
        )
        params: Final = MappingProxyType(_OBJECT_MAP_ADAPTER.validate_python(raw_params))
        base_headers: Final = self._config.validate_environment(
            api_key=api_key,
            headers=_EMPTY_HEADERS,
            model=model,
            messages=(),
            optional_params=optional_params,
            litellm_params=params,
        )
        header_source: Final[Mapping[str, object]] = extra_headers if extra_headers is not None else _EMPTY_OBJECT_MAP
        header_pairs: Final = tuple((key, str(value)) for key, value in header_source.items())
        headers: Final = MappingProxyType(
            dict(  # mutable-ok: temporary header map is immediately frozen
                chain(base_headers.items(), header_pairs)
            )
        )
        transformed_data: Final = self._config.transform_image_generation_request(
            model=model,
            prompt=prompt,
            optional_params=optional_params,
            litellm_params=params,
            headers=headers,
        )
        body_source: Final[Mapping[str, object]] = extra_body if extra_body is not None else _EMPTY_OBJECT_MAP
        data: Final = MappingProxyType(
            dict(  # mutable-ok: temporary request body is immediately frozen
                chain(transformed_data.items(), body_source.items())
            )
        )
        return _PreparedRequest(
            url=self._config.get_complete_url(
                api_base=self._api_base(params),
                api_key=api_key,
                model=model,
                optional_params=optional_params,
                litellm_params=params,
            ),
            headers=headers,
            data=data,
            params=params,
        )

    def _poll_sync(
        self,
        initial_response: httpx.Response,
        complete_url: str,
        headers: Mapping[str, str],
        client: HTTPHandler,
        timeout: float | httpx.Timeout | None,
        max_wait: float = DEFAULT_MAX_POLLING_TIME,
        interval: float = DEFAULT_POLLING_INTERVAL,
    ) -> httpx.Response:
        initial_task: Final = parse_toapis_task(initial_response)
        try:
            self._raise_if_terminal_failure(initial_task, initial_response.headers)
            if initial_task.status == "completed":
                return initial_response
            status_url: Final = self._config.get_status_url(complete_url, initial_task.id)
            deadline: Final = self._monotonic() + max_wait
            while True:
                self._wait_sync(interval, deadline)
                response = client.get(  # pyright: ignore[reportUnknownMemberType]  # HTTPHandler still uses unparameterized dicts
                    url=status_url,
                    headers=dict(headers),  # mutable-ok: legacy HTTP handler requires concrete headers
                    timeout=timeout,
                )
                if response.status_code == 429:
                    self._wait_sync(self._retry_after(response, interval), deadline)
                    continue
                task = parse_toapis_task(response)
                self._raise_if_terminal_failure(task, response.headers)
                if task.status == "completed":
                    return response
        except Exception as exc:
            mark_submission_outcome(exc, "accepted", provider_task_id=initial_task.id)
            raise

    async def _poll_async(
        self,
        initial_response: httpx.Response,
        complete_url: str,
        headers: Mapping[str, str],
        client: AsyncHTTPHandler,
        timeout: float | httpx.Timeout | None,
        max_wait: float = DEFAULT_MAX_POLLING_TIME,
        interval: float = DEFAULT_POLLING_INTERVAL,
    ) -> httpx.Response:
        initial_task: Final = parse_toapis_task(initial_response)
        try:
            self._raise_if_terminal_failure(initial_task, initial_response.headers)
            if initial_task.status == "completed":
                return initial_response
            status_url: Final = self._config.get_status_url(complete_url, initial_task.id)
            deadline: Final = self._monotonic() + max_wait
            while True:
                await self._wait_async(interval, deadline)
                response = await client.get(  # pyright: ignore[reportUnknownMemberType]  # AsyncHTTPHandler still uses unparameterized dicts
                    url=status_url,
                    headers=dict(headers),  # mutable-ok: legacy HTTP handler requires concrete headers
                    timeout=timeout,
                )
                if response.status_code == 429:
                    await self._wait_async(self._retry_after(response, interval), deadline)
                    continue
                task = parse_toapis_task(response)
                self._raise_if_terminal_failure(task, response.headers)
                if task.status == "completed":
                    return response
        except Exception as exc:
            mark_submission_outcome(exc, "accepted", provider_task_id=initial_task.id)
            raise

    def _post_sync(
        self,
        client: HTTPHandler,
        prepared: _PreparedRequest,
        timeout: float | httpx.Timeout | None,
    ) -> httpx.Response:
        response: Final[httpx.Response | None] = client.post(  # pyright: ignore[reportUnknownMemberType]  # HTTPHandler still uses unparameterized dicts
            url=prepared.url,
            headers=dict(prepared.headers),  # mutable-ok: legacy HTTP handler requires concrete headers
            json=dict(prepared.data),  # mutable-ok: legacy HTTP handler requires a concrete JSON dict
            timeout=timeout,
        )
        return self._require_response(response)

    async def _post_async(
        self,
        client: AsyncHTTPHandler,
        prepared: _PreparedRequest,
        timeout: float | httpx.Timeout | None,
    ) -> httpx.Response:
        response: Final[httpx.Response | None] = await client.post(  # pyright: ignore[reportUnknownMemberType]  # AsyncHTTPHandler still uses unparameterized dicts
            url=prepared.url,
            headers=dict(prepared.headers),  # mutable-ok: legacy HTTP handler requires concrete headers
            json=dict(prepared.data),  # mutable-ok: legacy HTTP handler requires a concrete JSON dict
            timeout=timeout,
        )
        return self._require_response(response)

    def _wait_sync(self, delay: float, deadline: float) -> None:
        remaining: Final = deadline - self._monotonic()
        if remaining <= 0:
            self._raise_timeout()
        self._sync_sleep(min(delay, remaining))
        if self._monotonic() >= deadline:
            self._raise_timeout()

    async def _wait_async(self, delay: float, deadline: float) -> None:
        remaining: Final = deadline - self._monotonic()
        if remaining <= 0:
            self._raise_timeout()
        await self._async_sleep(min(delay, remaining))
        if self._monotonic() >= deadline:
            self._raise_timeout()

    @staticmethod
    def _api_base(params: Mapping[str, object]) -> str | None:
        value: Final = params.get("api_base")
        return value if isinstance(value, str) else None

    @staticmethod
    def _log_request(logging_obj: LiteLLMLoggingObj, prompt: str, prepared: _PreparedRequest) -> None:
        logging_obj.pre_call(  # pyright: ignore[reportUnknownMemberType]  # Logging.pre_call remains untyped
            input=prompt,
            api_key="",
            additional_args={  # mutable-ok: Logging.pre_call consumes mutable additional args
                "complete_input_dict": prepared.data,
                "api_base": prepared.url,
                "headers": prepared.headers,
            },
        )

    def _to_image_response(
        self,
        model: str,
        response: httpx.Response,
        prepared: _PreparedRequest,
        optional_params: Mapping[str, object],
        logging_obj: LiteLLMLoggingObj,
    ) -> ImageResponse:
        return self._config.transform_image_generation_response(
            model=model,
            raw_response=response,
            model_response=ImageResponse(),
            logging_obj=logging_obj,
            request_data=prepared.data,
            optional_params=optional_params,
            litellm_params=prepared.params,
            encoding=None,
        )

    @staticmethod
    def _require_response(response: httpx.Response | None) -> httpx.Response:
        if response is None:
            raise BaseLLMException(
                status_code=500,
                message="ToAPIs returned no HTTP response",
                headers={},  # mutable-ok: BaseLLMException requires concrete response headers
            )
        return response

    @staticmethod
    def _raise_if_terminal_failure(task: ToAPISTaskResponse, headers: httpx.Headers) -> None:
        if task.status != "failed":
            return
        message: Final = f"{task.error.code}: {task.error.message}" if task.error else "Image generation failed"
        raise BaseLLMException(status_code=500, message=message, headers=headers)

    @staticmethod
    def _raise_timeout() -> NoReturn:
        raise BaseLLMException(
            status_code=408,
            message="ToAPIs image task timed out",
            headers={},  # mutable-ok: BaseLLMException requires concrete response headers
        )

    @staticmethod
    def _retry_after(response: httpx.Response, default: float) -> float:
        raw_value: Final = _OBJECT_ADAPTER.validate_python(response.headers.get("Retry-After"))
        if not isinstance(raw_value, str):
            return default
        try:
            return max(float(raw_value), default)
        except ValueError:
            return default


toapis_image_generation: Final = ToAPISImageGeneration()
