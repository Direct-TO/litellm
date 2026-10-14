"""Single-submission transport for ToAPIs Seedance video tasks."""

from collections.abc import Mapping
from typing import Final

import httpx
from pydantic import TypeAdapter

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.submission_utils import (
    classify_http_submission_outcome,
    copy_submission_metadata,
    mark_reexecution_blocked,
    mark_submission_outcome,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

from ..common_utils import parse_toapis_video_create_task

_SEEDANCE_VIDEO_MODELS: Final = frozenset({"seedance-2", "seedance-2-fast", "seedance-2-mini", "seedance-2-5"})
_OBJECT_ADAPTER: Final = TypeAdapter(dict[str, object])
_TASK_FIELDS: Final = frozenset({"id", "task_id", "video_id"})


class ToAPISVideoSubmissionError(BaseLLMException):
    """Preserve submission provenance across the generic HTTP error handler."""


def is_seedance_video_model(model: object) -> bool:
    return isinstance(model, str) and model in _SEEDANCE_VIDEO_MODELS


def _error_nodes(value: object, depth: int = 0) -> tuple[Mapping[str, object], ...]:
    if depth > 8:
        return ()
    if isinstance(value, str):
        try:
            return _error_nodes(_OBJECT_ADAPTER.validate_json(value), depth + 1)
        except ValueError:
            return ()
    if not isinstance(value, dict):
        return ()
    node = _OBJECT_ADAPTER.validate_python(value)
    return (node,) + tuple(
        child for key in ("error", "message", "data") for child in _error_nodes(node.get(key), depth + 1)
    )


def _finish(response: httpx.Response) -> httpx.Response:
    if response.status_code >= 400:
        nodes = _error_nodes(response.text)
        task_id = next(
            (
                value
                for node in nodes
                for field in ("task_id", "video_id", "id")
                if isinstance(value := node.get(field), str) and value
            ),
            None,
        )
        outcome = classify_http_submission_outcome(response)
        if task_id:
            outcome = "accepted"
        elif any(
            node.get("status") in ("queued", "pending", "in_progress", "completed", "accepted", "unknown")
            for node in nodes
        ):
            outcome = "unknown"
        error = mark_submission_outcome(
            ToAPISVideoSubmissionError(
                status_code=response.status_code,
                message=response.text,
                headers=response.headers,
            ),
            outcome,
            task_id,
        )
        if outcome in ("accepted", "unknown"):
            mark_reexecution_blocked(error)
        raise error
    try:
        # Validate accepted responses before the generic transformer so an invalid
        # envelope cannot cause an outer Router retry or a second paid task.
        parse_toapis_video_create_task(response)
    except BaseLLMException as exc:
        raise mark_reexecution_blocked(
            copy_submission_metadata(
                exc,
                ToAPISVideoSubmissionError(
                    status_code=exc.status_code,
                    message=str(exc),
                    headers=response.headers,
                ),
            )
        ) from exc
    return response


def _transport_error(error: httpx.RequestError) -> ToAPISVideoSubmissionError:
    return mark_reexecution_blocked(
        mark_submission_outcome(
            ToAPISVideoSubmissionError(
                status_code=408 if isinstance(error, httpx.TimeoutException) else 502,
                message=f"ToAPIs video submission outcome unknown ({type(error).__name__}); request was not retried",
            ),
            "unknown",
        )
    )


def video_generation(
    client: HTTPHandler,
    data: dict[str, object],
    url: str,
    headers: Mapping[str, str],
    timeout: float | httpx.Timeout,
) -> httpx.Response:
    try:
        response = client.client.request("POST", url, headers=headers, json=data, timeout=timeout)
    except httpx.RequestError as exc:
        raise _transport_error(exc) from exc
    return _finish(response)


async def async_video_generation(
    client: AsyncHTTPHandler,
    data: dict[str, object],
    url: str,
    headers: Mapping[str, str],
    timeout: float | httpx.Timeout,
) -> httpx.Response:
    try:
        # Bypass the wrapper's RemoteProtocolError retry: a lost response can
        # already represent an accepted, billable video task.
        response = await client.client.request("POST", url, headers=headers, json=data, timeout=timeout)
    except httpx.RequestError as exc:
        raise _transport_error(exc) from exc
    return _finish(response)
