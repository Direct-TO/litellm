"""One-shot recovery of Seedance 2.5's explicit virtual-person image rejection."""

import asyncio
import hashlib
import json
import re
import time
from collections.abc import Generator, Mapping
from dataclasses import dataclass
from typing import Final, Literal, NamedTuple
from uuid import uuid4

import httpx
from pydantic import TypeAdapter

from litellm._logging import verbose_logger
from litellm.litellm_core_utils.litellm_logging import Logging
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.submission_utils import (
    classify_http_submission_outcome,
    copy_submission_metadata,
    mark_reexecution_blocked,
    mark_submission_outcome,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.videos.intent import get_video_intent_metadata

from ..common_utils import build_toapis_endpoint, parse_toapis_video_create_task

REVIEW_TIMEOUT_SECONDS: Final = 120.0
REVIEW_POLL_INTERVAL_SECONDS: Final = 5.0
_OBJECT_ADAPTER: Final = TypeAdapter(dict[str, object])
_IMAGES_ADAPTER: Final = TypeAdapter(list[object])
_AVATAR_PATH: Final = "/v1/videos/doubao-seedance-2-0/private-avatar"
_IMAGE_REJECTION: Final = re.compile(
    r"input image\s+((?:['\"]?content\[\d+\]['\"]?\s*)+)may contain real person", re.IGNORECASE
)
_TASK_FIELDS: Final = frozenset({"id", "task_id", "video_id"})
_IDEMPOTENCY_HEADERS: Final = frozenset({"idempotency-key", "x-idempotency-key"})


class ToAPISVideoRecoveryError(BaseLLMException):
    """Keep submission provenance when crossing the generic HTTP error handler."""


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


def _rejected_images(response: httpx.Response, data: Mapping[str, object]) -> tuple[str, tuple[int, ...]] | None:
    if response.status_code != 400 or data.get("model") != "seedance-2-5" or data.get("content") is not None:
        return None
    nodes = _error_nodes(response.text)
    if any(
        any(node.get(field) for field in _TASK_FIELDS)
        or node.get("status") in ("queued", "pending", "in_progress", "completed", "accepted", "unknown")
        for node in nodes
    ):
        return None
    positions: set[int] = set()
    for node in nodes:
        code, message = node.get("code"), node.get("message")
        if not isinstance(code, str) or not isinstance(message, str):
            continue
        # ToAPIs can flatten PrivacyInformation into this wrapper code. The
        # indexed image-rejection message below is still required for recovery.
        if code.split(".")[-1] != "PrivacyInformation" and code != "fail_to_fetch_task":
            continue
        match = _IMAGE_REJECTION.search(message)
        if match:
            positions.update(int(index[1]) - 1 for index in re.finditer(r"content\[(\d+)\]", match[1]))
    if not positions:
        return None
    field = "image_with_roles" if data.get("image_with_roles") is not None else "image_urls"
    if data.get("image_with_roles") is not None and data.get("image_urls") is not None:
        return None
    images = data.get(field)
    if not isinstance(images, list) or not data.get("prompt"):
        return None
    image_list = _IMAGES_ADAPTER.validate_python(images)
    # ToAPIs' Seedance content starts with the prompt at content[0], followed
    # by the ordered images. Never guess a different offset or fall back to all images.
    for index in positions:
        if index < 0 or index >= len(image_list):
            return None
        image = image_list[index]
        url = _OBJECT_ADAPTER.validate_python(image).get("url") if isinstance(image, dict) else image
        if not isinstance(url, str):
            return None
        try:
            parsed = httpx.URL(url)
        except httpx.InvalidURL:
            return None
        if parsed.scheme not in ("http", "https") or not parsed.host or parsed.userinfo:
            return None
    return field, tuple(sorted(positions))


@dataclass(frozen=True)
class _Request:
    method: Literal["GET", "POST"]
    url: str
    headers: Mapping[str, str]
    data: Mapping[str, object] | None
    timeout: float | httpx.Timeout
    stage: Literal["generation", "review"] = "generation"


class _Result(NamedTuple):
    response: httpx.Response
    data: dict[str, object]


_Steps = Generator[_Request | float | _Result, httpx.Response | None, None]


def _received(response: httpx.Response | None) -> httpx.Response:
    if response is None:
        raise RuntimeError("A ToAPIs HTTP step must return a response")
    return response


def _image_url(image: object) -> str:
    value = _OBJECT_ADAPTER.validate_python(image).get("url") if isinstance(image, dict) else image
    if not isinstance(value, str):
        raise TypeError("A ToAPIs image must have a string URL")
    return value


def _review_error(message: str) -> ToAPISVideoRecoveryError:
    return mark_reexecution_blocked(
        mark_submission_outcome(
            ToAPISVideoRecoveryError(
                status_code=400,
                message=f"ToAPIs PrivacyInformation recovery: {message}; video generation was not resubmitted",
            ),
            "rejected",
        )
    )


def _finish(response: httpx.Response, data: dict[str, object], recovered: bool = False) -> _Result:
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
            ToAPISVideoRecoveryError(
                status_code=response.status_code,
                message=response.text,
                headers=response.headers,
            ),
            outcome,
            task_id,
        )
        if recovered or outcome in ("accepted", "unknown"):
            mark_reexecution_blocked(error)
        raise error
    try:
        # Validate before returning to the generic transformer so a malformed accepted
        # response keeps its unknown outcome even across a nested Router fallback.
        parse_toapis_video_create_task(response)
    except BaseLLMException as exc:
        raise mark_reexecution_blocked(
            copy_submission_metadata(
                exc,
                ToAPISVideoRecoveryError(
                    status_code=exc.status_code,
                    message=str(exc),
                    headers=response.headers,
                ),
            )
        ) from exc
    return _Result(response, data)


def _review_data(response: httpx.Response) -> Mapping[str, object]:
    try:
        payload = _OBJECT_ADAPTER.validate_json(response.content)
    except ValueError as exc:
        raise _review_error("invalid private-avatar response") from exc
    envelope = payload
    data = envelope.get("data")
    if response.status_code != 200 or envelope.get("success") is not True or not isinstance(data, dict):
        raise _review_error(f"private-avatar request failed (HTTP {response.status_code})")
    return _OBJECT_ADAPTER.validate_python(data)


def _identifier(data: Mapping[str, object], field: str) -> str:
    value = data.get(field)
    if not isinstance(value, str) or re.fullmatch(r"[A-Za-z0-9_-]+", value) is None:
        raise _review_error(f"private-avatar returned an invalid {field}")
    return value


def _remaining(deadline: float) -> float:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise _review_error("private-avatar review timed out")
    return remaining


def _review_timeout(timeout: float | httpx.Timeout, deadline: float) -> httpx.Timeout:
    remaining = min(30.0, _remaining(deadline))
    original = timeout if isinstance(timeout, httpx.Timeout) else httpx.Timeout(timeout)
    return httpx.Timeout(
        **{key: min(value, remaining) if value is not None else remaining for key, value in original.as_dict().items()}
    )


def _create_steps(
    data: dict[str, object], url: str, headers: Mapping[str, str], timeout: float | httpx.Timeout, logging_obj: Logging
) -> _Steps:
    response = _received((yield _Request("POST", url, headers, data, timeout)))
    logging_params = logging_obj.model_call_details.get("litellm_params", {})
    if get_video_intent_metadata({**logging_params, "_router_call_type": "avideo_generation"}) is not None:
        # The automatic-intent entry point promises a single submission, including
        # when a separate private-avatar recovery would otherwise submit again.
        yield _finish(response, data)
        return
    rejected = _rejected_images(response, data)
    if rejected is None:
        yield _finish(response, data)
        return
    field, positions = rejected
    # Derive the review endpoint from the actual generation URL, retaining any gateway prefix.
    endpoint = httpx.URL(url)
    if not endpoint.path.endswith("/v1/videos/generations"):
        yield _finish(response, data)
        return
    base = str(endpoint.copy_with(path=endpoint.path.removesuffix("/videos/generations")))
    avatar_url = build_toapis_endpoint(base, _AVATAR_PATH)
    review_headers = {
        key: value for key, value in headers.items() if key.lower() not in _IDEMPOTENCY_HEADERS | {"content-length"}
    }
    review_limit = timeout.read if isinstance(timeout, httpx.Timeout) else timeout
    deadline = time.monotonic() + min(
        REVIEW_TIMEOUT_SECONDS, review_limit if review_limit is not None else REVIEW_TIMEOUT_SECONDS
    )
    images = _IMAGES_ADAPTER.validate_python(data[field])
    reviewed: dict[str, str] = {}
    verbose_logger.info("ToAPIs Seedance private-avatar review started for image positions %s", positions)
    for index in positions:
        image = images[index]
        source = _image_url(image)
        if source not in reviewed:
            # One group per distinct image avoids assuming different images depict the same character.
            group_response = _received(
                (
                    yield _Request(
                        "POST",
                        avatar_url + "/groups",
                        review_headers,
                        {"name": "litellm-virtual-avatar-" + uuid4().hex},
                        _review_timeout(timeout, deadline),
                        "review",
                    )
                ),
            )
            group_id = _identifier(_review_data(group_response), "group_id")
            asset_response = _received(
                (
                    yield _Request(
                        "POST",
                        avatar_url + "/assets",
                        review_headers,
                        {
                            "group_id": group_id,
                            "asset_type": "image",
                            "source_url": source,
                            "name": f"reference-{index + 1}",
                        },
                        _review_timeout(timeout, deadline),
                        "review",
                    )
                ),
            )
            asset = _review_data(asset_response)
            asset_id = _identifier(asset, "asset_id")
            verbose_logger.info(
                "ToAPIs private-avatar submitted: image=%s group=%s asset=%s", index + 1, group_id, asset_id
            )
            while True:
                _remaining(deadline)
                if asset.get("asset_id") != asset_id:
                    raise _review_error("private-avatar query returned a different asset_id")
                status = asset.get("status")
                if status == "active":
                    asset_url = asset.get("asset_url")
                    if asset_url is not None and asset_url != f"asset://{asset_id}":
                        raise _review_error(f"private-avatar returned an inconsistent asset_url for {asset_id}")
                    reviewed[source] = f"asset://{asset_id}"
                    break
                if status != "processing":
                    raise _review_error(f"private-avatar review did not pass for asset {asset_id} (status={status})")
                yield min(REVIEW_POLL_INTERVAL_SECONDS, _remaining(deadline))
                poll_response = _received(
                    (
                        yield _Request(
                            "GET",
                            avatar_url + "/assets/" + asset_id,
                            review_headers,
                            None,
                            _review_timeout(timeout, deadline),
                            "review",
                        )
                    ),
                )
                asset = _review_data(poll_response)
        images[index] = (
            {**_OBJECT_ADAPTER.validate_python(image), "url": reviewed[source]}
            if isinstance(image, dict)
            else reviewed[source]
        )
    _remaining(deadline)
    retry_data = {**data, field: images}
    # A corrected body is a distinct request; do not reuse a key that may cache the original rejection.
    retry_headers = {
        key: "litellm-avatar-" + hashlib.sha256((value + json.dumps(images, sort_keys=True)).encode()).hexdigest()
        if key.lower() in _IDEMPOTENCY_HEADERS
        else value
        for key, value in headers.items()
        if key.lower() != "content-length"
    }
    verbose_logger.info("ToAPIs Seedance private-avatar review passed; resubmitting video once")
    logging_obj.pre_call(  # pyright: ignore[reportUnknownMemberType]  # logging callback has an untyped public signature
        input=str(data.get("prompt", "")),
        api_key="",
        additional_args={
            "complete_input_dict": retry_data,
            "api_base": url,
            "headers": retry_headers,
        },
    )
    # This is deliberately outside any recovery loop: a second rejection is returned unchanged.
    retry_response = _received((yield _Request("POST", url, retry_headers, retry_data, timeout)))
    yield _finish(retry_response, retry_data, recovered=True)


def _transport_error(error: httpx.RequestError, request: _Request) -> ToAPISVideoRecoveryError:
    if request.stage == "review":
        return _review_error(f"private-avatar transport failed ({type(error).__name__})")
    return mark_reexecution_blocked(
        mark_submission_outcome(
            ToAPISVideoRecoveryError(
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
    logging_obj: Logging,
) -> _Result:
    steps = _create_steps(data, url, headers, timeout, logging_obj)
    response: httpx.Response | None = None
    try:
        while True:
            step = steps.send(response)
            if isinstance(step, _Result):
                return step
            if not isinstance(step, _Request):
                time.sleep(step)
                response = None
            else:
                try:
                    response = client.client.request(
                        step.method, step.url, headers=step.headers, json=step.data, timeout=step.timeout
                    )
                except httpx.RequestError as exc:
                    raise _transport_error(exc, step) from exc
    finally:
        steps.close()


async def async_video_generation(
    client: AsyncHTTPHandler,
    data: dict[str, object],
    url: str,
    headers: Mapping[str, str],
    timeout: float | httpx.Timeout,
    logging_obj: Logging,
) -> _Result:
    steps = _create_steps(data, url, headers, timeout, logging_obj)
    response: httpx.Response | None = None
    try:
        while True:
            step = steps.send(response)
            if isinstance(step, _Result):
                return step
            if not isinstance(step, _Request):
                await asyncio.sleep(step)
                response = None
            else:
                try:
                    # Use the caller's transport, but do not reopen/re-POST on RemoteProtocolError:
                    # a lost response can already represent an accepted, billable video task.
                    response = await client.client.request(
                        step.method, step.url, headers=step.headers, json=step.data, timeout=step.timeout
                    )
                except httpx.RequestError as exc:
                    raise _transport_error(exc, step) from exc
    finally:
        steps.close()
