import base64
import io
import json
from unittest.mock import Mock

import httpx
import pytest

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.submission_utils import get_provider_task_id, get_submission_outcome
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.zexapi.image_generation.async_handler import ZexAPIAsyncImages
from litellm.llms.zexapi.image_generation.async_transformation import (
    ZexAPIAsyncImageEditConfig,
    async_edit_references,
    async_image_params,
    normalize_async_references,
)

PNG = b"\x89PNG\r\n\x1a\nfixture"
URL = "https://files.example/result.png"
MODEL = "gpt-image-2-2K"


def task(status="queued", task_id="task/original;id", **kwargs):
    return {"id": task_id, "object": "image", "status": status, "created_at": 1, **kwargs}


class Clock:
    now = 0.0

    def monotonic(self):
        return self.now

    def sleep(self, duration):
        self.now += duration

    async def asleep(self, duration):
        self.sleep(duration)


def handler(clock=None):
    clock = clock or Clock()
    return ZexAPIAsyncImages(sync_sleep=clock.sleep, async_sleep=clock.asleep, monotonic=clock.monotonic)


def arguments(client, **kwargs):
    return dict(
        model=MODEL,
        prompt="edit image",
        optional_params={"aspect_ratio": "16:9"},
        litellm_params={"api_base": "https://zexapi.example/v1"},
        api_key="test-key",
        timeout=20,
        logging_obj=Mock(),
        client=client,
        **kwargs,
    )


@pytest.mark.parametrize("model,tier", [("gpt-image-2", "1K"), (MODEL, "2K"), ("gpt-image-2-4K", "4K")])
def test_fixed_models_keep_matching_tiers_and_auto_unspecified(model, tier):
    assert async_image_params(model, {"resolution": tier.lower(), "aspect_ratio": "16:9"}) == {"aspect_ratio": "16:9"}
    assert async_image_params(model, {"resolution": "auto", "aspect_ratio": "auto"}) == {}
    assert async_image_params(model, {}) == {}


@pytest.mark.parametrize(
    "params",
    [
        {"resolution": "1K"},
        {"resolution": "4K"},
        {"resolution": "invalid"},
        {"size": "1280x720"},
        {"aspect_ratio": "9:16", "size": "2560x1440"},
        {"aspect_ratio": "2:1"},
        {"aspect_ratio": "2560x1440"},
        {"mask": "mask"},
        {"background": "transparent"},
        {"input_fidelity": "high"},
        {"quality": "high"},
        {"n": 2},
        {"response_format": "b64_json"},
        {"imageConfig": {"imageSize": "2K"}},
    ],
)
def test_invalid_and_unsupported_semantics_are_not_dropped(params):
    with pytest.raises(litellm.UnsupportedParamsError):
        ZexAPIAsyncImageEditConfig().map_openai_params(params, MODEL, drop_params=True)


def test_pixel_size_must_match_model_tier():
    assert async_image_params(MODEL, {"size": "2560x1440"}) == {"size": "2560x1440"}
    assert async_image_params(MODEL, {"size": "2560x1440", "aspect_ratio": "16:9"}) == {"size": "2560x1440"}


def test_reference_files_are_complete_and_keep_their_original_positions():
    stream = io.BytesIO(PNG)
    stream.seek(3)
    refs = async_edit_references([stream, ("reference.png", PNG, "image/png")], MODEL)
    assert stream.tell() == 3
    assert refs == ["data:image/png;base64," + base64.b64encode(PNG).decode()] * 2
    assert normalize_async_references([{"url": URL}, refs[0]], MODEL) == [URL, refs[0]]


@pytest.mark.parametrize(
    "refs", ["file:///secret", "data:image/png;base64,???", "data:text/plain;base64,YQ==", [URL] * 9, {"invalid": URL}]
)
def test_invalid_references_are_rejected(refs):
    with pytest.raises(litellm.UnsupportedParamsError):
        normalize_async_references(refs, MODEL)


def test_sync_create_poll_and_result_keep_original_model_and_task_id():
    calls = []
    replies = iter([task(), task("processing"), task("completed", model="gpt-image-2", url=URL)])

    def respond(request):
        calls.append(request)
        return httpx.Response(200, json=next(replies))

    client = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond)))
    try:
        result = handler().image_generation(**arguments(client, image=[PNG], is_edit=True))
    finally:
        client.client.close()
    assert [r.method for r in calls] == ["POST", "GET", "GET"]
    assert str(calls[0].url) == "https://zexapi.example/v1/videos"
    assert all(str(r.url) == "https://zexapi.example/v1/videos/task%2Foriginal%3Bid" for r in calls[1:])
    assert json.loads(calls[0].content) == {
        "model": MODEL,
        "prompt": "edit image",
        "aspect_ratio": "16:9",
        "images": ["data:image/png;base64," + base64.b64encode(PNG).decode()],
    }
    assert result.data[0].url == URL
    assert result._hidden_params["task_id"] == "task/original;id"
    assert result._hidden_params["model"] == MODEL


@pytest.mark.asyncio
async def test_async_create_poll_and_result_preserve_references():
    calls = []
    replies = iter([task(), task("in_progress"), task("completed", url=URL)])

    def respond(request):
        calls.append(request)
        return httpx.Response(200, json=next(replies))

    client = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    args = arguments(client, aimg_generation=True)
    args["optional_params"] = {"image_url": [URL]}
    try:
        result = await handler().image_generation(**args)
    finally:
        await client.client.aclose()
    assert [r.method for r in calls] == ["POST", "GET", "GET"]
    assert json.loads(calls[0].content) == {"model": MODEL, "prompt": "edit image", "images": [URL]}
    assert result.data[0].url == URL


@pytest.mark.parametrize(
    "response,outcome",
    [
        (httpx.Response(429, text="rate limited"), "rejected"),
        (httpx.Response(503, text="no available channel"), "rejected"),
        (httpx.Response(503, text="gateway failure"), "unknown"),
        (httpx.Response(200, text="invalid JSON"), "unknown"),
        (httpx.Response(200, json={"status": "queued"}), "unknown"),
    ],
)
def test_initial_rejection_or_unknown_never_polls(response, outcome):
    calls = []

    def respond(request):
        calls.append(request)
        return response

    client = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond)))
    try:
        with pytest.raises(BaseLLMException) as caught:
            handler().image_generation(**arguments(client))
    finally:
        client.client.close()
    assert get_submission_outcome(caught.value) == outcome
    assert get_provider_task_id(caught.value) is None
    assert [r.method for r in calls] == ["POST"]


@pytest.mark.parametrize(
    "reply",
    [
        httpx.Response(503, text="no available channel"),
        httpx.Response(200, text="broken JSON"),
        httpx.Response(200, json=task("completed")),
        httpx.Response(200, json=task("completed", task_id="other", url=URL)),
        httpx.Response(200, json=task("failed", error={"code": "upstream_error", "message": "rejected content"})),
        httpx.Response(200, json=task("completed", object="video", url=URL)),
    ],
)
def test_poll_failures_keep_accepted_task_and_never_recreate(reply):
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(200, json=task()) if request.method == "POST" else reply

    client = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond)))
    try:
        with pytest.raises(BaseLLMException) as caught:
            handler().image_generation(**arguments(client))
    finally:
        client.client.close()
    assert get_submission_outcome(caught.value) == "accepted"
    assert get_provider_task_id(caught.value) == "task/original;id"
    assert [r.method for r in calls] == ["POST", "GET"]


@pytest.mark.asyncio
@pytest.mark.parametrize("after_acceptance", [False, True])
async def test_transport_failure_is_not_retried(after_acceptance):
    calls = []

    def respond(request):
        calls.append(request)
        if after_acceptance and request.method == "POST":
            return httpx.Response(200, json=task())
        raise httpx.RemoteProtocolError("connection closed")

    client = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        with pytest.raises(httpx.RemoteProtocolError) as caught:
            await handler().image_generation(**arguments(client, aimg_generation=True))
    finally:
        await client.client.aclose()
    assert get_submission_outcome(caught.value) == ("accepted" if after_acceptance else "unknown")
    assert get_provider_task_id(caught.value) == ("task/original;id" if after_acceptance else None)
    assert [r.method for r in calls].count("POST") == 1


def test_polling_timeout_uses_request_budget_and_retains_task_id():
    clock = Clock()
    calls = []

    def respond(request):
        calls.append(request)
        return httpx.Response(200, json=task())

    client = HTTPHandler(client=httpx.Client(transport=httpx.MockTransport(respond)))
    args = arguments(client)
    args["timeout"] = 11
    try:
        with pytest.raises(BaseLLMException) as caught:
            handler(clock).image_generation(**args)
    finally:
        client.client.close()
    assert clock.now == 11
    assert caught.value.status_code == 408
    assert get_submission_outcome(caught.value) == "accepted"
    assert get_provider_task_id(caught.value) == "task/original;id"
    assert [r.method for r in calls] == ["POST", "GET", "GET"]
