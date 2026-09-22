import base64
import copy
import json
from unittest.mock import Mock

import httpx
import pytest

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.submission_utils import get_submission_outcome, is_reexecution_blocked
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.toapis.image_generation import reference_upload
from litellm.llms.toapis.image_generation.handler import ToAPISImageGeneration
from litellm.llms.toapis.image_generation.transformation import ToAPISImageGenerationConfig

PNG = base64.b64decode("iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+/l9sAAAAASUVORK5CYII=")
INLINE = "data:image/png;base64," + base64.b64encode(PNG).decode()
REMOTE = "https://cdn.example/reference.png"
UPLOADED = "https://files.example/uploaded.png"
BASE = "https://gateway.example/prefix/v1"


@pytest.fixture(autouse=True)
def mockable_transport(monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)


def upload_response(url=UPLOADED):
    return {
        "success": True,
        "message": "",
        "data": {"id": "upload-one", "url": url, "mime_type": "image/png", "size": len(PNG)},
    }


def completed():
    return {
        "id": "task-one",
        "object": "generation.task",
        "status": "completed",
        "result": {"type": "image", "data": [{"url": "https://files.example/result.png"}]},
    }


def kwargs(model, params):
    return dict(
        model=model,
        prompt="blue teapot",
        optional_params=params,
        litellm_params={"api_base": BASE},
        api_key="test-key",
        logging_obj=Mock(),
        timeout=10,
        extra_headers={"Content-Type": "application/json", "Idempotency-Key": "generation-one"},
    )


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize(
    "model,field,objects",
    [
        ("gpt-image-2.5-flare", "reference_images", False),
        ("gpt-image-2.5-sunburst-vip", "reference_images", False),
        ("gemini-3-pro-image-preview", "image_urls", True),
    ],
)
@pytest.mark.asyncio
async def test_upload_then_generate_preserves_order_and_payload(async_mode, model, field, objects, respx_mock):
    original = [INLINE, REMOTE, {"url": INLINE}]
    snapshot = copy.deepcopy(original)
    params = ToAPISImageGenerationConfig().map_openai_params({"image_url": original, "size": "1:1"}, {}, model, False)
    uploads = respx_mock.post(BASE + "/uploads/images").respond(200, json=upload_response())
    creates = respx_mock.post(BASE + "/images/generations").respond(200, json=completed())
    handler = ToAPISImageGeneration()
    if async_mode:
        client = AsyncHTTPHandler()
        try:
            result = await handler.async_image_generation(**kwargs(model, params), client=client)
        finally:
            await client.client.aclose()
    else:
        client = HTTPHandler()
        try:
            result = handler.image_generation(**kwargs(model, params), client=client)
        finally:
            client.client.close()
    assert result.data[0].url == "https://files.example/result.png"
    assert uploads.call_count == creates.call_count == 1
    assert [call.request.url.path for call in respx_mock.calls] == [
        "/prefix/v1/uploads/images",
        "/prefix/v1/images/generations",
    ]
    upload = uploads.calls[0].request
    assert upload.headers["authorization"] == "Bearer test-key"
    assert upload.headers["content-type"].startswith("multipart/form-data; boundary=")
    assert "idempotency-key" not in upload.headers
    assert b'name="file"; filename="reference.png"' in upload.content
    assert b"Content-Type: image/png" in upload.content
    assert PNG in upload.content
    create = creates.calls[0].request
    assert create.headers["idempotency-key"] == "generation-one"
    body = json.loads(create.content)
    urls = [UPLOADED, REMOTE, UPLOADED]
    assert body[field] == ([{"url": url} for url in urls] if objects else urls)
    assert b"base64" not in create.content
    assert original == snapshot
    assert params[field][0] == ({"url": INLINE} if objects else INLINE)


@pytest.mark.parametrize("mime", ["image/png", "image/jpeg", "image/webp", "image/gif"])
def test_supported_data_url_mime_types(mime):
    image = reference_upload.decode_reference(f"data:{mime};base64,AQID")
    assert image.content == b"\x01\x02\x03"
    assert image.mime_type == mime


@pytest.mark.parametrize(
    "value",
    [
        "data:image/png;base64,",
        "data:image/png;base64,%%%",
        "data:image/png;base64,A",
        "data:image/svg+xml;base64,AQID",
        "data:text/plain;base64,AQID",
        "data:image/png,AQID",
        "blob:https://example.com/one",
        "file:///tmp/image.png",
        "AQID",
        "https://",
        "",
    ],
)
def test_invalid_references_fail_pure_mapper(value, respx_mock):
    with pytest.raises(litellm.UnsupportedParamsError):
        ToAPISImageGenerationConfig().map_openai_params(
            {"image_url": [REMOTE, value]}, {}, "gpt-image-2.5-flare", False
        )
    assert len(respx_mock.calls) == 0


def test_decoded_size_boundary(monkeypatch):
    monkeypatch.setattr(reference_upload, "MAX_IMAGE_BYTES", 2)
    assert reference_upload.decode_reference("data:image/png;base64,AQI=").content == b"\x01\x02"
    # Same encoded length, but decoding reveals an oversized file.
    with pytest.raises(ValueError, match="exceeds"):
        reference_upload.decode_reference("data:image/png;base64,AQID")
    with pytest.raises(ValueError, match="exceeds"):
        reference_upload.decode_reference("data:image/png;base64,AQIDBA==")


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.parametrize(
    "failure", ["http", "timeout", "envelope", "malformed", "missing", "bad_url", "redirect", "invalid_later_reference"]
)
@pytest.mark.asyncio
async def test_upload_failure_never_submits_or_retries(async_mode, failure, respx_mock):
    def respond(request):
        if failure == "timeout":
            raise httpx.ReadTimeout("lost upload response", request=request)
        if failure == "http":
            return httpx.Response(429, text="rate limit")
        if failure == "envelope":
            return httpx.Response(200, json={"success": False, "message": "failed"})
        if failure == "malformed":
            return httpx.Response(200, text="not JSON")
        if failure == "missing":
            return httpx.Response(200, json={"success": True, "message": ""})
        if failure == "bad_url":
            return httpx.Response(200, json=upload_response(""))
        if failure == "redirect":
            return httpx.Response(307, headers={"Location": "https://other.example/upload"})
        pytest.fail("All references must be validated before upload")

    upload = respx_mock.post(BASE + "/uploads/images").mock(side_effect=respond)
    params = {"reference_images": [INLINE, "blob:invalid"] if failure == "invalid_later_reference" else [INLINE]}
    handler = ToAPISImageGeneration()
    if async_mode:
        client = AsyncHTTPHandler()
        try:
            with pytest.raises(BaseLLMException) as error:
                await handler.async_image_generation(**kwargs("gpt-image-2.5-flare", params), client=client)
        finally:
            await client.client.aclose()
    else:
        client = HTTPHandler()
        try:
            with pytest.raises(BaseLLMException) as error:
                handler.image_generation(**kwargs("gpt-image-2.5-flare", params), client=client)
        finally:
            client.client.close()
    assert get_submission_outcome(error.value) == "rejected"
    assert is_reexecution_blocked(error.value)
    assert upload.call_count == (0 if failure == "invalid_later_reference" else 1)
    assert len(respx_mock.calls) == upload.call_count


@pytest.mark.asyncio
async def test_url_only_skips_upload_and_extra_body_inline_is_prepared(respx_mock):
    creates = respx_mock.post(BASE + "/images/generations").respond(200, json=completed())
    uploads = respx_mock.post(BASE + "/uploads/images").respond(200, json=upload_response())
    client = AsyncHTTPHandler()
    try:
        handler = ToAPISImageGeneration()
        await handler.async_image_generation(
            **kwargs("gpt-image-2.5-flare", {"reference_images": [REMOTE]}), client=client
        )
        assert uploads.call_count == 0
        await handler.async_image_generation(
            **kwargs("gpt-image-2.5-flare", {}), client=client, extra_body={"reference_images": [INLINE]}
        )
        assert uploads.call_count == 1
        assert json.loads(creates.calls[-1].request.content)["reference_images"] == [UPLOADED]
    finally:
        await client.client.aclose()


def reference_router():
    return litellm.Router(
        model_list=[
            {
                "model_name": "images",
                "litellm_params": {
                    "model": "toapis/gpt-image-2.5-flare" + suffix,
                    "api_base": BASE,
                    "api_key": "test-key",
                    "order": order,
                    "num_retries": 0,
                },
            }
            for order, suffix in [(1, ""), (2, "-vip")]
        ],
        num_retries=2,
        enable_weighted_failover=True,
        weighted_failover_policy={"failure_scope": "deployment", "status_codes": [400, 429]},
    )


@pytest.mark.asyncio
async def test_router_upload_failure_does_not_fall_back(respx_mock):
    router = reference_router()
    upload = respx_mock.post(BASE + "/uploads/images").respond(429, text="rate limit")
    with pytest.raises(litellm.RateLimitError) as error:
        await router.aimage_generation(model="images", prompt="blue teapot", image_url=INLINE)
    assert is_reexecution_blocked(error.value)
    assert get_submission_outcome(error.value) == "rejected"
    assert upload.call_count == len(respx_mock.calls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["completed", "rejected", "accepted_failed"])
async def test_router_inline_reference_preserves_generation_failover_rules(outcome, respx_mock):
    router = reference_router()
    uploads = respx_mock.post(BASE + "/uploads/images").respond(200, json=upload_response())
    bodies = []

    def generate(request):
        body = json.loads(request.content)
        bodies.append(body)
        assert body["reference_images"] == [UPLOADED, REMOTE]
        if outcome == "rejected" and not body["model"].endswith("-vip"):
            return httpx.Response(400, text="no available channel")
        if outcome == "accepted_failed":
            return httpx.Response(200, json={**completed(), "status": "failed", "result": None})
        return httpx.Response(200, json=completed())

    respx_mock.post(BASE + "/images/generations").mock(side_effect=generate)
    if outcome == "accepted_failed":
        with pytest.raises(Exception) as error:
            await router.aimage_generation(model="images", prompt="teapot", image_url=[INLINE, REMOTE])
        assert get_submission_outcome(error.value) == "accepted"
    else:
        result = await router.aimage_generation(model="images", prompt="teapot", image_url=[INLINE, REMOTE])
        assert result.data[0].url == "https://files.example/result.png"
    expected = ["gpt-image-2.5-flare"] + (["gpt-image-2.5-flare-vip"] if outcome == "rejected" else [])
    assert [body["model"] for body in bodies] == expected
    # Each deployment uses its own upload credentials; no cross-deployment cache.
    assert uploads.call_count == len(expected)


def test_sync_sdk_accepts_inline_reference_with_full_endpoint_base(respx_mock):
    uploads = respx_mock.post(BASE + "/uploads/images").respond(200, json=upload_response())
    creates = respx_mock.post(BASE + "/images/generations").respond(200, json=completed())
    result = litellm.image_generation(
        model="toapis/gpt-image-2.5-flare",
        api_base=BASE + "/images/generations",
        api_key="test-key",
        prompt="teapot",
        image_url=INLINE,
    )
    assert result.data[0].url == "https://files.example/result.png"
    assert uploads.call_count == creates.call_count == 1
    assert json.loads(creates.calls[0].request.content)["reference_images"] == [UPLOADED]
