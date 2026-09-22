import asyncio
import copy
from datetime import datetime, timezone
from email.parser import BytesParser
from email.policy import default
from types import SimpleNamespace
from typing import Any

import httpx
import orjson
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.responses import Response

import litellm
from litellm.llms.base_llm.submission_utils import mark_submission_outcome
from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.common_request_processing import require_resolved_model
from litellm.proxy.image_endpoints import endpoints
from litellm.proxy.spend_tracking.spend_tracking_utils import get_logging_payload


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["rejected", "accepted", "unknown", None])
async def test_image_failure_exposes_only_adapter_submission_provenance(monkeypatch, outcome):
    async def passthrough(**kwargs):
        return kwargs["data"]

    async def fail(**kwargs):
        error = litellm.APIConnectionError(message="Cannot connect to host", model="gpt-image-2", llm_provider="toapis")
        if outcome:
            mark_submission_outcome(error, outcome)
        raise error

    async def ignore(**kwargs):
        pass

    monkeypatch.setattr("litellm.proxy.proxy_server.add_litellm_data_to_request", passthrough)
    monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {})
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.proxy_logging_obj", SimpleNamespace(pre_call_hook=passthrough, post_call_failure_hook=ignore))
    monkeypatch.setattr(endpoints, "route_request", fail)

    async def receive():
        return {"type": "http.request", "body": orjson.dumps({"model": "gpt-image-2", "prompt": "test"}), "more_body": False}

    request = Request({"type": "http", "method": "POST", "path": "/v1/images/generations", "headers": []}, receive)
    with pytest.raises(ProxyException) as caught:
        await endpoints.image_generation(request, Response(), UserAPIKeyAuth())
    assert caught.value.headers.get("x-litellm-submission-outcome") == outcome


def test_generation_model_is_required_when_no_server_default_is_configured():
    with pytest.raises(ProxyException) as exc_info:
        require_resolved_model(None)

    error = exc_info.value
    assert getattr(error, "code", None) == "400"
    assert getattr(error, "param", None) == "model"
    assert getattr(error, "openai_code", None) == "missing_required_parameter"
    assert require_resolved_model("gpt-image-2") == "gpt-image-2"


@pytest.mark.parametrize("image_field", ["image", "image[]"])
def test_image_edit_preserves_repeated_multipart_images_and_mask(monkeypatch, image_field):
    captured_data: dict[str, Any] = {}

    async def fake_base_process(self, **kwargs):
        captured_data.update(self.data)
        return {"data": [{"url": "https://files.example/result.png"}]}

    monkeypatch.setattr(
        endpoints.ProxyBaseLLMRequestProcessing,
        "base_process_llm_request",
        fake_base_process,
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {})
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.proxy_config", {})
    monkeypatch.setattr("litellm.proxy.proxy_server.user_model", None)

    app = FastAPI()
    app.include_router(endpoints.router)
    app.dependency_overrides[endpoints.user_api_key_auth] = UserAPIKeyAuth
    with TestClient(app) as client:
        response = client.post(
            "/v1/images/edits",
            data={"prompt": "keep the subject", "model": "gpt-image-2", "aspect_ratio": "16:9", "resolution": "2K"},
            files=[
                (image_field, ("reference-1.png", b"first", "image/png")),
                (image_field, ("reference-2.png", b"second", "image/png")),
                ("mask", ("mask.png", b"mask", "image/png")),
            ],
        )

    assert response.status_code == 200
    assert response.json() == {"data": [{"url": "https://files.example/result.png"}]}
    assert captured_data["prompt"] == "keep the subject"
    assert captured_data["model"] == "gpt-image-2"
    assert captured_data["aspect_ratio"] == "16:9"
    assert captured_data["resolution"] == "2K"
    assert [image.read() for image in captured_data["image"]] == [b"first", b"second"]
    assert [mask.read() for mask in captured_data["mask"]] == [b"mask"]


def test_zexapi_edit_proxy_maps_ratio_and_tier_through_router(monkeypatch, respx_mock):
    captured = []

    def respond(request):
        captured.append(request)
        return httpx.Response(200, json={"created": 1, "data": [{"url": "https://files.example/result.png"}]})

    respx_mock.post("https://edit.example/v1/images/edits").mock(side_effect=respond)
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    router = litellm.Router(
        model_list=[
            {
                "model_name": "gpt-image-2",
                "litellm_params": {
                    "model": "zexapi/" + model,
                    "api_base": "https://edit.example/v1",
                    "api_key": "test-key",
                },
                "model_info": {"id": model, "supported_endpoints": ["/v1/images/edits"]},
            }
            for model in ["image2", "gpt-image2"]
        ],
        num_retries=0,
    )

    async def process(self, **kwargs):
        return await router.aimage_edit(
            **{
                key: self.data[key]
                for key in ["model", "prompt", "image", "mask", "n", "aspect_ratio", "resolution"]
                if key in self.data
            }
        )

    monkeypatch.setattr(endpoints.ProxyBaseLLMRequestProcessing, "base_process_llm_request", process)
    monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {})
    monkeypatch.setattr("litellm.proxy.proxy_server.user_model", None)
    app = FastAPI()
    app.include_router(endpoints.router)
    app.dependency_overrides[endpoints.user_api_key_auth] = UserAPIKeyAuth
    with TestClient(app) as client:
        response = client.post(
            "/v1/images/edits",
            data={
                "model": "gpt-image-2",
                "prompt": "blue circle",
                "n": "1",
                "aspect_ratio": "16:9",
                "resolution": "2K",
            },
            files=[("image", ("reference.png", b"\x89PNG\r\n\x1a\nfixture", "image/png"))],
        )
    assert response.status_code == 200
    assert len(captured) == 1
    request = captured[0]
    message = BytesParser(policy=default).parsebytes(
        f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode() + request.content
    )
    fields = {
        part.get_param("name", header="content-disposition"): part.get_payload(decode=True).decode()
        for part in message.iter_parts()
        if part.get_filename() is None
    }
    assert fields == {"model": "gpt-image2", "prompt": "blue circle", "n": "1", "size": "2560x1440"}
    assert response.json()["data"][0]["url"] == "https://files.example/result.png"


@pytest.mark.asyncio
async def test_image_generation_prompt_rerouting(monkeypatch):
    """Ensure image prompts are exposed to guardrails and restored afterwards."""

    async def fake_add_litellm_data_to_request(**kwargs):
        return kwargs["data"]

    async def fake_update_request_status(**_: Any) -> None:
        await asyncio.sleep(0)

    proxy_logger_calls: dict[str, Any] = {}

    async def fake_pre_call_hook(*, user_api_key_dict, data, call_type):  # type: ignore[override]
        proxy_logger_calls["pre_call_input"] = copy.deepcopy(data)
        modified = {
            **data,
            "messages": [
                {
                    "role": "user",
                    "content": "sanitized prompt",
                }
            ],
        }
        return modified

    async def fake_post_call_failure_hook(**_: Any) -> None:
        return None

    async def fake_post_call_success_hook(*, data, user_api_key_dict, response):
        return response

    async def fake_post_call_response_headers_hook(**kwargs):
        return {"x-callback-test": "value"}

    fake_proxy_logger = SimpleNamespace(
        pre_call_hook=fake_pre_call_hook,
        update_request_status=fake_update_request_status,
        post_call_failure_hook=fake_post_call_failure_hook,
        post_call_success_hook=fake_post_call_success_hook,
        post_call_response_headers_hook=fake_post_call_response_headers_hook,
    )

    captured_route_request_data: dict[str, Any] = {}

    async def fake_route_request(*, data, **kwargs):  # type: ignore[override]
        captured_route_request_data.update(data)

        async def _inner():
            class FakeResponse(dict):
                _hidden_params = {}

            return FakeResponse(result="ok")

        return _inner()

    scope = {
        "type": "http",
        "method": "POST",
        "path": "/v1/images/generations",
        "headers": [],
    }
    body = orjson.dumps({"model": "gpt-image-2", "prompt": "original prompt"})

    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    request = Request(scope, receive)
    response = Response()
    user_api_key = UserAPIKeyAuth()

    monkeypatch.setattr(
        "litellm.proxy.proxy_server.add_litellm_data_to_request",
        fake_add_litellm_data_to_request,
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {})
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.proxy_config", {})
    monkeypatch.setattr("litellm.proxy.proxy_server.proxy_logging_obj", fake_proxy_logger)
    monkeypatch.setattr("litellm.proxy.proxy_server.user_model", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.version", "test-version")
    monkeypatch.setattr(
        "litellm.proxy.common_request_processing.ProxyBaseLLMRequestProcessing.get_custom_headers",
        classmethod(lambda *args, **kwargs: {}),
    )
    monkeypatch.setattr("litellm.proxy.image_endpoints.endpoints.route_request", fake_route_request)

    result = await endpoints.image_generation(
        request=request,
        fastapi_response=response,
        user_api_key_dict=user_api_key,
    )
    await asyncio.sleep(0)

    assert result == {"result": "ok"}
    pre_call_input = proxy_logger_calls["pre_call_input"]
    assert pre_call_input["messages"][0]["content"] == "original prompt"
    assert captured_route_request_data["prompt"] == "sanitized prompt"
    assert "messages" not in captured_route_request_data
    assert response.headers.get("x-callback-test") == "value"


@pytest.mark.asyncio
async def test_image_preflight_failures_have_distinct_stable_spend_log_ids(monkeypatch):
    payloads = []
    routed_ids = []

    async def passthrough(**kwargs):
        return kwargs["data"]

    async def reject_preflight(*, data, **kwargs):
        routed_ids.append(data["litellm_call_id"])
        raise litellm.BadRequestError(
            message="no deployment supporting image generation parameter(s): resolution",
            model=data["model"],
            llm_provider="",
        )

    async def record_failure(*, request_data, original_exception, **kwargs):
        assert isinstance(original_exception, litellm.BadRequestError)
        request_data["litellm_params"] = {"metadata": {"status": "failure"}}
        now = datetime.now(timezone.utc)
        first = get_logging_payload(request_data, {}, now, now)
        repeated = get_logging_payload(request_data, {}, now, now)
        assert first["request_id"] == repeated["request_id"]
        assert orjson.loads(first["metadata"])["status"] == "failure"
        payloads.append(first)

    monkeypatch.setattr("litellm.proxy.proxy_server.add_litellm_data_to_request", passthrough)
    monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {})
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", None)
    monkeypatch.setattr("litellm.proxy.proxy_server.proxy_config", {})
    monkeypatch.setattr("litellm.proxy.proxy_server.user_model", None)
    monkeypatch.setattr(
        "litellm.proxy.proxy_server.proxy_logging_obj",
        SimpleNamespace(
            pre_call_hook=passthrough,
            post_call_failure_hook=record_failure,
        ),
    )
    monkeypatch.setattr(endpoints, "route_request", reject_preflight)

    for _ in range(2):

        async def receive():
            return {
                "type": "http.request",
                "more_body": False,
                "body": orjson.dumps(
                    {
                        "model": "gemini-2.5-flash-image-preview",
                        "prompt": "test",
                        "resolution": "2K",
                        "litellm_call_id": "client-reused-id",
                    }
                ),
            }

        request = Request({"type": "http", "method": "POST", "path": "/v1/images/generations", "headers": []}, receive)
        with pytest.raises(ProxyException) as exc_info:
            await endpoints.image_generation(request, Response(), UserAPIKeyAuth())
        assert str(exc_info.value.code) == "400"

    assert len(payloads) == 2
    assert [payload["request_id"] for payload in payloads] == routed_ids
    assert len(set(routed_ids)) == 2
    assert all(call_id not in ("None", "", "client-reused-id") for call_id in routed_ids)
