import asyncio
import copy
from types import SimpleNamespace
from typing import Any

import orjson
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient
from starlette.requests import Request
from starlette.responses import Response

from litellm.proxy._types import ProxyException, UserAPIKeyAuth
from litellm.proxy.common_request_processing import require_resolved_model
from litellm.proxy.image_endpoints import endpoints


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
            data={"prompt": "keep the subject", "model": "gpt-image-2"},
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
    assert [image.read() for image in captured_data["image"]] == [b"first", b"second"]
    assert [mask.read() for mask in captured_data["mask"]] == [b"mask"]


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
