import json

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import litellm
from litellm.proxy import proxy_server
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.video_endpoints import endpoints
from litellm.router import Router


@pytest.mark.parametrize(
    "model,operation,seconds",
    [
        ("seedance-2", None, "5"),
        ("seedance-2-5", None, "5"),
        ("seedance-2-5", "generate", "5"),
        ("seedance-2-5", "edit", None),
        ("seedance-2-5", "edit", "-1"),
        ("seedance-2-5", "extend", None),
        ("seedance-2-5", "extend", "15"),
    ],
)
def test_json_gateway_request_survives_proxy_router_and_provider(respx_mock, monkeypatch, model, operation, seconds):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(proxy_server, "general_settings", {})
    monkeypatch.setattr(proxy_server, "user_model", None)
    gateway = Router(
        model_list=[
            {
                "model_name": "video",
                "litellm_params": {
                    "model": f"toapis/{model}",
                    "api_key": "fixture-key",
                    "api_base": "https://fixture-upstream.example",
                },
            }
        ],
        num_retries=0,
    )

    async def dispatch(self, **kwargs):
        assert kwargs["route_type"] == "avideo_generation"
        return await gateway.avideo_generation(**self.data)

    async def auth():
        return UserAPIKeyAuth(api_key="fixture-proxy-key")

    monkeypatch.setattr(ProxyBaseLLMRequestProcessing, "base_process_llm_request", dispatch)
    app = FastAPI()
    app.include_router(endpoints.router)
    app.dependency_overrides[endpoints.user_api_key_auth] = auth
    native = respx_mock.post("https://fixture-upstream.example/v1/videos/generations").mock(
        return_value=httpx.Response(200, json={"id": "fixture-task", "object": "generation.task", "status": "queued"})
    )
    refs = [
        {"type": kind, "role": "reference", "url": f"https://media.example/{kind}-{index}"}
        for index, kind in enumerate(["image", "image", "video", "audio"])
    ]
    payload = {
        "model": "video",
        "prompt": "fixture",
        "resolution": "1080p",
        "references": refs,
    }
    if operation is not None:
        payload["operation"] = operation
    if seconds is not None:
        payload["seconds"] = seconds
    if operation not in ("edit", "extend"):
        payload["aspect_ratio"] = "16:9"
    with TestClient(app) as client:
        response = client.post(
            "/v1/videos",
            json=payload,
        )
    assert response.status_code == 200, response.text
    assert response.json()["id"].startswith("video_")
    assert len(native.calls) == 1
    outgoing = json.loads(native.calls[0].request.content)
    assert outgoing["resolution"] == "1080p"
    assert outgoing["aspect_ratio"] == ("adaptive" if operation in ("edit", "extend") else "16:9")
    if model == "seedance-2-5":
        assert outgoing["video_operation"] == (operation or "generate")
    assert outgoing["duration"] == (int(seconds) if seconds is not None else -1)
    assert len(outgoing["image_with_roles"]) == 2
    assert outgoing["video_with_roles"] == [{"url": refs[2]["url"], "role": "reference_video"}]
    assert outgoing["audio_with_roles"] == [{"url": refs[3]["url"], "role": "reference_audio"}]
    assert not {"size", "width", "height", "references", "operation", "seconds"}.intersection(outgoing)
