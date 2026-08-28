import json

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.toapis.image_generation.transformation import ToAPISImageGenerationConfig


def test_toapis_image_request_and_base_url(monkeypatch):
    monkeypatch.setenv("TOAPIS_API_KEY", "test-key")
    config = ToAPISImageGenerationConfig()

    assert config.get_complete_url(None, None, "gpt-image-2", {}, {}) == "https://toapis.com/v1/images/generations"
    assert (
        config.get_complete_url("https://example.com/v1", None, "gpt-image-2", {}, {})
        == "https://example.com/v1/images/generations"
    )
    assert config.validate_environment({}, "gpt-image-2", [], {}, {}) == {
        "Authorization": "Bearer test-key",
        "Content-Type": "application/json",
    }
    mapped = config.map_openai_params(
        non_default_params={"size": "16:9"},
        optional_params={},
        model="gpt-image-2",
        drop_params=False,
    )
    assert mapped == {"size": "16:9", "resolution": "1k", "response_format": "url"}
    assert config.transform_image_generation_request(
        model="gpt-image-2",
        prompt="city",
        optional_params=mapped,
        litellm_params={},
        headers={},
    ) == {
        "model": "gpt-image-2",
        "prompt": "city",
        "size": "16:9",
        "resolution": "1k",
        "response_format": "url",
    }


def test_toapis_api_base_rejects_query_and_fragment():
    config = ToAPISImageGenerationConfig()

    with pytest.raises(ValueError, match="query string or fragment"):
        config.get_complete_url("https://example.com/v1?tenant=a", None, "gpt-image-2", {}, {})
    with pytest.raises(ValueError, match="query string or fragment"):
        config.get_complete_url("https://example.com/v1#fragment", None, "gpt-image-2", {}, {})


def test_toapis_public_image_generation_forwards_provider_fields(respx_mock):
    route = respx_mock.post("https://toapis.com/v1/images/generations").respond(
        json={
            "id": "task_img_123",
            "object": "generation.task",
            "model": "gpt-image-2",
            "status": "completed",
            "progress": 100,
            "created_at": 1703884800,
            "result": {"type": "image", "data": [{"url": "https://files.example/image.png"}]},
        }
    )

    response = litellm.image_generation(
        model="toapis/gpt-image-2",
        prompt="city",
        api_key="test-key",
        size="16:9",
    )

    assert response.data[0].url == "https://files.example/image.png"
    assert route.calls[0].request.headers["Authorization"] == "Bearer test-key"
    assert json.loads(route.calls[0].request.content) == {
        "model": "gpt-image-2",
        "prompt": "city",
        "size": "16:9",
        "resolution": "1k",
        "response_format": "url",
    }


@pytest.mark.parametrize(
    "model,size,response_format",
    [
        ("other-model", "1024x1024", "url"),
        ("gpt-image-2", "999x999", "url"),
        ("gpt-image-2", "1024x1024", "b64_json"),
    ],
)
def test_toapis_image_service_rejects_unsupported_contract(model, size, response_format):
    with pytest.raises(litellm.UnsupportedParamsError):
        ToAPISImageGenerationConfig().map_openai_params(
            non_default_params={"size": size, "response_format": response_format},
            optional_params={},
            model=model,
            drop_params=False,
        )


@pytest.mark.asyncio
async def test_toapis_public_async_image_generation():
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "id": "task_img_123",
                "object": "generation.task",
                "model": "gpt-image-2",
                "status": "completed",
                "progress": 100,
                "created_at": 1703884800,
                "result": {"type": "image", "data": [{"url": "https://files.example/image.png"}]},
            },
            request=request,
        )

    handler = AsyncHTTPHandler()
    await handler.close()
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        handler.client = client
        response = await litellm.aimage_generation(
            model="toapis/gpt-image-2",
            prompt="city",
            api_key="test-key",
            client=handler,
        )

    assert response.data[0].url == "https://files.example/image.png"
