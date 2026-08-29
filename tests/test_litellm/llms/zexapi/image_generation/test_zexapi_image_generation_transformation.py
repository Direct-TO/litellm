import json
from unittest.mock import Mock

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.zexapi.image_generation.transformation import ZexAPIImageGenerationConfig


def test_zexapi_image_generation_normalizes_to_url(monkeypatch):
    monkeypatch.setenv("ZEXAPI_API_KEY", "test-key")
    config = ZexAPIImageGenerationConfig()
    result = config.transform_image_generation_response(
        model="image2",
        raw_response=httpx.Response(
            200,
            json={
                "created": 1782108238,
                "data": [
                    {"url": "https://files.example/image.png", "b64_json": "aW1hZ2U="},
                    {"url": "https://files.example/image-2.png", "b64_json": "aW1hZ2Uy"},
                ],
            },
        ),
        model_response=litellm.ImageResponse(),
        logging_obj=Mock(),
        request_data={},
        optional_params={},
        litellm_params={},
        encoding=None,
    )

    assert config.get_complete_url(None, None, "image2", {}, {}) == "https://zexapi.com/v1/images/generations"
    assert config.validate_environment({}, "image2", [], {}, {}) == {
        "Authorization": "Bearer test-key",
        "Content-Type": "application/json",
    }
    assert result.data[0].url == "https://files.example/image.png"
    assert result.data[0].b64_json is None
    assert result.data[1].url == "https://files.example/image-2.png"
    assert result.data[1].b64_json is None


def test_zexapi_generation_request_is_openai_shaped():
    config = ZexAPIImageGenerationConfig()
    mapped = config.map_openai_params(
        non_default_params={"size": "9:16"},
        optional_params={},
        model="image2",
        drop_params=False,
    )
    assert mapped == {"size": "720x1280", "response_format": "url"}
    assert config.transform_image_generation_request(
        model="image2",
        prompt="poster",
        optional_params=mapped,
        litellm_params={},
        headers={},
    ) == {
        "model": "image2",
        "prompt": "poster",
        "size": "720x1280",
        "response_format": "url",
    }


def test_zexapi_api_base_rejects_query_and_fragment():
    config = ZexAPIImageGenerationConfig()

    with pytest.raises(ValueError, match="query string or fragment"):
        config.get_complete_url("https://example.com/v1?tenant=a", None, "image2", {}, {})
    with pytest.raises(ValueError, match="query string or fragment"):
        config.get_complete_url("https://example.com/v1#fragment", None, "image2", {}, {})


def test_zexapi_public_image_generation_normalizes_to_url(respx_mock):
    route = respx_mock.post("https://zexapi.com/v1/images/generations").respond(
        json={
            "created": 1782108238,
            "data": [{"url": "https://files.example/image.png", "b64_json": "aW1hZ2U="}],
        }
    )

    response = litellm.image_generation(
        model="zexapi/image2",
        prompt="poster",
        api_key="test-key",
        size="1024x1024",
    )

    assert response.data[0].url == "https://files.example/image.png"
    assert response.data[0].b64_json is None
    assert route.calls[0].request.headers["Authorization"] == "Bearer test-key"
    assert json.loads(route.calls[0].request.content) == {
        "model": "image2",
        "prompt": "poster",
        "size": "1024x1024",
        "response_format": "url",
    }


@pytest.mark.asyncio
async def test_zexapi_public_async_image_generation():
    def respond(request: httpx.Request) -> httpx.Response:
        return httpx.Response(
            200,
            json={
                "created": 1782108238,
                "data": [{"url": "https://files.example/image.png", "b64_json": "aW1hZ2U="}],
            },
            request=request,
        )

    handler = AsyncHTTPHandler()
    await handler.close()
    async with httpx.AsyncClient(transport=httpx.MockTransport(respond)) as client:
        handler.client = client
        response = await litellm.aimage_generation(
            model="zexapi/image2",
            prompt="poster",
            api_key="test-key",
            client=handler,
        )

    assert response.data[0].url == "https://files.example/image.png"
    assert response.data[0].b64_json is None


@pytest.mark.parametrize(
    "model,size,response_format",
    [
        ("gpt-image2", "1024x1024", "url"),
        ("image2", "999x999", "url"),
        ("image2", "1024x1024", "b64_json"),
    ],
)
def test_zexapi_image_service_rejects_unsupported_contract(model, size, response_format):
    with pytest.raises(litellm.UnsupportedParamsError):
        ZexAPIImageGenerationConfig().map_openai_params(
            non_default_params={"size": size, "response_format": response_format},
            optional_params={},
            model=model,
            drop_params=False,
        )
