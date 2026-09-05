import json
from unittest.mock import Mock

import httpx
import pytest

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.submission_utils import get_submission_outcome
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.zexapi.image_generation.transformation import (
    ZexAPIBananaImageGenerationConfig,
    ZexAPIImageGenerationConfig,
    get_zexapi_image_generation_config,
)


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


@pytest.mark.parametrize(
    ("config", "model"),
    (
        (ZexAPIImageGenerationConfig(), "image2"),
        (ZexAPIBananaImageGenerationConfig(), "gemini-3.1-flash-image-preview"),
    ),
)
def test_zexapi_invalid_success_response_has_unknown_submission_outcome(config, model):
    with pytest.raises(BaseLLMException) as exc_info:
        config.transform_image_generation_response(
            model=model,
            raw_response=httpx.Response(200, text="not-json"),
            model_response=litellm.ImageResponse(),
            logging_obj=Mock(),
            request_data={},
            optional_params={},
            litellm_params={},
            encoding=None,
        )

    assert get_submission_outcome(exc_info.value) == "unknown"


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


def test_zexapi_image2_maps_aspect_ratio_alias():
    mapped = ZexAPIImageGenerationConfig().map_openai_params(
        non_default_params={"aspect_ratio": "16:9", "n": 1},
        optional_params={},
        model="image2",
        drop_params=False,
    )

    assert mapped == {"size": "1280x720", "response_format": "url"}


def test_zexapi_image2_rejects_multiple_images():
    with pytest.raises(litellm.UnsupportedParamsError, match="supports n=1 only"):
        ZexAPIImageGenerationConfig().map_openai_params(
            non_default_params={"n": 2},
            optional_params={},
            model="image2",
            drop_params=False,
        )


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


@pytest.mark.parametrize("resolution", ["2K", "4K"])
def test_zexapi_image2_rejects_unsupported_resolution_before_network(respx_mock, resolution):
    route = respx_mock.post("https://zexapi.com/v1/images/generations").respond(json={"data": []})

    with pytest.raises(litellm.UnsupportedParamsError, match="does not support resolution"):
        litellm.image_generation(
            model="zexapi/image2",
            prompt="poster",
            api_key="test-key",
            resolution=resolution,
        )

    assert not route.called


@pytest.mark.parametrize(
    "resolution,ratio,expected_size",
    [
        ("1K", "1:1", "1024x1024"),
        ("2k", "16:9", "2560x1440"),
        ("4K", "21:9", "3696x1584"),
    ],
)
def test_zexapi_gpt_image2_maps_resolution_to_provider_pixel_size(resolution, ratio, expected_size):
    mapped = ZexAPIImageGenerationConfig().map_openai_params(
        non_default_params={"aspect_ratio": ratio, "resolution": resolution},
        optional_params={},
        model="gpt-image2",
        drop_params=False,
    )

    assert mapped == {"size": expected_size, "response_format": "url"}


def test_zexapi_gpt_image2_infers_resolution_from_native_pixel_size():
    mapped = ZexAPIImageGenerationConfig().map_openai_params(
        non_default_params={"size": "2880x2880"},
        optional_params={},
        model="gpt-image2",
        drop_params=False,
    )

    assert mapped == {"size": "2880x2880", "response_format": "url"}


def test_zexapi_public_gpt_image2_consumes_resolution_before_network(respx_mock):
    route = respx_mock.post("https://zexapi.com/v1/images/generations").respond(
        json={"created": 1782108238, "data": [{"url": "https://files.example/image.png"}]}
    )

    response = litellm.image_generation(
        model="zexapi/gpt-image2",
        prompt="poster",
        api_key="test-key",
        size="16:9",
        resolution="4K",
    )

    assert response.data[0].url == "https://files.example/image.png"
    assert json.loads(route.calls[0].request.content) == {
        "model": "gpt-image2",
        "prompt": "poster",
        "size": "3840x2160",
        "response_format": "url",
    }


def test_zexapi_banana_maps_openai_request_to_gemini_native_contract():
    config = ZexAPIBananaImageGenerationConfig()
    mapped = config.map_openai_params(
        non_default_params={
            "size": "16:9",
            "imageConfig": {"imageSize": "2k"},
            "image_url": [
                "https://files.example/reference.png",
                "data:image/png;base64,aW1hZ2U=",
            ],
        },
        optional_params={},
        model="gemini-3.1-flash-image-preview",
        drop_params=False,
    )
    request = config.transform_image_generation_request(
        model="gemini-3.1-flash-image-preview",
        prompt="city",
        optional_params=mapped,
        litellm_params={},
        headers={},
    )

    assert isinstance(
        get_zexapi_image_generation_config("gemini-3.1-flash-image-preview"),
        ZexAPIBananaImageGenerationConfig,
    )
    assert config.get_complete_url(None, None, "gemini-3.1-flash-image-preview", {}, {}) == (
        "https://zexapi.com/v1beta/models/gemini-3.1-flash-image-preview:generateContent"
    )
    assert request == {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": "city"},
                    {"fileData": {"fileUri": "https://files.example/reference.png"}},
                    {"inlineData": {"mimeType": "image/png", "data": "aW1hZ2U="}},
                ],
            }
        ],
        "generationConfig": {
            "responseModalities": ["IMAGE"],
            "imageConfig": {"aspectRatio": "16:9", "imageSize": "2K"},
        },
        "response_format": "url",
    }


def test_zexapi_public_banana_image_generation(respx_mock):
    route = respx_mock.post("https://zexapi.com/v1beta/models/gemini-3.1-flash-image-preview:generateContent").respond(
        json={
            "model": "gemini-3.1-flash-image-preview",
            "data": [{"url": "https://files.example/banana.png"}],
        }
    )

    response = litellm.image_generation(
        model="zexapi/gemini-3.1-flash-image-preview",
        prompt="city",
        api_key="test-key",
        size="16:9",
        resolution="2k",
        image_url="https://files.example/reference.png",
    )

    assert response.data[0].url == "https://files.example/banana.png"
    assert json.loads(route.calls[0].request.content) == {
        "contents": [
            {
                "role": "user",
                "parts": [
                    {"text": "city"},
                    {"fileData": {"fileUri": "https://files.example/reference.png"}},
                ],
            }
        ],
        "generationConfig": {
            "responseModalities": ["IMAGE"],
            "imageConfig": {"aspectRatio": "16:9", "imageSize": "2K"},
        },
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
        ("gpt-image-3", "1024x1024", "url"),
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


def test_zexapi_banana_rejects_non_object_image_config():
    with pytest.raises(litellm.UnsupportedParamsError, match="imageConfig must be an object"):
        ZexAPIBananaImageGenerationConfig().map_openai_params(
            non_default_params={"imageConfig": ["2K"]},
            optional_params={},
            model="gemini-3.1-flash-image-preview",
            drop_params=False,
        )


def test_zexapi_banana_rejects_unknown_image_config_fields():
    with pytest.raises(litellm.UnsupportedParamsError, match="does not support imageConfig field.*google_search"):
        ZexAPIBananaImageGenerationConfig().map_openai_params(
            non_default_params={"imageConfig": {"imageSize": "2K", "google_search": True}},
            optional_params={},
            model="gemini-3.1-flash-image-preview",
            drop_params=False,
        )
