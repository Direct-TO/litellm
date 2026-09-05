import json

import httpx
import pytest

import litellm
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.toapis.image_generation.transformation import ToAPISImageGenerationConfig

_GPT_IMAGE_2_PIXEL_CASES = (
    ("1024x1024", "1:1", "1k"),
    ("1536x1024", "3:2", "1k"),
    ("1024x1536", "2:3", "1k"),
    ("1024x768", "4:3", "1k"),
    ("768x1024", "3:4", "1k"),
    ("1280x1024", "5:4", "1k"),
    ("1024x1280", "4:5", "1k"),
    ("1536x864", "16:9", "1k"),
    ("864x1536", "9:16", "1k"),
    ("2048x1024", "2:1", "1k"),
    ("1024x2048", "1:2", "1k"),
    ("2016x864", "21:9", "1k"),
    ("864x2016", "9:21", "1k"),
    ("2048x2048", "1:1", "2k"),
    ("2048x1360", "3:2", "2k"),
    ("1360x2048", "2:3", "2k"),
    ("2048x1536", "4:3", "2k"),
    ("1536x2048", "3:4", "2k"),
    ("2560x2048", "5:4", "2k"),
    ("2048x2560", "4:5", "2k"),
    ("2048x1152", "16:9", "2k"),
    ("1152x2048", "9:16", "2k"),
    ("2688x1344", "2:1", "2k"),
    ("1344x2688", "1:2", "2k"),
    ("2688x1152", "21:9", "2k"),
    ("1152x2688", "9:21", "2k"),
    ("2880x2880", "1:1", "4k"),
    ("3520x2336", "3:2", "4k"),
    ("2336x3520", "2:3", "4k"),
    ("3312x2480", "4:3", "4k"),
    ("2480x3312", "3:4", "4k"),
    ("3216x2576", "5:4", "4k"),
    ("2576x3216", "4:5", "4k"),
    ("3840x2160", "16:9", "4k"),
    ("2160x3840", "9:16", "4k"),
    ("3840x1920", "2:1", "4k"),
    ("1920x3840", "1:2", "4k"),
    ("3840x1648", "21:9", "4k"),
    ("1648x3840", "9:21", "4k"),
)


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
            "result": {
                "type": "image",
                "data": [
                    {"url": "https://files.example/image.png"},
                    {"url": "https://files.example/image-2.png"},
                ],
            },
        }
    )

    response = litellm.image_generation(
        model="toapis/gpt-image-2",
        prompt="city",
        api_key="test-key",
        size="16:9",
        resolution="1K",
        image_url="https://files.example/reference.png",
    )

    assert [image.url for image in response.data] == [
        "https://files.example/image.png",
        "https://files.example/image-2.png",
    ]
    assert route.calls[0].request.headers["Authorization"] == "Bearer test-key"
    assert json.loads(route.calls[0].request.content) == {
        "model": "gpt-image-2",
        "prompt": "city",
        "size": "16:9",
        "resolution": "1k",
        "response_format": "url",
        "reference_images": ["https://files.example/reference.png"],
    }


@pytest.mark.parametrize(
    "model,resolution",
    [
        ("gemini-2.5-flash-image-preview", "1K"),
        ("gemini-3-pro-image-preview", "4K"),
        ("gemini-3.1-flash-image-preview", "0.5K"),
    ],
)
def test_toapis_banana_models_map_resolution_and_reference_images(model, resolution):
    mapped = ToAPISImageGenerationConfig().map_openai_params(
        non_default_params={
            "size": "16:9",
            "imageConfig": {"imageSize": resolution, "google_search": True},
            "image_url": ["https://files.example/reference.png"],
            "n": 1,
        },
        optional_params={},
        model=model,
        drop_params=False,
    )

    assert mapped == {
        "size": "16:9",
        "response_format": "url",
        "n": 1,
        "metadata": {"resolution": resolution, "google_search": True},
        "image_urls": [{"url": "https://files.example/reference.png"}],
    }


def test_toapis_gpt_image_2_maps_reference_images_without_changing_public_contract():
    mapped = ToAPISImageGenerationConfig().map_openai_params(
        non_default_params={"size": "1:1", "image_url": "https://files.example/reference.png"},
        optional_params={},
        model="gpt-image-2",
        drop_params=False,
    )

    assert mapped == {
        "size": "1:1",
        "resolution": "1k",
        "response_format": "url",
        "reference_images": ["https://files.example/reference.png"],
    }


@pytest.mark.parametrize("resolution,expected", [("2K", "2k"), ("4K", "4k")])
def test_toapis_gpt_image_2_supports_high_resolution(resolution, expected):
    mapped = ToAPISImageGenerationConfig().map_openai_params(
        non_default_params={"resolution": resolution},
        optional_params={},
        model="gpt-image-2",
        drop_params=False,
    )

    assert mapped == {"size": "1:1", "resolution": expected, "response_format": "url"}


@pytest.mark.parametrize("pixel_size,ratio,resolution", _GPT_IMAGE_2_PIXEL_CASES)
def test_toapis_gpt_image_2_maps_official_pixel_sizes(pixel_size, ratio, resolution):
    mapped = ToAPISImageGenerationConfig().map_openai_params(
        non_default_params={"size": pixel_size},
        optional_params={},
        model="gpt-image-2",
        drop_params=False,
    )

    assert mapped == {"size": ratio, "resolution": resolution, "response_format": "url"}


def test_toapis_gpt_image_2_rejects_pixel_resolution_conflict():
    with pytest.raises(litellm.UnsupportedParamsError, match="must describe the same resolution"):
        ToAPISImageGenerationConfig().map_openai_params(
            non_default_params={"size": "2048x1152", "resolution": "1K"},
            optional_params={},
            model="gpt-image-2",
            drop_params=False,
        )


def test_toapis_gpt_image_2_rejects_conflicting_aspect_ratio_aliases():
    with pytest.raises(litellm.UnsupportedParamsError, match="must describe the same aspect ratio"):
        ToAPISImageGenerationConfig().map_openai_params(
            non_default_params={"size": "1:1", "aspect_ratio": "16:9"},
            optional_params={},
            model="gpt-image-2",
            drop_params=False,
        )


def test_toapis_gpt_image_2_rejects_unknown_image_config_fields():
    with pytest.raises(litellm.UnsupportedParamsError, match="does not support imageConfig field.*google_search"):
        ToAPISImageGenerationConfig().map_openai_params(
            non_default_params={"imageConfig": {"imageSize": "1K", "google_search": True}},
            optional_params={},
            model="gpt-image-2",
            drop_params=False,
        )


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


def test_toapis_banana_rejects_non_object_image_config():
    with pytest.raises(litellm.UnsupportedParamsError, match="imageConfig must be an object"):
        ToAPISImageGenerationConfig().map_openai_params(
            non_default_params={"imageConfig": "2K"},
            optional_params={},
            model="gemini-3.1-flash-image-preview",
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
