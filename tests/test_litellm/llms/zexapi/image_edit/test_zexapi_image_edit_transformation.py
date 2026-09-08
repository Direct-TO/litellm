from unittest.mock import Mock

import httpx
import pytest
from email.parser import BytesParser
from email.policy import default

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.submission_utils import get_submission_outcome
from litellm.llms.zexapi.image_edit.transformation import (
    ZexAPIImageEditConfig,
    get_zexapi_image_edit_config,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler


@pytest.mark.parametrize(
    "resolution,size", [("1K", "1280x720"), ("2K", "2560x1440"), ("2k", "2560x1440"), ("4K", "3840x2160")]
)
def test_zexapi_edit_maps_tier_and_ratio_without_changing_other_edit_options(resolution, size):
    assert ZexAPIImageEditConfig().map_openai_params(
        image_edit_optional_params={
            "aspect_ratio": "16:9",
            "resolution": resolution,
            "n": 1,
            "response_format": "b64_json",
            "background": "transparent",
        },
        model="gpt-image2",
        drop_params=True,
    ) == {"size": size, "n": 1, "response_format": "b64_json", "background": "transparent"}


@pytest.mark.parametrize(
    "params",
    [
        {"size": "3840x2160", "resolution": "2K"},
        {"size": "2560x1440", "aspect_ratio": "9:16"},
        {"resolution": "8K"},
        {"resolution": "auto"},
        {"aspect_ratio": "2:1"},
    ],
)
def test_zexapi_edit_rejects_conflicting_or_unsupported_dimensions(params):
    with pytest.raises(litellm.UnsupportedParamsError):
        ZexAPIImageEditConfig().map_openai_params(params, model="gpt-image2", drop_params=True)


def test_zexapi_edit_does_not_invent_size_for_an_unspecified_request():
    assert ZexAPIImageEditConfig().map_openai_params({"n": 1}, model="gpt-image2", drop_params=False) == {"n": 1}


def test_zexapi_edit_keeps_consistent_explicit_pixels():
    assert ZexAPIImageEditConfig().map_openai_params(
        {"size": "2560x1440", "aspect_ratio": "16:9", "resolution": "2K"}, model="gpt-image2", drop_params=False
    ) == {"size": "2560x1440"}


@pytest.mark.asyncio
async def test_zexapi_public_async_edit_sends_mapped_size_and_preserves_files():
    captured = []

    def respond(request):
        captured.append(request)
        return httpx.Response(200, json={"created": 1, "data": [{"url": "https://files.example/edit.png"}]})

    client = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(respond))
    try:
        await litellm.aimage_edit(
            model="zexapi/gpt-image2",
            prompt="make it blue",
            image=[b"\x89PNG\r\n\x1a\nfirst", b"\x89PNG\r\n\x1a\nsecond"],
            mask=b"\x89PNG\r\n\x1a\nmask",
            n=1,
            aspect_ratio="16:9",
            resolution="2K",
            background="transparent",
            api_key="test-key",
            api_base="https://edit.example/v1",
            client=client,
        )
    finally:
        await client.client.aclose()
    assert len(captured) == 1
    request = captured[0]
    assert str(request.url) == "https://edit.example/v1/images/edits"
    message = BytesParser(policy=default).parsebytes(
        f"Content-Type: {request.headers['content-type']}\r\n\r\n".encode() + request.content
    )
    parts = list(message.iter_parts())
    fields = {
        part.get_param("name", header="content-disposition"): part.get_payload(decode=True).decode()
        for part in parts
        if part.get_filename() is None
    }
    assert fields == {
        "model": "gpt-image2",
        "prompt": "make it blue",
        "n": "1",
        "size": "2560x1440",
        "background": "transparent",
    }
    assert [part.get_param("name", header="content-disposition") for part in parts if part.get_filename()] == [
        "image[]",
        "image[]",
        "mask",
    ]


def test_zexapi_image_edit_uses_provider_url_and_preserves_dual_response(monkeypatch):
    monkeypatch.setenv("ZEXAPI_API_KEY", "test-key")
    config = ZexAPIImageEditConfig()
    result = config.transform_image_edit_response(
        model="image2",
        raw_response=httpx.Response(
            200,
            json={
                "created": 1782108238,
                "data": [
                    {"url": "https://files.example/edit.png", "b64_json": "ZWRpdA=="},
                    {"url": "https://files.example/edit-2.png", "b64_json": "ZWRpdDI="},
                ],
            },
        ),
        logging_obj=Mock(),
    )

    assert config.get_complete_url("image2", None, {}) == "https://zexapi.com/v1/images/edits"
    assert config.validate_environment({}, "image2", litellm_params={}) == {"Authorization": "Bearer test-key"}
    assert result.data[0].url == "https://files.example/edit.png"
    assert result.data[0].b64_json == "ZWRpdA=="
    assert result.data[1].url == "https://files.example/edit-2.png"
    assert result.data[1].b64_json == "ZWRpdDI="


def test_zexapi_invalid_image_edit_success_has_unknown_submission_outcome():
    with pytest.raises(BaseLLMException) as exc_info:
        ZexAPIImageEditConfig().transform_image_edit_response(
            model="image2",
            raw_response=httpx.Response(200, text="not-json"),
            logging_obj=Mock(),
        )

    assert get_submission_outcome(exc_info.value) == "unknown"


def test_zexapi_public_image_edit_uses_multipart_endpoint(respx_mock):
    route = respx_mock.post("https://zexapi.com/v1/images/edits").respond(
        json={
            "created": 1782108238,
            "data": [{"url": "https://files.example/edit.png", "b64_json": "ZWRpdA=="}],
        }
    )

    response = litellm.image_edit(
        model="zexapi/image2",
        prompt="make it gray",
        image=b"\x89PNG\r\n\x1a\n",
        api_key="test-key",
        size="1024x1024",
    )

    assert response.data[0].url == "https://files.example/edit.png"
    assert response.data[0].b64_json == "ZWRpdA=="
    assert route.calls[0].request.headers["Authorization"] == "Bearer test-key"
    assert "multipart/form-data" in route.calls[0].request.headers["Content-Type"]


def test_zexapi_gpt_image2_edit_accepts_high_resolution_size():
    config = get_zexapi_image_edit_config("gpt-image2")

    assert isinstance(config, ZexAPIImageEditConfig)
    assert config.map_openai_params(
        image_edit_optional_params={"size": "2048x2048"},
        model="gpt-image2",
        drop_params=False,
    ) == {"size": "2048x2048"}


def test_zexapi_image2_edit_rejects_high_resolution_size():
    config = ZexAPIImageEditConfig()

    with pytest.raises(litellm.UnsupportedParamsError, match="does not support size"):
        config.map_openai_params(
            image_edit_optional_params={"size": "2048x2048"},
            model="image2",
            drop_params=False,
        )


def test_zexapi_public_gpt_image2_edit_preserves_provider_model(respx_mock):
    route = respx_mock.post("https://zexapi.com/v1/images/edits").respond(
        json={"created": 1782108238, "data": [{"url": "https://files.example/edit.png"}]}
    )

    response = litellm.image_edit(
        model="zexapi/gpt-image2",
        prompt="make it gray",
        image=b"\x89PNG\r\n\x1a\n",
        api_key="test-key",
        size="2048x2048",
    )

    assert response.data[0].url == "https://files.example/edit.png"
    assert b"gpt-image2" in route.calls[0].request.content
