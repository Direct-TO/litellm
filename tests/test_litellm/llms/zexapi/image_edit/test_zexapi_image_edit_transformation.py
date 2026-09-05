from unittest.mock import Mock

import httpx
import pytest

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.submission_utils import get_submission_outcome
from litellm.llms.zexapi.image_edit.transformation import (
    ZexAPIImageEditConfig,
    get_zexapi_image_edit_config,
)


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
