from unittest.mock import Mock

import httpx

import litellm
from litellm.llms.zexapi.image_edit.transformation import ZexAPIImageEditConfig


def test_zexapi_image_edit_uses_provider_url_and_preserves_dual_response(monkeypatch):
    monkeypatch.setenv("ZEXAPI_API_KEY", "test-key")
    config = ZexAPIImageEditConfig()
    result = config.transform_image_edit_response(
        model="image2",
        raw_response=httpx.Response(
            200,
            json={
                "created": 1782108238,
                "data": [{"url": "https://files.example/edit.png", "b64_json": "ZWRpdA=="}],
            },
        ),
        logging_obj=Mock(),
    )

    assert config.get_complete_url("image2", None, {}) == "https://zexapi.com/v1/images/edits"
    assert config.validate_environment({}, "image2", litellm_params={}) == {"Authorization": "Bearer test-key"}
    assert result.data[0].url == "https://files.example/edit.png"
    assert result.data[0].b64_json == "ZWRpdA=="


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
