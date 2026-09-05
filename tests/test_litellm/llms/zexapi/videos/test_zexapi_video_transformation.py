import json
from unittest.mock import Mock

import httpx
import pytest

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.zexapi.videos.transformation import ZexAPIVideoConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.utils import decode_video_id_with_provider


def test_zexapi_video_json_request_and_environment(monkeypatch):
    monkeypatch.setenv("ZEXAPI_API_KEY", "test-key")
    config = ZexAPIVideoConfig()
    mapped = config.map_openai_params(
        video_create_optional_params={
            "size": "1920x1080",
            "images": ["https://files.example/reference.png"],
        },
        model="sora-2-12s",
        drop_params=False,
    )
    data, files, url = config.transform_video_create_request(
        model="sora-2-12s",
        prompt="advertisement",
        api_base=config.get_complete_url("sora-2-12s", None, {}),
        video_create_optional_request_params=mapped,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert data == {
        "model": "sora-2-12s",
        "prompt": "advertisement",
        "size": "1920x1080",
        "images": ["https://files.example/reference.png"],
    }
    assert not files
    assert url == "https://zexapi.com/v1/videos"
    assert config.use_multipart_form_data() is False
    assert config.validate_environment({}, "sora-2-12s") == {"Authorization": "Bearer test-key"}


def test_zexapi_video_accepts_duration_matching_model_slug():
    assert (
        ZexAPIVideoConfig().map_openai_params(
            video_create_optional_params={"seconds": "12"},
            model="sora-2-12s",
            drop_params=False,
        )
        == {}
    )


def test_zexapi_video_rejects_duration_mismatching_model_slug():
    with pytest.raises(ValueError, match="generates 12 seconds"):
        ZexAPIVideoConfig().map_openai_params(
            video_create_optional_params={"seconds": "5"},
            model="sora-2-12s",
            drop_params=False,
        )


@pytest.mark.parametrize(
    "model,seconds",
    [("veo_3_1-fast", "8"), ("veo_3_1-lite", "8"), ("omni_flash-10s", "10")],
)
def test_zexapi_video_accepts_documented_fixed_family_durations(model, seconds):
    assert (
        ZexAPIVideoConfig().map_openai_params(
            video_create_optional_params={"seconds": seconds},
            model=model,
            drop_params=False,
        )
        == {}
    )


@pytest.mark.parametrize("status", ["processing", "in_progress"])
def test_zexapi_task_response_normalizes_status_and_exposes_url(status):
    config = ZexAPIVideoConfig()
    result = config.transform_video_create_response(
        model="sora-2-12s",
        raw_response=httpx.Response(
            200,
            json={
                "id": "task_123",
                "object": "video",
                "model": "sora-2-12s",
                "status": status,
                "progress": 45,
                "created_at": 1709876543,
                "url": "https://files.example/video.mp4",
            },
        ),
        logging_obj=Mock(),
        custom_llm_provider="zexapi",
    )
    decoded = decode_video_id_with_provider(result.id)

    assert result.object == "video"
    assert result.status == "in_progress"
    assert result.output_url == "https://files.example/video.mp4"
    assert decoded["custom_llm_provider"] == "zexapi"
    assert decoded["model_id"] == "sora-2-12s"


def test_zexapi_video_config_rejects_image_tasks():
    with pytest.raises(BaseLLMException, match="use image_generation"):
        ZexAPIVideoConfig().transform_video_create_response(
            model="nano_banana_2",
            raw_response=httpx.Response(
                200,
                json={
                    "id": "task_123",
                    "object": "image",
                    "model": "nano_banana_2",
                    "status": "queued",
                },
            ),
            logging_obj=Mock(),
            custom_llm_provider="zexapi",
        )


def test_zexapi_failed_task_normalizes_string_error():
    result = ZexAPIVideoConfig().transform_video_status_retrieve_response(
        raw_response=httpx.Response(
            200,
            json={
                "id": "task_123",
                "object": "video",
                "status": "failed",
                "progress": 0,
                "error": "content policy",
            },
        ),
        logging_obj=Mock(),
        custom_llm_provider="zexapi",
    )

    assert result.error == {"code": "generation_failed", "message": "content policy"}


def test_zexapi_failed_task_without_error_gets_public_error():
    result = ZexAPIVideoConfig().transform_video_status_retrieve_response(
        raw_response=httpx.Response(
            200,
            json={
                "id": "task_123",
                "object": "video",
                "status": "failed",
                "progress": 0,
            },
        ),
        logging_obj=Mock(),
        custom_llm_provider="zexapi",
    )

    assert result.error == {
        "code": "provider_failed",
        "message": "The provider reported that video generation failed",
    }


def test_zexapi_completed_task_without_output_url_becomes_structured_failure():
    result = ZexAPIVideoConfig().transform_video_status_retrieve_response(
        raw_response=httpx.Response(
            200,
            json={
                "id": "task_123",
                "object": "video",
                "status": "completed",
                "progress": 100,
            },
        ),
        logging_obj=Mock(),
        custom_llm_provider="zexapi",
    )

    assert result.status == "failed"
    assert result.output_url is None
    assert result.error == {
        "code": "provider_contract_error",
        "message": "The provider completed video generation without a downloadable output URL",
    }


def test_zexapi_video_status_url_encodes_task_id():
    url, data = ZexAPIVideoConfig().transform_video_status_retrieve_request(
        video_id="task/../other?x=1",
        api_base="https://zexapi.com/v1/videos",
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert url == "https://zexapi.com/v1/videos/task%2F..%2Fother%3Fx%3D1"
    assert data == {}


def test_zexapi_public_video_generation_forwards_images(respx_mock):
    route = respx_mock.post("https://zexapi.com/v1/videos").respond(
        json={
            "id": "task_123",
            "object": "video",
            "model": "sora-2-12s",
            "status": "queued",
            "progress": 0,
            "created_at": 1709876543,
        }
    )

    response = litellm.video_generation(
        model="zexapi/sora-2-12s",
        prompt="advertisement",
        api_key="test-key",
        size="1920x1080",
        extra_body={"images": ["https://files.example/reference.png"]},
    )

    assert response.status == "queued"
    assert route.calls[0].request.headers["Authorization"] == "Bearer test-key"
    assert json.loads(route.calls[0].request.content) == {
        "model": "sora-2-12s",
        "prompt": "advertisement",
        "size": "1920x1080",
        "images": ["https://files.example/reference.png"],
    }


def test_zexapi_public_video_generation_supports_multipart_reference(respx_mock):
    route = respx_mock.post("https://zexapi.com/v1/videos").respond(
        json={
            "id": "task_123",
            "object": "video",
            "model": "sora-2-12s",
            "status": "queued",
            "progress": 0,
            "created_at": 1709876543,
        }
    )

    response = litellm.video_generation(
        model="zexapi/sora-2-12s",
        prompt="advertisement",
        api_key="test-key",
        size="1920x1080",
        input_reference=b"\x89PNG\r\n\x1a\n",
    )

    assert response.status == "queued"
    assert "multipart/form-data" in route.calls[0].request.headers["Content-Type"]
    assert b'name="input_reference"' in route.calls[0].request.content
