import json
from unittest.mock import Mock

import httpx
import pytest

import litellm
from litellm.llms.toapis.videos.transformation import ToAPISVideoConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.utils import decode_video_id_with_provider


def test_toapis_video_request_maps_openai_params_and_uses_json(monkeypatch):
    monkeypatch.setenv("TOAPIS_API_KEY", "test-key")
    config = ToAPISVideoConfig()
    mapped = config.map_openai_params(
        video_create_optional_params={"seconds": "12", "size": "1920x1080"},
        model="sora-2-vvip",
        drop_params=False,
    )
    data, files, url = config.transform_video_create_request(
        model="sora-2-vvip",
        prompt="waves",
        api_base=config.get_complete_url("sora-2-vvip", None, {}),
        video_create_optional_request_params=mapped,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert mapped == {"duration": 12, "aspect_ratio": "16:9"}
    assert data == {
        "model": "sora-2-vvip",
        "prompt": "waves",
        "duration": 12,
        "aspect_ratio": "16:9",
    }
    assert not files
    assert url == "https://toapis.com/v1/videos/generations"
    assert config.use_multipart_form_data() is False
    assert config.validate_environment({}, "sora-2-vvip") == {
        "Authorization": "Bearer test-key",
        "Content-Type": "application/json",
    }


def test_toapis_video_rejects_local_reference_input():
    with pytest.raises(ValueError, match="extra_body.image_urls"):
        ToAPISVideoConfig().map_openai_params(
            video_create_optional_params={"input_reference": b"image"},
            model="sora-2-vvip",
            drop_params=False,
        )


@pytest.mark.parametrize(
    "size,expected",
    [
        ("1024x1024", "1:1"),
        ("1920x1080", "16:9"),
        ("1080x1920", "9:16"),
        ("21:9", "21:9"),
    ],
)
def test_toapis_video_maps_pixel_sizes_to_aspect_ratios(size, expected):
    assert ToAPISVideoConfig._aspect_ratio(size) == expected


def test_toapis_video_task_response_exposes_output_url_and_provider_id():
    config = ToAPISVideoConfig()
    response = httpx.Response(
        200,
        json={
            "id": "video_task_123",
            "object": "generation.task",
            "model": "sora-2-vvip",
            "status": "completed",
            "progress": 100,
            "created_at": 1703884800,
            "completed_at": 1703884900,
            "expires_at": 1703971300,
            "result": {
                "type": "video",
                "data": [{"url": "https://files.example/video.mp4", "format": "mp4"}],
            },
        },
    )

    result = config.transform_video_create_response(
        model="sora-2-vvip",
        raw_response=response,
        logging_obj=Mock(),
        custom_llm_provider="toapis",
    )
    decoded = decode_video_id_with_provider(result.id)

    assert result.status == "completed"
    assert result.output_url == "https://files.example/video.mp4"
    assert result.expires_at == 1703971300
    assert decoded["custom_llm_provider"] == "toapis"
    assert decoded["model_id"] == "sora-2-vvip"
    assert decoded["video_id"] == "video_task_123"


def test_toapis_video_status_url_encodes_task_id():
    url, data = ToAPISVideoConfig().transform_video_status_retrieve_request(
        video_id="task/../other?x=1",
        api_base="https://toapis.com/v1/videos/generations",
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert url == "https://toapis.com/v1/videos/generations/task%2F..%2Fother%3Fx%3D1"
    assert data == {}


def test_toapis_public_video_generation_normalizes_pending_status(respx_mock):
    route = respx_mock.post("https://toapis.com/v1/videos/generations").respond(
        json={
            "id": "video_task_123",
            "object": "generation.task",
            "model": "sora-2-vvip",
            "status": "pending",
            "progress": 0,
            "created_at": 1703884800,
        }
    )

    response = litellm.video_generation(
        model="toapis/sora-2-vvip",
        prompt="waves",
        api_key="test-key",
        seconds="12",
        size="1920x1080",
        extra_body={"image_urls": ["https://files.example/reference.png"]},
    )

    assert response.status == "queued"
    assert route.calls[0].request.headers["Authorization"] == "Bearer test-key"
    assert json.loads(route.calls[0].request.content) == {
        "model": "sora-2-vvip",
        "prompt": "waves",
        "duration": 12,
        "aspect_ratio": "16:9",
        "image_urls": ["https://files.example/reference.png"],
    }
