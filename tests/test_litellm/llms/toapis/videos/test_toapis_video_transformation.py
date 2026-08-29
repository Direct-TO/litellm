import json
from io import BytesIO
from unittest.mock import Mock

import httpx
import pytest

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.toapis.videos.transformation import ToAPISVideoConfig
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.utils import decode_video_id_with_provider


def test_toapis_video_request_maps_openai_params_and_uses_json(monkeypatch):
    monkeypatch.setenv("TOAPIS_API_KEY", "test-key")
    config = ToAPISVideoConfig()
    mapped = config.map_openai_params(
        video_create_optional_params={"seconds": "12", "size": "1920x1080"},
        model="seedance-2-5",
        drop_params=False,
    )
    data, files, url = config.transform_video_create_request(
        model="seedance-2-5",
        prompt="waves",
        api_base=config.get_complete_url("seedance-2-5", None, {}),
        video_create_optional_request_params=mapped,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )

    assert mapped == {"duration": 12, "aspect_ratio": "16:9"}
    assert data == {
        "model": "seedance-2-5",
        "prompt": "waves",
        "duration": 12,
        "aspect_ratio": "16:9",
    }
    assert not files
    assert url == "https://toapis.com/v1/videos/generations"
    assert config.use_multipart_form_data() is False
    assert config.validate_environment({}, "seedance-2-5") == {
        "Authorization": "Bearer test-key",
        "Content-Type": "application/json",
    }


def test_toapis_video_keeps_local_reference_for_upload():
    reference = b"\x89PNG\r\n\x1a\n"
    assert ToAPISVideoConfig().map_openai_params(
        video_create_optional_params={"input_reference": reference},
        model="seedance-2-5",
        drop_params=False,
    ) == {"input_reference": reference}


def test_toapis_video_service_rejects_other_models():
    with pytest.raises(litellm.UnsupportedParamsError):
        ToAPISVideoConfig().map_openai_params(
            video_create_optional_params={},
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
            "model": "seedance-2-5",
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
        model="seedance-2-5",
        raw_response=response,
        logging_obj=Mock(),
        custom_llm_provider="toapis",
    )
    decoded = decode_video_id_with_provider(result.id)

    assert result.status == "completed"
    assert result.output_url == "https://files.example/video.mp4"
    assert result.expires_at == 1703971300
    assert decoded["custom_llm_provider"] == "toapis"
    assert decoded["model_id"] == "seedance-2-5"
    assert decoded["video_id"] == "video_task_123"


def test_toapis_completed_task_without_output_url_becomes_structured_failure():
    result = ToAPISVideoConfig().transform_video_status_retrieve_response(
        raw_response=httpx.Response(
            200,
            json={
                "id": "video_task_123",
                "object": "generation.task",
                "status": "completed",
                "progress": 100,
                "result": {"type": "video", "data": []},
            },
        ),
        logging_obj=Mock(),
        custom_llm_provider="toapis",
    )

    assert result.status == "failed"
    assert result.output_url is None
    assert result.error == {
        "code": "provider_contract_error",
        "message": "The provider completed video generation without a downloadable output URL",
    }


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
            "model": "seedance-2-5",
            "status": "pending",
            "progress": 0,
            "created_at": 1703884800,
        }
    )

    response = litellm.video_generation(
        model="toapis/seedance-2-5",
        prompt="waves",
        api_key="test-key",
        seconds="12",
        size="1920x1080",
        extra_body={
            "resolution": "720p",
            "generate_audio": True,
            "image_with_roles": [{"url": "https://files.example/reference.png", "role": "reference_image"}],
        },
    )

    assert response.status == "queued"
    assert route.calls[0].request.headers["Authorization"] == "Bearer test-key"
    assert json.loads(route.calls[0].request.content) == {
        "model": "seedance-2-5",
        "prompt": "waves",
        "duration": 12,
        "aspect_ratio": "16:9",
        "resolution": "720p",
        "generate_audio": True,
        "image_with_roles": [{"url": "https://files.example/reference.png", "role": "reference_image"}],
    }


def test_toapis_public_video_generation_uploads_input_reference(respx_mock):
    upload_route = respx_mock.post("https://toapis.com/v1/uploads/images").respond(
        json={
            "success": True,
            "message": "uploaded",
            "data": {
                "id": "image_123",
                "url": "https://files.example/reference.png",
                "mime_type": "image/png",
                "size": 8,
            },
        }
    )
    create_route = respx_mock.post("https://toapis.com/v1/videos/generations").respond(
        json={
            "id": "video_task_123",
            "object": "generation.task",
            "model": "seedance-2-5",
            "status": "queued",
            "progress": 0,
        }
    )

    reference = BytesIO(b"\x89PNG\r\n\x1a\n")
    reference.seek(8)
    response = litellm.video_generation(
        model="toapis/seedance-2-5",
        prompt="waves",
        api_key="test-key",
        seconds="12",
        input_reference=reference,
    )

    assert response.status == "queued"
    upload_request = upload_route.calls[0].request
    assert upload_request.headers["Authorization"] == "Bearer test-key"
    assert "multipart/form-data" in upload_request.headers["Content-Type"]
    assert b'name="file"' in upload_request.content
    assert b'name="purpose"' in upload_request.content
    assert b"\x89PNG\r\n\x1a\n" in upload_request.content
    assert json.loads(create_route.calls[0].request.content) == {
        "model": "seedance-2-5",
        "prompt": "waves",
        "duration": 12,
        "image_with_roles": [{"url": "https://files.example/reference.png", "role": "reference_image"}],
    }


def test_toapis_failed_upload_without_data_maps_to_bad_gateway():
    with pytest.raises(BaseLLMException) as exc_info:
        ToAPISVideoConfig().transform_video_create_input_reference_upload_response(
            raw_response=httpx.Response(
                200,
                json={"success": False, "message": "invalid image"},
            ),
            video_create_optional_request_params={"input_reference": b"image"},
        )

    assert exc_info.value.status_code == 502
    assert "invalid image" in str(exc_info.value)


@pytest.mark.asyncio
async def test_toapis_async_video_generation_uploads_input_reference():
    captured_create_bodies = []

    def handle(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/v1/uploads/images":
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "message": "uploaded",
                    "data": {
                        "id": "image_123",
                        "url": "https://files.example/reference.png",
                        "mime_type": "image/png",
                        "size": 8,
                    },
                },
            )
        captured_create_bodies.append(json.loads(request.content))
        return httpx.Response(
            200,
            json={
                "id": "video_task_123",
                "object": "generation.task",
                "model": "seedance-2-5",
                "status": "queued",
                "progress": 0,
            },
        )

    client = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(transport=httpx.MockTransport(handle))

    try:
        response = await litellm.avideo_generation(
            model="toapis/seedance-2-5",
            prompt="waves",
            api_key="test-key",
            seconds="12",
            input_reference=b"\x89PNG\r\n\x1a\n",
            client=client,
        )
    finally:
        await client.client.aclose()

    assert response.status == "queued"
    assert captured_create_bodies[0]["image_with_roles"] == [
        {"url": "https://files.example/reference.png", "role": "reference_image"}
    ]
