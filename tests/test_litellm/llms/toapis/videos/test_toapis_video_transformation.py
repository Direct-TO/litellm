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
    ) == {
        "input_reference": reference,
        "_toapis_reference_format": "image_with_roles",
    }


def test_toapis_video_service_rejects_other_models():
    with pytest.raises(litellm.UnsupportedParamsError):
        ToAPISVideoConfig().map_openai_params(
            video_create_optional_params={},
            model="undocumented-video-model",
            drop_params=False,
        )


@pytest.mark.parametrize(
    "model,size,expected_field,expected_value",
    [
        ("seedance-2", "1920x1080", "aspect_ratio", "16:9"),
        ("wan3.0-video", "1920x1080", "ratio", "16:9"),
        ("Veo3.1-fast-official", "1920x1080", "size", "1920x1080"),
        ("MiniMax-H3", "9:16", "aspect_ratio", "9:16"),
        ("kling-v3", "1080x1920", "aspect_ratio", "9:16"),
        ("grok-video-1.0", "1:1", "aspect_ratio", "1:1"),
    ],
)
def test_toapis_video_maps_documented_model_size_contract(model, size, expected_field, expected_value):
    mapped = ToAPISVideoConfig().map_openai_params(
        video_create_optional_params={"seconds": "8", "size": size},
        model=model,
        drop_params=False,
    )

    assert mapped == {"duration": 8, expected_field: expected_value}


@pytest.mark.parametrize(
    "model,expected_format",
    [
        ("gemini-omni-flash", "image_urls"),
        ("seedance-2-mini", "image_with_roles"),
        ("kling-v2-6", "reference_images"),
        ("kling-v3-omni", "metadata_image_list"),
        ("grok-video-1.5", "image"),
    ],
)
def test_toapis_video_selects_model_specific_reference_contract(model, expected_format):
    mapped = ToAPISVideoConfig().map_openai_params(
        video_create_optional_params={"input_reference": b"\x89PNG\r\n\x1a\n"},
        model=model,
        drop_params=False,
    )

    assert mapped["_toapis_reference_format"] == expected_format


@pytest.mark.parametrize(
    "reference_format,expected",
    [
        (
            "image_with_roles",
            {
                "metadata": {"seed": 42},
                "image_with_roles": [{"url": "https://files.example/ref.png", "role": "reference_image"}],
            },
        ),
        ("image_urls", {"metadata": {"seed": 42}, "image_urls": ["https://files.example/ref.png"]}),
        (
            "reference_images",
            {"metadata": {"seed": 42}, "reference_images": ["https://files.example/ref.png"]},
        ),
        ("image", {"metadata": {"seed": 42}, "image": "https://files.example/ref.png"}),
        (
            "metadata_image_list",
            {"metadata": {"seed": 42, "image_list": [{"image_url": "https://files.example/ref.png"}]}},
        ),
    ],
)
def test_toapis_video_uploaded_reference_uses_model_specific_field(reference_format, expected):
    result = ToAPISVideoConfig().transform_video_create_input_reference_upload_response(
        raw_response=httpx.Response(
            200,
            json={
                "success": True,
                "message": "uploaded",
                "data": {
                    "id": "image_123",
                    "url": "https://files.example/ref.png",
                    "mime_type": "image/png",
                    "size": 8,
                },
            },
        ),
        video_create_optional_request_params={
            "input_reference": b"image",
            "_toapis_reference_format": reference_format,
            "metadata": {"seed": 42},
        },
    )

    assert result == expected


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


@pytest.mark.parametrize(
    "response_id,task_id,expected_task_id",
    [
        ("video_task_123", "video_task_123", "video_task_123"),
        ("video_resource_123", "video_task_123", "video_task_123"),
        ("video_task_123", None, "video_task_123"),
    ],
)
@pytest.mark.parametrize(
    "status,expected_status",
    [("", "queued"), ("pending", "queued"), ("queued", "queued"), ("in_progress", "in_progress"), ("completed", "completed"), ("failed", "failed")],
)
def test_toapis_video_create_accepts_live_task_envelope(response_id, task_id, expected_task_id, status, expected_status):
    payload = {
        "id": response_id,
        "object": "video",
        "model": "seedance-2-5",
        "status": status,
        "progress": 100 if status in ("completed", "failed") else 25 if status == "in_progress" else 0,
        "created_at": 1788419339,
    }
    if task_id is not None:
        payload["task_id"] = task_id
    if status == "completed":
        payload["result"] = {"type": "video", "data": [{"url": "https://files.example/video.mp4"}]}
    if status == "failed":
        payload["error"] = {"code": "generation_failed", "message": "Provider generation failed"}

    result = ToAPISVideoConfig().transform_video_create_response(
        model="seedance-2-5",
        raw_response=httpx.Response(200, json=payload),
        logging_obj=Mock(),
        custom_llm_provider="toapis",
    )
    decoded = decode_video_id_with_provider(result.id)

    assert result.object == "video"
    assert result.status == expected_status
    assert result.progress == payload["progress"]
    assert decoded["video_id"] == expected_task_id
    assert result.output_url == ("https://files.example/video.mp4" if status == "completed" else None)
    assert result.error == payload.get("error")


@pytest.mark.parametrize("status", ["", "pending", "queued", "in_progress", "completed", "failed"])
def test_toapis_video_status_rejects_live_create_task_envelope(status):
    with pytest.raises(BaseLLMException) as exc_info:
        ToAPISVideoConfig().transform_video_status_retrieve_response(
            raw_response=httpx.Response(
                200,
                json={
                    "id": "video_task_123",
                    "task_id": "video_task_123",
                    "object": "video",
                    "status": status,
                    "progress": 0,
                },
            ),
            logging_obj=Mock(),
            custom_llm_provider="toapis",
        )

    assert exc_info.value.status_code == 502
    assert "Invalid ToAPIs task response" in str(exc_info.value)


@pytest.mark.parametrize(
    "overrides",
    [
        {"status": "unknown"},
        {"status": None},
        {"status": 0},
        {"id": "", "task_id": " "},
        {"id": None, "task_id": None},
    ],
)
def test_toapis_video_create_rejects_invalid_task_envelope(overrides):
    with pytest.raises(BaseLLMException) as exc_info:
        ToAPISVideoConfig().transform_video_create_response(
            model="wan3.0-video",
            raw_response=httpx.Response(200, json={"id": "video_task_123", "object": "video", "status": "queued", **overrides}),
            logging_obj=Mock(),
            custom_llm_provider="toapis",
        )

    assert exc_info.value.status_code == 502
    assert "Invalid ToAPIs task response" in str(exc_info.value)


@pytest.mark.parametrize(
    "status_code,expected_status_code",
    [
        (302, 502),
        (429, 429),
        (502, 502),
    ],
)
@pytest.mark.parametrize("status", ["", "queued", "in_progress"])
def test_toapis_video_create_only_normalizes_success_responses(status_code, expected_status_code, status):
    with pytest.raises(BaseLLMException) as exc_info:
        ToAPISVideoConfig().transform_video_create_response(
            model="seedance-2-5",
            raw_response=httpx.Response(
                status_code,
                json={
                    "id": "video_task_123",
                    "object": "video",
                    "status": status,
                },
            ),
            logging_obj=Mock(),
            custom_llm_provider="toapis",
        )

    assert exc_info.value.status_code == expected_status_code


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


@pytest.mark.parametrize("model", ["seedance-2-5", "wan3.0-video"])
@pytest.mark.parametrize("status,expected_status", [("", "queued"), ("pending", "queued"), ("queued", "queued"), ("in_progress", "in_progress")])
def test_toapis_public_video_generation_accepts_live_create_response(respx_mock, model, status, expected_status):
    route = respx_mock.post("https://toapis.com/v1/videos/generations").respond(
        json={
            "id": "video_task_123",
            "task_id": "video_task_123",
            "object": "video",
            "model": model,
            "status": status,
            "progress": 0,
            "created_at": 1788419339,
        }
    )

    response = litellm.video_generation(
        model=f"toapis/{model}",
        prompt="waves",
        api_key="test-key",
        seconds="4",
        size="1920x1080",
    )
    decoded = decode_video_id_with_provider(response.id)

    assert response.status == expected_status
    assert decoded["video_id"] == "video_task_123"
    assert decoded["model_id"] == model
    assert route.call_count == 1


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
