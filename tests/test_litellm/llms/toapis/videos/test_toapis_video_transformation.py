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

_SEEDANCE_PRIVATE_REVIEW_MODELS = ["seedance-2", "seedance-2-fast", "seedance-2-mini", "seedance-2-5"]


def _create_request_data(model, params):
    return ToAPISVideoConfig().transform_video_create_request(
        model=model,
        prompt="virtual character waves",
        api_base="https://toapis.com/v1/videos/generations",
        video_create_optional_request_params=params,
        litellm_params=GenericLiteLLMParams(),
        headers={},
    )[0]


@pytest.mark.parametrize("model", _SEEDANCE_PRIVATE_REVIEW_MODELS)
@pytest.mark.parametrize(
    "media",
    [
        {"image_with_roles": [{"url": "https://files.example/first.png", "role": "first_frame"}]},
        {"image_urls": ["https://files.example/first.png", "https://files.example/second.png"]},
        {"video_with_roles": [{"url": "https://files.example/ref.mp4", "role": "reference_video"}]},
        {"audio_with_roles": [{"url": "https://files.example/ref.mp3", "role": "reference_audio"}]},
    ],
)
def test_seedance_media_requests_enable_private_review_and_preserve_inputs(model, media):
    params = {"duration": 4, "resolution": "480p", "generate_audio": False, **media}

    assert _create_request_data(model, params) == {
        "model": model,
        "prompt": "virtual character waves",
        **params,
        "private_asset_review": True,
    }
    assert "private_asset_review" not in params


@pytest.mark.parametrize(
    "url", ["HTTPS://files.example/ref.png", "HtTp://files.example/ref.png", " https://files.example/ref.png "]
)
@pytest.mark.parametrize("field", ["image_urls", "image_with_roles", "video_with_roles", "audio_with_roles"])
def test_seedance_source_url_scheme_detection_preserves_original_url(field, url):
    media = {field: [url] if field == "image_urls" else [{"url": url, "role": "reference_image"}]}

    assert _create_request_data("seedance-2", media) == {
        "model": "seedance-2",
        "prompt": "virtual character waves",
        **media,
        "private_asset_review": True,
    }


@pytest.mark.parametrize("model", _SEEDANCE_PRIVATE_REVIEW_MODELS)
@pytest.mark.parametrize(
    "media", [{}, {"image_with_roles": []}, {"image_urls": [], "video_with_roles": [], "audio_with_roles": []}]
)
def test_seedance_text_requests_do_not_enable_private_review(model, media):
    assert "private_asset_review" not in _create_request_data(model, {"duration": 4, **media})


@pytest.mark.parametrize("model", ["wan3.0-video", "MiniMax-H3", "grok-video-1.0", "veo3.1-fast"])
def test_other_toapis_models_do_not_automatically_enable_private_review(model):
    data = _create_request_data(
        model,
        {"image_with_roles": [{"url": "https://files.example/ref.png", "role": "reference_image"}]},
    )

    assert "private_asset_review" not in data


@pytest.mark.parametrize("model", _SEEDANCE_PRIVATE_REVIEW_MODELS)
@pytest.mark.parametrize("review", [True, False])
def test_seedance_explicit_private_review_is_preserved(model, review):
    data = _create_request_data(
        model,
        {
            "image_with_roles": [{"url": "https://files.example/ref.png", "role": "reference_image"}],
            "private_asset_review": review,
        },
    )

    assert data["private_asset_review"] is review


@pytest.mark.parametrize("review", ["true", "false", 0, 1, None, {}, []])
def test_seedance_private_review_rejects_non_boolean_values(review):
    with pytest.raises(litellm.UnsupportedParamsError, match="private_asset_review must be a boolean"):
        _create_request_data("seedance-2", {"private_asset_review": review})


@pytest.mark.parametrize(
    "media",
    [
        {"image_with_roles": [{"url": "asset://old-image", "role": "reference_image"}]},
        {"image_urls": ["https://files.example/ref.png", "asset://old-image"]},
        {"video_with_roles": [{"url": "asset://old-video", "role": "reference_video"}]},
        {"audio_with_roles": [{"url": "asset://old-audio", "role": "reference_audio"}]},
    ],
)
def test_seedance_legacy_assets_do_not_implicitly_enable_private_review(media):
    assert "private_asset_review" not in _create_request_data("seedance-2", media)
    assert _create_request_data("seedance-2", {**media, "private_asset_review": False})["private_asset_review"] is False

    with pytest.raises(litellm.UnsupportedParamsError, match="cannot be combined with asset://"):
        _create_request_data("seedance-2", {**media, "private_asset_review": True})


@pytest.mark.parametrize("asset_url", ["ASSET://old-image", "AsSeT://old-image", " asset://old-image "])
def test_seedance_legacy_asset_scheme_cannot_bypass_private_review_conflict(asset_url):
    media = {"image_urls": ["https://files.example/ref.png", asset_url]}

    assert "private_asset_review" not in _create_request_data("seedance-2", media)
    with pytest.raises(litellm.UnsupportedParamsError, match="cannot be combined with asset://"):
        _create_request_data("seedance-2", {**media, "private_asset_review": True})


@pytest.mark.parametrize("model", _SEEDANCE_PRIVATE_REVIEW_MODELS)
@pytest.mark.parametrize("url", ["HTTPS://files.example/ref.png", "HtTp://files.example/ref.png"])
def test_seedance_canonical_references_accept_case_insensitive_source_url_schemes(respx_mock, model, url):
    route = respx_mock.post("https://toapis.com/v1/videos/generations").respond(
        json={"id": "video_task_123", "object": "generation.task", "model": model, "status": "queued"}
    )

    litellm.video_generation(
        model=f"toapis/{model}",
        prompt="waves",
        api_key="test-key",
        references=[{"type": "image", "url": url}],
    )

    body = json.loads(route.calls[0].request.content)
    assert body["private_asset_review"] is True
    assert body["image_with_roles"] == [{"url": url, "role": "reference_image"}]


@pytest.mark.parametrize("model", _SEEDANCE_PRIVATE_REVIEW_MODELS)
@pytest.mark.parametrize("extra_body,expected_review", [({}, True), ({"private_asset_review": False}, False)])
def test_seedance_canonical_references_enable_review_after_extra_body_merge(
    respx_mock, model, extra_body, expected_review
):
    route = respx_mock.post("https://toapis.com/v1/videos/generations").respond(
        json={"id": "video_task_123", "object": "generation.task", "model": model, "status": "queued"}
    )
    references = [
        {"type": "image", "url": "https://files.example/second.png"},
        {"type": "video", "url": "https://files.example/ref.mp4"},
        {"type": "image", "url": "https://files.example/first.png"},
        {"type": "audio", "url": "https://files.example/ref.mp3"},
    ]

    response = litellm.video_generation(
        model=f"toapis/{model}",
        prompt="virtual character waves",
        api_key="test-key",
        seconds="4",
        resolution="480p",
        aspect_ratio="16:9",
        references=references,
        extra_body=extra_body,
    )

    assert response.status == "queued"
    assert route.call_count == 1
    assert json.loads(route.calls[0].request.content) == {
        "model": model,
        "prompt": "virtual character waves",
        "duration": 4,
        "resolution": "480p",
        "aspect_ratio": "16:9",
        "image_with_roles": [
            {"url": "https://files.example/second.png", "role": "reference_image"},
            {"url": "https://files.example/first.png", "role": "reference_image"},
        ],
        "video_with_roles": [{"url": "https://files.example/ref.mp4", "role": "reference_video"}],
        "audio_with_roles": [{"url": "https://files.example/ref.mp3", "role": "reference_audio"}],
        "private_asset_review": expected_review,
        **({"video_operation": "generate"} if model == "seedance-2-5" else {}),
    }


def test_seedance_private_review_extra_body_overrides_top_level(respx_mock):
    route = respx_mock.post("https://toapis.com/v1/videos/generations").respond(
        json={"id": "video_task_123", "object": "generation.task", "model": "seedance-2", "status": "queued"}
    )

    litellm.video_generation(
        model="toapis/seedance-2",
        prompt="waves",
        api_key="test-key",
        private_asset_review=True,
        extra_body={
            "image_with_roles": [{"url": "https://files.example/ref.png", "role": "reference_image"}],
            "private_asset_review": False,
        },
    )

    assert json.loads(route.calls[0].request.content)["private_asset_review"] is False


@pytest.mark.parametrize(
    "extra_body,message",
    [
        ({"image_urls": ["asset://old-image"], "private_asset_review": True}, "cannot be combined with asset://"),
        ({"image_urls": ["https://files.example/ref.png"], "private_asset_review": "true"}, "must be a boolean"),
    ],
)
def test_seedance_private_review_invalid_extra_body_rejected_before_submission(respx_mock, extra_body, message):
    route = respx_mock.post("https://toapis.com/v1/videos/generations").respond(
        json={"id": "video_task_123", "object": "generation.task", "status": "queued"}
    )

    with pytest.raises(litellm.UnsupportedParamsError, match=message):
        litellm.video_generation(
            model="toapis/seedance-2",
            prompt="waves",
            api_key="test-key",
            extra_body=extra_body,
        )

    assert route.call_count == 0


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
    [
        ("", "queued"),
        ("pending", "queued"),
        ("queued", "queued"),
        ("in_progress", "in_progress"),
        ("completed", "completed"),
        ("failed", "failed"),
    ],
)
def test_toapis_video_create_accepts_live_task_envelope(
    response_id, task_id, expected_task_id, status, expected_status
):
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
            raw_response=httpx.Response(
                200, json={"id": "video_task_123", "object": "video", "status": "queued", **overrides}
            ),
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
        "private_asset_review": True,
    }


@pytest.mark.parametrize("model", ["seedance-2-5", "wan3.0-video"])
@pytest.mark.parametrize(
    "status,expected_status",
    [("", "queued"), ("pending", "queued"), ("queued", "queued"), ("in_progress", "in_progress")],
)
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
        "private_asset_review": True,
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
    assert captured_create_bodies[0]["private_asset_review"] is True
