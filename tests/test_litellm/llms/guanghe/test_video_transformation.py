import json
from unittest.mock import Mock

import httpx
import pytest

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.submission_utils import get_submission_outcome
from litellm.llms.guanghe.videos.transformation import (
    DEFAULT_API_BASE,
    MODEL_RESOLUTIONS,
    GuangheVideoConfig,
    build_endpoint,
)
from litellm.router import Router
from litellm.types.router import GenericLiteLLMParams
from litellm.types.videos.utils import decode_video_id_with_provider, encode_video_id_with_provider

MODEL = "seedance2.5_企业折"
BASE = "https://guanghe.example/api/inspiration-waterfall/v1"
REF = {"type": "image", "role": "reference", "url": "https://media.example/person.jpg"}


@pytest.fixture(autouse=True)
def isolate_transport(monkeypatch, respx_mock):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(
        "litellm.llms.guanghe.videos.reference_upload._validated_url", lambda url: (url, httpx.URL(url).host)
    )
    respx_mock.get(REF["url"]).respond(200, content=b"reference", headers={"Content-Type": "image/jpeg"})
    respx_mock.post(BASE + "/files/upload").respond(200, json={"success": True, "data": {"url": REF["url"]}})


@pytest.mark.parametrize("model", MODEL_RESOLUTIONS)
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_public_create_contract(model, asynchronous, respx_mock):
    route = respx_mock.post(BASE + "/video/tasks").respond(
        202, json={"success": True, "data": {"job_id": "job-one", "status": "accepting"}}
    )
    kwargs = dict(
        model="guanghe/" + model,
        prompt="fly",
        seconds="4",
        resolution="480p",
        aspect_ratio="3:4",
        references=[REF],
        api_key="test-key",
        api_base=BASE,
        extra_headers={"idempotency-key": "business-one"},
    )
    video = await litellm.avideo_generation(**kwargs) if asynchronous else litellm.video_generation(**kwargs)
    assert video.status == "queued"
    assert decode_video_id_with_provider(video.id) == {
        "video_id": "job-one",
        "custom_llm_provider": "guanghe",
        "model_id": model,
    }
    assert route.call_count == 1
    request = route.calls[0].request
    assert request.headers.get_list("idempotency-key") == ["business-one"]
    assert request.headers["authorization"] == "Bearer test-key"
    assert json.loads(request.content) == {
        "model_id": model,
        "prompt": "fly",
        "params": {"duration": 4, "resolution": "480p", "aspectRatio": "3:4", "imageUrls": [REF["url"]]},
    }


@pytest.mark.parametrize(
    "status,expected",
    [
        ("accepting", "queued"),
        ("unknown", "queued"),
        ("processing", "in_progress"),
        ("success", "completed"),
        ("succeeded", "completed"),
        ("failed", "failed"),
        ("cancelled", "failed"),
        ("manual_review", "failed"),
    ],
)
def test_statuses(status, expected):
    response = httpx.Response(
        200,
        json={
            "success": True,
            "data": {
                "task_id": "task-one",
                "model_id": MODEL,
                "status": status,
                "video_url": "https://cdn.example/result.mp4" if expected == "completed" else "",
                "error_message": "",
                "credits_cost": 28,
            },
        },
    )
    video = GuangheVideoConfig().transform_video_status_retrieve_response(response, Mock(), "guanghe")
    assert video.status == expected
    assert video._hidden_params["provider_status"] == status
    assert video._hidden_params["credits_cost"] == 28
    if expected == "failed":
        assert video.error["code"] == status
    if status == "manual_review":
        assert "Refund is not confirmed" in video.error["message"]


def test_completed_without_url_is_contract_failure():
    response = httpx.Response(200, json={"success": True, "data": {"task_id": "task-one", "status": "success"}})
    video = GuangheVideoConfig().transform_video_status_retrieve_response(response, Mock(), "guanghe")
    assert video.status == "failed"
    assert video.error["code"] == "provider_contract_error"


@pytest.mark.parametrize(
    "payload,outcome",
    [
        ({"success": True, "data": {"status": "accepting"}}, "unknown"),
        ({"success": True, "data": {"job_id": "existing", "status": "unrecognized"}}, "accepted"),
        ({"success": False, "code": "invalid_params", "message": "bad duration"}, "rejected"),
        ({"success": False, "code": "upstream_error", "message": "unknown"}, "unknown"),
        ({"success": False, "code": "invalid_params", "data": {"job_id": "existing"}}, "accepted"),
    ],
)
def test_response_provenance(payload, outcome):
    with pytest.raises(BaseLLMException) as caught:
        GuangheVideoConfig().transform_video_create_response(
            MODEL, httpx.Response(200, json=payload), Mock(), "guanghe"
        )
    assert get_submission_outcome(caught.value) == outcome


@pytest.mark.parametrize("failure", ["timeout", "bad_envelope", "server_error"])
async def test_router_does_not_retry_uncertain_submissions(failure, respx_mock):
    route = respx_mock.post(BASE + "/video/tasks")
    if failure == "timeout":
        route.mock(side_effect=httpx.ReadTimeout("unknown submission"))
    elif failure == "server_error":
        route.respond(502, json={"success": False, "message": "unknown", "code": "upstream_error"})
    else:
        route.respond(202, json={"unexpected": True})
    router = Router(
        model_list=[
            {
                "model_name": "gh",
                "litellm_params": {"model": "guanghe/" + MODEL, "api_key": "test-key", "api_base": BASE},
                "model_info": {"mode": "video_generation"},
            }
        ],
        num_retries=3,
    )
    with pytest.raises(Exception) as caught:
        await router.avideo_generation(
            model="gh", prompt="fly", seconds="4", resolution="480p", aspect_ratio="3:4", references=[REF]
        )
    assert get_submission_outcome(caught.value) == "unknown"
    assert route.call_count == 1


async def test_transport_replay_keeps_idempotency_key(respx_mock):
    route = respx_mock.post(BASE + "/video/tasks").mock(
        side_effect=[
            httpx.ConnectError("lost connection"),
            httpx.Response(202, json={"success": True, "data": {"job_id": "one", "status": "accepting"}}),
        ]
    )
    video = await litellm.avideo_generation(model="guanghe/" + MODEL, prompt="fly", api_key="test-key", api_base=BASE)
    assert video.status == "queued"
    assert route.call_count == 2
    assert route.calls[0].request.headers["idempotency-key"] == route.calls[1].request.headers["idempotency-key"]


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_query_and_content_redirect(asynchronous, respx_mock):
    video_id = encode_video_id_with_provider("task-one", "guanghe", MODEL)
    status = respx_mock.get(BASE + "/video/tasks/task-one").respond(
        200,
        json={
            "success": True,
            "data": {
                "task_id": "task-one",
                "model_id": MODEL,
                "status": "success",
                "video_url": "https://cdn.example/result.mp4",
            },
        },
    )
    respx_mock.get(BASE + "/video/tasks/task-one/download").respond(
        302, headers={"Location": "https://cdn.example/result.mp4"}
    )
    cdn = respx_mock.get("https://cdn.example/result.mp4").respond(
        200, content=b"video-bytes", headers={"Content-Type": "video/mp4"}
    )
    kwargs = dict(video_id=video_id, api_key="test-key", api_base=BASE)
    video = await litellm.avideo_status(**kwargs) if asynchronous else litellm.video_status(**kwargs)
    content = await litellm.avideo_content(**kwargs) if asynchronous else litellm.video_content(**kwargs)
    assert video.status == "completed"
    assert content == b"video-bytes"
    assert status.call_count == 1
    assert "authorization" not in cdn.calls[0].request.headers


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("success", [False, True])
async def test_local_upload_before_create(asynchronous, success, respx_mock):
    uploaded = (
        {"success": True, "data": {"url": REF["url"]}} if success else {"success": False, "message": "upload rejected"}
    )
    upload = respx_mock.post(BASE + "/files/upload").respond(200, json=uploaded)
    create = respx_mock.post(BASE + "/video/tasks").respond(
        202, json={"success": True, "data": {"job_id": "one", "status": "accepting"}}
    )
    kwargs = dict(
        model="guanghe/" + MODEL,
        prompt="fly",
        input_reference=b"\x89PNG\r\n\x1a\n",
        seconds="4",
        api_key="test-key",
        api_base=BASE,
    )
    if success:
        video = await litellm.avideo_generation(**kwargs) if asynchronous else litellm.video_generation(**kwargs)
        assert video.status == "queued"
        assert json.loads(create.calls[0].request.content)["params"]["imageUrls"] == [REF["url"]]
    else:
        with pytest.raises(Exception):
            if asynchronous:
                await litellm.avideo_generation(**kwargs)
            else:
                litellm.video_generation(**kwargs)
        assert create.call_count == 0
    assert upload.call_count == 1
    assert upload.calls[0].request.headers["content-type"].startswith("multipart/form-data;")


@pytest.mark.parametrize(
    "params",
    [
        {"seconds": "3"},
        {"seconds": "4.5"},
        {"resolution": "4K"},
        {"size": "480x640"},
        {"resolution": "480p", "extra_body": {"params": {"duration": 20}}},
        {"references": [REF], "imageUrls": ["https://other.example/image"]},
        {"references": [REF, {**REF, "role": "first_frame"}]},
    ],
)
def test_reject_unsupported_or_conflicting_inputs(params):
    with pytest.raises((ValueError, litellm.UnsupportedParamsError)):
        GuangheVideoConfig().map_openai_params(params, MODEL, True)


def test_multimodal_and_frame_mapping():
    config = GuangheVideoConfig()
    refs = [
        {"type": kind, "role": "reference", "url": f"https://media.example/{kind}"}
        for kind in ("image", "video", "audio")
    ]
    mapped = config.map_openai_params({"references": refs}, MODEL, False)
    assert [mapped[key] for key in ("imageUrls", "videoUrls", "audioUrls")] == [[ref["url"]] for ref in refs]
    frames = [{**REF, "role": "first_frame"}, {**REF, "role": "last_frame", "url": "https://media.example/end.jpg"}]
    mapped = config.map_openai_params({"references": frames}, MODEL, False)
    assert mapped["firstFrameUrl"] == REF["url"]
    assert mapped["lastFrameUrl"] == frames[1]["url"]


@pytest.mark.parametrize(
    "base", [None, "https://newapi.aitoken.name", DEFAULT_API_BASE, DEFAULT_API_BASE + "/video/tasks"]
)
def test_endpoint_normalization(base):
    assert build_endpoint(base, "video/tasks") == DEFAULT_API_BASE + "/video/tasks"


def test_download_path_encoding_and_json_failure():
    config = GuangheVideoConfig()
    video_id = encode_video_id_with_provider("a/b?c", "guanghe", MODEL)
    url, _ = config.transform_video_content_request(video_id, BASE + "/video/tasks", GenericLiteLLMParams(), {})
    assert url.endswith("/a%2Fb%3Fc/download")
    with pytest.raises(BaseLLMException):
        config.transform_video_content_response(
            httpx.Response(200, json={"success": False, "code": "task_not_ready"}), Mock()
        )


@pytest.mark.parametrize("mime", ["text/html", "application/xml", "text/plain", ""])
def test_download_rejects_error_pages(mime):
    with pytest.raises(BaseLLMException):
        GuangheVideoConfig().transform_video_content_response(
            httpx.Response(200, content=b"error page", headers={"Content-Type": mime}), Mock()
        )
