import copy
import ipaddress
import json
import socket
from types import SimpleNamespace

import httpx
import pytest

import litellm
from litellm.llms.base_llm.submission_utils import get_submission_outcome, is_reexecution_blocked
from litellm.llms.guanghe.videos import reference_upload
from litellm.router import Router

BASE = "https://guanghe.example/prefix/v1"
MODEL = "guanghe/seedance2.5_企业折"
IMAGE = "https://media.example/person.png?signature=original"
VIDEO = "https://media.example/clip.mp4"
UPLOADED = "https://oss.example/signed-reference?signature=provider"


@pytest.fixture(autouse=True)
def network(monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "ssl_verify", True)

    def resolve(host, port, **kwargs):
        try:
            address = str(ipaddress.ip_address(host))
        except ValueError:
            address = "93.184.216.34"
        return [(socket.AF_INET, socket.SOCK_STREAM, socket.IPPROTO_TCP, "", (address, port))]

    monkeypatch.setattr("litellm.litellm_core_utils.url_utils.socket.getaddrinfo", resolve)


async def generate(asynchronous, references=None, **extra):
    args = dict(
        model=MODEL,
        prompt="edit the teacher",
        seconds="-1",
        resolution="480p",
        api_key="provider-key",
        api_base=BASE,
        extra_headers={"Idempotency-Key": "generation-one", "X-Secret": "provider-only"},
    )
    args.update(extra)
    if references is not None:
        args["references"] = references
    return await litellm.avideo_generation(**args) if asynchronous else litellm.video_generation(**args)


def create_route(mock):
    return mock.post(BASE + "/video/tasks").respond(
        202, json={"success": True, "data": {"job_id": "one", "status": "accepting"}}
    )


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_reuploads_images_videos_preserving_order_and_audio(asynchronous, respx_mock):
    refs = [
        {"type": "image", "url": IMAGE},
        {"type": "video", "url": VIDEO},
        {"type": "image", "url": IMAGE},
        {"type": "audio", "url": "https://media.example/speech.mp3"},
    ]
    original = copy.deepcopy(refs)
    picture = respx_mock.get(IMAGE).respond(200, content=b"picture", headers={"Content-Type": "image/png"})
    movie = respx_mock.get(VIDEO).respond(200, content=b"movie", headers={"Content-Type": "video/mp4"})
    upload = respx_mock.post(BASE + "/files/upload").mock(
        side_effect=[
            httpx.Response(
                200,
                json={
                    "success": True,
                    "data": {"signed_url": UPLOADED, "url": "https://wrong.example/raw", "status": "ready"},
                },
            ),
            httpx.Response(200, json={"success": True, "data": {"url": UPLOADED + "-video"}}),
        ]
    )
    create = create_route(respx_mock)
    result = await generate(asynchronous, refs)
    assert result.status == "queued"
    assert refs == original
    assert picture.call_count == movie.call_count == 1
    assert upload.call_count == 2 and create.call_count == 1
    params = json.loads(create.calls[0].request.content)["params"]
    assert params == {
        "duration": -1,
        "resolution": "480p",
        "aspectRatio": "adaptive",
        "imageUrls": [UPLOADED, UPLOADED],
        "videoUrls": [UPLOADED + "-video"],
        "audioUrls": [refs[-1]["url"]],
    }
    for route in (picture, movie):
        headers = route.calls[0].request.headers
        assert not ({"authorization", "x-secret", "idempotency-key", "cookie"} & set(headers))
    for index, call in enumerate(upload.calls):
        req = call.request
        assert req.headers["authorization"] == "Bearer provider-key"
        assert "idempotency-key" not in req.headers
        assert req.headers["content-type"].startswith("multipart/form-data;")
        assert (b"picture" if index == 0 else b"movie") in req.content
        assert b"input_material" in req.content and b'filename="reference.' in req.content
    assert create.calls[0].request.headers["idempotency-key"] == "generation-one"
    assert respx_mock.calls[-1].request.url.path.endswith("/video/tasks")


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_first_last_frames_keep_their_roles(asynchronous, respx_mock):
    download = respx_mock.get(IMAGE).respond(200, content=b"frame", headers={"Content-Type": "image/png"})
    last = respx_mock.get(IMAGE + "-last").respond(200, content=b"last-frame", headers={"Content-Type": "image/png"})
    upload = respx_mock.post(BASE + "/files/upload").mock(
        side_effect=[
            httpx.Response(200, json={"success": True, "data": {"url": UPLOADED}}),
            httpx.Response(200, json={"success": True, "data": {"url": UPLOADED + "-last"}}),
        ]
    )
    create = create_route(respx_mock)
    await generate(
        asynchronous,
        [
            {"type": "image", "role": "first_frame", "url": IMAGE},
            {"type": "image", "role": "last_frame", "url": IMAGE + "-last"},
        ],
    )
    params = json.loads(create.calls[0].request.content)["params"]
    assert params["firstFrameUrl"] == UPLOADED
    assert params["lastFrameUrl"] == UPLOADED + "-last"
    assert download.call_count == last.call_count == 1
    assert upload.call_count == 2


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "case", ["http", "mime", "empty", "oversized_header", "oversized_body", "redirect_private", "credentials"]
)
async def test_bad_sources_never_upload_or_create(asynchronous, case, respx_mock, monkeypatch):
    source = IMAGE
    if case == "credentials":
        source = "https://user:password@media.example/person.png"
    route = respx_mock.get(source)
    if case == "http":
        route.respond(404)
    elif case == "mime":
        route.respond(200, content=b"HTML", headers={"Content-Type": "text/html"})
    elif case == "empty":
        route.respond(200, content=b"", headers={"Content-Type": "image/png"})
    elif case == "redirect_private":
        route.respond(302, headers={"Location": "http://169.254.169.254/latest/meta-data"})
    else:
        monkeypatch.setattr(reference_upload, "MAX_UPLOAD_BYTES", 4)
        headers = {"Content-Type": "image/png", "Content-Length": "10" if case == "oversized_header" else "1"}
        route.respond(200, content=b"0123456789", headers=headers)
    upload = respx_mock.post(BASE + "/files/upload").respond(200)
    create = create_route(respx_mock)
    with pytest.raises(Exception):
        await generate(asynchronous, [{"type": "image", "url": source}])
    assert upload.call_count == create.call_count == 0
    assert all(call.request.url.host != "169.254.169.254" for call in respx_mock.calls)


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize(
    "failure", ["business", "http", "malformed", "missing_url", "unsafe_url", "not_ready", "timeout", "redirect"]
)
async def test_failed_upload_never_creates_video(asynchronous, failure, respx_mock):
    respx_mock.get(VIDEO).respond(200, content=b"movie", headers={"Content-Type": "video/mp4"})
    upload = respx_mock.post(BASE + "/files/upload")
    if failure == "timeout":
        upload.mock(side_effect=httpx.ReadTimeout("secret URL should not be exposed"))
    elif failure == "malformed":
        upload.respond(200, text="not JSON")
    elif failure == "redirect":
        upload.respond(
            302, headers={"Location": "https://other.example/upload"}, json={"success": True, "data": {"url": UPLOADED}}
        )
    else:
        body = {"success": True, "data": {"url": UPLOADED}}
        if failure == "business":
            body = {"success": False, "message": "quota exhausted"}
        if failure == "missing_url":
            body = {"success": True, "data": {}}
        if failure == "unsafe_url":
            body["data"]["url"] = "https://user:secret@oss.example/file"
        if failure == "not_ready":
            body["data"]["status"] = "processing"
        upload.respond(503 if failure == "http" else 200, json=body)
    create = create_route(respx_mock)
    with pytest.raises(Exception) as caught:
        await generate(asynchronous, [{"type": "video", "url": VIDEO}])
    assert create.call_count == 0 and upload.call_count == 1
    assert get_submission_outcome(caught.value) == "rejected"
    assert is_reexecution_blocked(caught.value)


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_preparation_deadline_stops_generation_after_upload(asynchronous, respx_mock, monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(reference_upload, "time", SimpleNamespace(monotonic=lambda: clock[0]))
    respx_mock.get(IMAGE).respond(200, content=b"image", headers={"Content-Type": "image/png"})

    def finish_after_deadline(request):
        clock[0] = 11.0
        return httpx.Response(200, json={"success": True, "data": {"url": UPLOADED}})

    upload = respx_mock.post(BASE + "/files/upload").mock(side_effect=finish_after_deadline)
    create = create_route(respx_mock)
    with pytest.raises(Exception) as caught:
        await generate(asynchronous, [{"type": "image", "url": IMAGE}], timeout=10)
    assert upload.call_count == 1 and create.call_count == 0
    assert get_submission_outcome(caught.value) == "rejected"
    assert is_reexecution_blocked(caught.value)


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_later_upload_failure_preserves_inputs_and_never_generates(asynchronous, respx_mock):
    refs = [{"type": "image", "url": IMAGE}, {"type": "video", "url": VIDEO}]
    original = copy.deepcopy(refs)
    respx_mock.get(IMAGE).respond(200, content=b"image", headers={"Content-Type": "image/png"})
    respx_mock.get(VIDEO).respond(200, content=b"video", headers={"Content-Type": "video/mp4"})
    upload = respx_mock.post(BASE + "/files/upload").mock(
        side_effect=[
            httpx.Response(200, json={"success": True, "data": {"url": UPLOADED}}),
            httpx.Response(200, json={"success": False, "message": "video upload rejected"}),
        ]
    )
    create = create_route(respx_mock)
    with pytest.raises(Exception):
        await generate(asynchronous, refs)
    assert refs == original
    assert upload.call_count == 2 and create.call_count == 0


async def test_generation_transport_retry_reuses_uploaded_body(respx_mock):
    download = respx_mock.get(IMAGE).respond(200, content=b"picture", headers={"Content-Type": "image/png"})
    upload = respx_mock.post(BASE + "/files/upload").respond(200, json={"success": True, "data": {"url": UPLOADED}})
    create = respx_mock.post(BASE + "/video/tasks").mock(
        side_effect=[
            httpx.ConnectError("not connected"),
            httpx.Response(202, json={"success": True, "data": {"job_id": "one", "status": "accepting"}}),
        ]
    )
    video = await generate(True, [{"type": "image", "url": IMAGE}])
    assert video.status == "queued"
    assert upload.call_count == download.call_count == 1 and create.call_count == 2
    assert create.calls[0].request.content == create.calls[1].request.content
    assert create.calls[0].request.headers["idempotency-key"] == create.calls[1].request.headers["idempotency-key"]


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_redirects_and_octet_stream_extension(asynchronous, respx_mock):
    respx_mock.get(VIDEO).respond(302, headers={"Location": "https://cdn.example/clip.mp4?signature=kept"})
    cdn = respx_mock.get("https://cdn.example/clip.mp4?signature=kept").respond(
        200, content=b"movie", headers={"Content-Type": "application/octet-stream"}
    )
    upload = respx_mock.post(BASE + "/files/upload").respond(200, json={"success": True, "data": {"url": UPLOADED}})
    create_route(respx_mock)
    await generate(asynchronous, [{"type": "video", "url": VIDEO}])
    assert "authorization" not in cdn.calls[0].request.headers
    assert b"Content-Type: video/mp4" in upload.calls[0].request.content


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_no_references_and_local_image_do_not_double_upload(asynchronous, respx_mock):
    create = create_route(respx_mock)
    await generate(asynchronous)
    assert len(respx_mock.calls) == 1
    upload = respx_mock.post(BASE + "/files/upload").respond(200, json={"success": True, "data": {"url": UPLOADED}})
    await generate(asynchronous, input_reference=b"\x89PNG\r\n\x1a\n", resolution=None)
    assert upload.call_count == 1 and create.call_count == 2
    assert all(call.request.method == "POST" for call in respx_mock.calls)


async def test_router_upload_failure_does_not_retry_or_fallback(respx_mock):
    respx_mock.get(IMAGE).respond(200, content=b"picture", headers={"Content-Type": "image/png"})
    upload = respx_mock.post(BASE + "/files/upload").respond(200, json={"success": False, "message": "upload failed"})
    create = create_route(respx_mock)
    backup = respx_mock.post("https://backup.example/v1/videos/generations").respond(200)
    router = Router(
        model_list=[
            {"model_name": "primary", "litellm_params": {"model": MODEL, "api_base": BASE, "api_key": "test"}},
            {
                "model_name": "backup",
                "litellm_params": {
                    "model": "toapis/seedance-2-5",
                    "api_base": "https://backup.example/v1",
                    "api_key": "test",
                },
            },
        ],
        num_retries=3,
        fallbacks=[{"primary": ["backup"]}],
    )
    with pytest.raises(Exception):
        await router.avideo_generation(model="primary", prompt="edit", references=[{"type": "image", "url": IMAGE}])
    assert upload.call_count == 1
    assert create.call_count == backup.call_count == 0
