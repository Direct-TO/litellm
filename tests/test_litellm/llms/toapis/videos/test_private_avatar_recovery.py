import copy
import json
from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from openai import APIError

import litellm
from litellm.llms.base_llm.submission_utils import get_provider_task_id, get_submission_outcome, is_reexecution_blocked
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.llms.toapis.videos import handler
from litellm.types.videos.utils import decode_video_id_with_provider


def privacy_error(indices=(1, 4)):
    inner = {
        "code": "seedance_upstream_error",
        "data": None,
        "error": {
            "code": "***.PrivacyInformation",
            "message": "The request failed because the input image "
            + " ".join(f"'content[{index}]'" for index in indices)
            + " may contain real person. Request id: upstream-request-123",
        },
    }
    return {"code": "fail_to_fetch_task", "message": json.dumps(inner), "data": None}


class AvatarServer:
    def __init__(self, first=None, second=None, review_status="active", fail_second_asset=False):
        self.first = first if first is not None else httpx.Response(400, json=privacy_error())
        self.second = (
            second
            if second is not None
            else httpx.Response(200, json={"id": "task-recovered", "object": "generation.task", "status": "queued"})
        )
        self.review_status = review_status
        self.fail_second_asset = fail_second_asset
        self.requests = []
        self.creates = []
        self.uploads = []
        self.groups = []
        self.polls = []

    def __call__(self, request):
        self.requests.append(request)
        path = request.url.path
        if path.endswith("/videos/generations"):
            self.creates.append(json.loads(request.content))
            response = self.first if len(self.creates) == 1 else self.second
            if isinstance(response, Exception):
                raise response
            return httpx.Response(response.status_code, headers=response.headers, content=response.content)
        if path.endswith("/uploads/images"):
            return httpx.Response(
                200,
                json={
                    "success": True,
                    "message": "",
                    "data": {
                        "id": "image-local",
                        "url": "https://files.example/local.png",
                        "mime_type": "image/png",
                        "size": 8,
                    },
                },
            )
        if path.endswith("/groups"):
            self.groups.append(json.loads(request.content))
            data = {"group_id": f"group-{len(self.groups)}"}
        elif path.endswith("/assets"):
            self.uploads.append(json.loads(request.content))
            data = {"asset_id": f"asset-{len(self.uploads)}", "status": "processing"}
        elif "/assets/" in path:
            self.polls.append(request)
            asset_id = path.rsplit("/", 1)[-1]
            data = {
                "asset_id": asset_id,
                "asset_url": "asset://" + asset_id,
                "status": "failed" if self.fail_second_asset and asset_id == "asset-2" else self.review_status,
            }
        else:
            raise AssertionError(f"Unexpected endpoint: {request.method} {path}")
        return httpx.Response(200, json={"success": True, "message": "", "data": data})


async def generate(server, use_async, **overrides):
    params = {
        "model": "toapis/seedance-2-5",
        "prompt": "Keep all four fictional references in order",
        "api_key": "deployment-key",
        "api_base": "https://toapis.cn/v1",
        "seconds": "30",
        "resolution": "720p",
        "aspect_ratio": "16:9",
        "references": [
            {"type": "image", "url": f"https://files.example/ref-{index}.png", "role": "reference"}
            for index in range(1, 5)
        ],
    }
    params.update(overrides)
    if use_async:
        client = AsyncHTTPHandler()
        await client.client.aclose()
        client.client = httpx.AsyncClient(transport=httpx.MockTransport(server))
        try:
            return await litellm.avideo_generation(client=client, **params)
        finally:
            await client.client.aclose()
    client = HTTPHandler()
    client.client.close()
    client.client = httpx.Client(transport=httpx.MockTransport(server))
    try:
        return litellm.video_generation(client=client, **params)
    finally:
        client.client.close()


@pytest.fixture(autouse=True)
def fast_polling(monkeypatch):
    monkeypatch.setattr(handler, "REVIEW_POLL_INTERVAL_SECONDS", 0.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("base", ["https://toapis.cn/v1", "https://gateway.example/prefix/v1/videos/generations"])
async def test_review_only_flagged_images_and_retry_with_preserved_body(use_async, base):
    server = AvatarServer()
    references = [
        {"type": "image", "url": f"https://files.example/ref-{index}.png", "role": "reference"} for index in range(1, 5)
    ]
    original = copy.deepcopy(references)
    response = await generate(
        server,
        use_async,
        api_base=base,
        references=references,
        extra_body={"generate_audio": True},
        extra_headers={"Idempotency-Key": "original-key", "X-Custom": "keep"},
    )
    assert response.status == "queued"
    assert decode_video_id_with_provider(response.id)["video_id"] == "task-recovered"
    assert references == original
    assert len(server.creates) == 2
    assert [upload["source_url"] for upload in server.uploads] == [original[0]["url"], original[3]["url"]]
    expected = copy.deepcopy(server.creates[0])
    expected["image_with_roles"][0]["url"] = "asset://asset-1"
    expected["image_with_roles"][3]["url"] = "asset://asset-2"
    assert server.creates[1] == expected
    assert len(server.polls) == 2
    for request in server.requests:
        assert request.headers["Authorization"] == "Bearer deployment-key"
        assert request.headers["X-Custom"] == "keep"
        assert request.url.host == httpx.URL(base).host
        if "private-avatar" in request.url.path:
            assert "Idempotency-Key" not in request.headers
            assert "/videos/generations/v1/" not in request.url.path
    assert server.requests[0].headers["Idempotency-Key"] == "original-key"
    assert server.requests[-1].headers["Idempotency-Key"].startswith("litellm-avatar-")


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
async def test_native_image_urls_deduplicates_only_selected_sources(use_async):
    server = AvatarServer(first=httpx.Response(400, json=privacy_error((1, 2))))
    urls = ["https://files.example/same.png"] * 2
    await generate(
        server, use_async, references=None, resolution=None, aspect_ratio=None, extra_body={"image_urls": urls}
    )
    assert len(server.groups) == len(server.uploads) == 1
    assert server.creates[1]["image_urls"] == ["asset://asset-1", "asset://asset-1"]
    assert urls == ["https://files.example/same.png"] * 2


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
async def test_uploaded_local_reference_can_be_reviewed(use_async):
    server = AvatarServer(first=httpx.Response(400, json=privacy_error((1,))))
    await generate(
        server, use_async, references=None, resolution=None, aspect_ratio=None, input_reference=b"\x89PNG\r\n\x1a\n"
    )
    assert server.uploads[0]["source_url"] == "https://files.example/local.png"
    assert server.creates[1]["image_with_roles"] == [{"url": "asset://asset-1", "role": "reference_image"}]


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize(
    "first",
    [
        httpx.Response(400, json={"code": "InvalidParameter", "message": "ordinary failure"}),
        httpx.Response(400, text="PrivacyInformation: input image 'content[1]' may contain real person"),
        httpx.Response(400, json={"error": {"code": "PrivacyInformation", "message": "may contain real person"}}),
        httpx.Response(400, json=privacy_error((0, 1))),
        httpx.Response(400, json=privacy_error((1, 99))),
        httpx.Response(403, json=privacy_error()),
        httpx.Response(503, json=privacy_error()),
        httpx.Response(400, json={**privacy_error(), "task_id": "already-accepted"}),
        httpx.Response(400, json={**privacy_error(), "data": {"id": "already-accepted"}}),
        httpx.Response(400, json={**privacy_error(), "status": "accepted"}),
    ],
)
async def test_ambiguous_or_unrelated_failures_never_submit_review(use_async, first):
    server = AvatarServer(first=first)
    with pytest.raises(APIError) as caught:
        await generate(server, use_async)
    assert caught.value.status_code == first.status_code
    assert len(server.creates) == 1
    assert server.groups == server.uploads == []
    if "already-accepted" in first.text:
        assert get_submission_outcome(caught.value) == "accepted"
        assert get_provider_task_id(caught.value) == "already-accepted"


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
async def test_existing_asset_and_unknown_content_layout_do_not_trigger_review(use_async):
    for extra in [
        {"image_with_roles": [{"url": "asset://old", "role": "reference_image"}]},
        {"image_urls": ["https://files.example/ref.png"], "content": [{"type": "text", "text": "custom"}]},
        {
            "image_with_roles": [{"url": "https://files.example/ref.png"}],
            "image_urls": ["https://files.example/ref.png"],
        },
    ]:
        server = AvatarServer(first=httpx.Response(400, json=privacy_error((1,))))
        with pytest.raises(APIError):
            await generate(server, use_async, references=None, resolution=None, aspect_ratio=None, extra_body=extra)
        assert len(server.creates) == 1
        assert not server.groups


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
async def test_failed_accepted_task_does_not_start_another_video(use_async):
    server = AvatarServer(
        first=httpx.Response(
            200,
            json={
                "id": "accepted-task",
                "object": "generation.task",
                "status": "failed",
                "error": {"code": "PrivacyInformation", "message": "input image 'content[1]' may contain real person"},
            },
        )
    )
    response = await generate(server, use_async)
    assert response.status == "failed"
    assert len(server.creates) == 1
    assert not server.groups


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
async def test_second_rejection_is_returned_without_another_recovery(use_async):
    server = AvatarServer(second=httpx.Response(400, json=privacy_error()))
    with pytest.raises(litellm.BadRequestError, match="PrivacyInformation") as caught:
        await generate(server, use_async)
    assert is_reexecution_blocked(caught.value)
    assert len(server.creates) == 2
    assert len(server.uploads) == len(server.groups) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("failure", ["failed", "unknown", "second_failed", "timeout"])
async def test_all_assets_must_pass_before_any_resubmission(use_async, failure, monkeypatch):
    server = AvatarServer(
        review_status=failure if failure in ("failed", "unknown") else "active",
        fail_second_asset=failure == "second_failed",
    )
    if failure == "timeout":
        monkeypatch.setattr(handler, "REVIEW_TIMEOUT_SECONDS", 0.0)
    with pytest.raises(litellm.BadRequestError, match="video generation was not resubmitted") as caught:
        await generate(server, use_async)
    assert get_submission_outcome(caught.value) == "rejected"
    assert is_reexecution_blocked(caught.value)
    assert len(server.creates) == 1
    if failure == "second_failed":
        assert len(server.polls) == 2


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("stage", ["first", "second"])
@pytest.mark.parametrize("exception", [httpx.ReadTimeout("lost response"), httpx.RemoteProtocolError("lost response")])
async def test_uncertain_video_submission_preserves_provenance_without_reposting(use_async, stage, exception):
    server = AvatarServer(**{stage: exception})
    with pytest.raises(APIError) as caught:
        await generate(server, use_async)
    assert get_submission_outcome(caught.value) == "unknown"
    assert len(server.creates) == (1 if stage == "first" else 2)
    assert len(server.uploads) == (0 if stage == "first" else 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
async def test_review_transport_failure_never_retries_video(use_async):
    server = AvatarServer()

    def fail_review(request):
        if "/private-avatar/" in request.url.path:
            raise httpx.ReadTimeout("request outcome unknown")
        return server(request)

    with pytest.raises(litellm.BadRequestError, match="private-avatar transport failed"):
        await generate(fail_review, use_async)
    assert len(server.creates) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("model", ["toapis/seedance-2", "toapis/wan3.0-video"])
async def test_other_models_keep_existing_error_behavior(use_async, model):
    server = AvatarServer()
    with pytest.raises(litellm.BadRequestError):
        await generate(server, use_async, model=model, seconds="8")
    assert len(server.creates) == 1
    assert server.groups == []


def test_review_respects_remaining_total_budget_and_preserves_component_timeouts(monkeypatch):
    clock = Mock()
    clock.monotonic.return_value = 10.0
    monkeypatch.setattr(handler, "time", clock)
    timeout = handler._review_timeout(httpx.Timeout(connect=2, read=60, write=60, pool=3), deadline=15.0)
    assert timeout.as_dict() == {"connect": 2, "read": 5, "write": 5, "pool": 3}
    clock.monotonic.return_value = 15.0
    with pytest.raises(handler.ToAPISVideoRecoveryError, match="timed out"):
        handler._review_timeout(timeout, deadline=15.0)


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
async def test_processing_review_stops_at_total_deadline(use_async, monkeypatch):
    now = [0.0]

    def advance(seconds):
        now[0] += seconds

    async def async_advance(seconds):
        advance(seconds)

    monkeypatch.setattr(handler, "time", SimpleNamespace(monotonic=lambda: now[0], sleep=advance))
    monkeypatch.setattr(handler, "asyncio", SimpleNamespace(sleep=async_advance))
    monkeypatch.setattr(handler, "REVIEW_TIMEOUT_SECONDS", 6.0)
    monkeypatch.setattr(handler, "REVIEW_POLL_INTERVAL_SECONDS", 5.0)
    server = AvatarServer(review_status="processing")
    with pytest.raises(litellm.BadRequestError, match="review timed out") as caught:
        await generate(server, use_async)
    assert is_reexecution_blocked(caught.value)
    assert now[0] == 6
    assert len(server.creates) == len(server.polls) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("failure", ["http", "json", "success_false", "missing_id", "asset_id", "asset_url"])
async def test_invalid_review_responses_fail_without_generation(use_async, failure):
    server = AvatarServer()

    def invalid_review(request):
        response = server(request)
        if "/private-avatar/" not in request.url.path:
            return response
        if failure == "http":
            return httpx.Response(403, json={"message": "denied"})
        if failure == "json":
            return httpx.Response(200, text="not json")
        if failure == "success_false":
            return httpx.Response(200, json={"success": False, "data": {"group_id": "group"}})
        if failure == "missing_id":
            return httpx.Response(200, json={"success": True, "data": {}})
        if request.method == "GET":
            data = response.json()
            data["data"][failure] = "different" if failure == "asset_id" else "https://unexpected.example/ref.png"
            return httpx.Response(200, json=data)
        return response

    with pytest.raises(litellm.BadRequestError) as caught:
        await generate(invalid_review, use_async)
    assert is_reexecution_blocked(caught.value)
    assert len(server.creates) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["review", "retry"])
async def test_router_cannot_repeat_recovery_via_retry_policy_order_or_fallback(monkeypatch, failure):
    from litellm.types.router import RetryPolicy

    server = AvatarServer(
        review_status="failed" if failure == "review" else "active", second=httpx.Response(400, json=privacy_error())
    )
    deployments = []

    async def routed_generate(**kwargs):
        deployments.append(kwargs["api_key"])
        return await generate(server, True)

    monkeypatch.setattr("litellm.videos.avideo_generation", routed_generate)
    model_list = [
        {
            "model_name": group,
            "litellm_params": {
                "model": "toapis/seedance-2-5",
                "api_key": key,
                "weight": 1,
                "order": order,
            },
            "model_info": {"id": key, "supported_endpoints": ["/v1/videos"]},
        }
        for group, key, order in [
            ("video-model", "primary", 1),
            ("video-model", "next-order", 2),
            ("backup-model", "fallback", 1),
        ]
    ]
    router = litellm.Router(
        model_list=model_list,
        num_retries=3,
        retry_after=0,
        retry_policy=RetryPolicy(BadRequestErrorRetries=3),
        fallbacks=[{"video-model": ["backup-model"]}],
        enable_weighted_failover=True,
    )
    with pytest.raises(litellm.BadRequestError) as caught:
        await router.avideo_generation(model="video-model", prompt="fictional character")
    assert get_submission_outcome(caught.value) == "rejected"
    assert is_reexecution_blocked(caught.value)
    assert deployments == ["primary"]
    assert len(server.creates) == (1 if failure == "review" else 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("stage", ["first", "second"])
async def test_malformed_accepted_response_blocks_outer_reexecution(use_async, stage):
    server = AvatarServer(**{stage: httpx.Response(200, json={"object": "video"})})
    with pytest.raises(litellm.BadGatewayError) as caught:
        await generate(server, use_async)
    assert get_submission_outcome(caught.value) == "unknown"
    assert is_reexecution_blocked(caught.value)
    assert len(server.creates) == (1 if stage == "first" else 2)


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["review", "retry", "transport", "malformed"])
async def test_recovery_in_middle_of_fallback_chain_stops_later_targets(monkeypatch, failure):
    from litellm.llms.base_llm.submission_utils import mark_submission_outcome

    response = {
        "review": httpx.Response(400, json=privacy_error()),
        "retry": httpx.Response(400, json=privacy_error()),
        "transport": httpx.ReadTimeout("lost response"),
        "malformed": httpx.Response(200, json={}),
    }[failure]
    server = AvatarServer(review_status="failed" if failure == "review" else "active", second=response)
    attempted = []

    async def call(**kwargs):
        attempted.append(kwargs["model"])
        if kwargs["model"] == "primary":
            raise mark_submission_outcome(
                litellm.BadRequestError(
                    message="ordinary rejection",
                    model="primary",
                    llm_provider="toapis",
                ),
                "rejected",
            )
        if kwargs["model"] == "review":
            return await generate(server, True)
        raise AssertionError("A recovery failure must never reach the last fallback")

    router = litellm.Router(
        model_list=[
            {"model_name": model, "litellm_params": {"model": "toapis/seedance-2-5", "api_key": model}}
            for model in ("primary", "review", "last")
        ],
        num_retries=0,
        fallbacks=[{"primary": ["review", "last"]}],
    )
    with pytest.raises(APIError) as caught:
        await router.async_function_with_fallbacks(
            original_function=call,
            model="primary",
            _router_call_type="avideo_generation",
            metadata={},
            litellm_metadata={},
        )
    assert is_reexecution_blocked(caught.value)
    assert attempted == ["primary", "review"]
    assert get_submission_outcome(caught.value) == ("unknown" if failure in ("transport", "malformed") else "rejected")
