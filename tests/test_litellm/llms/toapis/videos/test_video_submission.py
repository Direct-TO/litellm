import copy
import json

import httpx
import pytest
from openai import APIError

import litellm
from litellm.llms.base_llm.submission_utils import (
    get_provider_task_id,
    get_submission_outcome,
    is_reexecution_blocked,
    mark_submission_outcome,
)
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler
from litellm.types.videos.utils import decode_video_id_with_provider


def privacy_error():
    return {
        "code": "fail_to_fetch_task",
        "message": json.dumps(
            {
                "code": "seedance_upstream_error",
                "error": {
                    "code": "PrivacyInformation",
                    "message": "The input image 'content[1]' may contain real person",
                },
            }
        ),
        "data": None,
    }


class SubmissionServer:
    """Reject every endpoint other than the single video-create request."""

    def __init__(self, response=None):
        self.response = (
            response
            if response is not None
            else httpx.Response(200, json={"id": "video-task", "object": "generation.task", "status": "queued"})
        )
        self.requests = []
        self.creates = []

    def __call__(self, request):
        self.requests.append(request)
        assert request.method == "POST"
        assert request.url.path.endswith("/videos/generations"), f"Unexpected endpoint: {request.url}"
        self.creates.append(json.loads(request.content))
        if isinstance(self.response, Exception):
            raise self.response
        return httpx.Response(self.response.status_code, headers=self.response.headers, content=self.response.content)


async def generate(server, use_async, model="toapis/seedance-2", **overrides):
    params = {
        "model": model,
        "prompt": "A fictional character smiles and waves",
        "api_key": "deployment-key",
        "api_base": "https://toapis.cn/v1",
        "seconds": "4",
        "resolution": "480p",
        "aspect_ratio": "16:9",
        "references": [{"type": "image", "url": "https://files.example/fictional.png", "role": "reference"}],
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


@pytest.fixture(params=["toapis/seedance-2", "toapis/seedance-2-mini", "toapis/seedance-2-fast", "toapis/seedance-2-5"])
def seedance_model(request):
    return request.param


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("base", ["https://toapis.cn/v1", "https://gateway.example/prefix/v1/videos/generations"])
async def test_seedance_submits_once_with_private_asset_review_and_original_urls(use_async, base, seedance_model):
    server = SubmissionServer()
    references = [
        {"type": "image", "url": f"https://files.example/ref-{index}.png", "role": "reference"} for index in (1, 2)
    ]
    original = copy.deepcopy(references)
    response = await generate(
        server,
        use_async,
        model=seedance_model,
        api_base=base,
        references=references,
        extra_body={"generate_audio": True},
        extra_headers={"Idempotency-Key": "business-video-001", "X-Custom": "keep"},
    )
    assert response.status == "queued"
    assert decode_video_id_with_provider(response.id)["video_id"] == "video-task"
    assert references == original
    assert len(server.creates) == len(server.requests) == 1
    body = server.creates[0]
    assert body["model"] == seedance_model.removeprefix("toapis/")
    assert body["private_asset_review"] is True
    assert body["image_with_roles"] == [{"url": reference["url"], "role": "reference_image"} for reference in original]
    assert body["prompt"] == "A fictional character smiles and waves"
    assert body["duration"] == 4
    assert body["resolution"] == "480p"
    assert body["aspect_ratio"] == "16:9"
    assert body["generate_audio"] is True
    request = server.requests[0]
    assert request.url.host == httpx.URL(base).host
    assert request.headers["Authorization"] == "Bearer deployment-key"
    assert request.headers["X-Custom"] == "keep"
    assert request.headers["Idempotency-Key"] == "business-video-001"


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
async def test_person_rejection_returns_original_error_without_review_or_resubmission(use_async, seedance_model):
    server = SubmissionServer(httpx.Response(400, json=privacy_error()))
    with pytest.raises(litellm.BadRequestError, match="may contain real person") as caught:
        await generate(server, use_async, model=seedance_model)
    assert get_submission_outcome(caught.value) == "rejected"
    assert get_provider_task_id(caught.value) is None
    assert len(server.creates) == len(server.requests) == 1
    assert server.creates[0]["private_asset_review"] is True
    assert server.creates[0]["image_with_roles"][0]["url"] == "https://files.example/fictional.png"


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("status", ["queued", "in_progress", "failed"])
async def test_accepted_task_status_and_id_are_preserved_without_reposting(use_async, status):
    error = {"code": "PrivacyInformation", "message": "Virtual portrait review failed"} if status == "failed" else None
    server = SubmissionServer(
        httpx.Response(200, json={"id": "accepted-task", "object": "generation.task", "status": status, "error": error})
    )
    response = await generate(server, use_async)
    assert response.status == status
    assert decode_video_id_with_provider(response.id)["video_id"] == "accepted-task"
    if status == "failed":
        assert response.error == error
    assert len(server.creates) == len(server.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
async def test_video_creation_envelope_from_live_verification_keeps_provider_task_id(use_async):
    server = SubmissionServer(
        httpx.Response(
            200, json={"id": "public-video-id", "task_id": "actual-provider-task", "object": "video", "status": ""}
        )
    )
    response = await generate(server, use_async)
    assert response.status == "queued"
    assert decode_video_id_with_provider(response.id)["video_id"] == "actual-provider-task"
    assert len(server.creates) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("exception", [httpx.ReadTimeout("lost response"), httpx.RemoteProtocolError("lost response")])
async def test_lost_submission_response_is_unknown_and_never_reposted(use_async, exception):
    server = SubmissionServer(exception)
    with pytest.raises(APIError) as caught:
        await generate(server, use_async)
    assert get_submission_outcome(caught.value) == "unknown"
    assert is_reexecution_blocked(caught.value)
    assert len(server.creates) == len(server.requests) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("task_field", ["task_id", "video_id", "id"])
async def test_http_rejection_with_task_id_preserves_accepted_provenance(use_async, task_field):
    server = SubmissionServer(httpx.Response(400, json={**privacy_error(), "data": {task_field: "accepted-task"}}))
    with pytest.raises(litellm.BadRequestError) as caught:
        await generate(server, use_async)
    assert get_submission_outcome(caught.value) == "accepted"
    assert get_provider_task_id(caught.value) == "accepted-task"
    assert is_reexecution_blocked(caught.value)
    assert len(server.creates) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("status", ["accepted", "unknown"])
async def test_error_with_possible_acceptance_is_not_safe_to_resubmit(use_async, status):
    server = SubmissionServer(httpx.Response(400, json={**privacy_error(), "data": {"status": status}}))
    with pytest.raises(APIError) as caught:
        await generate(server, use_async)
    assert get_submission_outcome(caught.value) == "unknown"
    assert is_reexecution_blocked(caught.value)
    assert len(server.creates) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize(
    "response", [httpx.Response(200, text="not json"), httpx.Response(200, json={"object": "video"})]
)
async def test_malformed_success_response_is_unknown_and_blocks_reexecution(use_async, response):
    server = SubmissionServer(response)
    with pytest.raises(litellm.BadGatewayError) as caught:
        await generate(server, use_async)
    assert get_submission_outcome(caught.value) == "unknown"
    assert is_reexecution_blocked(caught.value)
    assert len(server.creates) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("use_async", [False, True])
@pytest.mark.parametrize("model", ["toapis/grok-video-1.0", "toapis/wan3.0-video"])
async def test_other_models_do_not_enable_private_asset_review(use_async, model):
    server = SubmissionServer()
    response = await generate(server, use_async, model=model)
    assert response.status == "queued"
    assert "private_asset_review" not in server.creates[0]
    assert len(server.creates) == 1


def unsafe_response(failure):
    return {
        "transport": httpx.ReadTimeout("lost response"),
        "malformed": httpx.Response(200, json={}),
        "accepted": httpx.Response(400, json={**privacy_error(), "task_id": "accepted-task"}),
        "unknown": httpx.Response(400, json={**privacy_error(), "status": "unknown"}),
    }[failure]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["transport", "malformed", "accepted", "unknown"])
async def test_router_retry_deployment_order_and_fallback_do_not_repeat_unsafe_submission(monkeypatch, failure):
    from litellm.types.router import RetryPolicy

    server = SubmissionServer(unsafe_response(failure))
    deployments = []

    async def routed_generate(**kwargs):
        deployments.append(kwargs["api_key"])
        return await generate(server, True)

    monkeypatch.setattr("litellm.videos.avideo_generation", routed_generate)
    model_list = [
        {
            "model_name": group,
            "litellm_params": {"model": "toapis/seedance-2", "api_key": key, "weight": 1, "order": order},
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
        retry_policy=RetryPolicy(BadRequestErrorRetries=3, InternalServerErrorRetries=3),
        fallbacks=[{"video-model": ["backup-model"]}],
        enable_weighted_failover=True,
    )
    with pytest.raises(APIError) as caught:
        await router.avideo_generation(model="video-model", prompt="fictional character")
    assert is_reexecution_blocked(caught.value)
    assert get_submission_outcome(caught.value) == ("accepted" if failure == "accepted" else "unknown")
    assert deployments == ["primary"]
    assert len(server.creates) == 1


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["transport", "malformed", "accepted"])
async def test_unsafe_submission_in_fallback_chain_stops_remaining_targets(monkeypatch, failure):
    server = SubmissionServer(unsafe_response(failure))
    attempted = []

    async def call(**kwargs):
        attempted.append(kwargs["model"])
        if kwargs["model"] == "primary":
            raise mark_submission_outcome(
                litellm.BadRequestError(message="ordinary rejection", model="primary", llm_provider="toapis"),
                "rejected",
            )
        if kwargs["model"] == "seedance":
            return await generate(server, True)
        raise AssertionError("An uncertain or accepted submission must not reach the last fallback")

    router = litellm.Router(
        model_list=[
            {"model_name": model, "litellm_params": {"model": "toapis/seedance-2", "api_key": model}}
            for model in ("primary", "seedance", "last")
        ],
        num_retries=0,
        fallbacks=[{"primary": ["seedance", "last"]}],
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
    assert get_submission_outcome(caught.value) == ("accepted" if failure == "accepted" else "unknown")
    assert attempted == ["primary", "seedance"]
    assert len(server.creates) == 1
