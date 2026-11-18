"""Keep retired Fast tasks readable while new work uses the replacement channel."""

import json

import pytest

import litellm
from litellm.llms.guanghe.videos.transformation import FAST_KEY100_MODEL
from litellm.router import Router
from litellm.types.router import RouterRateLimitError
from litellm.types.videos.utils import decode_video_id_with_provider, encode_video_id_with_provider

GROUP = "seedance-2-fast"
LEGACY_MODEL = "seedance2.0-fast（企业折扣）"
LEGACY = "https://legacy-guanghe.example/api/inspiration-waterfall/v1"
CURRENT = "https://key100-guanghe.example/api/inspiration-waterfall/v1"
BACKUP = "https://backup-toapis.example/v1"


@pytest.fixture(autouse=True)
def isolate_transport(monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)


def gateway(legacy_only=False, legacy_priority=99):
    models = [
        {
            "model_name": GROUP,
            "litellm_params": {
                "model": "guanghe/" + LEGACY_MODEL,
                "api_base": LEGACY,
                "api_key": "legacy-key",
                "order": legacy_priority,
                "num_retries": 0,
            },
            "model_info": {
                "id": "legacy-fast",
                "mode": "video_generation",
                "supported_endpoints": [],
                "disable_background_health_check": True,
                "blocked": False,
            },
        }
    ]
    if not legacy_only:
        for model, base, key, priority in (
            ("guanghe/" + FAST_KEY100_MODEL, CURRENT, "current-key", 1),
            ("toapis/seedance-2-fast", BACKUP, "backup-key", 2),
        ):
            models.append(
                {
                    "model_name": GROUP,
                    "litellm_params": {
                        "model": model,
                        "api_base": base,
                        "api_key": key,
                        "order": priority,
                        "num_retries": 0,
                    },
                    "model_info": {
                        "id": key,
                        "mode": "video_generation",
                        "supported_endpoints": ["/v1/videos"],
                    },
                }
            )
    return Router(
        model_list=models,
        num_retries=0,
        enable_weighted_failover=True,
        weighted_failover_policy={
            "call_types": ["avideo_generation"],
            "status_codes": [403, 429, 503],
            "submission_outcomes": ["rejected"],
            "failure_scope": "provider",
        },
    )


def create_request():
    return {
        "model": GROUP,
        "prompt": "A red paper boat drifts gently on water.",
        "seconds": "5",
        "resolution": "480p",
        "aspect_ratio": "16:9",
        "generate_audio": True,
    }


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_retired_deployment_never_receives_new_generation(asynchronous, respx_mock):
    current = respx_mock.post(CURRENT + "/video/tasks").respond(
        202, json={"success": True, "data": {"job_id": "new-task", "status": "accepting"}}
    )
    router = gateway()
    result = (
        await router.avideo_generation(**create_request())
        if asynchronous
        else router.video_generation(**create_request())
    )
    assert decode_video_id_with_provider(result.id)["model_id"] == FAST_KEY100_MODEL
    assert current.call_count == 1
    assert current.calls[0].request.headers["authorization"] == "Bearer current-key"
    assert json.loads(current.calls[0].request.content)["model_id"] == FAST_KEY100_MODEL
    assert len(respx_mock.calls) == 1


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("operation", ["status", "content"])
async def test_retired_deployment_keeps_old_task_reads(asynchronous, operation, respx_mock):
    task_id = encode_video_id_with_provider("old-task", "guanghe", LEGACY_MODEL)
    suffix = "/download" if operation == "content" else ""
    route = respx_mock.get(LEGACY + "/video/tasks/old-task" + suffix)
    if operation == "content":
        route.respond(200, content=b"legacy-video", headers={"Content-Type": "video/mp4"})
    else:
        route.respond(
            200,
            json={
                "success": True,
                "data": {"task_id": "old-task", "model_id": LEGACY_MODEL, "status": "processing"},
            },
        )
    router = gateway()
    method = getattr(router, ("a" if asynchronous else "") + "video_" + operation)
    args = {"model": GROUP, "video_id": task_id}
    result = await method(**args) if asynchronous else method(**args)
    assert result == b"legacy-video" if operation == "content" else result.status == "in_progress"
    assert route.call_count == 1
    assert route.calls[0].request.headers["authorization"] == "Bearer legacy-key"
    assert len(respx_mock.calls) == 1


async def test_rejected_new_channel_falls_back_without_retired_create(respx_mock):
    current = respx_mock.post(CURRENT + "/video/tasks").respond(
        403, json={"success": False, "code": "model_not_allowed", "message": "create rejected"}
    )
    backup = respx_mock.post(BACKUP + "/videos/generations").respond(
        200, json={"id": "backup-task", "object": "generation.task", "status": "queued"}
    )
    result = await gateway().avideo_generation(**create_request())
    assert decode_video_id_with_provider(result.id)["custom_llm_provider"] == "toapis"
    assert current.call_count == backup.call_count == 1
    assert len(respx_mock.calls) == 2


@pytest.mark.parametrize("asynchronous", [False, True])
@pytest.mark.parametrize("operation", ["status", "content"])
async def test_missing_old_task_never_switches_channel(asynchronous, operation, respx_mock):
    suffix = "/download" if operation == "content" else ""
    route = respx_mock.get(LEGACY + "/video/tasks/missing" + suffix).respond(
        404, json={"success": False, "code": "task_not_found", "message": "Task not found"}
    )
    router = gateway()
    method = getattr(router, ("a" if asynchronous else "") + "video_" + operation)
    args = {"model": GROUP, "video_id": encode_video_id_with_provider("missing", "guanghe", LEGACY_MODEL)}
    # The async router can surface cooldown exhaustion after the original 404;
    # either way the task must remain pinned to its original deployment.
    if asynchronous:
        with pytest.raises((litellm.NotFoundError, RouterRateLimitError)):
            await method(**args)
    else:
        with pytest.raises(litellm.NotFoundError):
            method(**args)
    assert route.call_count >= 1
    assert all(call.request.url.host == "legacy-guanghe.example" for call in respx_mock.calls)


@pytest.mark.parametrize("asynchronous", [False, True])
async def test_retired_only_pool_rejects_generation_before_network(asynchronous, respx_mock):
    router = gateway(legacy_only=True)
    if asynchronous:
        with pytest.raises(litellm.BadRequestError, match="no deployment supporting endpoint"):
            await router.avideo_generation(**create_request())
    else:
        with pytest.raises(litellm.UnsupportedParamsError, match="retired for new generation"):
            router.video_generation(**create_request())
    assert len(respx_mock.calls) == 0


async def test_misprioritized_retired_deployment_is_filtered_for_async_create(respx_mock):
    current = respx_mock.post(CURRENT + "/video/tasks").respond(
        202, json={"success": True, "data": {"job_id": "new-task", "status": "accepting"}}
    )
    result = await gateway(legacy_priority=0).avideo_generation(**create_request())
    assert decode_video_id_with_provider(result.id)["model_id"] == FAST_KEY100_MODEL
    assert current.call_count == len(respx_mock.calls) == 1


def test_misprioritized_retired_deployment_fails_closed_for_sync_create(respx_mock):
    # Sync generic dispatch does not apply endpoint capability filtering. The
    # retired-model guard still prevents sending an invalid upstream request.
    router = gateway(legacy_priority=0)
    with pytest.raises(litellm.UnsupportedParamsError, match="retired for new generation"):
        router.video_generation(**create_request())
    assert len(respx_mock.calls) == 0
