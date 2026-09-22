"""Exercise the configured primary/backup pools without paid provider requests."""

import copy
import json
from pathlib import Path

import httpx
import pytest
import yaml

import litellm
from litellm.router import Router
from litellm.types.videos.utils import decode_video_id_with_provider, encode_video_id_with_provider

GROUPS = ("seedance-2", "seedance-2-5", "seedance-2-fast", "seedance-2-mini")
ROOT = Path(__file__).resolve().parents[4]
CONFIG = yaml.safe_load((ROOT / "litellm/proxy/dev_config.yaml").read_text(encoding="utf-8"))
GUANGHE = "https://guanghe.example/api/inspiration-waterfall/v1"
TOAPIS = "https://toapis.example/v1"


@pytest.fixture(autouse=True)
def isolate_transport(monkeypatch, respx_mock):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(
        "litellm.llms.guanghe.videos.reference_upload._validated_url", lambda url: (url, httpx.URL(url).host)
    )
    respx_mock.get("https://media.example/person.jpg").respond(
        200, content=b"reference", headers={"Content-Type": "image/jpeg"}
    )
    respx_mock.post(GUANGHE + "/files/upload").respond(
        200, json={"success": True, "data": {"url": "https://media.example/person.jpg"}}
    )


def gateway(group, paused_primary=False):
    models = copy.deepcopy([row for row in CONFIG["model_list"] if row["model_name"] == group])
    assert len(models) == 2
    for row in models:
        provider = row["litellm_params"]["model"].split("/", 1)[0]
        row["litellm_params"].update(
            {"api_key": provider + "-key", "api_base": GUANGHE if provider == "guanghe" else TOAPIS}
        )
        row["model_info"]["id"] = group + "-" + provider
        if provider == "guanghe" and paused_primary:
            row["model_info"]["blocked"] = True
    # Config file position must not override the explicit order.
    return Router(model_list=list(reversed(models)), num_retries=0, **CONFIG["router_settings"])


def request(group, **overrides):
    return {
        "model": group,
        "prompt": "fly",
        "seconds": "4",
        "resolution": "480p",
        "aspect_ratio": "3:4",
        "references": [{"type": "image", "url": "https://media.example/person.jpg"}],
        **overrides,
    }


def provider_routes(respx_mock, primary_status=202, primary_body=None):
    primary = respx_mock.post(GUANGHE + "/video/tasks").respond(
        primary_status, json=primary_body or {"success": True, "data": {"job_id": "gh-one", "status": "accepting"}}
    )
    backup = respx_mock.post(TOAPIS + "/videos/generations").respond(
        200, json={"id": "ta-one", "object": "generation.task", "status": "queued"}
    )
    return primary, backup


@pytest.mark.parametrize("group", GROUPS)
def test_config_has_two_ordered_providers_per_seedance_group(group):
    rows = [row for row in CONFIG["model_list"] if row["model_name"] == group]
    assert [(row["litellm_params"]["model"].split("/", 1)[0], row["litellm_params"]["order"]) for row in rows] == [
        ("guanghe", 1),
        ("toapis", 2),
    ]
    assert not any(row["model_name"].startswith("guanghe-seedance-") for row in CONFIG["model_list"])


@pytest.mark.parametrize("group", GROUPS)
async def test_compatible_request_prefers_guanghe(group, respx_mock):
    primary, backup = provider_routes(respx_mock)
    result = await gateway(group).avideo_generation(**request(group))
    assert decode_video_id_with_provider(result.id)["custom_llm_provider"] == "guanghe"
    assert primary.call_count == 1
    assert backup.call_count == 0


@pytest.mark.parametrize("group", GROUPS)
async def test_platform_default_audio_request_still_prefers_guanghe(group, respx_mock):
    primary, backup = provider_routes(respx_mock)
    result = await gateway(group).avideo_generation(**request(group, generate_audio=True))
    assert decode_video_id_with_provider(result.id)["custom_llm_provider"] == "guanghe"
    assert primary.call_count == 1
    assert backup.call_count == 0
    assert "generate_audio" not in json.loads(primary.calls[0].request.content)["params"]


@pytest.mark.parametrize("group", GROUPS)
async def test_explicit_mute_selects_provider_that_supports_it(group, respx_mock):
    primary, backup = provider_routes(respx_mock)
    result = await gateway(group).avideo_generation(**request(group, generate_audio=False))
    assert decode_video_id_with_provider(result.id)["custom_llm_provider"] == "toapis"
    assert primary.call_count == 0
    assert backup.call_count == 1
    assert json.loads(backup.calls[0].request.content)["generate_audio"] is False


@pytest.mark.parametrize("group", GROUPS)
@pytest.mark.parametrize("status", [403, 429])
async def test_explicit_rejection_uses_toapis_backup(group, status, respx_mock):
    primary, backup = provider_routes(
        respx_mock,
        status,
        {
            "success": False,
            "code": "model_not_allowed" if status == 403 else "rate_limit_exceeded",
            "message": "create rejected",
        },
    )
    result = await gateway(group).avideo_generation(**request(group))
    assert decode_video_id_with_provider(result.id)["custom_llm_provider"] == "toapis"
    assert primary.call_count == backup.call_count == 1


@pytest.mark.parametrize("group", GROUPS)
@pytest.mark.parametrize("kind", ["timeout", "unknown_503", "bad_envelope", "content_rejection"])
async def test_uncertain_or_content_failure_never_creates_backup(group, kind, respx_mock):
    primary, backup = provider_routes(respx_mock)
    if kind == "timeout":
        primary.mock(side_effect=httpx.ReadTimeout("submission result unknown"))
    elif kind == "bad_envelope":
        primary.respond(202, json={"success": True, "data": {"status": "accepting"}})
    else:
        primary.respond(
            503 if kind == "unknown_503" else 400,
            json={
                "success": False,
                "code": "upstream_error" if kind == "unknown_503" else "invalid_request",
                "message": "unknown upstream status" if kind == "unknown_503" else "content policy rejected",
            },
        )
    with pytest.raises(Exception):
        await gateway(group).avideo_generation(**request(group))
    assert primary.call_count == 1
    assert backup.call_count == 0


@pytest.mark.parametrize("group", GROUPS)
async def test_paused_primary_uses_backup(group, respx_mock):
    primary, backup = provider_routes(respx_mock)
    result = await gateway(group, paused_primary=True).avideo_generation(**request(group))
    assert decode_video_id_with_provider(result.id)["custom_llm_provider"] == "toapis"
    assert primary.call_count == 0
    assert backup.call_count == 1


async def test_unsupported_primary_parameters_select_compatible_backup(respx_mock):
    primary, backup = provider_routes(respx_mock)
    result = await gateway("seedance-2").avideo_generation(**request("seedance-2", resolution="4K"))
    assert decode_video_id_with_provider(result.id)["custom_llm_provider"] == "toapis"
    assert primary.call_count == 0
    assert backup.call_count == 1


@pytest.mark.parametrize("group", GROUPS)
@pytest.mark.parametrize("provider", ["guanghe", "toapis"])
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_status_uses_task_provider_even_with_shared_model_name(group, provider, asynchronous, respx_mock):
    router = gateway(group)
    row = next(row for row in router.model_list if row["litellm_params"]["model"].startswith(provider + "/"))
    provider_model = row["litellm_params"]["model"].split("/", 1)[1]
    video_id = encode_video_id_with_provider("task-original", provider, provider_model)
    primary = respx_mock.get(GUANGHE + "/video/tasks/task-original").respond(
        200, json={"success": True, "data": {"task_id": "task-original", "status": "processing"}}
    )
    backup = respx_mock.get(TOAPIS + "/videos/generations/task-original").respond(
        200, json={"id": "task-original", "object": "generation.task", "status": "in_progress"}
    )
    if provider == "guanghe":
        route = respx_mock.get(GUANGHE + "/video/tasks/task-original").respond(
            200,
            json={
                "success": True,
                "data": {"task_id": "task-original", "model_id": provider_model, "status": "processing"},
            },
        )
    else:
        route = respx_mock.get(TOAPIS + "/videos/generations/task-original").respond(
            200,
            json={"id": "task-original", "object": "generation.task", "model": provider_model, "status": "in_progress"},
        )
    kwargs = {"model": group, "video_id": video_id, "custom_llm_provider": provider}
    result = await router.avideo_status(**kwargs) if asynchronous else router.video_status(**kwargs)
    assert result.status == "in_progress"
    assert route.call_count == 1
    assert route.calls[0].request.headers["authorization"] == "Bearer " + provider + "-key"
    assert (backup if provider == "guanghe" else primary).call_count == 0


@pytest.mark.parametrize("provider", ["guanghe", "toapis"])
@pytest.mark.parametrize("asynchronous", [False, True])
async def test_content_stays_with_original_provider(provider, asynchronous, respx_mock):
    group = "seedance-2-5"
    router = gateway(group)
    row = next(row for row in router.model_list if row["litellm_params"]["model"].startswith(provider + "/"))
    model = row["litellm_params"]["model"].split("/", 1)[1]
    video_id = encode_video_id_with_provider("task-original", provider, model)
    primary = respx_mock.get(GUANGHE + "/video/tasks/task-original/download").respond(
        200, content=b"guanghe-video", headers={"Content-Type": "video/mp4"}
    )
    backup = respx_mock.get(TOAPIS + "/videos/generations/task-original/content").respond(
        200, content=b"toapis-video", headers={"Content-Type": "video/mp4"}
    )
    kwargs = {"model": group, "video_id": video_id, "custom_llm_provider": provider}
    content = await router.avideo_content(**kwargs) if asynchronous else router.video_content(**kwargs)
    expected, other = (primary, backup) if provider == "guanghe" else (backup, primary)
    assert content == (provider + "-video").encode()
    assert expected.call_count == 1
    assert other.call_count == 0
    assert expected.calls[0].request.headers["authorization"] == "Bearer " + provider + "-key"


@pytest.mark.parametrize("status", [400, 401, 402, 422])
async def test_order_cannot_bypass_media_failure_policy(status, respx_mock):
    primary, backup = provider_routes(
        respx_mock,
        status,
        {"success": False, "code": "invalid_request", "message": "explicit rejection outside failover policy"},
    )
    with pytest.raises(Exception):
        await gateway("seedance-2-5").avideo_generation(**request("seedance-2-5"))
    assert primary.call_count == 1
    assert backup.call_count == 0


@pytest.mark.parametrize("encoded_model", ["seedance-2-5", "toapis/seedance-2-5", "seedance-2-5-toapis", ""])
async def test_legacy_task_model_identifiers_remain_readable(encoded_model, respx_mock):
    router = gateway("seedance-2-5")
    video_id = encode_video_id_with_provider("legacy", "toapis", encoded_model)
    backup = respx_mock.get(TOAPIS + "/videos/generations/legacy").respond(
        200, json={"id": "legacy", "object": "generation.task", "status": "in_progress"}
    )
    result = await router.avideo_status(model="seedance-2-5", video_id=video_id)
    assert result.status == "in_progress"
    assert backup.call_count == 1


async def test_missing_original_provider_never_queries_another_provider(respx_mock):
    router = gateway("seedance-2-5")
    video_id = encode_video_id_with_provider("foreign-task", "zexapi", "seedance-2-5")
    with pytest.raises(litellm.NotFoundError):
        await router.avideo_status(model="seedance-2-5", video_id=video_id)
    assert len(respx_mock.calls) == 0


@pytest.mark.parametrize("provider", ["guanghe", "toapis"])
async def test_task_read_failure_does_not_fall_back_to_other_provider(provider, respx_mock):
    router = gateway("seedance-2-5")
    row = next(row for row in router.model_list if row["litellm_params"]["model"].startswith(provider + "/"))
    model = row["litellm_params"]["model"].split("/", 1)[1]
    video_id = encode_video_id_with_provider("missing", provider, model)
    primary = respx_mock.get(GUANGHE + "/video/tasks/missing").respond(
        200, json={"success": True, "data": {"task_id": "missing", "status": "processing"}}
    )
    backup = respx_mock.get(TOAPIS + "/videos/generations/missing").respond(
        200, json={"id": "missing", "object": "generation.task", "status": "in_progress"}
    )
    original, other = (primary, backup) if provider == "guanghe" else (backup, primary)
    original.respond(404, json={"success": False, "code": "task_not_found", "message": "Task not found"})
    with pytest.raises(Exception):
        await router.avideo_status(model="seedance-2-5", video_id=video_id)
    assert original.call_count >= 1
    assert other.call_count == 0
