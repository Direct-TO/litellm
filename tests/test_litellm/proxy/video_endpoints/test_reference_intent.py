import asyncio
import copy
import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest
from fastapi import Response

import litellm
from litellm.proxy._types import ConfigGeneralSettings, UserAPIKeyAuth
from litellm.proxy.common_request_processing import ProxyBaseLLMRequestProcessing
from litellm.proxy.video_endpoints.intent import prepare_video_intent
from litellm.router import Router
from litellm.types.router import RetryPolicy
from litellm.types.videos.intent import VIDEO_INTENT_METADATA_KEY

SETTINGS = {"video_reference_intent": {"classifier_model": "intent-chat"}}


def request():
    return {
        "model": "seedance-2-5",
        "prompt": "将视频1中的花朵改成黄色",
        "operation": "generate",
        "seconds": "4",
        "resolution": "480p",
        "aspect_ratio": "16:9",
        "references": [{"type": "video", "url": "https://media.example/video.mp4?private=secret"}],
        "metadata": {"user_api_key_user_id": "user-1"},
        "litellm_call_id": "parent-id",
    }


def decision(intent="edit", source=1, duration_kind="unspecified", duration=None):
    return {
        "intent": intent,
        "source_video_index": source,
        "duration_kind": duration_kind,
        "duration_seconds": duration,
    }


def classifier(result):
    content = result if isinstance(result, str) else json.dumps(result)
    return SimpleNamespace(
        get_model_list=lambda **kwargs: [object()],
        acompletion=AsyncMock(
            return_value=SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=content))])
        ),
    )


@pytest.mark.asyncio
async def test_edit_rewrites_only_required_parameters_and_records_both_versions():
    data = request()
    data["metadata"]["user_api_key"] = "a" * 64
    original = copy.deepcopy(data)
    gateway = classifier(decision())
    await prepare_video_intent(data, SETTINGS, gateway)
    assert data["model"] == original["model"]
    assert data["prompt"] == original["prompt"]
    assert data["references"] == original["references"]
    assert data["operation"] == "edit" and data["seconds"] == "-1"
    assert "aspect_ratio" not in data
    assert data["disable_fallbacks"] is True and data["num_retries"] == data["max_retries"] == 0
    audit = data["metadata"][VIDEO_INTENT_METADATA_KEY]
    assert audit["original"]["seconds"] == "4"
    assert audit["effective"]["seconds"] == "-1"
    assert audit["required_provider"] == "toapis"
    from litellm.litellm_core_utils.litellm_logging import get_standard_logging_metadata

    logged = get_standard_logging_metadata(data["metadata"])
    assert logged["spend_logs_metadata"][VIDEO_INTENT_METADATA_KEY]["effective"]["seconds"] == "-1"
    call = gateway.acompletion.call_args.kwargs
    assert call["metadata"]["parent_video_call_id"] == "parent-id"
    assert call["metadata"]["user_api_key_user_id"] == "user-1"
    assert call["metadata"]["user_api_key"] == "a" * 64
    assert "private=secret" not in json.dumps(call, default=str)


@pytest.mark.asyncio
@pytest.mark.parametrize("duration_kind,duration,expected", [("unspecified", None, "4"), ("additional", 6, "6")])
async def test_continuation_keeps_new_clip_duration_and_generate_mode(duration_kind, duration, expected):
    data = request()
    data["prompt"] = "从视频1结尾继续向后拍摄"
    await prepare_video_intent(
        data, SETTINGS, classifier(decision("extend", duration_kind=duration_kind, duration=duration))
    )
    assert data["operation"] == "generate"
    assert data["seconds"] == expected
    assert data["aspect_ratio"] == "16:9"
    assert data["metadata"][VIDEO_INTENT_METADATA_KEY]["required_provider"] == "toapis"


@pytest.mark.asyncio
async def test_generate_preserves_request_and_does_not_restrict_provider():
    data = request()
    original = copy.deepcopy(data)
    await prepare_video_intent(data, SETTINGS, classifier(decision("generate", source=None)))
    for key in ("prompt", "model", "operation", "seconds", "aspect_ratio", "resolution", "references"):
        assert data[key] == original[key]
    assert data["metadata"][VIDEO_INTENT_METADATA_KEY]["required_provider"] is None


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "change", [{"model": "other"}, {"references": []}, {"operation": "edit"}, {"operation": "extend"}]
)
async def test_unrelated_requests_bypass_classification(change):
    data = request()
    data.update(change)
    gateway = classifier(decision())
    await prepare_video_intent(data, SETTINGS, gateway)
    gateway.acompletion.assert_not_called()


@pytest.mark.asyncio
async def test_disabled_and_forged_decision_cannot_select_provider():
    data = request()
    data["metadata"][VIDEO_INTENT_METADATA_KEY] = {"intent": "edit", "required_provider": "toapis"}
    gateway = classifier(decision())
    await prepare_video_intent(data, {}, gateway)
    gateway.acompletion.assert_not_called()
    assert VIDEO_INTENT_METADATA_KEY not in data["metadata"]


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "result",
    [
        decision("unsupported"),
        decision("unclear"),
        decision(source=2),
        decision("extend", duration_kind="total", duration=12),
        decision("extend", duration_kind="additional", duration=31),
        decision("extend", duration_kind="additional"),
    ],
)
async def test_unexecutable_intent_stops_before_video_submission(result):
    with pytest.raises(litellm.BadRequestError):
        await prepare_video_intent(request(), SETTINGS, classifier(result))


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "content",
    ["not json", "{}", json.dumps({**decision(), "source_video_index": True}), json.dumps({**decision(), "extra": 1})],
)
async def test_invalid_classifier_response_fails_closed(content):
    with pytest.raises(litellm.InternalServerError, match="尚未提交"):
        await prepare_video_intent(request(), SETTINGS, classifier(content))


@pytest.mark.asyncio
async def test_classifier_deadline_stops_before_generation():
    gateway = classifier(decision())

    async def slow(**kwargs):
        await asyncio.sleep(1)

    gateway.acompletion.side_effect = slow
    with pytest.raises(litellm.InternalServerError, match="尚未提交"):
        await prepare_video_intent(
            request(), {"video_reference_intent": {"classifier_model": "other-chat", "timeout": 0.01}}, gateway
        )
    assert gateway.acompletion.call_count == 1


@pytest.mark.asyncio
async def test_model_is_configurable_and_missing_model_fails_closed():
    gateway = classifier(decision())
    await prepare_video_intent(request(), {"video_reference_intent": {"classifier_model": "another-chat"}}, gateway)
    assert gateway.acompletion.call_args.kwargs["model"] == "another-chat"
    gateway.get_model_list = lambda **kwargs: []
    with pytest.raises(litellm.InternalServerError, match="未配置"):
        await prepare_video_intent(request(), SETTINGS, gateway)


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "override",
    [
        {"api_base": "https://elsewhere.example"},
        {"extra_body": {"video_operation": "extend"}},
        {"video_with_roles": []},
    ],
)
async def test_conflicting_native_overrides_rejected_before_classifier(override):
    data = request()
    data.update(override)
    gateway = classifier(decision())
    with pytest.raises(litellm.BadRequestError):
        await prepare_video_intent(data, SETTINGS, gateway)
    gateway.acompletion.assert_not_called()


def make_router():
    return Router(
        model_list=[
            {
                "model_name": "seedance-2-5",
                "litellm_params": {
                    # An alternate reference-capable deployment tests selection
                    # without depending on optional, uncommitted provider code.
                    "model": "toapis/seedance-2",
                    "api_key": "fixture",
                    "api_base": "https://alternate.example",
                },
            },
            {
                "model_name": "seedance-2-5",
                "litellm_params": {
                    "model": "toapis/seedance-2-5",
                    "api_key": "fixture",
                    "api_base": "https://toapis.example",
                },
            },
            {
                "model_name": "intent-chat",
                "litellm_params": {
                    "model": "openai/fixture-chat",
                    "api_key": "fixture",
                    "api_base": "https://chat.example/v1",
                },
            },
        ],
        num_retries=2,
        retry_policy=RetryPolicy(RateLimitErrorRetries=3),
        enable_weighted_failover=True,
        fallbacks=[{"seedance-2-5": ["another-video-group"]}],
    )


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "intent,operation,duration,ratio", [("edit", "edit", -1, "adaptive"), ("extend", "generate", 4, "16:9")]
)
async def test_real_router_and_adapter_pin_special_intents_to_toapis(
    respx_mock, monkeypatch, intent, operation, duration, ratio
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    gateway = make_router()
    mock_classifier = classifier(decision(intent)).acompletion
    monkeypatch.setattr(gateway, "acompletion", mock_classifier)
    upstream = respx_mock.post("https://toapis.example/v1/videos/generations").respond(
        200, json={"id": "task-1", "object": "generation.task", "status": "queued"}
    )
    data = request()
    await prepare_video_intent(data, SETTINGS, gateway)
    result = await gateway.avideo_generation(**data)
    assert result.status == "queued"
    body = json.loads(upstream.calls[0].request.content)
    assert body["video_operation"] == operation
    assert body["duration"] == duration
    assert body["aspect_ratio"] == ratio
    assert body["prompt"] == data["prompt"]
    assert body["resolution"] == "480p"
    assert "metadata" not in body
    assert len(upstream.calls) == 1


@pytest.mark.asyncio
async def test_provider_failure_never_retries_or_falls_back_even_with_router_policies(respx_mock, monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    gateway = make_router()
    monkeypatch.setattr(gateway, "acompletion", classifier(decision("extend")).acompletion)
    upstream = respx_mock.post("https://toapis.example/v1/videos/generations").respond(
        429, json={"error": {"message": "busy"}}
    )
    data = request()
    await prepare_video_intent(data, SETTINGS, gateway)
    with pytest.raises(Exception):
        await gateway.avideo_generation(**data)
    assert len(upstream.calls) == 1


@pytest.mark.asyncio
async def test_ordinary_generation_can_use_both_providers_but_missing_toapis_stops_special_intent():
    gateway = make_router()
    data = request()
    await prepare_video_intent(data, SETTINGS, classifier(decision("generate", source=None)))
    kwargs = {**data, "_router_call_type": "avideo_generation"}
    candidates = gateway.get_model_list(model_name="seedance-2-5")
    assert len(gateway._filter_deployments_by_video_generation_params(data["model"], candidates, kwargs)) == 2
    await prepare_video_intent(data, SETTINGS, classifier(decision("extend")))
    kwargs = {**data, "_router_call_type": "avideo_generation"}
    with pytest.raises(litellm.BadRequestError, match="requires ToAPIs"):
        gateway._filter_deployments_by_video_generation_params(data["model"], candidates[:1], kwargs)


@pytest.mark.asyncio
async def test_common_processing_classifies_after_prechecks_before_route(monkeypatch):
    from litellm.proxy import common_request_processing as processing

    data = request()
    processor = ProxyBaseLLMRequestProcessing(data=data)
    order = []

    async def prechecks(**kwargs):
        order.append("prechecks")
        return data, MagicMock()

    async def prepare(*args):
        order.append("intent")

    async def route(**kwargs):
        order.append("route")
        raise RuntimeError("stop before submission")

    monkeypatch.setattr(processor, "_pre_call_with_fallbacks", prechecks)
    monkeypatch.setattr(processor, "_has_post_call_guardrails", lambda: False)
    monkeypatch.setattr("litellm.proxy.video_endpoints.intent.prepare_video_intent", prepare)
    monkeypatch.setattr(processing, "route_request", route)
    with pytest.raises(RuntimeError, match="stop before submission"):
        await processor._process_llm_request(
            request=MagicMock(),
            fastapi_response=Response(),
            user_api_key_dict=UserAPIKeyAuth(),
            route_type="avideo_generation",
            proxy_logging_obj=SimpleNamespace(during_call_hook=AsyncMock()),
            general_settings=SETTINGS,
            proxy_config=MagicMock(),
        )
    assert order == ["prechecks", "intent", "route"]


def test_general_settings_validate_classifier_configuration():
    settings = ConfigGeneralSettings(**SETTINGS)
    assert settings.video_reference_intent.classifier_model == "intent-chat"
    assert ConfigGeneralSettings().video_reference_intent is None


@pytest.mark.asyncio
async def test_private_avatar_recovery_does_not_resubmit_automatic_intent(respx_mock, monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    gateway = make_router()
    monkeypatch.setattr(gateway, "acompletion", classifier(decision()).acompletion)
    upstream = respx_mock.post("https://toapis.example/v1/videos/generations").respond(
        400,
        json={
            "error": {
                "code": "PrivacyInformation",
                "message": "input image 'content[1]' may contain real person",
            }
        },
    )
    data = request()
    data["references"].append({"type": "image", "url": "https://media.example/person.png"})
    await prepare_video_intent(data, SETTINGS, gateway)
    with pytest.raises(litellm.BadRequestError, match="PrivacyInformation"):
        await gateway.avideo_generation(**data)
    assert len(upstream.calls) == 1


@pytest.mark.asyncio
async def test_classifier_and_video_use_real_router_with_mocked_http(respx_mock, monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    chat = respx_mock.post("https://chat.example/v1/chat/completions").respond(
        200,
        json={
            "id": "chat-1",
            "object": "chat.completion",
            "created": 1,
            "model": "fixture-chat",
            "choices": [
                {
                    "index": 0,
                    "finish_reason": "stop",
                    "message": {"role": "assistant", "content": json.dumps(decision())},
                }
            ],
            "usage": {"prompt_tokens": 10, "completion_tokens": 10, "total_tokens": 20},
        },
    )
    video = respx_mock.post("https://toapis.example/v1/videos/generations").respond(
        200, json={"id": "task-1", "object": "generation.task", "status": "queued"}
    )
    gateway = make_router()
    data = request()
    await prepare_video_intent(data, SETTINGS, gateway)
    await gateway.avideo_generation(**data)
    assert len(chat.calls) == len(video.calls) == 1
    assert json.loads(video.calls[0].request.content)["duration"] == -1


@pytest.mark.asyncio
async def test_classifier_failure_does_not_use_global_retry_policy(respx_mock, monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    chat = respx_mock.post("https://chat.example/v1/chat/completions").respond(
        429, json={"error": {"message": "busy", "type": "rate_limit_error"}}
    )
    gateway = make_router()
    with pytest.raises(litellm.InternalServerError, match="尚未提交"):
        await prepare_video_intent(request(), SETTINGS, gateway)
    assert len(chat.calls) == 1


@pytest.mark.asyncio
async def test_multiple_reference_numbers_and_order_are_preserved():
    data = request()
    data["references"].insert(0, {"type": "image", "url": "https://media.example/image.png"})
    data["references"].append({"type": "video", "url": "https://media.example/second.mp4"})
    data["prompt"] = "修改视频2的颜色，参考图片1和视频1"
    refs = copy.deepcopy(data["references"])
    await prepare_video_intent(data, SETTINGS, classifier(decision(source=2)))
    assert data["references"] == refs
    assert data["metadata"][VIDEO_INTENT_METADATA_KEY]["source_video_index"] == 2
