import httpx
import pytest
from pydantic import ValidationError

import litellm
from litellm import Router
from litellm.litellm_core_utils.exception_mapping_utils import exception_type
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.submission_utils import (
    SubmissionOutcome,
    get_submission_outcome,
    mark_submission_outcome,
)
from litellm.llms.toapis.common_utils import parse_toapis_video_create_task
from litellm.types.router import UpdateRouterConfig, WeightedFailoverPolicy
from litellm.types.videos.main import VideoObject


def _model_list() -> list[dict]:
    return [
        {
            "model_name": "media-model",
            "litellm_params": {
                "model": "zexapi/image2",
                "api_key": "bad-key",
                "weight": 1,
                "num_retries": 0,
                "order": 1,
            },
            "model_info": {"id": "A", "supported_endpoints": ["/v1/images/generations"]},
        },
        {
            "model_name": "media-model",
            "litellm_params": {
                "model": "zexapi/gpt-image2",
                "api_key": "same-provider-key",
                "weight": 1,
                "num_retries": 0,
                "order": 2,
            },
            "model_info": {"id": "A2", "supported_endpoints": ["/v1/images/generations"]},
        },
        {
            "model_name": "media-model",
            "litellm_params": {
                "model": "toapis/gpt-image-2",
                "api_key": "good-key",
                "weight": 0,
                "num_retries": 0,
                "order": 2,
            },
            "model_info": {"id": "B", "supported_endpoints": ["/v1/images/generations"]},
        },
    ]


def _policy() -> WeightedFailoverPolicy:
    return WeightedFailoverPolicy(
        call_types=["aimage_generation", "avideo_generation"],
        status_codes=[403, 429, 503],
        submission_outcomes=["rejected"],
        failure_scope="provider",
    )


def _video_model_list() -> list[dict]:
    return [
        {
            "model_name": "video-model",
            "litellm_params": {
                "model": "zexapi/veo_3_1-fast",
                "api_key": "bad-video-key",
                "weight": 1,
                "num_retries": 0,
            },
            "model_info": {"id": "video-A", "supported_endpoints": ["/v1/videos"]},
        },
        {
            "model_name": "video-model",
            "litellm_params": {
                "model": "toapis/veo3.1-fast",
                "api_key": "good-video-key",
                "weight": 0,
                "num_retries": 0,
            },
            "model_info": {"id": "video-B", "supported_endpoints": ["/v1/videos"]},
        },
    ]


def test_weighted_failover_policy_round_trips_through_update_schema():
    config = UpdateRouterConfig(
        enable_weighted_failover=True,
        weighted_failover_policy=_policy(),
    )

    dumped = config.model_dump(exclude_none=True)
    assert dumped["enable_weighted_failover"] is True
    assert dumped["weighted_failover_policy"] == {
        "call_types": ["aimage_generation", "avideo_generation"],
        "status_codes": [403, 429, 503],
        "submission_outcomes": ["rejected"],
        "failure_scope": "provider",
    }


@pytest.mark.parametrize(
    "payload",
    (
        {"call_types": []},
        {"call_types": ["avideo_generation", "avideo_generation"]},
        {"status_codes": [200]},
        {"status_codes": [503, 503]},
        {"submission_outcomes": []},
        {"failure_scope": "region"},
        {"unexpected_filter": True},
    ),
)
def test_weighted_failover_policy_rejects_malformed_filters(payload: dict):
    with pytest.raises(ValidationError):
        WeightedFailoverPolicy(**payload)


def test_router_update_settings_accepts_weighted_failover_policy_dict():
    router = Router(model_list=_model_list())

    router.update_settings(
        enable_weighted_failover=True,
        weighted_failover_policy=_policy().model_dump(),
    )

    assert router.enable_weighted_failover is True
    assert router.weighted_failover_policy == _policy()
    assert router.get_settings()["weighted_failover_policy"] == _policy()


def test_status_queries_are_not_blocked_by_create_submission_outcome():
    router = Router(
        model_list=_model_list(),
        enable_weighted_failover=True,
        weighted_failover_policy=_policy(),
    )
    error = mark_submission_outcome(
        litellm.ServiceUnavailableError(
            message="status temporarily unavailable",
            llm_provider="toapis",
            model="media-model",
        ),
        "unknown",
    )

    assert (
        router._submission_outcome_blocks_reexecution(
            exception=error,
            kwargs={"_router_call_type": "avideo_status"},
        )
        is False
    )


def test_router_rejects_fail_open_weighted_failover_settings():
    with pytest.raises(TypeError, match="enable_weighted_failover must be a bool"):
        Router(model_list=_model_list(), enable_weighted_failover="false")  # type: ignore[arg-type]
    with pytest.raises(TypeError, match="weighted_failover_policy must be a mapping"):
        Router(model_list=_model_list(), weighted_failover_policy=["bad"])  # type: ignore[arg-type]

    router = Router(model_list=_model_list())
    with pytest.raises(TypeError, match="enable_weighted_failover must be a bool"):
        router.update_settings(enable_weighted_failover="false")
    with pytest.raises(TypeError, match="weighted_failover_policy must be a mapping"):
        router.update_settings(weighted_failover_policy=["bad"])


@pytest.mark.asyncio
@pytest.mark.parametrize(
    ("status_code", "outcome", "call_type", "should_fail_over"),
    (
        (403, "rejected", "avideo_generation", True),
        (429, "rejected", "aimage_generation", True),
        (503, "rejected", "avideo_generation", True),
        (400, "rejected", "avideo_generation", False),
        (502, "unknown", "avideo_generation", False),
        (503, "accepted", "aimage_generation", False),
        (503, "unknown", "aimage_generation", False),
        (503, "rejected", "avideo_status", False),
        (503, None, "avideo_generation", False),
    ),
)
async def test_weighted_failover_policy_requires_every_filter(
    monkeypatch,
    status_code: int,
    outcome: SubmissionOutcome | None,
    call_type: str,
    should_fail_over: bool,
):
    router = Router(
        model_list=_model_list(),
        routing_strategy="simple-shuffle",
        enable_weighted_failover=True,
        weighted_failover_policy=_policy(),
    )
    calls: list[dict] = []

    async def fake_run_async_fallback(*args, **kwargs):
        calls.append(kwargs)
        return "fallback-ok"

    monkeypatch.setattr("litellm.router.run_async_fallback", fake_run_async_fallback)
    error = litellm.APIError(
        status_code=status_code,
        message="provider failure",
        llm_provider="test-provider",
        model="media-model",
    )
    if outcome is not None:
        mark_submission_outcome(error, outcome)
    error.failed_deployment_id = "A"

    result = await router._maybe_run_weighted_failover(
        exception=error,
        original_model_group="media-model",
        all_deployments=_model_list(),
        args=(),
        kwargs={"metadata": {}, "_router_call_type": call_type},
        input_kwargs={},
    )

    assert result == ("fallback-ok" if should_fail_over else None)
    assert len(calls) == int(should_fail_over)
    if should_fail_over:
        excluded_ids = calls[0]["fallback_model_group"][0]["_excluded_deployment_ids"]
        assert set(excluded_ids) == {"A", "A2"}


@pytest.mark.asyncio
async def test_provider_scope_uses_static_provider_when_failed_id_is_dynamic(monkeypatch):
    router = Router(
        model_list=_model_list(),
        routing_strategy="simple-shuffle",
        enable_weighted_failover=True,
        weighted_failover_policy=_policy(),
    )
    calls: list[dict] = []

    async def fake_run_async_fallback(*args, **kwargs):
        calls.append(kwargs)
        return "fallback-ok"

    monkeypatch.setattr("litellm.router.run_async_fallback", fake_run_async_fallback)
    error = mark_submission_outcome(
        litellm.ServiceUnavailableError(
            message="no available channel",
            llm_provider="zexapi",
            model="media-model",
        ),
        "rejected",
    )
    router._stamp_failed_deployment_id_with_effective_model_info(
        error,
        _model_list()[0],
        {"model_info": {"id": "dynamic-A"}},
    )

    result = await router._maybe_run_weighted_failover(
        exception=error,
        original_model_group="media-model",
        all_deployments=_model_list(),
        args=(),
        kwargs={"metadata": {}, "_router_call_type": "aimage_generation"},
        input_kwargs={},
    )

    assert result == "fallback-ok"
    assert error.failed_deployment_id == "dynamic-A"
    excluded_ids = calls[0]["fallback_model_group"][0]["_excluded_deployment_ids"]
    assert set(excluded_ids) == {"dynamic-A", "A", "A2"}


@pytest.mark.asyncio
async def test_deployment_scope_excludes_static_deployment_when_failed_id_is_dynamic(monkeypatch):
    policy = _policy().model_copy(update={"failure_scope": "deployment"})
    router = Router(
        model_list=_model_list(),
        routing_strategy="simple-shuffle",
        enable_weighted_failover=True,
        weighted_failover_policy=policy,
    )
    calls: list[dict] = []

    async def fake_run_async_fallback(*args, **kwargs):
        calls.append(kwargs)
        return "fallback-ok"

    monkeypatch.setattr("litellm.router.run_async_fallback", fake_run_async_fallback)
    error = mark_submission_outcome(
        litellm.ServiceUnavailableError(
            message="no available channel",
            llm_provider="zexapi",
            model="media-model",
        ),
        "rejected",
    )
    router._stamp_failed_deployment_id_with_effective_model_info(
        error,
        _model_list()[0],
        {"model_info": {"id": "dynamic-A"}},
    )

    result = await router._maybe_run_weighted_failover(
        exception=error,
        original_model_group="media-model",
        all_deployments=_model_list(),
        args=(),
        kwargs={"metadata": {}, "_router_call_type": "aimage_generation"},
        input_kwargs={},
    )

    assert result == "fallback-ok"
    excluded_ids = calls[0]["fallback_model_group"][0]["_excluded_deployment_ids"]
    assert set(excluded_ids) == {"dynamic-A", "A"}


def test_provider_submission_metadata_survives_exception_mapping():
    raw_error = mark_submission_outcome(
        BaseLLMException(status_code=503, message="no available channel"),
        "rejected",
    )

    with pytest.raises(litellm.ServiceUnavailableError) as exc_info:
        exception_type(
            model="media-model",
            original_exception=raw_error,
            custom_llm_provider="toapis",
        )

    assert get_submission_outcome(exc_info.value) == "rejected"


def test_toapis_video_create_distinguishes_rejection_from_uncertain_acceptance():
    with pytest.raises(BaseLLMException) as rejected:
        parse_toapis_video_create_task(httpx.Response(503, text="no available channel"))
    with pytest.raises(BaseLLMException) as unknown:
        parse_toapis_video_create_task(
            httpx.Response(200, json={"object": "video", "status": "", "model": "seedance-2-5"})
        )
    with pytest.raises(BaseLLMException) as ambiguous_503:
        parse_toapis_video_create_task(httpx.Response(503, text="service temporarily unavailable"))

    assert get_submission_outcome(rejected.value) == "rejected"
    assert get_submission_outcome(unknown.value) == "unknown"
    assert get_submission_outcome(ambiguous_503.value) == "unknown"


@pytest.mark.asyncio
async def test_async_image_generation_fails_over_only_after_explicit_rejection(monkeypatch):
    attempted_keys: list[str] = []

    async def fake_image_generation(**kwargs):
        attempted_keys.append(kwargs["api_key"])
        if kwargs["api_key"] == "bad-key":
            raise mark_submission_outcome(
                litellm.ServiceUnavailableError(
                    message="no available channel",
                    llm_provider="test-provider",
                    model="media-model",
                ),
                "rejected",
            )
        return litellm.ImageResponse(data=[])

    monkeypatch.setattr(litellm, "aimage_generation", fake_image_generation)
    router = Router(
        model_list=_model_list(),
        routing_strategy="simple-shuffle",
        num_retries=2,
        enable_weighted_failover=True,
        weighted_failover_policy=_policy(),
    )

    await router.aimage_generation(model="media-model", prompt="poster")

    assert attempted_keys == ["bad-key", "good-key"]


@pytest.mark.asyncio
@pytest.mark.parametrize("status_code", (403, 429))
async def test_async_video_generation_fails_over_across_providers(monkeypatch, status_code: int):
    attempted_keys: list[str] = []

    async def fake_video_generation(**kwargs):
        attempted_keys.append(kwargs["api_key"])
        if kwargs["api_key"] == "bad-video-key":
            raise mark_submission_outcome(
                litellm.APIError(
                    status_code=status_code,
                    message="provider rejected create",
                    llm_provider="zexapi",
                    model="video-model",
                ),
                "rejected",
            )
        return VideoObject(id="video-ok", object="video", status="queued")

    monkeypatch.setattr("litellm.videos.avideo_generation", fake_video_generation)
    router = Router(
        model_list=_video_model_list(),
        routing_strategy="simple-shuffle",
        num_retries=2,
        enable_weighted_failover=True,
        weighted_failover_policy=_policy(),
    )

    response = await router.avideo_generation(model="video-model", prompt="city")

    assert response.id == "video-ok"
    assert attempted_keys == ["bad-video-key", "good-video-key"]


@pytest.mark.asyncio
async def test_single_provider_pool_returns_original_rejection_without_resubmitting(monkeypatch):
    attempted_keys: list[str] = []
    original_error = mark_submission_outcome(
        litellm.ServiceUnavailableError(
            message="no available channel",
            llm_provider="zexapi",
            model="media-model",
        ),
        "rejected",
    )

    async def fake_image_generation(**kwargs):
        attempted_keys.append(kwargs["api_key"])
        raise original_error

    monkeypatch.setattr(litellm, "aimage_generation", fake_image_generation)
    router = Router(
        model_list=_model_list()[:2],
        routing_strategy="simple-shuffle",
        num_retries=2,
        enable_weighted_failover=True,
        weighted_failover_policy=_policy(),
    )

    with pytest.raises(litellm.ServiceUnavailableError) as exc_info:
        await router.aimage_generation(model="media-model", prompt="poster")

    assert exc_info.value is original_error
    assert attempted_keys == ["bad-key"]


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ("accepted", "unknown", None))
async def test_async_image_generation_never_reexecutes_uncertain_submission(
    monkeypatch, outcome: SubmissionOutcome | None
):
    attempted_keys: list[str] = []

    async def fake_image_generation(**kwargs):
        attempted_keys.append(kwargs["api_key"])
        error = litellm.ServiceUnavailableError(
            message="poll failed",
            llm_provider="zexapi",
            model="media-model",
        )
        if outcome is not None:
            mark_submission_outcome(
                error,
                outcome,
                provider_task_id="task-accepted" if outcome == "accepted" else None,
            )
        raise error

    monkeypatch.setattr(litellm, "aimage_generation", fake_image_generation)
    backup = {
        "model_name": "backup-model",
        "litellm_params": {"model": "toapis/gpt-image-2", "api_key": "backup-key"},
        "model_info": {"id": "backup", "supported_endpoints": ["/v1/images/generations"]},
    }
    router = Router(
        model_list=[*_model_list(), backup],
        routing_strategy="simple-shuffle",
        num_retries=2,
        fallbacks=[{"media-model": ["backup-model"]}],
        enable_weighted_failover=True,
        weighted_failover_policy=_policy(),
    )

    with pytest.raises(litellm.ServiceUnavailableError):
        await router.aimage_generation(model="media-model", prompt="poster")

    assert attempted_keys == ["bad-key"]


@pytest.mark.asyncio
@pytest.mark.parametrize("second_outcome", ("accepted", "unknown", None))
async def test_weighted_second_attempt_uncertainty_stops_before_cross_group_fallback(
    monkeypatch,
    second_outcome: SubmissionOutcome | None,
):
    attempted_keys: list[str] = []
    second_error = litellm.ServiceUnavailableError(
        message="second provider result is uncertain",
        llm_provider="toapis",
        model="media-model",
    )
    if second_outcome is not None:
        mark_submission_outcome(
            second_error,
            second_outcome,
            provider_task_id="second-task" if second_outcome == "accepted" else None,
        )

    async def fake_image_generation(**kwargs):
        attempted_keys.append(kwargs["api_key"])
        if kwargs["api_key"] == "bad-key":
            raise mark_submission_outcome(
                litellm.ServiceUnavailableError(
                    message="no available channel",
                    llm_provider="zexapi",
                    model="media-model",
                ),
                "rejected",
            )
        if kwargs["api_key"] == "good-key":
            raise second_error
        return litellm.ImageResponse(data=[])

    monkeypatch.setattr(litellm, "aimage_generation", fake_image_generation)
    backup = {
        "model_name": "backup-model",
        "litellm_params": {"model": "openai/gpt-image-1", "api_key": "backup-key"},
        "model_info": {"id": "backup", "supported_endpoints": ["/v1/images/generations"]},
    }
    router = Router(
        model_list=[*_model_list(), backup],
        routing_strategy="simple-shuffle",
        num_retries=2,
        fallbacks=[{"media-model": ["backup-model"]}],
        enable_weighted_failover=True,
        weighted_failover_policy=_policy(),
    )

    with pytest.raises(litellm.ServiceUnavailableError) as exc_info:
        await router.aimage_generation(model="media-model", prompt="poster")

    assert exc_info.value is second_error
    assert attempted_keys == ["bad-key", "good-key"]
