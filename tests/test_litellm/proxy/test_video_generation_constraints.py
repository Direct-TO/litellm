"""The public catalogue and router preflight must agree before a supplier is called."""

from unittest.mock import AsyncMock

import pytest
from pydantic import ValidationError

from litellm import BadRequestError, Router
from litellm.proxy import proxy_server, utils
from litellm.proxy._types import UserAPIKeyAuth
from litellm.types.router import Deployment
from litellm.types.videos.constraints import VideoGenerationLimits
from litellm.videos.constraints import deployment_video_variants

OMNI = "toapis/gemini-omni-1.2-flash-preview"


def deployment(model, name="video", **info):
    return {
        "model_name": name,
        "litellm_params": {"model": model, "api_key": "offline-fixture"},
        "model_info": {"mode": "video_generation", "supported_endpoints": ["/v1/videos"], **info},
    }


@pytest.mark.asyncio
async def test_model_list_exposes_only_routable_combinations(monkeypatch):
    # Deliberately serve a single public model through disjoint deployments.
    rows = [
        deployment(OMNI, id="portrait", video_generation_limits={"aspect_ratios": ["9:16"], "max_seconds": 8}),
        deployment(OMNI, id="landscape", video_generation_limits={"aspect_ratios": ["16:9"], "min_seconds": 9}),
        deployment("toapis/grok-video-1.0", blocked=True),
        deployment("toapis/seedance-2-5", supported_endpoints=[]),
        deployment("toapis/seedance-2-mini", team_id="another-team"),
    ]
    router = Router(model_list=rows, num_retries=0)
    monkeypatch.setattr(proxy_server, "llm_router", router)
    monkeypatch.setattr(proxy_server, "general_settings", {})
    monkeypatch.setattr(utils, "get_available_models_for_user", AsyncMock(return_value=["video"]))
    auth = UserAPIKeyAuth(team_id="my-team")
    plain = await proxy_server.model_list(user_api_key_dict=auth)
    assert "metadata" not in plain["data"][0]
    response = await proxy_server.model_list(user_api_key_dict=auth, include_metadata=True)
    rules = response["data"][0]["metadata"]["generation_constraints"]
    assert rules["version"] == 1
    assert [(v["aspect_ratio"], v["seconds"]) for v in rules["variants"]] == [
        (["9:16"], list(range(4, 9))),
        (["16:9"], list(range(9, 16))),
    ]
    candidates = router.get_model_list(model_name="video")[:2]
    for ratio, seconds, expected in [("16:9", 15, 1), ("9:16", 8, 0)]:
        accepted = router._filter_deployments_by_video_generation_params(
            "video",
            candidates,
            {"_router_call_type": "video_generation", "aspect_ratio": ratio, "seconds": str(seconds)},
        )
        assert accepted == [candidates[expected]]
    for params in [{"aspect_ratio": "9:16", "seconds": 15}, {"seconds": 9}, {"aspect_ratio": "1:1"}, {}]:
        with pytest.raises(BadRequestError):
            router._filter_deployments_by_video_generation_params(
                "video", candidates, {"_router_call_type": "video_generation", **params}
            )


def test_omni_and_all_current_video_groups_have_specific_rules():
    omni = deployment_video_variants(OMNI)[0]
    assert omni == {
        "aspect_ratio": [None, "16:9", "9:16"],
        "seconds": [None, *range(4, 16)],
        "resolution": [None, "720p"],
    }
    for name in ("grok-video-1.0", "seedance-2", "seedance-2-fast", "seedance-2-mini", "seedance-2-5"):
        assert deployment_video_variants("toapis/" + name)
    assert deployment_video_variants("guanghe/seedance2.0-fast（国内key100）")
    assert deployment_video_variants("guanghe/seedance2.0-fast（企业折扣）") is None
    old_omni = deployment_video_variants("toapis/gemini-omni-flash")
    assert all("9:16" not in v["aspect_ratio"] for v in old_omni if "1080p" in v["resolution"])
    frame_variant = next(v for v in deployment_video_variants("toapis/seedance-2-5") if v.get("frames"))
    assert frame_variant["aspect_ratio"] == [None] and frame_variant["seconds"] == [None, -1]


def test_limits_survive_deployment_json_round_trip_and_reject_inverted_ranges():
    row = Deployment(**deployment(OMNI, video_generation_limits={"max_seconds": 10}))
    restored = Deployment.model_validate_json(row.model_dump_json())
    assert restored.model_info.video_generation_limits.max_seconds == 10
    with pytest.raises(ValidationError):
        VideoGenerationLimits(min_seconds=16, max_seconds=10)


@pytest.mark.asyncio
@pytest.mark.parametrize("params", [{"seconds": "16"}, {"aspect_ratio": "1:1"}, {"seconds": "-1"}])
async def test_invalid_public_request_never_submits(params, respx_mock):
    router = Router(model_list=[deployment(OMNI)], num_retries=0)
    with pytest.raises(BadRequestError):
        await router.avideo_generation(model="video", prompt="offline validation", **params)
    assert not respx_mock.calls
