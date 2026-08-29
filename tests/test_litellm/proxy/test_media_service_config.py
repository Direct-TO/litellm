from pathlib import Path

import pytest
import yaml

from litellm import Router
from litellm.llms.toapis.image_generation.transformation import ToAPISImageGenerationConfig
from litellm.llms.zexapi.image_generation.transformation import ZexAPIImageGenerationConfig
from litellm.proxy._types import ConfigGeneralSettings
from litellm.types.router import AllowedFailsPolicy, RouterGeneralSettings


def _dev_config() -> dict:
    config_path = Path(__file__).resolve().parents[3] / "litellm" / "proxy" / "dev_config.yaml"
    with config_path.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def test_media_services_hide_provider_deployments_behind_model_groups():
    config = _dev_config()
    model_list = config["model_list"]
    image_deployments = [
        item["litellm_params"]["model"] for item in model_list if item["model_name"] == "image-generation"
    ]
    generation_only_deployments = [
        item["litellm_params"]["model"]
        for item in model_list
        if item["model_name"] == "image-generation-only-toapis"
    ]
    video_deployments = [
        item["litellm_params"]["model"] for item in model_list if item["model_name"] == "video-generation"
    ]

    assert image_deployments == ["zexapi/image2"]
    assert generation_only_deployments == ["toapis/gpt-image-2"]
    assert video_deployments == ["toapis/seedance-2-5"]


def test_media_service_defaults_and_failure_cooldown_policy():
    config = _dev_config()
    general_settings = ConfigGeneralSettings(**config["general_settings"])
    media_deployments = [
        item
        for item in config["model_list"]
        if item["model_name"]
        in {"image-generation", "image-generation-only-toapis", "video-generation"}
    ]

    assert general_settings.image_generation_model == "image-generation"
    assert general_settings.video_generation_model == "video-generation"
    assert general_settings.model_list_generation_only is True
    assert RouterGeneralSettings().pass_through_all_models is False
    assert "router_settings" not in config
    for deployment in media_deployments:
        assert deployment["litellm_params"]["num_retries"] == 0
        assert deployment["model_info"]["allowed_fails"] == 2
        assert deployment["model_info"]["cooldown_time"] == 60
        policy = AllowedFailsPolicy(**deployment["model_info"]["allowed_fails_policy"])
        assert policy.AuthenticationErrorAllowedFails == 0
        assert policy.TimeoutErrorAllowedFails == 2
        assert policy.RateLimitErrorAllowedFails == 2
        assert policy.InternalServerErrorAllowedFails == 2
        assert policy.ServiceUnavailableErrorAllowedFails == 2
        assert policy.BadGatewayErrorAllowedFails == 2
        assert policy.NotFoundErrorAllowedFails == 1
        assert policy.BadRequestErrorAllowedFails is None
        assert policy.ContentPolicyViolationErrorAllowedFails is None


def test_router_builds_extensible_media_deployment_groups(monkeypatch):
    monkeypatch.setenv("TOAPIS_API_KEY", "fake-toapis-key")
    monkeypatch.setenv("ZEXAPI_API_KEY", "fake-zexapi-key")
    config = _dev_config()
    media_models = [
        item
        for item in config["model_list"]
        if item["model_name"]
        in {"image-generation", "image-generation-only-toapis", "video-generation"}
    ]
    router = Router(model_list=media_models)

    groups = tuple((item["model_name"], item["litellm_params"]["model"]) for item in router.model_list)
    assert groups == (
        ("image-generation", "zexapi/image2"),
        ("image-generation-only-toapis", "toapis/gpt-image-2"),
        ("video-generation", "toapis/seedance-2-5"),
    )
    assert router.routing_strategy == "simple-shuffle"
    for deployment in router.model_list:
        assert deployment["litellm_params"]["num_retries"] == 0
        assert deployment["model_info"]["allowed_fails"] == 2
        assert deployment["model_info"]["cooldown_time"] == 60


@pytest.mark.parametrize(
    "ratio,zexapi_size",
    [
        ("1:1", "1024x1024"),
        ("16:9", "1280x720"),
        ("9:16", "720x1280"),
        ("3:2", "1248x832"),
        ("2:3", "832x1248"),
        ("4:3", "1152x864"),
        ("3:4", "864x1152"),
        ("5:4", "1120x896"),
        ("4:5", "896x1120"),
        ("21:9", "1456x624"),
    ],
)
def test_image_service_maps_same_public_aspect_ratio_to_each_provider(ratio, zexapi_size):
    toapis_params = ToAPISImageGenerationConfig().map_openai_params(
        non_default_params={"size": ratio},
        optional_params={},
        model="gpt-image-2",
        drop_params=False,
    )
    zexapi_params = ZexAPIImageGenerationConfig().map_openai_params(
        non_default_params={"size": ratio},
        optional_params={},
        model="image2",
        drop_params=False,
    )

    assert toapis_params == {"size": ratio, "resolution": "1k", "response_format": "url"}
    assert zexapi_params == {"size": zexapi_size, "response_format": "url"}
