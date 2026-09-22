import json
from pathlib import Path

import pytest
import yaml

from litellm import Router
from litellm.llms.toapis.image_generation.transformation import ToAPISImageGenerationConfig
from litellm.llms.zexapi.image_generation.transformation import ZexAPIImageGenerationConfig
from litellm.proxy._types import ConfigGeneralSettings
from litellm.proxy.common_utils.model_capability import resolve_model_capability
from litellm.types.router import AllowedFailsPolicy, RouterGeneralSettings, WeightedFailoverPolicy

pytestmark = pytest.mark.usefixtures("local_model_cost_map")

_TOAPIS_VIDEO_MODELS = {
    "gemini-omni-flash",
    "gemini-omni-flash-preview-official",
    "veo3.1-fast",
    "veo3.1-quality",
    "veo3.1-lite",
    "Veo3.1-fast-official",
    "Veo3.1-quality-official",
    "Veo3.1-lite-official",
    "seedance-2",
    "seedance-2-fast",
    "seedance-2-mini",
    "seedance-2-5",
    "wan3.0-video",
    "happyhorse-1.1",
    "MiniMax-H3",
    "kling-v2-6",
    "kling-v3",
    "kling-3.0-turbo",
    "kling-v3-omni",
    "kling-video-o1",
    "grok-video-1.0",
    "grok-video-1.5",
}
_ZEXAPI_VEO_MODELS = {
    "veo_3_1",
    "veo_3_1-fl",
    "veo_3_1-hd",
    "veo_3_1-hd-fl",
    "veo_3_1-4K",
    "veo_3_1-4K-fl",
    "veo_3_1-fast",
    "veo_3_1-fast-fl",
    "veo_3_1-fast-hd",
    "veo_3_1-fast-fl-hd",
    "veo_3_1-fast-4K",
    "veo_3_1-fast-4K-fl",
    "veo_3_1-lite",
    "veo_3_1-lite-fl",
    "veo_3_1-lite-hd",
    "veo_3_1-lite-hd-fl",
    "veo_3_1-lite-4K",
    "veo_3_1-lite-4K-fl",
}
_IMAGE_25_DEPLOYMENTS = {
    f"toapis/gpt-image-2.5-{variant}{suffix}"
    for variant in ("flare", "sunburst")
    for suffix in ("", "-vip")
}
_EXPECTED_MEDIA_DEPLOYMENTS = {
    *_IMAGE_25_DEPLOYMENTS,
    "toapis/gemini-2.5-flash-image-preview",
    "toapis/gemini-3-pro-image-preview",
    "toapis/gemini-3.1-flash-image-preview",
    "zexapi/gemini-3-pro-image-preview",
    "zexapi/gemini-3.1-flash-image-preview",
    "zexapi/omni_flash-10s",
    "zexapi/omni_flash-10s-fl",
    "guanghe/seedance2.0_企业折扣",
    "guanghe/seedance2.5_企业折",
    "guanghe/seedance2.0-fast（企业折扣）",
    "guanghe/seedance2.0-mini（企业折扣）",
    *(f"toapis/{model}" for model in _TOAPIS_VIDEO_MODELS),
    *(f"zexapi/{model}" for model in _ZEXAPI_VEO_MODELS),
}


def _dev_config() -> dict:
    config_path = Path(__file__).resolve().parents[3] / "litellm" / "proxy" / "dev_config.yaml"
    with config_path.open(encoding="utf-8") as config_file:
        return yaml.safe_load(config_file)


def test_media_catalogs_match_and_cover_every_configured_deployment():
    root = Path(__file__).resolve().parents[3]
    main_catalog = json.loads((root / "model_prices_and_context_window.json").read_text(encoding="utf-8"))
    backup_catalog = json.loads(
        (root / "litellm" / "model_prices_and_context_window_backup.json").read_text(encoding="utf-8")
    )
    configured_media_models = {
        item["litellm_params"]["model"]
        for item in _dev_config()["model_list"]
        if item["litellm_params"]["model"].startswith(("toapis/", "zexapi/", "guanghe/"))
    }

    assert main_catalog == backup_catalog
    assert configured_media_models == _EXPECTED_MEDIA_DEPLOYMENTS
    assert configured_media_models <= main_catalog.keys()
    assert all(
        main_catalog[model]["mode"] in {"image_generation", "video_generation"} for model in configured_media_models
    )


def test_media_models_hide_provider_deployments_behind_public_model_names():
    config = _dev_config()
    model_list = config["model_list"]
    gpt_image_deployments = [
        item["litellm_params"]["model"] for item in model_list if item["model_name"].startswith("image-2.5-")
    ]
    banana_deployments = [
        item["litellm_params"]["model"] for item in model_list if item["model_name"] == "gemini-3.1-flash-image-preview"
    ]
    veo_deployments = [item["litellm_params"]["model"] for item in model_list if item["model_name"] == "veo3.1-fast"]
    public_names = {item["model_name"] for item in model_list}

    assert set(gpt_image_deployments) == _IMAGE_25_DEPLOYMENTS
    assert "gpt-image-2" not in public_names
    for variant in ("flare", "sunburst"):
        group = [item for item in model_list if item["model_name"] == f"image-2.5-{variant}"]
        assert [(item["litellm_params"]["model"], item["litellm_params"]["order"]) for item in group] == [
            (f"toapis/gpt-image-2.5-{variant}", 1),
            (f"toapis/gpt-image-2.5-{variant}-vip", 2),
        ]
    assert banana_deployments == [
        "toapis/gemini-3.1-flash-image-preview",
        "zexapi/gemini-3.1-flash-image-preview",
    ]
    assert veo_deployments == ["toapis/veo3.1-fast", "zexapi/veo_3_1-fast"]
    assert {"image-generation", "image-generation-only-toapis", "video-generation"}.isdisjoint(public_names)


def test_media_service_defaults_and_failure_cooldown_policy():
    config = _dev_config()
    general_settings = ConfigGeneralSettings(**config["general_settings"])
    media_deployments = [
        item
        for item in config["model_list"]
        if item["litellm_params"]["model"].startswith(("toapis/", "zexapi/", "guanghe/"))
    ]

    assert general_settings.completion_model is None
    assert general_settings.image_generation_model is None
    assert general_settings.video_generation_model is None
    assert general_settings.model_list_generation_only is True
    assert RouterGeneralSettings().pass_through_all_models is False
    router_settings = config["router_settings"]
    failover_policy = WeightedFailoverPolicy(**router_settings["weighted_failover_policy"])
    assert router_settings["routing_strategy"] == "simple-shuffle"
    assert router_settings["enable_weighted_failover"] is True
    assert failover_policy.call_types == ["aimage_generation", "avideo_generation"]
    assert failover_policy.status_codes == [403, 429, 503]
    assert failover_policy.submission_outcomes == ["rejected"]
    assert failover_policy.failure_scope == "provider"
    assert len(media_deployments) == 55
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
    monkeypatch.setenv("GUANGHE_API_KEY", "fake-guanghe-key")
    config = _dev_config()
    media_models = [
        item
        for item in config["model_list"]
        if item["model_name"] in {"image-2.5-flare", "image-2.5-sunburst", "gemini-3.1-flash-image-preview", "veo3.1-fast"}
    ]
    router = Router(model_list=media_models, **config["router_settings"])

    groups = tuple((item["model_name"], item["litellm_params"]["model"]) for item in router.model_list)
    assert groups == (
        ("image-2.5-flare", "toapis/gpt-image-2.5-flare"),
        ("image-2.5-flare", "toapis/gpt-image-2.5-flare-vip"),
        ("image-2.5-sunburst", "toapis/gpt-image-2.5-sunburst"),
        ("image-2.5-sunburst", "toapis/gpt-image-2.5-sunburst-vip"),
        ("gemini-3.1-flash-image-preview", "toapis/gemini-3.1-flash-image-preview"),
        ("gemini-3.1-flash-image-preview", "zexapi/gemini-3.1-flash-image-preview"),
        ("veo3.1-fast", "toapis/veo3.1-fast"),
        ("veo3.1-fast", "zexapi/veo_3_1-fast"),
    )
    assert router.routing_strategy == "simple-shuffle"
    assert router.enable_weighted_failover is True
    assert router.weighted_failover_policy == WeightedFailoverPolicy(
        call_types=["aimage_generation", "avideo_generation"],
        status_codes=[403, 429, 503],
        submission_outcomes=["rejected"],
        failure_scope="provider",
        model_group_overrides={
            name: WeightedFailoverPolicy(
                call_types=["aimage_generation"],
                status_codes=[400, 401, 403, 404, 422, 429, 503],
                submission_outcomes=["rejected"],
                failure_scope="deployment",
            )
            for name in ("image-2.5-flare", "image-2.5-sunburst")
        },
    )
    for deployment in router.model_list:
        assert deployment["litellm_params"]["num_retries"] == 0
        assert deployment["model_info"]["allowed_fails"] == 2
        assert deployment["model_info"]["cooldown_time"] == 60


def test_all_public_media_models_have_generation_capabilities(monkeypatch):
    monkeypatch.setenv("TOAPIS_API_KEY", "fake-toapis-key")
    monkeypatch.setenv("ZEXAPI_API_KEY", "fake-zexapi-key")
    monkeypatch.setenv("GUANGHE_API_KEY", "fake-guanghe-key")
    config = _dev_config()
    media_models = [
        item
        for item in config["model_list"]
        if item["litellm_params"]["model"].startswith(("toapis/", "zexapi/", "guanghe/"))
    ]
    router = Router(model_list=media_models)
    public_names = {item["model_name"] for item in media_models}

    for model_name in public_names:
        expected = (
            "image"
            if any(
                item["model_name"] == model_name
                and item["litellm_params"]["model"]
                in {
                    *_IMAGE_25_DEPLOYMENTS,
                    "toapis/gemini-2.5-flash-image-preview",
                    "toapis/gemini-3-pro-image-preview",
                    "zexapi/gemini-3-pro-image-preview",
                    "toapis/gemini-3.1-flash-image-preview",
                    "zexapi/gemini-3.1-flash-image-preview",
                }
                for item in media_models
            )
            else "video"
        )
        assert resolve_model_capability(model_name, router) == expected


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


@pytest.mark.parametrize(
    "resolution,zexapi_size",
    [
        ("2K", "2048x2048"),
        ("4K", "2880x2880"),
    ],
)
def test_gpt_image_2_high_resolution_maps_to_both_capable_providers(resolution, zexapi_size):
    toapis_params = ToAPISImageGenerationConfig().map_openai_params(
        non_default_params={"size": "1:1", "resolution": resolution},
        optional_params={},
        model="gpt-image-2",
        drop_params=False,
    )
    zexapi_params = ZexAPIImageGenerationConfig().map_openai_params(
        non_default_params={"size": "1:1", "resolution": resolution},
        optional_params={},
        model="gpt-image2",
        drop_params=False,
    )

    assert toapis_params == {
        "size": "1:1",
        "resolution": resolution.lower(),
        "response_format": "url",
    }
    assert zexapi_params == {"size": zexapi_size, "response_format": "url"}
