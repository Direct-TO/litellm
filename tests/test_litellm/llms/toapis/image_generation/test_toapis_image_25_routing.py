import json
from pathlib import Path

import httpx
import pytest
import yaml

import litellm
from litellm import Router
from litellm.llms.base_llm.submission_utils import get_submission_outcome
from litellm.llms.toapis.image_generation.transformation import ToAPISImageGenerationConfig
from litellm.types.router import UpdateRouterConfig, WeightedFailoverPolicy


@pytest.fixture(autouse=True)
def use_mockable_httpx(monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)


def config():
    root = Path(__file__).resolve().parents[5]
    return yaml.safe_load((root / "litellm/proxy/dev_config.yaml").read_text(encoding="utf-8"))


def router_for(variant, monkeypatch):
    monkeypatch.setenv("TOAPIS_API_KEY", "test-key")
    data = config()
    models = [item for item in data["model_list"] if item["model_name"] == f"image-2.5-{variant}"]
    return Router(model_list=models, num_retries=0, **data["router_settings"])


def completed(model):
    return {
        "id": "task-image-25",
        "object": "generation.task",
        "model": model,
        "status": "completed",
        "result": {"type": "image", "data": [{"url": "https://files.example/blue-teapot.png"}]},
    }


@pytest.mark.parametrize("variant", ["flare", "sunburst"])
@pytest.mark.parametrize("resolution,pixels", [("1K", "1536x864"), ("2K", "2048x1152"), ("4K", "3840x2160")])
def test_normal_and_vip_preserve_same_image_contract(variant, resolution, pixels):
    mapper = ToAPISImageGenerationConfig()
    request = {
        "size": "16:9", "resolution": resolution, "quality": "high", "background": "transparent",
        "image_url": ["https://files.example/reference.png"], "n": 1,
    }
    normal = mapper.map_openai_params(request, {}, f"gpt-image-2.5-{variant}", False)
    vip = mapper.map_openai_params(request, {}, f"gpt-image-2.5-{variant}-vip", False)
    assert normal == {
        "size": "16:9", "resolution": resolution, "response_format": "url", "quality": "high",
        "background": "transparent", "reference_images": request["image_url"], "n": 1,
    }
    assert vip == {key: value for key, value in {**normal, "size": pixels}.items() if key != "resolution"}


@pytest.mark.parametrize("variant", ["flare", "sunburst"])
def test_vip_custom_pixels_and_quality(variant):
    mapper = ToAPISImageGenerationConfig()
    vip = f"gpt-image-2.5-{variant}-vip"
    mapped = mapper.map_openai_params({"size": "1600x1024", "quality": "max"}, {}, vip, False)
    assert mapped == {"size": "1600x1024", "response_format": "url", "quality": "max"}
    with pytest.raises(litellm.UnsupportedParamsError, match="cannot be combined"):
        mapper.map_openai_params({"size": "1600x1024", "resolution": "2K"}, {}, vip, False)
    with pytest.raises(litellm.UnsupportedParamsError, match="same aspect ratio"):
        mapper.map_openai_params({"size": "1600x1024", "aspect_ratio": "1:1"}, {}, vip, False)
    with pytest.raises(litellm.UnsupportedParamsError, match="VIP quality"):
        mapper.map_openai_params({"quality": "ultra"}, {}, vip, False)
    normal = mapper.map_openai_params({"quality": "low"}, {}, f"gpt-image-2.5-{variant}", False)
    assert normal["quality"] == "high"


@pytest.mark.asyncio
@pytest.mark.parametrize("variant", ["flare", "sunburst"])
async def test_healthy_normal_never_calls_vip(variant, monkeypatch, respx_mock):
    bodies = []

    def respond(request):
        body = json.loads(request.content)
        bodies.append(body)
        return httpx.Response(200, json=completed(body["model"]))

    respx_mock.post("https://api.toapis.com/v1/images/generations").mock(side_effect=respond)
    router = router_for(variant, monkeypatch)
    response = await router.aimage_generation(model=f"image-2.5-{variant}", prompt="blue teapot", size="1:1")
    assert response.data[0].url == "https://files.example/blue-teapot.png"
    assert [body["model"] for body in bodies] == [f"gpt-image-2.5-{variant}"]
    assert bodies[0]["resolution"] == "1K"


@pytest.mark.asyncio
@pytest.mark.parametrize("variant", ["flare", "sunburst"])
@pytest.mark.parametrize("status", [400, 401, 403, 404, 422, 429, 503])
async def test_rejected_normal_falls_back_once_with_reference_and_size(variant, status, monkeypatch, respx_mock):
    bodies = []

    def respond(request):
        body = json.loads(request.content)
        bodies.append(body)
        if not body["model"].endswith("-vip"):
            return httpx.Response(status, json={"error": {"message": "no available channel"}})
        return httpx.Response(200, json=completed(body["model"]))

    respx_mock.post("https://api.toapis.com/v1/images/generations").mock(side_effect=respond)
    router = router_for(variant, monkeypatch)
    response = await router.aimage_generation(
        model=f"image-2.5-{variant}", prompt="blue teapot", size="16:9", resolution="2K",
        quality="high", background="transparent", image_url="https://files.example/source.png",
    )
    assert response.data[0].url == "https://files.example/blue-teapot.png"
    assert [body["model"] for body in bodies] == [f"gpt-image-2.5-{variant}", f"gpt-image-2.5-{variant}-vip"]
    assert bodies[0]["size"] == "16:9"
    assert bodies[0]["resolution"] == "2K"
    assert bodies[1]["size"] == "2048x1152"
    assert "resolution" not in bodies[1]
    for body in bodies:
        assert body["quality"] == "high"
        assert body["background"] == "transparent"
        assert body["reference_images"] == ["https://files.example/source.png"]


@pytest.mark.asyncio
@pytest.mark.parametrize("failure", ["timeout", "ambiguous_503", "server_500", "invalid_json", "accepted_failed"])
async def test_unknown_or_accepted_generation_never_calls_vip(failure, monkeypatch, respx_mock):
    bodies = []

    def respond(request):
        body = json.loads(request.content)
        bodies.append(body)
        if failure == "timeout":
            raise httpx.ReadTimeout("response lost", request=request)
        if failure == "ambiguous_503":
            return httpx.Response(503, text="service unavailable")
        if failure == "server_500":
            return httpx.Response(500, text="internal error")
        if failure == "invalid_json":
            return httpx.Response(200, json={"unexpected": True})
        return httpx.Response(200, json={**completed(body["model"]), "status": "failed", "result": None})

    respx_mock.post("https://api.toapis.com/v1/images/generations").mock(side_effect=respond)
    router = router_for("flare", monkeypatch)
    with pytest.raises(Exception) as exc:
        await router.aimage_generation(model="image-2.5-flare", prompt="blue teapot")
    assert [body["model"] for body in bodies] == ["gpt-image-2.5-flare"]
    assert get_submission_outcome(exc.value) == ("accepted" if failure == "accepted_failed" else "unknown")


@pytest.mark.asyncio
async def test_both_rejected_stop_after_one_attempt_each(monkeypatch, respx_mock):
    bodies = []

    def respond(request):
        bodies.append(json.loads(request.content))
        return httpx.Response(429, text="rate limited")

    respx_mock.post("https://api.toapis.com/v1/images/generations").mock(side_effect=respond)
    router = router_for("flare", monkeypatch)
    with pytest.raises(litellm.RateLimitError):
        await router.aimage_generation(model="image-2.5-flare", prompt="blue teapot")
    assert [body["model"] for body in bodies] == ["gpt-image-2.5-flare", "gpt-image-2.5-flare-vip"]


def test_override_roundtrip_and_other_model_groups_keep_provider_scope():
    policy = WeightedFailoverPolicy(**config()["router_settings"]["weighted_failover_policy"])
    settings = UpdateRouterConfig(weighted_failover_policy=policy).model_dump(exclude_none=True)
    router = Router(model_list=[])
    router.update_settings(**settings)
    assert router._get_weighted_failover_policy({"model": "image-2.5-flare"}).failure_scope == "deployment"
    assert router._get_weighted_failover_policy({"model": "seedance-2"}).failure_scope == "provider"
    assert router._get_weighted_failover_policy({"model": "seedance-2"}).status_codes == [403, 429, 503]
