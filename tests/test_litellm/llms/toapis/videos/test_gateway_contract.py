import httpx
import pytest

import litellm
from litellm.llms.toapis.videos.transformation import ToAPISVideoConfig
from litellm.router import Router
from litellm.videos.contract import validate_video_contract
from litellm.videos.utils import VideoGenerationRequestUtils


def reference(kind="image", role="reference", name="one"):
    return {"type": kind, "role": role, "url": f"https://media.example/{name}"}


def mapped(model="seedance-2", **params):
    return VideoGenerationRequestUtils.get_optional_params_video_generation(model, ToAPISVideoConfig(), params)


@pytest.mark.parametrize("resolution", ["480p", "720p", "1080p", "4K"])
def test_seedance_preserves_resolution_and_ratio(resolution):
    result = mapped(resolution=resolution, aspect_ratio="16:9", seconds="5")
    assert result == {"resolution": resolution.lower(), "aspect_ratio": "16:9", "duration": 5}


def test_resolution_is_not_silently_dropped_with_drop_params(monkeypatch):
    monkeypatch.setattr(litellm, "drop_params", True)
    with pytest.raises(litellm.UnsupportedParamsError, match="resolution"):
        mapped("seedance-2-fast", resolution="1080p")


@pytest.mark.parametrize(
    "model",
    [
        "kling-v2-6",
        "kling-3.0-turbo",
        "Veo3.1-lite-official",
    ],
)
@pytest.mark.parametrize(
    "params",
    [{"resolution": "720p"}, {"aspect_ratio": "16:9"}, {"references": [reference()]}],
)
def test_models_without_gateway_contract_reject_new_fields(model, params, monkeypatch):
    monkeypatch.setattr(litellm, "drop_params", True)
    config = ToAPISVideoConfig()
    assert not {"resolution", "aspect_ratio", "references"}.intersection(config.get_supported_openai_params(model))
    with pytest.raises(litellm.UnsupportedParamsError, match="no documented mapping"):
        mapped(model, **params)
    with pytest.raises(litellm.UnsupportedParamsError, match="no documented mapping"):
        config.map_openai_params(params, model=model, drop_params=True)


def test_multimodal_order_and_roles_are_preserved():
    refs = [
        reference(name="one"),
        reference(name="two"),
        reference("video", name="motion"),
        reference("audio", name="sound"),
    ]
    result = mapped(references=refs, resolution="720p")
    assert result["image_with_roles"] == [{"url": item["url"], "role": "reference_image"} for item in refs[:2]]
    assert result["video_with_roles"] == [{"url": refs[2]["url"], "role": "reference_video"}]
    assert result["audio_with_roles"] == [{"url": refs[3]["url"], "role": "reference_audio"}]
    assert "references" not in result


def test_explicit_frames_and_mixed_input_rejection():
    first = reference(role="first_frame", name="first")
    last = reference(role="last_frame", name="last")
    result = mapped(references=[last, first], resolution="1080p", aspect_ratio="16:9")
    assert result["image_with_roles"] == [
        {"url": last["url"], "role": "last_frame"},
        {"url": first["url"], "role": "first_frame"},
    ]
    with pytest.raises(litellm.UnsupportedParamsError, match="mixed"):
        mapped(references=[first, reference()])


def test_wan_and_h3_map_documented_distinct_fields():
    refs = [reference(), reference("video"), reference("audio")]
    wan = mapped("wan3.0-video", references=refs, aspect_ratio="9:16", resolution="1080p")
    assert wan["ratio"] == "9:16"
    assert wan["reference_images"] == [refs[0]["url"]]
    assert wan["video_list"] == [{"video_url": refs[1]["url"]}]
    assert mapped("MiniMax-H3", references=refs, resolution="2K")["resolution"] == "2K"


def test_happyhorse_and_veo_preserve_reference_operation():
    refs = [reference(name="one"), reference(name="two"), reference(name="three")]
    happy = mapped("happyhorse-1.1", references=refs, resolution="1080p")
    assert happy["action"] == "reference-to-video"
    assert happy["resolution"] == "1080P"
    assert happy["reference_images"] == [ref["url"] for ref in refs]
    veo = mapped("veo3.1-fast", references=refs, resolution="4K")
    assert veo["metadata"] == {"resolution": "4k", "generation_type": "reference"}
    assert "generation_type" not in veo
    frames = [reference(role="last_frame", name="last"), reference(role="first_frame", name="first")]
    veo_frames = mapped("veo3.1-fast", references=frames, resolution="1080p")
    assert veo_frames["metadata"] == {"resolution": "1080p", "generation_type": "frame"}
    assert veo_frames["image_urls"] == [frames[1]["url"], frames[0]["url"]]


@pytest.mark.parametrize(
    "params",
    [
        {"resolution": "2160"},
        {"resolution": "8K"},
        {"resolution": "720p", "size": "1280x720"},
        {"aspect_ratio": "1920x1080"},
        {"references": [reference("video", "first_frame")]},
        {"references": [reference(role="last_frame")]},
        {"references": [{"type": "image", "url": "file:///tmp/image.png"}]},
    ],
)
def test_invalid_contract_is_rejected(params):
    with pytest.raises(ValueError):
        validate_video_contract(params)


def test_router_filters_before_submission():
    candidates = [
        {"model_name": "video", "litellm_params": {"model": f"toapis/{model}"}, "model_info": {"id": model}}
        for model in ["seedance-2-fast", "seedance-2"]
    ]
    kwargs = {"_router_call_type": "avideo_generation", "resolution": "1080p", "aspect_ratio": "16:9"}
    assert Router._filter_deployments_by_video_generation_params("video", candidates, kwargs) == candidates[1:]
    with pytest.raises(litellm.BadRequestError, match="no deployment"):
        Router._filter_deployments_by_video_generation_params("video", candidates[:1], kwargs)


def test_router_checks_conflicting_deployment_defaults():
    good = {"model_name": "video", "litellm_params": {"model": "toapis/seedance-2"}, "model_info": {"id": "good"}}
    bad = {**good, "litellm_params": {**good["litellm_params"], "size": "1280x720"}, "model_info": {"id": "bad"}}
    kwargs = {"_router_call_type": "avideo_generation", "resolution": "1080p", "aspect_ratio": "16:9"}
    assert Router._filter_deployments_by_video_generation_params("video", [bad, good], kwargs) == [good]


def test_safe_native_options_survive_but_canonical_overrides_fail():
    assert mapped(resolution="720p", extra_body={"watermark": True})["watermark"] is True
    with pytest.raises(litellm.UnsupportedParamsError, match="override"):
        mapped(resolution="720p", extra_body={"resolution": "1080p"})
    with pytest.raises(litellm.UnsupportedParamsError, match="seconds"):
        mapped(resolution="720p", seconds="five")
    with pytest.raises(litellm.UnsupportedParamsError, match="override"):
        mapped(resolution="720p", seconds="5", extra_body={"duration": 99})
    with pytest.raises(litellm.UnsupportedParamsError, match="aspect_ratio"):
        mapped("wan3.0-video", aspect_ratio="21:9")


@pytest.mark.asyncio
async def test_public_create_sends_canonical_references_to_native_endpoint_once(respx_mock, monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route = respx_mock.post("https://gateway-fixture.example/v1/videos/generations").mock(
        return_value=httpx.Response(200, json={"id": "task-fixture", "object": "generation.task", "status": "queued"})
    )
    result = await litellm.avideo_generation(
        model="toapis/seedance-2",
        prompt="fixture",
        seconds="5",
        resolution="1080p",
        aspect_ratio="16:9",
        references=[reference(), reference("video")],
        api_key="fixture-key",
        api_base="https://gateway-fixture.example",
    )
    import json

    body = json.loads(route.calls[0].request.content)
    assert result.id and len(route.calls) == 1
    assert body["resolution"] == "1080p" and body["aspect_ratio"] == "16:9"
    assert body["video_with_roles"][0]["role"] == "reference_video"
    assert not {"references", "size", "width", "height"}.intersection(body)
