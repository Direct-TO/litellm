"""Regression coverage for the 13 enabled video models audited on 2026-09-11.

Expected payloads follow provider documentation, independent of mapper tables.
All HTTP creation routes are mocked; these tests do not create paid tasks.
"""

import json

import pytest

import litellm
from litellm.llms.toapis.videos.transformation import ToAPISVideoConfig
from litellm.llms.zexapi.videos.transformation import ZexAPIVideoConfig
from litellm.router import Router
from litellm.videos.utils import VideoGenerationRequestUtils


def ref(name="one", kind="image", role="reference"):
    return {"url": f"https://media.example/{name}", "type": kind, "role": role}


def mapped(model, provider="toapis", **params):
    config = ToAPISVideoConfig() if provider == "toapis" else ZexAPIVideoConfig()
    return VideoGenerationRequestUtils.get_optional_params_video_generation(model, config, params)


ENABLED_CASES = [
    ("toapis", "gemini-omni-flash", "6", {"image_urls": [ref()["url"]]}),
    ("toapis", "gemini-omni-flash-preview-official", "6", {"image_urls": [ref()["url"]]}),
    ("toapis", "grok-video-1.0", "10", {"reference_images": [ref()["url"]]}),
    ("toapis", "grok-video-1.5", "10", {"image": ref()["url"]}),
    ("toapis", "seedance-2", "5", {"image_with_roles": [{"url": ref()["url"], "role": "reference_image"}]}),
    ("toapis", "seedance-2-fast", "5", {"image_with_roles": [{"url": ref()["url"], "role": "reference_image"}]}),
    ("toapis", "seedance-2-mini", "5", {"image_with_roles": [{"url": ref()["url"], "role": "reference_image"}]}),
    ("toapis", "seedance-2-5", "5", {"image_with_roles": [{"url": ref()["url"], "role": "reference_image"}]}),
    ("toapis", "wan3.0-video", "5", {"reference_images": [ref()["url"]]}),
    ("toapis", "kling-v3", "5", {"reference_images": [ref()["url"]]}),
    ("toapis", "happyhorse-1.1", "5", {"action": "reference-to-video", "reference_images": [ref()["url"]]}),
    ("zexapi", "omni_flash-10s", "10", {"images": [ref()["url"]]}),
    ("zexapi", "omni_flash-10s-fl", "10", {"images": [ref()["url"]]}),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("provider,model,seconds,media", ENABLED_CASES)
async def test_enabled_model_router_submits_documented_payload_once(
    provider, model, seconds, media, respx_mock, monkeypatch
):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    endpoint = "/v1/videos/generations" if provider == "toapis" else "/v1/videos"
    response = {
        "id": "fixture-task",
        "object": "generation.task" if provider == "toapis" else "video",
        "status": "queued",
    }
    route = respx_mock.post("https://provider-fixture.example" + endpoint).respond(200, json=response)
    router = Router(
        model_list=[
            {
                "model_name": "public-video",
                "litellm_params": {
                    "model": f"{provider}/{model}",
                    "api_key": "fixture-key",
                    "api_base": "https://provider-fixture.example/v1",
                },
            }
        ],
        num_retries=0,
    )
    references = [ref(role="first_frame" if model.endswith("-fl") else "reference")]
    result = await router.avideo_generation(
        model="public-video",
        prompt="contract fixture",
        seconds=seconds,
        resolution="720p",
        aspect_ratio="16:9",
        references=references,
    )
    expected = {"model": model, "prompt": "contract fixture", **media}
    if provider == "zexapi":
        expected["size"] = "1280x720"
    else:
        expected["duration"] = int(seconds)
        expected["ratio" if model == "wan3.0-video" else "aspect_ratio"] = "16:9"
        expected["mode" if model == "kling-v3" else "resolution"] = (
            "std" if model == "kling-v3" else "720P" if model == "happyhorse-1.1" else "720p"
        )
    assert result.id
    assert route.call_count == 1
    assert json.loads(route.calls[0].request.content) == expected


def test_kling_explicit_frames_and_mixed_references_keep_roles_and_order():
    refs = [ref("end", role="last_frame"), ref("style"), ref("start", role="first_frame")]
    assert mapped("kling-v3", references=refs, resolution="1080p") == {
        "mode": "pro",
        "image_with_roles": [
            {"url": refs[0]["url"], "role": "last_frame"},
            {"url": refs[1]["url"], "role": "reference_image"},
            {"url": refs[2]["url"], "role": "first_frame"},
        ],
    }


def test_grok_main_image_and_regular_images_are_distinct():
    refs = [ref("style"), ref("main", role="first_frame")]
    assert mapped("grok-video-1.0", references=refs, seconds="15", resolution="480p", aspect_ratio="3:2") == {
        "image": refs[1]["url"],
        "reference_images": [refs[0]["url"]],
        "duration": 15,
        "resolution": "480p",
        "aspect_ratio": "3:2",
    }


def test_official_gemini_maps_three_videos_without_veo_parameters():
    refs = [ref(str(i), kind="video") for i in range(3)]
    assert mapped("gemini-omni-flash-preview-official", references=refs, resolution="720p", seconds="8") == {
        "video_list": [{"video_url": r["url"]} for r in refs],
        "resolution": "720p",
        "duration": 8,
    }


def test_happyhorse_video_edit_maps_source_and_reference_images():
    video, image = ref("source", kind="video"), ref("style")
    assert mapped("happyhorse-1.1", references=[video, image], resolution="1080p") == {
        "action": "video-edit",
        "url": video["url"],
        "reference_images": [image["url"]],
        "resolution": "1080P",
    }


def test_zexapi_frames_are_sorted_and_fixed_spec_not_forwarded():
    refs = [ref("end", role="last_frame"), ref("start", role="first_frame")]
    assert mapped(
        "omni_flash-10s-fl", "zexapi", references=refs, resolution="720p", aspect_ratio="9:16", seconds="-1"
    ) == {
        "size": "720x1280",
        "images": [refs[1]["url"], refs[0]["url"]],
    }


def test_zexapi_video_edit_uses_images_transport():
    video = ref("clip", kind="video")
    assert mapped("omni_flash-10s", "zexapi", references=[video], resolution="720p") == {"images": [video["url"]]}


@pytest.mark.parametrize(
    "model,limit", [("gemini-omni-flash", 3), ("gemini-omni-flash-preview-official", 10), ("grok-video-1.0", 8)]
)
def test_reference_image_boundary(model, limit):
    refs = [ref(str(i)) for i in range(limit)]
    assert mapped(model, references=refs)
    with pytest.raises(litellm.UnsupportedParamsError, match=f"at most {limit}"):
        mapped(model, references=refs + [ref("extra")])


INVALID_CASES = [
    ("toapis", "gemini-omni-flash", {"resolution": "1080p", "aspect_ratio": "9:16"}, "1080p only"),
    ("toapis", "gemini-omni-flash", {"resolution": "480p"}, "resolution"),
    ("toapis", "gemini-omni-flash", {"references": [ref(kind="video")]}, "video references"),
    ("toapis", "gemini-omni-flash", {"references": [ref(role="first_frame")]}, "first/last-frame"),
    ("toapis", "gemini-omni-flash-preview-official", {"resolution": "1080p"}, "resolution"),
    ("toapis", "gemini-omni-flash-preview-official", {"references": [ref(), ref(kind="video")]}, "without images"),
    (
        "toapis",
        "gemini-omni-flash-preview-official",
        {"references": [ref(kind="video")], "aspect_ratio": "16:9"},
        "cannot honor",
    ),
    ("toapis", "gemini-omni-flash-preview-official", {"references": [ref(kind="audio")]}, "audio references"),
    ("toapis", "grok-video-1.5", {"resolution": "720p", "references": []}, "exactly one"),
    ("toapis", "grok-video-1.5", {"references": [ref(), ref("two")]}, "exactly one"),
    (
        "toapis",
        "grok-video-1.5",
        {"references": [ref(role="first_frame"), ref("end", role="last_frame")]},
        "last frames",
    ),
    ("toapis", "grok-video-1.0", {"aspect_ratio": "4:3"}, "aspect_ratio"),
    ("toapis", "grok-video-1.0", {"resolution": "1080p"}, "resolution"),
    ("toapis", "happyhorse-1.1", {"references": [ref(kind="video"), ref("two", kind="video")]}, "one video"),
    (
        "toapis",
        "happyhorse-1.1",
        {"references": [ref(role="first_frame")], "aspect_ratio": "16:9"},
        "source image ratio",
    ),
    ("zexapi", "omni_flash-10s", {"resolution": "1080p"}, "720p"),
    ("zexapi", "omni_flash-10s", {"resolution": "720p", "seconds": "6"}, "10 seconds"),
    ("zexapi", "omni_flash-10s", {"aspect_ratio": "1:1"}, "aspect_ratio"),
    ("zexapi", "omni_flash-10s", {"references": [ref(str(i)) for i in range(8)]}, "at most 7"),
    ("zexapi", "omni_flash-10s", {"references": [ref(kind="audio")]}, "audio references"),
    ("zexapi", "omni_flash-10s", {"references": [ref(role="first_frame")]}, "omni_flash-10s-fl"),
    ("zexapi", "omni_flash-10s", {"references": [ref(), ref(kind="video")]}, "mixing"),
    ("zexapi", "omni_flash-10s-fl", {"references": [ref()]}, "explicit first_frame"),
    ("zexapi", "omni_flash-10s-fl", {"resolution": "720p"}, "explicit first_frame"),
    ("zexapi", "omni_flash-10s", {"resolution": "720p", "images": [ref()["url"]]}, "cannot be combined"),
]


@pytest.mark.asyncio
@pytest.mark.parametrize("provider,model,params,error", INVALID_CASES)
async def test_invalid_combination_stops_before_submission_even_with_drop_params(
    provider, model, params, error, respx_mock, monkeypatch
):
    monkeypatch.setattr(litellm, "drop_params", True)
    candidates = [
        {
            "model_name": "public-video",
            "litellm_params": {"model": f"{provider}/{model}"},
            "model_info": {"id": "fixture"},
        }
    ]
    with pytest.raises(litellm.BadRequestError, match=error):
        Router._filter_deployments_by_video_generation_params(
            "public-video", candidates, {"_router_call_type": "avideo_generation", **params}
        )
    with pytest.raises(litellm.UnsupportedParamsError, match=error):
        mapped(model, provider, **params)
    assert not respx_mock.calls


@pytest.mark.parametrize(
    "model,seconds",
    [
        ("gemini-omni-flash", "8"),
        ("gemini-omni-flash-preview-official", "11"),
        ("grok-video-1.0", "16"),
        ("grok-video-1.5", "-1"),
        ("seedance-2", "3"),
        ("seedance-2-fast", "16"),
        ("seedance-2-mini", "-1"),
        ("seedance-2-5", "31"),
        ("wan3.0-video", "1"),
        ("happyhorse-1.1", "2"),
        ("kling-v3", "16"),
    ],
)
def test_documented_duration_limits(model, seconds):
    with pytest.raises(litellm.UnsupportedParamsError, match="seconds"):
        mapped(model, resolution="720p", seconds=seconds)


@pytest.mark.parametrize("provider,model", [("toapis", "gemini-omni-flash"), ("zexapi", "omni_flash-10s")])
def test_canonical_mapping_rejects_native_overrides(provider, model):
    for extra in [{"image_urls": [ref()["url"]]}, {"resolution": "1080p"}, {"duration": 20}]:
        with pytest.raises(litellm.UnsupportedParamsError, match="override"):
            mapped(model, provider, resolution="720p", extra_body=extra)


def test_unadapted_provider_model_is_not_accidentally_enabled():
    with pytest.raises(litellm.UnsupportedParamsError, match="no documented mapping"):
        mapped("veo_3_1-lite", "zexapi", resolution="720p")


def test_happyhorse_edit_source_cannot_be_overridden():
    params = {"references": [ref(kind="video")], "resolution": "720p"}
    with pytest.raises(litellm.UnsupportedParamsError, match="override"):
        mapped("happyhorse-1.1", **params, extra_body={"url": "https://media.example/wrong-source"})
    with pytest.raises(litellm.UnsupportedParamsError, match="overrides"):
        mapped("happyhorse-1.1", **params, url="https://media.example/wrong-source")


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "provider,model,params",
    [
        ("toapis", "gemini-omni-flash-preview-official", {"resolution": "1080p"}),
        ("toapis", "grok-video-1.5", {"resolution": "720p", "references": []}),
        ("zexapi", "omni_flash-10s-fl", {"resolution": "720p", "references": [ref()]}),
    ],
)
async def test_public_router_invalid_request_never_creates_task(provider, model, params, respx_mock):
    router = Router(
        model_list=[
            {
                "model_name": "public-video",
                "litellm_params": {
                    "model": f"{provider}/{model}",
                    "api_key": "fixture-key",
                    "api_base": "https://provider-fixture.example/v1",
                },
            }
        ],
        num_retries=0,
    )
    with pytest.raises(litellm.BadRequestError, match="no deployment supporting video parameters"):
        await router.avideo_generation(model="public-video", prompt="invalid fixture", **params)
    assert not respx_mock.calls


@pytest.mark.parametrize(
    "provider,model,expected",
    [
        ("toapis", "gemini-omni-flash", {"resolution": "720p"}),
        ("toapis", "gemini-omni-flash-preview-official", {"resolution": "720p"}),
        ("toapis", "grok-video-1.0", {"resolution": "720p"}),
        ("zexapi", "omni_flash-10s", {}),
    ],
)
def test_text_to_video_without_references(provider, model, expected):
    assert mapped(model, provider, resolution="720p", references=[]) == expected
