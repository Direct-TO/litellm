"""Unified operation semantics and rejection before any paid submission."""

import pytest

import litellm
from litellm.llms.openai.videos.transformation import OpenAIVideoConfig
from litellm.llms.toapis.videos.transformation import ToAPISVideoConfig
from litellm.router import Router
from litellm.videos.utils import VideoGenerationRequestUtils

VIDEO = {"type": "video", "role": "reference", "url": "https://media.example/source.mp4"}


def mapped(model="seedance-2-5", **params):
    return VideoGenerationRequestUtils.get_optional_params_video_generation(model, ToAPISVideoConfig(), params)


@pytest.mark.parametrize("operation", [None, "generate"])
def test_default_generation_with_video_is_never_edit(operation):
    params = {"references": [VIDEO], "seconds": "8", "aspect_ratio": "16:9"}
    if operation is not None:
        params["operation"] = operation
    result = mapped(**params)
    assert result == {
        "video_operation": "generate",
        "duration": 8,
        "aspect_ratio": "16:9",
        "video_with_roles": [{"url": VIDEO["url"], "role": "reference_video"}],
    }


@pytest.mark.parametrize("operation", ["edit", "extend"])
@pytest.mark.parametrize("seconds", [None, "-1"])
def test_automatic_duration_and_ratio(operation, seconds):
    params = {"operation": operation, "references": [VIDEO]}
    if seconds is not None:
        params["seconds"] = seconds
    assert mapped(**params) == {
        "video_operation": operation,
        "duration": -1,
        "aspect_ratio": "adaptive",
        "video_with_roles": [{"url": VIDEO["url"], "role": "reference_video"}],
    }


@pytest.mark.parametrize("seconds", ["4", "15", "30"])
def test_extend_duration_is_output_total_without_adding_source_duration(seconds):
    result = mapped(operation="extend", seconds=seconds, references=[VIDEO])
    assert result["duration"] == int(seconds)


INVALID = [
    ({"operation": "remix"}, "operation"),
    ({"operation": ""}, "operation"),
    ({"operation": ["edit"]}, "operation"),
    ({"operation": "edit"}, "requires at least one video"),
    ({"operation": "extend", "references": []}, "requires at least one video"),
    ({"operation": "edit", "references": [{**VIDEO, "type": "image"}]}, "requires at least one video"),
    ({"operation": "edit", "references": [VIDEO], "seconds": "5"}, "automatic seconds"),
    ({"operation": "extend", "references": [VIDEO], "seconds": "3"}, "seconds"),
    ({"operation": "extend", "references": [VIDEO], "seconds": "31"}, "seconds"),
    ({"operation": "extend", "references": [VIDEO], "seconds": "4.5"}, "seconds"),
    ({"operation": "edit", "references": [VIDEO], "aspect_ratio": "16:9"}, "automatic aspect_ratio"),
    ({"operation": "extend", "references": [VIDEO], "aspect_ratio": "16:9"}, "automatic aspect_ratio"),
    ({"operation": "edit", "references": [VIDEO], "resolution": "4K"}, "resolution"),
    ({"operation": "edit", "references": [VIDEO, {**VIDEO, "type": "image", "role": "first_frame"}]}, "mixed"),
    ({"operation": "extend", "references": [VIDEO] * 11}, "count"),
    ({"operation": "edit", "references": [VIDEO], "tools": [{"type": "web_search"}]}, "tools"),
    ({"operation": "edit", "references": [VIDEO], "video_operation": "generate"}, "overrides"),
    ({"operation": "extend", "references": [VIDEO], "duration": 5}, "overrides"),
    ({"operation": "edit", "references": [VIDEO], "action": "text-to-video"}, "overrides"),
    ({"operation": "edit", "references": [VIDEO], "extra_body": {"video_operation": "generate"}}, "override"),
    ({"operation": "extend", "references": [VIDEO], "extra_body": {"duration": 5}}, "override"),
    ({"extra_body": {"operation": "edit"}}, "top-level"),
]


@pytest.mark.parametrize("params,error", INVALID)
def test_invalid_operations_rejected_by_router_and_sender_with_drop_params(params, error, monkeypatch, respx_mock):
    monkeypatch.setattr(litellm, "drop_params", True)
    candidates = [{"model_name": "video", "litellm_params": {"model": "toapis/seedance-2-5"}}]
    with pytest.raises(litellm.BadRequestError, match=error):
        Router._filter_deployments_by_video_generation_params(
            "video", candidates, {"_router_call_type": "avideo_generation", **params}
        )
    with pytest.raises(litellm.UnsupportedParamsError, match=error):
        mapped(**params)
    assert not respx_mock.calls


@pytest.mark.parametrize("operation", ["edit", "extend"])
def test_router_excludes_same_alias_deployments_without_operation_support(operation):
    candidates = [
        {"model_name": "video", "litellm_params": {"model": model}}
        for model in ("toapis/seedance-2", "zexapi/omni_flash-10s", "toapis/seedance-2-5")
    ]
    params = {"_router_call_type": "avideo_generation", "operation": operation, "references": [VIDEO]}
    assert Router._filter_deployments_by_video_generation_params("video", candidates, params) == candidates[2:]
    with pytest.raises(litellm.BadRequestError, match="no deployment"):
        Router._filter_deployments_by_video_generation_params("video", candidates[:2], params)


def test_router_checks_native_operation_in_deployment_defaults():
    good = {"model_name": "video", "litellm_params": {"model": "toapis/seedance-2-5"}}
    bad = {**good, "litellm_params": {**good["litellm_params"], "video_operation": "generate"}}
    params = {"_router_call_type": "avideo_generation", "operation": "edit", "references": [VIDEO]}
    assert Router._filter_deployments_by_video_generation_params("video", [bad, good], params) == [good]


@pytest.mark.parametrize("model", ["happyhorse-1.1", "gemini-omni-flash-preview-official"])
@pytest.mark.parametrize("params", [{}, {"operation": "generate"}])
def test_existing_edit_models_require_explicit_edit(model, params):
    with pytest.raises(litellm.UnsupportedParamsError, match="requires operation='edit'"):
        mapped(model, references=[VIDEO], **params)
    assert mapped(model, operation="edit", references=[VIDEO])
    with pytest.raises(litellm.UnsupportedParamsError, match="not supported"):
        mapped(model, operation="extend", references=[VIDEO])


def test_generate_on_unadapted_provider_is_default_and_not_forwarded():
    result = VideoGenerationRequestUtils.get_optional_params_video_generation(
        "sora-2", OpenAIVideoConfig(), {"operation": "generate", "seconds": "8"}
    )
    assert "operation" not in result
    with pytest.raises(litellm.UnsupportedParamsError, match="no documented mapping"):
        VideoGenerationRequestUtils.get_optional_params_video_generation(
            "sora-2", OpenAIVideoConfig(), {"operation": "edit", "references": [VIDEO]}
        )


@pytest.mark.parametrize("extra", [{"operation": "edit"}, {"operation": "generate"}])
def test_request_extraction_cannot_hide_operation_in_extra_body(extra):
    params = VideoGenerationRequestUtils.get_requested_video_generation_optional_param({"extra_body": extra})
    with pytest.raises(litellm.UnsupportedParamsError, match="top-level"):
        mapped(**params)
