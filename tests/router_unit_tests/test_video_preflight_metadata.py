"""Proxy logging metadata must not make valid video deployments ineligible."""

from copy import deepcopy
from importlib.util import find_spec

import pytest

import litellm
from litellm.router import Router

MODELS = [
    "toapis/seedance-2-fast",
    pytest.param(
        "guanghe/seedance2.0-fast（企业折扣）",
        marks=pytest.mark.skipif(
            find_spec("litellm.llms.guanghe") is None,
            reason="Guanghe integration is not present in this checkout",
        ),
    ),
]


def video_request(call_type="avideo_generation"):
    return {
        "_router_call_type": call_type,
        "seconds": "5",
        "resolution": "480p",
        "aspect_ratio": "1:1",
        "generate_audio": True,
        "references": [{"type": "image", "role": "reference", "url": "https://media.example/reference.png"}],
    }


@pytest.mark.parametrize("call_type", ["video_generation", "avideo_generation"])
@pytest.mark.parametrize("metadata", [{}, {"headers": {"user-agent": "undici"}, "queue_time_seconds": 0.03}])
@pytest.mark.parametrize("metadata_source", ["request", "deployment"])
@pytest.mark.parametrize("model", MODELS)
def test_router_preserves_video_candidates_and_logging_metadata(call_type, metadata, metadata_source, model):
    candidates = [{"model_name": "video", "litellm_params": {"model": model}}]
    request = video_request(call_type)
    if metadata_source == "request":
        request["metadata"] = deepcopy(metadata)
    else:
        for candidate in candidates:
            candidate["litellm_params"]["metadata"] = deepcopy(metadata)
    original = deepcopy((candidates, request))

    assert Router._filter_deployments_by_video_generation_params("video", candidates, request) == candidates
    for candidate in candidates:
        assert Router._filter_deployments_by_video_generation_params("video", candidate, request) == candidate
    assert (candidates, request) == original


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize(
    "overrides,error",
    [
        ({"resolution": "8K"}, "resolution"),
        ({"image_with_roles": [{"url": "https://media.example/native.png"}]}, "overrides|unsupported parameters"),
        ({"extra_body": {"metadata": {"resolution": "1080p"}}}, "override"),
        ({"extra_body": {"duration": 99}}, "override"),
    ],
)
def test_logging_metadata_does_not_bypass_video_parameter_validation(model, overrides, error, monkeypatch):
    monkeypatch.setattr(litellm, "drop_params", True)
    candidate = {"model_name": "video", "litellm_params": {"model": model}}
    request = {**video_request(), "metadata": {"session_id": "fixture"}, **overrides}

    with pytest.raises(litellm.BadRequestError, match=error):
        Router._filter_deployments_by_video_generation_params("video", [candidate], request)
