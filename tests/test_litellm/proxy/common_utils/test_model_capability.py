from unittest.mock import MagicMock, Mock

import pytest

from litellm.proxy.common_utils.model_capability import (
    generation_only_model_listing_enabled,
    resolve_model_capability,
)


def _deployment(mode: str | None = None) -> dict[str, object]:
    return {
        "model_name": "public-model",
        "litellm_params": {"model": "provider/backend-model"},
        "model_info": {"mode": mode} if mode is not None else {},
    }


@pytest.mark.parametrize(
    ("mode", "expected"),
    [
        ("chat", "text"),
        ("completion", "text"),
        ("image_generation", "image"),
        ("video_generation", "video"),
        ("audio_speech", "audio"),
        ("audio_transcription", None),
        ("responses", None),
        ("image_edit", None),
        ("embedding", None),
        ("rerank", None),
        ("ocr", None),
    ],
)
def test_resolve_model_capability_from_cost_map(mode, expected):
    assert (
        resolve_model_capability(
            model_name="public-model",
            llm_router=None,
            get_model_info=lambda _model: {"mode": mode},
        )
        == expected
    )


def test_configured_deployment_mode_takes_precedence():
    router = MagicMock()
    router.get_model_list.return_value = [_deployment(mode="video_generation")]
    get_model_info = Mock()

    assert resolve_model_capability("public-model", router, get_model_info=get_model_info) == "video"
    get_model_info.assert_not_called()


def test_backend_cost_map_resolves_custom_alias_capability():
    router = MagicMock()
    deployment = _deployment()
    router.get_model_list.return_value = [deployment]
    get_model_info = Mock(return_value={"mode": "image_generation"})

    assert resolve_model_capability("public-model", router, get_model_info=get_model_info) == "image"
    get_model_info.assert_called_once_with("provider/backend-model")


def test_same_capability_across_deployments_is_publishable():
    router = MagicMock()
    router.get_model_list.return_value = [
        _deployment(mode="chat"),
        _deployment(mode="completion"),
    ]

    assert resolve_model_capability("public-model", router) == "text"


@pytest.mark.parametrize(
    "modes",
    [
        ("chat", "image_generation"),
        ("image_generation", "image_edit"),
        ("image_generation", "embedding"),
    ],
)
def test_conflicting_or_non_generation_deployment_fails_closed(modes):
    router = MagicMock()
    router.get_model_list.return_value = [_deployment(mode=mode) for mode in modes]

    assert resolve_model_capability("public-model", router) is None


@pytest.mark.parametrize(
    ("settings", "expected"),
    [
        ({"model_list_generation_only": True}, True),
        ({"model_list_generation_only": False}, False),
        ({"model_list_generation_only": "true"}, False),
        ({}, False),
    ],
)
def test_generation_only_model_listing_requires_strict_boolean(settings, expected):
    assert generation_only_model_listing_enabled(settings) is expected
