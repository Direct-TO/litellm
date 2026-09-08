import pytest

import litellm
from litellm import Router


def _image_model_list(
    include_zexapi: bool = True,
    include_zexapi_highres: bool = False,
) -> list[dict]:
    deployments = [
        {
            "model_name": "gpt-image-2",
            "litellm_params": {
                "model": "toapis/gpt-image-2",
                "api_key": "fake-toapis-key",
            },
            "model_info": {
                "id": "toapis-image",
                "supported_endpoints": ["/v1/images/generations"],
            },
        }
    ]
    if include_zexapi:
        deployments.insert(
            0,
            {
                "model_name": "gpt-image-2",
                "litellm_params": {
                    "model": "zexapi/image2",
                    "api_key": "fake-zexapi-key",
                },
                "model_info": {
                    "id": "zexapi-image",
                    "supported_endpoints": ["/v1/images/generations", "/v1/images/edits"],
                },
            },
        )
    if include_zexapi_highres:
        deployments.insert(
            1 if include_zexapi else 0,
            {
                "model_name": "gpt-image-2",
                "litellm_params": {
                    "model": "zexapi/gpt-image2",
                    "api_key": "fake-zexapi-key",
                },
                "model_info": {
                    "id": "zexapi-gpt-image",
                    "supported_endpoints": ["/v1/images/generations", "/v1/images/edits"],
                },
            },
        )
    return deployments


def _banana_model_list() -> list[dict]:
    return [
        {
            "model_name": "gemini-3.1-flash-image-preview",
            "litellm_params": {
                "model": "toapis/gemini-3.1-flash-image-preview",
                "api_key": "fake-toapis-key",
            },
            "model_info": {
                "id": "toapis-banana",
                "supported_endpoints": ["/v1/images/generations"],
            },
        },
        {
            "model_name": "gemini-3.1-flash-image-preview",
            "litellm_params": {
                "model": "zexapi/gemini-3.1-flash-image-preview",
                "api_key": "fake-zexapi-key",
            },
            "model_info": {
                "id": "zexapi-banana",
                "supported_endpoints": ["/v1/images/generations"],
            },
        },
    ]


def _mixed_media_model_list() -> list[dict]:
    return [
        {
            "model_name": "mixed-media",
            "litellm_params": {"model": "toapis/gpt-image-2", "api_key": "fake-toapis-key"},
            "model_info": {
                "id": "image-only",
                "supported_endpoints": ["/v1/images/generations"],
            },
        },
        {
            "model_name": "mixed-media",
            "litellm_params": {"model": "toapis/seedance-2-5", "api_key": "fake-toapis-key"},
            "model_info": {"id": "video-only", "supported_endpoints": ["/v1/videos"]},
        },
    ]


def test_video_generation_call_type_filters_out_image_only_deployment():
    deployments = _mixed_media_model_list()

    filtered = Router._filter_deployments_by_supported_endpoint(
        model="mixed-media",
        healthy_deployments=deployments,
        request_kwargs={"_router_call_type": "avideo_generation"},
    )

    assert isinstance(filtered, list)
    assert [deployment["model_info"]["id"] for deployment in filtered] == ["video-only"]


@pytest.mark.asyncio
async def test_image_edit_router_selects_only_edit_capable_deployment(monkeypatch):
    captured: dict[str, object] = {}

    async def fake_image_edit(**kwargs):
        captured.update(kwargs)
        return litellm.ImageResponse(data=[])

    monkeypatch.setattr(litellm, "aimage_edit", fake_image_edit)
    router = Router(model_list=_image_model_list())

    await router.aimage_edit(model="gpt-image-2", image=b"image", prompt="edit")

    assert captured["model"] == "zexapi/image2"
    assert captured["custom_llm_provider"] == "zexapi"
    assert "_router_call_type" not in captured


@pytest.mark.asyncio
async def test_image_edit_router_rejects_group_without_edit_capable_deployment(monkeypatch):
    async def fake_image_edit(**kwargs):
        raise AssertionError("unsupported deployment must be rejected before provider dispatch")

    monkeypatch.setattr(litellm, "aimage_edit", fake_image_edit)
    router = Router(model_list=_image_model_list(include_zexapi=False))

    with pytest.raises(litellm.BadRequestError, match="no deployment supporting endpoint /v1/images/edits"):
        await router.aimage_edit(model="gpt-image-2", image=b"image", prompt="edit")


@pytest.mark.asyncio
async def test_image_edit_router_uses_base_model_catalog_for_endpoint_capability(monkeypatch):
    async def fake_image_edit(**kwargs):
        raise AssertionError("generation-only base model must be rejected before provider dispatch")

    monkeypatch.setattr(litellm, "aimage_edit", fake_image_edit)
    router = Router(
        model_list=[
            {
                "model_name": "custom-image-deployment",
                "litellm_params": {
                    "model": "toapis/custom-image-deployment",
                    "api_key": "fake-toapis-key",
                },
                "model_info": {
                    "id": "custom-image",
                    "base_model": "toapis/gpt-image-2",
                },
            }
        ]
    )

    with pytest.raises(litellm.BadRequestError, match="no deployment supporting endpoint /v1/images/edits"):
        await router.aimage_edit(model="custom-image-deployment", image=b"image", prompt="edit")


@pytest.mark.asyncio
async def test_image_generation_router_selects_deployment_supporting_requested_size(monkeypatch):
    captured: dict[str, object] = {}

    async def fake_image_generation(**kwargs):
        captured.update(kwargs)
        return litellm.ImageResponse(data=[])

    monkeypatch.setattr(litellm, "aimage_generation", fake_image_generation)
    router = Router(model_list=_image_model_list())

    await router.aimage_generation(model="gpt-image-2", prompt="poster", size="2:1")

    assert captured["model"] == "toapis/gpt-image-2"
    assert "_router_call_type" not in captured


@pytest.mark.asyncio
async def test_image_generation_router_selects_only_deployment_supporting_resolution(monkeypatch):
    captured: dict[str, object] = {}

    async def fake_image_generation(**kwargs):
        captured.update(kwargs)
        return litellm.ImageResponse(data=[])

    monkeypatch.setattr(litellm, "aimage_generation", fake_image_generation)
    router = Router(model_list=_image_model_list())

    await router.aimage_generation(model="gpt-image-2", prompt="poster", resolution="1K")

    assert captured["model"] in {"zexapi/image2", "toapis/gpt-image-2"}


@pytest.mark.asyncio
async def test_image_edit_router_selects_high_resolution_deployment(monkeypatch):
    captured: dict[str, object] = {}

    async def fake_image_edit(**kwargs):
        captured.update(kwargs)
        return litellm.ImageResponse(data=[])

    monkeypatch.setattr(litellm, "aimage_edit", fake_image_edit)
    router = Router(model_list=_image_model_list(include_zexapi_highres=True))

    await router.aimage_edit(
        model="gpt-image-2",
        image=b"image",
        prompt="edit",
        size="2048x2048",
    )

    assert captured["model"] == "zexapi/gpt-image2"


@pytest.mark.asyncio
@pytest.mark.parametrize("drop_params", [False, True])
async def test_image_edit_router_uses_tier_to_exclude_1k_deployment(monkeypatch, drop_params):
    captured = []

    async def fake_image_edit(**kwargs):
        captured.append(kwargs)
        return litellm.ImageResponse(data=[])

    monkeypatch.setattr(litellm, "aimage_edit", fake_image_edit)
    router = Router(model_list=_image_model_list(include_zexapi_highres=True), num_retries=0)
    await router.aimage_edit(
        model="gpt-image-2",
        image=b"image",
        prompt="edit",
        aspect_ratio="16:9",
        resolution="2K",
        drop_params=drop_params,
    )
    assert len(captured) == 1
    assert captured[0]["model"] == "zexapi/gpt-image2"
    assert captured[0]["aspect_ratio"] == "16:9"
    assert captured[0]["resolution"] == "2K"


@pytest.mark.asyncio
async def test_image_edit_router_rejects_2k_without_a_compatible_deployment(monkeypatch):
    called = []

    async def fake_image_edit(**kwargs):
        called.append(kwargs)
        return litellm.ImageResponse(data=[])

    monkeypatch.setattr(litellm, "aimage_edit", fake_image_edit)
    router = Router(model_list=_image_model_list(), num_retries=0)
    with pytest.raises(litellm.BadRequestError, match="image edit parameter"):
        await router.aimage_edit(
            model="gpt-image-2", image=b"image", prompt="edit", aspect_ratio="16:9", resolution="2K", drop_params=True
        )
    assert called == []


@pytest.mark.asyncio
async def test_image_generation_router_rejects_deployment_that_would_drop_nested_image_config(monkeypatch):
    captured: dict[str, object] = {}

    async def fake_image_generation(**kwargs):
        captured.update(kwargs)
        return litellm.ImageResponse(data=[])

    monkeypatch.setattr(litellm, "aimage_generation", fake_image_generation)
    router = Router(model_list=_banana_model_list())

    await router.aimage_generation(
        model="gemini-3.1-flash-image-preview",
        prompt="poster",
        imageConfig={"imageSize": "2K", "google_search": True},
    )

    assert captured["model"] == "toapis/gemini-3.1-flash-image-preview"


def test_sync_image_generation_router_applies_parameter_preflight(monkeypatch):
    captured: dict[str, object] = {}

    def fake_image_generation(**kwargs):
        captured.update(kwargs)
        return litellm.ImageResponse(data=[])

    monkeypatch.setattr(litellm, "image_generation", fake_image_generation)
    router = Router(model_list=_image_model_list())

    router.image_generation(model="gpt-image-2", prompt="poster", size="2:1")

    assert captured["model"] == "toapis/gpt-image-2"
    assert "_router_call_type" not in captured


@pytest.mark.asyncio
@pytest.mark.parametrize(
    "request_params",
    [
        {"resolution": "2K"},
        {"resolution": "4K"},
        {"imageConfig": {"imageSize": "2K"}},
        {"imageConfig": {"imageSize": "4K"}},
    ],
)
async def test_image_generation_router_routes_high_resolution_to_compatible_deployment(
    monkeypatch,
    request_params,
):
    captured: dict[str, object] = {}

    async def fake_image_generation(**kwargs):
        captured.update(kwargs)
        return litellm.ImageResponse(data=[])

    monkeypatch.setattr(litellm, "aimage_generation", fake_image_generation)
    router = Router(model_list=_image_model_list())

    await router.aimage_generation(model="gpt-image-2", prompt="poster", **request_params)

    assert captured["model"] == "toapis/gpt-image-2"


def test_image_generation_router_excludes_low_resolution_zexapi_deployment():
    deployments = _image_model_list(include_zexapi_highres=True)

    filtered = Router._filter_deployments_by_image_generation_params(
        model="gpt-image-2",
        healthy_deployments=deployments,
        request_kwargs={"_router_call_type": "aimage_generation", "resolution": "4K"},
    )

    assert isinstance(filtered, list)
    assert {deployment["litellm_params"]["model"] for deployment in filtered} == {
        "zexapi/gpt-image2",
        "toapis/gpt-image-2",
    }


@pytest.mark.parametrize(
    "pixel_size,expected_model",
    [
        ("2048x1152", "toapis/gpt-image-2"),
        ("2560x1440", "zexapi/gpt-image2"),
    ],
)
def test_image_generation_router_preserves_exact_provider_pixel_contract(pixel_size, expected_model):
    deployments = _image_model_list(include_zexapi_highres=True)

    filtered = Router._filter_deployments_by_image_generation_params(
        model="gpt-image-2",
        healthy_deployments=deployments,
        request_kwargs={"_router_call_type": "aimage_generation", "n": 1, "size": pixel_size},
    )

    assert isinstance(filtered, list)
    assert [deployment["litellm_params"]["model"] for deployment in filtered] == [expected_model]


@pytest.mark.asyncio
async def test_image_generation_router_honors_request_level_drop_params(monkeypatch):
    captured: dict[str, object] = {}

    async def fake_image_generation(**kwargs):
        captured.update(kwargs)
        return litellm.ImageResponse(data=[])

    monkeypatch.setattr(litellm, "aimage_generation", fake_image_generation)
    router = Router(model_list=_image_model_list(include_zexapi=False))

    await router.aimage_generation(
        model="gpt-image-2",
        prompt="poster",
        quality="high",
        drop_params=True,
    )

    assert captured["model"] == "toapis/gpt-image-2"


@pytest.mark.asyncio
async def test_image_generation_router_honors_additional_drop_params(monkeypatch):
    captured: dict[str, object] = {}

    async def fake_image_generation(**kwargs):
        captured.update(kwargs)
        return litellm.ImageResponse(data=[])

    monkeypatch.setattr(litellm, "aimage_generation", fake_image_generation)
    router = Router(model_list=_image_model_list(include_zexapi=True, include_zexapi_highres=False))

    await router.aimage_generation(
        model="gpt-image-2",
        prompt="poster",
        resolution="4K",
        additional_drop_params=["resolution"],
    )

    assert captured["model"] in {"zexapi/image2", "toapis/gpt-image-2"}
