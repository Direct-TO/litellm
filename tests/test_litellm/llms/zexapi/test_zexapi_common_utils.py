import httpx
import pytest

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.submission_utils import get_submission_outcome
from litellm.llms.openai_like.json_loader import JSONProviderRegistry
from litellm.llms.zexapi.common_utils import parse_zexapi_task, raise_for_zexapi_error
from litellm.llms.zexapi.image_edit.transformation import ZexAPIImageEditConfig
from litellm.llms.zexapi.image_generation.transformation import (
    ZexAPIBananaImageGenerationConfig,
    ZexAPIImageGenerationConfig,
)
from litellm.utils import ProviderConfigManager


def test_zexapi_provider_registration_and_resolution():
    model, provider, _, api_base = get_llm_provider("zexapi/image2")

    assert model == "image2"
    assert provider == "zexapi"
    assert api_base == "https://zexapi.com/v1"
    assert JSONProviderRegistry.exists("zexapi")
    assert litellm.LlmProviders.ZEXAPI.value == "zexapi"
    assert "zexapi" in litellm.provider_list
    assert "zexapi" not in litellm.openai_compatible_providers


def test_zexapi_http_rejection_and_invalid_acceptance_have_distinct_submission_outcomes():
    with pytest.raises(BaseLLMException) as rejected:
        raise_for_zexapi_error(httpx.Response(429, text="rate limited"))
    with pytest.raises(BaseLLMException) as rejected_channel:
        raise_for_zexapi_error(httpx.Response(503, text="No available channel for model"))
    with pytest.raises(BaseLLMException) as ambiguous_503:
        raise_for_zexapi_error(httpx.Response(503, text="service temporarily unavailable"))
    with pytest.raises(BaseLLMException) as unknown:
        parse_zexapi_task(httpx.Response(200, json={"object": "video", "status": "queued"}))

    assert get_submission_outcome(rejected.value) == "rejected"
    assert get_submission_outcome(rejected_channel.value) == "rejected"
    assert get_submission_outcome(ambiguous_503.value) == "unknown"
    assert get_submission_outcome(unknown.value) == "unknown"


def test_zexapi_media_configs_are_registered():
    assert isinstance(
        ProviderConfigManager.get_provider_image_generation_config("image2", litellm.LlmProviders.ZEXAPI),
        ZexAPIImageGenerationConfig,
    )
    assert isinstance(
        ProviderConfigManager.get_provider_image_edit_config("image2", litellm.LlmProviders.ZEXAPI),
        ZexAPIImageEditConfig,
    )
    assert isinstance(
        ProviderConfigManager.get_provider_image_generation_config("gpt-image2", litellm.LlmProviders.ZEXAPI),
        ZexAPIImageGenerationConfig,
    )
    assert isinstance(
        ProviderConfigManager.get_provider_image_edit_config("gpt-image2", litellm.LlmProviders.ZEXAPI),
        ZexAPIImageEditConfig,
    )
    assert ProviderConfigManager.get_provider_image_edit_config("unknown-image", litellm.LlmProviders.ZEXAPI) is None
    assert isinstance(
        ProviderConfigManager.get_provider_image_generation_config(
            "gemini-3.1-flash-image-preview", litellm.LlmProviders.ZEXAPI
        ),
        ZexAPIBananaImageGenerationConfig,
    )
    model_cost = litellm.get_model_cost_map(url="")

    assert model_cost["zexapi/image2"]["mode"] == "image_generation"
    assert model_cost["zexapi/gpt-image2"]["mode"] == "image_generation"
    assert model_cost["zexapi/gemini-3.1-flash-image-preview"]["mode"] == "image_generation"
    assert model_cost["zexapi/omni_flash-10s"]["mode"] == "video_generation"
    assert model_cost["zexapi/veo_3_1-fast"]["mode"] == "video_generation"


def test_zexapi_chat_completion_uses_provider_url_and_key(respx_mock, monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route = respx_mock.post("https://zexapi.com/v1/chat/completions").respond(
        json={
            "id": "chatcmpl-image-123",
            "object": "chat.completion",
            "created": 1784359143,
            "model": "image2",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "![image](https://files.example/image.png)"},
                    "finish_reason": "stop",
                }
            ],
        }
    )

    response = litellm.completion(
        model="zexapi/image2",
        messages=[{"role": "user", "content": "poster"}],
        api_key="test-key",
    )

    assert response.choices[0].message.content == "![image](https://files.example/image.png)"
    assert route.calls[0].request.headers["Authorization"] == "Bearer test-key"
