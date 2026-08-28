import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.openai_like.json_loader import JSONProviderRegistry
from litellm.llms.zexapi.image_edit.transformation import ZexAPIImageEditConfig
from litellm.llms.zexapi.image_generation.transformation import ZexAPIImageGenerationConfig
from litellm.utils import ProviderConfigManager


def test_zexapi_provider_registration_and_resolution():
    model, provider, _, api_base = get_llm_provider("zexapi/gpt-image2")

    assert model == "gpt-image2"
    assert provider == "zexapi"
    assert api_base == "https://zexapi.com/v1"
    assert JSONProviderRegistry.exists("zexapi")
    assert litellm.LlmProviders.ZEXAPI.value == "zexapi"
    assert "zexapi" in litellm.provider_list
    assert "zexapi" not in litellm.openai_compatible_providers


def test_zexapi_media_configs_are_registered():
    assert isinstance(
        ProviderConfigManager.get_provider_image_generation_config("gpt-image2", litellm.LlmProviders.ZEXAPI),
        ZexAPIImageGenerationConfig,
    )
    assert isinstance(
        ProviderConfigManager.get_provider_image_edit_config("gpt-image2", litellm.LlmProviders.ZEXAPI),
        ZexAPIImageEditConfig,
    )
    model_cost = litellm.get_model_cost_map(url="")

    assert model_cost["zexapi/gpt-image2"]["mode"] == "image_generation"
    assert model_cost["zexapi/sora-2-12s"]["mode"] == "video_generation"


def test_zexapi_chat_completion_uses_provider_url_and_key(respx_mock, monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route = respx_mock.post("https://zexapi.com/v1/chat/completions").respond(
        json={
            "id": "chatcmpl-image-123",
            "object": "chat.completion",
            "created": 1784359143,
            "model": "gpt-image2",
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
        model="zexapi/gpt-image2",
        messages=[{"role": "user", "content": "poster"}],
        api_key="test-key",
    )

    assert response.choices[0].message.content == "![image](https://files.example/image.png)"
    assert route.calls[0].request.headers["Authorization"] == "Bearer test-key"
