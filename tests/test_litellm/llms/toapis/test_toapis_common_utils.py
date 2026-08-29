import httpx
import pytest

import litellm
from litellm.litellm_core_utils.get_llm_provider_logic import get_llm_provider
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.openai_like.json_loader import JSONProviderRegistry
from litellm.llms.toapis.common_utils import ToAPISModelInfo, parse_toapis_task
from litellm.utils import ProviderConfigManager


def _task_response(status: str) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": "task_123",
            "object": "generation.task",
            "model": "gpt-image-2",
            "status": status,
            "progress": 0,
            "created_at": 1703884800,
        },
    )


def test_toapis_provider_registration_and_resolution():
    model, provider, _, api_base = get_llm_provider("toapis/gpt-5.6-terra")

    assert model == "gpt-5.6-terra"
    assert provider == "toapis"
    assert api_base == "https://toapis.com/v1"
    assert litellm.LlmProviders.TOAPIS.value == "toapis"
    assert "toapis" in litellm.provider_list
    assert "toapis" not in litellm.openai_compatible_providers


def test_toapis_pending_task_status_normalizes_to_queued():
    task = parse_toapis_task(_task_response("pending"))

    assert task.status == "queued"


def test_toapis_unknown_task_status_is_rejected():
    with pytest.raises(BaseLLMException, match="Invalid ToAPIs task response") as exc_info:
        parse_toapis_task(_task_response("mystery"))

    assert exc_info.value.status_code == 502


def test_toapis_json_config_supports_responses():
    provider = JSONProviderRegistry.get("toapis")

    assert provider is not None
    assert provider.api_key_env == "TOAPIS_API_KEY"
    assert provider.api_base_env == "TOAPIS_API_BASE"
    assert JSONProviderRegistry.supports_responses_api("toapis") is True

    responses_config = ProviderConfigManager.get_provider_responses_api_config(
        provider="toapis",
        model="toapis/gpt-5.3-codex-official",
    )

    assert responses_config is not None
    assert responses_config.get_complete_url(None, {}) == "https://toapis.com/v1/responses"


def test_toapis_dynamic_model_listing_uses_provider_credentials(respx_mock, monkeypatch):
    monkeypatch.setenv("TOAPIS_API_KEY", "test-key")
    route = respx_mock.get(
        "https://toapis.com/v1/models",
        headers={"Authorization": "Bearer test-key"},
    ).respond(
        json={
            "success": True,
            "object": "list",
            "data": [
                {"id": "gpt-5.6-terra", "object": "model"},
                {"id": "claude-sonnet-4-6", "object": "model"},
            ],
        }
    )

    models = ToAPISModelInfo().get_models()

    assert models == ["gpt-5.6-terra", "claude-sonnet-4-6"]
    assert route.called


def test_toapis_get_valid_models_uses_dynamic_model_info(respx_mock):
    respx_mock.get("https://toapis.com/v1/models").respond(
        json={"success": True, "object": "list", "data": [{"id": "gpt-5.6-terra"}]}
    )

    models = litellm.get_valid_models(
        check_provider_endpoint=True,
        custom_llm_provider="toapis",
        api_key="model-list-test-key",
    )

    assert models == ["gpt-5.6-terra"]


def test_toapis_media_models_are_registered():
    model_cost = litellm.get_model_cost_map(url="")

    assert model_cost["toapis/gpt-image-2"]["mode"] == "image_generation"
    assert model_cost["toapis/seedance-2-5"]["mode"] == "video_generation"


def test_toapis_chat_completion_uses_provider_url_and_key(respx_mock, monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    route = respx_mock.post("https://toapis.com/v1/chat/completions").respond(
        json={
            "id": "chatcmpl-123",
            "object": "chat.completion",
            "created": 1703884800,
            "model": "gpt-5.6-terra",
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": "hello"},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 1, "completion_tokens": 1, "total_tokens": 2},
        }
    )

    response = litellm.completion(
        model="toapis/gpt-5.6-terra",
        messages=[{"role": "user", "content": "hello"}],
        api_key="test-key",
    )

    assert response.choices[0].message.content == "hello"
    assert route.calls[0].request.headers["Authorization"] == "Bearer test-key"


def test_toapis_responses_api_uses_provider_url_and_key(respx_mock):
    route = respx_mock.post("https://toapis.com/v1/responses").respond(
        json={
            "id": "resp_123",
            "object": "response",
            "created_at": 1734366691,
            "status": "completed",
            "model": "gpt-5.3-codex-official",
            "output": [
                {
                    "type": "message",
                    "id": "msg_1",
                    "status": "completed",
                    "role": "assistant",
                    "content": [{"type": "output_text", "text": "done", "annotations": []}],
                }
            ],
            "parallel_tool_calls": True,
            "usage": {
                "input_tokens": 1,
                "output_tokens": 1,
                "total_tokens": 2,
                "output_tokens_details": {"reasoning_tokens": 0},
            },
            "error": None,
            "incomplete_details": None,
            "instructions": None,
            "metadata": None,
            "temperature": None,
            "tool_choice": "auto",
            "tools": [],
            "top_p": None,
            "max_output_tokens": None,
            "previous_response_id": None,
            "reasoning": None,
            "truncation": None,
            "user": None,
        }
    )

    response = litellm.responses(
        model="toapis/gpt-5.3-codex-official",
        input="hello",
        api_key="test-key",
    )

    assert response.output[0].content[0].text == "done"
    assert route.calls[0].request.headers["Authorization"] == "Bearer test-key"
