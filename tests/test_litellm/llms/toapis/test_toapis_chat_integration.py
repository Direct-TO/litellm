"""Exercise the ToAPIs chat transport; mocked replies do not certify upstream capabilities."""

import json

import pytest

import litellm

MODEL = "toapis/gpt-6.1-terra"
UPSTREAM_MODEL = "gpt-6.1-terra"
BASE_URL = "https://toapis.cn/v1"
MESSAGES = [{"role": "user", "content": "Return the title of a short video."}]
TOOL = {
    "type": "function",
    "function": {
        "name": "save_title",
        "description": "Save the title of the video.",
        "parameters": {
            "type": "object",
            "properties": {"title": {"type": "string"}},
            "required": ["title"],
            "additionalProperties": False,
        },
    },
}
TOOL_CHOICE = {"type": "function", "function": {"name": "save_title"}}


@pytest.fixture(autouse=True)
def isolated_toapis_transport(monkeypatch):
    monkeypatch.setattr(litellm, "disable_aiohttp_transport", True)
    monkeypatch.setattr(litellm, "api_key", None)
    monkeypatch.setattr(litellm, "api_base", None)
    monkeypatch.setenv("TOAPIS_API_KEY", "toapis-chat-test-key")
    monkeypatch.setenv("TOAPIS_API_BASE", BASE_URL)


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_terra_chat_preserves_downstream_json_request(respx_mock, asynchronous):
    """Startork's JSON-object chat path must keep the model and response-format contract."""
    options = {"response_format": {"type": "json_object"}}
    route = respx_mock.post(f"{BASE_URL}/chat/completions").respond(
        json={
            "id": "chatcmpl-terra",
            "object": "chat.completion",
            "created": 1703884800,
            "model": UPSTREAM_MODEL,
            "choices": [
                {
                    "index": 0,
                    "message": {"role": "assistant", "content": '{"title":"A short video"}'},
                    "finish_reason": "stop",
                }
            ],
            "usage": {"prompt_tokens": 12, "completion_tokens": 8, "total_tokens": 20},
        }
    )

    kwargs = {"model": MODEL, "messages": MESSAGES, "max_retries": 0, **options}
    response = await litellm.acompletion(**kwargs) if asynchronous else litellm.completion(**kwargs)

    assert route.call_count == 1
    assert len(respx_mock.calls) == 1
    request = route.calls[0].request
    assert request.headers["Authorization"] == "Bearer toapis-chat-test-key"
    payload = json.loads(request.content)
    assert payload["model"] == UPSTREAM_MODEL
    assert payload["messages"] == MESSAGES
    for key, value in options.items():
        assert payload[key] == value
    assert "input" not in payload
    assert response.choices[0].finish_reason == "stop"
    assert response.usage.total_tokens == 20
    assert json.loads(response.choices[0].message.content) == {"title": "A short video"}


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
@pytest.mark.parametrize("stream", [False, True], ids=["nonstreaming", "streaming"])
async def test_terra_rejects_tool_requests_before_sending_http(respx_mock, asynchronous, stream):
    """The verified backend lacks tool calls; do not spend on a silently unfulfilled tool request."""
    kwargs = {
        "model": MODEL,
        "messages": MESSAGES,
        "tools": [TOOL],
        "tool_choice": TOOL_CHOICE,
        "stream": stream,
        "max_retries": 0,
    }

    with pytest.raises(litellm.UnsupportedParamsError, match="does not support parameters") as error:
        if asynchronous:
            await litellm.acompletion(**kwargs)
        else:
            litellm.completion(**kwargs)

    assert "tools" in str(error.value)
    assert "tool_choice" in str(error.value)
    assert len(respx_mock.calls) == 0


def _sse_events(deltas, finish_reason):
    events = []
    for delta in deltas:
        events.append(
            {
                "id": "chatcmpl-terra-stream",
                "object": "chat.completion.chunk",
                "created": 1703884800,
                "model": UPSTREAM_MODEL,
                "choices": [{"index": 0, "delta": delta, "finish_reason": None}],
            }
        )
    events.append(
        {
            "id": "chatcmpl-terra-stream",
            "object": "chat.completion.chunk",
            "created": 1703884800,
            "model": UPSTREAM_MODEL,
            "choices": [{"index": 0, "delta": {}, "finish_reason": finish_reason}],
        }
    )
    return "".join(f"data: {json.dumps(event)}\n\n" for event in events) + "data: [DONE]\n\n"


@pytest.mark.parametrize("asynchronous", [False, True], ids=["sync", "async"])
async def test_terra_chat_parses_sse_text_deltas(respx_mock, asynchronous):
    """Decode actual HTTP SSE bytes without replacing the provider's streaming parser."""
    deltas = [{"role": "assistant", "content": ""}, {"content": "A short "}, {"content": "video"}]
    route = respx_mock.post(f"{BASE_URL}/chat/completions").respond(
        text=_sse_events(deltas, "stop"),
        headers={"Content-Type": "text/event-stream"},
    )

    kwargs = {"model": MODEL, "messages": MESSAGES, "stream": True, "max_retries": 0}
    stream = await litellm.acompletion(**kwargs) if asynchronous else litellm.completion(**kwargs)
    chunks = [chunk async for chunk in stream] if asynchronous else list(stream)

    assert route.call_count == 1
    assert len(respx_mock.calls) == 1
    request = route.calls[0].request
    assert request.headers["Authorization"] == "Bearer toapis-chat-test-key"
    payload = json.loads(request.content)
    assert payload["model"] == UPSTREAM_MODEL
    assert payload["messages"] == MESSAGES
    assert payload["stream"] is True
    assert [chunk.choices[0].finish_reason for chunk in chunks if chunk.choices[0].finish_reason] == ["stop"]
    assert "".join(chunk.choices[0].delta.content or "" for chunk in chunks) == "A short video"
