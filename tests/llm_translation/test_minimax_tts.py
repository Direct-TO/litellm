"""Network-free contracts for MiniMax Speech 2.8."""

import json
from pathlib import Path
from unittest.mock import MagicMock, patch

import httpx
import pytest

import litellm
from litellm.llms.minimax.text_to_speech.contract import normalize_speech_request
from litellm.llms.minimax.text_to_speech.transformation import MinimaxException, MinimaxTextToSpeechConfig

CONFIG = MinimaxTextToSpeechConfig()


def request_body(**kwargs):
    voice, params = CONFIG.map_openai_params(
        "speech-2.8-hd",
        kwargs.pop("optional_params", {}),
        voice=kwargs.pop("voice", None),
        kwargs=kwargs,
    )
    return CONFIG.transform_text_to_speech_request("speech-2.8-hd", "台词不改写。", voice, params, {}, {})["dict_body"]


@pytest.mark.parametrize(
    "emotion", [None, "auto", "happy", "sad", "angry", "fearful", "disgusted", "surprised", "calm"]
)
def test_explicit_emotion(emotion):
    body = request_body(voice_setting={"voice_id": "female-yujie", "emotion": emotion})
    assert body["text"] == "台词不改写。"
    assert body["voice_setting"]["voice_id"] == "female-yujie"
    if emotion in (None, "auto"):
        assert "emotion" not in body["voice_setting"]
    else:
        assert body["voice_setting"]["emotion"] == emotion


@pytest.mark.parametrize(
    "values",
    [
        {"emotion": "愤怒"},
        {"emotion": "坚定"},
        {"emotion": "whisper"},
        {"emotion": "fluent"},
        {"speed": 0.49},
        {"speed": 2.01},
        {"speed": True},
        {"speed": "1"},
        {"vol": 0},
        {"vol": 11},
        {"pitch": 13},
        {"pitch": 0.1},
        {"speed": float("nan")},
        {"persona": "a detective"},
        {"emotionIntensity": 0.8},
    ],
)
def test_invalid_controls_rejected(values):
    with pytest.raises(litellm.BadRequestError):
        request_body(voice_setting={"voice_id": "male-qn-badao", **values})


def test_no_voice_fabrication_or_instruction_guessing():
    with pytest.raises(litellm.BadRequestError, match="voice_id"):
        request_body()
    with pytest.raises(litellm.BadRequestError, match="instructions"):
        request_body(voice="male-qn-badao", optional_params={"instructions": "情绪：愤怒；情绪强度：80%"})
    assert request_body(default_voice_id="male-qn-badao")["voice_setting"]["voice_id"] == "male-qn-badao"
    # IDs are opaque. No OpenAI alias -> guessed MiniMax persona conversion.
    assert request_body(voice="alloy")["voice_setting"]["voice_id"] == "alloy"


@pytest.mark.parametrize(
    "data",
    [
        {"text": "a", "input": "b"},
        {"text": "a", "voice": "a", "voice_setting": {"voice_id": "b"}},
        {"text": "a", "speed": 1, "voice_setting": {"speed": 2}},
        {"text": "a", "response_format": "mp3", "audio_setting": {"format": "wav"}},
        {"text": "a", "voice_setting": {"emotion": "happy"}, "extra_body": {"voice_setting": {"emotion": "sad"}}},
    ],
)
def test_conflicting_aliases_rejected(data):
    with pytest.raises(ValueError, match="Conflicting"):
        normalize_speech_request({"model": "speech", **data})


def test_canonical_request_to_sdk():
    raw = {
        "model": "speech",
        "text": "原文。",
        "voice_setting": {"voice_id": "female-yujie", "speed": 1.2},
        "audio_setting": {"format": "wav"},
    }
    normalized = normalize_speech_request(raw)
    assert normalized["input"] == "原文。"
    assert normalized["voice"] == "female-yujie"
    assert normalized["response_format"] == "wav"
    assert raw["text"] == "原文。"


def test_native_kwargs_and_extra_body_identical_and_no_metadata_leak():
    native = {
        "voice_setting": {"voice_id": "female-yujie", "emotion": "angry", "pitch": -2},
        "audio_setting": {"format": "wav", "sample_rate": 24000},
        "language_boost": "Chinese",
    }
    direct = request_body(**native, metadata={"secret": "not-upstream"})
    assert direct == request_body(extra_body=native)
    assert "metadata" not in direct
    assert direct["audio_setting"]["format"] == "wav"
    with pytest.raises(litellm.BadRequestError, match="Conflicting"):
        request_body(voice="male-qn-badao", **native)


@pytest.mark.parametrize(
    "base", ["https://api.minimax.cn", "https://api.minimax.cn/v1/", "https://api.minimax.cn/v1/t2a_v2"]
)
def test_base_url(base):
    assert CONFIG.get_complete_url("speech-2.8-hd", base, {}) == "https://api.minimax.cn/v1/t2a_v2"


def provider_response(**overrides):
    data = {
        "data": {"audio": b"RIFF-test-audio".hex(), "status": 2},
        "base_resp": {"status_code": 0},
        "extra_info": {"audio_format": "wav", "usage_characters": 22},
    }
    data.update(overrides)
    return httpx.Response(200, json=data, request=httpx.Request("POST", "https://api.minimax.cn/v1/t2a_v2"))


def test_audio_decode_and_business_rejection():
    response = CONFIG.transform_text_to_speech_response("speech-2.8-hd", provider_response(), MagicMock())
    assert response.content == b"RIFF-test-audio"
    assert response.response.headers["content-type"] == "audio/wav"
    assert response._hidden_params["minimax_usage_characters"] == 22
    with pytest.raises(MinimaxException) as exc:
        CONFIG.transform_text_to_speech_response(
            "speech-2.8-hd", provider_response(data=None, base_resp={"status_code": 1004}), MagicMock()
        )
    assert exc.value.status_code == 401


@pytest.mark.parametrize("async_mode", [False, True])
@pytest.mark.asyncio
async def test_sdk_actual_http_boundary(async_mode):
    from unittest.mock import AsyncMock

    target = "litellm.llms.custom_httpx.http_handler." + ("AsyncHTTPHandler" if async_mode else "HTTPHandler") + ".post"
    mock = AsyncMock(return_value=provider_response()) if async_mode else MagicMock(return_value=provider_response())
    with patch(target, mock):
        kwargs = {
            "model": "minimax/speech-2.8-hd",
            "input": "原文。",
            "api_base": "https://api.minimax.cn/v1",
            "api_key": "not-real",
            "voice": "female-yujie",
            "response_format": "wav",
            "extra_body": {"voice_setting": {"emotion": "angry"}, "language_boost": "Chinese"},
        }
        response = await litellm.aspeech(**kwargs) if async_mode else litellm.speech(**kwargs)
    assert response.content == b"RIFF-test-audio"
    mock.assert_called_once()
    body = mock.call_args.kwargs["json"]
    assert body["text"] == "原文。"
    assert body["voice_setting"]["emotion"] == "angry"
    assert body["voice_setting"]["voice_id"] == "female-yujie"
    assert body["audio_setting"]["format"] == "wav"
    assert "instructions" not in body


@pytest.mark.parametrize(
    "relative", ["model_prices_and_context_window.json", "litellm/model_prices_and_context_window_backup.json"]
)
def test_model_catalog(relative):
    info = json.loads((Path(__file__).parents[2] / relative).read_text(encoding="utf-8"))
    for name in ("hd", "turbo"):
        assert info[f"minimax/speech-2.8-{name}"]["mode"] == "audio_speech"
