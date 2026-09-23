"""Explicit MiniMax speech parameters; no natural-language inference."""

from copy import deepcopy
from typing import Annotated, Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, StrictFloat, StrictInt, StrictStr

Number = StrictInt | StrictFloat
EMOTIONS = ("happy", "sad", "angry", "fearful", "disgusted", "surprised", "calm")


class VoiceSettings(BaseModel):
    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)
    voice_id: Annotated[StrictStr, Field(min_length=1, max_length=256)] | None = None
    emotion: Literal["auto", "happy", "sad", "angry", "fearful", "disgusted", "surprised", "calm"] | None = None
    speed: Annotated[Number, Field(ge=0.5, le=2)] | None = None
    pitch: Annotated[StrictInt, Field(ge=-12, le=12)] | None = None
    vol: Annotated[Number, Field(gt=0, le=10)] | None = None
    text_normalization: StrictBool | None = None
    latex_read: StrictBool | None = None


class AudioSettings(BaseModel):
    model_config = ConfigDict(extra="forbid")
    format: Literal["mp3", "wav", "flac", "pcm"] = "mp3"
    sample_rate: Literal[8000, 16000, 22050, 24000, 32000, 44100] = 32000
    bitrate: Literal[32000, 64000, 128000, 256000] = 128000
    channel: Literal[1, 2] = 1


LANGUAGES = frozenset(
    (
        "Chinese",
        "Chinese,Yue",
        "English",
        "Arabic",
        "Russian",
        "Spanish",
        "French",
        "Portuguese",
        "German",
        "Turkish",
        "Dutch",
        "Ukrainian",
        "Vietnamese",
        "Indonesian",
        "Japanese",
        "Italian",
        "Korean",
        "Thai",
        "Polish",
        "Romanian",
        "Greek",
        "Czech",
        "Finnish",
        "Hindi",
        "Bulgarian",
        "Danish",
        "Hebrew",
        "Malay",
        "Persian",
        "Slovak",
        "Swedish",
        "Croatian",
        "Filipino",
        "Hungarian",
        "Norwegian",
        "Slovenian",
        "Catalan",
        "Nynorsk",
        "Tamil",
        "Afrikaans",
        "auto",
    )
)
NATIVE_PARAMS = frozenset(
    {
        "voice_setting",
        "audio_setting",
        "voice_id",
        "emotion",
        "vol",
        "pitch",
        "text_normalization",
        "latex_read",
        "sample_rate",
        "bitrate",
        "channel",
        "language_boost",
        "pronunciation_dict",
        "voice_modify",
        "subtitle_enable",
        "subtitle_type",
        "aigc_watermark",
        "output_format",
    }
)


def merge_explicit(target: dict, source: dict, prefix: str = "") -> None:
    """Equal aliases are fine; conflicting supplied values are never silently overwritten."""
    for key, value in source.items():
        if value is None:
            continue
        path = f"{prefix}{key}"
        if isinstance(value, dict) and key in target and isinstance(target[key], dict):
            merge_explicit(target[key], value, f"{path}.")
        elif key in target and target[key] is not None and target[key] != value:
            raise ValueError(f"Conflicting values for {path}")
        else:
            target[key] = deepcopy(value)


def normalize_speech_request(data: dict) -> dict:
    """Normalize the public text contract to the existing SDK/router argument names."""
    result = dict(data)
    if isinstance(result.get("extra_body"), dict):
        result["extra_body"] = deepcopy(result["extra_body"])
    if "text" in result:
        text = result.pop("text")
        if "input" in result and result["input"] != text:
            raise ValueError("Conflicting text and input")
        result["input"] = text
    if not isinstance(result.get("model"), str) or not result["model"].strip():
        raise ValueError("model is required")
    if not isinstance(result.get("input"), str) or not result["input"].strip():
        raise ValueError("text is required (input is the legacy alias)")
    extra = result.get("extra_body") or {}
    if not isinstance(extra, dict):
        raise ValueError("extra_body must be an object")
    for group in ("voice_setting", "audio_setting"):
        nested = result.get(group)
        from_extra = extra.get(group)
        if nested is not None and not isinstance(nested, dict):
            raise ValueError(f"{group} must be an object")
        if from_extra is not None and not isinstance(from_extra, dict):
            raise ValueError(f"extra_body.{group} must be an object")
        combined = deepcopy(nested or {})
        merge_explicit(combined, from_extra or {}, f"{group}.")
        if combined:
            result[group] = combined
    for alias, group, key in (
        ("voice", "voice_setting", "voice_id"),
        ("speed", "voice_setting", "speed"),
        ("response_format", "audio_setting", "format"),
    ):
        value = result.get(group, {}).get(key)
        if value is not None:
            merge_explicit(result, {alias: value})
    result.setdefault("voice", None)  # Router requires the argument; adapter resolves deployment default.
    return result


def build_minimax_params(optional_params: dict, voice: Any, kwargs: dict | None) -> tuple[str, dict]:
    params: dict = {}
    for source in (kwargs or {}, optional_params):
        if source.get("stream") not in (None, False):
            raise ValueError("This speech endpoint returns completed audio; stream is not supported")
        if source.get("instructions") not in (None, ""):
            raise ValueError("MiniMax does not support instructions; use voice_setting.emotion")
        if "emotionIntensity" in source or "emotion_intensity" in source:
            raise ValueError("MiniMax does not support emotion intensity")
        extra = source.get("extra_body") or {}
        if not isinstance(extra, dict):
            raise ValueError("extra_body must be an object")
        unknown = set(extra) - NATIVE_PARAMS
        if unknown:
            raise ValueError(f"Unsupported MiniMax extra_body fields: {', '.join(sorted(unknown))}")
        merge_explicit(params, extra)
        merge_explicit(params, {k: v for k, v in source.items() if k in NATIVE_PARAMS})
    voice_data = params.pop("voice_setting", {})
    audio_data = params.pop("audio_setting", {})
    if not isinstance(voice_data, dict) or not isinstance(audio_data, dict):
        raise ValueError("voice_setting and audio_setting must be objects")
    for key in VoiceSettings.model_fields:
        if key in params:
            merge_explicit(voice_data, {key: params.pop(key)}, "voice_setting.")
    for key in ("sample_rate", "bitrate", "channel"):
        if key in params:
            merge_explicit(audio_data, {key: params.pop(key)}, "audio_setting.")
    if voice is not None:
        merge_explicit(voice_data, {"voice_id": voice}, "voice_setting.")
    if optional_params.get("speed") is not None:
        merge_explicit(voice_data, {"speed": optional_params["speed"]}, "voice_setting.")
    if optional_params.get("response_format") is not None:
        merge_explicit(audio_data, {"format": optional_params["response_format"]}, "audio_setting.")
    if not voice_data.get("voice_id"):
        voice_data["voice_id"] = (kwargs or {}).get("default_voice_id")
    settings = VoiceSettings.model_validate(voice_data).model_dump(exclude_none=True)
    if not settings.get("voice_id") or not settings["voice_id"].strip():
        raise ValueError("voice_setting.voice_id is required unless default_voice_id is configured")
    if settings.get("emotion") == "auto":
        settings.pop("emotion")
    language = params.get("language_boost")
    if language is not None and (not isinstance(language, str) or language not in LANGUAGES):
        raise ValueError("Unsupported language_boost")
    if params.pop("output_format", "hex") != "hex":
        raise ValueError("LiteLLM speech returns audio bytes; output_format must be hex")
    params.update(voice_setting=settings, audio_setting=AudioSettings.model_validate(audio_data).model_dump())
    return settings["voice_id"], params


SPEECH_REQUEST_SCHEMA = {
    "type": "object",
    "required": ["model"],
    "anyOf": [{"required": ["text"]}, {"required": ["input"]}],
    "properties": {
        "model": {"type": "string", "minLength": 1},
        "text": {"type": "string", "minLength": 1, "description": "Text to synthesize, unchanged."},
        "input": {"type": "string", "description": "Legacy alias of text; conflicts return 400."},
        "voice_setting": VoiceSettings.model_json_schema(),
        "audio_setting": AudioSettings.model_json_schema(),
        "language_boost": {"type": "string", "enum": sorted(LANGUAGES)},
    },
}
