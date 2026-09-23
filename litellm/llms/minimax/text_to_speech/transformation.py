"""MiniMax HTTP speech adapter with explicit provider parameters."""

import base64
import binascii
from typing import Any

import httpx

import litellm
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.text_to_speech.transformation import BaseTextToSpeechConfig, TextToSpeechRequestData
from litellm.llms.minimax.text_to_speech.contract import build_minimax_params
from litellm.secret_managers.main import get_secret_str
from litellm.types.llms.openai import HttpxBinaryResponseContent


class MinimaxException(BaseLLMException):
    pass


class MinimaxTextToSpeechConfig(BaseTextToSpeechConfig):
    TTS_BASE_URL = "https://api.minimax.io"
    TTS_ENDPOINT_PATH = "/v1/t2a_v2"
    FORMAT_MAPPINGS = {"mp3": "mp3", "pcm": "pcm", "wav": "wav", "flac": "flac"}
    AUDIO_MIME_TYPES = {
        "mp3": "audio/mpeg",
        "pcm": "application/octet-stream",
        "wav": "audio/wav",
        "flac": "audio/flac",
    }

    def get_supported_openai_params(self, model: str) -> list:
        return ["voice", "response_format", "speed"]

    def map_openai_params(
        self,
        model: str,
        optional_params: dict,
        voice: str | dict | None = None,
        drop_params: bool = False,
        kwargs: dict[str, Any] | None = None,
    ) -> tuple[str | None, dict]:
        try:
            return build_minimax_params(optional_params, voice, kwargs)
        except ValueError as exc:
            raise litellm.BadRequestError(message=str(exc), model=model, llm_provider="minimax") from exc

    def validate_environment(
        self,
        headers: dict,
        model: str,
        api_key: str | None = None,
        api_base: str | None = None,
    ) -> dict:
        api_key = api_key or litellm.api_key or get_secret_str("MINIMAX_API_KEY")
        if not api_key:
            raise ValueError("MiniMax API key is required. Set MINIMAX_API_KEY or pass api_key.")
        return {**headers, "Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    def get_error_class(self, error_message: str, status_code: int, headers: dict | httpx.Headers) -> BaseLLMException:
        return MinimaxException(message=error_message, status_code=status_code, headers=headers)

    def transform_text_to_speech_request(
        self,
        model: str,
        input: str,
        voice: str | None,
        optional_params: dict,
        litellm_params: dict,
        headers: dict,
    ) -> TextToSpeechRequestData:
        if not isinstance(input, str) or not input.strip() or len(input) >= 10000:
            raise litellm.BadRequestError(
                message="text must contain 1 to 9999 characters", model=model, llm_provider="minimax"
            )
        # Revalidate at the provider boundary so direct handler users cannot bypass validation.
        try:
            _, params = build_minimax_params({}, voice, optional_params)
        except ValueError as exc:
            raise litellm.BadRequestError(message=str(exc), model=model, llm_provider="minimax") from exc
        return TextToSpeechRequestData(
            dict_body={**params, "model": model, "text": input, "stream": False, "output_format": "hex"},
            headers={"Content-Type": "application/json"},
        )

    def transform_text_to_speech_response(
        self,
        model: str,
        raw_response: httpx.Response,
        logging_obj: Any,
    ) -> HttpxBinaryResponseContent:
        try:
            response = raw_response.json()
        except ValueError as exc:
            raise MinimaxException(status_code=502, message="MiniMax returned invalid JSON", headers={}) from exc
        if not isinstance(response, dict):
            raise MinimaxException(status_code=502, message="MiniMax returned an invalid response", headers={})
        base = response.get("base_resp") or {}
        error_code = base.get("status_code", response.get("status", 0))
        if error_code != 0:
            # A provider business rejection must not escape as HTTP 200.
            status = {1004: 401, 2049: 401, 1002: 429, 1039: 429, 2013: 400, 2038: 403}.get(error_code, 502)
            raise MinimaxException(
                status_code=status,
                message=f"MiniMax TTS error {error_code}: {base.get('status_msg') or response.get('ced') or 'rejected'}",
                headers=dict(raw_response.headers),
            )
        data = response.get("data") or {}
        encoded = data.get("audio")
        try:
            if encoded:
                audio = bytes.fromhex(encoded)
            elif response.get("audio_file"):
                audio = base64.b64decode(response["audio_file"], validate=True)
            else:
                raise ValueError("missing audio")
            if not audio:
                raise ValueError("empty audio")
        except (ValueError, TypeError, binascii.Error) as exc:
            raise MinimaxException(
                status_code=502, message="MiniMax returned missing or invalid audio", headers={}
            ) from exc
        info = response.get("extra_info") or {}
        headers = dict(raw_response.headers)
        headers.pop("content-encoding", None)
        headers.pop("transfer-encoding", None)
        headers["content-length"] = str(len(audio))
        headers["content-type"] = self.AUDIO_MIME_TYPES.get(info.get("audio_format"), "application/octet-stream")
        result = HttpxBinaryResponseContent(
            httpx.Response(200, headers=headers, content=audio, request=raw_response.request)
        )
        result._hidden_params.update(
            minimax_usage_characters=info.get("usage_characters"), minimax_trace_id=response.get("trace_id")
        )
        return result

    def get_complete_url(self, model: str, api_base: str | None, litellm_params: dict) -> str:
        base = (api_base or get_secret_str("MINIMAX_API_BASE") or self.TTS_BASE_URL).rstrip("/")
        if base.endswith("/t2a_v2"):
            return base
        if base.endswith("/v1"):
            return f"{base}/t2a_v2"
        return f"{base}{self.TTS_ENDPOINT_PATH}"
