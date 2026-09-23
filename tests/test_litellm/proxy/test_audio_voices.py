"""Local HTTP tests: no provider calls, shared database, or paid generations."""

from copy import deepcopy
from datetime import datetime, timezone
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import httpx
import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

import litellm
from litellm.proxy import proxy_server
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.audio_endpoints import voices
from litellm.proxy.auth.auth_utils import get_model_from_request
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth


class MemoryTable:
    def __init__(self):
        self.rows = {}

    async def create(self, data):
        key = data["unified_object_id"]
        if key in self.rows:
            raise ValueError("unique constraint")
        row = SimpleNamespace(
            **{**data, "file_object": deepcopy(data["file_object"].data), "created_at": datetime.now(timezone.utc)}
        )
        self.rows[key] = row
        return row

    async def update(self, where, data):
        row = self.rows[where["unified_object_id"]]
        for k, v in data.items():
            setattr(row, k, deepcopy(v.data) if k == "file_object" else v)
        return row

    async def find_unique(self, where):
        return self.rows.get(where["unified_object_id"])

    async def find_many(self, where):
        return [r for r in self.rows.values() if all(getattr(r, k) == v for k, v in where.items())]


@pytest.fixture
def api(monkeypatch):
    memory = MemoryTable()
    monkeypatch.setattr(
        proxy_server, "prisma_client", SimpleNamespace(db=SimpleNamespace(litellm_managedobjecttable=memory))
    )
    llm_router = litellm.Router(
        model_list=[
            {
                "model_name": "voice-model",
                "litellm_params": {
                    "model": "minimax/speech-2.8-hd",
                    "api_base": "https://api.minimax.cn",
                    "api_key": "account-one",
                    "default_voice_id": "female-yujie",
                },
                "model_info": {"id": "deployment-one"},
            },
            {
                "model_name": "other-model",
                "litellm_params": {
                    "model": "minimax/speech-2.8-turbo",
                    "api_base": "https://api.minimax.cn",
                    "api_key": "account-two",
                },
                "model_info": {"id": "deployment-two"},
            },
        ],
        num_retries=0,
    )
    monkeypatch.setattr(proxy_server, "llm_router", llm_router)
    monkeypatch.setattr(proxy_server, "user_model", None)
    monkeypatch.setattr(
        proxy_server,
        "proxy_logging_obj",
        MagicMock(
            pre_call_hook=AsyncMock(side_effect=lambda **kw: kw["data"]),
            post_call_failure_hook=AsyncMock(),
            post_call_response_headers_hook=AsyncMock(return_value={}),
            update_request_status=AsyncMock(),
        ),
    )
    monkeypatch.setattr(proxy_server, "add_litellm_data_to_request", AsyncMock(side_effect=lambda **kw: kw["data"]))
    auth = UserAPIKeyAuth(api_key="key-one", user_id="user-one")
    app = FastAPI()
    app.include_router(voices.router)
    app.add_api_route("/v1/audio/speech", proxy_server.audio_speech, methods=["POST"])
    app.dependency_overrides[user_api_key_auth] = lambda: auth
    calls = []

    async def fake_upstream(deployment, path, *, payload=None, audio=None):
        calls.append((deployment, path, payload, audio))
        if path == "get_voice":
            return {"system_voice": [{"voice_id": "female-yujie", "voice_name": "御姐"}]}
        if path == "voice_design":
            return {"voice_id": payload["voice_id"], "trial_audio": b"ID3-preview".hex()}
        if path == "files/upload":
            return {"file": {"file_id": 12345}}
        if path == "voice_clone":
            return {"demo_audio": "https://cdn.example/preview.mp3"}
        raise AssertionError(path)

    monkeypatch.setattr(voices, "upstream", fake_upstream)
    yield SimpleNamespace(
        client=TestClient(app, raise_server_exceptions=False),
        auth=auth,
        app=app,
        calls=calls,
        memory=memory,
        router=llm_router,
    )
    llm_router.discard()


def design(api, key="test-design"):
    return api.client.post(
        "/v1/audio/voices/design",
        headers={"Idempotency-Key": key},
        json={"model": "voice-model", "prompt": "成熟、低沉的男声", "preview_text": "原文。"},
    )


def test_design_query_and_idempotent_replay(api):
    first = design(api)
    assert first.status_code == 200, first.text
    voice_id = first.json()["voice_id"]
    assert first.json()["preview_audio"]["data"]
    assert design(api).json()["voice_id"] == voice_id
    assert len(api.calls) == 1
    listing = api.client.get("/v1/audio/voices", params={"model": "voice-model"})
    assert listing.status_code == 200, listing.text
    assert {v["voice_id"] for v in listing.json()["data"]} == {"female-yujie", voice_id}
    # Database-backed records survive a different handler/client instance.
    second_client = TestClient(api.app)
    assert voice_id in str(second_client.get("/v1/audio/voices?model=voice-model").json())


def test_clone_upload_then_create_on_same_account(api):
    response = api.client.post(
        "/v1/audio/voices/clone",
        data={"model": "voice-model", "preview_text": "试听。"},
        files={"audio": ("reference.wav", b"RIFF-test-reference", "audio/wav")},
    )
    assert response.status_code == 200, response.text
    assert [c[1] for c in api.calls] == ["files/upload", "voice_clone"]
    assert api.calls[1][2]["file_id"] == 12345
    assert api.calls[1][2]["voice_id"] == response.json()["voice_id"]
    assert api.calls[0][0]["model_info"]["id"] == api.calls[1][0]["model_info"]["id"]


def test_foreign_owner_and_account_are_not_usable(api):
    voice_id = design(api).json()["voice_id"]
    payload = {"model": "other-model", "text": "台词", "voice_setting": {"voice_id": voice_id}}
    assert api.client.post("/v1/audio/speech", json=payload).status_code == 409
    api.auth.user_id = "another-user"
    payload["model"] = "voice-model"
    assert api.client.post("/v1/audio/speech", json=payload).status_code == 404
    listing = api.client.get("/v1/audio/voices?model=voice-model").json()["data"]
    assert all(v["type"] == "system" for v in listing)


def test_uncertain_creation_does_not_retry(api, monkeypatch):
    upstream = AsyncMock(side_effect=HTTPException(502, "timeout"))
    monkeypatch.setattr(voices, "upstream", upstream)
    first = design(api)
    assert first.status_code == 502
    assert first.json()["detail"]["status"] == "unknown"
    assert design(api).status_code == 409
    upstream.assert_awaited_once()


@pytest.mark.parametrize(
    "payload",
    [
        {"model": "voice-model", "text": "a", "input": "b"},
        {"model": "voice-model", "text": "a", "voice_setting": {"emotion": "愤怒"}},
        {"model": "voice-model", "text": "a", "voice_setting": {"speed": 9}},
        {"model": "voice-model", "text": "a", "instructions": "情绪：高兴"},
        {"model": "voice-model", "text": "a", "api_base": "https://evil.example"},
    ],
)
def test_bad_contract_rejected_without_upstream(api, payload):
    response = api.client.post("/v1/audio/speech", json=payload)
    assert response.status_code == 400, response.text
    assert api.calls == []


def test_query_model_is_in_auth_scope():
    assert (
        get_model_from_request({}, "/v1/audio/voices", request_query_params={"model": "private-model"})
        == "private-model"
    )


def test_database_required_before_creation(api, monkeypatch):
    monkeypatch.setattr(proxy_server, "prisma_client", None)
    assert design(api).status_code == 503
    assert api.calls == []


def test_managed_voice_cannot_bypass_ownership_without_router(api, monkeypatch):
    voice_id = design(api).json()["voice_id"]
    monkeypatch.setattr(proxy_server, "llm_router", None)
    response = api.client.post("/v1/audio/speech", json={"model": "voice-model", "text": "台词。", "voice": voice_id})
    assert response.status_code == 503


def test_idempotency_key_conflicting_payload(api):
    assert design(api).status_code == 200
    response = api.client.post(
        "/v1/audio/voices/design",
        headers={"Idempotency-Key": "test-design"},
        json={"model": "voice-model", "prompt": "different", "preview_text": "原文。"},
    )
    assert response.status_code == 409
    assert len(api.calls) == 1


def test_paused_deployment_blocks_voice_creation(api, monkeypatch):
    monkeypatch.setattr(api.router, "_is_deployment_blocked", lambda d: True)
    assert design(api).status_code == 400
    assert api.calls == []


def test_clone_without_preview_does_not_request_synthesis(api):
    response = api.client.post(
        "/v1/audio/voices/clone",
        data={"model": "voice-model"},
        files={"audio": ("reference.wav", b"RIFF-reference", "audio/wav")},
    )
    assert response.status_code == 200
    assert "text" not in api.calls[1][2]
    assert "model" not in api.calls[1][2]


def test_openapi_documents_all_voice_endpoints(api):
    spec = api.client.get("/openapi.json").json()
    assert "/v1/audio/voices/design" in spec["paths"]
    assert "/v1/audio/voices/clone" in spec["paths"]
    clone = spec["paths"]["/v1/audio/voices/clone"]["post"]
    assert "multipart/form-data" in clone["requestBody"]["content"]


def test_speech_http_bytes_and_account_binding(api, monkeypatch):
    voice_id = design(api).json()["voice_id"]
    raw = httpx.Response(
        200,
        json={
            "base_resp": {"status_code": 0},
            "data": {"audio": b"RIFF-audio".hex()},
            "extra_info": {"audio_format": "wav"},
        },
        request=httpx.Request("POST", "https://api.minimax.cn/v1/t2a_v2"),
    )
    post = AsyncMock(return_value=raw)
    monkeypatch.setattr("litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post", post)
    response = api.client.post(
        "/v1/audio/speech",
        json={
            "model": "voice-model",
            "text": "不改写的台词。",
            "voice_setting": {"voice_id": voice_id, "emotion": "auto"},
            "audio_setting": {"format": "wav"},
        },
    )
    assert response.status_code == 200, response.text
    assert response.content == b"RIFF-audio"
    assert response.headers["content-type"] == "audio/wav"
    post.assert_awaited_once()
    kwargs = post.call_args.kwargs
    assert kwargs["headers"]["Authorization"] == "Bearer account-one"
    assert kwargs["json"]["voice_setting"]["voice_id"] == voice_id
    assert "emotion" not in kwargs["json"]["voice_setting"]
    assert kwargs["json"]["text"] == "不改写的台词。"


def test_missing_voice_uses_only_deployment_default(api, monkeypatch):
    raw = httpx.Response(
        200,
        json={
            "base_resp": {"status_code": 0},
            "data": {"audio": b"ID3-audio".hex()},
            "extra_info": {"audio_format": "mp3"},
        },
        request=httpx.Request("POST", "https://api.minimax.cn/v1/t2a_v2"),
    )
    post = AsyncMock(return_value=raw)
    monkeypatch.setattr("litellm.llms.custom_httpx.http_handler.AsyncHTTPHandler.post", post)
    response = api.client.post("/v1/audio/speech", json={"model": "voice-model", "text": "台词。"})
    assert response.status_code == 200, response.text
    assert post.call_args.kwargs["json"]["voice_setting"]["voice_id"] == "female-yujie"
    assert api.client.post("/v1/audio/speech", json={"model": "other-model", "text": "台词。"}).status_code == 400
