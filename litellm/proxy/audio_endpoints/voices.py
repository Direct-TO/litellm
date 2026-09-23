"""Account-bound voice resources backed by the existing managed object table.

No provider request is retried here. Reserve the resource before creating it so
an uncertain upstream result remains queryable and cannot be silently recreated.
"""

import base64
import hashlib
import json
from pathlib import Path
from typing import Annotated, Any, Literal
from uuid import uuid4

import httpx
from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, Request, UploadFile
from pydantic import BaseModel, ConfigDict, Field, StrictStr

import litellm
from litellm.llms.minimax.text_to_speech.contract import build_minimax_params, normalize_speech_request
from litellm.llms.minimax.text_to_speech.transformation import MinimaxTextToSpeechConfig
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.auth.user_api_key_auth import user_api_key_auth
from litellm.secret_managers.main import get_secret_str

router = APIRouter(tags=["audio"])
VOICE_PREFIX = "voice_"
PURPOSE = "audio_voice"
MAX_AUDIO_BYTES = 20 * 1024 * 1024


class VoiceDesignRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    model: Annotated[StrictStr, Field(min_length=1)]
    prompt: Annotated[StrictStr, Field(min_length=1, max_length=4000)]
    preview_text: Annotated[StrictStr, Field(min_length=1, max_length=500)]


class PreviewAudio(BaseModel):
    content_type: str | None = None
    data: str | None = Field(default=None, description="Base64 audio from voice design")
    url: str | None = Field(default=None, description="Upstream preview URL from voice cloning")


class VoiceResource(BaseModel):
    voice_id: str
    type: Literal["system", "designed", "cloned"]
    status: Literal["creating", "ready", "unknown"]
    model: str | None = None
    created_at: str | None = None
    name: str | None = None
    description: list[str] | str | None = None
    preview_audio: PreviewAudio | None = None


class VoiceList(BaseModel):
    object: Literal["list"] = "list"
    data: list[VoiceResource]


def owner_scope(auth: UserAPIKeyAuth) -> str:
    # Never trust client metadata/user fields for resource ownership.
    identity = auth.user_id or auth.api_key
    if not identity:
        raise HTTPException(403, "An authenticated owner is required for voice resources")
    return hashlib.sha256(json.dumps([auth.team_id, identity]).encode()).hexdigest()


def table():
    from litellm.proxy import proxy_server

    if proxy_server.prisma_client is None:
        raise HTTPException(503, "Managed voices require the LiteLLM database")
    return proxy_server.prisma_client.db.litellm_managedobjecttable


def metadata(row) -> dict:
    value = row.file_object
    return json.loads(value) if isinstance(value, str) else dict(value)


def credentials(deployment: dict) -> tuple[str, str, str]:
    params = deployment["litellm_params"]
    model, provider, dynamic_key, base = litellm.get_llm_provider(
        model=params["model"], custom_llm_provider=params.get("custom_llm_provider"), api_base=params.get("api_base")
    )
    if provider != "minimax" or model not in ("speech-2.8-hd", "speech-2.8-turbo"):
        raise HTTPException(400, "Voice management currently supports MiniMax Speech 2.8 HD/Turbo")
    key = params.get("api_key") or dynamic_key or get_secret_str("MINIMAX_API_KEY")
    if isinstance(key, str) and key.startswith("os.environ/"):
        key = get_secret_str(key[len("os.environ/") :])
    if not isinstance(key, str) or not key:
        raise HTTPException(503, "MiniMax credentials are not configured")
    url = MinimaxTextToSpeechConfig().get_complete_url(model, base, {})
    return url.removesuffix("/t2a_v2"), key, model


def account_binding(deployment: dict) -> str:
    base, key, _ = credentials(deployment)
    return hashlib.sha256(json.dumps([base, key]).encode()).hexdigest()


def deployments_for(model: str, auth: UserAPIKeyAuth) -> list[dict]:
    from litellm.proxy import proxy_server

    llm_router = proxy_server.llm_router
    if llm_router is None:
        raise HTTPException(503, "Voice resources require a configured model router")
    deployments = llm_router.get_model_list(model_name=model, team_id=auth.team_id) or []
    return [
        d
        for d in deployments
        if llm_router._deployment_usable_by_team(d, auth.team_id) and not llm_router._is_deployment_blocked(d)
    ]


async def choose_deployment(model: str, auth: UserAPIKeyAuth, binding: str | None = None) -> dict:
    from litellm.proxy import proxy_server

    candidates = deployments_for(model, auth)
    if not candidates:
        raise HTTPException(400, "Model has no available deployments")
    if binding:
        candidates = [d for d in candidates if account_binding(d) == binding]
        if not candidates:
            raise HTTPException(409, "The voice's original upstream account is unavailable for this model")
        # Select only from the original account, including its current block/cooldown checks.
        requested = str(candidates[0]["model_info"]["id"])
    else:
        requested = model
    deployment = await proxy_server.llm_router.async_get_available_deployment(
        model=requested,
        messages=[],
        request_kwargs={"metadata": {"user_api_key_team_id": auth.team_id}},
    )
    if str(deployment["model_info"]["id"]) not in {str(d["model_info"]["id"]) for d in candidates}:
        raise HTTPException(403, "Selected deployment is outside the requested model scope")
    credentials(deployment)
    return deployment


async def upstream(deployment: dict, path: str, *, payload: dict | None = None, audio: tuple | None = None) -> dict:
    base, key, _ = credentials(deployment)
    # Explicitly zero retries; never send the proxy key or client-controlled URL.
    async with httpx.AsyncClient(timeout=90, follow_redirects=False) as client:
        kwargs: dict[str, Any] = {"headers": {"Authorization": f"Bearer {key}"}}
        if audio:
            kwargs.update(data={"purpose": "voice_clone"}, files={"file": audio})
        else:
            kwargs["json"] = payload
        try:
            response = await client.post(f"{base}/{path}", **kwargs)
        except httpx.HTTPError as exc:
            raise HTTPException(
                502, "MiniMax request outcome is unknown; do not automatically recreate the voice"
            ) from exc
    try:
        data = response.json()
    except ValueError as exc:
        raise HTTPException(502, "MiniMax returned invalid JSON; request outcome is unknown") from exc
    if not isinstance(data, dict):
        raise HTTPException(502, "MiniMax returned an invalid response")
    code = (data.get("base_resp") or {}).get("status_code")
    if response.status_code != 200 or code != 0:
        status = {1004: 401, 2049: 401, 2013: 400, 2038: 403, 1002: 429, 1039: 429}.get(code, 502)
        # Upstream messages can contain request data; expose the code, not credentials/body.
        raise HTTPException(status, f"MiniMax rejected the request (HTTP {response.status_code}, code {code})")
    return data


def public_voice(row) -> dict:
    info = metadata(row)
    return {
        "voice_id": row.unified_object_id,
        "type": info["type"],
        "model": info["model"],
        "status": row.status,
        "created_at": row.created_at.isoformat() if row.created_at else None,
        **({"preview_audio": info["preview_audio"]} if info.get("preview_audio") else {}),
    }


async def reserve_voice(
    auth: UserAPIKeyAuth, model: str, kind: str, fingerprint: str, request_key: str | None, deployment: dict
) -> tuple[Any, bool]:
    from prisma import Json

    db = table()
    owner = owner_scope(auth)
    if request_key and len(request_key) > 200:
        raise HTTPException(400, "Idempotency-Key must be at most 200 characters")
    suffix = hashlib.sha256(f"{owner}:{kind}:{request_key}".encode()).hexdigest()[:32] if request_key else uuid4().hex
    voice_id = VOICE_PREFIX + suffix
    info = {
        "model": model,
        "type": kind,
        "fingerprint": fingerprint,
        "account_binding": account_binding(deployment),
        "upstream_voice_id": voice_id,
        "deployment_id": str(deployment["model_info"]["id"]),
    }
    try:
        row = await db.create(
            data={
                "unified_object_id": voice_id,
                "model_object_id": "audio:" + voice_id,
                "file_purpose": PURPOSE,
                "file_object": Json(info),
                "status": "creating",
                "created_by": owner,
                "team_id": auth.team_id,
                "batch_processed": True,
            }
        )
        return row, True
    except Exception:
        existing = await db.find_unique(where={"unified_object_id": voice_id})
        if existing is None:
            raise HTTPException(503, "Could not reserve voice metadata; no upstream creation was attempted") from None
        if existing.created_by != owner or metadata(existing).get("fingerprint") != fingerprint:
            raise HTTPException(409, "Idempotency-Key was already used with different parameters") from None
        if existing.status != "ready":
            raise HTTPException(
                409,
                {
                    "voice_id": voice_id,
                    "status": existing.status,
                    "message": "Creation is incomplete or uncertain; no new request was sent",
                },
            ) from None
        return existing, False


async def create_voice(
    model: str,
    kind: str,
    auth: UserAPIKeyAuth,
    request_key: str | None,
    *,
    prompt: str | None = None,
    preview_text: str | None = None,
    audio: tuple | None = None,
) -> dict:
    table()  # Fail before upload/design if durable ownership storage is unavailable.
    deployment = await choose_deployment(model, auth)
    fingerprint = hashlib.sha256(
        json.dumps(
            [model, kind, prompt, preview_text, hashlib.sha256(audio[1]).hexdigest() if audio else None],
            ensure_ascii=False,
        ).encode()
    ).hexdigest()
    row, fresh = await reserve_voice(auth, model, kind, fingerprint, request_key, deployment)
    if not fresh:
        return public_voice(row)
    info = metadata(row)
    try:
        if kind == "designed":
            data = await upstream(
                deployment,
                "voice_design",
                payload={"voice_id": info["upstream_voice_id"], "prompt": prompt, "preview_text": preview_text},
            )
            returned_id = data.get("voice_id")
            if not isinstance(returned_id, str) or not returned_id:
                raise HTTPException(502, "Voice design returned no voice_id")
            info["upstream_voice_id"] = returned_id
            trial = data.get("trial_audio")
            if isinstance(trial, str) and trial:
                try:
                    decoded = bytes.fromhex(trial)
                except ValueError as exc:
                    raise HTTPException(502, "Voice design returned invalid trial audio") from exc
                mime = "application/octet-stream"
                if decoded.startswith(b"RIFF") and decoded[8:12] == b"WAVE":
                    mime = "audio/wav"
                elif decoded.startswith(b"fLaC"):
                    mime = "audio/flac"
                elif decoded.startswith(b"ID3") or (len(decoded) > 1 and decoded[0] == 255 and decoded[1] & 224 == 224):
                    mime = "audio/mpeg"
                info["preview_audio"] = {"content_type": mime, "data": base64.b64encode(decoded).decode()}
        else:
            uploaded = await upstream(deployment, "files/upload", audio=audio)
            file_id = (uploaded.get("file") or {}).get("file_id")
            if not isinstance(file_id, int):
                raise HTTPException(502, "Voice upload returned no file_id")
            payload = {"file_id": file_id, "voice_id": info["upstream_voice_id"]}
            if preview_text:
                payload.update(text=preview_text, model=credentials(deployment)[2])
            data = await upstream(deployment, "voice_clone", payload=payload)
            if data.get("input_sensitive"):
                raise HTTPException(400, "MiniMax rejected the reference recording")
            if data.get("demo_audio"):
                info["preview_audio"] = {"url": data["demo_audio"]}
        info["upstream_usage"] = data.get("extra_info")
        from prisma import Json

        saved = await table().update(
            where={"unified_object_id": row.unified_object_id}, data={"file_object": Json(info), "status": "ready"}
        )
        return public_voice(saved)
    except Exception as exc:
        try:
            from prisma import Json

            await table().update(
                where={"unified_object_id": row.unified_object_id},
                data={"file_object": Json(info), "status": "unknown"},
            )
        except Exception:
            pass  # The original creating reservation remains; it still blocks a duplicate submission.
        status = exc.status_code if isinstance(exc, HTTPException) else 502
        raise HTTPException(
            status,
            {
                "voice_id": row.unified_object_id,
                "status": "unknown",
                "message": "Voice creation did not finish reliably. Check the original voice; do not recreate automatically.",
            },
        ) from None


@router.post("/v1/audio/voices/design", response_model=VoiceResource, response_model_exclude_none=True)
async def design_voice(body: VoiceDesignRequest, request: Request, auth: UserAPIKeyAuth = Depends(user_api_key_auth)):
    if not body.prompt.strip() or not body.preview_text.strip() or not body.model.strip():
        raise HTTPException(400, "model, prompt and preview_text must be nonempty")
    return await create_voice(
        body.model,
        "designed",
        auth,
        request.headers.get("Idempotency-Key"),
        prompt=body.prompt,
        preview_text=body.preview_text,
    )


@router.post("/v1/audio/voices/clone", response_model=VoiceResource, response_model_exclude_none=True)
async def clone_voice(
    request: Request,
    model: str = Form(...),
    audio: UploadFile = File(...),
    preview_text: str | None = Form(None),
    auth: UserAPIKeyAuth = Depends(user_api_key_auth),
):
    try:
        content = await audio.read(MAX_AUDIO_BYTES + 1)
        extension = Path(audio.filename or "").suffix.lower()
        if extension not in (".mp3", ".m4a", ".wav") or not content or len(content) > MAX_AUDIO_BYTES:
            raise HTTPException(400, "audio must be a nonempty MP3/M4A/WAV file up to 20 MiB")
        if not model.strip() or (preview_text is not None and (not preview_text.strip() or len(preview_text) > 1000)):
            raise HTTPException(400, "model is required; preview_text must contain 1 to 1000 characters")
        return await create_voice(
            model,
            "cloned",
            auth,
            request.headers.get("Idempotency-Key"),
            preview_text=preview_text,
            audio=("reference" + extension, content, audio.content_type),
        )
    finally:
        await audio.close()


@router.get("/v1/audio/voices", response_model=VoiceList, response_model_exclude_none=True)
async def list_voices(model: str = Query(..., min_length=1), auth: UserAPIKeyAuth = Depends(user_api_key_auth)):
    db = table()
    deployment = await choose_deployment(model, auth)
    data = await upstream(deployment, "get_voice", payload={"voice_type": "system"})
    system = [
        {
            "voice_id": v["voice_id"],
            "type": "system",
            "name": v.get("voice_name"),
            "description": v.get("description", []),
            "status": "ready",
        }
        for v in data.get("system_voice", [])
    ]
    # Newly created voices are available here even before MiniMax's first-use activation.
    bindings = {account_binding(d) for d in deployments_for(model, auth)}
    rows = await db.find_many(where={"file_purpose": PURPOSE, "created_by": owner_scope(auth)})
    custom = [public_voice(r) for r in rows if metadata(r).get("account_binding") in bindings]
    return {"object": "list", "data": system + custom}


async def prepare_speech(data: dict, auth: UserAPIKeyAuth) -> dict:
    """Bind a managed voice to its account before dispatch, using the public model's scope."""
    from litellm.proxy import proxy_server

    result = normalize_speech_request(data)
    llm_router = proxy_server.llm_router
    if llm_router is None or not isinstance(llm_router, litellm.Router):
        if str(result.get("voice") or "").startswith(VOICE_PREFIX):
            raise HTTPException(503, "Managed voices require a configured model router")
        return result
    candidates = deployments_for(result["model"], auth)
    if not candidates or not all(
        d["litellm_params"].get("custom_llm_provider") == "minimax"
        or d["litellm_params"]["model"].startswith("minimax/")
        for d in candidates
    ):
        if str(result.get("voice") or "").startswith(VOICE_PREFIX):
            raise HTTPException(400, "Managed voices require a MiniMax model group")
        return result
    forbidden = {"api_base", "api_key", "custom_llm_provider", "user_config", "specific_deployment"}
    if forbidden.intersection(result):
        raise HTTPException(400, "MiniMax voice requests must use server-configured model credentials")
    # Validate all explicit controls before issuing even the read-only voice-list call.
    validation_data = dict(result)
    if not result.get("voice"):
        validation_data["default_voice_id"] = "validation-placeholder"
    build_minimax_params(
        {k: result[k] for k in ("instructions", "speed", "response_format") if k in result},
        result.get("voice"),
        validation_data,
    )
    voice_id = result.get("voice")
    binding = None
    row = None
    if isinstance(voice_id, str) and voice_id.startswith(VOICE_PREFIX):
        row = await table().find_unique(where={"unified_object_id": voice_id})
        if row is None or row.file_purpose != PURPOSE or row.created_by != owner_scope(auth):
            raise HTTPException(404, "Voice not found")
        if row.status != "ready":
            raise HTTPException(409, "Voice creation is incomplete; check its original status")
        binding = metadata(row)["account_binding"]
    deployment = await choose_deployment(result["model"], auth, binding)
    if row:
        voice_id = metadata(row)["upstream_voice_id"]
    else:
        voice_id = voice_id or deployment["litellm_params"].get("default_voice_id")
        if not voice_id:
            raise HTTPException(400, "voice_setting.voice_id is required unless default_voice_id is configured")
        native = await upstream(deployment, "get_voice", payload={"voice_type": "system"})
        if voice_id not in {v["voice_id"] for v in native.get("system_voice", [])}:
            raise HTTPException(400, "Use a system voice_id or a managed voice created by this caller")
    result["voice"] = voice_id
    result["voice_setting"] = {**result.get("voice_setting", {}), "voice_id": voice_id}
    if isinstance(result.get("extra_body", {}).get("voice_setting"), dict):
        result["extra_body"]["voice_setting"].pop("voice_id", None)
    result["model"] = str(deployment["model_info"]["id"])
    # Bound audio must never retry/fall back to another account or create a second paid result.
    result.update(disable_fallbacks=True, num_retries=0, max_retries=0)
    return result
