"""Stage Guanghe image/video references before submitting a generation task."""

import asyncio
import mimetypes
import time
from collections.abc import Mapping
from typing import Final, NamedTuple

import httpx
from pydantic import TypeAdapter, ValidationError

from litellm.litellm_core_utils.url_utils import validate_url
from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.submission_utils import mark_reexecution_blocked, mark_submission_outcome
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

from .transformation import build_endpoint, parse_guanghe_response

MAX_UPLOAD_BYTES: Final = 50 * 1024 * 1024
MAX_REFERENCE_REDIRECTS: Final = 5
_URLS: Final = TypeAdapter(list[str])


class _Reference(NamedTuple):
    field: str
    position: int | None
    kind: str
    url: str


def _failure(message: str, status: int = 400) -> BaseLLMException:
    # No generation has been submitted. Do not let Router retry the upload or
    # silently turn a failed staging operation into a different provider call.
    return mark_reexecution_blocked(
        mark_submission_outcome(
            BaseLLMException(status_code=status, message="Guanghe reference: " + message), "rejected"
        )
    )


def _references(params: Mapping[str, object]) -> list[_Reference]:
    result: list[_Reference] = []
    for field, kind in (("imageUrls", "image"), ("videoUrls", "video")):
        value = _urls(params.get(field, []))
        for index, url in enumerate(value):
            result.append(_Reference(field, index, kind, url))
    for field in ("firstFrameUrl", "lastFrameUrl"):
        value = params.get(field)
        if value is not None:
            if not isinstance(value, str):
                raise _failure(f"{field} must be a URL")
            result.append(_Reference(field, None, "image", value))
    return result


def _urls(value: object) -> list[str]:
    try:
        return _URLS.validate_python(value, strict=True)
    except ValidationError as exc:
        raise _failure("reference fields must be arrays of URLs") from exc


def _timeout(original: float | httpx.Timeout, deadline: float) -> httpx.Timeout:
    remaining = deadline - time.monotonic()
    if remaining <= 0:
        raise _failure("download/upload timed out before generation", 408)
    configured = original if isinstance(original, httpx.Timeout) else httpx.Timeout(original)
    return httpx.Timeout(
        **{k: min(v, remaining) if v is not None else remaining for k, v in configured.as_dict().items()}
    )


def _deadline(timeout: float | httpx.Timeout) -> float:
    seconds = timeout.read if isinstance(timeout, httpx.Timeout) else timeout
    return time.monotonic() + (seconds if seconds is not None else 600.0)


def _validated_url(url: str) -> tuple[str, str]:
    parsed = httpx.URL(url)
    if parsed.scheme not in ("http", "https") or not parsed.host or parsed.userinfo:
        raise _failure("source URL must use HTTP(S) without credentials")
    return validate_url(url)


def _download_mime(response: httpx.Response, url: str, kind: str) -> str:
    if response.status_code != 200:
        raise _failure(f"download failed (HTTP {response.status_code})", 502)
    response_headers = dict(response.headers.multi_items())
    length = response_headers.get("content-length")
    if length is not None and (not length.isdigit() or int(length) > MAX_UPLOAD_BYTES):
        raise _failure("reference exceeds the 50 MiB limit or has an invalid Content-Length")
    mime = response_headers.get("content-type", "").split(";", 1)[0].strip().lower()
    if mime in ("", "application/octet-stream"):
        mime = mimetypes.guess_type(httpx.URL(url).path)[0] or ""
    if not mime.startswith(kind + "/"):
        raise _failure(f"download did not return an {kind} file")
    return mime


def _append(content: bytearray, chunk: bytes, timeout: float | httpx.Timeout, deadline: float) -> None:
    _timeout(timeout, deadline)
    if len(content) + len(chunk) > MAX_UPLOAD_BYTES:
        raise _failure("reference exceeds the 50 MiB limit")
    content.extend(chunk)


def _redirect(response: httpx.Response, url: str) -> str:
    location = dict(response.headers.multi_items()).get("location")
    if not location:
        raise _failure("download redirect has no Location")
    return str(httpx.URL(url).join(location))


def _download(
    client: httpx.Client, ref: _Reference, timeout: float | httpx.Timeout, deadline: float
) -> tuple[bytes, str]:
    url = ref.url
    for _ in range(MAX_REFERENCE_REDIRECTS):
        target, host = _validated_url(url)
        with client.stream("GET", target, headers={"Host": host}, timeout=_timeout(timeout, deadline)) as response:
            if response.is_redirect:
                url = _redirect(response, url)
                continue
            mime = _download_mime(response, url, ref.kind)
            content = bytearray()
            for chunk in response.iter_bytes(8192):
                _append(content, chunk, timeout, deadline)
            if not content:
                raise _failure("download returned an empty file")
            return bytes(content), mime
    raise _failure("too many download redirects")


async def _async_download(
    client: httpx.AsyncClient, ref: _Reference, timeout: float | httpx.Timeout, deadline: float
) -> tuple[bytes, str]:
    url = ref.url
    for _ in range(MAX_REFERENCE_REDIRECTS):
        target, host = await asyncio.to_thread(_validated_url, url)
        async with client.stream(
            "GET", target, headers={"Host": host}, timeout=_timeout(timeout, deadline)
        ) as response:
            if response.is_redirect:
                url = _redirect(response, url)
                continue
            mime = _download_mime(response, url, ref.kind)
            content = bytearray()
            async for chunk in response.aiter_bytes(8192):
                _append(content, chunk, timeout, deadline)
            if not content:
                raise _failure("download returned an empty file")
            return bytes(content), mime
    raise _failure("too many download redirects")


def _upload_headers(headers: Mapping[str, str]) -> dict[str, str]:
    return {k: v for k, v in headers.items() if k.lower() not in ("content-type", "content-length", "idempotency-key")}


def _uploaded_url(response: httpx.Response) -> str:
    if response.is_redirect:
        raise _failure("upload endpoint returned a redirect", 502)
    data = parse_guanghe_response(response)  # HTTP 200 + success=false is a provider failure.
    url = data.get("signed_url") or data.get("url")
    if not isinstance(url, str):
        raise _failure("upload returned no usable URL", 502)
    parsed = httpx.URL(url)
    if parsed.scheme not in ("http", "https") or not parsed.host or parsed.userinfo:
        raise _failure("upload returned an invalid URL", 502)
    if data.get("status") not in (None, "ready"):
        raise _failure("uploaded file is not ready", 502)
    return url


def _replace(params: dict[str, object], ref: _Reference, url: str) -> None:
    if ref.position is None:
        params[ref.field] = url
    else:
        copied = _urls(params[ref.field])
        copied[ref.position] = url
        params[ref.field] = copied


def prepare_reference_uploads(
    params: Mapping[str, object],
    api_base: str | None,
    headers: Mapping[str, str],
    client: HTTPHandler,
    timeout: float | httpx.Timeout,
) -> dict[str, object]:
    prepared = dict(params)
    references = _references(params)
    if not references:
        return prepared
    uploaded: dict[tuple[str, str], str] = {}
    deadline = _deadline(timeout)
    try:
        # A separate, credential-free client prevents provider Authorization,
        # caller extra headers and cookies from reaching arbitrary media hosts.
        with httpx.Client(follow_redirects=False) as media:
            for ref in references:
                identity = (ref.kind, ref.url)
                if identity not in uploaded:
                    content, mime = _download(media, ref, timeout, deadline)
                    response = client.client.post(
                        build_endpoint(api_base, "files/upload"),
                        headers=_upload_headers(headers),
                        data={"file_type": "input_material", "source": "upload"},
                        files={"file": ("reference" + (mimetypes.guess_extension(mime) or ""), content, mime)},
                        timeout=_timeout(timeout, deadline),
                        follow_redirects=False,
                    )
                    uploaded[identity] = _uploaded_url(response)
                    del content
                _replace(prepared, ref, uploaded[identity])
            _timeout(timeout, deadline)
    except BaseLLMException as exc:
        raise mark_reexecution_blocked(mark_submission_outcome(exc, "rejected"))
    except (httpx.HTTPError, httpx.InvalidURL, ValueError) as exc:
        raise _failure(f"download/upload failed ({type(exc).__name__}); generation was not submitted", 502) from exc
    return prepared


async def async_prepare_reference_uploads(
    params: Mapping[str, object],
    api_base: str | None,
    headers: Mapping[str, str],
    client: AsyncHTTPHandler,
    timeout: float | httpx.Timeout,
) -> dict[str, object]:
    prepared = dict(params)
    references = _references(params)
    if not references:
        return prepared
    uploaded: dict[tuple[str, str], str] = {}
    deadline = _deadline(timeout)
    try:
        async with httpx.AsyncClient(follow_redirects=False) as media:
            for ref in references:
                identity = (ref.kind, ref.url)
                if identity not in uploaded:
                    content, mime = await _async_download(media, ref, timeout, deadline)
                    response = await client.client.post(
                        build_endpoint(api_base, "files/upload"),
                        headers=_upload_headers(headers),
                        data={"file_type": "input_material", "source": "upload"},
                        files={"file": ("reference" + (mimetypes.guess_extension(mime) or ""), content, mime)},
                        timeout=_timeout(timeout, deadline),
                        follow_redirects=False,
                    )
                    uploaded[identity] = _uploaded_url(response)
                    del content
                _replace(prepared, ref, uploaded[identity])
            _timeout(timeout, deadline)
    except BaseLLMException as exc:
        raise mark_reexecution_blocked(mark_submission_outcome(exc, "rejected"))
    except (httpx.HTTPError, httpx.InvalidURL, ValueError) as exc:
        raise _failure(f"download/upload failed ({type(exc).__name__}); generation was not submitted", 502) from exc
    return prepared
