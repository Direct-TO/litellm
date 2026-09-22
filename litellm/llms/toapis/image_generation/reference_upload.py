"""Convert inline image references to ToAPIs URLs before generation submission."""

import base64
import binascii
from collections.abc import Mapping, Sequence
from types import MappingProxyType
from typing import Final, NamedTuple

import httpx
from pydantic import TypeAdapter

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.submission_utils import mark_reexecution_blocked, mark_submission_outcome
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler, HTTPHandler

from ..common_utils import parse_toapis_image_upload

MAX_IMAGE_BYTES: Final = 10 * 1024 * 1024
_EXTENSIONS: Final = MappingProxyType(
    {"image/png": "png", "image/jpeg": "jpg", "image/webp": "webp", "image/gif": "gif"}
)
_REFERENCE_OBJECT: Final = TypeAdapter(dict[str, object])
_REFERENCE_ITEMS: Final = TypeAdapter(tuple[object, ...])


class InlineImage(NamedTuple):
    content: bytes
    mime_type: str
    filename: str


def decode_reference(value: str) -> InlineImage | None:
    """Validate without I/O; None denotes an existing HTTP(S) reference."""
    if not value.lower().startswith("data:"):
        url: Final = httpx.URL(value)
        if url.scheme not in ("http", "https") or not url.host or url.userinfo:
            raise ValueError("image_url requires an HTTP(S) URL or a base64 image Data URL")
        return None
    header, separator, encoded = value.partition(",")
    mime_type: Final = header[5:].removesuffix(";base64").lower()
    if not separator or not header.endswith(";base64") or mime_type not in _EXTENSIONS:
        raise ValueError("image_url Data URLs must use base64 PNG, JPEG, WebP or GIF")
    if len(encoded) > 4 * ((MAX_IMAGE_BYTES + 2) // 3):
        raise ValueError("image_url decoded image exceeds the 10 MiB upload limit")
    try:
        content: Final = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("image_url contains invalid base64 image data") from exc
    if not content:
        raise ValueError("image_url contains an empty base64 image")
    if len(content) > MAX_IMAGE_BYTES:
        raise ValueError("image_url decoded image exceeds the 10 MiB upload limit")
    return InlineImage(content, mime_type, "reference." + _EXTENSIONS[mime_type])


def _reference_url(item: object) -> str:
    value: Final = _REFERENCE_OBJECT.validate_python(item).get("url") if isinstance(item, Mapping) else item
    if not isinstance(value, str):
        raise TypeError("ToAPIs reference images must be URL strings or {'url': ...} objects")
    return value


def _references(data: Mapping[str, object]) -> Mapping[str, InlineImage]:
    # Validate every reference before the first upload, including extra_body overrides.
    images: Final[dict[str, InlineImage]] = {}  # mutable-ok: request-local upload deduplication accumulator
    for field in ("reference_images", "image_urls"):
        items = data.get(field)
        if items is None:
            continue
        if not isinstance(items, Sequence) or isinstance(items, (str, bytes, bytearray)):
            raise TypeError("ToAPIs reference image fields must be arrays")
        for item in _REFERENCE_ITEMS.validate_python(items):
            url = _reference_url(item)
            if url not in images:
                image = decode_reference(url)
                if image is not None:
                    images[url] = image
    return MappingProxyType(images)


def _replace_references(data: Mapping[str, object], uploaded: Mapping[str, str]) -> Mapping[str, object]:
    if not uploaded:
        return data
    result: Final = dict(data)  # mutable-ok: private copy of the outgoing JSON body
    for field in ("reference_images", "image_urls"):
        if data.get(field) is None:
            continue
        items: list[object] = []  # mutable-ok: construct an ordered JSON array without mutating caller input
        for item in _REFERENCE_ITEMS.validate_python(data[field]):
            url = _reference_url(item)
            replacement = uploaded.get(url, url)
            items.append(
                {**_REFERENCE_OBJECT.validate_python(item), "url": replacement}  # mutable-ok: JSON object for upstream
                if isinstance(item, Mapping)
                else replacement
            )
        result[field] = items
    return MappingProxyType(result)


def _upload_headers(headers: Mapping[str, str]) -> Mapping[str, str]:
    return MappingProxyType(
        {
            key: value
            for key, value in headers.items()
            if key.lower() not in ("content-type", "content-length", "idempotency-key")
        }
    )


def _uploaded_url(response: httpx.Response) -> str:
    url: Final = parse_toapis_image_upload(response).url
    try:
        if url.lower().startswith("data:") or decode_reference(url) is not None:
            raise ValueError("Inline upload result")
    except (ValueError, httpx.InvalidURL) as exc:
        raise BaseLLMException(status_code=502, message="ToAPIs upload did not return an HTTP(S) URL") from exc
    return url


def _failure_status(exc: Exception) -> int:
    if isinstance(exc, BaseLLMException):
        return exc.status_code
    if isinstance(exc, httpx.HTTPStatusError):
        return exc.response.status_code if exc.response.status_code >= 400 else 502
    if isinstance(exc, (TypeError, ValueError, httpx.InvalidURL)):
        return 400
    return 408 if isinstance(exc, httpx.TimeoutException) else 502


def _failure(exc: Exception) -> BaseLLMException:
    # An upload is not a generation submission. Do not retry it through Router/VIP.
    return mark_reexecution_blocked(
        mark_submission_outcome(
            BaseLLMException(
                status_code=_failure_status(exc),
                message="ToAPIs reference image preparation failed before generation"
                + (": " + str(exc) if isinstance(exc, (TypeError, ValueError)) else " (" + type(exc).__name__ + ")"),
            ),
            "rejected",
        )
    )


def upload_references(
    data: Mapping[str, object],
    url: str,
    headers: Mapping[str, str],
    client: HTTPHandler,
    timeout: float | httpx.Timeout | None,
) -> Mapping[str, object]:
    try:
        images: Final = _references(data)
        uploaded: Final[dict[str, str]] = {}  # mutable-ok: accumulate results within this deployment call only
        for source, image in images.items():
            response = client.client.post(
                url,
                headers=_upload_headers(headers),
                data={"purpose": "generation"},  # mutable-ok: httpx multipart form fields
                files={"file": (image.filename, image.content, image.mime_type)},  # mutable-ok: httpx multipart files
                timeout=timeout if timeout is not None else client.client.timeout,
                follow_redirects=False,
            )
            response.raise_for_status()
            uploaded[source] = _uploaded_url(response)
        return _replace_references(data, uploaded)
    except Exception as exc:
        raise _failure(exc) from exc


async def async_upload_references(
    data: Mapping[str, object],
    url: str,
    headers: Mapping[str, str],
    client: AsyncHTTPHandler,
    timeout: float | httpx.Timeout | None,
) -> Mapping[str, object]:
    try:
        images: Final = _references(data)
        uploaded: Final[dict[str, str]] = {}  # mutable-ok: accumulate results within this deployment call only
        for source, image in images.items():
            response = await client.client.post(
                url,
                headers=_upload_headers(headers),
                data={"purpose": "generation"},  # mutable-ok: httpx multipart form fields
                files={"file": (image.filename, image.content, image.mime_type)},  # mutable-ok: httpx multipart files
                timeout=timeout if timeout is not None else client.timeout,
                follow_redirects=False,
            )
            response.raise_for_status()
            uploaded[source] = _uploaded_url(response)
        return _replace_references(data, uploaded)
    except Exception as exc:
        raise _failure(exc) from exc
