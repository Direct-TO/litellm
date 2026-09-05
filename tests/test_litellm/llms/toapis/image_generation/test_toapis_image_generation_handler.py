from collections.abc import Mapping
from unittest.mock import Mock

import httpx
import pytest

from litellm.llms.base_llm.chat.transformation import BaseLLMException
from litellm.llms.base_llm.submission_utils import get_provider_task_id, get_submission_outcome
from litellm.llms.toapis.image_generation.handler import ToAPISImageGeneration
from litellm.llms.toapis.image_generation.transformation import ToAPISImageGenerationConfig
from litellm.types.utils import ImageResponse


def _task_response(
    status: str,
    task_id: str = "task_img_123",
    result: dict | None = None,
    error: dict | None = None,
) -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "id": task_id,
            "object": "generation.task",
            "model": "gpt-image-2",
            "status": status,
            "progress": 100 if status in {"completed", "failed"} else 0,
            "created_at": 1703884800,
            "completed_at": 1703884810 if status == "completed" else None,
            "expires_at": 1703971210 if status == "completed" else None,
            "result": result,
            "error": error,
        },
    )


class _SyncSequenceClient:
    def __init__(self, responses: tuple[httpx.Response, ...]) -> None:
        self._responses = iter(responses)
        self.urls: list[str] = []

    def get(self, *, url: str, headers: Mapping[str, str], timeout: object) -> httpx.Response:
        self.urls.append(url)
        return next(self._responses)


class _AsyncSequenceClient:
    def __init__(self, responses: tuple[httpx.Response, ...]) -> None:
        self._responses = iter(responses)
        self.urls: list[str] = []

    async def get(self, *, url: str, headers: Mapping[str, str], timeout: object) -> httpx.Response:
        self.urls.append(url)
        return next(self._responses)


class _AdvancingClock:
    def __init__(self) -> None:
        self.now = 0.0
        self.sleeps: list[float] = []

    def monotonic(self) -> float:
        return self.now

    def sleep(self, delay: float) -> None:
        self.sleeps.append(delay)
        self.now += delay


def test_toapis_sync_polling_normalizes_pending_status_and_result():
    handler = ToAPISImageGeneration(sync_sleep=lambda _: None, monotonic=lambda: 0.0)
    client = _SyncSequenceClient(
        (
            _task_response(
                "completed",
                task_id="task/with query?",
                result={"type": "image", "data": [{"url": "https://files.example/image.png"}]},
            ),
        )
    )
    final_response = handler._poll_sync(
        initial_response=_task_response("pending", task_id="task/with query?"),
        complete_url="https://toapis.com/v1/images/generations",
        headers={"Authorization": "Bearer test-key"},
        client=client,
        timeout=10,
    )
    result = ToAPISImageGenerationConfig().transform_image_generation_response(
        model="gpt-image-2",
        raw_response=final_response,
        model_response=ImageResponse(),
        logging_obj=Mock(),
        request_data={},
        optional_params={},
        litellm_params={},
        encoding=None,
    )

    assert client.urls == ["https://toapis.com/v1/images/generations/task%2Fwith%20query%3F"]
    assert result.data[0].url == "https://files.example/image.png"
    assert result.data[0].provider_specific_fields == {
        "task_id": "task/with query?",
        "expires_at": 1703971210,
    }


@pytest.mark.asyncio
async def test_toapis_async_polling_matches_sync_behavior():
    async def no_sleep(_: float) -> None:
        return None

    handler = ToAPISImageGeneration(async_sleep=no_sleep, monotonic=lambda: 0.0)
    client = _AsyncSequenceClient(
        (
            _task_response(
                "completed",
                result={"type": "image", "data": [{"url": "https://files.example/image.png"}]},
            ),
        )
    )

    response = await handler._poll_async(
        initial_response=_task_response("queued"),
        complete_url="https://toapis.com/v1/images/generations",
        headers={"Authorization": "Bearer test-key"},
        client=client,
        timeout=10,
    )

    assert response.json()["status"] == "completed"
    assert client.urls == ["https://toapis.com/v1/images/generations/task_img_123"]


def test_toapis_failed_task_raises_provider_error():
    handler = ToAPISImageGeneration(sync_sleep=lambda _: None, monotonic=lambda: 0.0)

    with pytest.raises(BaseLLMException, match="generation_failed: content policy") as exc_info:
        handler._poll_sync(
            initial_response=_task_response(
                "failed",
                error={"code": "generation_failed", "message": "content policy"},
            ),
            complete_url="https://toapis.com/v1/images/generations",
            headers={},
            client=_SyncSequenceClient(()),
            timeout=10,
        )

    assert exc_info.value.status_code == 500
    assert get_submission_outcome(exc_info.value) == "accepted"
    assert get_provider_task_id(exc_info.value) == "task_img_123"


def test_toapis_retry_after_cannot_exceed_polling_deadline():
    clock = _AdvancingClock()
    handler = ToAPISImageGeneration(sync_sleep=clock.sleep, monotonic=clock.monotonic)
    client = _SyncSequenceClient((httpx.Response(429, headers={"Retry-After": "3600"}),))

    with pytest.raises(BaseLLMException) as exc_info:
        handler._poll_sync(
            initial_response=_task_response("queued"),
            complete_url="https://toapis.com/v1/images/generations",
            headers={},
            client=client,
            timeout=10,
            max_wait=120,
            interval=5,
        )

    assert exc_info.value.status_code == 408
    assert get_submission_outcome(exc_info.value) == "accepted"
    assert get_provider_task_id(exc_info.value) == "task_img_123"
    assert clock.sleeps == [5, 115]


def test_toapis_initial_http_failure_is_safe_to_fail_over():
    handler = ToAPISImageGeneration(sync_sleep=lambda _: None, monotonic=lambda: 0.0)

    with pytest.raises(BaseLLMException) as exc_info:
        handler._poll_sync(
            initial_response=httpx.Response(503, text="no available channel"),
            complete_url="https://toapis.com/v1/images/generations",
            headers={},
            client=_SyncSequenceClient(()),
            timeout=10,
        )

    assert exc_info.value.status_code == 503
    assert get_submission_outcome(exc_info.value) == "rejected"
    assert get_provider_task_id(exc_info.value) is None


def test_toapis_poll_http_failure_preserves_accepted_task_provenance():
    handler = ToAPISImageGeneration(sync_sleep=lambda _: None, monotonic=lambda: 0.0)

    with pytest.raises(BaseLLMException) as exc_info:
        handler._poll_sync(
            initial_response=_task_response("queued", task_id="accepted-task"),
            complete_url="https://toapis.com/v1/images/generations",
            headers={},
            client=_SyncSequenceClient((httpx.Response(503, text="poll unavailable"),)),
            timeout=10,
        )

    assert exc_info.value.status_code == 503
    assert get_submission_outcome(exc_info.value) == "accepted"
    assert get_provider_task_id(exc_info.value) == "accepted-task"
