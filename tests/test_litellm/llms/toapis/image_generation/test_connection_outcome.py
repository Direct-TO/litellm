from types import SimpleNamespace
from unittest.mock import Mock

import httpx
import pytest
from aiohttp import ClientConnectorError

from litellm.llms.base_llm.submission_utils import get_provider_task_id, get_submission_outcome, is_reexecution_blocked
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.toapis.image_generation.handler import ToAPISImageGeneration


@pytest.mark.asyncio
@pytest.mark.parametrize("phase", ["create", "poll", "read"])
async def test_connect_failure_is_rejected_only_before_acceptance(phase):
    requests = []

    async def transport(request):
        requests.append(request.method)
        if phase == "poll" and request.method == "POST":
            return httpx.Response(200, json={"id": "accepted-task", "object": "generation.task", "status": "queued"})
        if phase == "read":
            raise httpx.ReadError("response lost", request=request)
        connector_error = ClientConnectorError(
            SimpleNamespace(host="api.toapis.com", port=443, ssl=True), OSError(111, "connection refused")
        )
        raise httpx.ConnectError(str(connector_error), request=request) from connector_error

    async def no_sleep(_):
        pass

    handler = ToAPISImageGeneration(async_sleep=no_sleep)
    client = object.__new__(AsyncHTTPHandler)
    client.timeout = 10
    async with httpx.AsyncClient(transport=httpx.MockTransport(transport)) as http:
        client.client = http
        with pytest.raises(httpx.RequestError) as caught:
            await handler.async_image_generation(
                model="gpt-image-2", prompt="test", optional_params={}, litellm_params={},
                logging_obj=Mock(), timeout=10, api_key="test-key", client=client,
            )
    assert requests == (["POST", "GET"] if phase == "poll" else ["POST"])
    assert get_submission_outcome(caught.value) == {"create": "rejected", "poll": "accepted", "read": "unknown"}[phase]
    assert get_provider_task_id(caught.value) == ("accepted-task" if phase == "poll" else None)
    if phase == "create":
        assert is_reexecution_blocked(caught.value)
