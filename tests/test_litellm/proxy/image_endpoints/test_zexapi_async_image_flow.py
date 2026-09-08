import base64
import importlib
import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from types import SimpleNamespace
from unittest.mock import AsyncMock

import httpx
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

import litellm
from litellm.litellm_core_utils.get_model_cost_map import GetModelCostMap
from litellm.llms.base_llm.submission_utils import get_provider_task_id, get_submission_outcome
from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.zexapi.image_generation.async_handler import ZexAPIAsyncImages
from litellm.proxy._types import UserAPIKeyAuth
from litellm.proxy.image_endpoints import endpoints

PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAIAAAACCAYAAABytg0kAAAACXBIWXMAAAPoAAAD6AG1e1JrAAAAEUlEQVR4nGPQq/3/H4QZYAwAWewKpRUlAtEAAAAASUVORK5CYII="
)


def model_list(base_url):
    return [
        {
            "model_name": "async-image",
            "litellm_params": {"model": "zexapi/" + model, "api_base": base_url, "api_key": "test-key"},
            "model_info": {"id": model, "supported_endpoints": ["/v1/images/generations", "/v1/images/edits"]},
        }
        for model in ["gpt-image-2", "gpt-image-2-2K", "gpt-image-2-4K"]
    ]


@pytest.mark.parametrize("model", ["gpt-image-2", "gpt-image-2-2K", "gpt-image-2-4K"])
def test_bundled_catalog_routes_async_images_as_images_not_videos(monkeypatch, model):
    monkeypatch.setattr(litellm, "model_cost", GetModelCostMap.load_local_model_cost_map())
    deployment = {
        "model_name": "async-image",
        "litellm_params": {"model": "zexapi/" + model, "api_key": "test-key"},
    }
    for call_type in ("image_generation", "aimage_generation", "image_edit", "aimage_edit"):
        assert litellm.Router._filter_deployments_by_supported_endpoint(
            "async-image", [deployment], {"_router_call_type": call_type}
        ) == [deployment]
    with pytest.raises(litellm.BadRequestError, match="no deployment supporting endpoint /v1/videos"):
        litellm.Router._filter_deployments_by_supported_endpoint(
            "async-image", [deployment], {"_router_call_type": "avideo_generation"}
        )


@pytest.fixture
def upstream():
    requests = []
    state = {"poll_error": False}

    class Upstream(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def reply(self, payload, status=200, mime="application/json"):
            data = payload if isinstance(payload, bytes) else json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", mime)
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def do_POST(self):
            body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
            requests.append(("POST", self.path, body))
            self.reply(
                {"id": "task-original", "object": "image", "status": "queued", "model": body["model"], "created_at": 1}
            )

        def do_GET(self):
            requests.append(("GET", self.path, None))
            if self.path == "/image.png":
                self.reply(PNG, mime="image/png")
            elif state["poll_error"]:
                self.reply({"error": "temporary query failure"}, status=503)
            else:
                # Real ZexAPI queries normalize the model back to the family name.
                self.reply(
                    {
                        "id": "task-original",
                        "object": "image",
                        "status": "completed",
                        "model": "gpt-image-2",
                        "url": f"http://127.0.0.1:{self.server.server_port}/image.png",
                        "created_at": 1,
                    }
                )

    server = ThreadingHTTPServer(("127.0.0.1", 0), Upstream)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_port}", requests, state
    finally:
        server.shutdown()
        server.server_close()
        thread.join()


def install_handler(monkeypatch):
    async def no_sleep(_):
        return None

    monkeypatch.setattr(
        importlib.import_module("litellm.images.main"), "zexapi_async_images", ZexAPIAsyncImages(async_sleep=no_sleep)
    )
    client = AsyncHTTPHandler()
    client.client = httpx.AsyncClient(trust_env=False)
    monkeypatch.setattr(
        "litellm.llms.zexapi.image_generation.async_handler.http_handlers.get_async_httpx_client", lambda **_: client
    )
    return client


def proxy_app(monkeypatch, router):
    async def add_data(**kwargs):
        return kwargs["data"]

    async def pre_call(**kwargs):
        return kwargs["data"]

    async def post_call(**kwargs):
        return kwargs["response"]

    logger = SimpleNamespace(
        pre_call_hook=pre_call,
        update_request_status=AsyncMock(),
        post_call_success_hook=post_call,
        post_call_failure_hook=AsyncMock(),
        post_call_response_headers_hook=AsyncMock(return_value={}),
    )
    monkeypatch.setattr("litellm.proxy.proxy_server.add_litellm_data_to_request", add_data)
    monkeypatch.setattr("litellm.proxy.proxy_server.general_settings", {})
    monkeypatch.setattr("litellm.proxy.proxy_server.llm_router", router)
    monkeypatch.setattr("litellm.proxy.proxy_server.proxy_config", {})
    monkeypatch.setattr("litellm.proxy.proxy_server.proxy_logging_obj", logger)
    monkeypatch.setattr("litellm.proxy.proxy_server.user_model", None)
    monkeypatch.setattr(
        endpoints.ProxyBaseLLMRequestProcessing, "get_custom_headers", classmethod(lambda *args, **kwargs: {})
    )

    async def edit_process(self, **kwargs):
        return await router.aimage_edit(
            **{
                key: self.data[key]
                for key in ["model", "prompt", "image", "mask", "n", "aspect_ratio", "resolution"]
                if key in self.data
            }
        )

    monkeypatch.setattr(endpoints.ProxyBaseLLMRequestProcessing, "base_process_llm_request", edit_process)
    app = FastAPI()
    app.include_router(endpoints.router)
    app.dependency_overrides[endpoints.user_api_key_auth] = UserAPIKeyAuth
    return app


@pytest.mark.parametrize("kind", ["generations", "edits"])
@pytest.mark.parametrize("tier", ["1K", "2K", "4K"])
@pytest.mark.parametrize("qualified_base_model", [False, True])
def test_public_image_api_returns_final_image_from_async_tcp_upstream(
    monkeypatch, upstream, kind, tier, qualified_base_model
):
    base, requests, _ = upstream
    client = install_handler(monkeypatch)
    deployments = model_list(base)
    for deployment in deployments:
        deployment["model_info"].pop("supported_endpoints")
        if qualified_base_model:
            deployment["model_info"]["base_model"] = deployment["litellm_params"]["model"]
    router = litellm.Router(model_list=deployments, num_retries=0)
    app = proxy_app(monkeypatch, router)
    fields = {"model": "async-image", "prompt": "blue circle", "aspect_ratio": "16:9", "resolution": tier}
    with TestClient(app) as proxy:
        if kind == "edits":
            response = proxy.post(
                "/v1/images/edits", data=fields, files=[("image", ("reference.png", PNG, "image/png"))]
            )
        else:
            response = proxy.post(
                "/v1/images/generations",
                json={**fields, "image_url": ["data:image/png;base64," + base64.b64encode(PNG).decode()]},
            )
        proxy.portal.call(client.client.aclose)
    assert response.status_code == 200
    assert [(method, path) for method, path, _ in requests] == [
        ("POST", "/v1/videos"),
        ("GET", "/v1/videos/task-original"),
    ]
    assert requests[0][2] == {
        "model": "gpt-image-2" if tier == "1K" else "gpt-image-2-" + tier,
        "prompt": "blue circle",
        "aspect_ratio": "16:9",
        "images": ["data:image/png;base64," + base64.b64encode(PNG).decode()],
    }
    image = response.json()["data"][0]
    assert image["url"] == base + "/image.png"
    assert "_hidden_params" not in response.json()
    assert httpx.get(image["url"], trust_env=False).content == PNG


@pytest.mark.asyncio
async def test_router_does_not_retry_or_fallback_after_query_failure(monkeypatch, upstream):
    base, requests, state = upstream
    state["poll_error"] = True
    client = install_handler(monkeypatch)
    deployments = model_list(base)
    deployments.append(
        {
            "model_name": "fallback-image",
            "litellm_params": {"model": "zexapi/gpt-image-2-2K", "api_base": base, "api_key": "test-key"},
        }
    )
    router = litellm.Router(model_list=deployments, num_retries=3, fallbacks=[{"async-image": ["fallback-image"]}])
    try:
        with pytest.raises(litellm.ServiceUnavailableError) as caught:
            await router.aimage_edit(model="async-image", image=PNG, prompt="edit", resolution="2K")
    finally:
        await client.client.aclose()
    assert get_submission_outcome(caught.value) == "accepted"
    assert get_provider_task_id(caught.value) == "task-original"
    assert [(method, path) for method, path, _ in requests] == [
        ("POST", "/v1/videos"),
        ("GET", "/v1/videos/task-original"),
    ]


@pytest.mark.asyncio
async def test_router_rejects_unsupported_mask_before_any_submission(monkeypatch, upstream):
    base, requests, _ = upstream
    client = install_handler(monkeypatch)
    router = litellm.Router(model_list=model_list(base), num_retries=0)
    try:
        with pytest.raises(litellm.BadRequestError, match="image edit parameter"):
            await router.aimage_edit(
                model="async-image", image=PNG, mask=PNG, prompt="edit", resolution="2K", drop_params=True
            )
    finally:
        await client.client.aclose()
    assert requests == []


@pytest.mark.asyncio
@pytest.mark.parametrize("entrypoint", ["sdk", "router"])
@pytest.mark.parametrize("params", [{"mask": PNG}, {"input_fidelity": "high"}])
async def test_generation_rejects_edit_semantics_even_with_drop_params(monkeypatch, upstream, entrypoint, params):
    base, requests, _ = upstream
    client = install_handler(monkeypatch)
    router = litellm.Router(model_list=model_list(base), num_retries=0)
    try:
        with pytest.raises(litellm.BadRequestError):
            if entrypoint == "router":
                await router.aimage_generation(
                    model="async-image", prompt="edit", resolution="2K", drop_params=True, **params
                )
            else:
                await litellm.aimage_generation(
                    model="zexapi/gpt-image-2-2K",
                    api_base=base,
                    api_key="test-key",
                    prompt="edit",
                    resolution="2K",
                    drop_params=True,
                    **params,
                )
    finally:
        await client.client.aclose()
    assert requests == []


@pytest.mark.asyncio
async def test_generation_honors_explicit_additional_drop_params(monkeypatch, upstream):
    base, requests, _ = upstream
    client = install_handler(monkeypatch)
    try:
        response = await litellm.aimage_generation(
            model="zexapi/gpt-image-2-2K",
            api_base=base,
            api_key="test-key",
            prompt="generate",
            mask=PNG,
            additional_drop_params=["mask"],
        )
    finally:
        await client.client.aclose()
    assert response.data[0].url == base + "/image.png"
    assert requests[0][2] == {"model": "gpt-image-2-2K", "prompt": "generate"}
    assert [method for method, _, _ in requests] == ["POST", "GET"]
