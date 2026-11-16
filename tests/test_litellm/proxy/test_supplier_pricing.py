from unittest.mock import MagicMock

import httpx
import pytest

from litellm.proxy import supplier_pricing


@pytest.mark.asyncio
async def test_prices_use_each_deployments_credential_and_preserve_discount(monkeypatch):
    requests = []

    class Client:
        def __init__(self, **kwargs):
            pass

        async def get(self, url, **kwargs):
            requests.append((url, kwargs))
            price = "9" if kwargs["headers"]["Authorization"] == "Bearer account-one" else "8"
            return httpx.Response(
                200,
                request=httpx.Request("GET", url),
                json={
                    "currency": "USD",
                    "data": [
                        {
                            "id": "seedance-2-5",
                            "prices": [
                                {
                                    "unit": "1m_tokens",
                                    "unit_price": price,
                                    "conditions": {"resolution": "720p", "has_video_input": False},
                                    "private_field": "omit",
                                }
                            ],
                        }
                    ],
                },
            )

        async def close(self):
            pass

    monkeypatch.setattr(supplier_pricing, "AsyncHTTPHandler", Client)
    router = MagicMock()
    router.get_deployment_credentials_with_provider.side_effect = [
        {
            "model": "toapis/seedance-2-5",
            "custom_llm_provider": "toapis",
            "api_key": "account-one",
            "api_base": "https://toapis.cn/v1",
        },
        {
            "model": "toapis/seedance-2-5",
            "custom_llm_provider": "toapis",
            "api_key": "account-two",
            "api_base": "https://toapis.cn",
        },
        {"model": "guanghe/video", "custom_llm_provider": "guanghe", "api_key": "other-secret"},
    ]
    models = [{"model_name": "video", "model_info": {"id": str(i)}} for i in range(3)]
    result = await supplier_pricing.attach_supplier_pricing(models, router, "team")
    assert len(requests) == 2
    assert all(url == "https://toapis.cn/v1/pricing" for url, _ in requests)
    assert all(options["follow_redirects"] is False for _, options in requests)
    assert [row["supplier_pricing"]["prices"][0]["unit_price"] for row in result[:2]] == ["9", "8"]
    assert result[2]["supplier_pricing"]["status"] == "unsupported"
    assert result[0]["supplier_pricing"]["account_adjusted"] is True
    assert "private_field" not in str(result)
    assert "account-one" not in str(result)
    assert all("supplier_pricing" not in row for row in models)


@pytest.mark.asyncio
async def test_failed_catalogue_remains_unknown_not_free(monkeypatch):
    class Client:
        def __init__(self, **kwargs):
            pass

        async def get(self, url, **kwargs):
            raise RuntimeError("private upstream error")

        async def close(self):
            pass

    monkeypatch.setattr(supplier_pricing, "AsyncHTTPHandler", Client)
    router = MagicMock()
    router.get_deployment_credentials_with_provider.return_value = {
        "model": "toapis/seedance-2",
        "custom_llm_provider": "toapis",
        "api_key": "secret",
    }
    result = await supplier_pricing.attach_supplier_pricing([{"model_info": {"id": "one"}}], router)
    assert result[0]["supplier_pricing"]["status"] == "unavailable"
    assert result[0]["supplier_pricing"]["prices"] == []
    assert "private upstream error" not in str(result)
