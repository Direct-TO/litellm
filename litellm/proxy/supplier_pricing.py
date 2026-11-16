"""Read supplier account prices for already-authorized router deployments. No inference or spend."""

import asyncio
from datetime import datetime, timezone
from typing import Any

from litellm.llms.custom_httpx.http_handler import AsyncHTTPHandler
from litellm.llms.toapis.common_utils import build_toapis_endpoint, get_toapis_api_key
from litellm.secret_managers.main import get_secret_str


def _resolve(value: Any) -> str | None:
    if not isinstance(value, str):
        return None
    return get_secret_str(value.split("/", 1)[1]) if value.startswith("os.environ/") else value


async def attach_supplier_pricing(models: list[dict], router: Any, team_id: str | None = None) -> list[dict]:
    """Caller must pass only visible deployments, after all key/team access filters."""
    semaphore = asyncio.Semaphore(4)

    async def read(model: dict) -> dict:
        result = {**model}
        info: dict = {"status": "unavailable", "prices": [], "checked_at": datetime.now(timezone.utc).isoformat()}
        result["supplier_pricing"] = info
        deployment_id = (model.get("model_info") or {}).get("id")
        if not isinstance(deployment_id, str):
            return result
        credentials = router.get_deployment_credentials_with_provider(deployment_id, team_id=team_id)
        if not credentials:
            return result
        provider = credentials.get("custom_llm_provider")
        upstream_model = credentials.get("model", "")
        if not provider and isinstance(upstream_model, str) and "/" in upstream_model:
            provider = upstream_model.split("/", 1)[0]
        info["provider"] = provider
        if provider != "toapis":
            info["status"] = "unsupported"
            return result
        upstream_model = upstream_model.removeprefix("toapis/")
        info["upstream_model"] = upstream_model
        key = get_toapis_api_key(_resolve(credentials.get("api_key")))
        if not key:
            return result
        async with semaphore:
            client = AsyncHTTPHandler(timeout=12)
            try:
                response = await client.get(
                    build_toapis_endpoint(_resolve(credentials.get("api_base")), "/v1/pricing"),
                    params={"model": upstream_model},
                    headers={"Authorization": f"Bearer {key}"},
                    follow_redirects=False,
                    timeout=12,
                )
                response.raise_for_status()
                payload = response.json()
                if payload.get("currency") != "USD":
                    return result
                rows = [
                    row
                    for item in payload.get("data", [])
                    if item.get("id") == upstream_model
                    for row in item.get("prices", [])
                ]
                # Explicit projection: never return credentials, raw errors, or unrelated response fields.
                fields = (
                    "group",
                    "conditions",
                    "charge_type",
                    "price_basis",
                    "unit",
                    "unit_price",
                    "input_unit_price",
                    "minimum_charge_usd",
                )
                info.update(
                    status="available" if rows else "unavailable",
                    currency="USD",
                    account_adjusted=True,
                    prices=[
                        {field: row[field] for field in fields if field in row} for row in rows if isinstance(row, dict)
                    ],
                )
            except Exception:
                # A missing/unavailable catalogue is not a zero price. Keep normal model discovery usable.
                info["status"] = "unavailable"
            finally:
                await client.close()
        return result

    return list(await asyncio.gather(*(read(model) for model in models)))
