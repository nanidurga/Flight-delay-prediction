"""
render.py — Thin async client for the Render REST API v1.

All methods raise RenderError on HTTP failure.
No business logic — sequencing lives in orchestrator.py.
"""

import httpx

RENDER_API = "https://api.render.com/v1"


class RenderError(Exception):
    """Raised when a Render API call fails."""


class RenderClient:
    def __init__(self, api_key: str):
        self._headers = {
            "Authorization": f"Bearer {api_key}",
            "Accept": "application/json",
            "Content-Type": "application/json",
        }

    async def _get_owner_id(self) -> str:
        """Fetch the first owner (user or team) ID for this API key."""
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    f"{RENDER_API}/owners",
                    headers=self._headers,
                )
                resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RenderError(f"get_owner_id failed: {exc}") from exc
        owners = resp.json()
        if not owners:
            raise RenderError("No owners found for this Render API key.")
        return owners[0]["owner"]["id"]

    async def create_service(self, github_repo: str, service_name: str) -> dict:
        """
        Create a new Render web service linked to a GitHub repo.

        Returns {"service_id": str, "deploy_id": str}
        """
        owner_id = await self._get_owner_id()
        payload = {
            "type": "web_service",
            "name": service_name,
            "ownerId": owner_id,
            "serviceDetails": {
                "runtime": "python",
                "plan": "free",
                "region": "oregon",
                "envSpecificDetails": {
                    "buildCommand": "pip install -r requirements.txt",
                    "startCommand": "uvicorn api.main:app --host 0.0.0.0 --port $PORT",
                },
                "envVars": [{"key": "PYTHON_VERSION", "value": "3.11.0"}],
            },
            "repo": f"https://github.com/{github_repo}",
            "branch": "master",
            "autoDeploy": "yes",
        }
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{RENDER_API}/services",
                    headers=self._headers,
                    json=payload,
                )
                resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RenderError(f"create_service failed: {exc}") from exc

        data = resp.json()
        return {
            "service_id": data["service"]["id"],
            "deploy_id": data.get("deployId", ""),
        }

    async def trigger_deploy(self, service_id: str) -> str:
        """Trigger a new deploy on an existing service. Returns deploy_id."""
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{RENDER_API}/services/{service_id}/deploys",
                    headers=self._headers,
                    json={},
                )
                resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RenderError(f"trigger_deploy failed: {exc}") from exc

        return resp.json()["deploy"]["id"]

    async def get_deploy(self, service_id: str, deploy_id: str) -> dict:
        """Return deploy info including status field."""
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    f"{RENDER_API}/services/{service_id}/deploys/{deploy_id}",
                    headers=self._headers,
                )
                resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RenderError(f"get_deploy failed: {exc}") from exc

        data = resp.json()
        # API returns the deploy object directly (no "deploy" wrapper)
        return data.get("deploy", data)

    async def get_service_status(self, service_id: str) -> dict:
        """Return service state and live URL."""
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    f"{RENDER_API}/services/{service_id}",
                    headers=self._headers,
                )
                resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RenderError(f"get_service_status failed: {exc}") from exc

        data = resp.json()
        # API returns the service object directly (no "service" wrapper)
        svc = data.get("service", data)
        suspended = svc.get("suspended", "not_suspended")
        details = svc.get("serviceDetails", {})
        # URL may be at serviceDetails.url or serviceDetails.serviceDetails.url depending on API version
        url = details.get("url") or details.get("serviceDetails", {}).get("url", "")
        # Fallback: derive from service slug
        if not url:
            slug = svc.get("slug", "")
            url = f"https://{slug}.onrender.com" if slug else ""
        return {
            "service_id": service_id,
            "state": "suspended" if suspended == "suspended" else "live",
            "url": url,
        }

    async def get_logs(self, service_id: str, tail: int = 100) -> list[str]:
        """Return last `tail` log lines as a list of strings."""
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    f"{RENDER_API}/services/{service_id}/logs",
                    headers=self._headers,
                    params={"tail": tail},
                )
                resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RenderError(f"get_logs failed: {exc}") from exc

        return [entry["message"] for entry in resp.json()]
