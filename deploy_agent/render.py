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

    async def create_service(self, github_repo: str, service_name: str) -> dict:
        """
        Create a new Render web service linked to a GitHub repo.

        Returns {"service_id": str, "deploy_id": str}
        """
        payload = {
            "type": "web_service",
            "name": service_name,
            "ownerId": None,
            "serviceDetails": {
                "runtime": "python",
                "buildCommand": "pip install -r requirements.txt",
                "startCommand": "uvicorn api.main:app --host 0.0.0.0 --port $PORT",
                "envVars": [{"key": "PYTHON_VERSION", "value": "3.11.0"}],
                "plan": "free",
                "region": "oregon",
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

    async def get_deploy(self, deploy_id: str) -> dict:
        """Return deploy info including status field."""
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    f"{RENDER_API}/deploys/{deploy_id}",
                    headers=self._headers,
                )
                resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise RenderError(f"get_deploy failed: {exc}") from exc

        return resp.json()["deploy"]

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

        svc = resp.json()["service"]
        suspended = svc.get("suspended", "not_suspended")
        return {
            "service_id": service_id,
            "state": "suspended" if suspended == "suspended" else "live",
            "url": svc.get("serviceDetails", {}).get("url", ""),
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
