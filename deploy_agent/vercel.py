"""
vercel.py — Thin async client for the Vercel REST API.

Deploy strategy:
  1. Create project linked to GitHub repo (POST /v9/projects)
  2. Set VITE_API_URL env var (POST /v9/projects/{id}/env)
  3. Create a deploy hook (POST /v9/projects/{id}/deploy-hooks)
  4. Trigger deploy via hook URL (POST {hookUrl})
  5. Poll latest deployment for READY state

All methods raise VercelError on HTTP failure.
"""

import httpx

VERCEL_API = "https://api.vercel.com"


class VercelError(Exception):
    """Raised when a Vercel API call fails."""


class VercelClient:
    def __init__(self, token: str):
        self._headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json",
        }

    async def create_project(self, github_repo: str, project_name: str) -> str:
        """
        Create a Vercel project linked to a GitHub repo.
        Sets root_directory=frontend, framework=vite.
        Returns project_id string.
        """
        payload = {
            "name": project_name,
            "framework": "vite",
            "rootDirectory": "frontend",
            "buildCommand": "npm run build",
            "outputDirectory": "dist",
            "gitRepository": {
                "type": "github",
                "repo": github_repo,
            },
        }
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{VERCEL_API}/v9/projects",
                    headers=self._headers,
                    json=payload,
                )
                resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise VercelError(f"create_project failed: {exc}") from exc

        return resp.json()["id"]

    async def set_env_var(self, project_id: str, key: str, value: str) -> None:
        """Add or update an environment variable on a Vercel project."""
        payload = {
            "key": key,
            "value": value,
            "type": "plain",
            "target": ["production", "preview", "development"],
        }
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{VERCEL_API}/v9/projects/{project_id}/env",
                    headers=self._headers,
                    json=payload,
                )
                resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise VercelError(f"set_env_var failed: {exc}") from exc

    async def create_deploy_hook(self, project_id: str, hook_name: str) -> str:
        """
        Create a deploy hook for the project.
        Returns the hook URL (used to trigger deploys without auth).
        """
        payload = {"name": hook_name, "ref": "main"}
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(
                    f"{VERCEL_API}/v9/projects/{project_id}/deploy-hooks",
                    headers=self._headers,
                    json=payload,
                )
                resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise VercelError(f"create_deploy_hook failed: {exc}") from exc

        return resp.json()["hook"]["url"]

    async def trigger_via_hook(self, hook_url: str) -> str:
        """POST to a deploy hook URL to trigger a deployment. Returns job_id."""
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.post(hook_url, json={})
                resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise VercelError(f"trigger_via_hook failed: {exc}") from exc

        return resp.json()["job"]["id"]

    async def get_latest_deployment(self, project_id: str) -> dict:
        """
        Return state and URL of the most recent deployment.
        State will be one of: QUEUED, BUILDING, READY, ERROR, CANCELED
        """
        try:
            async with httpx.AsyncClient(timeout=30) as client:
                resp = await client.get(
                    f"{VERCEL_API}/v9/projects/{project_id}/deployments",
                    headers=self._headers,
                    params={"limit": 1},
                )
                resp.raise_for_status()
        except httpx.HTTPStatusError as exc:
            raise VercelError(f"get_latest_deployment failed: {exc}") from exc

        deployments = resp.json().get("deployments", [])
        if not deployments:
            return {"state": "NONE", "url": ""}

        d = deployments[0]
        raw_url = d.get("url", "")
        return {
            "deployment_id": d.get("uid", ""),
            "state": d.get("readyState", "UNKNOWN"),
            "url": f"https://{raw_url}" if raw_url and not raw_url.startswith("http") else raw_url,
        }
