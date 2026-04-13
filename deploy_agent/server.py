"""
server.py — FastMCP entry point for the deploy agent.

Registers 4 tools into Claude Code:
  - deploy_full_stack
  - render_get_logs
  - render_get_status
  - vercel_get_status

Run with:  python deploy_agent/server.py
Claude Code registers this via ~/.claude/settings.json mcpServers config.
"""

import asyncio
import json
import sys
from pathlib import Path

# Ensure the MTP project root is on sys.path so `deploy_agent.*` imports work
# regardless of how the script is invoked (python server.py vs python -m ...)
sys.path.insert(0, str(Path(__file__).parent.parent))

from mcp.server.fastmcp import FastMCP

from deploy_agent.config import load_config, ConfigError
from deploy_agent.render import RenderClient
from deploy_agent.vercel import VercelClient
from deploy_agent import orchestrator

# ── Initialise FastMCP ──────────────────────────────────────────────────────
mcp = FastMCP("deploy-agent")

# ── Load config + build clients at import time ──────────────────────────────
_ENV_PATH = Path(__file__).parent / ".env"

try:
    _cfg = load_config(env_path=_ENV_PATH)
    _render = RenderClient(api_key=_cfg["RENDER_API_KEY"])
    _vercel = VercelClient(token=_cfg["VERCEL_TOKEN"])
    _default_service_id = _cfg.get("RENDER_SERVICE_ID")
    _default_project_id = _cfg.get("VERCEL_PROJECT_ID")
except ConfigError as e:
    # Server still starts — tools will return the config error message
    _render = None
    _vercel = None
    _default_service_id = None
    _default_project_id = None
    _config_error = str(e)
else:
    _config_error = None


def _check_config():
    if _config_error:
        raise RuntimeError(f"Deploy agent misconfigured: {_config_error}")


# ── Tools ────────────────────────────────────────────────────────────────────

@mcp.tool()
async def deploy_full_stack(
    github_repo: str,
    service_name: str = "mtp-flight-api",
) -> str:
    """
    Deploy the MTP Flight Delay app end-to-end.

    Steps:
      1. Create / reuse Render web service for the FastAPI backend
      2. Wait for backend deploy to go live (up to 15 min)
      3. Create / reuse Vercel project for the React frontend
      4. Inject backend URL as VITE_API_URL environment variable
      5. Trigger Vercel frontend deployment
      6. Wait for frontend to be READY (up to 10 min)

    Args:
        github_repo: GitHub repo in "owner/repo" format, e.g. "21MA23002/mtp-flight-delay"
        service_name: Name for the Render service (default: "mtp-flight-api")

    Returns JSON with backend_url, frontend_url, render_service_id, vercel_project_id.
    """
    _check_config()
    result = await orchestrator.deploy_full_stack(
        render=_render,
        vercel=_vercel,
        github_repo=github_repo,
        service_name=service_name,
    )
    return json.dumps(result, indent=2)


@mcp.tool()
async def render_get_logs(service_id: str = "") -> str:
    """
    Fetch the last 100 lines of runtime logs from your Render backend service.

    Args:
        service_id: Render service ID (e.g. "srv-xxxx").
                    Defaults to RENDER_SERVICE_ID from config if not provided.

    Returns log lines as a newline-separated string.
    """
    _check_config()
    sid = service_id or _default_service_id
    if not sid:
        return "Error: no service_id provided and RENDER_SERVICE_ID not set in config."
    lines = await orchestrator.get_render_logs(render=_render, service_id=sid)
    return "\n".join(lines)


@mcp.tool()
async def render_get_status(service_id: str = "") -> str:
    """
    Check the current status of your Render backend service.

    Args:
        service_id: Render service ID (e.g. "srv-xxxx").
                    Defaults to RENDER_SERVICE_ID from config if not provided.

    Returns JSON with state and url fields.
    """
    _check_config()
    sid = service_id or _default_service_id
    if not sid:
        return "Error: no service_id provided and RENDER_SERVICE_ID not set in config."
    result = await orchestrator.get_render_status(render=_render, service_id=sid)
    return json.dumps(result, indent=2)


@mcp.tool()
async def vercel_get_status(project_id: str = "") -> str:
    """
    Check the current status of your Vercel frontend deployment.

    Args:
        project_id: Vercel project ID (e.g. "prj-xxxx").
                    Defaults to VERCEL_PROJECT_ID from config if not provided.

    Returns JSON with state and url fields.
    """
    _check_config()
    pid = project_id or _default_project_id
    if not pid:
        return "Error: no project_id provided and VERCEL_PROJECT_ID not set in config."
    result = await orchestrator.get_vercel_status(vercel=_vercel, project_id=pid)
    return json.dumps(result, indent=2)


# ── Entry point ──────────────────────────────────────────────────────────────

if __name__ == "__main__":
    mcp.run()
