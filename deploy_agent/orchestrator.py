"""
orchestrator.py — Sequences Render + Vercel calls for full-stack deployment.

No HTTP calls here — all HTTP is in render.py and vercel.py.
Receives pre-built RenderClient and VercelClient instances.
"""

import asyncio


async def deploy_full_stack(
    render,
    vercel,
    github_repo: str,
    service_name: str = "",
) -> dict:
    """
    Deploy backend to Render then frontend to Vercel.

    Steps:
      1. Create (or reuse) Render service
      2. Poll until Render deploy is live (timeout 15 min)
      3. Fetch live URL from Render
      4. Create (or reuse) Vercel project
      5. Set VITE_API_URL = render live URL
      6. Create deploy hook + trigger deploy
      7. Poll until Vercel deployment is READY (timeout 10 min)

    Returns dict with status, backend_url, frontend_url, render_service_id, vercel_project_id.
    """
    if not service_name:
        service_name = github_repo.split("/")[-1]

    # Step 1: Create Render service
    render_result = await render.create_service(
        github_repo=github_repo,
        service_name=service_name,
    )
    service_id = render_result["service_id"]
    deploy_id = render_result["deploy_id"]

    # Step 2: Poll until deploy is live (max 15 min = 90 × 10s)
    backend_url = await _wait_for_render_deploy(render, service_id, deploy_id, timeout_polls=90)

    # Step 3: Create Vercel project
    project_name = github_repo.split("/")[-1]
    project_id = await vercel.create_project(
        github_repo=github_repo,
        project_name=project_name,
    )

    # Step 4: Inject Render URL as VITE_API_URL
    await vercel.set_env_var(
        project_id=project_id,
        key="VITE_API_URL",
        value=backend_url,
    )

    # Step 5: Create deploy hook and trigger
    hook_url = await vercel.create_deploy_hook(project_id, "claude-deploy")
    await vercel.trigger_via_hook(hook_url)

    # Step 6: Poll until READY (max 10 min = 60 × 10s)
    frontend_result = await _wait_for_vercel_deploy(vercel, project_id, timeout_polls=60)

    return {
        "status": "success",
        "backend_url": backend_url,
        "frontend_url": frontend_result["url"],
        "render_service_id": service_id,
        "vercel_project_id": project_id,
    }


async def _wait_for_render_deploy(
    render, service_id: str, deploy_id: str, timeout_polls: int
) -> str:
    """Poll Render deploy until live. Returns live_url. Raises TimeoutError if exceeded."""
    terminal = {"live", "failed", "canceled", "deactivated"}

    for _ in range(timeout_polls):
        deploy = await render.get_deploy(service_id, deploy_id)
        status = deploy.get("status", "")
        if status == "live":
            svc = await render.get_service_status(service_id)
            return svc["url"]
        if status in terminal:
            raise RuntimeError(f"Render deploy ended with status: {status}")
        await asyncio.sleep(10)

    raise TimeoutError(
        f"Render deploy did not go live within {timeout_polls * 10 // 60} minutes. "
        f"service_id={service_id} — call render_get_logs() for details."
    )


async def _wait_for_vercel_deploy(vercel, project_id: str, timeout_polls: int) -> dict:
    """Poll Vercel until deployment is READY. Returns deployment dict."""
    terminal = {"READY", "ERROR", "CANCELED"}

    for _ in range(timeout_polls):
        dep = await vercel.get_latest_deployment(project_id)
        if dep["state"] in terminal:
            if dep["state"] == "READY":
                return dep
            raise RuntimeError(f"Vercel deployment ended with state: {dep['state']}")
        await asyncio.sleep(10)

    raise TimeoutError(
        f"Vercel deployment did not reach READY within {timeout_polls * 10 // 60} minutes."
    )


async def get_render_logs(render, service_id: str) -> list[str]:
    """Return last 100 Render log lines."""
    return await render.get_logs(service_id)


async def get_render_status(render, service_id: str) -> dict:
    """Return Render service state and URL."""
    return await render.get_service_status(service_id)


async def get_vercel_status(vercel, project_id: str) -> dict:
    """Return latest Vercel deployment state and URL."""
    return await vercel.get_latest_deployment(project_id)
