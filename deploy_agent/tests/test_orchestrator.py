import pytest
from unittest.mock import AsyncMock, MagicMock


@pytest.fixture
def mock_render():
    client = MagicMock()
    client.create_service = AsyncMock(return_value={
        "service_id": "srv-abc123", "deploy_id": "dep-xyz"
    })
    client.get_deploy = AsyncMock(return_value={"status": "live", "id": "dep-xyz"})
    client.get_service_status = AsyncMock(return_value={
        "state": "live",
        "url": "https://mtp-flight-api.onrender.com",
        "service_id": "srv-abc123"
    })
    client.get_logs = AsyncMock(return_value=["log line 1", "log line 2"])
    client.trigger_deploy = AsyncMock(return_value="dep-new123")
    return client


@pytest.fixture
def mock_vercel():
    client = MagicMock()
    client.create_project = AsyncMock(return_value="prj-abc123")
    client.set_env_var = AsyncMock()
    client.create_deploy_hook = AsyncMock(return_value="https://api.vercel.com/hook/xyz")
    client.trigger_via_hook = AsyncMock(return_value="job-123")
    client.get_latest_deployment = AsyncMock(return_value={
        "state": "READY",
        "url": "https://mtp-flight-delay.vercel.app",
        "deployment_id": "dpl-abc"
    })
    return client


@pytest.mark.asyncio
async def test_deploy_full_stack_returns_both_urls(mock_render, mock_vercel):
    """deploy_full_stack returns backend_url and frontend_url on success."""
    from deploy_agent.orchestrator import deploy_full_stack

    result = await deploy_full_stack(
        render=mock_render,
        vercel=mock_vercel,
        github_repo="21MA23002/mtp-flight-delay",
        service_name="mtp-flight-api",
    )

    assert result["status"] == "success"
    assert result["backend_url"] == "https://mtp-flight-api.onrender.com"
    assert result["frontend_url"] == "https://mtp-flight-delay.vercel.app"
    assert result["render_service_id"] == "srv-abc123"
    assert result["vercel_project_id"] == "prj-abc123"


@pytest.mark.asyncio
async def test_vercel_receives_render_url_as_env_var(mock_render, mock_vercel):
    """VITE_API_URL set on Vercel equals the Render live URL."""
    from deploy_agent.orchestrator import deploy_full_stack

    await deploy_full_stack(
        render=mock_render,
        vercel=mock_vercel,
        github_repo="21MA23002/mtp-flight-delay",
        service_name="mtp-flight-api",
    )

    mock_vercel.set_env_var.assert_awaited_once_with(
        project_id="prj-abc123",
        key="VITE_API_URL",
        value="https://mtp-flight-api.onrender.com",
    )


@pytest.mark.asyncio
async def test_deploy_full_stack_sequence_order(mock_render, mock_vercel):
    """Render deploy completes before Vercel project is created."""
    call_order = []

    mock_render.create_service = AsyncMock(
        side_effect=lambda **kw: call_order.append("render_create") or
                                  {"service_id": "srv-abc123", "deploy_id": "dep-xyz"}
    )
    mock_render.get_deploy = AsyncMock(
        side_effect=lambda *a: call_order.append("render_poll") or
                               {"status": "live", "id": "dep-xyz"}
    )
    mock_vercel.create_project = AsyncMock(
        side_effect=lambda **kw: call_order.append("vercel_create") or "prj-abc123"
    )

    from deploy_agent.orchestrator import deploy_full_stack
    await deploy_full_stack(
        render=mock_render,
        vercel=mock_vercel,
        github_repo="21MA23002/mtp-flight-delay",
        service_name="mtp-flight-api",
    )

    render_create_idx = call_order.index("render_create")
    render_poll_idx = call_order.index("render_poll")
    vercel_create_idx = call_order.index("vercel_create")

    assert render_create_idx < render_poll_idx < vercel_create_idx


@pytest.mark.asyncio
async def test_get_render_logs_delegates_to_render_client(mock_render, mock_vercel):
    """get_render_logs calls render.get_logs and returns lines."""
    from deploy_agent.orchestrator import get_render_logs

    lines = await get_render_logs(render=mock_render, service_id="srv-abc123")

    mock_render.get_logs.assert_awaited_once_with("srv-abc123")
    assert lines == ["log line 1", "log line 2"]


@pytest.mark.asyncio
async def test_get_render_status_delegates_to_render_client(mock_render):
    """get_render_status calls render.get_service_status and returns result."""
    from deploy_agent.orchestrator import get_render_status

    result = await get_render_status(render=mock_render, service_id="srv-abc123")

    mock_render.get_service_status.assert_awaited_once_with("srv-abc123")
    assert result["state"] == "live"


@pytest.mark.asyncio
async def test_get_vercel_status_delegates_to_vercel_client(mock_vercel):
    """get_vercel_status calls vercel.get_latest_deployment and returns result."""
    from deploy_agent.orchestrator import get_vercel_status

    result = await get_vercel_status(vercel=mock_vercel, project_id="prj-abc123")

    mock_vercel.get_latest_deployment.assert_awaited_once_with("prj-abc123")
    assert result["state"] == "READY"
