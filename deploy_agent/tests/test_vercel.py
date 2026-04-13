import pytest
import httpx
from unittest.mock import AsyncMock, MagicMock, patch


@pytest.fixture
def vercel_client():
    from deploy_agent.vercel import VercelClient
    return VercelClient(token="vtest")


@pytest.mark.asyncio
async def test_create_project_returns_project_id(vercel_client):
    """create_project POSTs to /v9/projects and returns project_id."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "id": "prj-abc123",
        "name": "mtp-flight-delay",
    }
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
        result = await vercel_client.create_project(
            github_repo="21MA23002/mtp-flight-delay",
            project_name="mtp-flight-delay"
        )

    assert result == "prj-abc123"


@pytest.mark.asyncio
async def test_set_env_var_calls_correct_endpoint(vercel_client):
    """set_env_var POSTs to /v9/projects/{id}/env."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
        await vercel_client.set_env_var(
            project_id="prj-abc123",
            key="VITE_API_URL",
            value="https://mtp-flight-api.onrender.com"
        )

    call_url = mock_post.call_args[0][0]
    assert "prj-abc123/env" in call_url


@pytest.mark.asyncio
async def test_create_deploy_hook_returns_hook_url(vercel_client):
    """create_deploy_hook returns the hook URL string."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    # Vercel returns the updated project object; hooks live at link.deployHooks
    mock_response.json.return_value = {
        "link": {
            "deployHooks": [
                {
                    "id": "hook-xyz",
                    "name": "claude-deploy",
                    "ref": "master",
                    "url": "https://api.vercel.com/v1/integrations/deploy/abc/xyz",
                    "createdAt": 1000,
                }
            ]
        }
    }
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
        url = await vercel_client.create_deploy_hook("prj-abc123", "claude-deploy")

    assert url == "https://api.vercel.com/v1/integrations/deploy/abc/xyz"


@pytest.mark.asyncio
async def test_trigger_via_hook_posts_to_hook_url(vercel_client):
    """trigger_via_hook POSTs to the hook URL and returns job_id."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {"job": {"id": "job-123"}}
    mock_response.raise_for_status = MagicMock()

    hook_url = "https://api.vercel.com/v1/integrations/deploy/abc/xyz"
    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response) as mock_post:
        job_id = await vercel_client.trigger_via_hook(hook_url)

    assert mock_post.call_args[0][0] == hook_url
    assert job_id == "job-123"


@pytest.mark.asyncio
async def test_get_latest_deployment_returns_state_and_url(vercel_client):
    """get_latest_deployment returns state and URL of most recent deployment."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "deployments": [
            {
                "uid": "dpl-abc",
                "readyState": "READY",
                "url": "mtp-flight-delay-xyz.vercel.app"
            }
        ]
    }
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response):
        result = await vercel_client.get_latest_deployment("prj-abc123")

    assert result["state"] == "READY"
    assert result["url"] == "https://mtp-flight-delay-xyz.vercel.app"


@pytest.mark.asyncio
async def test_vercel_error_raised_on_http_error(vercel_client):
    """VercelError is raised when Vercel API returns 4xx/5xx."""
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "403 Forbidden", request=MagicMock(), response=MagicMock()
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
        from deploy_agent.vercel import VercelError
        with pytest.raises(VercelError):
            await vercel_client.create_project("user/repo", "name")
