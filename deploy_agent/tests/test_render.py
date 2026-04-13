import pytest
import httpx
from unittest.mock import AsyncMock, patch, MagicMock


@pytest.fixture
def render_client():
    from deploy_agent.render import RenderClient
    return RenderClient(api_key="rnd_test")


@pytest.mark.asyncio
async def test_create_service_returns_service_and_deploy_ids(render_client):
    """create_service POSTs to /v1/services and returns service_id + deploy_id."""
    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.json.return_value = {
        "service": {"id": "srv-abc123", "name": "mtp-flight-api"},
        "deployId": "dep-xyz789"
    }
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
        result = await render_client.create_service(
            github_repo="21MA23002/mtp-flight-delay",
            service_name="mtp-flight-api"
        )

    assert result["service_id"] == "srv-abc123"
    assert result["deploy_id"] == "dep-xyz789"


@pytest.mark.asyncio
async def test_get_deploy_returns_state(render_client):
    """get_deploy GETs /v1/deploys/{id} and returns state."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "deploy": {"id": "dep-xyz789", "status": "live"}
    }
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response):
        result = await render_client.get_deploy("dep-xyz789")

    assert result["status"] == "live"


@pytest.mark.asyncio
async def test_get_service_status_returns_live_url(render_client):
    """get_service_status returns service state and live URL."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = {
        "service": {
            "id": "srv-abc123",
            "serviceDetails": {"url": "https://mtp-flight-api.onrender.com"},
            "suspended": "not_suspended"
        }
    }
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response):
        result = await render_client.get_service_status("srv-abc123")

    assert result["url"] == "https://mtp-flight-api.onrender.com"
    assert result["state"] == "live"


@pytest.mark.asyncio
async def test_get_logs_returns_lines(render_client):
    """get_logs returns list of log line strings."""
    mock_response = MagicMock()
    mock_response.status_code = 200
    mock_response.json.return_value = [
        {"message": "Starting server..."},
        {"message": "Uvicorn running on 0.0.0.0:10000"},
    ]
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.get", new_callable=AsyncMock, return_value=mock_response):
        lines = await render_client.get_logs("srv-abc123")

    assert lines == ["Starting server...", "Uvicorn running on 0.0.0.0:10000"]


@pytest.mark.asyncio
async def test_trigger_deploy_returns_deploy_id(render_client):
    """trigger_deploy POSTs to /v1/services/{id}/deploys and returns deploy_id."""
    mock_response = MagicMock()
    mock_response.status_code = 201
    mock_response.json.return_value = {"deploy": {"id": "dep-new123"}}
    mock_response.raise_for_status = MagicMock()

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
        deploy_id = await render_client.trigger_deploy("srv-abc123")

    assert deploy_id == "dep-new123"


@pytest.mark.asyncio
async def test_render_error_raised_on_http_error(render_client):
    """RenderError is raised when Render API returns 4xx/5xx."""
    mock_response = MagicMock()
    mock_response.raise_for_status.side_effect = httpx.HTTPStatusError(
        "401 Unauthorized", request=MagicMock(), response=MagicMock()
    )

    with patch("httpx.AsyncClient.post", new_callable=AsyncMock, return_value=mock_response):
        from deploy_agent.render import RenderError
        with pytest.raises(RenderError):
            await render_client.create_service("user/repo", "name")
