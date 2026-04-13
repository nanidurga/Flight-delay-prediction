import os
import pytest
from pathlib import Path
from unittest.mock import patch


def test_loads_from_env_file(tmp_path, monkeypatch):
    """Tokens in .env file are returned correctly."""
    env_file = tmp_path / ".env"
    env_file.write_text(
        "RENDER_API_KEY=rnd_test\n"
        "VERCEL_TOKEN=vtest\n"
        "RENDER_SERVICE_ID=srv-test\n"
        "VERCEL_PROJECT_ID=prj-test\n"
    )
    # Ensure os.environ does NOT have the keys
    monkeypatch.delenv("RENDER_API_KEY", raising=False)
    monkeypatch.delenv("VERCEL_TOKEN", raising=False)

    from deploy_agent.config import load_config
    cfg = load_config(env_path=env_file)

    assert cfg["RENDER_API_KEY"] == "rnd_test"
    assert cfg["VERCEL_TOKEN"] == "vtest"
    assert cfg["RENDER_SERVICE_ID"] == "srv-test"
    assert cfg["VERCEL_PROJECT_ID"] == "prj-test"


def test_env_vars_override_missing_dotenv(monkeypatch):
    """os.environ is used when .env file is absent."""
    monkeypatch.setenv("RENDER_API_KEY", "rnd_from_env")
    monkeypatch.setenv("VERCEL_TOKEN", "v_from_env")
    monkeypatch.setenv("RENDER_SERVICE_ID", "")
    monkeypatch.setenv("VERCEL_PROJECT_ID", "")

    from deploy_agent.config import load_config
    cfg = load_config(env_path=Path("/nonexistent/.env"))

    assert cfg["RENDER_API_KEY"] == "rnd_from_env"
    assert cfg["VERCEL_TOKEN"] == "v_from_env"


def test_dotenv_takes_priority_over_os_environ(tmp_path, monkeypatch):
    """.env file value wins over os.environ when both present."""
    env_file = tmp_path / ".env"
    env_file.write_text("RENDER_API_KEY=rnd_dotenv\nVERCEL_TOKEN=v_dotenv\n")
    monkeypatch.setenv("RENDER_API_KEY", "rnd_os_env")
    monkeypatch.setenv("VERCEL_TOKEN", "v_os_env")

    from deploy_agent.config import load_config
    cfg = load_config(env_path=env_file)

    assert cfg["RENDER_API_KEY"] == "rnd_dotenv"
    assert cfg["VERCEL_TOKEN"] == "v_dotenv"


def test_raises_config_error_when_required_key_missing(monkeypatch):
    """ConfigError raised when a required key is absent from both sources."""
    monkeypatch.delenv("RENDER_API_KEY", raising=False)
    monkeypatch.delenv("VERCEL_TOKEN", raising=False)

    from deploy_agent.config import load_config, ConfigError
    with pytest.raises(ConfigError, match="RENDER_API_KEY"):
        load_config(env_path=Path("/nonexistent/.env"))


def test_service_and_project_ids_are_optional(tmp_path, monkeypatch):
    """RENDER_SERVICE_ID and VERCEL_PROJECT_ID are optional (None if absent)."""
    env_file = tmp_path / ".env"
    env_file.write_text("RENDER_API_KEY=rnd_test\nVERCEL_TOKEN=vtest\n")
    monkeypatch.delenv("RENDER_SERVICE_ID", raising=False)
    monkeypatch.delenv("VERCEL_PROJECT_ID", raising=False)

    from deploy_agent.config import load_config
    cfg = load_config(env_path=env_file)

    assert cfg["RENDER_SERVICE_ID"] is None
    assert cfg["VERCEL_PROJECT_ID"] is None
