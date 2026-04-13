"""
config.py — Token loading for the deploy agent.

Priority:
  1. deploy_agent/.env  (loaded via python-dotenv)
  2. os.environ         (populated by Claude Code mcpServers.env config)
  3. ConfigError        if required key still missing
"""

import os
from pathlib import Path
from dotenv import dotenv_values


class ConfigError(Exception):
    """Raised when a required API token cannot be found."""


_REQUIRED = ["RENDER_API_KEY", "VERCEL_TOKEN"]
_OPTIONAL = ["RENDER_SERVICE_ID", "VERCEL_PROJECT_ID"]

# Default .env path: same directory as this file
_DEFAULT_ENV = Path(__file__).parent / ".env"


def load_config(env_path: Path = _DEFAULT_ENV) -> dict:
    """
    Load deployment tokens.

    Returns a dict with keys:
      RENDER_API_KEY, VERCEL_TOKEN,
      RENDER_SERVICE_ID (or None), VERCEL_PROJECT_ID (or None)

    Raises ConfigError if a required key is missing from both sources.
    """
    # Step 1: read .env file (returns empty dict if file absent)
    dotenv = dotenv_values(env_path) if env_path.exists() else {}

    cfg = {}

    for key in _REQUIRED:
        # .env takes priority; fall back to os.environ
        value = dotenv.get(key) or os.environ.get(key)
        if not value:
            raise ConfigError(
                f"Required token '{key}' not found.\n"
                f"  Option 1: add it to {env_path}\n"
                f"  Option 2: add it to mcpServers.env in ~/.claude/settings.json"
            )
        cfg[key] = value

    for key in _OPTIONAL:
        value = dotenv.get(key) or os.environ.get(key) or None
        cfg[key] = value if value else None

    return cfg
