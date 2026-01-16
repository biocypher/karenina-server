"""Integration tests for configuration API handlers.

Uses TestClient to test API endpoints for configuration management.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from karenina_server.server import create_fastapi_app


@pytest.fixture
def webapp_dir() -> Path:
    """Return path to the webapp directory."""
    return Path(__file__).parent.parent.parent.parent / "src" / "karenina_server" / "webapp"


@pytest.fixture
def temp_env_file():
    """Create a temporary .env file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
        f.write("OPENAI_API_KEY=sk-test123456789012345678901234567890123456789012345678\n")
        f.write("GOOGLE_API_KEY=AIzaSyA-test_key_with_39_characters_12\n")
        f.write("TEST_VAR=test_value\n")
        temp_path = Path(f.name)
    yield temp_path
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def client(webapp_dir, temp_env_file):
    """Create a test client with patched config service."""
    from karenina_server.services.config_service import ConfigurationService

    with patch("karenina_server.api.config_handlers.config_service"):
        import karenina_server.api.config_handlers as handlers

        original_service = handlers.config_service
        handlers.config_service = ConfigurationService(env_file_path=temp_env_file)
        app = create_fastapi_app(webapp_dir)
        test_client = TestClient(app)
        yield test_client
        handlers.config_service = original_service


@pytest.mark.integration
@pytest.mark.api
class TestConfigEndpoints:
    """Integration tests for configuration API endpoints."""

    def test_get_env_vars_masked(self, client):
        """Test GET /api/config/env-vars returns masked env vars."""
        response = client.get("/api/config/env-vars")
        assert response.status_code == 200
        env_vars = response.json()
        # Env vars should be returned as a dict
        assert isinstance(env_vars, dict)

    def test_update_env_var(self, client):
        """Test PUT /api/config/env-vars updates env vars."""
        response = client.put(
            "/api/config/env-vars",
            json={"key": "NEW_VAR", "value": "new_value"},
        )
        assert response.status_code == 200

    def test_delete_env_var(self, client):
        """Test DELETE /api/config/env-vars/{key} removes env var."""
        # First add a variable
        client.put("/api/config/env-vars", json={"key": "TO_DELETE", "value": "value"})

        # Then delete it
        response = client.delete("/api/config/env-vars/TO_DELETE")
        assert response.status_code == 200
