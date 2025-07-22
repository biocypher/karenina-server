"""Tests for configuration API handlers."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from karenina_server.api.config_handlers import router
from karenina_server.services.config_service import ConfigurationService
from karenina_server.services.defaults_service import DefaultsService


@pytest.fixture
def temp_env_file():
    """Create a temporary .env file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
        f.write("TEST_VAR=test_value\n")
        f.write("OPENAI_API_KEY=sk-test123456789012345678901234567890123456789012345678\n")
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def temp_defaults_file():
    """Create a temporary defaults.json file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        defaults = {
            "default_interface": "langchain",
            "default_provider": "google_genai",
            "default_model": "gemini-pro",
            "saved_at": "2023-01-01T00:00:00Z",
        }
        json.dump(defaults, f, indent=2)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def test_client(temp_env_file, temp_defaults_file):
    """Create test client with mocked services."""
    from fastapi import FastAPI

    app = FastAPI()

    # Create real service instances with test files
    real_config_service = ConfigurationService(temp_env_file)
    real_defaults_service = DefaultsService(temp_defaults_file)

    # Mock the global services with our real instances
    with (
        patch("karenina_server.api.config_handlers.config_service", real_config_service),
        patch("karenina_server.api.config_handlers.defaults_service", real_defaults_service),
    ):
        app.include_router(router, prefix="/api/config")
        yield TestClient(app)


class TestConfigHandlers:
    """Test cases for configuration API handlers."""

    def test_get_env_vars_masked(self, test_client):
        """Test getting masked environment variables."""
        response = test_client.get("/api/config/env-vars")

        assert response.status_code == 200
        data = response.json()
        assert "TEST_VAR" in data
        assert "OPENAI_API_KEY" in data
        # API key should be masked
        assert data["OPENAI_API_KEY"].endswith("5678")
        assert data["OPENAI_API_KEY"].startswith("*")

    def test_unmasked_endpoint_removed(self, test_client):
        """Test that unmasked endpoint has been removed."""
        response = test_client.get("/api/config/env-vars/unmasked")
        # Should return 404 or 405 since the endpoint was completely removed
        # 404 = Not Found, 405 = Method Not Allowed (both indicate unavailable)
        assert response.status_code in [404, 405]

    def test_get_env_file_contents(self, test_client):
        """Test getting .env file contents."""
        response = test_client.get("/api/config/env-file")

        assert response.status_code == 200
        data = response.json()
        assert "content" in data
        assert "TEST_VAR=test_value" in data["content"]

    def test_update_env_var_valid(self, test_client):
        """Test updating a valid environment variable."""
        payload = {"key": "NEW_VAR", "value": "new_value"}

        response = test_client.put("/api/config/env-vars", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert "Successfully updated NEW_VAR" in data["message"]

    def test_update_env_var_invalid_key(self, test_client):
        """Test updating with invalid key format."""
        payload = {
            "key": "invalid-key",  # Should be uppercase with underscores
            "value": "test_value",
        }

        response = test_client.put("/api/config/env-vars", json=payload)

        assert response.status_code == 422  # Pydantic validation error

    def test_update_env_var_too_long_key(self, test_client):
        """Test updating with overly long key."""
        payload = {
            "key": "A" * 101,  # Exceeds 100 character limit
            "value": "test_value",
        }

        response = test_client.put("/api/config/env-vars", json=payload)

        assert response.status_code == 422  # Pydantic validation error

    def test_update_env_var_too_long_value(self, test_client):
        """Test updating with overly long value."""
        payload = {
            "key": "TEST_VAR",
            "value": "x" * 1001,  # Exceeds 1000 character limit
        }

        response = test_client.put("/api/config/env-vars", json=payload)

        assert response.status_code == 422  # Pydantic validation error

    def test_bulk_update_valid(self, test_client):
        """Test bulk update with valid variables."""
        payload = {"variables": [{"key": "VAR_ONE", "value": "value1"}, {"key": "VAR_TWO", "value": "value2"}]}

        response = test_client.put("/api/config/env-vars/bulk", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert "All variables updated successfully" in data["message"]
        assert "VAR_ONE" in data["updated"]
        assert "VAR_TWO" in data["updated"]

    def test_bulk_update_too_many_variables(self, test_client):
        """Test bulk update with too many variables."""
        payload = {"variables": [{"key": f"VAR_{i}", "value": f"value{i}"} for i in range(51)]}

        response = test_client.put("/api/config/env-vars/bulk", json=payload)

        assert response.status_code == 422  # Pydantic validation error

    def test_bulk_update_validation_failure(self, test_client):
        """Test bulk update with validation failure (should be atomic)."""
        payload = {
            "variables": [
                {"key": "VALID_VAR", "value": "valid_value"},
                {"key": "invalid-key", "value": "value2"},  # Invalid key format
            ]
        }

        response = test_client.put("/api/config/env-vars/bulk", json=payload)

        assert response.status_code == 422  # Should fail validation before any updates

    def test_update_env_file_valid(self, test_client):
        """Test updating entire .env file."""
        payload = {"content": "NEW_VAR=new_value\nANOTHER_VAR=another_value\n"}

        response = test_client.put("/api/config/env-file", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert "Successfully updated .env file" in data["message"]

    def test_update_env_file_too_large(self, test_client):
        """Test updating with overly large file content."""
        payload = {
            "content": "x" * 50001  # Exceeds 50KB limit
        }

        response = test_client.put("/api/config/env-file", json=payload)

        assert response.status_code == 422  # Pydantic validation error

    def test_delete_env_var(self, test_client):
        """Test deleting an environment variable."""
        response = test_client.delete("/api/config/env-vars/TEST_VAR")

        assert response.status_code == 200
        data = response.json()
        assert "Successfully removed TEST_VAR" in data["message"]

    def test_get_config_status(self, test_client):
        """Test getting configuration status."""
        response = test_client.get("/api/config/status")

        assert response.status_code == 200
        data = response.json()
        assert "variables" in data
        assert isinstance(data["variables"], dict)

    def test_get_defaults(self, test_client):
        """Test getting default configuration."""
        response = test_client.get("/api/config/defaults")

        assert response.status_code == 200
        data = response.json()
        assert "default_interface" in data
        assert "default_provider" in data
        assert "default_model" in data

    def test_update_defaults_valid(self, test_client):
        """Test updating default configuration."""
        payload = {"default_interface": "openrouter", "default_provider": "openai", "default_model": "gpt-4"}

        response = test_client.put("/api/config/defaults", json=payload)

        assert response.status_code == 200
        data = response.json()
        assert "Default configuration saved successfully" in data["message"]

    def test_update_defaults_invalid_interface(self, test_client):
        """Test updating with invalid interface."""
        payload = {"default_interface": "invalid_interface", "default_provider": "openai", "default_model": "gpt-4"}

        response = test_client.put("/api/config/defaults", json=payload)

        assert response.status_code == 400  # Should fail validation

    def test_get_defaults_file_status(self, test_client):
        """Test getting defaults file status."""
        response = test_client.get("/api/config/defaults/status")

        assert response.status_code == 200
        data = response.json()
        assert "file_exists" in data
        assert "file_path" in data

    def test_reset_defaults(self, test_client):
        """Test resetting defaults to fallback values."""
        response = test_client.post("/api/config/defaults/reset")

        assert response.status_code == 200
        data = response.json()
        assert "Defaults reset to fallback values" in data["message"]


class TestSecurityFeatures:
    """Test security-related features."""

    def test_error_message_sanitization(self, test_client):
        """Test that error messages are sanitized."""
        # This test might need to be adapted based on how we can trigger specific errors
        # For now, test that invalid path doesn't expose system details
        pass

    def test_input_validation_prevents_injection(self, test_client):
        """Test that input validation prevents common injection attempts."""
        # Test special characters in keys
        payload = {
            "key": "TEST_VAR; rm -rf /",  # Shell injection attempt
            "value": "test_value",
        }

        response = test_client.put("/api/config/env-vars", json=payload)
        assert response.status_code == 422  # Should be blocked by regex validation

    def test_api_key_storage(self, test_client):
        """Test API key storage with any format (validation removed)."""
        # Test traditional OpenAI key format
        payload = {"key": "OPENAI_API_KEY", "value": "sk-proj-abcdefghijklmnopqrstuvwxyz1234567890"}

        response = test_client.put("/api/config/env-vars", json=payload)
        assert response.status_code == 200

        # Test non-standard key format - should now succeed
        payload = {"key": "OPENAI_API_KEY", "value": "invalid-key-format"}

        response = test_client.put("/api/config/env-vars", json=payload)
        assert response.status_code == 200  # Should succeed since validation is removed
