"""Tests for configuration service."""

import os
import tempfile
from pathlib import Path

import pytest

from karenina_server.services.config_service import ConfigurationService


@pytest.fixture
def temp_env_file():
    """Create a temporary .env file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
        f.write("OPENAI_API_KEY=sk-test123456789012345678901234567890123456789012345678\n")
        f.write("GOOGLE_API_KEY=AIzaSyA-test_key_with_39_characters_12\n")
        f.write("TEST_VAR=test_value\n")
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def config_service(temp_env_file):
    """Create a configuration service with temporary .env file."""
    return ConfigurationService(env_file_path=temp_env_file)


class TestConfigurationService:
    """Test cases for ConfigurationService."""

    def test_read_env_vars_masked(self, config_service):
        """Test reading environment variables with masking."""
        env_vars = config_service.read_env_vars(mask_secrets=True)

        assert "OPENAI_API_KEY" in env_vars
        assert "GOOGLE_API_KEY" in env_vars
        assert "TEST_VAR" in env_vars

        # Check masking
        assert env_vars["OPENAI_API_KEY"].endswith("5678")
        assert env_vars["OPENAI_API_KEY"].startswith("*" * (48 - 4))
        assert env_vars["GOOGLE_API_KEY"].endswith("s_12")
        assert env_vars["TEST_VAR"] == "test_value"  # Non-API key not masked

    def test_read_env_vars_unmasked(self, config_service):
        """Test reading environment variables without masking."""
        env_vars = config_service.read_env_vars(mask_secrets=False)

        assert env_vars["OPENAI_API_KEY"] == "sk-test123456789012345678901234567890123456789012345678"
        assert env_vars["GOOGLE_API_KEY"] == "AIzaSyA-test_key_with_39_characters_12"
        assert env_vars["TEST_VAR"] == "test_value"

    def test_update_env_var(self, config_service):
        """Test updating an environment variable."""
        # Update existing variable
        config_service.update_env_var("TEST_VAR", "new_value")
        env_vars = config_service.read_env_vars(mask_secrets=False)
        assert env_vars["TEST_VAR"] == "new_value"

        # Add new variable
        config_service.update_env_var("NEW_VAR", "new_var_value")
        env_vars = config_service.read_env_vars(mask_secrets=False)
        assert env_vars["NEW_VAR"] == "new_var_value"

    def test_validate_api_key_format(self, config_service):
        """Test API key format validation."""
        # Valid OpenAI key
        is_valid, error = config_service._validate_api_key(
            "OPENAI_API_KEY", "sk-proj123456789012345678901234567890123456789012345"
        )
        assert is_valid
        assert error is None

        # Invalid OpenAI key (wrong prefix)
        is_valid, error = config_service._validate_api_key("OPENAI_API_KEY", "invalid-key")
        assert not is_valid
        assert "Invalid format" in error

        # Valid Google key
        is_valid, error = config_service._validate_api_key("GOOGLE_API_KEY", "AIzaSyA-test_key_with_39_characters_123")
        assert is_valid
        assert error is None

        # Unknown key type (allow any format)
        is_valid, error = config_service._validate_api_key("UNKNOWN_KEY", "any-value")
        assert is_valid
        assert error is None

    def test_remove_env_var(self, config_service):
        """Test removing an environment variable."""
        # Remove existing variable
        config_service.remove_env_var("TEST_VAR")
        env_vars = config_service.read_env_vars(mask_secrets=False)
        assert "TEST_VAR" not in env_vars

        # Removing non-existent variable should not raise error
        config_service.remove_env_var("NON_EXISTENT_VAR")

    def test_get_env_file_contents(self, config_service):
        """Test getting raw .env file contents."""
        contents = config_service.get_env_file_contents()
        assert "OPENAI_API_KEY=sk-test" in contents
        assert "GOOGLE_API_KEY=AIzaSyA-test" in contents
        assert "TEST_VAR=test_value" in contents

    def test_update_env_file_contents(self, config_service):
        """Test updating entire .env file contents."""
        new_contents = "NEW_VAR1=value1\nNEW_VAR2=value2\n"
        config_service.update_env_file_contents(new_contents)

        # Check file was updated
        contents = config_service.get_env_file_contents()
        assert contents == new_contents

        # Check variables are loaded
        env_vars = config_service.read_env_vars(mask_secrets=False)
        assert env_vars["NEW_VAR1"] == "value1"
        assert env_vars["NEW_VAR2"] == "value2"
        assert "OPENAI_API_KEY" not in env_vars  # Old vars removed

    def test_validate_provider_config(self, config_service):
        """Test provider configuration validation."""
        # OpenAI configured
        is_valid, error = config_service.validate_provider_config("openai")
        assert is_valid
        assert error is None

        # Google configured
        is_valid, error = config_service.validate_provider_config("google_genai")
        assert is_valid
        assert error is None

        # Anthropic not configured
        is_valid, error = config_service.validate_provider_config("anthropic")
        assert not is_valid
        assert "Missing ANTHROPIC_API_KEY" in error

        # Unknown provider
        is_valid, error = config_service.validate_provider_config("unknown")
        assert not is_valid
        assert "Unknown provider" in error

    def test_backup_and_restore(self, config_service, temp_env_file):
        """Test backup creation and restoration."""
        # Create backup
        config_service._create_backup()
        backup_path = temp_env_file.with_suffix(".env.backup")
        assert backup_path.exists()

        # Modify file
        config_service.update_env_var("TEST_VAR", "modified_value")
        env_vars = config_service.read_env_vars(mask_secrets=False)
        assert env_vars["TEST_VAR"] == "modified_value"

        # Restore from backup
        config_service._restore_backup()
        env_vars = config_service.read_env_vars(mask_secrets=False)
        assert env_vars["TEST_VAR"] == "test_value"  # Original value restored

        # Cleanup backup
        if backup_path.exists():
            backup_path.unlink()

    def test_file_permissions(self, config_service, temp_env_file):
        """Test that file permissions are set correctly."""
        # Update file to ensure permissions are set
        config_service.update_env_var("TEST_VAR", "new_value")

        # Check file permissions (should be 600)
        stat_info = os.stat(temp_env_file)
        mode = stat_info.st_mode & 0o777
        assert mode == 0o600

    def test_empty_env_file(self):
        """Test handling of non-existent .env file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            service = ConfigurationService(env_file_path=env_path)

            # Reading non-existent file should return empty dict
            env_vars = service.read_env_vars()
            assert env_vars == {}

            # Getting contents should return empty string
            contents = service.get_env_file_contents()
            assert contents == ""

            # Adding a variable should create the file
            service.update_env_var("NEW_VAR", "value")
            assert env_path.exists()
            env_vars = service.read_env_vars(mask_secrets=False)
            assert env_vars["NEW_VAR"] == "value"
