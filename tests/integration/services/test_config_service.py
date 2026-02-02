"""Integration tests for configuration service.

Uses file I/O with temporary files.
"""

import os
import tempfile
import threading
import time
from pathlib import Path
from unittest.mock import patch

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
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def config_service(temp_env_file):
    """Create a configuration service with temporary .env file."""
    return ConfigurationService(env_file_path=temp_env_file)


@pytest.mark.integration
@pytest.mark.service
class TestConfigurationService:
    """Test cases for ConfigurationService."""

    def test_read_env_vars_masked(self, config_service):
        """Test reading environment variables with masking."""
        env_vars = config_service.read_env_vars(mask_secrets=True)

        assert "OPENAI_API_KEY" in env_vars
        assert "GOOGLE_API_KEY" in env_vars
        assert "TEST_VAR" in env_vars
        assert env_vars["OPENAI_API_KEY"].endswith("5678")
        assert env_vars["OPENAI_API_KEY"].startswith("*")

    def test_read_env_vars_unmasked(self, config_service):
        """Test reading environment variables without masking."""
        env_vars = config_service.read_env_vars(mask_secrets=False)

        assert env_vars["OPENAI_API_KEY"] == "sk-test123456789012345678901234567890123456789012345678"
        assert env_vars["TEST_VAR"] == "test_value"

    def test_update_env_var(self, config_service):
        """Test updating an environment variable."""
        config_service.update_env_var("TEST_VAR", "new_value")
        env_vars = config_service.read_env_vars(mask_secrets=False)
        assert env_vars["TEST_VAR"] == "new_value"

        config_service.update_env_var("NEW_VAR", "new_var_value")
        env_vars = config_service.read_env_vars(mask_secrets=False)
        assert env_vars["NEW_VAR"] == "new_var_value"

    def test_remove_env_var(self, config_service):
        """Test removing an environment variable."""
        config_service.remove_env_var("TEST_VAR")
        env_vars = config_service.read_env_vars(mask_secrets=False)
        assert "TEST_VAR" not in env_vars
        config_service.remove_env_var("NON_EXISTENT_VAR")

    def test_get_env_file_contents(self, config_service):
        """Test getting raw .env file contents."""
        contents = config_service.get_env_file_contents()
        assert "OPENAI_API_KEY=sk-test" in contents
        assert "TEST_VAR=test_value" in contents

    def test_update_env_file_contents(self, config_service):
        """Test updating entire .env file contents."""
        new_contents = "NEW_VAR1=value1\nNEW_VAR2=value2\n"
        config_service.update_env_file_contents(new_contents)

        contents = config_service.get_env_file_contents()
        assert contents == new_contents

        env_vars = config_service.read_env_vars(mask_secrets=False)
        assert env_vars["NEW_VAR1"] == "value1"
        assert "OPENAI_API_KEY" not in env_vars

    def test_backup_and_restore(self, config_service, temp_env_file):
        """Test backup creation and restoration."""
        config_service._create_backup()
        backup_path = temp_env_file.with_suffix(".env.backup")
        assert backup_path.exists()

        config_service.update_env_var("TEST_VAR", "modified_value")
        env_vars = config_service.read_env_vars(mask_secrets=False)
        assert env_vars["TEST_VAR"] == "modified_value"

        config_service._restore_backup()
        env_vars = config_service.read_env_vars(mask_secrets=False)
        assert env_vars["TEST_VAR"] == "test_value"

        if backup_path.exists():
            backup_path.unlink()

    def test_file_permissions(self, config_service, temp_env_file):
        """Test that file permissions are set correctly."""
        config_service.update_env_var("TEST_VAR", "new_value")

        stat_info = os.stat(temp_env_file)
        mode = stat_info.st_mode & 0o777
        assert mode == 0o600

    def test_empty_env_file(self):
        """Test handling of non-existent .env file."""
        with tempfile.TemporaryDirectory() as temp_dir:
            env_path = Path(temp_dir) / ".env"
            service = ConfigurationService(env_file_path=env_path)

            env_vars = service.read_env_vars()
            assert env_vars == {}

            contents = service.get_env_file_contents()
            assert contents == ""

            service.update_env_var("NEW_VAR", "value")
            assert env_path.exists()


@pytest.mark.integration
@pytest.mark.service
class TestConfigurationServiceSecurity:
    """Security-focused test cases for ConfigurationService."""

    def test_path_traversal_protection(self):
        """Test that path traversal attacks are prevented."""
        with pytest.raises(ValueError, match="Path outside allowed directories"):
            ConfigurationService(Path("/etc/passwd"))

    def test_file_permissions_on_creation(self, temp_env_file):
        """Test that files are created with secure permissions."""
        temp_env_file.unlink()
        service = ConfigurationService(temp_env_file)
        service.update_env_var("TEST_VAR", "test_value")

        stat = temp_env_file.stat()
        assert stat.st_mode & 0o777 == 0o600

    def test_bulk_update_functionality(self, config_service):
        """Test that bulk updates work correctly."""
        config_service.update_env_var("VAR1", "value1")
        config_service.update_env_var("VAR2", "value2")

        updates = [
            ("VAR1", "new_value1"),
            ("OPENAI_API_KEY", "any-key-format"),
            ("VAR2", "new_value2"),
        ]

        config_service.update_env_vars_bulk(updates)

        env_vars = config_service.read_env_vars(mask_secrets=False)
        assert env_vars["VAR1"] == "new_value1"
        assert env_vars["VAR2"] == "new_value2"

    def test_backup_and_restore_on_failure(self, config_service):
        """Test that backup and restore works on operation failure."""
        config_service.update_env_var("TEST_VAR", "original_value")

        with patch("karenina_server.services.config_service.set_key") as mock_set_key:
            mock_set_key.side_effect = Exception("Simulated file error")

            with pytest.raises(Exception, match="Simulated file error"):
                config_service.update_env_var("TEST_VAR", "new_value")

        env_vars = config_service.read_env_vars(mask_secrets=False)
        assert env_vars["TEST_VAR"] == "original_value"

    def test_large_input_handling(self, config_service):
        """Test handling of unusually large inputs."""
        long_value = "x" * 10000
        config_service.update_env_var("LONG_VAR", long_value)

        env_vars = config_service.read_env_vars(mask_secrets=False)
        assert env_vars["LONG_VAR"] == long_value

    def test_concurrent_access_safety(self, config_service):
        """Test basic concurrent access patterns."""
        errors = []

        def update_worker(worker_id):
            try:
                for i in range(3):
                    key = f"THREAD_{worker_id}_{i}"
                    value = f"worker_{worker_id}_value_{i}"
                    config_service.update_env_var(key, value)
                    time.sleep(0.001)
            except Exception as e:
                errors.append(e)

        threads = []
        for i in range(2):
            thread = threading.Thread(target=update_worker, args=(i,))
            threads.append(thread)
            thread.start()

        for thread in threads:
            thread.join()

        assert len(errors) == 0

        env_vars = config_service.read_env_vars(mask_secrets=False)
        thread_vars = [key for key in env_vars if key.startswith("THREAD_")]
        assert len(thread_vars) > 0
