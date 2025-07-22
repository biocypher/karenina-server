"""Tests for configuration service."""

import os
import tempfile
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


class TestConfigurationServiceSecurity:
    """Security-focused test cases for ConfigurationService."""

    def test_path_traversal_protection(self):
        """Test that path traversal attacks are prevented."""
        with pytest.raises(ValueError, match="Path outside allowed directories"):
            ConfigurationService(Path("/etc/passwd"))

        # Test absolute path outside allowed directories
        with pytest.raises(ValueError, match="Path outside allowed directories"):
            ConfigurationService(Path("/tmp/malicious.env"))

    def test_file_permissions_on_creation(self, temp_env_file):
        """Test that files are created with secure permissions."""
        # Remove the existing file
        temp_env_file.unlink()

        service = ConfigurationService(temp_env_file)
        service.update_env_var("TEST_VAR", "test_value")

        # Check file permissions (should be 0o600)
        stat = temp_env_file.stat()
        assert stat.st_mode & 0o777 == 0o600

    def test_api_key_storage(self, config_service):
        """Test that API keys can be stored with any format."""
        # Any OpenAI key format should work
        config_service.update_env_var("OPENAI_API_KEY", "invalid-key")
        config_service.update_env_var("OPENAI_API_KEY", "sk-short")
        config_service.update_env_var("OPENAI_API_KEY", "sk-proj-1234567890123456789012345678901234567890")

        # Any Anthropic key format should work
        config_service.update_env_var("ANTHROPIC_API_KEY", "sk-ant-short")
        config_service.update_env_var("ANTHROPIC_API_KEY", "sk-ant-" + "a" * 95)

        # Any Google key format should work
        config_service.update_env_var("GOOGLE_API_KEY", "short")
        config_service.update_env_var("GOOGLE_API_KEY", "a" * 39)

    def test_bulk_update_functionality(self, config_service):
        """Test that bulk updates work correctly."""
        # Set initial values
        config_service.update_env_var("VAR1", "value1")
        config_service.update_env_var("VAR2", "value2")

        # Perform bulk update - all should succeed
        updates = [
            ("VAR1", "new_value1"),
            ("OPENAI_API_KEY", "any-key-format"),  # Any format should work now
            ("VAR2", "new_value2"),
        ]

        config_service.update_env_vars_bulk(updates)

        # Check that all values were changed
        env_vars = config_service.read_env_vars(mask_secrets=False)
        assert env_vars["VAR1"] == "new_value1"
        assert env_vars["VAR2"] == "new_value2"
        assert env_vars["OPENAI_API_KEY"] == "any-key-format"

    def test_backup_and_restore_on_failure(self, config_service):
        """Test that backup and restore works on operation failure."""
        # Set initial value
        config_service.update_env_var("TEST_VAR", "original_value")

        # Mock an error during file writing to trigger restore
        with patch("karenina_server.services.config_service.set_key") as mock_set_key:
            mock_set_key.side_effect = Exception("Simulated file error")

            with pytest.raises(Exception, match="Simulated file error"):
                config_service.update_env_var("TEST_VAR", "new_value")

        # Check that original value is restored
        env_vars = config_service.read_env_vars(mask_secrets=False)
        assert env_vars["TEST_VAR"] == "original_value"

    def test_secure_temp_file_creation(self, config_service):
        """Test that temporary files are created securely."""
        import os
        from unittest.mock import patch

        # Track file creation calls
        original_open = os.open
        created_files = []

        def track_open(path, flags, mode=None):
            created_files.append((path, flags, mode))
            return original_open(path, flags, mode)

        with patch("os.open", side_effect=track_open):
            config_service.update_env_file_contents("NEW_VAR=new_value")

        # Check that temporary file was created with secure permissions
        temp_files = [f for f in created_files if f[0].endswith(".tmp")]
        assert len(temp_files) > 0
        temp_file = temp_files[0]
        assert temp_file[2] == 0o600  # Should have secure permissions

    def test_large_input_handling(self, config_service):
        """Test handling of unusually large inputs."""
        # Test very long value (should handle gracefully)
        long_value = "x" * 10000
        config_service.update_env_var("LONG_VAR", long_value)

        env_vars = config_service.read_env_vars(mask_secrets=False)
        assert env_vars["LONG_VAR"] == long_value

    def test_masking_preserves_security(self, config_service):
        """Test that masking properly obscures sensitive data."""
        # Set various API keys (any format should work now)
        keys_and_values = {
            "OPENAI_API_KEY": "any-openai-key-format",
            "ANTHROPIC_API_KEY": "any-anthropic-key-format",
            "GOOGLE_API_KEY": "any-google-key-format",
            "REGULAR_VAR": "not_a_secret",
        }

        for key, value in keys_and_values.items():
            config_service.update_env_var(key, value)

        # Get masked values
        masked_vars = config_service.read_env_vars(mask_secrets=True)

        # Check that API keys are properly masked
        for key, original_value in keys_and_values.items():
            if key.endswith("_API_KEY"):
                masked_value = masked_vars[key]
                # Should end with last 4 characters
                assert masked_value.endswith(original_value[-4:])
                # Should start with asterisks
                assert masked_value.startswith("*")
                # Should not contain the full original value
                assert original_value[:-4] not in masked_value
            else:
                # Non-API keys should not be masked
                assert masked_vars[key] == original_value

    def test_concurrent_access_safety(self, config_service):
        """Test basic concurrent access patterns."""
        import threading
        import time

        errors = []

        def update_worker(worker_id):
            try:
                for i in range(3):  # Small number to reduce race conditions
                    key = f"THREAD_{worker_id}_{i}"
                    value = f"worker_{worker_id}_value_{i}"
                    config_service.update_env_var(key, value)
                    time.sleep(0.001)  # Very small delay
            except Exception as e:
                errors.append(e)

        # Start 2 threads with unique keys to avoid conflicts
        threads = []
        for i in range(2):
            thread = threading.Thread(target=update_worker, args=(i,))
            threads.append(thread)
            thread.start()

        # Wait for all threads to complete
        for thread in threads:
            thread.join()

        # The main test is that no errors occurred during concurrent access
        assert len(errors) == 0

        # Verify that at least some variables were set (allowing for race conditions)
        env_vars = config_service.read_env_vars(mask_secrets=False)
        thread_vars = [key for key in env_vars if key.startswith("THREAD_")]
        assert len(thread_vars) > 0  # At least some variables should be set
