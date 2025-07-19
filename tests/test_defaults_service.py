"""Tests for defaults service."""

import json
import tempfile
from pathlib import Path

import pytest

from karenina_server.services.defaults_service import DefaultsService


@pytest.fixture
def temp_defaults_file():
    """Create a temporary defaults.json file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        initial_defaults = {
            "default_interface": "langchain",
            "default_provider": "openai",
            "default_model": "gpt-4",
            "saved_at": "2025-07-19T07:45:00Z",
        }
        json.dump(initial_defaults, f)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def defaults_service(temp_defaults_file):
    """Create a defaults service with temporary defaults file."""
    return DefaultsService(defaults_file_path=temp_defaults_file)


@pytest.fixture
def empty_defaults_service():
    """Create a defaults service with non-existent file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        non_existent_file = Path(temp_dir) / "nonexistent.json"
        yield DefaultsService(defaults_file_path=non_existent_file)


class TestDefaultsService:
    """Test cases for DefaultsService."""

    def test_get_defaults_from_file(self, defaults_service):
        """Test reading defaults from existing file."""
        defaults = defaults_service.get_defaults()

        assert defaults["default_interface"] == "langchain"
        assert defaults["default_provider"] == "openai"
        assert defaults["default_model"] == "gpt-4"

    def test_get_defaults_fallback(self, empty_defaults_service):
        """Test fallback defaults when file doesn't exist."""
        defaults = empty_defaults_service.get_defaults()

        assert defaults["default_interface"] == "langchain"
        assert defaults["default_provider"] == "google_genai"
        assert defaults["default_model"] == "gemini-pro"

    def test_save_defaults(self, empty_defaults_service):
        """Test saving defaults to file."""
        new_defaults = {"default_interface": "openrouter", "default_provider": "anthropic", "default_model": "claude-3"}

        empty_defaults_service.save_defaults(new_defaults)

        # Verify saved defaults
        saved_defaults = empty_defaults_service.get_defaults()
        assert saved_defaults["default_interface"] == "openrouter"
        assert saved_defaults["default_provider"] == "anthropic"
        assert saved_defaults["default_model"] == "claude-3"

    def test_validate_defaults_success(self, defaults_service):
        """Test successful validation of defaults."""
        valid_defaults = {"default_interface": "langchain", "default_provider": "openai", "default_model": "gpt-4"}

        is_valid, error = defaults_service._validate_defaults(valid_defaults)
        assert is_valid
        assert error is None

    def test_validate_defaults_invalid_interface(self, defaults_service):
        """Test validation with invalid interface."""
        invalid_defaults = {
            "default_interface": "invalid_interface",
            "default_provider": "openai",
            "default_model": "gpt-4",
        }

        is_valid, error = defaults_service._validate_defaults(invalid_defaults)
        assert not is_valid
        assert "Invalid interface" in error

    def test_validate_defaults_missing_field(self, defaults_service):
        """Test validation with missing required field."""
        invalid_defaults = {
            "default_interface": "langchain",
            "default_provider": "openai",
            # missing default_model
        }

        is_valid, error = defaults_service._validate_defaults(invalid_defaults)
        assert not is_valid
        assert "Missing required field" in error

    def test_validate_defaults_empty_provider(self, defaults_service):
        """Test validation with empty provider."""
        invalid_defaults = {"default_interface": "langchain", "default_provider": "", "default_model": "gpt-4"}

        is_valid, error = defaults_service._validate_defaults(invalid_defaults)
        assert not is_valid
        assert "Provider cannot be empty" in error

    def test_save_invalid_defaults(self, defaults_service):
        """Test that saving invalid defaults raises ValueError."""
        invalid_defaults = {"default_interface": "invalid", "default_provider": "openai", "default_model": "gpt-4"}

        with pytest.raises(ValueError, match="Invalid interface"):
            defaults_service.save_defaults(invalid_defaults)

    def test_reset_to_fallback(self, defaults_service):
        """Test resetting to fallback defaults."""
        # Verify file exists initially
        assert defaults_service.defaults_file_path.exists()

        # Reset to fallback
        defaults_service.reset_to_fallback()

        # File should be removed
        assert not defaults_service.defaults_file_path.exists()

        # Should return fallback defaults
        defaults = defaults_service.get_defaults()
        assert defaults["default_provider"] == "google_genai"  # fallback value

    def test_get_file_status_existing_file(self, defaults_service):
        """Test file status for existing file."""
        status = defaults_service.get_file_status()

        assert status["file_exists"] is True
        assert status["using_fallback"] is False
        assert status["saved_at"] == "2025-07-19T07:45:00Z"
        assert status["last_modified"] is not None

    def test_get_file_status_non_existing_file(self, empty_defaults_service):
        """Test file status for non-existing file."""
        status = empty_defaults_service.get_file_status()

        assert status["file_exists"] is False
        assert status["using_fallback"] is True
        assert status["saved_at"] is None
        assert status["last_modified"] is None

    def test_backup_and_restore(self, defaults_service):
        """Test backup creation and restoration."""
        original_defaults = defaults_service.get_defaults()

        # Modify defaults
        new_defaults = {"default_interface": "openrouter", "default_provider": "test", "default_model": "test-model"}
        defaults_service.save_defaults(new_defaults)

        # Verify modification
        modified_defaults = defaults_service.get_defaults()
        assert modified_defaults["default_provider"] == "test"

        # Restore from backup
        defaults_service._restore_backup()

        # Verify restoration
        restored_defaults = defaults_service.get_defaults()
        assert restored_defaults["default_provider"] == original_defaults["default_provider"]
