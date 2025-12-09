"""Service for managing persistent default LLM configuration settings."""
# ruff: noqa: B904  # Intentionally suppress exception chaining for security

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class DefaultsService:
    """Service for managing default LLM configuration persistence."""

    def __init__(self, defaults_file_path: Path | None = None):
        """Initialize defaults service.

        Args:
            defaults_file_path: Path to defaults.json file. If None, uses default location.
        """
        if defaults_file_path is None:
            # Default to defaults.json file in project root (same directory as start-karenina.sh)
            project_root = Path(__file__).parent.parent.parent.parent.parent
            defaults_file_path = project_root / "defaults.json"

        # Validate and canonicalize path to prevent traversal attacks
        self.defaults_file_path = self._validate_file_path(defaults_file_path)

        # Hardcoded fallback defaults
        self.fallback_defaults: dict[str, str | None] = {
            "default_interface": "langchain",
            "default_provider": "anthropic",
            "default_model": "claude-haiku-4-5",
            "default_endpoint_base_url": None,
        }

    def _validate_file_path(self, file_path: Path) -> Path:
        """Validate file path to prevent directory traversal attacks.

        Args:
            file_path: Path to validate

        Returns:
            Canonicalized safe path

        Raises:
            ValueError: If path is unsafe
        """
        try:
            # Resolve to absolute path and normalize
            resolved_path = file_path.resolve()

            # Get project root for comparison
            project_root = Path(__file__).parent.parent.parent.parent.parent.resolve()

            # Ensure the path is within project directory or user home
            user_home = Path.home().resolve()

            # Get system temp directory (for tests)
            import tempfile

            system_temp = Path(tempfile.gettempdir()).resolve()

            # Forbidden paths (but allow system temp for testing)
            forbidden_paths = [
                Path("/etc"),
                Path("/usr"),
                Path("/bin"),
                Path("/sbin"),
                Path("/root"),
            ]

            # Check if path is in forbidden directories
            for forbidden in forbidden_paths:
                if resolved_path.is_relative_to(forbidden.resolve()):
                    raise ValueError(f"Path outside allowed directories: {resolved_path}")

            # Allow paths within project root, user home, or system temp directory
            if (
                resolved_path.is_relative_to(project_root)
                or resolved_path.is_relative_to(user_home)
                or resolved_path.is_relative_to(system_temp)
            ):
                return resolved_path

            raise ValueError(f"Path outside allowed directories: {resolved_path}")

        except (OSError, ValueError) as e:
            if "Path outside allowed directories" in str(e):
                raise e
            raise ValueError(f"Invalid file path: {e}")

    def _validate_defaults(self, defaults: dict[str, str | None]) -> tuple[bool, str | None]:
        """Validate default configuration values.

        Args:
            defaults: Dictionary of default values to validate

        Returns:
            Tuple of (is_valid, error_message)
        """
        required_fields = ["default_interface", "default_provider", "default_model"]

        # Check required fields
        for field in required_fields:
            if field not in defaults:
                return False, f"Missing required field: {field}"

        # Validate interface
        valid_interfaces = ["langchain", "openrouter", "openai_endpoint"]
        if defaults["default_interface"] not in valid_interfaces:
            return False, f"Invalid interface. Must be one of: {', '.join(valid_interfaces)}"

        # For openai_endpoint, require endpoint_base_url
        if defaults["default_interface"] == "openai_endpoint":
            endpoint_url = defaults.get("default_endpoint_base_url")
            if not endpoint_url:
                return False, "endpoint_base_url is required for openai_endpoint interface"

            # Validate URL format
            try:
                from urllib.parse import urlparse

                result = urlparse(endpoint_url)
                if not all([result.scheme, result.netloc]):
                    return False, "Invalid endpoint_base_url format (must include scheme and host)"
            except Exception:
                return False, "Invalid endpoint_base_url format"

        # Validate provider (basic string validation)
        provider = defaults["default_provider"]
        if not provider or not isinstance(provider, str) or len(provider.strip()) == 0:
            return False, "Provider cannot be empty"

        # Validate model (basic string validation)
        model = defaults["default_model"]
        if not model or not isinstance(model, str) or len(model.strip()) == 0:
            return False, "Model cannot be empty"

        return True, None

    def get_defaults(self) -> dict[str, str | None]:
        """Get current default configuration.

        Returns:
            Dictionary of default configuration values
        """
        try:
            if self.defaults_file_path.exists():
                with open(self.defaults_file_path) as f:
                    saved_defaults = json.load(f)

                # Validate saved defaults
                is_valid, error = self._validate_defaults(saved_defaults)
                if is_valid:
                    logger.info("Loaded defaults from file")
                    return {
                        "default_interface": saved_defaults["default_interface"],
                        "default_provider": saved_defaults["default_provider"],
                        "default_model": saved_defaults["default_model"],
                        "default_endpoint_base_url": saved_defaults.get("default_endpoint_base_url"),
                    }
                else:
                    logger.warning(f"Invalid saved defaults: {error}, using fallback")

        except Exception as e:
            logger.error(f"Error reading defaults file: {e}, using fallback")

        # Return fallback defaults if file doesn't exist or is invalid
        logger.info("Using fallback defaults")
        return self.fallback_defaults.copy()

    def save_defaults(self, defaults: dict[str, str | None]) -> None:
        """Save default configuration to file.

        Args:
            defaults: Dictionary of default values to save

        Raises:
            ValueError: If validation fails
            IOError: If file operations fail
        """
        # Validate defaults
        is_valid, error_msg = self._validate_defaults(defaults)
        if not is_valid:
            raise ValueError(error_msg)

        # Create backup if file exists
        self._create_backup()

        try:
            # Prepare data to save
            data_to_save = {
                "default_interface": defaults["default_interface"],
                "default_provider": defaults["default_provider"],
                "default_model": defaults["default_model"],
                "default_endpoint_base_url": defaults.get("default_endpoint_base_url"),
                "saved_at": datetime.utcnow().isoformat() + "Z",
            }

            # Write to file with secure permissions
            # Use temporary file and atomic move
            temp_file = self.defaults_file_path.with_suffix(".tmp")
            fd = os.open(str(temp_file), os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(data_to_save, f, indent=2)
            except:
                os.close(fd)
                raise

            # Atomic move to replace original file
            temp_file.replace(self.defaults_file_path)

            logger.info(f"Saved defaults to {self.defaults_file_path}")

        except Exception as e:
            logger.error(f"Error saving defaults: {e}")
            self._restore_backup()
            raise

    def reset_to_fallback(self) -> None:
        """Reset defaults to hardcoded fallback values.

        This removes the defaults.json file, causing the system to use fallback defaults.
        """
        try:
            if self.defaults_file_path.exists():
                # Create backup before removal
                self._create_backup()
                self.defaults_file_path.unlink()
                logger.info("Reset defaults to fallback values")
        except Exception as e:
            logger.error(f"Error resetting defaults: {e}")
            raise

    def _create_backup(self) -> None:
        """Create a backup of the current defaults file."""
        if self.defaults_file_path.exists():
            backup_path = self.defaults_file_path.with_suffix(".json.backup")
            shutil.copy2(self.defaults_file_path, backup_path)
            logger.debug(f"Created backup at {backup_path}")

    def _restore_backup(self) -> None:
        """Restore defaults file from backup."""
        backup_path = self.defaults_file_path.with_suffix(".json.backup")
        if backup_path.exists():
            shutil.copy2(backup_path, self.defaults_file_path)
            logger.info("Restored defaults file from backup")

    def get_file_status(self) -> dict[str, Any]:
        """Get status information about the defaults file.

        Returns:
            Dictionary with file status information
        """
        status = {
            "file_exists": self.defaults_file_path.exists(),
            "file_path": str(self.defaults_file_path),
            "using_fallback": False,
            "last_modified": None,
            "saved_at": None,
        }

        if status["file_exists"]:
            try:
                # Get file modification time
                stat = self.defaults_file_path.stat()
                status["last_modified"] = datetime.fromtimestamp(stat.st_mtime).isoformat()

                # Get saved_at from file content
                with open(self.defaults_file_path) as f:
                    data = json.load(f)
                    status["saved_at"] = data.get("saved_at")

            except Exception as e:
                logger.error(f"Error getting file status: {e}")
                status["using_fallback"] = True
        else:
            status["using_fallback"] = True

        return status
