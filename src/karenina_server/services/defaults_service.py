"""Service for managing persistent default LLM configuration settings."""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path

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

        self.defaults_file_path = defaults_file_path

        # Hardcoded fallback defaults
        self.fallback_defaults = {
            "default_interface": "langchain",
            "default_provider": "google_genai",
            "default_model": "gemini-pro",
        }

    def _validate_defaults(self, defaults: dict[str, str]) -> tuple[bool, str | None]:
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
        valid_interfaces = ["langchain", "openrouter"]
        if defaults["default_interface"] not in valid_interfaces:
            return False, f"Invalid interface. Must be one of: {', '.join(valid_interfaces)}"

        # Validate provider (basic string validation)
        provider = defaults["default_provider"]
        if not provider or not isinstance(provider, str) or len(provider.strip()) == 0:
            return False, "Provider cannot be empty"

        # Validate model (basic string validation)
        model = defaults["default_model"]
        if not model or not isinstance(model, str) or len(model.strip()) == 0:
            return False, "Model cannot be empty"

        return True, None

    def get_defaults(self) -> dict[str, str]:
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
                    }
                else:
                    logger.warning(f"Invalid saved defaults: {error}, using fallback")

        except Exception as e:
            logger.error(f"Error reading defaults file: {e}, using fallback")

        # Return fallback defaults if file doesn't exist or is invalid
        logger.info("Using fallback defaults")
        return self.fallback_defaults.copy()

    def save_defaults(self, defaults: dict[str, str]) -> None:
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
                "saved_at": datetime.utcnow().isoformat() + "Z",
            }

            # Write to file
            with open(self.defaults_file_path, "w") as f:
                json.dump(data_to_save, f, indent=2)

            # Set proper file permissions
            self.defaults_file_path.chmod(0o644)

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

    def get_file_status(self) -> dict[str, any]:
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
