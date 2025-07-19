"""Configuration service for managing .env file and environment variables."""
# ruff: noqa: B904  # Intentionally suppress exception chaining for security

import logging
import os
import re
import shutil
from pathlib import Path

from dotenv import dotenv_values, set_key, unset_key

logger = logging.getLogger(__name__)


class ConfigurationService:
    """Service for managing application configuration via .env file."""

    def __init__(self, env_file_path: Path | None = None):
        """Initialize configuration service.

        Args:
            env_file_path: Path to .env file. If None, uses default location.
        """
        if env_file_path is None:
            # Default to .env file in project root (same directory as start-karenina.sh)
            project_root = Path(__file__).parent.parent.parent.parent.parent
            env_file_path = project_root / ".env"

        # Validate and canonicalize path to prevent traversal attacks
        self.env_file_path = self._validate_file_path(env_file_path)
        self.api_key_patterns = {
            # OpenAI keys: traditional sk- format and newer sk-proj- format
            "OPENAI_API_KEY": r"^sk-(proj-)?[a-zA-Z0-9]{20,}$",
            # Anthropic keys: sk-ant- prefix
            "ANTHROPIC_API_KEY": r"^sk-ant-[a-zA-Z0-9-]{95,}$",
            # Google API keys: 39 characters, alphanumeric with underscores/hyphens
            "GOOGLE_API_KEY": r"^[a-zA-Z0-9_-]{35,43}$",
            # OpenRouter keys: sk-or- prefix
            "OPENROUTER_API_KEY": r"^sk-or-[a-zA-Z0-9]{48,}$",
            # Gemini keys: alternative format
            "GEMINI_API_KEY": r"^[a-zA-Z0-9_-]{35,43}$",
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

    def _mask_api_key(self, key: str, value: str) -> str:
        """Mask API key for security, showing only last 4 characters.

        Args:
            key: The environment variable name
            value: The API key value

        Returns:
            Masked value or original if not an API key
        """
        if key.endswith("_API_KEY") and value and len(value) > 4:
            return "*" * (len(value) - 4) + value[-4:]
        return value

    def _validate_api_key(self, key: str, value: str) -> tuple[bool, str | None]:
        """Validate API key format.

        Args:
            key: The environment variable name
            value: The API key value

        Returns:
            Tuple of (is_valid, error_message)
        """
        if key not in self.api_key_patterns:
            # Not a known API key, allow any format
            return True, None

        pattern = self.api_key_patterns[key]
        if re.match(pattern, value):
            return True, None

        return False, f"Invalid format for {key}"

    def read_env_vars(self, mask_secrets: bool = True) -> dict[str, str]:
        """Read environment variables from .env file.

        Args:
            mask_secrets: Whether to mask sensitive values

        Returns:
            Dictionary of environment variables
        """
        if not self.env_file_path.exists():
            logger.warning(f".env file not found at {self.env_file_path}")
            return {}

        try:
            env_vars = dotenv_values(self.env_file_path)

            if mask_secrets:
                masked_vars = {}
                for key, value in env_vars.items():
                    if value:
                        masked_vars[key] = self._mask_api_key(key, value)
                    else:
                        masked_vars[key] = ""
                return masked_vars

            # Convert None values to empty strings for consistent typing
            return {k: v or "" for k, v in env_vars.items()}

        except Exception as e:
            logger.error(f"Error reading .env file: {e}")
            raise

    def update_env_var(self, key: str, value: str) -> None:
        """Update or add an environment variable in .env file.

        Args:
            key: The environment variable name
            value: The new value

        Raises:
            ValueError: If validation fails
            IOError: If file operations fail
        """
        # Validate API key format if applicable
        is_valid, error_msg = self._validate_api_key(key, value)
        if not is_valid:
            raise ValueError(error_msg)

        # Create backup before modification
        self._create_backup()

        try:
            # Ensure .env file exists with secure permissions
            if not self.env_file_path.exists():
                # Create file with secure permissions atomically
                fd = os.open(str(self.env_file_path), os.O_CREAT | os.O_WRONLY | os.O_EXCL, 0o600)
                os.close(fd)

            # Update or add the key
            set_key(str(self.env_file_path), key, value)

            # Also update the current process environment
            os.environ[key] = value

            logger.info(f"Updated environment variable: {key}")

        except Exception as e:
            logger.error(f"Error updating .env file: {e}")
            self._restore_backup()
            raise

    def remove_env_var(self, key: str) -> None:
        """Remove an environment variable from .env file.

        Args:
            key: The environment variable name to remove
        """
        if not self.env_file_path.exists():
            return

        # Create backup before modification
        self._create_backup()

        try:
            unset_key(str(self.env_file_path), key)

            # Also remove from current process environment
            if key in os.environ:
                del os.environ[key]

            logger.info(f"Removed environment variable: {key}")

        except Exception as e:
            logger.error(f"Error removing from .env file: {e}")
            self._restore_backup()
            raise

    def get_env_file_contents(self) -> str:
        """Get the raw contents of the .env file.

        Returns:
            The contents of the .env file as a string
        """
        if not self.env_file_path.exists():
            return ""

        try:
            return self.env_file_path.read_text()
        except Exception as e:
            logger.error(f"Error reading .env file: {e}")
            raise

    def update_env_file_contents(self, contents: str) -> None:
        """Update the entire .env file contents.

        Args:
            contents: The new contents for the .env file
        """
        # Create backup before modification
        self._create_backup()

        try:
            # Write new contents with secure permissions
            # Use temporary file and atomic move to ensure security
            temp_file = self.env_file_path.with_suffix(".tmp")
            fd = os.open(str(temp_file), os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o600)
            try:
                os.write(fd, contents.encode("utf-8"))
            finally:
                os.close(fd)

            # Atomic move to replace original file
            temp_file.replace(self.env_file_path)

            # Reload environment variables
            from dotenv import load_dotenv

            load_dotenv(self.env_file_path, override=True)

            logger.info("Updated .env file contents")

        except Exception as e:
            logger.error(f"Error writing .env file: {e}")
            self._restore_backup()
            raise

    def update_env_vars_bulk(self, updates: list[tuple[str, str]]) -> None:
        """Update multiple environment variables atomically.

        All updates are validated first, then applied as a single transaction.
        If any validation fails, no changes are made.

        Args:
            updates: List of (key, value) tuples to update

        Raises:
            ValueError: If any validation fails
            IOError: If file operations fail
        """
        # Phase 1: Validate all updates without making changes
        validation_errors = []
        for key, value in updates:
            is_valid, error_msg = self._validate_api_key(key, value)
            if not is_valid:
                validation_errors.append(f"{key}: {error_msg}")

        if validation_errors:
            raise ValueError(f"Validation failed: {'; '.join(validation_errors)}")

        # Phase 2: Create backup before any modifications
        self._create_backup()

        try:
            # Phase 3: Apply all updates
            for key, value in updates:
                # Ensure .env file exists with secure permissions
                if not self.env_file_path.exists():
                    fd = os.open(str(self.env_file_path), os.O_CREAT | os.O_WRONLY | os.O_EXCL, 0o600)
                    os.close(fd)

                # Update the key
                set_key(str(self.env_file_path), key, value)

                # Also update the current process environment
                os.environ[key] = value

            logger.info(f"Successfully updated {len(updates)} environment variables")

        except Exception as e:
            logger.error(f"Error during bulk update: {e}")
            self._restore_backup()
            raise

    def _create_backup(self) -> None:
        """Create a backup of the current .env file."""
        if self.env_file_path.exists():
            backup_path = self.env_file_path.with_suffix(".env.backup")
            shutil.copy2(self.env_file_path, backup_path)
            logger.debug(f"Created backup at {backup_path}")

    def _restore_backup(self) -> None:
        """Restore .env file from backup."""
        backup_path = self.env_file_path.with_suffix(".env.backup")
        if backup_path.exists():
            shutil.copy2(backup_path, self.env_file_path)
            logger.info("Restored .env file from backup")

    def validate_provider_config(self, provider: str) -> tuple[bool, str | None]:
        """Validate if a provider has required configuration.

        Args:
            provider: The provider name (openai, google_genai, anthropic, openrouter)

        Returns:
            Tuple of (is_configured, error_message)
        """
        env_vars = self.read_env_vars(mask_secrets=False)

        required_keys = {
            "openai": "OPENAI_API_KEY",
            "google_genai": "GOOGLE_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "openrouter": "OPENROUTER_API_KEY",
        }

        if provider not in required_keys:
            return False, f"Unknown provider: {provider}"

        required_key = required_keys[provider]
        if required_key not in env_vars or not env_vars[required_key]:
            return False, f"Missing {required_key} for {provider}"

        return True, None
