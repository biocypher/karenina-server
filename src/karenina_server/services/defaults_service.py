"""Service for managing persistent default LLM configuration settings."""
# ruff: noqa: B904  # Intentionally suppress exception chaining for security

import contextlib
import fcntl
import json
import logging
import os
import shutil
from collections.abc import Generator
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
            # Anthropic-specific defaults (for claude_tool and claude_agent_sdk interfaces)
            "default_anthropic_base_url": None,
            "default_anthropic_opus_model": None,
            "default_anthropic_sonnet_model": None,
            "default_anthropic_haiku_model": None,
        }

        # Lock file path (adjacent to defaults file)
        self._lock_file_path = self.defaults_file_path.with_suffix(".lock")

    @contextlib.contextmanager
    def _file_lock(self, exclusive: bool = False) -> Generator[None, None, None]:
        """Acquire a file lock for concurrent access safety.

        Uses fcntl.flock() for advisory locking. This prevents race conditions
        when multiple processes try to read/write the defaults file.

        Args:
            exclusive: If True, acquire exclusive (write) lock. If False, acquire shared (read) lock.

        Yields:
            None when lock is acquired.
        """
        # Ensure lock file parent directory exists
        self._lock_file_path.parent.mkdir(parents=True, exist_ok=True)

        # Open lock file (create if doesn't exist)
        lock_fd = os.open(str(self._lock_file_path), os.O_CREAT | os.O_RDWR, 0o644)
        try:
            # Acquire lock (blocking)
            lock_type = fcntl.LOCK_EX if exclusive else fcntl.LOCK_SH
            fcntl.flock(lock_fd, lock_type)
            try:
                yield
            finally:
                # Release lock
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)

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
        valid_interfaces = ["langchain", "openrouter", "openai_endpoint", "claude_tool", "claude_agent_sdk"]
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

        # Validate anthropic_base_url format if provided
        anthropic_base_url = defaults.get("default_anthropic_base_url")
        if anthropic_base_url:
            try:
                from urllib.parse import urlparse

                result = urlparse(anthropic_base_url)
                if not all([result.scheme, result.netloc]):
                    return False, "Invalid anthropic_base_url format (must include scheme and host)"
            except Exception:
                return False, "Invalid anthropic_base_url format"

        # For claude_tool and claude_agent_sdk, provider is implicitly anthropic
        interface = defaults["default_interface"]
        if interface in ("claude_tool", "claude_agent_sdk"):
            # Provider is auto-set to anthropic for these interfaces
            defaults["default_provider"] = "anthropic"

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

        Uses a shared (read) lock to prevent reading during concurrent writes.
        Handles FileNotFoundError and json.JSONDecodeError gracefully by
        returning fallback defaults.

        Returns:
            Dictionary of default configuration values
        """
        try:
            with self._file_lock(exclusive=False):
                # Try to open directly - handles TOCTOU by relying on exception
                # instead of exists() check
                try:
                    with open(self.defaults_file_path) as f:
                        saved_defaults = json.load(f)
                except FileNotFoundError:
                    logger.info("Defaults file not found, using fallback defaults")
                    return self.fallback_defaults.copy()
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON in defaults file: {e}, using fallback")
                    return self.fallback_defaults.copy()

                # Validate saved defaults
                is_valid, error = self._validate_defaults(saved_defaults)
                if is_valid:
                    logger.info("Loaded defaults from file")
                    return {
                        "default_interface": saved_defaults["default_interface"],
                        "default_provider": saved_defaults["default_provider"],
                        "default_model": saved_defaults["default_model"],
                        "default_endpoint_base_url": saved_defaults.get("default_endpoint_base_url"),
                        # Anthropic-specific defaults
                        "default_anthropic_base_url": saved_defaults.get("default_anthropic_base_url"),
                        "default_anthropic_opus_model": saved_defaults.get("default_anthropic_opus_model"),
                        "default_anthropic_sonnet_model": saved_defaults.get("default_anthropic_sonnet_model"),
                        "default_anthropic_haiku_model": saved_defaults.get("default_anthropic_haiku_model"),
                    }
                else:
                    logger.warning(f"Invalid saved defaults: {error}, using fallback")
                    return self.fallback_defaults.copy()

        except Exception as e:
            logger.error(f"Error reading defaults file: {e}, using fallback")

        # Return fallback defaults if file doesn't exist or is invalid
        logger.info("Using fallback defaults")
        return self.fallback_defaults.copy()

    def save_defaults(self, defaults: dict[str, str | None]) -> None:
        """Save default configuration to file.

        Uses an exclusive (write) lock to prevent concurrent reads/writes.
        Uses atomic write pattern (write to temp file, then rename) to ensure
        file is never in a partial state.

        Args:
            defaults: Dictionary of default values to save

        Raises:
            ValueError: If validation fails
            IOError: If file operations fail
        """
        # Validate defaults (outside lock since it doesn't touch files)
        is_valid, error_msg = self._validate_defaults(defaults)
        if not is_valid:
            raise ValueError(error_msg)

        with self._file_lock(exclusive=True):
            # Create backup if file exists (now protected by lock)
            self._create_backup_unlocked()

            try:
                # Prepare data to save
                data_to_save = {
                    "default_interface": defaults["default_interface"],
                    "default_provider": defaults["default_provider"],
                    "default_model": defaults["default_model"],
                    "default_endpoint_base_url": defaults.get("default_endpoint_base_url"),
                    # Anthropic-specific defaults
                    "default_anthropic_base_url": defaults.get("default_anthropic_base_url"),
                    "default_anthropic_opus_model": defaults.get("default_anthropic_opus_model"),
                    "default_anthropic_sonnet_model": defaults.get("default_anthropic_sonnet_model"),
                    "default_anthropic_haiku_model": defaults.get("default_anthropic_haiku_model"),
                    "saved_at": datetime.utcnow().isoformat() + "Z",
                }

                # Ensure parent directory exists
                self.defaults_file_path.parent.mkdir(parents=True, exist_ok=True)

                # Write to temp file with secure permissions, then atomic rename
                temp_file = self.defaults_file_path.with_suffix(".tmp")
                fd = os.open(str(temp_file), os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
                try:
                    with os.fdopen(fd, "w") as f:
                        json.dump(data_to_save, f, indent=2)
                except Exception:
                    os.close(fd)
                    raise

                # Atomic move to replace original file
                temp_file.replace(self.defaults_file_path)

                logger.info(f"Saved defaults to {self.defaults_file_path}")

            except Exception as e:
                logger.error(f"Error saving defaults: {e}")
                self._restore_backup_unlocked()
                raise

    def reset_to_fallback(self) -> None:
        """Reset defaults to hardcoded fallback values.

        Uses an exclusive lock. This removes the defaults.json file, causing the
        system to use fallback defaults.
        """
        with self._file_lock(exclusive=True):
            try:
                # Try to unlink directly - handles TOCTOU by relying on exception
                # instead of exists() check
                try:
                    # Create backup before removal
                    self._create_backup_unlocked()
                    self.defaults_file_path.unlink()
                    logger.info("Reset defaults to fallback values")
                except FileNotFoundError:
                    # File already doesn't exist, which is fine
                    logger.info("Defaults file already absent, nothing to reset")
            except Exception as e:
                logger.error(f"Error resetting defaults: {e}")
                raise

    def _create_backup(self) -> None:
        """Create a backup of the current defaults file (with lock)."""
        with self._file_lock(exclusive=False):
            self._create_backup_unlocked()

    def _create_backup_unlocked(self) -> None:
        """Create a backup of the current defaults file (caller must hold lock).

        Uses try/except instead of exists() check to handle TOCTOU race.
        """
        backup_path = self.defaults_file_path.with_suffix(".json.backup")
        try:
            shutil.copy2(self.defaults_file_path, backup_path)
            logger.debug(f"Created backup at {backup_path}")
        except FileNotFoundError:
            # Original file doesn't exist, nothing to backup
            pass

    def _restore_backup(self) -> None:
        """Restore defaults file from backup (with lock)."""
        with self._file_lock(exclusive=True):
            self._restore_backup_unlocked()

    def _restore_backup_unlocked(self) -> None:
        """Restore defaults file from backup (caller must hold lock).

        Uses try/except instead of exists() check to handle TOCTOU race.
        """
        backup_path = self.defaults_file_path.with_suffix(".json.backup")
        try:
            shutil.copy2(backup_path, self.defaults_file_path)
            logger.info("Restored defaults file from backup")
        except FileNotFoundError:
            # Backup file doesn't exist, nothing to restore
            logger.warning("No backup file found to restore")

    def get_file_status(self) -> dict[str, Any]:
        """Get status information about the defaults file.

        Uses a shared (read) lock for consistent status retrieval.

        Returns:
            Dictionary with file status information
        """
        status: dict[str, Any] = {
            "file_exists": False,
            "file_path": str(self.defaults_file_path),
            "using_fallback": True,
            "last_modified": None,
            "saved_at": None,
        }

        try:
            with self._file_lock(exclusive=False):
                # Try to stat and read the file directly - handles TOCTOU
                try:
                    # Get file modification time
                    stat = self.defaults_file_path.stat()
                    status["file_exists"] = True
                    status["using_fallback"] = False
                    status["last_modified"] = datetime.fromtimestamp(stat.st_mtime).isoformat()

                    # Get saved_at from file content
                    with open(self.defaults_file_path) as f:
                        data = json.load(f)
                        status["saved_at"] = data.get("saved_at")

                except FileNotFoundError:
                    # File doesn't exist, status already defaults to using_fallback=True
                    pass
                except json.JSONDecodeError as e:
                    logger.warning(f"Invalid JSON in defaults file during status check: {e}")
                    status["file_exists"] = True
                    status["using_fallback"] = True

        except Exception as e:
            logger.error(f"Error getting file status: {e}")

        return status
