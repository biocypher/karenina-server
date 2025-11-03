"""Service for managing benchmark configuration presets."""
# ruff: noqa: B904  # Intentionally suppress exception chaining for security

import json
import logging
import os
import shutil
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from karenina.schemas.workflow.verification import VerificationConfig

logger = logging.getLogger(__name__)


class BenchmarkPresetService:
    """Service for managing benchmark configuration preset persistence."""

    def __init__(self, presets_file_path: Path | None = None):
        """Initialize preset service.

        Args:
            presets_file_path: Path to benchmark_presets.json file. If None, uses default location.
        """
        if presets_file_path is None:
            # Default to benchmark_presets.json file in project root
            project_root = Path(__file__).parent.parent.parent.parent.parent
            presets_file_path = project_root / "benchmark_presets.json"

        # Validate and canonicalize path to prevent traversal attacks
        self.presets_file_path = self._validate_file_path(presets_file_path)

        # Ensure file exists with empty presets structure
        self._ensure_file_exists()

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

    def _ensure_file_exists(self) -> None:
        """Ensure presets file exists with empty structure if not present."""
        if not self.presets_file_path.exists():
            try:
                self.presets_file_path.parent.mkdir(parents=True, exist_ok=True)
                initial_data: dict[str, dict[str, Any]] = {"presets": {}}
                with open(self.presets_file_path, "w") as f:
                    json.dump(initial_data, f, indent=2)
                logger.info(f"Created initial presets file at {self.presets_file_path}")
            except Exception as e:
                logger.error(f"Error creating presets file: {e}")
                raise

    def _load_presets(self) -> dict[str, dict[str, Any]]:
        """Load presets from file.

        Returns:
            Dictionary of presets keyed by preset ID
        """
        try:
            with open(self.presets_file_path) as f:
                data: dict[str, Any] = json.load(f)
                presets: dict[str, dict[str, Any]] = data.get("presets", {})
                return presets
        except Exception as e:
            logger.error(f"Error loading presets: {e}")
            return {}

    def _save_presets(self, presets: dict[str, dict[str, Any]]) -> None:
        """Save presets to file with atomic write.

        Args:
            presets: Dictionary of presets to save

        Raises:
            IOError: If file operations fail
        """
        # Create backup if file exists
        self._create_backup()

        try:
            data_to_save = {"presets": presets}

            # Write to file with secure permissions
            # Use temporary file and atomic move
            temp_file = self.presets_file_path.with_suffix(".tmp")
            fd = os.open(str(temp_file), os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(data_to_save, f, indent=2)
            except:
                os.close(fd)
                raise

            # Atomic move to replace original file
            temp_file.replace(self.presets_file_path)

            logger.info(f"Saved presets to {self.presets_file_path}")

        except Exception as e:
            logger.error(f"Error saving presets: {e}")
            self._restore_backup()
            raise

    def _create_backup(self) -> None:
        """Create a backup of the current presets file."""
        if self.presets_file_path.exists():
            backup_path = self.presets_file_path.with_suffix(".json.backup")
            shutil.copy2(self.presets_file_path, backup_path)
            logger.debug(f"Created backup at {backup_path}")

    def _restore_backup(self) -> None:
        """Restore presets file from backup."""
        backup_path = self.presets_file_path.with_suffix(".json.backup")
        if backup_path.exists():
            shutil.copy2(backup_path, self.presets_file_path)
            logger.info("Restored presets file from backup")

    def _validate_preset_data(self, name: str, description: str | None, preset_id: str | None = None) -> None:
        """Validate preset metadata.

        Args:
            name: Preset name
            description: Optional description
            preset_id: Preset ID to exclude from uniqueness check (for updates)

        Raises:
            ValueError: If validation fails
        """
        # Validate name
        if not name or not isinstance(name, str) or len(name.strip()) == 0:
            raise ValueError("Preset name cannot be empty")

        if len(name) > 100:
            raise ValueError("Preset name cannot exceed 100 characters")

        # Validate description if provided
        if description is not None:
            if not isinstance(description, str):
                raise ValueError("Description must be a string")
            if len(description) > 500:
                raise ValueError("Description cannot exceed 500 characters")

        # Check name uniqueness
        presets = self._load_presets()
        for pid, preset in presets.items():
            # Skip the preset being updated
            if preset_id and pid == preset_id:
                continue
            if preset.get("name") == name:
                raise ValueError(f"A preset with name '{name}' already exists")

    def list_presets(self) -> dict[str, dict[str, Any]]:
        """Get all presets.

        Returns:
            Dictionary of presets keyed by preset ID
        """
        return self._load_presets()

    def get_preset(self, preset_id: str) -> dict[str, Any]:
        """Get a specific preset by ID.

        Args:
            preset_id: Preset ID

        Returns:
            Preset data dictionary

        Raises:
            ValueError: If preset not found
        """
        presets = self._load_presets()
        if preset_id not in presets:
            raise ValueError(f"Preset with ID '{preset_id}' not found")
        return presets[preset_id]

    def create_preset(
        self,
        name: str,
        config: VerificationConfig,
        description: str | None = None,
    ) -> dict[str, Any]:
        """Create a new preset.

        Args:
            name: Preset name
            config: VerificationConfig instance
            description: Optional description

        Returns:
            Created preset data

        Raises:
            ValueError: If validation fails
        """
        # Validate metadata
        self._validate_preset_data(name, description)

        # Generate UUID and timestamps
        preset_id = str(uuid.uuid4())
        now = datetime.now(UTC).isoformat()

        # Convert config to dict (Pydantic model_dump)
        config_dict = config.model_dump(mode="json")

        # Create preset
        preset = {
            "id": preset_id,
            "name": name,
            "description": description,
            "config": config_dict,
            "created_at": now,
            "updated_at": now,
        }

        # Load, update, and save
        presets = self._load_presets()
        presets[preset_id] = preset
        self._save_presets(presets)

        logger.info(f"Created preset '{name}' with ID {preset_id}")
        return preset

    def update_preset(
        self,
        preset_id: str,
        name: str | None = None,
        config: VerificationConfig | None = None,
        description: str | None = None,
    ) -> dict[str, Any]:
        """Update an existing preset.

        Args:
            preset_id: Preset ID to update
            name: New name (optional)
            config: New VerificationConfig (optional)
            description: New description (optional, use empty string to clear)

        Returns:
            Updated preset data

        Raises:
            ValueError: If preset not found or validation fails
        """
        # Load presets
        presets = self._load_presets()
        if preset_id not in presets:
            raise ValueError(f"Preset with ID '{preset_id}' not found")

        preset = presets[preset_id]

        # Update name if provided
        if name is not None:
            self._validate_preset_data(name, None, preset_id=preset_id)
            preset["name"] = name

        # Update config if provided
        if config is not None:
            config_dict = config.model_dump(mode="json")
            preset["config"] = config_dict

        # Update description if provided (None means don't change, empty string clears it)
        if description is not None:
            if len(description) > 500:
                raise ValueError("Description cannot exceed 500 characters")
            preset["description"] = description if description else None

        # Update timestamp
        preset["updated_at"] = datetime.now(UTC).isoformat()

        # Save
        presets[preset_id] = preset
        self._save_presets(presets)

        logger.info(f"Updated preset '{preset['name']}' (ID: {preset_id})")
        return preset

    def delete_preset(self, preset_id: str) -> None:
        """Delete a preset.

        Args:
            preset_id: Preset ID to delete

        Raises:
            ValueError: If preset not found
        """
        presets = self._load_presets()
        if preset_id not in presets:
            raise ValueError(f"Preset with ID '{preset_id}' not found")

        preset_name = presets[preset_id].get("name", "Unknown")
        del presets[preset_id]
        self._save_presets(presets)

        logger.info(f"Deleted preset '{preset_name}' (ID: {preset_id})")

    def get_file_status(self) -> dict[str, Any]:
        """Get status information about the presets file.

        Returns:
            Dictionary with file status information
        """
        status = {
            "file_exists": self.presets_file_path.exists(),
            "file_path": str(self.presets_file_path),
            "preset_count": 0,
            "last_modified": None,
        }

        if status["file_exists"]:
            try:
                # Get file modification time
                stat = self.presets_file_path.stat()
                status["last_modified"] = datetime.fromtimestamp(stat.st_mtime).isoformat()

                # Count presets
                presets = self._load_presets()
                status["preset_count"] = len(presets)

            except Exception as e:
                logger.error(f"Error getting file status: {e}")

        return status
