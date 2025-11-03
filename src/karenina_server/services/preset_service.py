"""Service for managing benchmark configuration presets."""

import json
import logging
import os
import uuid
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from karenina.schemas.workflow.verification import VerificationConfig

logger = logging.getLogger(__name__)


class BenchmarkPresetService:
    """Service for managing benchmark configuration preset persistence."""

    def __init__(self, presets_dir_path: Path | None = None):
        """Initialize preset service.

        Args:
            presets_dir_path: Path to presets directory. If None, uses default location from env or project root.
        """
        if presets_dir_path is None:
            # Check environment variable first
            env_presets_dir = os.getenv("KARENINA_PRESETS_DIR")
            if env_presets_dir:
                presets_dir_path = Path(env_presets_dir)
            else:
                # Default to benchmark_presets/ directory in project root
                project_root = Path(__file__).parent.parent.parent.parent.parent
                presets_dir_path = project_root / "benchmark_presets"

        # Validate and canonicalize path to prevent traversal attacks
        self.presets_dir_path = self._validate_dir_path(presets_dir_path)

        # Ensure directory exists
        self._ensure_dir_exists()

    def _validate_dir_path(self, dir_path: Path) -> Path:
        """Validate directory path to prevent directory traversal attacks.

        Args:
            dir_path: Path to validate

        Returns:
            Canonicalized safe path

        Raises:
            ValueError: If path is unsafe
        """
        try:
            # Resolve to absolute path and normalize
            resolved_path = dir_path.resolve()

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
                raise
            raise ValueError(f"Invalid directory path: {e}") from e

    def _ensure_dir_exists(self) -> None:
        """Ensure presets directory exists."""
        if not self.presets_dir_path.exists():
            try:
                self.presets_dir_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Created presets directory at {self.presets_dir_path}")
            except Exception as e:
                logger.error(f"Error creating presets directory: {e}")
                raise

    def _load_preset_from_file(self, filepath: Path) -> dict[str, Any] | None:
        """Load a single preset from a JSON file.

        Args:
            filepath: Path to preset JSON file

        Returns:
            Preset dictionary or None if load fails
        """
        try:
            with open(filepath) as f:
                preset: dict[str, Any] = json.load(f)
                return preset
        except Exception as e:
            logger.error(f"Error loading preset from {filepath}: {e}")
            return None

    def _scan_presets(self) -> dict[str, dict[str, Any]]:
        """Scan presets directory and load all preset files.

        Returns:
            Dictionary of presets keyed by preset ID
        """
        presets: dict[str, dict[str, Any]] = {}

        if not self.presets_dir_path.exists():
            return presets

        # Scan for .json files
        for filepath in self.presets_dir_path.glob("*.json"):
            preset = self._load_preset_from_file(filepath)
            if preset and "id" in preset:
                presets[preset["id"]] = preset

        return presets

    def _save_preset_to_file(self, preset: dict[str, Any], filename: str) -> None:
        """Save a preset to a JSON file.

        Args:
            preset: Preset dictionary
            filename: Filename to save as

        Raises:
            IOError: If file operations fail
        """
        filepath = self.presets_dir_path / filename

        try:
            # Write to file with secure permissions
            fd = os.open(str(filepath), os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
            try:
                with os.fdopen(fd, "w") as f:
                    json.dump(preset, f, indent=2)
            except:
                os.close(fd)
                raise

            logger.info(f"Saved preset to {filepath}")

        except Exception as e:
            logger.error(f"Error saving preset to {filepath}: {e}")
            raise

    def _find_preset_file(self, preset_id: str) -> Path | None:
        """Find the file path for a preset by its ID.

        Args:
            preset_id: Preset UUID

        Returns:
            Path to preset file or None if not found
        """
        # Scan all JSON files for matching ID
        for filepath in self.presets_dir_path.glob("*.json"):
            preset = self._load_preset_from_file(filepath)
            if preset and preset.get("id") == preset_id:
                return filepath
        return None

    def _validate_preset_data(self, name: str, description: str | None, preset_id: str | None = None) -> None:
        """Validate preset metadata with uniqueness check.

        Args:
            name: Preset name
            description: Optional description
            preset_id: Preset ID to exclude from uniqueness check (for updates)

        Raises:
            ValueError: If validation fails
        """
        # Use core validation for basic checks (length limits)
        from karenina.schemas.workflow.verification import VerificationConfig

        VerificationConfig.validate_preset_metadata(name, description)

        # Additional type check for description (server-specific)
        if description is not None and not isinstance(description, str):
            raise ValueError("Description must be a string")

        # Server-specific: Check name uniqueness across all presets
        presets = self._scan_presets()
        for pid, preset in presets.items():
            # Skip the preset being updated
            if preset_id and pid == preset_id:
                continue
            if preset.get("name") == name:
                raise ValueError(f"A preset with name '{name}' already exists")

    def list_presets(self) -> dict[str, dict[str, Any]]:
        """Get all presets by scanning the presets directory.

        Returns:
            Dictionary of presets keyed by preset ID
        """
        return self._scan_presets()

    def get_preset(self, preset_id: str) -> dict[str, Any]:
        """Get a specific preset by ID.

        Args:
            preset_id: Preset ID

        Returns:
            Preset data dictionary

        Raises:
            ValueError: If preset not found
        """
        filepath = self._find_preset_file(preset_id)
        if not filepath:
            raise ValueError(f"Preset with ID '{preset_id}' not found")

        preset = self._load_preset_from_file(filepath)
        if not preset:
            raise ValueError(f"Failed to load preset with ID '{preset_id}'")

        return preset

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
        from karenina.schemas.workflow.verification import VerificationConfig

        config_dict = config.model_dump(mode="json")

        # Sanitize model configurations using core utility
        if "answering_models" in config_dict:
            config_dict["answering_models"] = [
                VerificationConfig.sanitize_model_config(m) for m in config_dict["answering_models"]
            ]
        if "parsing_models" in config_dict:
            config_dict["parsing_models"] = [
                VerificationConfig.sanitize_model_config(m) for m in config_dict["parsing_models"]
            ]

        # Create preset structure using core utility
        preset: dict[str, Any] = VerificationConfig.create_preset_structure(
            preset_id=preset_id,
            name=name,
            description=description,
            config_dict=config_dict,
            created_at=now,
            updated_at=now,
        )

        # Generate filename and save using core utility
        filename = VerificationConfig.sanitize_preset_name(name)
        self._save_preset_to_file(preset, filename)

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
        # Find existing preset file
        old_filepath = self._find_preset_file(preset_id)
        if not old_filepath:
            raise ValueError(f"Preset with ID '{preset_id}' not found")

        # Load existing preset
        preset = self._load_preset_from_file(old_filepath)
        if not preset:
            raise ValueError(f"Failed to load preset with ID '{preset_id}'")

        old_name = preset.get("name")

        # Update name if provided
        if name is not None:
            self._validate_preset_data(name, None, preset_id=preset_id)
            preset["name"] = name

        # Update config if provided
        if config is not None:
            from karenina.schemas.workflow.verification import VerificationConfig

            config_dict = config.model_dump(mode="json")

            # Sanitize model configurations using core utility
            if "answering_models" in config_dict:
                config_dict["answering_models"] = [
                    VerificationConfig.sanitize_model_config(m) for m in config_dict["answering_models"]
                ]
            if "parsing_models" in config_dict:
                config_dict["parsing_models"] = [
                    VerificationConfig.sanitize_model_config(m) for m in config_dict["parsing_models"]
                ]

            preset["config"] = config_dict

        # Update description if provided (None means don't change, empty string clears it)
        if description is not None:
            if len(description) > 500:
                raise ValueError("Description cannot exceed 500 characters")
            preset["description"] = description if description else None

        # Update timestamp
        preset["updated_at"] = datetime.now(UTC).isoformat()

        # Determine new filename using core utility
        from karenina.schemas.workflow.verification import VerificationConfig

        new_name = preset["name"]
        new_filename = VerificationConfig.sanitize_preset_name(new_name)
        new_filepath = self.presets_dir_path / new_filename

        # If name changed, we need to check if file needs to be renamed
        if name is not None and old_name != new_name:
            # Check if new filename would conflict with an existing file (different preset)
            if new_filepath.exists():
                # Check if it's a different preset
                existing_preset = self._load_preset_from_file(new_filepath)
                if existing_preset and existing_preset.get("id") != preset_id:
                    # Conflict with another preset - this shouldn't happen due to name validation
                    # but handle it gracefully
                    raise ValueError("Filename conflict: another preset uses the same sanitized filename")

            # Safe to rename: delete old file, write new one
            old_filepath.unlink()
            self._save_preset_to_file(preset, new_filename)
        else:
            # No name change, just update the existing file
            self._save_preset_to_file(preset, old_filepath.name)

        logger.info(f"Updated preset '{preset['name']}' (ID: {preset_id})")
        return preset

    def delete_preset(self, preset_id: str) -> None:
        """Delete a preset.

        Args:
            preset_id: Preset ID to delete

        Raises:
            ValueError: If preset not found
        """
        filepath = self._find_preset_file(preset_id)
        if not filepath:
            raise ValueError(f"Preset with ID '{preset_id}' not found")

        # Load preset to get name for logging
        preset = self._load_preset_from_file(filepath)
        preset_name = preset.get("name", "Unknown") if preset else "Unknown"

        # Delete file
        filepath.unlink()

        logger.info(f"Deleted preset '{preset_name}' (ID: {preset_id})")

    def get_directory_status(self) -> dict[str, Any]:
        """Get status information about the presets directory.

        Returns:
            Dictionary with directory status information
        """
        status = {
            "directory_exists": self.presets_dir_path.exists(),
            "directory_path": str(self.presets_dir_path),
            "preset_count": 0,
        }

        if status["directory_exists"]:
            try:
                # Count presets
                presets = self._scan_presets()
                status["preset_count"] = len(presets)

            except Exception as e:
                logger.error(f"Error getting directory status: {e}")

        return status
