"""Service for managing MCP server configuration presets."""

import json
import logging
import os
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)


class MCPPresetService:
    """Service for managing MCP server presets stored as JSON files.

    Presets are stored in the mcp_presets/ directory. The directory location
    can be configured via the MCP_PRESETS_DIR environment variable, which
    should point to the parent directory (mcp_presets/ will be created inside).

    Defaults to mcp_presets/ in the current working directory if not specified.
    """

    def __init__(self, presets_dir_path: Path | None = None):
        """Initialize MCP preset service.

        Args:
            presets_dir_path: Path to MCP presets directory. If None, uses default location.
        """
        if presets_dir_path is None:
            # Check environment variable first, otherwise use current working directory
            env_presets_dir = os.getenv("MCP_PRESETS_DIR")
            presets_dir_path = Path(env_presets_dir) / "mcp_presets" if env_presets_dir else Path.cwd() / "mcp_presets"

        # Validate and canonicalize path
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

            # Forbidden paths
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
                logger.info(f"Created MCP presets directory at {self.presets_dir_path}")
            except Exception as e:
                logger.error(f"Error creating MCP presets directory: {e}")
                raise

    def _sanitize_preset_name(self, name: str) -> str:
        """Sanitize preset name to create a safe filename.

        Args:
            name: Preset name

        Returns:
            Safe filename (with .json extension)
        """
        # Remove or replace unsafe characters
        safe_name = "".join(c if c.isalnum() or c in (" ", "-", "_") else "_" for c in name)
        # Replace spaces with underscores and convert to lowercase
        safe_name = safe_name.replace(" ", "_").lower().strip("_")
        # Limit length
        safe_name = safe_name[:100]
        # Add .json extension
        return f"{safe_name}.json"

    def _load_preset_from_file(self, filepath: Path) -> dict[str, Any] | None:
        """Load a single MCP preset from a JSON file.

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
            logger.error(f"Error loading MCP preset from {filepath}: {e}")
            return None

    def _scan_presets(self) -> dict[str, dict[str, Any]]:
        """Scan presets directory and load all MCP preset files.

        Returns:
            Dictionary of presets keyed by server name
        """
        presets: dict[str, dict[str, Any]] = {}

        if not self.presets_dir_path.exists():
            return presets

        # Scan for .json files
        for filepath in self.presets_dir_path.glob("*.json"):
            preset = self._load_preset_from_file(filepath)
            if preset and "name" in preset and "url" in preset:
                presets[preset["name"]] = preset

        return presets

    def list_all_presets(self) -> dict[str, dict[str, Any]]:
        """Get all MCP presets from the presets directory.

        Returns:
            Dictionary of presets keyed by server name
        """
        return self._scan_presets()

    def save_preset(self, name: str, url: str, tools: list[str] | None = None) -> dict[str, Any]:
        """Save a new or update existing MCP preset.

        Args:
            name: Server name
            url: Server URL
            tools: Optional list of tool names to filter

        Returns:
            Saved preset data

        Raises:
            ValueError: If validation fails
        """
        # Validate inputs
        if not name or not isinstance(name, str):
            raise ValueError("Name must be a non-empty string")

        if not url or not isinstance(url, str):
            raise ValueError("URL must be a non-empty string")

        if tools is not None and not isinstance(tools, list):
            raise ValueError("Tools must be a list or None")

        # Create preset structure
        preset: dict[str, Any] = {
            "name": name,
            "url": url,
        }

        if tools:
            preset["tools"] = tools

        # Save to file
        filename = self._sanitize_preset_name(name)
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

            logger.info(f"Saved MCP preset '{name}' to {filepath}")
            return preset

        except Exception as e:
            logger.error(f"Error saving MCP preset to {filepath}: {e}")
            raise

    def delete_preset(self, name: str) -> None:
        """Delete an MCP preset.

        Args:
            name: Server name

        Raises:
            ValueError: If preset not found
        """
        # Check if this preset exists
        presets = self._scan_presets()
        if name not in presets:
            raise ValueError(f"Preset '{name}' not found")

        # Find and delete the file
        filename = self._sanitize_preset_name(name)
        filepath = self.presets_dir_path / filename

        if not filepath.exists():
            # Try to find any file containing this preset
            for file in self.presets_dir_path.glob("*.json"):
                preset = self._load_preset_from_file(file)
                if preset and preset.get("name") == name:
                    filepath = file
                    break
            else:
                raise ValueError(f"Preset file for '{name}' not found")

        # Delete file
        filepath.unlink()

        logger.info(f"Deleted MCP preset '{name}'")
