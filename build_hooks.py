"""Build hook to compile karenina-gui during package build.

This hook runs during ``uv build`` or ``pip wheel`` and:
1. Looks for a sibling ``karenina-gui`` checkout by default.
2. Runs ``npm ci`` (or ``npm install`` when no lockfile exists) and ``npm run build``.
3. Copies the GUI ``dist/`` output to ``src/karenina_server/webapp/dist/``.

Runtime installs must not require Node.js/npm. Release builds should either have
sibling GUI source available or pre-built assets already present in
``src/karenina_server/webapp/dist``.

Environment switches:
- ``KARENINA_SKIP_GUI_BUILD=1``: do not build sibling GUI; require existing assets.
- ``KARENINA_ALLOW_PLACEHOLDER_WEBAPP=1``: create a placeholder webapp if neither
  GUI source nor existing assets are available (local/dev fallback only).
- ``KARENINA_GUI_DIR=/path/to/karenina-gui``: override sibling GUI source path.
"""

from __future__ import annotations

import os
import shutil
import subprocess
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface

_TRUE_VALUES = {"1", "true", "yes", "on"}


def _env_enabled(name: str) -> bool:
    return os.environ.get(name, "").lower() in _TRUE_VALUES


class CustomBuildHook(BuildHookInterface):  # type: ignore[misc]
    """Hatch build hook to bundle karenina-gui with karenina-server."""

    PLUGIN_NAME = "custom"

    def initialize(self, _version: str, _build_data: dict[str, Any]) -> None:
        """Build the GUI before packaging."""
        root = Path(self.root)
        gui_dir = Path(os.environ.get("KARENINA_GUI_DIR", root.parent / "karenina-gui"))
        webapp_dist = root / "src" / "karenina_server" / "webapp" / "dist"

        if _env_enabled("KARENINA_SKIP_GUI_BUILD"):
            if self._has_built_assets(webapp_dist):
                self.app.display_info("KARENINA_SKIP_GUI_BUILD=1: using existing webapp assets")
                return
            raise RuntimeError(
                "KARENINA_SKIP_GUI_BUILD=1 was set, but pre-built webapp assets were not found at "
                f"{webapp_dist}. Run the GUI build first or unset KARENINA_SKIP_GUI_BUILD."
            )

        if not gui_dir.exists():
            if self._has_built_assets(webapp_dist):
                self.app.display_info("GUI source not found, using pre-built webapp assets")
                return
            if _env_enabled("KARENINA_ALLOW_PLACEHOLDER_WEBAPP"):
                self.app.display_warning(
                    f"karenina-gui not found at {gui_dir}. Creating placeholder webapp because "
                    "KARENINA_ALLOW_PLACEHOLDER_WEBAPP=1."
                )
                self._write_placeholder(webapp_dist)
                return
            raise RuntimeError(
                f"karenina-gui not found at {gui_dir} and no pre-built webapp assets exist at {webapp_dist}. "
                "Build with a sibling karenina-gui checkout, set KARENINA_GUI_DIR, or pre-populate assets. "
                "For local placeholder builds only, set KARENINA_ALLOW_PLACEHOLDER_WEBAPP=1."
            )

        self.app.display_info(f"Building karenina-gui from {gui_dir}")

        package_json = gui_dir / "package.json"
        if not package_json.exists():
            raise RuntimeError(f"package.json not found in {gui_dir}")

        install_cmd = ["npm", "ci"] if (gui_dir / "package-lock.json").exists() else ["npm", "install"]
        self._run(install_cmd, cwd=gui_dir, label="Installing npm dependencies")
        self._run(["npm", "run", "build"], cwd=gui_dir, label="Building webapp")

        gui_dist = gui_dir / "dist"
        if not self._has_built_assets(gui_dist):
            raise RuntimeError(f"Build completed but GUI dist assets were not found at {gui_dist}")

        if webapp_dist.exists():
            shutil.rmtree(webapp_dist)
        shutil.copytree(gui_dist, webapp_dist)

        self.app.display_info(f"Webapp built and copied to {webapp_dist}")

    @staticmethod
    def _has_built_assets(path: Path) -> bool:
        return path.is_dir() and (path / "index.html").is_file()

    @staticmethod
    def _write_placeholder(webapp_dist: Path) -> None:
        webapp_dist.mkdir(parents=True, exist_ok=True)
        (webapp_dist / "assets").mkdir(exist_ok=True)
        (webapp_dist / "index.html").write_text(
            "<!DOCTYPE html><html><head><title>Karenina Server</title></head>"
            "<body><h1>Web UI Not Available</h1>"
            "<p>This development build does not include the web UI.</p>"
            "</body></html>"
        )

    def _run(self, cmd: list[str], *, cwd: Path, label: str) -> None:
        self.app.display_info(f"{label}: {' '.join(cmd)}")
        try:
            subprocess.run(cmd, cwd=cwd, check=True, capture_output=True, text=True)
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"{label} failed: {e.stderr}") from e
        except FileNotFoundError as e:
            raise RuntimeError(
                "npm not found. Install Node.js/npm to build webapp assets, or build from a source distribution "
                "that already contains pre-built assets."
            ) from e
