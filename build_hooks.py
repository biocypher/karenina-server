"""Build hook to compile karenina-gui during package build.

This hook runs during ``uv build`` or ``pip wheel`` and:
1. Looks for a sibling ``karenina-gui`` checkout by default.
2. Runs ``npm ci`` (or ``npm install`` when no lockfile exists) and ``npm run build``.
3. Copies the GUI ``dist/`` output to ``src/karenina_server/webapp/dist/``.

Runtime installs must not require Node.js/npm, so a wheel is expected to carry
built assets. When no GUI source and no pre-built assets are available the hook
falls back to a placeholder webapp, which keeps source installs working (editable
checkouts, ``pip install git+...``) at the cost of shipping no UI. Release builds
set ``KARENINA_RELEASE_BUILD=1`` to turn that fallback into a hard error, so a
published artifact can never silently contain the placeholder.

Environment switches:
- ``KARENINA_RELEASE_BUILD=1``: refuse to build without real GUI assets, and
  refuse to reuse a previously written placeholder.
- ``KARENINA_SKIP_GUI_BUILD=1``: do not build sibling GUI; require existing assets.
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

# Written alongside the placeholder index.html so later builds can tell a
# placeholder apart from a real GUI build left over in the working tree.
PLACEHOLDER_MARKER = ".karenina-placeholder"

_RELEASE_HINT = (
    "Build with a sibling karenina-gui checkout, set KARENINA_GUI_DIR, or pre-populate "
    "the assets. Unset KARENINA_RELEASE_BUILD for a placeholder build without a UI."
)


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
        release_build = _env_enabled("KARENINA_RELEASE_BUILD")

        if _env_enabled("KARENINA_SKIP_GUI_BUILD"):
            if not self._has_built_assets(webapp_dist):
                raise RuntimeError(
                    "KARENINA_SKIP_GUI_BUILD=1 was set, but pre-built webapp assets were not found at "
                    f"{webapp_dist}. Run the GUI build first or unset KARENINA_SKIP_GUI_BUILD."
                )
            self._reject_placeholder_for_release(webapp_dist, release_build)
            self.app.display_info("KARENINA_SKIP_GUI_BUILD=1: using existing webapp assets")
            return

        if not gui_dir.exists():
            if self._has_built_assets(webapp_dist):
                self._reject_placeholder_for_release(webapp_dist, release_build)
                self.app.display_info("GUI source not found, using pre-built webapp assets")
                return
            if release_build:
                raise RuntimeError(
                    f"KARENINA_RELEASE_BUILD=1 was set, but karenina-gui was not found at {gui_dir} "
                    f"and no pre-built webapp assets exist at {webapp_dist}. {_RELEASE_HINT}"
                )
            self.app.display_warning(
                f"karenina-gui not found at {gui_dir}. Building a placeholder webapp: this install "
                "will serve the API but no web UI. Set KARENINA_RELEASE_BUILD=1 to make this an error."
            )
            self._write_placeholder(webapp_dist)
            return

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

    @classmethod
    def _reject_placeholder_for_release(cls, webapp_dist: Path, release_build: bool) -> None:
        """Stop a release build from shipping placeholder assets left by an earlier build."""
        if release_build and cls._is_placeholder(webapp_dist):
            raise RuntimeError(
                f"KARENINA_RELEASE_BUILD=1 was set, but the assets at {webapp_dist} are a placeholder "
                f"webapp from an earlier build, not a real GUI build. {_RELEASE_HINT}"
            )

    @staticmethod
    def _has_built_assets(path: Path) -> bool:
        return path.is_dir() and (path / "index.html").is_file()

    @staticmethod
    def _is_placeholder(webapp_dist: Path) -> bool:
        return (webapp_dist / PLACEHOLDER_MARKER).is_file()

    @staticmethod
    def _write_placeholder(webapp_dist: Path) -> None:
        webapp_dist.mkdir(parents=True, exist_ok=True)
        (webapp_dist / "assets").mkdir(exist_ok=True)
        (webapp_dist / PLACEHOLDER_MARKER).write_text(
            "This webapp/dist was generated as a placeholder because no karenina-gui\n"
            "source or pre-built assets were available at build time.\n"
        )
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
