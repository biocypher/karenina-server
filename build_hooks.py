"""Build hook to compile karenina-gui during package build.

This hook runs during `uv build` or `pip wheel` and:
1. Checks if karenina-gui source is available
2. Runs npm install and npm run build
3. Copies the dist/ output to karenina_server/webapp/dist/

If karenina-gui is not available but webapp/dist already exists (e.g., from sdist),
the hook skips the build step.
"""

import shutil
import subprocess
from pathlib import Path
from typing import Any

from hatchling.builders.hooks.plugin.interface import BuildHookInterface


class CustomBuildHook(BuildHookInterface):  # type: ignore[misc]
    """Hatch build hook to bundle karenina-gui with karenina-server."""

    PLUGIN_NAME = "custom"

    def initialize(self, _version: str, _build_data: dict[str, Any]) -> None:
        """Build the GUI before packaging.

        Args:
            _version: The version being built (unused, required by interface)
            _build_data: Build configuration data (unused, required by interface)
        """
        root = Path(self.root)
        gui_dir = root.parent / "karenina-gui"
        webapp_dist = root / "src" / "karenina_server" / "webapp" / "dist"

        # Skip if GUI source not available (e.g., installing from sdist)
        if not gui_dir.exists():
            if webapp_dist.exists():
                self.app.display_info("GUI source not found, using pre-built webapp assets")
                return
            # Create a minimal placeholder for dev/CI installs without GUI
            self.app.display_warning(
                f"karenina-gui not found at {gui_dir}. Creating placeholder webapp. "
                "The web UI will not be available. Install from PyPI for the full webapp."
            )
            webapp_dist.mkdir(parents=True, exist_ok=True)
            (webapp_dist / "index.html").write_text(
                "<!DOCTYPE html><html><head><title>Karenina Server</title></head>"
                "<body><h1>Web UI Not Available</h1>"
                "<p>This installation does not include the web UI. "
                "Install from PyPI or build with karenina-gui available.</p>"
                "</body></html>"
            )
            return

        self.app.display_info(f"Building karenina-gui from {gui_dir}")

        # Check for package.json
        package_json = gui_dir / "package.json"
        if not package_json.exists():
            raise RuntimeError(f"package.json not found in {gui_dir}")

        # Install npm dependencies if needed
        node_modules = gui_dir / "node_modules"
        if not node_modules.exists():
            self.app.display_info("Installing npm dependencies...")
            try:
                subprocess.run(
                    ["npm", "install"],
                    cwd=gui_dir,
                    check=True,
                    capture_output=True,
                    text=True,
                )
            except subprocess.CalledProcessError as e:
                raise RuntimeError(f"npm install failed: {e.stderr}") from e
            except FileNotFoundError as e:
                raise RuntimeError(
                    "npm not found. Please install Node.js and npm to build the webapp. Visit: https://nodejs.org/"
                ) from e

        # Build the GUI
        self.app.display_info("Building webapp...")
        try:
            subprocess.run(
                ["npm", "run", "build"],
                cwd=gui_dir,
                check=True,
                capture_output=True,
                text=True,
            )
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"npm run build failed: {e.stderr}") from e

        # Copy to webapp directory
        gui_dist = gui_dir / "dist"
        if not gui_dist.exists():
            raise RuntimeError("Build completed but dist directory not found")

        # Remove existing dist and copy new one
        if webapp_dist.exists():
            shutil.rmtree(webapp_dist)
        shutil.copytree(gui_dist, webapp_dist)

        self.app.display_info(f"Webapp built and copied to {webapp_dist}")
