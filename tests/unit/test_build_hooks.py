"""Tests for the webapp asset policy in the Hatch build hook.

The hook must keep source installs working (editable checkouts, ``pip install
git+...``) by falling back to a placeholder webapp, while guaranteeing that a
release build never ships that placeholder.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from build_hooks import PLACEHOLDER_MARKER, CustomBuildHook


class _RecordingApp:
    """Stand-in for the Hatch application object used for build output."""

    def __init__(self) -> None:
        self.info: list[str] = []
        self.warnings: list[str] = []

    def display_info(self, message: str) -> None:
        self.info.append(message)

    def display_warning(self, message: str) -> None:
        self.warnings.append(message)


class _Hook(CustomBuildHook):
    """CustomBuildHook with the Hatch plumbing replaced by plain attributes."""

    def __init__(self, root: Path) -> None:  # noqa: D107 - bypasses BuildHookInterface.__init__
        self._test_root = root
        self._test_app = _RecordingApp()

    @property
    def root(self) -> str:
        return str(self._test_root)

    @property
    def app(self) -> _RecordingApp:
        return self._test_app


@pytest.fixture
def clean_env(monkeypatch: pytest.MonkeyPatch) -> None:
    """Remove the hook's environment switches so each test sets its own."""
    for name in (
        "KARENINA_RELEASE_BUILD",
        "KARENINA_SKIP_GUI_BUILD",
        "KARENINA_GUI_DIR",
    ):
        monkeypatch.delenv(name, raising=False)


def _webapp_dist(root: Path) -> Path:
    return root / "src" / "karenina_server" / "webapp" / "dist"


def _write_real_assets(root: Path) -> Path:
    dist = _webapp_dist(root)
    (dist / "assets").mkdir(parents=True)
    (dist / "index.html").write_text("<html><body>Karenina GUI</body></html>")
    (dist / "assets" / "app.js").write_text("console.log('karenina')")
    return dist


def _run(hook: CustomBuildHook) -> None:
    hook.initialize("standard", {})


def _missing_gui(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    """Point the GUI lookup at a path that does not exist."""
    monkeypatch.setenv("KARENINA_GUI_DIR", str(tmp_path / "no-such-karenina-gui"))


def test_source_build_without_gui_falls_back_to_placeholder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_env: None
) -> None:
    """A plain source build with no GUI must succeed, not fail."""
    _missing_gui(monkeypatch, tmp_path)
    hook = _Hook(tmp_path)

    _run(hook)

    dist = _webapp_dist(tmp_path)
    assert (dist / "index.html").is_file()
    assert (dist / PLACEHOLDER_MARKER).is_file()
    assert any("placeholder webapp" in warning for warning in hook.app.warnings)


def test_release_build_without_gui_or_assets_fails(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_env: None
) -> None:
    _missing_gui(monkeypatch, tmp_path)
    monkeypatch.setenv("KARENINA_RELEASE_BUILD", "1")
    hook = _Hook(tmp_path)

    with pytest.raises(RuntimeError, match="no pre-built webapp assets exist"):
        _run(hook)

    assert not _webapp_dist(tmp_path).exists()


def test_release_build_refuses_stale_placeholder_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_env: None
) -> None:
    """A placeholder left by an earlier dev build must not pass as real assets."""
    _missing_gui(monkeypatch, tmp_path)
    CustomBuildHook._write_placeholder(_webapp_dist(tmp_path))
    monkeypatch.setenv("KARENINA_RELEASE_BUILD", "1")
    hook = _Hook(tmp_path)

    with pytest.raises(RuntimeError, match="are a placeholder"):
        _run(hook)


def test_release_build_accepts_real_prebuilt_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_env: None
) -> None:
    _missing_gui(monkeypatch, tmp_path)
    dist = _write_real_assets(tmp_path)
    monkeypatch.setenv("KARENINA_RELEASE_BUILD", "1")
    hook = _Hook(tmp_path)

    _run(hook)

    assert (dist / "index.html").read_text() == "<html><body>Karenina GUI</body></html>"
    assert hook.app.warnings == []


def test_source_build_reuses_existing_placeholder_without_error(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_env: None
) -> None:
    _missing_gui(monkeypatch, tmp_path)
    CustomBuildHook._write_placeholder(_webapp_dist(tmp_path))
    hook = _Hook(tmp_path)

    _run(hook)

    assert (_webapp_dist(tmp_path) / PLACEHOLDER_MARKER).is_file()


def test_skip_gui_build_requires_existing_assets(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_env: None
) -> None:
    monkeypatch.setenv("KARENINA_SKIP_GUI_BUILD", "1")
    hook = _Hook(tmp_path)

    with pytest.raises(RuntimeError, match="pre-built webapp assets were not found"):
        _run(hook)


def test_skip_gui_build_under_release_refuses_placeholder(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_env: None
) -> None:
    CustomBuildHook._write_placeholder(_webapp_dist(tmp_path))
    monkeypatch.setenv("KARENINA_SKIP_GUI_BUILD", "1")
    monkeypatch.setenv("KARENINA_RELEASE_BUILD", "1")
    hook = _Hook(tmp_path)

    with pytest.raises(RuntimeError, match="are a placeholder"):
        _run(hook)


def test_gui_build_replaces_stale_placeholder_marker(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_env: None
) -> None:
    """Building from GUI source must not leave the placeholder marker behind."""
    gui_dir = tmp_path / "karenina-gui"
    gui_dist = gui_dir / "dist"
    gui_dist.mkdir(parents=True)
    (gui_dir / "package.json").write_text("{}")
    (gui_dist / "index.html").write_text("<html><body>Real GUI</body></html>")

    CustomBuildHook._write_placeholder(_webapp_dist(tmp_path))
    monkeypatch.setenv("KARENINA_GUI_DIR", str(gui_dir))
    monkeypatch.setenv("KARENINA_RELEASE_BUILD", "1")

    hook = _Hook(tmp_path)
    commands: list[list[str]] = []

    def _fake_run(cmd: list[str], **_kwargs: Any) -> None:
        commands.append(cmd)

    monkeypatch.setattr(_Hook, "_run", lambda _self, cmd, **kwargs: _fake_run(cmd, **kwargs))

    _run(hook)

    dist = _webapp_dist(tmp_path)
    assert (dist / "index.html").read_text() == "<html><body>Real GUI</body></html>"
    assert not (dist / PLACEHOLDER_MARKER).exists()
    assert commands == [["npm", "install"], ["npm", "run", "build"]]


def test_gui_build_prefers_npm_ci_when_lockfile_present(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, clean_env: None
) -> None:
    gui_dir = tmp_path / "karenina-gui"
    gui_dist = gui_dir / "dist"
    gui_dist.mkdir(parents=True)
    (gui_dir / "package.json").write_text("{}")
    (gui_dir / "package-lock.json").write_text("{}")
    (gui_dist / "index.html").write_text("<html></html>")

    monkeypatch.setenv("KARENINA_GUI_DIR", str(gui_dir))
    hook = _Hook(tmp_path)
    commands: list[list[str]] = []
    monkeypatch.setattr(_Hook, "_run", lambda _self, cmd, **_kwargs: commands.append(cmd))

    _run(hook)

    assert commands[0] == ["npm", "ci"]
