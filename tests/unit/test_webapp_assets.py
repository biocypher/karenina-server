from pathlib import Path

import pytest

from karenina_server.server import build_webapp, find_webapp_directory


def test_build_webapp_uses_prebuilt_dist_without_package_json(tmp_path: Path) -> None:
    webapp = tmp_path / "webapp"
    dist = webapp / "dist"
    dist.mkdir(parents=True)
    (dist / "index.html").write_text("<html></html>")

    assert build_webapp(webapp) == dist


def test_build_webapp_accepts_dist_directory(tmp_path: Path) -> None:
    dist = tmp_path / "webapp" / "dist"
    dist.mkdir(parents=True)
    (dist / "index.html").write_text("<html></html>")

    assert build_webapp(dist) == dist


def test_build_webapp_errors_when_assets_and_source_are_missing(tmp_path: Path) -> None:
    webapp = tmp_path / "webapp"
    webapp.mkdir()

    with pytest.raises(FileNotFoundError, match="Webapp assets not found"):
        build_webapp(webapp)


def test_find_webapp_directory_accepts_dist_override(tmp_path: Path) -> None:
    dist = tmp_path / "webapp" / "dist"
    dist.mkdir(parents=True)
    (dist / "index.html").write_text("<html></html>")

    assert find_webapp_directory(str(dist)) == dist.parent.resolve()
