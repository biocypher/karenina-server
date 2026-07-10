from pathlib import Path

from fastapi.testclient import TestClient

from karenina_server.server import create_fastapi_app


def _write_dist(root: Path) -> None:
    dist = root / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    (dist / "index.html").write_text("<html><body>Karenina GUI</body></html>")
    (assets / "app.js").write_text("console.log('karenina')")


def test_fastapi_serves_packaged_index_assets_and_spa_routes(tmp_path: Path) -> None:
    webapp = tmp_path / "webapp"
    _write_dist(webapp)

    client = TestClient(create_fastapi_app(webapp))

    root = client.get("/")
    assert root.status_code == 200
    assert "Karenina GUI" in root.text

    asset = client.get("/assets/app.js")
    assert asset.status_code == 200
    assert "karenina" in asset.text

    spa_route = client.get("/benchmark/results")
    assert spa_route.status_code == 200
    assert "Karenina GUI" in spa_route.text
