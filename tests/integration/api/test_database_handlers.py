"""Integration tests for database handlers API endpoints.

Tests database connection, benchmark management, and verification run endpoints.
"""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def temp_sqlite_db():
    """Create a temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield f"sqlite:///{db_path}"
    Path(db_path).unlink(missing_ok=True)


@pytest.mark.integration
@pytest.mark.api
class TestDatabaseConnectEndpoint:
    """Test database connection endpoint."""

    def test_connect_without_storage_url(self, client):
        """Test connection without storage URL returns validation error."""
        response = client.post("/api/v2/databases/connections", json={})
        assert response.status_code == 422

    def test_connect_with_valid_url(self, client, temp_sqlite_db):
        """Test connection with valid SQLite URL."""
        response = client.post(
            "/api/v2/databases/connections",
            json={"storage_url": temp_sqlite_db, "create_if_missing": True},
        )
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

    def test_connect_with_invalid_url(self, client):
        """Test connection with invalid URL returns error."""
        response = client.post(
            "/api/v2/databases/connections",
            json={"storage_url": "invalid://url"},
        )
        assert response.status_code == 500


@pytest.mark.integration
@pytest.mark.api
class TestListBenchmarksEndpoint:
    """Test benchmark listing endpoint."""

    def test_list_benchmarks_no_storage_url(self, client):
        """Test listing benchmarks without storage URL returns validation error."""
        response = client.get("/api/v2/benchmarks")
        assert response.status_code == 422

    def test_list_benchmarks_with_storage_url(self, client, temp_sqlite_db):
        """Test listing benchmarks with valid storage URL."""
        # First initialize the database
        client.post(
            "/api/v2/databases/connections",
            json={"storage_url": temp_sqlite_db, "create_if_missing": True},
        )

        response = client.get(f"/api/v2/benchmarks?storage_url={temp_sqlite_db}")
        assert response.status_code == 200
        data = response.json()
        assert "benchmarks" in data
        assert isinstance(data["benchmarks"], list)


@pytest.mark.integration
@pytest.mark.api
class TestVerificationRunsEndpoint:
    """Test verification runs endpoint."""

    def test_list_verification_runs_no_storage_url(self, client):
        """Test listing runs without storage URL returns validation error."""
        response = client.get("/api/v2/verification-runs")
        assert response.status_code == 422

    def test_list_verification_runs_with_storage_url(self, client, temp_sqlite_db):
        """Test listing verification runs with valid storage URL."""
        # First initialize the database
        client.post(
            "/api/v2/databases/connections",
            json={"storage_url": temp_sqlite_db, "create_if_missing": True},
        )

        response = client.get(f"/api/v2/verification-runs?storage_url={temp_sqlite_db}")
        assert response.status_code == 200
        data = response.json()
        assert "runs" in data
        assert isinstance(data["runs"], list)


@pytest.mark.integration
@pytest.mark.api
class TestListDatabasesEndpoint:
    """Test database file listing endpoint."""

    def test_list_databases_default_path(self, client, monkeypatch):
        """Test listing databases in default path."""
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create some .db files
            (Path(temp_dir) / "test1.db").touch()
            (Path(temp_dir) / "test2.db").touch()
            (Path(temp_dir) / "notdb.txt").touch()

            # Use DB_PATH which is what the handler reads
            monkeypatch.setenv("DB_PATH", temp_dir)

            response = client.get("/api/v2/databases")
            assert response.status_code == 200
            data = response.json()
            assert "databases" in data
            # Should find at least the .db files we created
            db_names = [db["name"] for db in data["databases"]]
            assert "test1.db" in db_names
            assert "test2.db" in db_names
            assert "notdb.txt" not in db_names
