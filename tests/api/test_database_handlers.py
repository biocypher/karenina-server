"""Tests for database API handlers."""

import tempfile
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient


@pytest.fixture
def temp_sqlite_db():
    """Create a temporary SQLite database for testing."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name

    yield f"sqlite:///{db_path}"

    # Cleanup
    Path(db_path).unlink(missing_ok=True)


@pytest.fixture
def mock_app():
    """Create a mock FastAPI app with database routes."""
    app = FastAPI()

    # Import and register database routes
    from karenina_server.api.database_handlers import register_database_routes
    from karenina_server.server import (
        BenchmarkListResponse,
        BenchmarkLoadRequest,
        BenchmarkLoadResponse,
        DatabaseConnectRequest,
        DatabaseConnectResponse,
    )

    register_database_routes(
        app,
        DatabaseConnectRequest,
        DatabaseConnectResponse,
        BenchmarkListResponse,
        BenchmarkLoadRequest,
        BenchmarkLoadResponse,
    )

    return app


@pytest.fixture
def client(mock_app):
    """Create a test client."""
    return TestClient(mock_app)


class TestDatabaseConnect:
    """Test database connection endpoint."""

    def test_connect_to_new_database(self, client, temp_sqlite_db):
        """Test connecting to and creating a new database."""
        response = client.post("/api/database/connect", json={"storage_url": temp_sqlite_db, "create_if_missing": True})

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["storage_url"] == temp_sqlite_db
        assert data["benchmark_count"] == 0

    def test_connect_to_existing_database_with_benchmarks(self, client, temp_sqlite_db):
        """Test connecting to database with existing benchmarks."""
        # First, create database and add a benchmark
        from karenina.benchmark import Benchmark

        b = Benchmark.create(name="Test Benchmark", version="1.0.0")
        b.add_question("Question?", "Answer")
        b.save_to_db(temp_sqlite_db)

        # Now connect
        response = client.post("/api/database/connect", json={"storage_url": temp_sqlite_db, "create_if_missing": True})

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["benchmark_count"] == 1

    def test_connect_with_invalid_url(self, client):
        """Test connecting with invalid database URL."""
        response = client.post(
            "/api/database/connect", json={"storage_url": "invalid://url", "create_if_missing": True}
        )

        assert response.status_code == 500
        assert "error" in response.json()["detail"].lower() or "detail" in response.json()


class TestListBenchmarks:
    """Test benchmark listing endpoint."""

    def test_list_benchmarks_empty_database(self, client, temp_sqlite_db):
        """Test listing benchmarks from empty database."""
        # Initialize empty database
        from karenina.storage import DBConfig, init_database

        init_database(DBConfig(storage_url=temp_sqlite_db))

        response = client.get(f"/api/database/benchmarks?storage_url={temp_sqlite_db}")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["count"] == 0
        assert len(data["benchmarks"]) == 0

    def test_list_benchmarks_with_data(self, client, temp_sqlite_db):
        """Test listing benchmarks from database with data."""
        # Create benchmarks
        from karenina.benchmark import Benchmark

        b1 = Benchmark.create(name="Benchmark 1", version="1.0.0")
        b1.add_question("Q1?", "A1")
        b1.add_question("Q2?", "A2")
        b1.save_to_db(temp_sqlite_db)

        b2 = Benchmark.create(name="Benchmark 2", version="2.0.0")
        b2.add_question("Q3?", "A3")
        b2.save_to_db(temp_sqlite_db)

        response = client.get(f"/api/database/benchmarks?storage_url={temp_sqlite_db}")

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["count"] == 2
        assert len(data["benchmarks"]) == 2

        # Check benchmark details
        benchmark_names = {b["name"] for b in data["benchmarks"]}
        assert "Benchmark 1" in benchmark_names
        assert "Benchmark 2" in benchmark_names

        # Find Benchmark 1 and check question count
        b1_data = next(b for b in data["benchmarks"] if b["name"] == "Benchmark 1")
        assert b1_data["total_questions"] == 2

    def test_list_benchmarks_invalid_url(self, client):
        """Test listing benchmarks with invalid URL."""
        response = client.get("/api/database/benchmarks?storage_url=invalid://url")

        assert response.status_code == 500


class TestLoadBenchmark:
    """Test benchmark loading endpoint."""

    def test_load_benchmark_success(self, client, temp_sqlite_db):
        """Test successfully loading a benchmark."""
        # Create and save benchmark
        from karenina.benchmark import Benchmark

        b = Benchmark.create(name="Test Load", description="Test description", version="1.0.0", creator="Test User")
        b.add_question("Question 1?", "Answer 1")
        b.add_question("Question 2?", "Answer 2")
        b.save_to_db(temp_sqlite_db)

        response = client.post(
            "/api/database/load-benchmark", json={"storage_url": temp_sqlite_db, "benchmark_name": "Test Load"}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["benchmark_name"] == "Test Load"
        assert data["storage_url"] == temp_sqlite_db

        # Check checkpoint data structure
        checkpoint = data["checkpoint_data"]
        assert checkpoint["dataset_metadata"]["name"] == "Test Load"
        assert checkpoint["dataset_metadata"]["description"] == "Test description"
        assert checkpoint["dataset_metadata"]["version"] == "1.0.0"
        assert checkpoint["dataset_metadata"]["creator"] == "Test User"

        # Check questions
        assert len(checkpoint["questions"]) == 2

    def test_load_nonexistent_benchmark(self, client, temp_sqlite_db):
        """Test loading a benchmark that doesn't exist."""
        from karenina.storage import DBConfig, init_database

        init_database(DBConfig(storage_url=temp_sqlite_db))

        response = client.post(
            "/api/database/load-benchmark",
            json={"storage_url": temp_sqlite_db, "benchmark_name": "Nonexistent"},
        )

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()

    def test_load_benchmark_invalid_url(self, client):
        """Test loading benchmark with invalid URL."""
        response = client.post(
            "/api/database/load-benchmark", json={"storage_url": "invalid://url", "benchmark_name": "Test"}
        )

        assert response.status_code == 500


class TestInitDatabase:
    """Test database initialization endpoint."""

    def test_init_new_database(self, client):
        """Test initializing a new database."""
        with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
            db_path = f.name

        storage_url = f"sqlite:///{db_path}"

        try:
            response = client.post("/api/database/init", json={"storage_url": storage_url})

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["storage_url"] == storage_url

            # Verify database was created with tables
            from karenina.storage import DBConfig, get_engine
            from sqlalchemy import inspect

            engine = get_engine(DBConfig(storage_url=storage_url))
            inspector = inspect(engine)
            tables = inspector.get_table_names()

            expected_tables = [
                "benchmarks",
                "questions",
                "benchmark_questions",
                "verification_runs",
                "verification_results",
            ]
            for table in expected_tables:
                assert table in tables

        finally:
            Path(db_path).unlink(missing_ok=True)

    def test_init_database_invalid_url(self, client):
        """Test initializing database with invalid URL."""
        response = client.post("/api/database/init", json={"storage_url": "invalid://url"})

        assert response.status_code == 500
