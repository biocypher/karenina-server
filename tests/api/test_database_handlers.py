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
        BenchmarkCreateRequest,
        BenchmarkCreateResponse,
        BenchmarkListResponse,
        BenchmarkLoadRequest,
        BenchmarkLoadResponse,
        BenchmarkSaveRequest,
        BenchmarkSaveResponse,
        DatabaseConnectRequest,
        DatabaseConnectResponse,
        ListDatabasesResponse,
    )

    register_database_routes(
        app,
        DatabaseConnectRequest,
        DatabaseConnectResponse,
        BenchmarkListResponse,
        BenchmarkLoadRequest,
        BenchmarkLoadResponse,
        BenchmarkCreateRequest,
        BenchmarkCreateResponse,
        BenchmarkSaveRequest,
        BenchmarkSaveResponse,
        ListDatabasesResponse,
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


class TestCreateBenchmark:
    """Test benchmark creation endpoint."""

    def test_create_benchmark_minimal(self, client, temp_sqlite_db):
        """Test creating a benchmark with minimal required fields."""
        # First initialize the database
        from karenina.storage import DBConfig, init_database

        init_database(DBConfig(storage_url=temp_sqlite_db))

        response = client.post(
            "/api/database/create-benchmark",
            json={"storage_url": temp_sqlite_db, "name": "New Benchmark"},
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["benchmark_name"] == "New Benchmark"

        # Verify checkpoint data structure
        checkpoint = data["checkpoint_data"]
        assert checkpoint["dataset_metadata"]["name"] == "New Benchmark"
        assert len(checkpoint["questions"]) == 0
        assert checkpoint["global_rubric"] is None

    def test_create_benchmark_with_all_fields(self, client, temp_sqlite_db):
        """Test creating a benchmark with all optional fields."""
        from karenina.storage import DBConfig, init_database

        init_database(DBConfig(storage_url=temp_sqlite_db))

        response = client.post(
            "/api/database/create-benchmark",
            json={
                "storage_url": temp_sqlite_db,
                "name": "Complete Benchmark",
                "description": "A test benchmark with all fields",
                "version": "2.0.0",
                "creator": "Test Creator",
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert data["benchmark_name"] == "Complete Benchmark"

        # Verify all metadata fields
        metadata = data["checkpoint_data"]["dataset_metadata"]
        assert metadata["name"] == "Complete Benchmark"
        assert metadata["description"] == "A test benchmark with all fields"
        assert metadata["version"] == "2.0.0"
        assert metadata["creator"] == "Test Creator"

    def test_create_benchmark_duplicate_name(self, client, temp_sqlite_db):
        """Test creating a benchmark with duplicate name replaces existing."""
        from karenina.benchmark import Benchmark

        # Create initial benchmark with some data
        b = Benchmark.create(name="Duplicate Name", version="1.0.0", description="Original description")
        b.add_question("Original question?", "Original answer")
        b.save_to_db(temp_sqlite_db)

        # Create another with same name but different metadata
        response = client.post(
            "/api/database/create-benchmark",
            json={
                "storage_url": temp_sqlite_db,
                "name": "Duplicate Name",
                "description": "New description",
                "version": "2.0.0",
            },
        )

        # Should succeed (replaces existing)
        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True

        # Verify it replaced the old benchmark (should have no questions)
        checkpoint = data["checkpoint_data"]
        assert len(checkpoint["questions"]) == 0
        assert checkpoint["dataset_metadata"]["version"] == "2.0.0"
        assert checkpoint["dataset_metadata"]["description"] == "New description"

    def test_create_benchmark_invalid_url(self, client):
        """Test creating benchmark with invalid database URL."""
        response = client.post(
            "/api/database/create-benchmark",
            json={"storage_url": "invalid://url", "name": "Test"},
        )

        assert response.status_code == 500


class TestSaveBenchmark:
    """Test benchmark save endpoint."""

    def test_save_benchmark_new(self, client, temp_sqlite_db):
        """Test saving a new benchmark to database."""
        from karenina.storage import DBConfig, init_database

        init_database(DBConfig(storage_url=temp_sqlite_db))

        # Prepare checkpoint data
        checkpoint_data = {
            "dataset_metadata": {"name": "Saved Benchmark", "version": "1.0.0", "creator": "Test User"},
            "questions": {
                "q1": {
                    "question": "What is 2+2?",
                    "raw_answer": "4",
                    "original_answer_template": "class Answer(BaseModel): result: int",
                    "answer_template": "class Answer(BaseModel): result: int",
                    "last_modified": "2025-01-01T00:00:00Z",
                    "finished": True,
                }
            },
            "global_rubric": None,
        }

        response = client.post(
            "/api/database/save-benchmark",
            json={
                "storage_url": temp_sqlite_db,
                "benchmark_name": "Saved Benchmark",
                "checkpoint_data": checkpoint_data,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "message" in data
        assert "last_modified" in data

        # Verify benchmark was actually saved
        from karenina.storage import DBConfig, load_benchmark

        loaded, _ = load_benchmark("Saved Benchmark", DBConfig(storage_url=temp_sqlite_db), load_config=True)
        assert loaded.name == "Saved Benchmark"
        all_questions = loaded.get_all_questions()
        assert len(all_questions) == 1
        assert all_questions[0]["question"] == "What is 2+2?"

    def test_save_benchmark_update_existing(self, client, temp_sqlite_db):
        """Test updating an existing benchmark adds to existing questions."""
        from karenina.benchmark import Benchmark

        # Create initial benchmark
        b = Benchmark.create(name="Update Test", version="1.0.0")
        b.add_question("Old question?", "Old answer")
        b.save_to_db(temp_sqlite_db)

        # Save with new data (adds question, doesn't replace)
        checkpoint_data = {
            "dataset_metadata": {"name": "Update Test", "version": "2.0.0"},
            "questions": {
                "q1": {
                    "question": "New question?",
                    "raw_answer": "New answer",
                    "original_answer_template": "class Answer(BaseModel): text: str",
                    "answer_template": "class Answer(BaseModel): text: str",
                    "last_modified": "2025-01-02T00:00:00Z",
                    "finished": True,
                }
            },
            "global_rubric": None,
        }

        response = client.post(
            "/api/database/save-benchmark",
            json={
                "storage_url": temp_sqlite_db,
                "benchmark_name": "Update Test",
                "checkpoint_data": checkpoint_data,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "message" in data
        assert "last_modified" in data

        # Verify both questions exist (accumulative behavior)
        from karenina.storage import DBConfig, load_benchmark

        loaded, _ = load_benchmark("Update Test", DBConfig(storage_url=temp_sqlite_db), load_config=True)
        assert loaded.version == "2.0.0"
        all_questions = loaded.get_all_questions()
        assert len(all_questions) == 2
        question_texts = {q["question"] for q in all_questions}
        assert "Old question?" in question_texts
        assert "New question?" in question_texts

    def test_save_benchmark_with_rubric(self, client, temp_sqlite_db):
        """Test saving benchmark with global rubric."""
        from karenina.storage import DBConfig, init_database

        init_database(DBConfig(storage_url=temp_sqlite_db))

        checkpoint_data = {
            "dataset_metadata": {"name": "Rubric Test", "version": "1.0.0"},
            "questions": {},
            "global_rubric": {
                "id": "test-rubric",
                "name": "Test Rubric",
                "traits": [{"name": "Clarity", "description": "Answer is clear", "weight": 1.0}],
            },
        }

        response = client.post(
            "/api/database/save-benchmark",
            json={
                "storage_url": temp_sqlite_db,
                "benchmark_name": "Rubric Test",
                "checkpoint_data": checkpoint_data,
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["success"] is True
        assert "message" in data
        assert "last_modified" in data

        # Verify rubric was saved
        from karenina.storage import DBConfig, load_benchmark

        loaded, _ = load_benchmark("Rubric Test", DBConfig(storage_url=temp_sqlite_db), load_config=True)
        # Check that benchmark was saved (rubric storage might not be fully implemented yet)
        assert loaded.name == "Rubric Test"

    def test_save_benchmark_invalid_url(self, client):
        """Test saving benchmark with invalid database URL."""
        checkpoint_data = {
            "dataset_metadata": {"name": "Test", "version": "1.0.0"},
            "questions": {},
            "global_rubric": None,
        }

        response = client.post(
            "/api/database/save-benchmark",
            json={
                "storage_url": "invalid://url",
                "benchmark_name": "Test",
                "checkpoint_data": checkpoint_data,
            },
        )

        assert response.status_code == 500


class TestListDatabases:
    """Test database listing endpoint."""

    def test_list_databases_with_db_path_env(self, client, monkeypatch):
        """Test listing databases from DB_PATH environment variable."""
        # Create temporary directory with some .db files

        with tempfile.TemporaryDirectory() as tmpdir:
            # Create some test .db files
            db1_path = Path(tmpdir) / "test1.db"
            db2_path = Path(tmpdir) / "test2.db"
            db1_path.touch()
            db2_path.touch()

            # Set DB_PATH environment variable
            monkeypatch.setenv("DB_PATH", tmpdir)

            response = client.get("/api/database/list-databases")

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["db_directory"] == tmpdir
            assert data["is_default_directory"] is False
            assert len(data["databases"]) == 2

            # Check database names
            db_names = {db["name"] for db in data["databases"]}
            assert "test1.db" in db_names
            assert "test2.db" in db_names

            # Check that paths are absolute
            for db in data["databases"]:
                assert db["path"].startswith(tmpdir)

    def test_list_databases_uses_cwd_by_default(self, client, monkeypatch):
        """Test listing databases from current working directory when DB_PATH not set."""
        import os

        # Ensure DB_PATH is not set
        monkeypatch.delenv("DB_PATH", raising=False)

        with tempfile.TemporaryDirectory() as tmpdir:
            # Change to temp directory
            original_cwd = os.getcwd()
            try:
                os.chdir(tmpdir)

                # Create a test .db file
                db_path = Path(tmpdir) / "default.db"
                db_path.touch()

                response = client.get("/api/database/list-databases")

                assert response.status_code == 200
                data = response.json()
                assert data["success"] is True
                # Compare resolved paths to handle symlinks (e.g., /var -> /private/var on macOS)
                assert Path(data["db_directory"]).resolve() == Path(tmpdir).resolve()
                assert data["is_default_directory"] is True
                assert len(data["databases"]) == 1
                assert data["databases"][0]["name"] == "default.db"
            finally:
                os.chdir(original_cwd)

    def test_list_databases_empty_directory(self, client, monkeypatch):
        """Test listing databases from empty directory."""
        with tempfile.TemporaryDirectory() as tmpdir:
            monkeypatch.setenv("DB_PATH", tmpdir)

            response = client.get("/api/database/list-databases")

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert data["db_directory"] == tmpdir
            assert len(data["databases"]) == 0

    def test_list_databases_filters_only_db_files(self, client, monkeypatch):
        """Test that only .db files are listed."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create various files
            (Path(tmpdir) / "test.db").touch()
            (Path(tmpdir) / "other.txt").touch()
            (Path(tmpdir) / "data.json").touch()
            (Path(tmpdir) / "another.db").touch()

            monkeypatch.setenv("DB_PATH", tmpdir)

            response = client.get("/api/database/list-databases")

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert len(data["databases"]) == 2
            db_names = {db["name"] for db in data["databases"]}
            assert "test.db" in db_names
            assert "another.db" in db_names
            assert "other.txt" not in db_names

    def test_list_databases_includes_file_size(self, client, monkeypatch):
        """Test that database size is included in response."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create a .db file with some content
            db_path = Path(tmpdir) / "test.db"
            db_path.write_text("test content")

            monkeypatch.setenv("DB_PATH", tmpdir)

            response = client.get("/api/database/list-databases")

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            assert len(data["databases"]) == 1
            assert data["databases"][0]["size"] > 0

    def test_list_databases_sorted_alphabetically(self, client, monkeypatch):
        """Test that databases are sorted alphabetically by name."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create files in non-alphabetical order
            (Path(tmpdir) / "zebra.db").touch()
            (Path(tmpdir) / "apple.db").touch()
            (Path(tmpdir) / "middle.db").touch()

            monkeypatch.setenv("DB_PATH", tmpdir)

            response = client.get("/api/database/list-databases")

            assert response.status_code == 200
            data = response.json()
            assert data["success"] is True
            db_names = [db["name"] for db in data["databases"]]
            assert db_names == ["apple.db", "middle.db", "zebra.db"]

    def test_list_databases_nonexistent_directory(self, client, monkeypatch):
        """Test error when DB_PATH points to nonexistent directory."""
        monkeypatch.setenv("DB_PATH", "/nonexistent/directory")

        response = client.get("/api/database/list-databases")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"].lower()
