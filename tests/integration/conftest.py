"""Conftest for integration tests.

Integration tests use TestClient, temp files, threading, or database operations.
"""

import tempfile
from pathlib import Path

import pytest
from fastapi.testclient import TestClient


@pytest.fixture
def webapp_dir() -> Path:
    """Return path to the webapp directory."""
    return Path(__file__).parent.parent.parent / "src" / "karenina_server" / "webapp"


@pytest.fixture
def app(webapp_dir):
    """Create FastAPI app for testing."""
    from karenina_server.server import create_fastapi_app

    return create_fastapi_app(webapp_dir)


@pytest.fixture
def client(app):
    """Create TestClient for API testing."""
    return TestClient(app)


@pytest.fixture
def temp_env_file():
    """Shared temporary .env file fixture."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".env", delete=False) as f:
        f.write("OPENAI_API_KEY=sk-test123\n")
        f.write("TEST_VAR=test_value\n")
        temp_path = Path(f.name)
    yield temp_path
    if temp_path.exists():
        temp_path.unlink()


@pytest.fixture
def temp_presets_dir():
    """Shared temporary presets directory fixture."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield Path(temp_dir)


@pytest.fixture
def temp_defaults_file():
    """Shared temporary defaults file fixture."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        import json

        json.dump({"test_default": "value"}, f)
        temp_path = Path(f.name)
    yield temp_path
    if temp_path.exists():
        temp_path.unlink()
