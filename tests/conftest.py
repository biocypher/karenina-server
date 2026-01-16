"""Root conftest.py for karenina-server tests.

Defines pytest markers and shared fixtures used across all test tiers.
"""

import tempfile
from pathlib import Path

import pytest


def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line("markers", "unit: Pure logic tests - no I/O, no external calls")
    config.addinivalue_line("markers", "integration: TestClient, file I/O, threading tests")
    config.addinivalue_line("markers", "e2e: End-to-end workflow tests")
    config.addinivalue_line("markers", "slow: Tests taking > 1 second")
    config.addinivalue_line("markers", "api: API endpoint tests")
    config.addinivalue_line("markers", "service: Service layer tests")
    config.addinivalue_line("markers", "middleware: Middleware tests")


@pytest.fixture
def fixtures_dir() -> Path:
    """Return the root fixtures directory path."""
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def temp_sqlite_db():
    """Shared temp database fixture for tests requiring SQLite."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = f.name
    yield f"sqlite:///{db_path}"
    Path(db_path).unlink(missing_ok=True)
