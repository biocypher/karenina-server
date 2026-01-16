"""Conftest for end-to-end tests.

E2E tests exercise full workflows including database operations,
file I/O, and multi-step operations.
"""

import tempfile
from pathlib import Path

import pytest


@pytest.fixture
def e2e_fixtures_dir():
    """Return path to E2E fixtures."""
    return Path(__file__).parent.parent / "fixtures"


@pytest.fixture
def temp_db_path():
    """Create a temporary database path for E2E tests."""
    with tempfile.NamedTemporaryFile(suffix=".db", delete=False) as f:
        db_path = Path(f.name)
    yield db_path
    if db_path.exists():
        db_path.unlink()


@pytest.fixture
def e2e_fixture_mode(monkeypatch):
    """Enable E2E fixture mode for deterministic LLM responses."""
    monkeypatch.setenv("KARENINA_E2E_MODE", "true")
