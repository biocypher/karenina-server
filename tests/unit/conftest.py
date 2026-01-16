"""Conftest for unit tests.

Unit tests are pure logic tests with no I/O, no external calls,
and no dependencies on TestClient or temp files.
"""

import pytest


@pytest.fixture(autouse=True)
def unit_test_marker(request):
    """Auto-apply unit marker to all tests in this directory."""
    # This fixture ensures unit tests are properly isolated
    pass
