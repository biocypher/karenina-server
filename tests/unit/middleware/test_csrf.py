"""Unit tests for CSRF token and token store logic.

These are pure logic tests with no I/O, no TestClient, no external calls.
"""

import time

import pytest

from karenina_server.api.auth_handlers import CsrfToken, csrf_store


@pytest.fixture(autouse=True)
def clear_csrf_store():
    """Clear CSRF store before and after each test."""
    csrf_store.clear_all()
    yield
    csrf_store.clear_all()


@pytest.mark.unit
@pytest.mark.middleware
class TestCsrfToken:
    """Test CsrfToken dataclass."""

    def test_is_expired_not_expired(self):
        """Token should not be expired when within TTL."""
        token = CsrfToken(token="test", created_at=time.time(), last_used_at=time.time())
        assert not token.is_expired(3600)  # 1 hour TTL

    def test_is_expired_expired(self):
        """Token should be expired when past TTL."""
        old_time = time.time() - 7200  # 2 hours ago
        token = CsrfToken(token="test", created_at=old_time, last_used_at=old_time)
        assert token.is_expired(3600)  # 1 hour TTL


@pytest.mark.unit
@pytest.mark.middleware
class TestCsrfTokenStore:
    """Test CsrfTokenStore functionality."""

    def test_generate_token(self):
        """Should generate a unique token for each client."""
        token1 = csrf_store.generate_token("client1")
        token2 = csrf_store.generate_token("client2")

        assert token1 != token2
        assert len(token1) > 20  # Should be reasonably long

    def test_generate_token_replaces_old(self):
        """Generating a new token should replace the old one."""
        token1 = csrf_store.generate_token("client1")
        token2 = csrf_store.generate_token("client1")

        assert token1 != token2
        assert not csrf_store.validate_token("client1", token1)
        assert csrf_store.validate_token("client1", token2)

    def test_validate_token_success(self):
        """Should validate a correct token."""
        token = csrf_store.generate_token("client1")
        assert csrf_store.validate_token("client1", token)

    def test_validate_token_wrong_client(self):
        """Should reject a token from a different client."""
        token = csrf_store.generate_token("client1")
        assert not csrf_store.validate_token("client2", token)

    def test_validate_token_wrong_token(self):
        """Should reject an incorrect token."""
        csrf_store.generate_token("client1")
        assert not csrf_store.validate_token("client1", "wrong_token")

    def test_validate_token_expired(self):
        """Should reject an expired token."""
        # Set a very short TTL for testing
        original_ttl = csrf_store._ttl_seconds
        csrf_store.set_ttl(0.1)  # 100ms

        try:
            token = csrf_store.generate_token("client1")
            time.sleep(0.2)  # Wait for expiration
            assert not csrf_store.validate_token("client1", token)
        finally:
            csrf_store.set_ttl(original_ttl)

    def test_invalidate_token(self):
        """Should invalidate a token on request."""
        token = csrf_store.generate_token("client1")
        assert csrf_store.validate_token("client1", token)

        csrf_store.invalidate_token("client1")
        assert not csrf_store.validate_token("client1", token)

    def test_clear_all(self):
        """Should clear all tokens."""
        token1 = csrf_store.generate_token("client1")
        token2 = csrf_store.generate_token("client2")

        csrf_store.clear_all()

        assert not csrf_store.validate_token("client1", token1)
        assert not csrf_store.validate_token("client2", token2)
