"""Tests for CSRF protection (auth handlers and middleware)."""

import time
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi import FastAPI, HTTPException
from fastapi.testclient import TestClient

from karenina_server.api.auth_handlers import (
    CsrfToken,
    csrf_store,
    get_client_id,
)
from karenina_server.api.auth_handlers import (
    router as auth_router,
)
from karenina_server.middleware.csrf_middleware import CsrfMiddleware


@pytest.fixture(autouse=True)
def clear_csrf_store():
    """Clear CSRF store before and after each test."""
    csrf_store.clear_all()
    yield
    csrf_store.clear_all()


@pytest.fixture
def app_with_csrf():
    """Create a test app with CSRF middleware enabled."""
    from fastapi import Request
    from fastapi.responses import JSONResponse

    app = FastAPI()

    # Add exception handler for HTTPException (needed for middleware exceptions)
    @app.exception_handler(HTTPException)
    async def http_exception_handler(request: Request, exc: HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail},
        )

    app.add_middleware(CsrfMiddleware, enabled=True)
    app.include_router(auth_router, prefix="/api/auth")

    @app.get("/api/health")
    async def health():
        return {"status": "ok"}

    @app.post("/api/data")
    async def create_data():
        return {"created": True}

    @app.put("/api/data/{id}")
    async def update_data(id: str):
        return {"updated": id}

    @app.delete("/api/data/{id}")
    async def delete_data(id: str):
        return {"deleted": id}

    @app.patch("/api/data/{id}")
    async def patch_data(id: str):
        return {"patched": id}

    return app


@pytest.fixture
def client(app_with_csrf):
    """Test client with CSRF middleware enabled."""
    return TestClient(app_with_csrf, raise_server_exceptions=False)


@pytest.fixture
def app_csrf_disabled():
    """Create a test app with CSRF middleware disabled."""
    app = FastAPI()
    app.add_middleware(CsrfMiddleware, enabled=False)
    app.include_router(auth_router, prefix="/api/auth")

    @app.post("/api/data")
    async def create_data():
        return {"created": True}

    return app


@pytest.fixture
def client_disabled(app_csrf_disabled):
    """Test client with CSRF middleware disabled."""
    return TestClient(app_csrf_disabled)


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


class TestCsrfTokenEndpoint:
    """Test /api/auth/csrf-token endpoint."""

    def test_get_csrf_token(self, client):
        """Should return a CSRF token."""
        response = client.get("/api/auth/csrf-token")

        assert response.status_code == 200
        data = response.json()
        assert "token" in data
        assert isinstance(data["token"], str)
        assert len(data["token"]) > 20

    def test_get_csrf_token_sets_session_cookie(self, client):
        """Should set a session cookie if not present."""
        response = client.get("/api/auth/csrf-token")

        assert response.status_code == 200
        assert "karenina_session" in response.cookies

    def test_get_csrf_token_uses_existing_session(self, client):
        """Should use existing session cookie if present."""
        # First request to get session cookie
        response1 = client.get("/api/auth/csrf-token")
        session_cookie = response1.cookies.get("karenina_session")

        # Second request with the same cookie
        response2 = client.get("/api/auth/csrf-token", cookies={"karenina_session": session_cookie})

        # Should return same session cookie (reuse existing session)
        assert response2.cookies.get("karenina_session") == session_cookie
        assert response2.status_code == 200

    def test_get_csrf_token_compatibility_route(self, client):
        """Should return a CSRF token from the compatibility route /api/csrf-token."""
        response = client.get("/api/csrf-token")

        assert response.status_code == 200
        data = response.json()
        assert "token" in data
        assert isinstance(data["token"], str)


class TestCsrfMiddleware:
    """Test CSRF middleware validation."""

    def test_get_request_no_csrf_required(self, client):
        """GET requests should not require CSRF token."""
        response = client.get("/api/health")
        assert response.status_code == 200

    def test_post_request_without_csrf_token(self, client):
        """POST without CSRF token should be rejected."""
        response = client.post("/api/data")
        assert response.status_code == 403
        assert "CSRF token missing" in response.json()["detail"]

    def test_post_request_with_invalid_csrf_token(self, client):
        """POST with invalid CSRF token should be rejected."""
        response = client.post("/api/data", headers={"X-CSRF-Token": "invalid_token"})
        assert response.status_code == 403
        assert "Invalid or expired CSRF token" in response.json()["detail"]

    def test_post_request_with_valid_csrf_token(self, client):
        """POST with valid CSRF token should succeed."""
        # Get CSRF token
        token_response = client.get("/api/auth/csrf-token")
        token = token_response.json()["token"]
        session_cookie = token_response.cookies.get("karenina_session")

        # Use token in POST request
        response = client.post(
            "/api/data",
            headers={"X-CSRF-Token": token},
            cookies={"karenina_session": session_cookie},
        )
        assert response.status_code == 200
        assert response.json() == {"created": True}

    def test_put_request_with_valid_csrf_token(self, client):
        """PUT with valid CSRF token should succeed."""
        token_response = client.get("/api/auth/csrf-token")
        token = token_response.json()["token"]
        session_cookie = token_response.cookies.get("karenina_session")

        response = client.put(
            "/api/data/123",
            headers={"X-CSRF-Token": token},
            cookies={"karenina_session": session_cookie},
        )
        assert response.status_code == 200
        assert response.json() == {"updated": "123"}

    def test_delete_request_with_valid_csrf_token(self, client):
        """DELETE with valid CSRF token should succeed."""
        token_response = client.get("/api/auth/csrf-token")
        token = token_response.json()["token"]
        session_cookie = token_response.cookies.get("karenina_session")

        response = client.delete(
            "/api/data/456",
            headers={"X-CSRF-Token": token},
            cookies={"karenina_session": session_cookie},
        )
        assert response.status_code == 200
        assert response.json() == {"deleted": "456"}

    def test_patch_request_with_valid_csrf_token(self, client):
        """PATCH with valid CSRF token should succeed."""
        token_response = client.get("/api/auth/csrf-token")
        token = token_response.json()["token"]
        session_cookie = token_response.cookies.get("karenina_session")

        response = client.patch(
            "/api/data/789",
            headers={"X-CSRF-Token": token},
            cookies={"karenina_session": session_cookie},
        )
        assert response.status_code == 200
        assert response.json() == {"patched": "789"}

    def test_csrf_token_endpoint_exempt(self, client):
        """CSRF token endpoint should be accessible without token."""
        response = client.get("/api/auth/csrf-token")
        assert response.status_code == 200

    def test_health_endpoint_exempt(self, client):
        """Health endpoint should be exempt from CSRF."""
        response = client.get("/api/health")
        assert response.status_code == 200

    def test_csrf_disabled(self, client_disabled):
        """When CSRF is disabled, POST should work without token."""
        response = client_disabled.post("/api/data")
        assert response.status_code == 200
        assert response.json() == {"created": True}


class TestClientIdExtraction:
    """Test client ID extraction logic."""

    def test_get_client_id_with_session_cookie(self):
        """Should use session cookie when available."""
        from fastapi import Request
        from starlette.testclient import TestClient

        app = FastAPI()

        @app.get("/test")
        async def test_endpoint(request: Request):
            return {"client_id": get_client_id(request)}

        client = TestClient(app)
        response = client.get("/test", cookies={"karenina_session": "test_session"})

        assert response.json()["client_id"] == "session:test_session"

    def test_get_client_id_without_session_cookie(self):
        """Should fall back to IP-based ID when no session cookie."""
        from fastapi import Request

        app = FastAPI()

        @app.get("/test")
        async def test_endpoint(request: Request):
            return {"client_id": get_client_id(request)}

        client = TestClient(app)
        response = client.get("/test")

        client_id = response.json()["client_id"]
        assert client_id.startswith("ip:")


class TestCsrfMiddlewareExemptPaths:
    """Test CSRF middleware path exemptions."""

    def test_exempt_paths_static_assets(self):
        """Static asset paths should be exempt."""
        app = FastAPI()
        app.add_middleware(CsrfMiddleware, enabled=True)

        @app.post("/assets/main.js")
        async def fake_asset():
            return {"ok": True}

        client = TestClient(app)
        response = client.post("/assets/main.js")
        # If it was not exempt, it would return 403
        # Since there's no actual route, it returns 200 from the test endpoint
        assert response.status_code == 200

    def test_exempt_paths_websocket(self):
        """WebSocket paths should be exempt."""
        app = FastAPI()
        app.add_middleware(CsrfMiddleware, enabled=True)

        @app.post("/ws/connect")
        async def ws_endpoint():
            return {"ok": True}

        client = TestClient(app)
        response = client.post("/ws/connect")
        assert response.status_code == 200

    def test_custom_exempt_paths(self):
        """Custom exempt paths should work."""
        app = FastAPI()
        app.add_middleware(CsrfMiddleware, enabled=True, exempt_paths={"/api/custom-exempt"})

        @app.post("/api/custom-exempt")
        async def exempt_endpoint():
            return {"ok": True}

        client = TestClient(app)
        response = client.post("/api/custom-exempt")
        assert response.status_code == 200


class TestIntegrationWithRealApp:
    """Integration tests with the actual Karenina app."""

    @pytest.fixture
    def real_client(self):
        """Create a test client for the actual FastAPI app."""
        webapp_dir = Path(__file__).parent.parent / "webapp"

        # Temporarily enable CSRF for testing
        with patch.dict("os.environ", {"KARENINA_CSRF_ENABLED": "true"}):
            from karenina_server.server import create_fastapi_app

            app = create_fastapi_app(webapp_dir)
            return TestClient(app)

    def test_csrf_token_endpoint_exists(self, real_client):
        """CSRF token endpoint should be accessible in the real app."""
        response = real_client.get("/api/auth/csrf-token")
        assert response.status_code == 200
        assert "token" in response.json()
