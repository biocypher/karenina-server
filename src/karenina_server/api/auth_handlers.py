"""Authentication and CSRF protection handlers.

This module provides CSRF (Cross-Site Request Forgery) protection for the Karenina API.

CSRF Protection Flow:
1. Frontend calls GET /api/auth/csrf-token to obtain a token
2. Frontend includes the token in X-CSRF-Token header on mutation requests
3. Middleware validates the token on POST, PUT, DELETE, PATCH requests

Token Implementation:
- Tokens are generated per-session using cryptographically secure random bytes
- Tokens are stored in memory with a configurable TTL (default 24 hours)
- Tokens are automatically cleaned up when expired
"""

import secrets
import time
from dataclasses import dataclass
from threading import Lock
from typing import ClassVar

from fastapi import APIRouter, Request, Response
from pydantic import BaseModel


class CsrfTokenResponse(BaseModel):
    """Response model for CSRF token endpoint."""

    token: str


@dataclass
class CsrfToken:
    """Represents a CSRF token with expiration metadata."""

    token: str
    created_at: float
    last_used_at: float

    def is_expired(self, ttl_seconds: float) -> bool:
        """Check if the token has expired based on TTL from creation."""
        return time.time() - self.created_at > ttl_seconds


class CsrfTokenStore:
    """Thread-safe in-memory CSRF token store.

    Tokens are stored per client (identified by cookie or IP).
    This is suitable for single-server deployments. For multi-server
    deployments, consider using Redis or another shared store.
    """

    _instance: ClassVar["CsrfTokenStore | None"] = None
    _lock: ClassVar[Lock] = Lock()

    # Instance attributes (initialized in __new__)
    _tokens: dict[str, CsrfToken]
    _token_to_client: dict[str, str]
    _store_lock: Lock
    _ttl_seconds: float
    _token_length: int

    def __new__(cls) -> "CsrfTokenStore":
        """Singleton pattern for token store."""
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._tokens = {}
                    cls._instance._token_to_client = {}
                    cls._instance._store_lock = Lock()
                    cls._instance._ttl_seconds = 24 * 60 * 60  # 24 hours
                    cls._instance._token_length = 32  # 256 bits of entropy
        return cls._instance

    def generate_token(self, client_id: str) -> str:
        """Generate a new CSRF token for a client.

        Args:
            client_id: Unique identifier for the client (e.g., session cookie or IP)

        Returns:
            The generated token string
        """
        token = secrets.token_urlsafe(self._token_length)
        now = time.time()

        with self._store_lock:
            # Clean up old token for this client if exists
            if client_id in self._tokens:
                old_token = self._tokens[client_id].token
                self._token_to_client.pop(old_token, None)

            # Store new token
            csrf_token = CsrfToken(token=token, created_at=now, last_used_at=now)
            self._tokens[client_id] = csrf_token
            self._token_to_client[token] = client_id

            # Periodic cleanup of expired tokens
            self._cleanup_expired_tokens()

        return token

    def validate_token(self, client_id: str, token: str) -> bool:
        """Validate a CSRF token for a client.

        Args:
            client_id: Unique identifier for the client
            token: Token to validate

        Returns:
            True if token is valid for this client, False otherwise
        """
        with self._store_lock:
            stored = self._tokens.get(client_id)

            if stored is None:
                return False

            if stored.is_expired(self._ttl_seconds):
                # Clean up expired token
                self._tokens.pop(client_id, None)
                self._token_to_client.pop(stored.token, None)
                return False

            # Use constant-time comparison to prevent timing attacks
            if secrets.compare_digest(stored.token, token):
                # Update last used time
                stored.last_used_at = time.time()
                return True

            return False

    def invalidate_token(self, client_id: str) -> None:
        """Invalidate and remove a token for a client.

        Args:
            client_id: Unique identifier for the client
        """
        with self._store_lock:
            stored = self._tokens.pop(client_id, None)
            if stored:
                self._token_to_client.pop(stored.token, None)

    def _cleanup_expired_tokens(self) -> None:
        """Remove expired tokens from the store.

        This is called periodically during token generation.
        Should be called while holding _store_lock.
        """
        expired_clients = [
            client_id for client_id, token in self._tokens.items() if token.is_expired(self._ttl_seconds)
        ]

        for client_id in expired_clients:
            stored = self._tokens.pop(client_id, None)
            if stored:
                self._token_to_client.pop(stored.token, None)

    def set_ttl(self, ttl_seconds: float) -> None:
        """Configure token TTL (for testing)."""
        self._ttl_seconds = ttl_seconds

    def clear_all(self) -> None:
        """Clear all tokens (for testing)."""
        with self._store_lock:
            self._tokens.clear()
            self._token_to_client.clear()


# Global token store instance
csrf_store = CsrfTokenStore()


def get_client_id(request: Request) -> str:
    """Get a unique identifier for the client.

    Uses session cookie if available, falls back to IP address.
    In a real production deployment, you'd want proper session management.

    Args:
        request: FastAPI request object

    Returns:
        Client identifier string
    """
    # Try to get from a session cookie first
    session_cookie = request.cookies.get("karenina_session")
    if session_cookie:
        return f"session:{session_cookie}"

    # Fall back to client IP (less reliable due to NAT, proxies, etc.)
    client_host = request.client.host if request.client else "unknown"
    user_agent = request.headers.get("user-agent", "unknown")[:50]  # Truncate for safety

    # Combine IP and user-agent for slightly better fingerprinting
    return f"ip:{client_host}:{hash(user_agent)}"


router = APIRouter()


@router.get("/csrf-token", response_model=CsrfTokenResponse)
async def get_csrf_token(request: Request, response: Response) -> CsrfTokenResponse:
    """Generate and return a CSRF token for the client.

    The token should be included in the X-CSRF-Token header for all
    state-changing requests (POST, PUT, DELETE, PATCH).

    Returns:
        CsrfTokenResponse containing the CSRF token
    """
    client_id = get_client_id(request)

    # If no session cookie exists, create one for better client tracking
    if not request.cookies.get("karenina_session"):
        session_id = secrets.token_urlsafe(16)
        response.set_cookie(
            key="karenina_session",
            value=session_id,
            httponly=True,
            samesite="lax",
            max_age=24 * 60 * 60,  # 24 hours
        )
        # Use the new session ID for the CSRF token
        client_id = f"session:{session_id}"

    token = csrf_store.generate_token(client_id)

    return CsrfTokenResponse(token=token)
