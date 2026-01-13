"""CSRF validation middleware.

This middleware validates CSRF tokens on state-changing requests (POST, PUT, DELETE, PATCH).
"""

from collections.abc import Awaitable, Callable
from typing import Any

from fastapi import Request, Response
from starlette.middleware.base import BaseHTTPMiddleware

from karenina_server.api.auth_handlers import csrf_store, get_client_id


class CsrfMiddleware(BaseHTTPMiddleware):
    """Middleware to validate CSRF tokens on mutation requests.

    This middleware:
    - Skips validation for safe HTTP methods (GET, HEAD, OPTIONS, TRACE)
    - Skips validation for API documentation endpoints (/docs, /openapi.json)
    - Skips validation for the CSRF token endpoint itself
    - Validates X-CSRF-Token header on all other requests

    Configuration:
        enabled: Whether CSRF validation is enabled (default: True)
        exempt_paths: Set of paths to exempt from CSRF validation
    """

    SAFE_METHODS = frozenset({"GET", "HEAD", "OPTIONS", "TRACE"})
    CSRF_HEADER = "X-CSRF-Token"

    # Paths that are exempt from CSRF validation
    DEFAULT_EXEMPT_PATHS = frozenset(
        {
            "/docs",
            "/redoc",
            "/openapi.json",
            "/api/auth/csrf-token",
            "/api/health",
        }
    )

    def __init__(
        self,
        app: Callable[..., Any],
        enabled: bool = True,
        exempt_paths: set[str] | None = None,
    ) -> None:
        """Initialize CSRF middleware.

        Args:
            app: The ASGI application
            enabled: Whether CSRF validation is enabled
            exempt_paths: Additional paths to exempt from validation
        """
        super().__init__(app)
        self.enabled = enabled
        self.exempt_paths = self.DEFAULT_EXEMPT_PATHS.copy()
        if exempt_paths:
            self.exempt_paths = self.exempt_paths | exempt_paths

    async def dispatch(self, request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
        """Process the request and validate CSRF token if needed."""
        from starlette.responses import JSONResponse

        # Skip if disabled
        if not self.enabled:
            return await call_next(request)

        # Skip for safe methods
        if request.method in self.SAFE_METHODS:
            return await call_next(request)

        # Skip for exempt paths
        path = request.url.path
        if self._is_exempt_path(path):
            return await call_next(request)

        # Validate CSRF token
        token = request.headers.get(self.CSRF_HEADER)

        if not token:
            return JSONResponse(
                status_code=403,
                content={"detail": "CSRF token missing. Include X-CSRF-Token header with your request."},
            )

        client_id = get_client_id(request)

        if not csrf_store.validate_token(client_id, token):
            return JSONResponse(
                status_code=403,
                content={"detail": "Invalid or expired CSRF token. Please refresh the token and try again."},
            )

        # Token is valid, proceed with request
        return await call_next(request)

    def _is_exempt_path(self, path: str) -> bool:
        """Check if a path is exempt from CSRF validation.

        Args:
            path: The request path

        Returns:
            True if the path is exempt, False otherwise
        """
        # Exact match
        if path in self.exempt_paths:
            return True

        # Prefix match for static assets and websocket endpoints
        exempt_prefixes = ("/assets/", "/ws/")
        return any(path.startswith(prefix) for prefix in exempt_prefixes)


def create_csrf_middleware(enabled: bool = True, exempt_paths: set[str] | None = None) -> type[CsrfMiddleware]:
    """Factory function to create CSRF middleware with configuration.

    This allows configuring the middleware before adding it to the app.

    Args:
        enabled: Whether CSRF validation is enabled
        exempt_paths: Additional paths to exempt from validation

    Returns:
        Configured CSRF middleware class
    """

    class ConfiguredCsrfMiddleware(CsrfMiddleware):
        def __init__(self, app: Callable[..., Any]) -> None:
            super().__init__(app, enabled=enabled, exempt_paths=exempt_paths)

    return ConfiguredCsrfMiddleware
