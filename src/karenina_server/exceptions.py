"""Custom exceptions for karenina-server API.

This module defines application-specific exceptions that map to HTTP status codes.
These provide semantic meaning for error conditions and enable consistent error handling.

Usage:
    from karenina_server.exceptions import NotFoundError, ValidationError

    # In a handler:
    if not resource:
        raise NotFoundError("Resource not found")

    # In global exception handler, these map to appropriate HTTP status codes:
    # NotFoundError -> 404
    # ValidationError -> 400
    # ConflictError -> 409
    # ServiceUnavailableError -> 503
"""

from typing import Any


class APIError(Exception):
    """Base exception for API errors.

    All custom API exceptions should inherit from this class.
    Each subclass defines its associated HTTP status code.
    """

    status_code: int = 500
    default_detail: str = "Internal server error"

    def __init__(self, detail: str | None = None, **extra: Any) -> None:
        """Initialize the API error.

        Args:
            detail: Human-readable error message. If None, uses default_detail.
            **extra: Additional context to include in the error response.
        """
        self.detail = detail or self.default_detail
        self.extra = extra
        super().__init__(self.detail)


class NotFoundError(APIError):
    """Resource not found (HTTP 404).

    Use when a requested resource doesn't exist.

    Examples:
        - Benchmark not found
        - Database file not found
        - Verification run not found
    """

    status_code = 404
    default_detail = "Resource not found"


class ValidationError(APIError):
    """Request validation failed (HTTP 400).

    Use for client errors where the request is malformed or contains invalid data.

    Examples:
        - Invalid URL format
        - Missing required fields
        - Invalid field values
    """

    status_code = 400
    default_detail = "Validation error"


class ConflictError(APIError):
    """Resource conflict (HTTP 409).

    Use when the request conflicts with current state.

    Examples:
        - Benchmark with same name already exists
        - Duplicate question ID
        - Concurrent modification conflict
    """

    status_code = 409
    default_detail = "Resource conflict"


class ServiceUnavailableError(APIError):
    """Service or dependency unavailable (HTTP 503).

    Use when a required service or dependency is not available.

    Examples:
        - Storage functionality not installed
        - LLM service unavailable
        - External API temporarily down
    """

    status_code = 503
    default_detail = "Service unavailable"


class ForbiddenError(APIError):
    """Access forbidden (HTTP 403).

    Use when the user is authenticated but lacks permission.

    Examples:
        - File outside allowed directory
        - Protected resource
        - Insufficient privileges
    """

    status_code = 403
    default_detail = "Access forbidden"
