"""Karenina Server: FastAPI server for the Karenina benchmarking system.

This package provides the web API layer for Karenina, including:
- REST API endpoints for benchmark operations
- WebSocket support for real-time updates
- Async job management for long-running tasks
- File upload and processing capabilities
"""

__version__ = "0.1.0"

# Import services to make them available for testing
from . import services as services  # noqa: F401
