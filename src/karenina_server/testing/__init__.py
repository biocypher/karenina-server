"""Testing utilities for karenina-server.

This module provides E2E testing support, including LLM fixture mocking.
"""

from .e2e_fixture_mode import is_e2e_mode, setup_e2e_fixture_mode

__all__ = ["is_e2e_mode", "setup_e2e_fixture_mode"]
