"""E2E Fixture Mode for karenina-server.

When KARENINA_E2E_MODE=true, this module patches the LLM initialization
to use FixtureBackedLLMClient instead of real API calls.

This enables E2E tests to run against a real server while keeping LLM
responses deterministic and avoiding API costs.

Usage:
    Set environment variables before starting the server:
    - KARENINA_E2E_MODE=true
    - KARENINA_E2E_FIXTURE_DIR=/path/to/fixtures

The fixture directory should contain JSON fixture files organized by
SHA256 hash of the request messages, matching the format used by
karenina's FixtureBackedLLMClient.
"""

import contextlib
import logging
import os
from pathlib import Path
from typing import Any
from unittest.mock import patch

logger = logging.getLogger(__name__)

# Global state for E2E mode
_e2e_patches: list[Any] = []
_e2e_initialized = False


def is_e2e_mode() -> bool:
    """Check if E2E mode is enabled via environment variable."""
    return os.environ.get("KARENINA_E2E_MODE", "").lower() == "true"


def get_fixture_dir() -> Path | None:
    """Get the E2E fixture directory from environment."""
    fixture_dir = os.environ.get("KARENINA_E2E_FIXTURE_DIR")
    if fixture_dir:
        return Path(fixture_dir)
    return None


def setup_e2e_fixture_mode() -> bool:
    """Set up E2E fixture mode if enabled.

    This patches karenina's init_chat_model_unified to return a
    FixtureBackedLLMClient instead of a real LLM client.

    Returns:
        True if E2E mode was set up, False otherwise.
    """
    global _e2e_patches, _e2e_initialized

    if _e2e_initialized:
        return is_e2e_mode()

    _e2e_initialized = True

    if not is_e2e_mode():
        logger.debug("E2E mode not enabled")
        return False

    fixture_dir = get_fixture_dir()
    if not fixture_dir:
        logger.warning("KARENINA_E2E_MODE is true but KARENINA_E2E_FIXTURE_DIR not set")
        return False

    if not fixture_dir.exists():
        logger.warning(f"E2E fixture directory does not exist: {fixture_dir}")
        # Create it for convenience
        fixture_dir.mkdir(parents=True, exist_ok=True)
        logger.info(f"Created E2E fixture directory: {fixture_dir}")

    logger.info(f"Setting up E2E fixture mode with fixtures from: {fixture_dir}")

    try:
        # Import FixtureBackedLLMClient from karenina's utils package
        from karenina.utils.testing import FixtureBackedLLMClient

        # Create the fixture-backed client
        fixture_client = FixtureBackedLLMClient(fixture_dir)

        # Create a factory function that returns the fixture client
        def fixture_backed_init(*args: Any, **kwargs: Any) -> FixtureBackedLLMClient:
            """Return the fixture-backed LLM client instead of real client."""
            logger.debug(f"E2E mode: Returning fixture-backed client (args={args}, kwargs={kwargs})")
            return fixture_client

        # Patch init_chat_model_unified in all relevant modules
        # The function is defined in karenina.infrastructure.llm.interface
        # but may be imported elsewhere, so we patch at the source
        patch_targets = [
            "karenina.infrastructure.llm.interface.init_chat_model_unified",
            "karenina.infrastructure.llm.init_chat_model_unified",
        ]

        for target in patch_targets:
            try:
                p = patch(target, side_effect=fixture_backed_init)
                p.start()
                _e2e_patches.append(p)
                logger.debug(f"Patched: {target}")
            except Exception as e:
                logger.debug(f"Could not patch {target}: {e}")

        logger.info(f"E2E fixture mode enabled with {len(_e2e_patches)} patches")
        return True

    except ImportError as e:
        logger.error(f"Failed to import FixtureBackedLLMClient: {e}")
        logger.error("Make sure karenina is installed in the environment")
        return False
    except Exception as e:
        logger.error(f"Failed to set up E2E fixture mode: {e}")
        return False


def teardown_e2e_fixture_mode() -> None:
    """Tear down E2E fixture mode, removing patches.

    This is primarily for testing purposes.
    """
    global _e2e_patches, _e2e_initialized

    for p in _e2e_patches:
        with contextlib.suppress(Exception):
            p.stop()

    _e2e_patches = []
    _e2e_initialized = False
    logger.debug("E2E fixture mode teardown complete")
