"""Tests for the verification-result summary projection used by
``/api/v2/verification-results``.

Validates that the projection exposes ``failure`` and ``caveats`` fields
instead of the legacy ``completed_without_errors`` boolean.
"""

from __future__ import annotations

from typing import Any

import pytest


def _sample_result_dict(
    *,
    failure: dict[str, Any] | None = None,
    caveats: list[str] | None = None,
) -> dict[str, Any]:
    """Build a storage-layer result dict (shape of ``load_verification_results(as_dict=False)``)."""
    return {
        "id": 42,
        "run_id": "run-1",
        "metadata": {
            "question_id": "q-1",
            "question_text": "What is 2+2?",
            "answering": {"interface": "langchain", "model_name": "claude-haiku-4-5"},
            "parsing": {"interface": "langchain", "model_name": "claude-haiku-4-5"},
            "failure": failure,
            "caveats": caveats or [],
            "execution_time": 1.5,
            "timestamp": "2026-04-15T12:00:00Z",
        },
        "template": {"verify_result": True},
    }


@pytest.mark.unit
class TestVerificationResultSummaryProjection:
    """Tests for ``_result_to_summary_dict`` (the response projection)."""

    def test_passes_through_failure_when_failure_present(self) -> None:
        """Response dict should include the ``failure`` object as-is."""
        from karenina_server.api.database_handlers import (
            _result_to_summary_dict,
        )

        failure_payload = {
            "category": "timeout",
            "stage": "generate_answer",
            "reason": "timeout retries exhausted",
        }
        row = _sample_result_dict(failure=failure_payload)

        summary = _result_to_summary_dict(row)

        assert summary["failure"] == failure_payload
        assert summary["caveats"] == []
        assert "completed_without_errors" not in summary

    def test_passes_through_none_failure(self) -> None:
        """A passing result has ``failure=None`` and empty ``caveats``."""
        from karenina_server.api.database_handlers import (
            _result_to_summary_dict,
        )

        row = _sample_result_dict(failure=None)
        summary = _result_to_summary_dict(row)
        assert summary["failure"] is None
        assert summary["caveats"] == []

    def test_passes_through_caveats(self) -> None:
        """``caveats`` should propagate into the summary dict."""
        from karenina_server.api.database_handlers import (
            _result_to_summary_dict,
        )

        row = _sample_result_dict(
            failure=None,
            caveats=["retries_used", "partial_content"],
        )
        summary = _result_to_summary_dict(row)
        assert summary["caveats"] == ["retries_used", "partial_content"]

    def test_preserves_core_fields(self) -> None:
        """Core identification fields remain unchanged."""
        from karenina_server.api.database_handlers import (
            _result_to_summary_dict,
        )

        row = _sample_result_dict()
        summary = _result_to_summary_dict(row)
        assert summary["id"] == 42
        assert summary["run_id"] == "run-1"
        assert summary["question_id"] == "q-1"
        assert summary["answering_model"] == "langchain:claude-haiku-4-5"
        assert summary["template_verify_result"] is True
