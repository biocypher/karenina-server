"""Tests for ``VerificationResultSummary`` schema.

Ensures that ``/api/database/results`` response rows expose the new
``failure``/``caveats`` shape instead of the legacy
``completed_without_errors`` boolean. See the 2026-04-15
failure-state-harmonization plan for context.
"""

from __future__ import annotations

import pytest
from karenina.schemas.results.caveat import Caveat
from karenina.schemas.results.failure import Failure, FailureCategory

from karenina_server.schemas.database import VerificationResultSummary

REQUIRED_FIELDS = {
    "id": 1,
    "run_id": "run-1",
    "question_id": "q-1",
    "question_text": "What is 2+2?",
    "answering_model": "langchain:claude-haiku-4-5",
    "parsing_model": "langchain:claude-haiku-4-5",
    "template_verify_result": True,
    "execution_time": 1.5,
    "timestamp": "2026-04-15T12:00:00Z",
}


@pytest.mark.unit
class TestVerificationResultSummaryFields:
    """Verify the result summary exposes the new failure+caveats shape."""

    def test_legacy_completed_without_errors_rejected(self) -> None:
        """Passing the legacy field should now raise a validation error."""
        with pytest.raises(Exception):  # noqa: B017, PT011 - Pydantic wraps ValidationError
            VerificationResultSummary(
                **REQUIRED_FIELDS,
                completed_without_errors=True,
            )

    def test_accepts_failure_none(self) -> None:
        """A passing result has ``failure=None`` and an empty ``caveats`` list."""
        summary = VerificationResultSummary(**REQUIRED_FIELDS)
        assert summary.failure is None
        assert summary.caveats == []

    def test_accepts_structured_failure(self) -> None:
        """A failing row carries a structured :class:`Failure` object."""
        failure = Failure(
            category=FailureCategory.TIMEOUT,
            stage="generate_answer",
            reason="timeout retries exhausted",
        )
        summary = VerificationResultSummary(
            **REQUIRED_FIELDS,
            failure=failure,
        )
        assert summary.failure is not None
        assert summary.failure.category == FailureCategory.TIMEOUT
        # ``group`` is a computed field from the category mapping.
        assert summary.failure.group.value == "retry"

    def test_accepts_caveats_list(self) -> None:
        """``caveats`` accepts a list of :class:`Caveat` values."""
        summary = VerificationResultSummary(
            **REQUIRED_FIELDS,
            caveats=[Caveat.RETRIES_USED, Caveat.PARTIAL_CONTENT],
        )
        assert summary.caveats == [Caveat.RETRIES_USED, Caveat.PARTIAL_CONTENT]

    def test_round_trip_dumps_failure_and_caveats(self) -> None:
        """``model_dump`` emits ``failure`` and ``caveats`` keys in the JSON body."""
        summary = VerificationResultSummary(
            **REQUIRED_FIELDS,
            failure=Failure(
                category=FailureCategory.ABSTENTION,
                stage="abstention_check",
                reason="model declined",
            ),
            caveats=[Caveat.RETRIES_USED],
        )
        dumped = summary.model_dump(mode="json")
        assert "failure" in dumped
        assert "caveats" in dumped
        assert "completed_without_errors" not in dumped
        assert dumped["failure"]["category"] == "abstention"
        assert dumped["caveats"] == ["retries_used"]
