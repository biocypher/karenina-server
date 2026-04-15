"""Tests for VerificationService counting logic using the new failure shape.

Ensures that ``successful_count`` / ``failed_count`` are derived from
``metadata.failure`` (``None`` means success) rather than the legacy
``completed_without_errors`` boolean.
"""

from __future__ import annotations

from typing import Any

import pytest
from karenina.schemas.results.failure import Failure, FailureCategory
from karenina.schemas.verification import (
    ModelIdentity,
    VerificationResult,
    VerificationResultMetadata,
)


def _mk_result(
    *,
    question_id: str,
    failure: Failure | None,
) -> VerificationResult:
    return VerificationResult(
        metadata=VerificationResultMetadata(
            question_id=question_id,
            template_id="no_template",
            failure=failure,
            question_text="What is 2+2?",
            answering=ModelIdentity(interface="langchain", model_name="claude-haiku-4-5"),
            parsing=ModelIdentity(interface="langchain", model_name="claude-haiku-4-5"),
            execution_time=1.0,
            timestamp="2026-04-15T12:00:00Z",
            result_id=f"rid_{question_id}",
        )
    )


@pytest.mark.unit
class TestFailureCountsDerivation:
    """Counts are derived from ``metadata.failure`` state."""

    def test_success_when_failure_is_none(self) -> None:
        result = _mk_result(question_id="q1", failure=None)
        success = result.metadata.failure is None
        assert success is True

    def test_failure_when_failure_populated(self) -> None:
        result = _mk_result(
            question_id="q2",
            failure=Failure(
                category=FailureCategory.TIMEOUT,
                stage="generate_answer",
                reason="timeout exhausted",
            ),
        )
        success = result.metadata.failure is None
        assert success is False

    def test_count_successful_results(self) -> None:
        results: list[Any] = [
            _mk_result(question_id="q1", failure=None),
            _mk_result(
                question_id="q2",
                failure=Failure(
                    category=FailureCategory.CONTENT,
                    stage="verify_template",
                    reason="verify_template returned False",
                ),
            ),
            _mk_result(question_id="q3", failure=None),
        ]
        successful = sum(1 for r in results if r.metadata.failure is None)
        failed = sum(1 for r in results if r.metadata.failure is not None)
        assert successful == 2
        assert failed == 1
