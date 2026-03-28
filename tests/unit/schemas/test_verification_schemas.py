"""Tests for verification endpoint Pydantic models."""

import pytest
from karenina.schemas.verification.api_models import FinishedTemplate

from karenina_server.schemas.verification import FinishedTemplatePayload

# Shared required fields for constructing a valid FinishedTemplatePayload.
REQUIRED_FIELDS = {
    "question_id": "q1",
    "question_text": "What is the capital of France?",
    "question_preview": "What is the capital...",
    "template_code": "class Answer(BaseAnswer): pass",
    "last_modified": "2026-03-23T00:00:00Z",
}


@pytest.mark.unit
class TestFinishedTemplatePayload:
    """Tests for FinishedTemplatePayload field completeness."""

    def test_finished_template_payload_accepts_question_dynamic_rubric(self):
        """Payload should accept and preserve question_dynamic_rubric."""
        dynamic_rubric = {"traits": []}
        payload = FinishedTemplatePayload(
            **REQUIRED_FIELDS,
            question_dynamic_rubric=dynamic_rubric,
        )
        assert payload.question_dynamic_rubric == {"traits": []}

    def test_finished_template_payload_accepts_workspace_path(self):
        """Payload should accept and preserve workspace_path."""
        payload = FinishedTemplatePayload(
            **REQUIRED_FIELDS,
            workspace_path="/some/path",
        )
        assert payload.workspace_path == "/some/path"

    def test_finished_template_payload_defaults_new_fields_to_none(self):
        """Both new fields should default to None when omitted."""
        payload = FinishedTemplatePayload(**REQUIRED_FIELDS)
        assert payload.question_dynamic_rubric is None
        assert payload.workspace_path is None

    def test_finished_template_payload_round_trips_to_core_finished_template(self):
        """Payload with all fields should round-trip to core FinishedTemplate."""
        payload = FinishedTemplatePayload(
            **REQUIRED_FIELDS,
            raw_answer="Paris",
            finished=True,
            question_rubric={"traits": [{"name": "accuracy"}]},
            question_dynamic_rubric={"traits": [{"name": "dynamic_trait"}]},
            keywords=["geography"],
            few_shot_examples=[{"q": "Example?", "a": "Yes"}],
            workspace_path="/workspace/agents",
        )

        dumped = payload.model_dump()
        core = FinishedTemplate(**dumped)

        assert core.question_dynamic_rubric == {"traits": [{"name": "dynamic_trait"}]}
        assert core.workspace_path == "/workspace/agents"
        # Verify other fields survive the round-trip too.
        assert core.question_id == "q1"
        assert core.raw_answer == "Paris"
        assert core.question_rubric == {"traits": [{"name": "accuracy"}]}
        assert core.keywords == ["geography"]
