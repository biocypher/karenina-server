"""Unit tests for database handler helpers."""

import hashlib

import pytest

from karenina_server.api.database_handlers import _build_question_from_data


@pytest.mark.unit
class TestBuildQuestionFromData:
    """Tests for _build_question_from_data helper."""

    def test_preserves_core_fields(self):
        """Standard fields (question, raw_answer, keywords) are preserved."""
        q_data = {
            "question": "What is the capital of France?",
            "raw_answer": "Paris",
            "keywords": ["geography", "europe"],
            "few_shot_examples": [{"q": "Capital of Germany?", "a": "Berlin"}],
            "answer_notes": "Simple factual recall",
        }
        question = _build_question_from_data(q_data)

        assert question.question == "What is the capital of France?"
        assert question.raw_answer == "Paris"
        assert question.keywords == ["geography", "europe"]
        assert question.few_shot_examples == [{"q": "Capital of Germany?", "a": "Berlin"}]
        assert question.answer_notes == "Simple factual recall"

    def test_preserves_previously_missing_fields(self):
        """The 7 previously dropped fields are all preserved."""
        q_data = {
            "question": "What drug targets EGFR?",
            "raw_answer": "Erlotinib",
            "question_dynamic_rubric": {"criteria": "specificity"},
            "workspace_path": "task_01",
            "author": {"name": "Dr. Smith", "affiliation": "MIT"},
            "sources": [{"url": "https://example.com", "title": "Source 1"}],
            "custom_metadata": {"difficulty": "hard", "domain": "oncology"},
            "date_created": "2025-01-15T10:00:00",
            "date_modified": "2025-06-20T14:30:00",
        }
        question = _build_question_from_data(q_data)

        assert question.question_dynamic_rubric == {"criteria": "specificity"}
        assert question.workspace_path == "task_01"
        assert question.author == {"name": "Dr. Smith", "affiliation": "MIT"}
        assert question.sources == [{"url": "https://example.com", "title": "Source 1"}]
        assert question.custom_metadata == {"difficulty": "hard", "domain": "oncology"}
        assert question.date_created == "2025-01-15T10:00:00"
        assert question.date_modified == "2025-06-20T14:30:00"

    def test_filters_non_question_keys(self):
        """Non-Question keys are filtered out (no ValidationError from extra='forbid')."""
        q_data = {
            "question": "What is 2+2?",
            "raw_answer": "4",
            "finished": True,
            "last_modified": "2025-03-01T00:00:00",
            "some_random_key": "should be dropped",
        }
        # Should not raise ValidationError
        question = _build_question_from_data(q_data)
        assert question.question == "What is 2+2?"
        assert question.raw_answer == "4"
        # Non-Question fields should not appear on the model
        assert not hasattr(question, "finished")
        assert not hasattr(question, "last_modified")
        assert not hasattr(question, "some_random_key")

    def test_handles_legacy_tags(self):
        """A q_data with 'tags' instead of 'keywords' produces correct keywords."""
        q_data = {
            "question": "What is DNA?",
            "raw_answer": "Deoxyribonucleic acid",
            "tags": ["biology", "genetics"],
        }
        question = _build_question_from_data(q_data)
        assert question.keywords == ["biology", "genetics"]

    def test_handles_id_in_data(self):
        """A q_data with 'id' key does not cause error; computed id is correct."""
        q_data = {
            "question": "What is RNA?",
            "raw_answer": "Ribonucleic acid",
            "id": "some-stale-id-from-frontend",
        }
        question = _build_question_from_data(q_data)
        expected_id = hashlib.md5(b"What is RNA?").hexdigest()
        assert question.id == expected_id
