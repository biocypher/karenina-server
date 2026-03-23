"""Tests for rubric_utils agentic trait parsing (Issue 060)."""

import pytest
from karenina.schemas import AgenticRubricTrait, Rubric

from karenina_server.utils.rubric_utils import (
    build_rubric_from_dict,
    parse_agentic_trait,
)


@pytest.mark.unit
class TestParseAgenticTrait:
    """Tests for parse_agentic_trait()."""

    def test_minimal_agentic_trait(self):
        """Minimal dict produces a valid AgenticRubricTrait."""
        data = {
            "name": "checks_sources",
            "description": "Agent verifies cited sources exist",
            "kind": "boolean",
            "higher_is_better": True,
        }
        trait = parse_agentic_trait(data)
        assert isinstance(trait, AgenticRubricTrait)
        assert trait.name == "checks_sources"
        assert trait.kind == "boolean"

    def test_kind_normalization_binary_to_boolean(self):
        """Frontend 'binary' is normalized to 'boolean'."""
        data = {
            "name": "checks_sources",
            "description": "Agent verifies cited sources exist",
            "kind": "binary",
            "higher_is_better": True,
        }
        trait = parse_agentic_trait(data)
        assert trait.kind == "boolean"

    def test_kind_normalization_case_insensitive(self):
        """Kind normalization is case-insensitive."""
        data = {
            "name": "checks_sources",
            "description": "Agent verifies cited sources exist",
            "kind": "Binary",
            "higher_is_better": True,
        }
        trait = parse_agentic_trait(data)
        assert trait.kind == "boolean"

    def test_score_kind_with_bounds(self):
        """Score kind preserves min/max score."""
        data = {
            "name": "depth_of_analysis",
            "description": "How deeply the agent investigates",
            "kind": "score",
            "higher_is_better": True,
            "min_score": 1,
            "max_score": 10,
        }
        trait = parse_agentic_trait(data)
        assert trait.kind == "score"
        assert trait.min_score == 1
        assert trait.max_score == 10

    def test_all_agentic_specific_fields(self):
        """All agentic-specific fields are passed through."""
        data = {
            "name": "thorough_review",
            "description": "Agent reviews all files",
            "kind": "boolean",
            "higher_is_better": True,
            "context_mode": "workspace_only",
            "materialize_trace": False,
            "persist_trace": False,
            "max_turns": 20,
            "timeout_seconds": 300,
        }
        trait = parse_agentic_trait(data)
        assert trait.context_mode == "workspace_only"
        assert trait.materialize_trace is False
        assert trait.max_turns == 20
        assert trait.timeout_seconds == 300

    def test_does_not_mutate_input(self):
        """parse_agentic_trait does not modify the input dict."""
        data = {
            "name": "test",
            "description": "test trait",
            "kind": "binary",
            "higher_is_better": True,
        }
        original_kind = data["kind"]
        parse_agentic_trait(data)
        assert data["kind"] == original_kind


@pytest.mark.unit
class TestBuildRubricFromDictWithAgenticTraits:
    """Tests for agentic trait support in build_rubric_from_dict()."""

    def test_agentic_traits_only(self):
        """Rubric with only agentic traits is not None."""
        rubric_data = {
            "agentic_traits": [
                {
                    "name": "checks_sources",
                    "description": "Agent verifies cited sources exist",
                    "kind": "boolean",
                    "higher_is_better": True,
                },
            ],
        }
        rubric = build_rubric_from_dict(rubric_data)
        assert rubric is not None
        assert isinstance(rubric, Rubric)
        assert len(rubric.agentic_traits) == 1
        assert rubric.agentic_traits[0].name == "checks_sources"

    def test_agentic_traits_mixed_with_llm(self):
        """Rubric with both LLM and agentic traits includes both."""
        rubric_data = {
            "llm_traits": [
                {"name": "clarity", "description": "Is the response clear?"},
            ],
            "agentic_traits": [
                {
                    "name": "checks_sources",
                    "description": "Agent verifies cited sources exist",
                    "kind": "boolean",
                    "higher_is_better": True,
                },
            ],
        }
        rubric = build_rubric_from_dict(rubric_data)
        assert rubric is not None
        assert len(rubric.llm_traits) == 1
        assert len(rubric.agentic_traits) == 1

    def test_empty_agentic_traits_no_other_traits_returns_none(self):
        """Empty agentic_traits with no other traits returns None."""
        rubric_data: dict[str, list[dict[str, object]]] = {"agentic_traits": []}
        rubric = build_rubric_from_dict(rubric_data)
        assert rubric is None

    def test_multiple_agentic_traits(self):
        """Multiple agentic traits are all parsed."""
        rubric_data = {
            "agentic_traits": [
                {
                    "name": "trait_a",
                    "description": "First trait",
                    "kind": "boolean",
                    "higher_is_better": True,
                },
                {
                    "name": "trait_b",
                    "description": "Second trait",
                    "kind": "score",
                    "higher_is_better": True,
                    "min_score": 1,
                    "max_score": 5,
                },
            ],
        }
        rubric = build_rubric_from_dict(rubric_data)
        assert rubric is not None
        assert len(rubric.agentic_traits) == 2
        assert rubric.agentic_traits[0].kind == "boolean"
        assert rubric.agentic_traits[1].kind == "score"
