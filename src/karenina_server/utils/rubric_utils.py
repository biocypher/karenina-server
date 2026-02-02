"""Rubric parsing utilities shared across API handlers.

This module provides functions for converting frontend rubric dicts
to karenina Rubric objects.
"""

from typing import Any

from karenina.schemas import (
    CallableTrait,
    LLMRubricTrait,
    MetricRubricTrait,
    RegexTrait,
    Rubric,
)


def normalize_trait_kind(kind: str | None) -> str:
    """Normalize trait kind string to backend TraitKind values.

    The frontend may send different casing or naming conventions.
    This function maps them to the expected backend values.

    Args:
        kind: The trait kind from frontend (e.g., 'binary', 'Binary', 'score', 'Score')

    Returns:
        Normalized kind string ('boolean' or 'score')
    """
    if kind is None:
        return "score"
    if kind.lower() == "binary":
        return "boolean"
    if kind.lower() == "score":
        return "score"
    return kind


def parse_llm_trait(trait_data: dict[str, Any]) -> LLMRubricTrait:
    """Parse a dict into an LLMRubricTrait.

    Args:
        trait_data: Dict with trait properties from frontend.

    Returns:
        LLMRubricTrait instance.
    """
    kind = normalize_trait_kind(trait_data.get("kind"))
    return LLMRubricTrait(
        name=trait_data["name"],
        description=trait_data.get("description"),
        kind=kind,
        min_score=trait_data.get("min_score", 1) if kind == "score" else None,
        max_score=trait_data.get("max_score", 5) if kind == "score" else None,
    )


def parse_regex_trait(trait_data: dict[str, Any]) -> RegexTrait:
    """Parse a dict into a RegexTrait.

    Args:
        trait_data: Dict with trait properties from frontend.

    Returns:
        RegexTrait instance.
    """
    return RegexTrait(
        name=trait_data["name"],
        description=trait_data.get("description"),
        pattern=trait_data.get("pattern", ""),
        case_sensitive=trait_data.get("case_sensitive", True),
        invert_result=trait_data.get("invert_result", False),
    )


def parse_callable_trait(trait_data: dict[str, Any]) -> CallableTrait:
    """Parse a dict into a CallableTrait.

    Args:
        trait_data: Dict with trait properties from frontend.

    Returns:
        CallableTrait instance.
    """
    return CallableTrait(
        name=trait_data["name"],
        description=trait_data.get("description"),
        callable_code=trait_data.get("callable_code", b""),
        kind=trait_data.get("kind", "boolean"),
        min_score=trait_data.get("min_score"),
        max_score=trait_data.get("max_score"),
        invert_result=trait_data.get("invert_result", False),
    )


def parse_metric_trait(trait_data: dict[str, Any]) -> MetricRubricTrait:
    """Parse a dict into a MetricRubricTrait.

    Args:
        trait_data: Dict with trait properties from frontend.

    Returns:
        MetricRubricTrait instance.
    """
    return MetricRubricTrait(
        name=trait_data["name"],
        description=trait_data.get("description"),
        evaluation_mode=trait_data.get("evaluation_mode", "tp_only"),
        metrics=trait_data.get("metrics", []),
        tp_instructions=trait_data.get("tp_instructions", []),
        tn_instructions=trait_data.get("tn_instructions", []),
        repeated_extraction=trait_data.get("repeated_extraction", True),
    )


def build_rubric_from_dict(rubric_data: dict[str, Any]) -> Rubric | None:
    """Build a Rubric object from a frontend rubric dict.

    This function handles all trait types (LLM, regex, callable, metric)
    and returns a fully constructed Rubric object.

    Args:
        rubric_data: Dict containing trait lists from frontend.

    Returns:
        Rubric instance, or None if no traits are present.
    """
    llm_traits = [parse_llm_trait(t) for t in rubric_data.get("llm_traits", [])]
    regex_traits = [parse_regex_trait(t) for t in rubric_data.get("regex_traits", [])]
    callable_traits = [parse_callable_trait(t) for t in rubric_data.get("callable_traits", [])]
    metric_traits = [parse_metric_trait(t) for t in rubric_data.get("metric_traits", [])]

    if not any([llm_traits, regex_traits, callable_traits, metric_traits]):
        return None

    return Rubric(
        llm_traits=llm_traits,
        regex_traits=regex_traits,
        callable_traits=callable_traits,
        metric_traits=metric_traits,
    )
