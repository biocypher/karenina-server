"""Attribute access utilities for safe nested attribute retrieval."""

from typing import Any, TypeVar

T = TypeVar("T")


def get_attr_safe(obj: Any, attr: str, default: T | None = None) -> Any | T | None:
    """Safely get an attribute from an object.

    Combines existence check, hasattr check, and None check into a single call.
    Returns the attribute value if the object exists, has the attribute, and
    the attribute is not None. Otherwise returns the default.

    Args:
        obj: The object to get the attribute from (can be None).
        attr: The attribute name to retrieve.
        default: The default value to return if attribute access fails.

    Returns:
        The attribute value if accessible and not None, otherwise the default.

    Examples:
        >>> get_attr_safe(result.template, "verify_result")
        True
        >>> get_attr_safe(None, "anything")
        None
        >>> get_attr_safe(result.template, "missing_attr", False)
        False
    """
    if obj is None:
        return default
    if not hasattr(obj, attr):
        return default
    value = getattr(obj, attr)
    if value is None:
        return default
    return value


def has_attr_truthy(obj: Any, attr: str) -> bool:
    """Check if an object has an attribute with a truthy value.

    Args:
        obj: The object to check (can be None).
        attr: The attribute name to check.

    Returns:
        True if obj exists, has the attribute, and the attribute is truthy.

    Examples:
        >>> has_attr_truthy(result.rubric, "llm_trait_scores")
        True
        >>> has_attr_truthy(None, "anything")
        False
    """
    if obj is None:
        return False
    if not hasattr(obj, attr):
        return False
    return bool(getattr(obj, attr))
