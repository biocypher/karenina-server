"""Service for template builder operations.

Wraps the karenina core library's template converter, validator,
and pipeline utilities for use by the API handler.
"""

import logging
import threading
from typing import Any

from karenina.benchmark.authoring.answers.generator import (
    _smoke_test_generated_code,
)
from karenina.benchmark.verification.utils.class_discovery import find_answer_class
from karenina.benchmark.verification.utils.template_validation import (
    _build_exec_namespace,
    validate_answer_template,
)
from karenina.schemas.entities.template_spec import TemplateSpec

from karenina_server.services.template_converter import (
    detect_template_mode,
    python_to_spec,
    spec_to_python,
)

logger = logging.getLogger(__name__)


class TemplateBuilderService:
    """Service for template parsing, generation, validation, and testing."""

    def __init__(self) -> None:
        """Initialize the template builder service."""
        self._lock = threading.RLock()

    def parse_template(self, code: str) -> dict[str, Any]:
        """Parse Python code into TemplateSpec JSON with mode detection.

        Args:
            code: Python template source code.

        Returns:
            Dict with 'mode', 'spec' (or None), and 'error' (or None).
        """
        with self._lock:
            try:
                mode = detect_template_mode(code)
            except ValueError as e:
                return {"mode": "unknown", "spec": None, "error": str(e)}

            spec = None
            if mode in ("verified", "mixed"):
                try:
                    spec_obj = python_to_spec(code)
                    spec = spec_obj.model_dump(mode="json")
                except ValueError as e:
                    return {"mode": mode, "spec": None, "error": str(e)}

            return {"mode": mode, "spec": spec, "error": None}

    def generate_code(self, spec_dict: dict[str, Any]) -> str:
        """Generate Python code from TemplateSpec JSON.

        Runs the full generation pipeline: validate spec, generate code,
        and smoke test (exec + verify with ground truth values).

        Args:
            spec_dict: TemplateSpec as a JSON-compatible dict.

        Returns:
            Python source code string.

        Raises:
            ValueError: If the spec is invalid or the generated code
                fails the smoke test.
        """
        with self._lock:
            spec = TemplateSpec.model_validate(spec_dict)
            code: str = spec_to_python(spec)

            success, error_msg = _smoke_test_generated_code(code)
            if not success:
                raise ValueError(f"Generated code failed smoke test: {error_msg}")

            return code

    def validate_template(self, code: str) -> dict[str, Any]:
        """Run Quick Check validation on template code.

        Args:
            code: Python template source code.

        Returns:
            Dict with 'valid', 'errors', 'ground_truth_check', 'verify_check'.
        """
        with self._lock:
            # Step 1: Syntax/structure validation
            is_valid, error_msg, _ = validate_answer_template(code)
            if not is_valid:
                return {
                    "valid": False,
                    "errors": [error_msg or "Unknown validation error"],
                    "ground_truth_check": None,
                    "verify_check": None,
                }

            # Step 2: Exec and verify with ground truth
            ns = _build_exec_namespace()
            try:
                exec(code, ns)  # noqa: S102
                answer_cls = find_answer_class(ns)
            except Exception as e:
                return {
                    "valid": False,
                    "errors": [f"Failed to load template: {e}"],
                    "ground_truth_check": None,
                    "verify_check": None,
                }

            # Step 3: Check ground truth self-consistency
            ground_truth_ok = False
            verify_ok = False
            errors: list[str] = []

            try:
                verified_fields = answer_cls._get_verified_fields()
                if verified_fields:
                    kwargs = {name: meta.ground_truth for name, meta in verified_fields.items()}
                    instance = answer_cls(**kwargs)
                    ground_truth_ok = True

                    verify_result = instance.verify()
                    verify_ok = bool(verify_result)
                    if not verify_ok:
                        errors.append("verify() returned False with ground truth values")
                else:
                    # Classic template: just check it compiles
                    ground_truth_ok = True
                    verify_ok = True
            except Exception as e:
                errors.append(f"Smoke test failed: {e}")

            return {
                "valid": len(errors) == 0,
                "errors": errors,
                "ground_truth_check": ground_truth_ok,
                "verify_check": verify_ok,
            }

    def get_available_primitives(self) -> list[dict[str, Any]]:
        """List all available verification primitives with metadata.

        Returns:
            List of primitive info dicts.
        """
        from karenina.schemas.primitives.registry import _PRIMITIVE_REGISTRY

        primitives = []
        # Type-to-primitive applicability mapping
        type_map: dict[str, list[str]] = {
            "BooleanMatch": ["bool"],
            "ExactMatch": ["str", "int", "float"],
            "ContainsAny": ["str"],
            "ContainsAll": ["str"],
            "RegexMatch": ["str"],
            "SemanticMatch": ["str"],
            "NumericExact": ["int", "float"],
            "NumericTolerance": ["float"],
            "NumericRange": ["int", "float"],
            "SetContainment": ["list_str"],
            "OrderedMatch": ["list_str"],
            "LiteralMatch": ["literal", "str"],
            "DateMatch": ["date", "str"],
            "DateTolerance": ["date", "str"],
            "DateRange": ["date", "str"],
            "TraceRegex": ["bool"],
            "TraceContains": ["bool"],
            "TraceLength": ["bool"],
        }

        for name, cls in _PRIMITIVE_REGISTRY.items():
            schema = cls.model_json_schema()
            # Remove internal fields from properties
            props = {k: v for k, v in schema.get("properties", {}).items() if k != "type"}
            is_trace = name.startswith("Trace")

            primitives.append(
                {
                    "name": name,
                    "description": cls.__doc__ or "",
                    "parameters": props,
                    "applies_to": type_map.get(name, []),
                    "is_trace": is_trace,
                }
            )

        return primitives


# Module-level singleton
template_builder_service = TemplateBuilderService()
