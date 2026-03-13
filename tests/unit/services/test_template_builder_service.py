"""Unit tests for template builder service generate pipeline.

Tests that generate_code runs the full pipeline:
validate spec -> generate code -> smoke test.
"""

import pytest

from karenina_server.services.template_builder_service import TemplateBuilderService


@pytest.mark.unit
class TestGenerateCodePipeline:
    """Test that generate_code validates, generates, and smoke-tests."""

    def setup_method(self):
        self.service = TemplateBuilderService()

    def test_valid_spec_returns_executable_code(self):
        """Valid spec produces code that can be exec'd and verified."""
        spec = {
            "fields": [
                {
                    "name": "target",
                    "type": "str",
                    "description": "Protein target",
                    "ground_truth": "BCL2",
                    "verify_with": {"type": "ExactMatch"},
                },
            ],
        }
        result = self.service.generate_code(spec)
        assert "VerifiedField(" in result
        assert "ExactMatch(" in result
        assert "class Answer(BaseAnswer):" in result

    def test_unknown_primitive_rejected(self):
        """Spec with nonexistent primitive raises ValueError."""
        spec = {
            "fields": [
                {
                    "name": "target",
                    "type": "str",
                    "description": "target",
                    "ground_truth": "BCL2",
                    "verify_with": {"type": "FuzzyMatch", "threshold": 0.6},
                },
            ],
        }
        with pytest.raises(ValueError, match="FuzzyMatch"):
            self.service.generate_code(spec)

    def test_invalid_primitive_params_rejected(self):
        """Spec with wrong primitive parameters raises ValueError."""
        spec = {
            "fields": [
                {
                    "name": "target",
                    "type": "str",
                    "description": "target",
                    "ground_truth": "BCL2",
                    "verify_with": {"type": "ContainsAll", "terms": ["a"]},
                },
            ],
        }
        with pytest.raises(ValueError, match="ContainsAll"):
            self.service.generate_code(spec)

    def test_primitive_type_mismatch_rejected(self):
        """Spec with incompatible primitive/field type raises ValueError."""
        spec = {
            "fields": [
                {
                    "name": "flag",
                    "type": "bool",
                    "description": "flag",
                    "ground_truth": True,
                    "verify_with": {"type": "ExactMatch"},
                },
            ],
        }
        with pytest.raises(ValueError, match="not compatible"):
            self.service.generate_code(spec)

    def test_smoke_test_runs_on_generated_code(self):
        """Generated code passes smoke test (exec + verify with ground truth)."""
        spec = {
            "fields": [
                {
                    "name": "target",
                    "type": "str",
                    "description": "Protein target",
                    "ground_truth": "BCL2",
                    "verify_with": {"type": "ExactMatch"},
                },
                {
                    "name": "approved",
                    "type": "bool",
                    "description": "FDA approved",
                    "ground_truth": True,
                    "verify_with": {"type": "BooleanMatch"},
                },
            ],
        }
        code = self.service.generate_code(spec)

        # Verify the code actually works end-to-end
        from karenina.benchmark.verification.utils.class_discovery import find_answer_class
        from karenina.benchmark.verification.utils.template_validation import (
            _build_exec_namespace,
        )

        ns = _build_exec_namespace()
        exec(code, ns)  # noqa: S102
        answer_cls = find_answer_class(ns)
        verified_fields = answer_cls._get_verified_fields()
        kwargs = {name: meta.ground_truth for name, meta in verified_fields.items()}
        instance = answer_cls(**kwargs)
        assert instance.verify() is True

    def test_smoke_test_failure_raises(self):
        """If generated code fails smoke test, generate_code raises ValueError."""
        # A spec where ground truth values will cause verify to fail:
        # NumericRange expects value in range, but ground truth is outside it
        spec = {
            "fields": [
                {
                    "name": "score",
                    "type": "float",
                    "description": "Score value",
                    "ground_truth": 100.0,
                    "verify_with": {"type": "NumericRange", "min": 0.0, "max": 10.0},
                },
            ],
        }
        with pytest.raises(ValueError, match="smoke test"):
            self.service.generate_code(spec)
