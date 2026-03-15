"""API handlers for the template builder.

Endpoints:
    POST /api/v2/templates/builder/parse     Parse Python code to TemplateSpec JSON
    POST /api/v2/templates/builder/generate   TemplateSpec JSON to Python code
    POST /api/v2/templates/builder/validate   Quick Check validation
    POST /api/v2/templates/builder/test       Test with judge LLM
    GET  /api/v2/templates/builder/primitives List available primitives
"""

import logging

from fastapi import APIRouter, HTTPException

from karenina_server.schemas.template_builder import (
    PrimitiveInfo,
    PrimitiveListResponse,
    TemplateGenerateRequest,
    TemplateGenerateResponse,
    TemplateParseRequest,
    TemplateParseResponse,
    TemplateTestRequest,
    TemplateTestResponse,
    TemplateValidateRequest,
    TemplateValidateResponse,
)
from karenina_server.services.template_builder_service import template_builder_service

logger = logging.getLogger(__name__)

router = APIRouter()


def _sanitize_error_message(error: Exception) -> str:
    """Sanitize error message for API response."""
    msg = str(error)
    if len(msg) > 500:
        msg = msg[:500] + "..."
    return msg


@router.post("/v2/templates/builder/parse", response_model=TemplateParseResponse)
async def parse_template(request: TemplateParseRequest) -> TemplateParseResponse:
    """Parse Python template code into TemplateSpec JSON."""
    try:
        result = template_builder_service.parse_template(request.code)
        return TemplateParseResponse(
            mode=result["mode"],
            spec=result["spec"],
            error=result["error"],
            success=result["error"] is None,
        )
    except Exception as e:
        logger.error("Template parse error: %s", e)
        raise HTTPException(status_code=500, detail=_sanitize_error_message(e)) from e


@router.post("/v2/templates/builder/generate", response_model=TemplateGenerateResponse)
async def generate_template(request: TemplateGenerateRequest) -> TemplateGenerateResponse:
    """Generate Python code from TemplateSpec JSON."""
    try:
        code = template_builder_service.generate_code(request.spec)
        return TemplateGenerateResponse(code=code)
    except ValueError as e:
        return TemplateGenerateResponse(success=False, error=str(e))
    except Exception as e:
        logger.error("Template generate error: %s", e)
        raise HTTPException(status_code=500, detail=_sanitize_error_message(e)) from e


@router.post("/v2/templates/builder/validate", response_model=TemplateValidateResponse)
async def validate_template(request: TemplateValidateRequest) -> TemplateValidateResponse:
    """Run Quick Check validation on template code."""
    try:
        result = template_builder_service.validate_template(request.code)
        return TemplateValidateResponse(
            valid=result["valid"],
            errors=result["errors"],
            ground_truth_check=result["ground_truth_check"],
            verify_check=result["verify_check"],
        )
    except Exception as e:
        logger.error("Template validate error: %s", e)
        raise HTTPException(status_code=500, detail=_sanitize_error_message(e)) from e


@router.post("/v2/templates/builder/test", response_model=TemplateTestResponse)
async def test_template(request: TemplateTestRequest) -> TemplateTestResponse:
    """Test a template by parsing a sample response with a judge LLM.

    This endpoint requires a running LLM provider. It creates a minimal
    TemplateEvaluator, parses the sample response, and returns field results.
    """
    try:
        from karenina.adapters import get_parser
        from karenina.benchmark.verification.prompts.parsing.parsing_instructions import (
            TemplatePromptBuilder,
        )
        from karenina.benchmark.verification.utils.class_discovery import find_answer_class
        from karenina.benchmark.verification.utils.template_validation import (
            _build_exec_namespace,
        )
        from karenina.ports import Message
        from karenina.schemas.config import ModelConfig

        # Compile the template
        ns = _build_exec_namespace()
        exec(request.code, ns)  # noqa: S102
        answer_cls = find_answer_class(ns)

        # Create model config for the judge
        model_config = ModelConfig(
            id="template-test-judge",
            model_name=request.judge_model_name,
            model_provider=request.judge_model_provider,
            temperature=0.0,
        )

        # Build prompt
        builder = TemplatePromptBuilder(answer_class=answer_cls)
        system_text = builder.build_system_prompt()

        # Parse via adapter
        parser = get_parser(model_config)
        messages = [
            Message.system(system_text),
            Message.user(
                f"Parse the following response:\n\n"
                f"**QUESTION:** {request.question_text}\n\n"
                f"**RESPONSE:** {request.sample_response}"
            ),
        ]
        parse_result = parser.parse_to_pydantic(messages, answer_cls)

        if not isinstance(parse_result.parsed, answer_cls):
            return TemplateTestResponse(success=False, error="Judge failed to parse response.")

        parsed = parse_result.parsed
        verify_result = parsed.verify()

        # Get granular results if available
        granular = None
        field_results = None
        try:
            granular = parsed.verify_granular()
            field_results = parsed._compute_field_results()
        except Exception:
            pass

        # Build field values
        parsed_fields = {}
        for name in answer_cls.model_fields:
            if name not in ("id", "correct"):
                parsed_fields[name] = getattr(parsed, name, None)

        return TemplateTestResponse(
            parsed_fields=parsed_fields,
            verify_result=verify_result,
            verify_granular=granular,
            field_results=field_results,
        )

    except Exception as e:
        logger.error("Template test error: %s", e)
        return TemplateTestResponse(success=False, error=str(e))


@router.get("/v2/templates/builder/primitives", response_model=PrimitiveListResponse)
async def list_primitives() -> PrimitiveListResponse:
    """List all available verification primitives with parameter schemas."""
    try:
        primitives_data = template_builder_service.get_available_primitives()
        primitives = [PrimitiveInfo(**p) for p in primitives_data]
        return PrimitiveListResponse(primitives=primitives)
    except Exception as e:
        logger.error("Primitives list error: %s", e)
        raise HTTPException(status_code=500, detail=_sanitize_error_message(e)) from e
