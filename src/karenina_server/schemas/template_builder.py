"""Request/response schemas for the template builder API."""

from typing import Any

from pydantic import BaseModel, Field


class TemplateParseRequest(BaseModel):
    """Request to parse Python template code into TemplateSpec JSON."""

    code: str = Field(..., description="Python source code for a template class.")


class TemplateParseResponse(BaseModel):
    """Response from parsing a template."""

    success: bool = True
    mode: str = Field(..., description="Template mode: 'verified', 'classic', or 'mixed'.")
    spec: dict[str, Any] | None = Field(None, description="TemplateSpec JSON (only for verified/mixed templates).")
    error: str | None = None


class TemplateGenerateRequest(BaseModel):
    """Request to generate Python code from TemplateSpec JSON."""

    spec: dict[str, Any] = Field(..., description="TemplateSpec JSON.")


class TemplateGenerateResponse(BaseModel):
    """Response with generated Python code."""

    success: bool = True
    code: str | None = None
    error: str | None = None


class TemplateValidateRequest(BaseModel):
    """Request to validate a template (Quick Check)."""

    code: str = Field(..., description="Python source code for validation.")


class TemplateValidateResponse(BaseModel):
    """Response from template validation."""

    success: bool = True
    valid: bool = False
    errors: list[str] = Field(default_factory=list)
    ground_truth_check: bool | None = None
    verify_check: bool | None = None


class TemplateTestRequest(BaseModel):
    """Request to test a template against a sample response."""

    code: str = Field(..., description="Python template source code.")
    sample_response: str = Field(..., description="Sample LLM response to parse.")
    question_text: str = Field("", description="Original question text.")
    judge_model_provider: str = Field("anthropic", description="Judge model provider.")
    judge_model_name: str = Field("claude-haiku-4-5", description="Judge model name.")


class TemplateTestResponse(BaseModel):
    """Response from template testing with a judge LLM."""

    success: bool = True
    parsed_fields: dict[str, Any] | None = None
    verify_result: bool | None = None
    verify_granular: float | None = None
    field_results: dict[str, bool] | None = None
    error: str | None = None


class PrimitiveInfo(BaseModel):
    """Information about an available verification primitive."""

    name: str
    description: str
    parameters: dict[str, Any] = Field(default_factory=dict, description="JSON schema for primitive parameters.")
    applies_to: list[str] = Field(default_factory=list, description="Field types this primitive applies to.")
    is_trace: bool = False


class PrimitiveListResponse(BaseModel):
    """Response listing available verification primitives."""

    success: bool = True
    primitives: list[PrimitiveInfo] = Field(default_factory=list)
