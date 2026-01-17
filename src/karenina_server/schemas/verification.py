"""Verification endpoint Pydantic models for the Karenina API."""

from typing import Any

from pydantic import BaseModel, Field


class FinishedTemplatePayload(BaseModel):
    """Payload for a finished answer template in verification request.

    This is the server-side representation of template data sent from the GUI.
    Maps to karenina.schemas.workflow.FinishedTemplate.
    """

    question_id: str = Field(description="Unique identifier for the question")
    question_text: str = Field(description="Full text of the question")
    question_preview: str = Field(description="Truncated preview for UI display")
    raw_answer: str | None = Field(default=None, description="Ground truth answer from checkpoint")
    template_code: str = Field(description="Python code for the answer template")
    last_modified: str = Field(description="ISO timestamp of last modification")
    finished: bool = Field(default=True, description="Whether the template is finished")
    question_rubric: dict[str, Any] | None = Field(default=None, description="Question-specific rubric definition")
    keywords: list[str] | None = Field(default=None, description="Keywords for categorization")
    few_shot_examples: list[dict[str, str]] | None = Field(
        default=None, description="Few-shot examples for this question"
    )


class StartVerificationRequest(BaseModel):
    """Request to start a verification job.

    Contains all the information needed to run verification on a set of questions.
    """

    config: dict[str, Any] = Field(description="Verification configuration (parsed into VerificationConfig)")
    finished_templates: list[FinishedTemplatePayload] = Field(description="List of templates to verify")
    question_ids: list[str] | None = Field(default=None, description="Specific question IDs to verify (all if None)")
    run_name: str | None = Field(default=None, description="User-defined name for this verification run")
    storage_url: str | None = Field(default=None, description="Database URL for auto-saving results")
    benchmark_name: str | None = Field(default=None, description="Benchmark name for auto-saving results")


class StartVerificationResponse(BaseModel):
    """Response when starting a verification job.

    Follows the standard envelope pattern with success/error fields.
    """

    success: bool = Field(default=True, description="Whether the request succeeded")
    job_id: str = Field(description="Unique identifier for the verification job")
    run_name: str = Field(description="Name of the verification run (auto-generated if not provided)")
    status: str = Field(description="Current status of the job")
    message: str = Field(description="Human-readable status message")
    error: str | None = Field(default=None, description="Error message if request failed")


class ComputeSummaryRequest(BaseModel):
    """Request to compute summary statistics for verification results."""

    results: dict[str, Any] = Field(description="Dict of verification results (result_id -> VerificationResult data)")
    run_name: str | None = Field(default=None, description="Optional run name to filter by (null for all results)")


class ComputeSummaryResponse(BaseModel):
    """Response containing summary statistics.

    Wraps the dynamic summary from VerificationResultSet.get_summary() with envelope fields.
    """

    success: bool = Field(default=True, description="Whether the request succeeded")
    error: str | None = Field(default=None, description="Error message if request failed")
    summary: dict[str, Any] = Field(description="Summary statistics from VerificationResultSet.get_summary()")


class CompareModelsRequest(BaseModel):
    """Request to compare multiple models with per-model summaries and heatmap data."""

    results: dict[str, Any] = Field(description="Dict of verification results (result_id -> VerificationResult data)")
    models: list[dict[str, Any]] = Field(description="List of model configs to compare [{answering_model, mcp_config}]")
    parsing_model: str | None = Field(default=None, description="Parsing model to filter by for fair comparison")
    replicate: int | None = Field(default=None, description="Optional replicate number to filter by")


class CompareModelsResponse(BaseModel):
    """Response containing model comparison data.

    Note: Contains complex nested structures for heatmap and token data.
    """

    success: bool = Field(default=True, description="Whether the request succeeded")
    model_summaries: dict[str, Any] = Field(description="Dict mapping model display name to summary stats")
    heatmap_data: list[dict[str, Any]] = Field(description="List of questions with results by model")
    question_token_data: list[dict[str, Any]] = Field(description="Per-question token usage data for bar charts")
    error: str | None = Field(default=None, description="Error message if request failed")
