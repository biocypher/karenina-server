"""Template generation Pydantic models for the Karenina API."""

from typing import Any

from pydantic import BaseModel, SecretStr


class TemplateGenerationConfig(BaseModel):
    """Configuration for template generation."""

    model_provider: str
    model_name: str
    temperature: float = 0.1
    interface: str = "langchain"
    endpoint_base_url: str | None = None
    endpoint_api_key: SecretStr | None = None


class TemplateGenerationRequest(BaseModel):
    """Request for starting template generation."""

    questions: dict[str, Any]
    config: TemplateGenerationConfig
    force_regenerate: bool = False


class TemplateGenerationResponse(BaseModel):
    """Response for template generation start."""

    job_id: str
    status: str
    message: str


class TemplateGenerationStatusResponse(BaseModel):
    """Response for template generation status."""

    job_id: str
    status: str
    percentage: float
    current_question: str
    processed_count: int
    total_count: int
    duration_seconds: float | None = None
    last_task_duration: float | None = None
    error: str | None = None
    result: dict[str, Any] | None = None
    in_progress_questions: list[str] = []
