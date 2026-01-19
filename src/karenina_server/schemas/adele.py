"""ADeLe classification endpoint Pydantic models for the Karenina API."""

from typing import Literal

from pydantic import BaseModel, Field


class AdeleModelConfig(BaseModel):
    """Configuration for the model used in ADeLe classification."""

    interface: Literal["langchain", "openrouter", "openai_endpoint"] = Field(
        default="langchain",
        description="The interface to use for model initialization",
    )
    provider: str = Field(
        default="anthropic",
        description="Model provider (e.g., 'anthropic', 'openai', 'google_genai')",
    )
    model_name: str = Field(
        default="claude-3-5-haiku-latest",
        description="Model name to use for classification",
    )
    temperature: float = Field(
        default=0.0,
        ge=0.0,
        le=2.0,
        description="Temperature for LLM calls (0.0 for deterministic)",
    )
    endpoint_base_url: str | None = Field(
        default=None,
        description="Custom base URL for openai_endpoint interface",
    )
    endpoint_api_key: str | None = Field(
        default=None,
        description="API key for openai_endpoint interface",
    )
    trait_eval_mode: Literal["batch", "sequential"] = Field(
        default="batch",
        description=(
            "How to evaluate traits for each question. "
            "'batch' evaluates all traits in one LLM call (faster, cheaper). "
            "'sequential' evaluates each trait separately (potentially more accurate)."
        ),
    )


class AdeleTraitInfoResponse(BaseModel):
    """Information about a single ADeLe trait."""

    name: str = Field(description="Snake_case trait name (e.g., 'attention_and_scan')")
    code: str = Field(description="Original ADeLe code (e.g., 'AS')")
    description: str | None = Field(default=None, description="Trait description/header from the rubric")
    classes: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping from class name to class description",
    )
    class_names: list[str] = Field(
        default_factory=list,
        description="Ordered list of class names (from level 0 to 5)",
    )


class ListAdeleTraitsResponse(BaseModel):
    """Response for listing available ADeLe traits."""

    success: bool = Field(default=True, description="Whether the request succeeded")
    traits: list[AdeleTraitInfoResponse] = Field(description="List of available ADeLe traits")
    count: int = Field(description="Total number of traits available")
    error: str | None = Field(default=None, description="Error message if request failed")


class ClassifySingleQuestionRequest(BaseModel):
    """Request to classify a single question using ADeLe dimensions."""

    question_text: str = Field(description="The question text to classify")
    question_id: str | None = Field(default=None, description="Optional question identifier")
    trait_names: list[str] | None = Field(
        default=None,
        description="List of ADeLe trait names to evaluate. If None, evaluates all 18 traits.",
    )
    llm_config: AdeleModelConfig | None = Field(
        default=None,
        description="Optional model configuration. If None, uses default settings.",
    )


class ClassificationResultPayload(BaseModel):
    """Classification result for a single question."""

    question_id: str | None = Field(default=None, description="Question identifier")
    question_text: str = Field(description="The question text that was classified")
    scores: dict[str, int] = Field(
        default_factory=dict,
        description="Mapping from trait name to integer score (0-5). -1 indicates error.",
    )
    labels: dict[str, str] = Field(
        default_factory=dict,
        description="Mapping from trait name to class label",
    )
    model: str = Field(default="unknown", description="Model used for classification")
    classified_at: str = Field(default="", description="ISO timestamp of classification")


class ClassifySingleQuestionResponse(BaseModel):
    """Response for single question classification."""

    success: bool = Field(default=True, description="Whether the request succeeded")
    result: ClassificationResultPayload | None = Field(default=None, description="Classification result")
    error: str | None = Field(default=None, description="Error message if request failed")


class ClassifyBatchRequest(BaseModel):
    """Request to classify multiple questions using ADeLe dimensions."""

    questions: list[dict[str, str]] = Field(
        description="List of questions to classify. Each dict must have 'question_id' and 'question_text' keys."
    )
    trait_names: list[str] | None = Field(
        default=None,
        description="List of ADeLe trait names to evaluate. If None, evaluates all 18 traits.",
    )
    llm_config: AdeleModelConfig | None = Field(
        default=None,
        description="Optional model configuration. If None, uses default settings.",
    )


class StartClassifyBatchResponse(BaseModel):
    """Response when starting a batch classification job."""

    success: bool = Field(default=True, description="Whether the request succeeded")
    job_id: str = Field(description="Unique identifier for the classification job")
    status: str = Field(description="Current status of the job")
    message: str = Field(description="Human-readable status message")
    total_questions: int = Field(description="Total number of questions to classify")
    error: str | None = Field(default=None, description="Error message if request failed")


class ClassifyBatchProgressResponse(BaseModel):
    """Response for batch classification progress query."""

    success: bool = Field(default=True, description="Whether the request succeeded")
    job_id: str = Field(description="Job identifier")
    status: str = Field(description="Current status (pending, running, completed, failed, cancelled)")
    progress: float = Field(description="Progress percentage (0-100)")
    completed: int = Field(description="Number of questions classified")
    total: int = Field(description="Total questions to classify")
    message: str = Field(description="Human-readable status message")
    error: str | None = Field(default=None, description="Error message if job failed")


class ClassifyBatchResultsResponse(BaseModel):
    """Response containing batch classification results."""

    success: bool = Field(default=True, description="Whether the request succeeded")
    job_id: str = Field(description="Job identifier")
    status: str = Field(description="Job status")
    results: list[ClassificationResultPayload] = Field(
        default_factory=list, description="List of classification results"
    )
    error: str | None = Field(default=None, description="Error message if request failed")


class ClassifyBatchProgressMessage(BaseModel):
    """WebSocket message for batch classification progress updates."""

    type: str = Field(default="progress", description="Message type")
    job_id: str = Field(description="Job identifier")
    status: str = Field(description="Current status")
    progress: float = Field(description="Progress percentage (0-100)")
    completed: int = Field(description="Number of questions classified")
    total: int = Field(description="Total questions to classify")
    current_question_id: str | None = Field(default=None, description="Currently processing question")
    message: str = Field(description="Human-readable status message")


class UpdateQuestionMetadataRequest(BaseModel):
    """Request to update a question's custom_metadata with classification results."""

    question_id: str = Field(description="Question identifier")
    classification: ClassificationResultPayload = Field(description="Classification result to save")
