"""Database management Pydantic models for the Karenina API."""

from typing import Any

from pydantic import BaseModel, Field


class PaginationParams(BaseModel):
    """Pagination query parameters."""

    page: int = Field(default=1, ge=1, description="Page number (1-indexed)")
    page_size: int = Field(default=50, ge=1, le=100, description="Number of items per page (max 100)")


class PaginationMetadata(BaseModel):
    """Pagination metadata included in paginated responses."""

    page: int = Field(description="Current page number (1-indexed)")
    page_size: int = Field(description="Number of items per page")
    total: int = Field(description="Total number of items across all pages")
    total_pages: int = Field(description="Total number of pages")
    has_next: bool = Field(description="Whether there is a next page")
    has_prev: bool = Field(description="Whether there is a previous page")


class DatabaseConnectRequest(BaseModel):
    """Request for connecting to a database."""

    storage_url: str
    create_if_missing: bool = True


class DatabaseConnectResponse(BaseModel):
    """Response for database connection endpoint."""

    success: bool
    storage_url: str
    benchmark_count: int
    message: str
    error: str | None = None


class BenchmarkInfo(BaseModel):
    """Information about a benchmark."""

    id: str
    name: str
    total_questions: int
    finished_count: int
    unfinished_count: int
    last_modified: str | None = None


class BenchmarkListResponse(BaseModel):
    """Response for listing benchmarks with pagination support."""

    success: bool
    benchmarks: list[BenchmarkInfo]
    count: int  # Number of items in current page (for backward compatibility)
    pagination: PaginationMetadata | None = None  # Pagination metadata (optional for backward compat)
    error: str | None = None


class BenchmarkLoadRequest(BaseModel):
    """Request for loading a benchmark."""

    storage_url: str
    benchmark_name: str


class BenchmarkLoadResponse(BaseModel):
    """Response for benchmark load endpoint."""

    success: bool
    benchmark_name: str
    checkpoint_data: dict[str, Any]
    storage_url: str
    message: str
    error: str | None = None


class BenchmarkCreateRequest(BaseModel):
    """Request for creating a new benchmark."""

    storage_url: str
    name: str
    description: str | None = None
    version: str | None = None
    creator: str | None = None


class BenchmarkCreateResponse(BaseModel):
    """Response for benchmark creation endpoint."""

    success: bool
    benchmark_name: str
    checkpoint_data: dict[str, Any]
    storage_url: str
    message: str
    error: str | None = None


class DuplicateQuestionInfo(BaseModel):
    """Information about a duplicate question."""

    question_id: str
    question_text: str
    old_version: dict[str, Any]  # Full question data from database
    new_version: dict[str, Any]  # Full question data from current checkpoint


class BenchmarkSaveRequest(BaseModel):
    """Request for saving a benchmark."""

    storage_url: str
    benchmark_name: str
    checkpoint_data: dict[str, Any]
    detect_duplicates: bool = False  # If True, only detect duplicates without saving


class BenchmarkSaveResponse(BaseModel):
    """Response for benchmark save endpoint."""

    success: bool
    message: str
    last_modified: str | None = None
    duplicates: list[DuplicateQuestionInfo] | None = None  # Present when duplicates detected
    error: str | None = None


class DuplicateResolutionRequest(BaseModel):
    """Request for resolving duplicate questions."""

    storage_url: str
    benchmark_name: str
    checkpoint_data: dict[str, Any]
    resolutions: dict[str, str]  # Map of question_id -> "keep_old" | "keep_new"


class DuplicateResolutionResponse(BaseModel):
    """Response for duplicate resolution endpoint."""

    success: bool
    message: str
    last_modified: str
    kept_old_count: int
    kept_new_count: int
    error: str | None = None


class DatabaseInfo(BaseModel):
    """Information about a database file."""

    name: str
    path: str
    size: int | None = None


class ListDatabasesResponse(BaseModel):
    """Response for listing databases."""

    success: bool
    databases: list[DatabaseInfo]
    db_directory: str
    is_default_directory: bool
    error: str | None = None


class DeleteDatabaseRequest(BaseModel):
    """Request for deleting a database."""

    storage_url: str  # Must be sqlite:/// URL


class DeleteDatabaseResponse(BaseModel):
    """Response for database deletion endpoint."""

    success: bool
    message: str
    error: str | None = None


class DeleteBenchmarkRequest(BaseModel):
    """Request for deleting a benchmark."""

    storage_url: str
    benchmark_name: str


class DeleteBenchmarkResponse(BaseModel):
    """Response for benchmark deletion endpoint."""

    success: bool
    message: str
    deleted_questions: int = 0
    deleted_runs: int = 0
    error: str | None = None


class ImportResultsRequest(BaseModel):
    """Request for importing verification results."""

    storage_url: str
    json_data: dict[str, Any]
    benchmark_name: str
    run_name: str | None = None


class ImportResultsResponse(BaseModel):
    """Response for results import endpoint."""

    success: bool
    run_id: str
    imported_count: int
    message: str
    error: str | None = None


class VerificationRunInfo(BaseModel):
    """Information about a verification run."""

    id: str
    run_name: str | None
    benchmark_id: str
    benchmark_name: str
    status: str
    total_questions: int
    processed_count: int
    successful_count: int
    failed_count: int
    start_time: str | None
    end_time: str | None
    created_at: str
    is_imported: bool = False


class ListVerificationRunsRequest(BaseModel):
    """Request for listing verification runs."""

    storage_url: str
    benchmark_name: str | None = None


class ListVerificationRunsResponse(BaseModel):
    """Response for listing verification runs."""

    success: bool
    runs: list[VerificationRunInfo]
    count: int
    error: str | None = None


class LoadVerificationResultsRequest(BaseModel):
    """Request for loading verification results."""

    storage_url: str
    run_id: str | None = None
    benchmark_name: str | None = None
    question_id: str | None = None
    answering_model: str | None = None
    limit: int | None = None


class VerificationResultSummary(BaseModel):
    """Summary of a verification result."""

    id: int
    run_id: str
    question_id: str
    question_text: str
    answering_model: str
    parsing_model: str
    completed_without_errors: bool
    template_verify_result: Any
    execution_time: float
    timestamp: str


class LoadVerificationResultsResponse(BaseModel):
    """Response for loading verification results."""

    success: bool
    results: list[VerificationResultSummary]
    count: int
    error: str | None = None
