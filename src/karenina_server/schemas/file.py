"""File-related Pydantic models for the Karenina API."""

from typing import Any

from pydantic import BaseModel


class FilePreviewResponse(BaseModel):
    """Response for file preview endpoint."""

    success: bool
    total_rows: int | None = None
    columns: list[str] | None = None
    preview_rows: int | None = None
    data: list[dict[str, Any]] | None = None
    error: str | None = None


class KeywordColumnConfig(BaseModel):
    """Configuration for a keyword column with its separator."""

    column: str
    separator: str


class ExtractQuestionsRequest(BaseModel):
    """Request for extracting questions from a file."""

    file_id: str
    question_column: str
    answer_column: str
    sheet_name: str | None = None
    # Optional metadata column mappings
    author_name_column: str | None = None
    author_email_column: str | None = None
    author_affiliation_column: str | None = None
    url_column: str | None = None
    keywords_columns: list[dict[str, str]] | None = None


class ExtractQuestionsResponse(BaseModel):
    """Response for question extraction endpoint."""

    success: bool
    questions_count: int | None = None
    questions_data: dict[str, Any] | None = None
    error: str | None = None
