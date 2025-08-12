"""File upload, preview and extraction API handlers."""

import json
import tempfile
import time
import uuid
from pathlib import Path
from typing import Any

from fastapi import File, Form, HTTPException, UploadFile
from fastapi.responses import FileResponse

try:
    from karenina.questions.extractor import extract_and_generate_questions, get_file_preview

    EXTRACTOR_AVAILABLE = True
except ImportError:
    EXTRACTOR_AVAILABLE = False

try:
    from karenina.llm.manual_traces import ManualTraceError, get_manual_trace_count, load_manual_traces

    MANUAL_TRACES_AVAILABLE = True
except ImportError:
    MANUAL_TRACES_AVAILABLE = False

# Global storage for uploaded files (in production, use a proper database)
uploaded_files = {}


def generate_python_questions_file(questions_data: dict[str, Any]) -> str:
    """Generate Python file content from questions data."""

    # Header
    content = '''"""Auto-generated questions from extracted data."""

from karenina.schemas.question_class import Question

# Auto-generated questions

'''

    # Generate individual question objects
    question_objects = []
    for i, (_question_id, question_data) in enumerate(questions_data.items(), 1):
        # Create question object (ID auto-generated from question text)
        question_var = f"question_{i}"
        question_objects.append(question_var)

        # Escape strings for Python
        question_text = repr(question_data.get("question", ""))
        raw_answer = repr(question_data.get("raw_answer", ""))
        tags = question_data.get("tags", [])

        content += f"""{question_var} = Question(
    question={question_text},
    raw_answer={raw_answer},
    tags={tags}
)

"""

    # Add list of all questions
    newline_indent = ",\n    "
    content += f"""# List of all questions
all_questions = [
    {newline_indent.join(question_objects)},
]
"""

    return content


def register_file_routes(
    app: Any, FilePreviewResponse: Any, ExtractQuestionsRequest: Any, ExtractQuestionsResponse: Any
) -> None:
    """Register file-related routes."""

    @app.post("/api/upload-file")  # type: ignore[misc]
    async def upload_file_endpoint(file: UploadFile = File(...)) -> dict[str, Any]:
        """Upload a file for question extraction."""
        if not EXTRACTOR_AVAILABLE:
            raise HTTPException(status_code=500, detail="Question extractor not available")

        try:
            # Generate unique file ID
            file_id = str(uuid.uuid4())

            # Create temporary file
            temp_dir = Path(tempfile.gettempdir()) / "otarbench_uploads"
            temp_dir.mkdir(exist_ok=True)

            file_extension = Path(file.filename).suffix if file.filename else ""
            temp_file_path = temp_dir / f"{file_id}{file_extension}"

            # Save uploaded file
            with open(temp_file_path, "wb") as buffer:
                content = await file.read()
                buffer.write(content)

            # Store file info
            uploaded_files[file_id] = {
                "original_name": file.filename,
                "file_path": str(temp_file_path),
                "content_type": file.content_type,
                "size": len(content),
            }

            return {"success": True, "file_id": file_id, "filename": file.filename, "size": len(content)}

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error uploading file: {e!s}") from e

    @app.post("/api/preview-file", response_model=FilePreviewResponse)  # type: ignore[misc]
    async def preview_file_endpoint(
        file_id: str = Form(...), sheet_name: str | None = Form(None)
    ) -> FilePreviewResponse:
        """Get a preview of the uploaded file."""
        if not EXTRACTOR_AVAILABLE:
            raise HTTPException(status_code=500, detail="Question extractor not available")

        if file_id not in uploaded_files:
            raise HTTPException(status_code=404, detail="File not found")

        try:
            file_info = uploaded_files[file_id]
            preview_data = get_file_preview(file_info["file_path"], sheet_name)
            return FilePreviewResponse(**preview_data)

        except Exception as e:
            return FilePreviewResponse(success=False, error=f"Error previewing file: {e!s}")

    @app.post("/api/extract-questions", response_model=ExtractQuestionsResponse)  # type: ignore[misc]
    async def extract_questions_endpoint(request: ExtractQuestionsRequest) -> ExtractQuestionsResponse:
        """Extract questions from the uploaded file."""
        if not EXTRACTOR_AVAILABLE:
            raise HTTPException(status_code=500, detail="Question extractor not available")

        if request.file_id not in uploaded_files:
            raise HTTPException(status_code=404, detail="File not found")

        try:
            file_info = uploaded_files[request.file_id]

            # Extract questions and return as JSON
            questions_data = extract_and_generate_questions(
                file_path=file_info["file_path"],
                output_path="",  # Not used when return_json=True
                question_column=request.question_column,
                answer_column=request.answer_column,
                sheet_name=request.sheet_name,
                return_json=True,
            )

            return ExtractQuestionsResponse(
                success=True,
                questions_count=len(questions_data) if questions_data else 0,
                questions_data=questions_data,
            )

        except Exception as e:
            return ExtractQuestionsResponse(success=False, error=f"Error extracting questions: {e!s}")

    @app.post("/api/export-questions-python")  # type: ignore[misc]
    async def export_questions_python_endpoint(request: dict[str, Any]) -> FileResponse:
        """Export questions as a Python file."""
        try:
            questions_data = request.get("questions", {})

            if not questions_data:
                raise HTTPException(status_code=400, detail="No questions data provided")

            # Generate Python file content
            python_content = generate_python_questions_file(questions_data)

            # Create temporary file
            temp_dir = Path(tempfile.gettempdir()) / "otarbench_exports"
            temp_dir.mkdir(exist_ok=True)

            export_id = str(uuid.uuid4())
            python_file_path = temp_dir / f"questions_{export_id}.py"

            # Write Python file
            with open(python_file_path, "w", encoding="utf-8") as f:
                f.write(python_content)

            # Return file for download
            return FileResponse(
                path=python_file_path, filename=f"questions_{int(time.time())}.py", media_type="text/x-python"
            )

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error exporting Python file: {e!s}") from e

    @app.delete("/api/uploaded-files/{file_id}")  # type: ignore[misc]
    async def delete_uploaded_file_endpoint(file_id: str) -> dict[str, str]:
        """Delete an uploaded file."""
        if file_id not in uploaded_files:
            raise HTTPException(status_code=404, detail="File not found")

        try:
            file_info = uploaded_files[file_id]
            file_path_str = file_info.get("file_path")
            if not file_path_str:
                raise HTTPException(status_code=400, detail="File path not found")
            file_path = Path(str(file_path_str))

            # Delete the file if it exists
            if file_path.exists():
                file_path.unlink()

            # Remove from storage
            del uploaded_files[file_id]

            return {"message": f"File {file_id} deleted successfully"}

        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error deleting file: {e!s}") from e

    @app.post("/api/upload-manual-traces")  # type: ignore[misc]
    async def upload_manual_traces_endpoint(file: UploadFile = File(...)) -> dict[str, Any]:
        """Upload manual traces JSON file."""
        if not MANUAL_TRACES_AVAILABLE:
            raise HTTPException(status_code=500, detail="Manual traces functionality not available")

        try:
            # Read file content
            content = await file.read()

            # Parse JSON
            try:
                json_data = json.loads(content)
            except json.JSONDecodeError as e:
                raise HTTPException(status_code=400, detail=f"Invalid JSON format: {e}") from e

            # Load traces into the trace manager
            load_manual_traces(json_data)

            # Get count for response
            trace_count = get_manual_trace_count()

            return {
                "success": True,
                "message": f"Successfully loaded {trace_count} manual traces",
                "trace_count": trace_count,
                "filename": file.filename,
            }

        except ManualTraceError as e:
            raise HTTPException(status_code=400, detail=f"Manual trace validation error: {e}") from e
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Error uploading manual traces: {e}") from e

    @app.get("/api/manual-traces/status")  # type: ignore[misc]
    async def get_manual_traces_status() -> dict[str, Any]:
        """Get the status of loaded manual traces."""
        if not MANUAL_TRACES_AVAILABLE:
            raise HTTPException(status_code=500, detail="Manual traces functionality not available")

        trace_count = get_manual_trace_count()

        return {"loaded": trace_count > 0, "trace_count": trace_count}
