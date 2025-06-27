from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from karenina_server.server import create_fastapi_app


@pytest.fixture
def client():
    """Create test client."""
    # Create a temporary webapp directory for testing
    webapp_dir = Path(__file__).parent.parent / "webapp"
    app = create_fastapi_app(webapp_dir)
    return TestClient(app)


@pytest.fixture
def sample_questions():
    """Sample questions for testing."""
    return {
        "q1": {
            "question": "What is the capital of France?",
            "raw_answer": "Paris is the capital of France.",
            "source": "test",
        },
        "q2": {
            "question": "Is aspirin safe for children?",
            "raw_answer": "Aspirin is generally not recommended for children.",
            "source": "test",
        },
        "q3": {
            "question": "What causes diabetes?",
            "raw_answer": "Diabetes is caused by problems with insulin.",
            "source": "test",
        },
    }


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {"model_provider": "google_genai", "model_name": "gemini-2.0-flash", "temperature": 0.1}


@patch("karenina_server.server.LLM_AVAILABLE", True)
@patch("karenina_server.services.generation_service.generation_service")
def test_selective_template_generation(mock_service, client, sample_questions, sample_config):
    """Test generating templates for selected questions only."""
    # Mock the service to return a job ID
    mock_service.start_generation.return_value = "test-job-123"

    # Select only 2 out of 3 questions
    selected_questions = {"q1": sample_questions["q1"], "q3": sample_questions["q3"]}

    # Make request with selected questions
    response = client.post(
        "/api/generate-answer-templates", json={"questions": selected_questions, "config": sample_config}
    )

    # Verify response
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == "test-job-123"
    assert data["status"] == "started"
    assert "2 questions" in data["message"]

    # Verify service was called with correct parameters
    mock_service.start_generation.assert_called_once()
    call_args = mock_service.start_generation.call_args

    # Check that only selected questions were passed
    assert len(call_args.kwargs["questions_data"]) == 2
    assert "q1" in call_args.kwargs["questions_data"]
    assert "q3" in call_args.kwargs["questions_data"]
    assert "q2" not in call_args.kwargs["questions_data"]

    # Check config was passed correctly
    config = call_args.kwargs["config"]
    assert config.model_provider == "google_genai"
    assert config.model_name == "gemini-2.0-flash"
    assert config.temperature == 0.1


@patch("karenina_server.server.LLM_AVAILABLE", True)
@patch("karenina_server.services.generation_service.generation_service")
def test_progress_tracking_with_selection(mock_service, client):
    """Test progress tracking returns correct information for selected questions."""
    # Mock progress data
    mock_progress = {
        "job_id": "test-job-123",
        "status": "running",
        "total_questions": 2,  # Only 2 selected questions
        "processed_count": 1,
        "successful_count": 1,
        "failed_count": 0,
        "percentage": 50.0,
        "current_question": "What causes diabetes?...",
        "estimated_time_remaining": 5.2,
        "error_message": None,
        "start_time": 1234567890.0,
        "end_time": None,
    }

    mock_service.get_progress.return_value = mock_progress
    mock_service.jobs = {}  # No completed job yet

    # Get progress
    response = client.get("/api/generation-progress/test-job-123")

    # Verify response
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == "test-job-123"
    assert data["status"] == "running"
    assert data["percentage"] == 50.0
    assert data["processed_count"] == 1
    assert data["total_count"] == 2
    assert data["current_question"] == "What causes diabetes?..."
    assert data["estimated_time_remaining"] == 5.2


@patch("karenina_server.server.LLM_AVAILABLE", True)
@patch("karenina_server.services.generation_service.generation_service")
def test_completed_generation_with_results(mock_service, client):
    """Test completed generation returns correct results for selected questions."""
    # Mock completed job with results
    mock_job = Mock()
    mock_job.status = "completed"
    mock_job.result = {
        "templates": {
            "q1": {
                "success": True,
                "template_code": "class Answer(BaseAnswer):\n    capital: str = Field(...)",
                "error": None,
            },
            "q3": {
                "success": True,
                "template_code": "class Answer(BaseAnswer):\n    cause: str = Field(...)",
                "error": None,
            },
        },
        "total_templates": 2,
        "successful_generations": 2,
        "failed_generations": 0,
        "average_generation_time": 1.5,
        "model_info": {"name": "gemini-2.0-flash", "provider": "google_genai", "temperature": 0.1},
    }

    mock_progress = {
        "job_id": "test-job-123",
        "status": "completed",
        "total_questions": 2,
        "processed_count": 2,
        "successful_count": 2,
        "failed_count": 0,
        "percentage": 100.0,
        "current_question": "",
        "estimated_time_remaining": None,
        "error_message": None,
        "start_time": 1234567890.0,
        "end_time": 1234567892.0,
    }

    mock_service.get_progress.return_value = mock_progress
    mock_service.jobs = {"test-job-123": mock_job}

    # Get progress for completed job
    response = client.get("/api/generation-progress/test-job-123")

    # Verify response
    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == "test-job-123"
    assert data["status"] == "completed"
    assert data["percentage"] == 100.0
    assert data["processed_count"] == 2
    assert data["total_count"] == 2

    # Verify result is included
    assert data["result"] is not None
    result = data["result"]
    assert result["total_templates"] == 2
    assert result["successful_generations"] == 2
    assert result["failed_generations"] == 0
    assert len(result["templates"]) == 2
    assert "q1" in result["templates"]
    assert "q3" in result["templates"]


@patch("karenina_server.server.LLM_AVAILABLE", True)
@patch("karenina_server.services.generation_service.generation_service")
def test_empty_selection_error(mock_service, client, sample_config):
    """Test that empty question selection returns an error."""
    # Mock the service to raise an exception for empty questions
    mock_service.start_generation.side_effect = ValueError("No questions provided")

    # Try to generate with empty questions
    response = client.post(
        "/api/generate-answer-templates",
        json={
            "questions": {},  # Empty selection
            "config": sample_config,
        },
    )

    # Should return an error for empty selection
    assert response.status_code == 500
    assert "error" in response.json() or "detail" in response.json()


@patch("karenina_server.server.LLM_AVAILABLE", True)
@patch("karenina_server.services.generation_service.generation_service")
def test_single_question_selection(mock_service, client, sample_questions, sample_config):
    """Test generating templates for a single selected question."""
    mock_service.start_generation.return_value = "single-job-456"

    # Select only one question
    selected_questions = {"q2": sample_questions["q2"]}

    response = client.post(
        "/api/generate-answer-templates", json={"questions": selected_questions, "config": sample_config}
    )

    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == "single-job-456"
    assert "1 questions" in data["message"]

    # Verify only one question was passed
    call_args = mock_service.start_generation.call_args
    assert len(call_args.kwargs["questions_data"]) == 1
    assert "q2" in call_args.kwargs["questions_data"]


@patch("karenina_server.server.LLM_AVAILABLE", True)
@patch("karenina_server.services.generation_service.generation_service")
def test_all_questions_selection(mock_service, client, sample_questions, sample_config):
    """Test generating templates for all questions (default behavior)."""
    mock_service.start_generation.return_value = "all-job-789"

    response = client.post(
        "/api/generate-answer-templates",
        json={
            "questions": sample_questions,  # All questions
            "config": sample_config,
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["job_id"] == "all-job-789"
    assert "3 questions" in data["message"]

    # Verify all questions were passed
    call_args = mock_service.start_generation.call_args
    assert len(call_args.kwargs["questions_data"]) == 3
    assert "q1" in call_args.kwargs["questions_data"]
    assert "q2" in call_args.kwargs["questions_data"]
    assert "q3" in call_args.kwargs["questions_data"]
