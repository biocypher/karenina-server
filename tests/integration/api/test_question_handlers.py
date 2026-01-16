"""Integration tests for question selector API handlers.

Uses TestClient to test API endpoints for question selection.
"""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from karenina_server.server import create_fastapi_app


@pytest.fixture
def client():
    """Create test client."""
    webapp_dir = Path(__file__).parent.parent.parent.parent / "src" / "karenina_server" / "webapp"
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


@pytest.mark.integration
@pytest.mark.api
class TestSelectiveTemplateGeneration:
    """Test selective template generation endpoints."""

    @patch("karenina_server.server.LLM_AVAILABLE", True)
    @patch("karenina_server.services.generation_service.generation_service")
    def test_selective_template_generation(self, mock_service, client, sample_questions, sample_config):
        """Test generating templates for selected questions only."""
        mock_service.start_generation.return_value = "test-job-123"

        selected_questions = {"q1": sample_questions["q1"], "q3": sample_questions["q3"]}

        response = client.post(
            "/api/generate-answer-templates", json={"questions": selected_questions, "config": sample_config}
        )

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "test-job-123"
        assert data["status"] == "started"
        assert "2 questions" in data["message"]

    @patch("karenina_server.server.LLM_AVAILABLE", True)
    @patch("karenina_server.services.generation_service.generation_service")
    def test_progress_tracking_with_selection(self, mock_service, client):
        """Test progress tracking returns correct information for selected questions."""
        mock_progress = {
            "job_id": "test-job-123",
            "status": "running",
            "total_questions": 2,
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
        mock_service.jobs = {}

        response = client.get("/api/generation-progress/test-job-123")

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "test-job-123"
        assert data["status"] == "running"
        assert data["percentage"] == 50.0

    @patch("karenina_server.server.LLM_AVAILABLE", True)
    @patch("karenina_server.services.generation_service.generation_service")
    def test_completed_generation_with_results(self, mock_service, client):
        """Test completed generation returns correct results for selected questions."""
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

        response = client.get("/api/generation-progress/test-job-123")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "completed"
        assert data["result"]["total_templates"] == 2

    @patch("karenina_server.server.LLM_AVAILABLE", True)
    @patch("karenina_server.services.generation_service.generation_service")
    def test_single_question_selection(self, mock_service, client, sample_questions, sample_config):
        """Test generating templates for a single selected question."""
        mock_service.start_generation.return_value = "single-job-456"

        selected_questions = {"q2": sample_questions["q2"]}

        response = client.post(
            "/api/generate-answer-templates", json={"questions": selected_questions, "config": sample_config}
        )

        assert response.status_code == 200
        data = response.json()
        assert "1 questions" in data["message"]
