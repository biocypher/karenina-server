"""Integration tests for template generation API handlers.

Uses TestClient to test API endpoints for template generation.
"""

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from karenina_server.server import create_fastapi_app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    webapp_dir = Path(__file__).parent.parent.parent.parent / "src" / "karenina_server" / "webapp"
    app = create_fastapi_app(webapp_dir)
    return TestClient(app)


@pytest.fixture
def sample_questions():
    """Sample questions data for testing."""
    return {
        "test_1": {"question": "Is rofecoxib withdrawn?", "raw_answer": "Yes"},
        "test_2": {"question": "What is the capital of France?", "raw_answer": "Paris"},
    }


@pytest.mark.integration
@pytest.mark.api
class TestTemplateGenerationAPI:
    """Test cases for template generation API endpoints."""

    @patch("karenina_server.server.LLM_AVAILABLE", True)
    @patch("karenina_server.services.generation_service.generation_service")
    def test_start_generation_success(self, mock_service, client, sample_questions):
        """Test successful template generation start."""
        mock_service.start_generation.return_value = "test-job-id"

        response = client.post(
            "/api/generate-answer-templates",
            json={
                "questions": sample_questions,
                "config": {
                    "model_name": "gemini-2.0-flash",
                    "model_provider": "google_genai",
                    "temperature": 0,
                    "interface": "langchain",
                },
            },
        )

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "test-job-id"
        assert data["status"] == "started"
        assert "2 questions" in data["message"]

    @patch("karenina_server.server.LLM_AVAILABLE", False)
    def test_start_generation_llm_unavailable(self, client, sample_questions):
        """Test template generation when LLM is unavailable."""
        response = client.post(
            "/api/generate-answer-templates",
            json={
                "questions": sample_questions,
                "config": {
                    "model_name": "gemini-2.0-flash",
                    "model_provider": "google_genai",
                    "temperature": 0,
                    "interface": "langchain",
                },
            },
        )

        assert response.status_code == 503
        assert "LLM functionality not available" in response.json()["detail"]

    @patch("karenina_server.server.LLM_AVAILABLE", True)
    @patch("karenina_server.services.generation_service.generation_service")
    def test_get_progress_success(self, mock_service, client):
        """Test successful progress retrieval."""
        mock_status = {
            "job_id": "test-job-id",
            "status": "running",
            "total_questions": 2,
            "processed_count": 1,
            "current_question": "test_1",
            "estimated_time_remaining": 30,
            "error_message": None,
            "start_time": 1234567890,
            "last_update": 1234567891,
        }
        mock_service.get_progress.return_value = mock_status

        response = client.get("/api/generation-progress/test-job-id")

        assert response.status_code == 200
        data = response.json()
        assert data["job_id"] == "test-job-id"
        assert data["status"] == "running"

    @patch("karenina_server.server.LLM_AVAILABLE", True)
    @patch("karenina_server.services.generation_service.generation_service")
    def test_get_progress_job_not_found(self, mock_service, client):
        """Test progress retrieval for non-existent job."""
        mock_service.get_progress.return_value = None

        response = client.get("/api/generation-progress/nonexistent-job")

        assert response.status_code == 404
        assert "Job not found" in response.json()["detail"]

    @patch("karenina_server.server.LLM_AVAILABLE", True)
    @patch("karenina_server.services.generation_service.generation_service")
    def test_cancel_generation_success(self, mock_service, client):
        """Test successful job cancellation."""
        mock_service.cancel_job.return_value = True

        response = client.post("/api/cancel-generation/test-job-id")

        assert response.status_code == 200
        assert "Job cancelled successfully" in response.json()["message"]

    @patch("karenina_server.server.LLM_AVAILABLE", True)
    @patch("karenina_server.services.generation_service.generation_service")
    def test_cancel_generation_job_not_found(self, mock_service, client):
        """Test cancellation of non-existent job."""
        mock_service.cancel_job.return_value = False

        response = client.post("/api/cancel-generation/nonexistent-job")

        assert response.status_code == 404
        assert "Job not found or cannot be cancelled" in response.json()["detail"]

    def test_invalid_request_data(self, client):
        """Test API with invalid request data."""
        response = client.post(
            "/api/generate-answer-templates",
            json={
                "config": {"model_name": "gemini-2.0-flash", "model_provider": "google_genai", "interface": "langchain"}
            },
        )

        assert response.status_code == 422
