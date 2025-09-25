"""Tests for rubric API endpoints."""

from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from fastapi.testclient import TestClient

from karenina_server.server import create_fastapi_app


@pytest.fixture
def client():
    """Create a test client for the FastAPI app."""
    webapp_dir = Path(__file__).parent.parent / "webapp"
    app = create_fastapi_app(webapp_dir)
    return TestClient(app)


@pytest.fixture
def sample_rubric_data():
    """Sample rubric data for testing."""
    return {
        "traits": [
            {
                "name": "accuracy",
                "description": "Is the response factually accurate?",
                "kind": "boolean",
                "min_score": None,
                "max_score": None,
            },
            {
                "name": "completeness",
                "description": "How complete is the response?",
                "kind": "score",
                "min_score": 1,
                "max_score": 5,
            },
        ]
    }


@pytest.fixture
def sample_trait_generation_request():
    """Sample trait generation request data."""
    return {
        "questions": {
            "q1": {"question": "What is the capital of France?", "raw_answer": "Paris", "tags": []},
            "q2": {"question": "Explain photosynthesis.", "raw_answer": "Process plants use to make food", "tags": []},
        },
        "system_prompt": "Generate evaluation criteria for these questions.",
        "user_suggestions": ["clarity", "accuracy"],
        "config": {
            "model_provider": "google_genai",
            "model_name": "gemini-2.0-flash",
            "temperature": 0.1,
            "interface": "langchain",
        },
    }


class TestRubricCRUD:
    """Test CRUD operations for rubrics."""

    def test_create_rubric_success(self, client, sample_rubric_data):
        """Test successful rubric creation."""
        response = client.post("/api/rubric", json=sample_rubric_data)

        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Rubric saved successfully"

    def test_create_rubric_invalid_trait_name(self, client, sample_rubric_data):
        """Test rubric creation with invalid trait name."""
        sample_rubric_data["traits"][0]["name"] = ""

        response = client.post("/api/rubric", json=sample_rubric_data)

        assert response.status_code == 422
        detail = response.json()["detail"]
        assert isinstance(detail, list) and len(detail) > 0

    def test_create_rubric_missing_traits(self, client):
        """Test rubric creation with missing traits."""
        response = client.post("/api/rubric", json={})

        assert response.status_code == 400
        assert "must have at least one trait" in response.json()["detail"]

    def test_create_rubric_duplicate_trait_names(self, client, sample_rubric_data):
        """Test rubric creation with duplicate trait names."""
        # Make both traits have the same name
        sample_rubric_data["traits"][1]["name"] = sample_rubric_data["traits"][0]["name"]

        response = client.post("/api/rubric", json=sample_rubric_data)

        assert response.status_code == 400
        assert "must be unique" in response.json()["detail"]

    def test_get_rubric_none_exists(self, client):
        """Test getting rubric when none exists."""
        # Clear any existing rubric first
        client.delete("/api/rubric")

        response = client.get("/api/rubric")

        assert response.status_code == 200
        assert response.json() is None

    def test_get_rubric_after_create(self, client, sample_rubric_data):
        """Test getting rubric after creation."""
        # First create a rubric
        create_response = client.post("/api/rubric", json=sample_rubric_data)
        assert create_response.status_code == 200

        # Then get it
        get_response = client.get("/api/rubric")
        assert get_response.status_code == 200

        rubric_data = get_response.json()
        assert rubric_data is not None
        assert "traits" in rubric_data
        assert len(rubric_data["traits"]) == 2

    def test_update_rubric(self, client, sample_rubric_data):
        """Test updating an existing rubric."""
        # Create initial rubric
        client.post("/api/rubric", json=sample_rubric_data)

        # Update with different data
        updated_data = {
            "traits": [
                {
                    "name": "clarity",
                    "description": "Is the response clear?",
                    "kind": "boolean",
                }
            ]
        }

        response = client.post("/api/rubric", json=updated_data)
        assert response.status_code == 200

        # Verify the update
        get_response = client.get("/api/rubric")
        rubric_data = get_response.json()
        assert len(rubric_data["traits"]) == 1
        assert rubric_data["traits"][0]["name"] == "clarity"

    def test_delete_rubric(self, client, sample_rubric_data):
        """Test deleting a rubric."""
        # Create rubric first
        client.post("/api/rubric", json=sample_rubric_data)

        # Delete it
        delete_response = client.delete("/api/rubric")
        assert delete_response.status_code == 200
        assert delete_response.json()["message"] == "Rubric deleted successfully"

        # Verify it's gone
        get_response = client.get("/api/rubric")
        assert get_response.json() is None

    def test_delete_nonexistent_rubric(self, client):
        """Test deleting when no rubric exists."""
        response = client.delete("/api/rubric")
        assert response.status_code == 200
        assert response.json()["message"] == "Rubric deleted successfully"


class TestRubricValidation:
    """Test rubric validation logic."""

    def test_rubric_trait_validation(self, client):
        """Test validation of individual rubric traits."""
        invalid_trait_data = {
            "traits": [
                {
                    "name": "",  # Empty name should be caught by Pydantic
                    "description": "Valid description",
                    "kind": "boolean",
                }
            ]
        }

        response = client.post("/api/rubric", json=invalid_trait_data)

        # Should get validation error from Pydantic
        assert response.status_code == 422

    def test_rubric_invalid_trait_kind(self, client):
        """Test validation of invalid trait kind."""
        invalid_kind_data = {
            "traits": [
                {
                    "name": "test_trait",
                    "description": "Test description",
                    "kind": "invalid_kind",  # Should be 'boolean' or 'score'
                }
            ]
        }

        response = client.post("/api/rubric", json=invalid_kind_data)

        # Should get validation error
        assert response.status_code == 422


class TestRubricIntegration:
    """Test rubric integration scenarios."""

    def test_rubric_system_prompt_endpoint(self, client):
        """Test the default system prompt endpoint."""
        response = client.get("/api/rubric/default-system-prompt")

        assert response.status_code == 200
        data = response.json()
        assert "prompt" in data
        assert isinstance(data["prompt"], str)
        assert len(data["prompt"]) > 0
        # Check for key content in the prompt
        assert "rubric design" in data["prompt"].lower()
        assert "qualitative aspects" in data["prompt"].lower()


class TestRubricTraitGeneration:
    """Test rubric trait generation endpoints."""

    @patch("karenina_server.api.rubric_handlers.GenerationService")
    def test_generate_traits_success(self, mock_service_class, client, sample_trait_generation_request):
        """Test successful trait generation."""
        # Mock the generation service
        mock_service = Mock()
        mock_service_class.return_value = mock_service

        # Mock successful generation response
        mock_service.generate_rubric_traits.return_value = """
        [
            {
                "name": "accuracy",
                "description": "Is the response factually accurate?",
                "kind": "boolean"
            },
            {
                "name": "completeness",
                "description": "How complete is the response?",
                "kind": "score",
                "min_score": 1,
                "max_score": 5
            }
        ]
        """

        response = client.post("/api/generate-rubric-traits", json=sample_trait_generation_request)

        assert response.status_code == 200
        data = response.json()
        assert "traits" in data
        assert len(data["traits"]) == 2
        assert data["traits"][0]["name"] == "accuracy"
        assert data["traits"][1]["name"] == "completeness"

    def test_generate_traits_no_questions(self, client, sample_trait_generation_request):
        """Test trait generation with no questions."""
        sample_trait_generation_request["questions"] = {}

        response = client.post("/api/generate-rubric-traits", json=sample_trait_generation_request)

        assert response.status_code == 400
        assert "No questions provided" in response.json()["detail"]

    @patch("karenina_server.api.rubric_handlers.GenerationService")
    def test_generate_traits_fallback_parsing(self, mock_service_class, client, sample_trait_generation_request):
        """Test trait generation with response that triggers fallback parsing."""
        # Mock the generation service
        mock_service = Mock()
        mock_service_class.return_value = mock_service

        # Mock response that will trigger fallback traits
        mock_service.generate_rubric_traits.return_value = "This is not valid JSON"

        response = client.post("/api/generate-rubric-traits", json=sample_trait_generation_request)

        assert response.status_code == 200
        data = response.json()
        # Should get fallback traits when parsing fails
        assert "traits" in data
        assert len(data["traits"]) == 2
        assert data["traits"][0]["name"] == "clarity"
