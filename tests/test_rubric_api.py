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

    def test_create_rubric_empty_traits(self, client, sample_rubric_data):
        """Test rubric creation with empty traits list."""
        sample_rubric_data["traits"] = []

        response = client.post("/api/rubric", json=sample_rubric_data)

        assert response.status_code == 400
        assert "must have at least one trait" in response.json()["detail"]

    def test_create_rubric_duplicate_trait_names(self, client, sample_rubric_data):
        """Test rubric creation with duplicate trait names."""
        sample_rubric_data["traits"] = [
            {
                "name": "accuracy",
                "description": "First accuracy trait",
                "kind": "boolean",
                "min_score": None,
                "max_score": None,
            },
            {
                "name": "accuracy",
                "description": "Second accuracy trait",
                "kind": "score",
                "min_score": 1,
                "max_score": 5,
            },
        ]

        response = client.post("/api/rubric", json=sample_rubric_data)

        assert response.status_code == 400
        assert "Trait names must be unique" in response.json()["detail"]

    def test_get_rubric_exists(self, client, sample_rubric_data):
        """Test getting an existing rubric."""
        # First create a rubric
        client.post("/api/rubric", json=sample_rubric_data)

        # Then get it
        response = client.get("/api/rubric")

        assert response.status_code == 200
        data = response.json()
        assert len(data["traits"]) == 2

    def test_get_rubric_not_exists(self, client):
        """Test getting rubric when none exists."""
        # Clear any existing rubric first
        client.delete("/api/rubric")

        response = client.get("/api/rubric")

        assert response.status_code == 200
        assert response.json() is None

    def test_update_rubric(self, client, sample_rubric_data):
        """Test updating an existing rubric."""
        # Create initial rubric
        client.post("/api/rubric", json=sample_rubric_data)

        # Update it
        updated_data = sample_rubric_data.copy()
        updated_data["traits"].append({"name": "clarity", "description": "Is the response clear?", "kind": "boolean"})

        response = client.post("/api/rubric", json=updated_data)

        assert response.status_code == 200
        assert response.json()["message"] == "Rubric saved successfully"

        # Verify the update
        get_response = client.get("/api/rubric")
        data = get_response.json()
        assert len(data["traits"]) == 3

    def test_delete_rubric_exists(self, client, sample_rubric_data):
        """Test deleting an existing rubric."""
        # Create rubric
        client.post("/api/rubric", json=sample_rubric_data)

        # Delete it
        response = client.delete("/api/rubric")

        assert response.status_code == 200
        assert response.json()["message"] == "Rubric deleted successfully"

        # Verify deletion
        get_response = client.get("/api/rubric")
        assert get_response.json() is None

    def test_delete_rubric_not_exists(self, client):
        """Test deleting rubric when none exists."""
        response = client.delete("/api/rubric")

        assert response.status_code == 200
        assert response.json()["message"] == "Rubric deleted successfully"


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
                "kind": "boolean",
                "min_score": null,
                "max_score": null
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
    def test_generate_traits_llm_error(self, mock_service_class, client, sample_trait_generation_request):
        """Test trait generation when LLM fails."""
        # Mock the generation service to raise an error
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.generate_rubric_traits.side_effect = Exception("LLM API error")

        response = client.post("/api/generate-rubric-traits", json=sample_trait_generation_request)

        assert response.status_code == 500
        assert "Error generating rubric traits" in response.json()["detail"]

    @patch("karenina_server.api.rubric_handlers.GenerationService")
    def test_generate_traits_invalid_response(self, mock_service_class, client, sample_trait_generation_request):
        """Test trait generation with invalid LLM response."""
        # Mock the generation service
        mock_service = Mock()
        mock_service_class.return_value = mock_service

        # Mock invalid JSON response
        mock_service.generate_rubric_traits.return_value = "This is not valid JSON"

        response = client.post("/api/generate-rubric-traits", json=sample_trait_generation_request)

        assert response.status_code == 200
        data = response.json()
        # Should fall back to default traits
        assert "traits" in data
        assert len(data["traits"]) >= 1

    def test_generate_traits_minimal_request(self, client):
        """Test trait generation with minimal request data."""
        minimal_request = {
            "questions": {"q1": {"question": "Test question?", "raw_answer": "Test answer", "tags": []}},
            "config": {"model_name": "gemini-2.0-flash", "temperature": 0.1, "interface": "langchain"},
        }

        with patch("karenina_server.api.rubric_handlers.GenerationService") as mock_service_class:
            mock_service = Mock()
            mock_service_class.return_value = mock_service
            mock_service.generate_rubric_traits.return_value = "[]"

            response = client.post("/api/generate-rubric-traits", json=minimal_request)

            assert response.status_code == 200
            # Should use default values for optional parameters
            mock_service.generate_rubric_traits.assert_called_once()

    def test_generate_traits_with_suggestions(self, client, sample_trait_generation_request):
        """Test trait generation with user suggestions."""
        sample_trait_generation_request["user_suggestions"] = ["clarity", "depth", "relevance"]

        with patch("karenina_server.api.rubric_handlers.GenerationService") as mock_service_class:
            mock_service = Mock()
            mock_service_class.return_value = mock_service
            mock_service.generate_rubric_traits.return_value = "[]"

            response = client.post("/api/generate-rubric-traits", json=sample_trait_generation_request)

            assert response.status_code == 200
            # Verify suggestions were passed through
            call_args = mock_service.generate_rubric_traits.call_args
            user_prompt = call_args.kwargs["user_prompt"]
            assert "clarity" in user_prompt
            assert "depth" in user_prompt
            assert "relevance" in user_prompt


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
                    "description": "Valid description",
                    "kind": "invalid_kind",  # Invalid trait kind
                }
            ]
        }

        response = client.post("/api/rubric", json=invalid_kind_data)

        # Should get validation error from Pydantic
        assert response.status_code == 422


class TestRubricIntegration:
    """Integration tests for rubric functionality."""

    def test_complete_rubric_workflow(self, client):
        """Test complete workflow: create, read, update, delete."""
        # Step 1: Create rubric
        create_data = {"traits": [{"name": "initial_trait", "description": "Initial trait", "kind": "boolean"}]}

        create_response = client.post("/api/rubric", json=create_data)
        assert create_response.status_code == 200

        # Step 2: Read rubric
        read_response = client.get("/api/rubric")
        assert read_response.status_code == 200
        data = read_response.json()
        assert len(data["traits"]) == 1

        # Step 3: Update rubric
        update_data = data.copy()
        update_data["traits"].append(
            {
                "name": "added_trait",
                "description": "Added during update",
                "kind": "score",
                "min_score": 1,
                "max_score": 3,
            }
        )

        update_response = client.post("/api/rubric", json=update_data)
        assert update_response.status_code == 200

        # Verify update
        read_again_response = client.get("/api/rubric")
        updated_data = read_again_response.json()
        assert len(updated_data["traits"]) == 2

        # Step 4: Delete rubric
        delete_response = client.delete("/api/rubric")
        assert delete_response.status_code == 200

        # Verify deletion
        final_read_response = client.get("/api/rubric")
        assert final_read_response.json() is None

    @patch("karenina_server.api.rubric_handlers.GenerationService")
    def test_trait_generation_to_rubric_creation(self, mock_service_class, client):
        """Test workflow from trait generation to rubric creation."""
        # Step 1: Generate traits
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.generate_rubric_traits.return_value = """
        [
            {
                "name": "generated_accuracy",
                "description": "Generated accuracy trait",
                "kind": "boolean",
                "min_score": null,
                "max_score": null
            },
            {
                "name": "generated_completeness",
                "description": "Generated completeness trait",
                "kind": "score",
                "min_score": 1,
                "max_score": 5
            }
        ]
        """

        generation_request = {
            "questions": {"q1": {"question": "Test question?", "raw_answer": "Test answer", "tags": []}},
            "config": {"model_name": "gemini-2.0-flash", "temperature": 0.1, "interface": "langchain"},
        }

        generation_response = client.post("/api/generate-rubric-traits", json=generation_request)
        assert generation_response.status_code == 200
        generated_traits = generation_response.json()["traits"]

        # Step 2: Create rubric using generated traits
        rubric_data = {"traits": generated_traits}

        create_response = client.post("/api/rubric", json=rubric_data)
        assert create_response.status_code == 200

        # Step 3: Verify the created rubric
        read_response = client.get("/api/rubric")
        rubric = read_response.json()
        assert len(rubric["traits"]) == 2
        assert rubric["traits"][0]["name"] == "generated_accuracy"
        assert rubric["traits"][1]["name"] == "generated_completeness"

    def test_rubric_persistence_across_requests(self, client, sample_rubric_data):
        """Test that rubric persists across multiple requests."""
        # Create rubric in one request
        client.post("/api/rubric", json=sample_rubric_data)

        # Make multiple read requests to ensure persistence
        for _ in range(5):
            response = client.get("/api/rubric")
            assert response.status_code == 200
            data = response.json()
            assert len(data["traits"]) == 2

    def test_rubric_overwrite_behavior(self, client):
        """Test that creating a new rubric overwrites the existing one."""
        # Create first rubric
        first_rubric = {"traits": [{"name": "trait1", "description": "First trait", "kind": "boolean"}]}
        client.post("/api/rubric", json=first_rubric)

        # Create second rubric (should overwrite)
        second_rubric = {
            "traits": [
                {"name": "trait2", "description": "Second trait", "kind": "score", "min_score": 1, "max_score": 3}
            ]
        }
        client.post("/api/rubric", json=second_rubric)

        # Verify only second rubric exists
        response = client.get("/api/rubric")
        data = response.json()
        assert len(data["traits"]) == 1
        assert data["traits"][0]["name"] == "trait2"


class TestOpenRouterConfiguration:
    """Test OpenRouter-specific configuration handling."""

    def test_openrouter_config_no_default_provider(self):
        """Test that OpenRouter config doesn't default to google_genai."""
        from karenina_server.api.rubric_handlers import RubricTraitGenerationConfig

        # Create config with OpenRouter interface
        config = RubricTraitGenerationConfig(
            model_name="openrouter/cypher-alpha:free", temperature=0.1, interface="openrouter"
        )

        # Provider should be None, not defaulted to google_genai
        assert config.model_provider is None
        assert config.interface == "openrouter"
        assert config.model_name == "openrouter/cypher-alpha:free"

    @patch("karenina_server.api.rubric_handlers.GenerationService")
    def test_generate_traits_with_openrouter(self, mock_service_class, client):
        """Test trait generation with OpenRouter interface."""
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.generate_rubric_traits.return_value = """
        [
            {
                "name": "clarity",
                "description": "Is the response clear?",
                "kind": "boolean"
            }
        ]
        """

        request_data = {
            "questions": {"q1": {"question": "Test?", "raw_answer": "Answer", "tags": []}},
            "config": {
                "model_name": "openrouter/cypher-alpha:free",
                "temperature": 0.1,
                "interface": "openrouter",
                # Note: no model_provider field
            },
        }

        response = client.post("/api/generate-rubric-traits", json=request_data)
        assert response.status_code == 200

        # Verify the service was called with correct parameters
        call_args = mock_service.generate_rubric_traits.call_args
        assert call_args.kwargs["model_provider"] == ""  # Should be empty for OpenRouter
        assert call_args.kwargs["interface"] == "openrouter"
        assert call_args.kwargs["model_name"] == "openrouter/cypher-alpha:free"

    @patch("karenina_server.api.rubric_handlers.GenerationService")
    def test_generate_traits_langchain_vs_openrouter(self, mock_service_class, client):
        """Test different handling of LangChain vs OpenRouter interfaces."""
        mock_service = Mock()
        mock_service_class.return_value = mock_service
        mock_service.generate_rubric_traits.return_value = "[]"

        # Test 1: LangChain with provider
        langchain_request = {
            "questions": {"q1": {"question": "Test?", "raw_answer": "Answer", "tags": []}},
            "config": {
                "model_provider": "google_genai",
                "model_name": "gemini-2.0-flash",
                "temperature": 0.1,
                "interface": "langchain",
            },
        }

        response = client.post("/api/generate-rubric-traits", json=langchain_request)
        assert response.status_code == 200

        call_args = mock_service.generate_rubric_traits.call_args
        assert call_args.kwargs["model_provider"] == "google_genai"
        assert call_args.kwargs["interface"] == "langchain"

        # Test 2: OpenRouter without provider
        openrouter_request = {
            "questions": {"q1": {"question": "Test?", "raw_answer": "Answer", "tags": []}},
            "config": {"model_name": "openrouter/cypher-alpha:free", "temperature": 0.1, "interface": "openrouter"},
        }

        response = client.post("/api/generate-rubric-traits", json=openrouter_request)
        assert response.status_code == 200

        call_args = mock_service.generate_rubric_traits.call_args
        assert call_args.kwargs["model_provider"] == ""  # Empty for OpenRouter
        assert call_args.kwargs["interface"] == "openrouter"
