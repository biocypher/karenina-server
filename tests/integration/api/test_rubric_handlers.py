"""Integration tests for rubric API handlers.

Uses TestClient to test API endpoints for rubric management.
"""

from pathlib import Path

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
def sample_rubric_data():
    """Sample rubric data for testing."""
    return {
        "llm_traits": [
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


@pytest.mark.integration
@pytest.mark.api
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
        sample_rubric_data["llm_traits"][0]["name"] = ""
        response = client.post("/api/rubric", json=sample_rubric_data)
        assert response.status_code == 422

    def test_create_rubric_missing_traits(self, client):
        """Test rubric creation with missing traits."""
        response = client.post("/api/rubric", json={})
        assert response.status_code == 400
        assert "must have at least one trait" in response.json()["detail"]

    def test_create_rubric_duplicate_trait_names(self, client, sample_rubric_data):
        """Test rubric creation with duplicate trait names."""
        sample_rubric_data["llm_traits"][1]["name"] = sample_rubric_data["llm_traits"][0]["name"]
        response = client.post("/api/rubric", json=sample_rubric_data)
        assert response.status_code == 400
        assert "must be unique" in response.json()["detail"]

    def test_get_rubric_none_exists(self, client):
        """Test getting rubric when none exists."""
        client.delete("/api/rubric")
        response = client.get("/api/rubric")
        assert response.status_code == 200
        assert response.json() is None

    def test_get_rubric_after_create(self, client, sample_rubric_data):
        """Test getting rubric after creation."""
        create_response = client.post("/api/rubric", json=sample_rubric_data)
        assert create_response.status_code == 200

        get_response = client.get("/api/rubric")
        assert get_response.status_code == 200
        rubric_data = get_response.json()
        assert rubric_data is not None
        assert "llm_traits" in rubric_data
        assert len(rubric_data["llm_traits"]) == 2

    def test_update_rubric(self, client, sample_rubric_data):
        """Test updating an existing rubric."""
        client.post("/api/rubric", json=sample_rubric_data)

        updated_data = {
            "llm_traits": [
                {
                    "name": "clarity",
                    "description": "Is the response clear?",
                    "kind": "boolean",
                }
            ]
        }

        response = client.post("/api/rubric", json=updated_data)
        assert response.status_code == 200

        get_response = client.get("/api/rubric")
        rubric_data = get_response.json()
        assert len(rubric_data["llm_traits"]) == 1
        assert rubric_data["llm_traits"][0]["name"] == "clarity"

    def test_delete_rubric(self, client, sample_rubric_data):
        """Test deleting a rubric."""
        client.post("/api/rubric", json=sample_rubric_data)

        delete_response = client.delete("/api/rubric")
        assert delete_response.status_code == 200
        assert delete_response.json()["message"] == "Rubric deleted successfully"

        get_response = client.get("/api/rubric")
        assert get_response.json() is None


@pytest.mark.integration
@pytest.mark.api
class TestRubricValidation:
    """Test rubric validation logic."""

    def test_rubric_trait_validation(self, client):
        """Test validation of individual rubric traits."""
        invalid_trait_data = {
            "traits": [
                {
                    "name": "",
                    "description": "Valid description",
                    "kind": "boolean",
                }
            ]
        }
        response = client.post("/api/rubric", json=invalid_trait_data)
        assert response.status_code == 422

    def test_rubric_invalid_trait_kind(self, client):
        """Test validation of invalid trait kind."""
        invalid_kind_data = {
            "traits": [
                {
                    "name": "test_trait",
                    "description": "Test description",
                    "kind": "invalid_kind",
                }
            ]
        }
        response = client.post("/api/rubric", json=invalid_kind_data)
        assert response.status_code == 422
