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
        response = client.put("/api/v2/rubrics/current", json=sample_rubric_data)
        assert response.status_code == 200
        data = response.json()
        assert data["message"] == "Rubric saved successfully"

    def test_create_rubric_invalid_trait_name(self, client, sample_rubric_data):
        """Test rubric creation with invalid trait name."""
        sample_rubric_data["llm_traits"][0]["name"] = ""
        response = client.put("/api/v2/rubrics/current", json=sample_rubric_data)
        assert response.status_code == 422

    def test_create_rubric_missing_traits(self, client):
        """Test rubric creation with missing traits."""
        response = client.put("/api/v2/rubrics/current", json={})
        assert response.status_code == 400
        assert "must have at least one trait" in response.json()["detail"]

    def test_create_rubric_duplicate_trait_names(self, client, sample_rubric_data):
        """Test rubric creation with duplicate trait names."""
        sample_rubric_data["llm_traits"][1]["name"] = sample_rubric_data["llm_traits"][0]["name"]
        response = client.put("/api/v2/rubrics/current", json=sample_rubric_data)
        assert response.status_code == 400
        assert "must be unique" in response.json()["detail"]

    def test_get_rubric_none_exists(self, client):
        """Test getting rubric when none exists."""
        client.delete("/api/v2/rubrics/current")
        response = client.get("/api/v2/rubrics/current")
        assert response.status_code == 200
        assert response.json() is None

    def test_get_rubric_after_create(self, client, sample_rubric_data):
        """Test getting rubric after creation."""
        create_response = client.put("/api/v2/rubrics/current", json=sample_rubric_data)
        assert create_response.status_code == 200

        get_response = client.get("/api/v2/rubrics/current")
        assert get_response.status_code == 200
        rubric_data = get_response.json()
        assert rubric_data is not None
        assert "llm_traits" in rubric_data
        assert len(rubric_data["llm_traits"]) == 2

    def test_update_rubric(self, client, sample_rubric_data):
        """Test updating an existing rubric."""
        client.put("/api/v2/rubrics/current", json=sample_rubric_data)

        updated_data = {
            "llm_traits": [
                {
                    "name": "clarity",
                    "description": "Is the response clear?",
                    "kind": "boolean",
                }
            ]
        }

        response = client.put("/api/v2/rubrics/current", json=updated_data)
        assert response.status_code == 200

        get_response = client.get("/api/v2/rubrics/current")
        rubric_data = get_response.json()
        assert len(rubric_data["llm_traits"]) == 1
        assert rubric_data["llm_traits"][0]["name"] == "clarity"

    def test_delete_rubric(self, client, sample_rubric_data):
        """Test deleting a rubric."""
        client.put("/api/v2/rubrics/current", json=sample_rubric_data)

        delete_response = client.delete("/api/v2/rubrics/current")
        assert delete_response.status_code == 200
        assert delete_response.json()["message"] == "Rubric deleted successfully"

        get_response = client.get("/api/v2/rubrics/current")
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
        response = client.put("/api/v2/rubrics/current", json=invalid_trait_data)
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
        response = client.put("/api/v2/rubrics/current", json=invalid_kind_data)
        assert response.status_code == 422


@pytest.mark.integration
@pytest.mark.api
class TestLiteralKindTraitValidation:
    """Test API validation for literal kind LLM traits."""

    def test_literal_trait_success(self, client):
        """Test successful creation of rubric with literal kind trait."""
        rubric_data = {
            "llm_traits": [
                {
                    "name": "sentiment",
                    "description": "Classify the sentiment of the response",
                    "kind": "literal",
                    "classes": {
                        "positive": "Response has positive sentiment",
                        "neutral": "Response has neutral sentiment",
                        "negative": "Response has negative sentiment",
                    },
                    "higher_is_better": True,
                }
            ]
        }
        response = client.put("/api/v2/rubrics/current", json=rubric_data)
        assert response.status_code == 200

        # Verify the rubric was stored correctly
        get_response = client.get("/api/v2/rubrics/current")
        assert get_response.status_code == 200
        rubric = get_response.json()
        assert len(rubric["llm_traits"]) == 1
        trait = rubric["llm_traits"][0]
        assert trait["kind"] == "literal"
        assert trait["classes"] == {
            "positive": "Response has positive sentiment",
            "neutral": "Response has neutral sentiment",
            "negative": "Response has negative sentiment",
        }
        # min_score and max_score should be auto-derived
        assert trait["min_score"] == 0
        assert trait["max_score"] == 2  # 3 classes -> max index 2

    def test_literal_trait_missing_classes(self, client):
        """Test that literal kind trait without classes is rejected."""
        rubric_data = {
            "llm_traits": [
                {
                    "name": "category",
                    "description": "Categorize the response",
                    "kind": "literal",
                    # classes field is missing
                    "higher_is_better": True,
                }
            ]
        }
        response = client.put("/api/v2/rubrics/current", json=rubric_data)
        assert response.status_code == 422

    def test_literal_trait_too_few_classes(self, client):
        """Test that literal kind trait with only 1 class is rejected."""
        rubric_data = {
            "llm_traits": [
                {
                    "name": "category",
                    "description": "Categorize the response",
                    "kind": "literal",
                    "classes": {
                        "only_one": "Single class",
                    },
                    "higher_is_better": True,
                }
            ]
        }
        response = client.put("/api/v2/rubrics/current", json=rubric_data)
        assert response.status_code == 422

    def test_literal_trait_empty_class_name(self, client):
        """Test that literal kind trait with empty class name is rejected."""
        rubric_data = {
            "llm_traits": [
                {
                    "name": "category",
                    "description": "Categorize the response",
                    "kind": "literal",
                    "classes": {
                        "": "Empty name class",
                        "valid": "Valid class",
                    },
                    "higher_is_better": True,
                }
            ]
        }
        response = client.put("/api/v2/rubrics/current", json=rubric_data)
        assert response.status_code == 422

    def test_literal_trait_empty_class_description(self, client):
        """Test that literal kind trait with empty class description is rejected."""
        rubric_data = {
            "llm_traits": [
                {
                    "name": "category",
                    "description": "Categorize the response",
                    "kind": "literal",
                    "classes": {
                        "class_a": "",
                        "class_b": "Valid description",
                    },
                    "higher_is_better": True,
                }
            ]
        }
        response = client.put("/api/v2/rubrics/current", json=rubric_data)
        assert response.status_code == 422

    def test_literal_trait_duplicate_class_names(self, client):
        """Test that literal kind trait with duplicate class names (case-insensitive) is rejected."""
        rubric_data = {
            "llm_traits": [
                {
                    "name": "category",
                    "description": "Categorize the response",
                    "kind": "literal",
                    "classes": {
                        "Good": "Uppercase Good",
                        "good": "Lowercase good (duplicate)",
                    },
                    "higher_is_better": True,
                }
            ]
        }
        response = client.put("/api/v2/rubrics/current", json=rubric_data)
        assert response.status_code == 422

    def test_literal_trait_max_classes(self, client):
        """Test that literal kind trait with exactly 20 classes succeeds."""
        classes = {f"class_{i}": f"Description for class {i}" for i in range(20)}
        rubric_data = {
            "llm_traits": [
                {
                    "name": "many_categories",
                    "description": "Categorize into many options",
                    "kind": "literal",
                    "classes": classes,
                    "higher_is_better": True,
                }
            ]
        }
        response = client.put("/api/v2/rubrics/current", json=rubric_data)
        assert response.status_code == 200

        # Verify max_score is correctly derived
        get_response = client.get("/api/v2/rubrics/current")
        trait = get_response.json()["llm_traits"][0]
        assert trait["max_score"] == 19  # 20 classes -> max index 19

    def test_literal_trait_too_many_classes(self, client):
        """Test that literal kind trait with more than 20 classes is rejected."""
        classes = {f"class_{i}": f"Description for class {i}" for i in range(21)}
        rubric_data = {
            "llm_traits": [
                {
                    "name": "too_many_categories",
                    "description": "Too many options",
                    "kind": "literal",
                    "classes": classes,
                    "higher_is_better": True,
                }
            ]
        }
        response = client.put("/api/v2/rubrics/current", json=rubric_data)
        assert response.status_code == 422

    def test_literal_trait_mixed_with_other_kinds(self, client):
        """Test rubric with mixed trait kinds including literal."""
        rubric_data = {
            "llm_traits": [
                {
                    "name": "accuracy",
                    "description": "Is the response accurate?",
                    "kind": "boolean",
                    "higher_is_better": True,
                },
                {
                    "name": "quality",
                    "description": "Rate the quality",
                    "kind": "score",
                    "min_score": 1,
                    "max_score": 5,
                    "higher_is_better": True,
                },
                {
                    "name": "sentiment",
                    "description": "Classify sentiment",
                    "kind": "literal",
                    "classes": {
                        "positive": "Positive sentiment",
                        "negative": "Negative sentiment",
                    },
                    "higher_is_better": True,
                },
            ]
        }
        response = client.put("/api/v2/rubrics/current", json=rubric_data)
        assert response.status_code == 200

        # Verify all traits are stored correctly
        get_response = client.get("/api/v2/rubrics/current")
        rubric = get_response.json()
        assert len(rubric["llm_traits"]) == 3

        boolean_trait = next(t for t in rubric["llm_traits"] if t["name"] == "accuracy")
        assert boolean_trait["kind"] == "boolean"

        score_trait = next(t for t in rubric["llm_traits"] if t["name"] == "quality")
        assert score_trait["kind"] == "score"
        assert score_trait["min_score"] == 1
        assert score_trait["max_score"] == 5

        literal_trait = next(t for t in rubric["llm_traits"] if t["name"] == "sentiment")
        assert literal_trait["kind"] == "literal"
        assert literal_trait["min_score"] == 0
        assert literal_trait["max_score"] == 1  # 2 classes -> max index 1
