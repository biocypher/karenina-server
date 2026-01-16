"""Integration tests for benchmark preset API handlers.

Uses TestClient to test API endpoints for preset management.
"""

import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from karenina_server.server import create_fastapi_app


@pytest.fixture
def temp_presets_dir():
    """Create a temporary presets directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        yield temp_path


@pytest.fixture
def client(temp_presets_dir):
    """Create a test client with temporary presets directory."""
    webapp_dir = Path(__file__).parent.parent.parent.parent / "src" / "karenina_server" / "webapp"

    with patch("karenina_server.api.preset_handlers.preset_service") as mock_service:
        from karenina_server.services.preset_service import BenchmarkPresetService

        mock_service.return_value = None
        import karenina_server.api.preset_handlers as handlers

        original_service = handlers.preset_service
        handlers.preset_service = BenchmarkPresetService(presets_dir_path=temp_presets_dir)

        app = create_fastapi_app(webapp_dir)
        test_client = TestClient(app)
        yield test_client
        handlers.preset_service = original_service


@pytest.fixture
def sample_preset_data():
    """Sample preset data for testing."""
    return {
        "name": "Test Preset",
        "description": "A test preset configuration",
        "config": {
            "answering_models": [
                {
                    "id": "test-answering-1",
                    "model_provider": "openai",
                    "model_name": "gpt-4.1-mini",
                    "temperature": 0.0,
                    "interface": "langchain",
                    "system_prompt": "You are a helpful assistant.",
                    "max_retries": 3,
                }
            ],
            "parsing_models": [
                {
                    "id": "test-parsing-1",
                    "model_provider": "openai",
                    "model_name": "gpt-4.1-mini",
                    "temperature": 0.0,
                    "interface": "langchain",
                    "system_prompt": "Extract structured data.",
                    "max_retries": 3,
                }
            ],
            "replicate_count": 1,
            "rubric_enabled": True,
            "evaluation_mode": "template_and_rubric",
            "abstention_enabled": False,
            "deep_judgment_enabled": False,
        },
    }


@pytest.fixture
def complex_preset_data():
    """Complex preset with all features enabled."""
    return {
        "name": "Complex Preset",
        "description": "Full-featured configuration",
        "config": {
            "answering_models": [
                {
                    "id": "model-1",
                    "model_provider": "openai",
                    "model_name": "gpt-4.1-mini",
                    "temperature": 0.7,
                    "interface": "langchain",
                    "system_prompt": "Test prompt",
                    "max_retries": 5,
                    "mcp_urls_dict": {"tool1": "http://example.com"},
                    "mcp_tool_filter": ["tool1", "tool2"],
                }
            ],
            "parsing_models": [
                {
                    "id": "parser-1",
                    "model_provider": "google_genai",
                    "model_name": "gemini-pro",
                    "interface": "langchain",
                    "system_prompt": "Parse this",
                }
            ],
            "replicate_count": 3,
            "rubric_enabled": True,
            "rubric_trait_names": ["accuracy", "clarity"],
            "evaluation_mode": "rubric_only",
            "abstention_enabled": True,
            "deep_judgment_enabled": True,
            "deep_judgment_max_excerpts_per_attribute": 10,
            "deep_judgment_fuzzy_match_threshold": 0.85,
            "deep_judgment_search_enabled": True,
            "deep_judgment_search_tool": "tavily",
            "few_shot_config": {
                "enabled": True,
                "global_mode": "k-shot",
                "global_k": 3,
                "question_configs": {},
                "global_external_examples": [],
            },
        },
    }


@pytest.mark.integration
@pytest.mark.api
class TestPresetList:
    """Test GET /api/presets - list all presets."""

    def test_list_presets_empty(self, client):
        """Test listing presets when none exist."""
        response = client.get("/api/presets")
        assert response.status_code == 200
        data = response.json()
        assert "presets" in data
        assert data["presets"] == []

    def test_list_presets_with_data(self, client, sample_preset_data):
        """Test listing presets with existing data."""
        client.post("/api/presets", json=sample_preset_data)
        sample_preset_data2 = sample_preset_data.copy()
        sample_preset_data2["name"] = "Second Preset"
        client.post("/api/presets", json=sample_preset_data2)

        response = client.get("/api/presets")
        assert response.status_code == 200
        data = response.json()
        assert len(data["presets"]) == 2

    def test_list_presets_summary_features(self, client, complex_preset_data):
        """Test that list includes correct enabled features in summary."""
        response = client.post("/api/presets", json=complex_preset_data)
        assert response.status_code == 201

        response = client.get("/api/presets")
        data = response.json()
        preset = data["presets"][0]
        features = preset["summary"]["enabled_features"]

        assert "rubric" in features
        assert "abstention" in features
        assert "deep_judgment" in features


@pytest.mark.integration
@pytest.mark.api
class TestPresetCreate:
    """Test POST /api/presets - create preset."""

    def test_create_preset_success(self, client, sample_preset_data):
        """Test successful preset creation."""
        response = client.post("/api/presets", json=sample_preset_data)
        assert response.status_code == 201
        data = response.json()
        assert "preset" in data
        preset = data["preset"]
        assert preset["name"] == "Test Preset"

    def test_create_preset_duplicate_name(self, client, sample_preset_data):
        """Test that duplicate names are rejected."""
        response = client.post("/api/presets", json=sample_preset_data)
        assert response.status_code == 201
        response = client.post("/api/presets", json=sample_preset_data)
        assert response.status_code == 400
        assert "already exists" in response.json()["detail"]

    def test_create_preset_empty_name(self, client, sample_preset_data):
        """Test that empty names are rejected."""
        sample_preset_data["name"] = ""
        response = client.post("/api/presets", json=sample_preset_data)
        assert response.status_code == 422


@pytest.mark.integration
@pytest.mark.api
class TestPresetGet:
    """Test GET /api/presets/{preset_id} - get specific preset."""

    def test_get_preset_success(self, client, sample_preset_data):
        """Test successful preset retrieval."""
        create_response = client.post("/api/presets", json=sample_preset_data)
        preset_id = create_response.json()["preset"]["id"]

        response = client.get(f"/api/presets/{preset_id}")
        assert response.status_code == 200
        assert response.json()["preset"]["name"] == "Test Preset"

    def test_get_preset_not_found(self, client):
        """Test retrieving non-existent preset."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = client.get(f"/api/presets/{fake_id}")
        assert response.status_code == 404


@pytest.mark.integration
@pytest.mark.api
class TestPresetUpdate:
    """Test PUT /api/presets/{preset_id} - update preset."""

    def test_update_preset_name(self, client, sample_preset_data):
        """Test updating preset name."""
        create_response = client.post("/api/presets", json=sample_preset_data)
        preset_id = create_response.json()["preset"]["id"]

        update_data = {"name": "Updated Name"}
        response = client.put(f"/api/presets/{preset_id}", json=update_data)
        assert response.status_code == 200
        assert response.json()["preset"]["name"] == "Updated Name"

    def test_update_preset_not_found(self, client):
        """Test updating non-existent preset."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = client.put(f"/api/presets/{fake_id}", json={"name": "New Name"})
        assert response.status_code == 404


@pytest.mark.integration
@pytest.mark.api
class TestPresetDelete:
    """Test DELETE /api/presets/{preset_id} - delete preset."""

    def test_delete_preset_success(self, client, sample_preset_data):
        """Test successful preset deletion."""
        create_response = client.post("/api/presets", json=sample_preset_data)
        preset_id = create_response.json()["preset"]["id"]

        response = client.delete(f"/api/presets/{preset_id}")
        assert response.status_code == 200

        get_response = client.get(f"/api/presets/{preset_id}")
        assert get_response.status_code == 404

    def test_delete_preset_not_found(self, client):
        """Test deleting non-existent preset."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        response = client.delete(f"/api/presets/{fake_id}")
        assert response.status_code == 404


@pytest.mark.integration
@pytest.mark.api
class TestPresetIntegration:
    """Integration tests for full preset workflows."""

    def test_full_crud_workflow(self, client, sample_preset_data):
        """Test complete CRUD workflow."""
        # Create
        create_response = client.post("/api/presets", json=sample_preset_data)
        assert create_response.status_code == 201
        preset_id = create_response.json()["preset"]["id"]

        # Read
        get_response = client.get(f"/api/presets/{preset_id}")
        assert get_response.status_code == 200

        # Update
        update_response = client.put(f"/api/presets/{preset_id}", json={"name": "Updated"})
        assert update_response.status_code == 200

        # List
        list_response = client.get("/api/presets")
        assert len(list_response.json()["presets"]) == 1

        # Delete
        delete_response = client.delete(f"/api/presets/{preset_id}")
        assert delete_response.status_code == 200

        # Verify deletion
        get_response = client.get(f"/api/presets/{preset_id}")
        assert get_response.status_code == 404
