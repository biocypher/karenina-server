"""Tests for benchmark preset API endpoints."""

import json
import tempfile
from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from karenina_server.server import create_fastapi_app


@pytest.fixture
def temp_presets_file():
    """Create a temporary benchmark_presets.json file for testing."""
    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
        initial_presets = {"presets": {}}
        json.dump(initial_presets, f)
        temp_path = Path(f.name)

    yield temp_path

    # Cleanup
    if temp_path.exists():
        temp_path.unlink()
    # Also cleanup backup if it exists
    backup_path = temp_path.with_suffix(".json.backup")
    if backup_path.exists():
        backup_path.unlink()


@pytest.fixture
def client(temp_presets_file):
    """Create a test client with temporary presets file."""
    webapp_dir = Path(__file__).parent.parent / "webapp"

    # Patch the preset_service global instance to use our temp file
    with patch("karenina_server.api.preset_handlers.preset_service") as mock_service:
        from karenina_server.services.preset_service import BenchmarkPresetService

        mock_service.return_value = None  # Prevent double-calling
        # Replace the global with our test instance
        import karenina_server.api.preset_handlers as handlers

        original_service = handlers.preset_service
        handlers.preset_service = BenchmarkPresetService(presets_file_path=temp_presets_file)

        app = create_fastapi_app(webapp_dir)
        test_client = TestClient(app)

        yield test_client

        # Restore original
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
        # Create some presets
        client.post("/api/presets", json=sample_preset_data)

        sample_preset_data2 = sample_preset_data.copy()
        sample_preset_data2["name"] = "Second Preset"
        client.post("/api/presets", json=sample_preset_data2)

        # List presets
        response = client.get("/api/presets")

        assert response.status_code == 200
        data = response.json()
        assert len(data["presets"]) == 2

        # Verify summary fields
        preset = data["presets"][0]
        assert "id" in preset
        assert "name" in preset
        assert "description" in preset
        assert "created_at" in preset
        assert "updated_at" in preset
        assert "summary" in preset

        # Verify summary content
        summary = preset["summary"]
        assert "answering_model_count" in summary
        assert "parsing_model_count" in summary
        assert "total_model_count" in summary
        assert "replicate_count" in summary
        assert "enabled_features" in summary
        assert summary["answering_model_count"] == 1
        assert summary["parsing_model_count"] == 1
        assert summary["total_model_count"] == 2

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
        assert "deep_judgment_search" in features
        assert "few_shot" in features


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
        assert preset["description"] == "A test preset configuration"
        assert "id" in preset
        assert "created_at" in preset
        assert "updated_at" in preset

        # Verify config is preserved
        assert preset["config"]["replicate_count"] == 1
        assert len(preset["config"]["answering_models"]) == 1
        assert preset["config"]["answering_models"][0]["model_name"] == "gpt-4.1-mini"

    def test_create_preset_without_description(self, client, sample_preset_data):
        """Test creating preset without description."""
        del sample_preset_data["description"]

        response = client.post("/api/presets", json=sample_preset_data)

        assert response.status_code == 201
        data = response.json()
        assert data["preset"]["description"] is None

    def test_create_preset_complex_config(self, client, complex_preset_data):
        """Test creating preset with complex configuration."""
        response = client.post("/api/presets", json=complex_preset_data)

        assert response.status_code == 201
        data = response.json()
        preset = data["preset"]

        # Verify all config fields preserved
        assert preset["config"]["replicate_count"] == 3
        assert preset["config"]["rubric_trait_names"] == ["accuracy", "clarity"]
        assert preset["config"]["evaluation_mode"] == "rubric_only"
        assert preset["config"]["deep_judgment_search_tool"] == "tavily"
        assert preset["config"]["few_shot_config"]["global_k"] == 3
        assert preset["config"]["answering_models"][0]["mcp_urls_dict"] == {"tool1": "http://example.com"}

    def test_create_preset_duplicate_name(self, client, sample_preset_data):
        """Test that duplicate names are rejected."""
        # Create first preset
        response = client.post("/api/presets", json=sample_preset_data)
        assert response.status_code == 201

        # Try to create another with same name
        response = client.post("/api/presets", json=sample_preset_data)

        assert response.status_code == 400
        assert "already exists" in response.json()["detail"]

    def test_create_preset_empty_name(self, client, sample_preset_data):
        """Test that empty names are rejected."""
        sample_preset_data["name"] = ""

        response = client.post("/api/presets", json=sample_preset_data)

        assert response.status_code == 422  # Pydantic validation
        # Check that detail contains validation error info
        assert "detail" in response.json()

    def test_create_preset_name_too_long(self, client, sample_preset_data):
        """Test that overly long names are rejected."""
        sample_preset_data["name"] = "x" * 101

        response = client.post("/api/presets", json=sample_preset_data)

        assert response.status_code == 422  # Pydantic validation
        assert "detail" in response.json()

    def test_create_preset_description_too_long(self, client, sample_preset_data):
        """Test that overly long descriptions are rejected."""
        sample_preset_data["description"] = "x" * 501

        response = client.post("/api/presets", json=sample_preset_data)

        assert response.status_code == 422  # Pydantic validation
        assert "detail" in response.json()

    def test_create_preset_invalid_config(self, client, sample_preset_data):
        """Test that invalid verification config is rejected."""
        # Remove required field
        del sample_preset_data["config"]["answering_models"]

        response = client.post("/api/presets", json=sample_preset_data)

        assert response.status_code == 422  # Pydantic validation error

    def test_create_preset_missing_fields(self, client):
        """Test that missing required fields are rejected."""
        response = client.post("/api/presets", json={})

        assert response.status_code == 422  # Pydantic validation error


class TestPresetGet:
    """Test GET /api/presets/{preset_id} - get specific preset."""

    def test_get_preset_success(self, client, sample_preset_data):
        """Test successful preset retrieval."""
        # Create a preset
        create_response = client.post("/api/presets", json=sample_preset_data)
        preset_id = create_response.json()["preset"]["id"]

        # Retrieve it
        response = client.get(f"/api/presets/{preset_id}")

        assert response.status_code == 200
        data = response.json()
        assert "preset" in data

        preset = data["preset"]
        assert preset["id"] == preset_id
        assert preset["name"] == "Test Preset"
        assert preset["config"]["replicate_count"] == 1

    def test_get_preset_not_found(self, client):
        """Test retrieving non-existent preset."""
        fake_id = "00000000-0000-0000-0000-000000000000"

        response = client.get(f"/api/presets/{fake_id}")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_get_preset_invalid_id_format(self, client):
        """Test retrieving with invalid UUID format."""
        response = client.get("/api/presets/not-a-uuid")

        # API doesn't validate UUID format, just passes to service which returns 404
        assert response.status_code == 404


class TestPresetUpdate:
    """Test PUT /api/presets/{preset_id} - update preset."""

    def test_update_preset_name(self, client, sample_preset_data):
        """Test updating preset name."""
        # Create preset
        create_response = client.post("/api/presets", json=sample_preset_data)
        preset_id = create_response.json()["preset"]["id"]

        # Update name
        update_data = {"name": "Updated Name"}
        response = client.put(f"/api/presets/{preset_id}", json=update_data)

        assert response.status_code == 200
        data = response.json()
        assert data["preset"]["name"] == "Updated Name"
        assert data["preset"]["id"] == preset_id

    def test_update_preset_description(self, client, sample_preset_data):
        """Test updating preset description."""
        create_response = client.post("/api/presets", json=sample_preset_data)
        preset_id = create_response.json()["preset"]["id"]

        update_data = {"description": "New description"}
        response = client.put(f"/api/presets/{preset_id}", json=update_data)

        assert response.status_code == 200
        assert response.json()["preset"]["description"] == "New description"

    def test_update_preset_config(self, client, sample_preset_data):
        """Test updating preset configuration."""
        create_response = client.post("/api/presets", json=sample_preset_data)
        preset_id = create_response.json()["preset"]["id"]

        # Update config - ensure it's a valid VerificationConfig dict
        # Need to provide full config with required fields
        new_config = {
            "answering_models": sample_preset_data["config"]["answering_models"],
            "parsing_models": sample_preset_data["config"]["parsing_models"],
            "replicate_count": 5,  # Changed
            "rubric_enabled": False,  # Changed
        }
        update_data = {"config": new_config}

        response = client.put(f"/api/presets/{preset_id}", json=update_data)

        assert response.status_code == 200
        preset = response.json()["preset"]
        assert preset["config"]["replicate_count"] == 5
        assert preset["config"]["rubric_enabled"] is False

    def test_update_preset_all_fields(self, client, sample_preset_data):
        """Test updating all fields simultaneously."""
        create_response = client.post("/api/presets", json=sample_preset_data)
        preset_id = create_response.json()["preset"]["id"]

        new_config = sample_preset_data["config"].copy()
        new_config["replicate_count"] = 10
        update_data = {
            "name": "Completely Updated",
            "description": "New description",
            "config": new_config,
        }

        response = client.put(f"/api/presets/{preset_id}", json=update_data)

        assert response.status_code == 200
        preset = response.json()["preset"]
        assert preset["name"] == "Completely Updated"
        assert preset["description"] == "New description"
        assert preset["config"]["replicate_count"] == 10

    def test_update_preset_not_found(self, client):
        """Test updating non-existent preset."""
        fake_id = "00000000-0000-0000-0000-000000000000"
        update_data = {"name": "New Name"}

        response = client.put(f"/api/presets/{fake_id}", json=update_data)

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_update_preset_duplicate_name(self, client, sample_preset_data):
        """Test that updating to duplicate name is rejected."""
        # Create two presets
        client.post("/api/presets", json=sample_preset_data)

        sample_preset_data2 = sample_preset_data.copy()
        sample_preset_data2["name"] = "Second Preset"
        response2 = client.post("/api/presets", json=sample_preset_data2)
        preset2_id = response2.json()["preset"]["id"]

        # Try to update preset2 to have preset1's name
        update_data = {"name": "Test Preset"}
        response = client.put(f"/api/presets/{preset2_id}", json=update_data)

        assert response.status_code == 400
        assert "already exists" in response.json()["detail"]

    def test_update_preset_validation(self, client, sample_preset_data):
        """Test validation during update."""
        create_response = client.post("/api/presets", json=sample_preset_data)
        preset_id = create_response.json()["preset"]["id"]

        # Try empty name - Pydantic validates this
        response = client.put(f"/api/presets/{preset_id}", json={"name": ""})
        assert response.status_code == 422  # Pydantic validation

        # Try name too long - Pydantic validates this
        response = client.put(f"/api/presets/{preset_id}", json={"name": "x" * 101})
        assert response.status_code == 422  # Pydantic validation

    def test_update_preset_empty_body(self, client, sample_preset_data):
        """Test that empty update body is rejected."""
        create_response = client.post("/api/presets", json=sample_preset_data)
        preset_id = create_response.json()["preset"]["id"]

        response = client.put(f"/api/presets/{preset_id}", json={})

        assert response.status_code == 400
        assert "at least one field" in response.json()["detail"].lower()


class TestPresetDelete:
    """Test DELETE /api/presets/{preset_id} - delete preset."""

    def test_delete_preset_success(self, client, sample_preset_data):
        """Test successful preset deletion."""
        # Create preset
        create_response = client.post("/api/presets", json=sample_preset_data)
        preset_id = create_response.json()["preset"]["id"]

        # Delete it
        response = client.delete(f"/api/presets/{preset_id}")

        assert response.status_code == 200
        assert "deleted successfully" in response.json()["message"]

        # Verify it's gone
        get_response = client.get(f"/api/presets/{preset_id}")
        assert get_response.status_code == 404

    def test_delete_preset_not_found(self, client):
        """Test deleting non-existent preset."""
        fake_id = "00000000-0000-0000-0000-000000000000"

        response = client.delete(f"/api/presets/{fake_id}")

        assert response.status_code == 404
        assert "not found" in response.json()["detail"]

    def test_delete_preset_multiple(self, client, sample_preset_data):
        """Test deleting multiple presets."""
        # Create three presets
        ids = []
        for i in range(3):
            data = sample_preset_data.copy()
            data["name"] = f"Preset {i}"
            response = client.post("/api/presets", json=data)
            ids.append(response.json()["preset"]["id"])

        # Delete two of them
        client.delete(f"/api/presets/{ids[0]}")
        client.delete(f"/api/presets/{ids[2]}")

        # List remaining
        response = client.get("/api/presets")
        presets = response.json()["presets"]
        assert len(presets) == 1
        assert presets[0]["id"] == ids[1]


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
        assert get_response.json()["preset"]["name"] == "Test Preset"

        # Update
        update_response = client.put(f"/api/presets/{preset_id}", json={"name": "Updated Preset"})
        assert update_response.status_code == 200
        assert update_response.json()["preset"]["name"] == "Updated Preset"

        # List
        list_response = client.get("/api/presets")
        assert len(list_response.json()["presets"]) == 1

        # Delete
        delete_response = client.delete(f"/api/presets/{preset_id}")
        assert delete_response.status_code == 200

        # Verify deletion
        get_response = client.get(f"/api/presets/{preset_id}")
        assert get_response.status_code == 404

    def test_multiple_presets_management(self, client, sample_preset_data):
        """Test managing multiple presets simultaneously."""
        # Create multiple presets
        preset_ids = []
        for i in range(5):
            data = sample_preset_data.copy()
            data["name"] = f"Preset {i}"
            data["description"] = f"Description {i}"
            response = client.post("/api/presets", json=data)
            assert response.status_code == 201
            preset_ids.append(response.json()["preset"]["id"])

        # List all
        list_response = client.get("/api/presets")
        assert len(list_response.json()["presets"]) == 5

        # Update some
        client.put(f"/api/presets/{preset_ids[0]}", json={"name": "Updated 0"})
        client.put(f"/api/presets/{preset_ids[2]}", json={"description": "New desc"})

        # Delete some
        client.delete(f"/api/presets/{preset_ids[1]}")
        client.delete(f"/api/presets/{preset_ids[4]}")

        # Verify final state
        list_response = client.get("/api/presets")
        remaining = list_response.json()["presets"]
        assert len(remaining) == 3

        # Check updates persisted
        get_response = client.get(f"/api/presets/{preset_ids[0]}")
        assert get_response.json()["preset"]["name"] == "Updated 0"

    def test_preset_persistence_across_requests(self, client, sample_preset_data):
        """Test that presets persist across multiple API calls."""
        # Create preset
        create_response = client.post("/api/presets", json=sample_preset_data)
        preset_id = create_response.json()["preset"]["id"]

        # Make multiple read requests
        for _ in range(5):
            response = client.get(f"/api/presets/{preset_id}")
            assert response.status_code == 200
            assert response.json()["preset"]["name"] == "Test Preset"

    def test_concurrent_preset_operations(self, client, sample_preset_data):
        """Test multiple operations on different presets."""
        # Create two presets
        data1 = sample_preset_data.copy()
        data1["name"] = "Preset A"
        response1 = client.post("/api/presets", json=data1)
        id1 = response1.json()["preset"]["id"]

        data2 = sample_preset_data.copy()
        data2["name"] = "Preset B"
        response2 = client.post("/api/presets", json=data2)
        id2 = response2.json()["preset"]["id"]

        # Update both simultaneously (in sequence for testing)
        client.put(f"/api/presets/{id1}", json={"description": "Updated A"})
        client.put(f"/api/presets/{id2}", json={"description": "Updated B"})

        # Verify both updates
        response1 = client.get(f"/api/presets/{id1}")
        response2 = client.get(f"/api/presets/{id2}")

        assert response1.json()["preset"]["description"] == "Updated A"
        assert response2.json()["preset"]["description"] == "Updated B"
