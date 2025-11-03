"""Tests for benchmark preset service."""

import json
import tempfile
from pathlib import Path
from uuid import UUID

import pytest
from karenina.schemas.workflow.verification import VerificationConfig

from karenina_server.services.preset_service import BenchmarkPresetService


@pytest.fixture
def sample_verification_config():
    """Create a sample VerificationConfig for testing."""
    return VerificationConfig(
        answering_models=[
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
        parsing_models=[
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
        replicate_count=1,
        rubric_enabled=True,
        evaluation_mode="template_and_rubric",
        abstention_enabled=False,
        deep_judgment_enabled=False,
    )


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
def preset_service(temp_presets_file):
    """Create a preset service with temporary presets file."""
    return BenchmarkPresetService(presets_file_path=temp_presets_file)


@pytest.fixture
def empty_preset_service():
    """Create a preset service with non-existent file."""
    with tempfile.TemporaryDirectory() as temp_dir:
        non_existent_file = Path(temp_dir) / "nonexistent.json"
        yield BenchmarkPresetService(presets_file_path=non_existent_file)


class TestBenchmarkPresetService:
    """Test cases for BenchmarkPresetService."""

    # ===== CREATE TESTS =====

    def test_create_preset_basic(self, preset_service, sample_verification_config):
        """Test creating a basic preset."""
        preset = preset_service.create_preset(
            name="Test Preset",
            config=sample_verification_config,
            description="A test preset",
        )

        assert preset["name"] == "Test Preset"
        assert preset["description"] == "A test preset"
        assert "id" in preset
        assert "created_at" in preset
        assert "updated_at" in preset
        assert preset["created_at"] == preset["updated_at"]

        # Verify UUID format
        UUID(preset["id"])

        # Verify config is preserved
        assert preset["config"]["replicate_count"] == 1
        assert preset["config"]["rubric_enabled"] is True
        assert len(preset["config"]["answering_models"]) == 1
        assert len(preset["config"]["parsing_models"]) == 1

    def test_create_preset_without_description(self, preset_service, sample_verification_config):
        """Test creating a preset without description."""
        preset = preset_service.create_preset(
            name="No Description Preset",
            config=sample_verification_config,
        )

        assert preset["name"] == "No Description Preset"
        assert preset["description"] is None

    def test_create_preset_with_complex_config(self, preset_service):
        """Test creating a preset with all configuration options."""
        complex_config = VerificationConfig(
            answering_models=[
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
                },
                {
                    "id": "model-2",
                    "model_provider": "anthropic",
                    "model_name": "claude-3-5-sonnet-20241022",
                    "temperature": 0.0,
                    "interface": "openai_endpoint",
                    "system_prompt": "Another prompt",
                    "endpoint_base_url": "https://api.example.com",
                    "endpoint_api_key": "test-key",
                },
            ],
            parsing_models=[
                {
                    "id": "parser-1",
                    "model_provider": "google_genai",
                    "model_name": "gemini-pro",
                    "interface": "langchain",
                    "system_prompt": "Parse this",
                }
            ],
            replicate_count=3,
            parsing_only=False,
            rubric_enabled=True,
            rubric_trait_names=["accuracy", "clarity"],
            evaluation_mode="rubric_only",
            abstention_enabled=True,
            deep_judgment_enabled=True,
            deep_judgment_max_excerpts_per_attribute=10,
            deep_judgment_fuzzy_match_threshold=0.85,
            deep_judgment_excerpt_retry_attempts=3,
            deep_judgment_search_enabled=True,
            deep_judgment_search_tool="tavily",
            few_shot_config={
                "enabled": True,
                "global_mode": "k-shot",
                "global_k": 3,
                "question_configs": {"q1": {"mode": "custom", "k": 5}},
                "global_external_examples": [{"input": "test", "output": "result"}],
            },
        )

        preset = preset_service.create_preset(
            name="Complex Config",
            config=complex_config,
            description="Full-featured preset",
        )

        # Verify all fields preserved
        assert preset["config"]["replicate_count"] == 3
        assert preset["config"]["rubric_enabled"] is True
        assert preset["config"]["rubric_trait_names"] == ["accuracy", "clarity"]
        assert preset["config"]["evaluation_mode"] == "rubric_only"
        assert preset["config"]["abstention_enabled"] is True
        assert preset["config"]["deep_judgment_enabled"] is True
        assert preset["config"]["deep_judgment_search_tool"] == "tavily"
        assert preset["config"]["few_shot_config"]["enabled"] is True
        assert preset["config"]["few_shot_config"]["global_k"] == 3
        assert len(preset["config"]["answering_models"]) == 2
        assert preset["config"]["answering_models"][0]["mcp_urls_dict"] == {"tool1": "http://example.com"}
        assert preset["config"]["answering_models"][1]["endpoint_base_url"] == "https://api.example.com"

    def test_create_multiple_presets(self, preset_service, sample_verification_config):
        """Test creating multiple presets."""
        preset1 = preset_service.create_preset("Preset 1", sample_verification_config)
        preset2 = preset_service.create_preset("Preset 2", sample_verification_config)
        preset3 = preset_service.create_preset("Preset 3", sample_verification_config)

        assert preset1["id"] != preset2["id"]
        assert preset2["id"] != preset3["id"]

        all_presets = preset_service.list_presets()
        assert len(all_presets) == 3

    # ===== VALIDATION TESTS =====

    def test_create_preset_duplicate_name(self, preset_service, sample_verification_config):
        """Test that duplicate preset names are rejected."""
        preset_service.create_preset("Duplicate Name", sample_verification_config)

        with pytest.raises(ValueError, match="already exists"):
            preset_service.create_preset("Duplicate Name", sample_verification_config)

    def test_create_preset_name_too_long(self, preset_service, sample_verification_config):
        """Test that preset names exceeding max length are rejected."""
        long_name = "x" * 101  # Max is 100

        with pytest.raises(ValueError, match="cannot exceed 100 characters"):
            preset_service.create_preset(long_name, sample_verification_config)

    def test_create_preset_description_too_long(self, preset_service, sample_verification_config):
        """Test that descriptions exceeding max length are rejected."""
        long_description = "x" * 501  # Max is 500

        with pytest.raises(ValueError, match="cannot exceed 500 characters"):
            preset_service.create_preset("Test", sample_verification_config, description=long_description)

    def test_create_preset_empty_name(self, preset_service, sample_verification_config):
        """Test that empty preset names are rejected."""
        with pytest.raises(ValueError, match="cannot be empty"):
            preset_service.create_preset("", sample_verification_config)

        with pytest.raises(ValueError, match="cannot be empty"):
            preset_service.create_preset("   ", sample_verification_config)

    # ===== READ TESTS =====

    def test_get_preset(self, preset_service, sample_verification_config):
        """Test retrieving a preset by ID."""
        created = preset_service.create_preset("Test Preset", sample_verification_config)
        preset_id = created["id"]

        retrieved = preset_service.get_preset(preset_id)

        assert retrieved["id"] == preset_id
        assert retrieved["name"] == "Test Preset"
        assert retrieved["config"]["replicate_count"] == 1

    def test_get_nonexistent_preset(self, preset_service):
        """Test that retrieving non-existent preset raises ValueError."""
        fake_id = "00000000-0000-0000-0000-000000000000"

        with pytest.raises(ValueError, match="not found"):
            preset_service.get_preset(fake_id)

    def test_list_presets_empty(self, preset_service):
        """Test listing presets when none exist."""
        presets = preset_service.list_presets()
        assert presets == {}

    def test_list_presets_multiple(self, preset_service, sample_verification_config):
        """Test listing multiple presets."""
        preset1 = preset_service.create_preset("Preset A", sample_verification_config)
        preset2 = preset_service.create_preset("Preset B", sample_verification_config)

        presets = preset_service.list_presets()

        assert len(presets) == 2
        assert preset1["id"] in presets
        assert preset2["id"] in presets
        assert presets[preset1["id"]]["name"] == "Preset A"
        assert presets[preset2["id"]]["name"] == "Preset B"

    # ===== UPDATE TESTS =====

    def test_update_preset_name(self, preset_service, sample_verification_config):
        """Test updating a preset's name."""
        preset = preset_service.create_preset("Original Name", sample_verification_config)
        preset_id = preset["id"]
        original_updated_at = preset["updated_at"]

        # Wait a moment to ensure timestamp difference (in real tests, might mock time)
        import time

        time.sleep(0.01)

        updated = preset_service.update_preset(preset_id, name="New Name")

        assert updated["id"] == preset_id
        assert updated["name"] == "New Name"
        assert updated["updated_at"] != original_updated_at
        assert updated["created_at"] == preset["created_at"]

    def test_update_preset_description(self, preset_service, sample_verification_config):
        """Test updating a preset's description."""
        preset = preset_service.create_preset("Test", sample_verification_config, description="Old description")
        preset_id = preset["id"]

        updated = preset_service.update_preset(preset_id, description="New description")

        assert updated["description"] == "New description"

    def test_update_preset_config(self, preset_service, sample_verification_config):
        """Test updating a preset's configuration."""
        preset = preset_service.create_preset("Test", sample_verification_config)
        preset_id = preset["id"]

        # Create new config with different settings
        new_config = VerificationConfig(
            answering_models=sample_verification_config.answering_models,
            parsing_models=sample_verification_config.parsing_models,
            replicate_count=5,  # Changed
            rubric_enabled=False,  # Changed
        )

        updated = preset_service.update_preset(preset_id, config=new_config)

        assert updated["config"]["replicate_count"] == 5
        assert updated["config"]["rubric_enabled"] is False

    def test_update_preset_all_fields(self, preset_service, sample_verification_config):
        """Test updating all fields simultaneously."""
        preset = preset_service.create_preset("Original", sample_verification_config, description="Original desc")
        preset_id = preset["id"]

        new_config = VerificationConfig(
            answering_models=sample_verification_config.answering_models,
            parsing_models=sample_verification_config.parsing_models,
            replicate_count=10,
        )

        updated = preset_service.update_preset(
            preset_id,
            name="Updated Name",
            description="Updated desc",
            config=new_config,
        )

        assert updated["name"] == "Updated Name"
        assert updated["description"] == "Updated desc"
        assert updated["config"]["replicate_count"] == 10

    def test_update_nonexistent_preset(self, preset_service):
        """Test that updating non-existent preset raises ValueError."""
        fake_id = "00000000-0000-0000-0000-000000000000"

        with pytest.raises(ValueError, match="not found"):
            preset_service.update_preset(fake_id, name="New Name")

    def test_update_preset_duplicate_name(self, preset_service, sample_verification_config):
        """Test that updating to a duplicate name is rejected."""
        preset_service.create_preset("Preset 1", sample_verification_config)
        preset2 = preset_service.create_preset("Preset 2", sample_verification_config)

        with pytest.raises(ValueError, match="already exists"):
            preset_service.update_preset(preset2["id"], name="Preset 1")

    def test_update_preset_name_validation(self, preset_service, sample_verification_config):
        """Test name validation during updates."""
        preset = preset_service.create_preset("Test", sample_verification_config)

        with pytest.raises(ValueError, match="cannot be empty"):
            preset_service.update_preset(preset["id"], name="")

        with pytest.raises(ValueError, match="cannot exceed 100 characters"):
            preset_service.update_preset(preset["id"], name="x" * 101)

    # ===== DELETE TESTS =====

    def test_delete_preset(self, preset_service, sample_verification_config):
        """Test deleting a preset."""
        preset = preset_service.create_preset("To Delete", sample_verification_config)
        preset_id = preset["id"]

        # Verify exists
        assert preset_service.get_preset(preset_id) is not None

        # Delete
        preset_service.delete_preset(preset_id)

        # Verify deleted
        with pytest.raises(ValueError, match="not found"):
            preset_service.get_preset(preset_id)

    def test_delete_nonexistent_preset(self, preset_service):
        """Test that deleting non-existent preset raises ValueError."""
        fake_id = "00000000-0000-0000-0000-000000000000"

        with pytest.raises(ValueError, match="not found"):
            preset_service.delete_preset(fake_id)

    def test_delete_multiple_presets(self, preset_service, sample_verification_config):
        """Test deleting multiple presets."""
        preset1 = preset_service.create_preset("Preset 1", sample_verification_config)
        preset2 = preset_service.create_preset("Preset 2", sample_verification_config)
        preset3 = preset_service.create_preset("Preset 3", sample_verification_config)

        preset_service.delete_preset(preset1["id"])
        preset_service.delete_preset(preset3["id"])

        remaining = preset_service.list_presets()
        assert len(remaining) == 1
        assert preset2["id"] in remaining

    # ===== FILE OPERATIONS TESTS =====

    def test_file_creation(self, empty_preset_service, sample_verification_config):
        """Test that preset file is created automatically on init."""
        # File is created automatically by _ensure_file_exists() in __init__
        assert empty_preset_service.presets_file_path.exists()

        # Verify it has correct structure
        with open(empty_preset_service.presets_file_path) as f:
            data = json.load(f)
        assert "presets" in data
        assert data["presets"] == {}

        # Create a preset and verify it's saved
        empty_preset_service.create_preset("First Preset", sample_verification_config)

        with open(empty_preset_service.presets_file_path) as f:
            data = json.load(f)
        assert len(data["presets"]) == 1

    def test_file_persistence(self, temp_presets_file, sample_verification_config):
        """Test that presets persist across service instances."""
        # Create preset with first service instance
        service1 = BenchmarkPresetService(presets_file_path=temp_presets_file)
        preset = service1.create_preset("Persistent", sample_verification_config)
        preset_id = preset["id"]

        # Read with new service instance
        service2 = BenchmarkPresetService(presets_file_path=temp_presets_file)
        retrieved = service2.get_preset(preset_id)

        assert retrieved["name"] == "Persistent"

    def test_backup_creation(self, preset_service, sample_verification_config):
        """Test that backup is created during updates."""
        preset_service.create_preset("Test", sample_verification_config)

        backup_path = preset_service.presets_file_path.with_suffix(".json.backup")
        assert backup_path.exists()

    def test_backup_restore(self, preset_service, sample_verification_config):
        """Test backup restoration after failed operation."""
        # Create first preset - backup will be empty, file will have preset1
        preset1 = preset_service.create_preset("Test1", sample_verification_config)
        preset1_id = preset1["id"]

        # Create second preset - backup will have preset1, file will have preset1+preset2
        preset2 = preset_service.create_preset("Test2", sample_verification_config)
        preset2_id = preset2["id"]

        # Verify backup was created and contains preset1 only (state before preset2 was added)
        backup_path = preset_service.presets_file_path.with_suffix(".json.backup")
        assert backup_path.exists()

        # Simulate corruption by manually editing file
        with open(preset_service.presets_file_path, "w") as f:
            f.write("invalid json{")

        # Restore from backup
        preset_service._restore_backup()

        # Verify file is restored (should have preset1 but not preset2)
        with open(preset_service.presets_file_path) as f:
            data = json.load(f)
        assert preset1_id in data["presets"]
        assert data["presets"][preset1_id]["name"] == "Test1"
        # preset2 should not be in backup since backup was created before preset2 was added
        assert preset2_id not in data["presets"]

        # Verify service can read preset1
        restored = preset_service.get_preset(preset1_id)
        assert restored["name"] == "Test1"

        # Verify preset2 is gone
        with pytest.raises(ValueError, match="not found"):
            preset_service.get_preset(preset2_id)

    # ===== INTEGRATION TESTS =====

    def test_full_crud_cycle(self, preset_service, sample_verification_config):
        """Test complete CRUD cycle for a preset."""
        # Create
        created = preset_service.create_preset(
            "CRUD Test",
            sample_verification_config,
            description="Testing CRUD operations",
        )
        preset_id = created["id"]

        # Read
        retrieved = preset_service.get_preset(preset_id)
        assert retrieved["name"] == "CRUD Test"

        # Update
        updated = preset_service.update_preset(preset_id, name="Updated CRUD Test")
        assert updated["name"] == "Updated CRUD Test"

        # Delete
        preset_service.delete_preset(preset_id)
        with pytest.raises(ValueError):
            preset_service.get_preset(preset_id)

    def test_concurrent_presets(self, preset_service, sample_verification_config):
        """Test managing multiple presets simultaneously."""
        # Create several presets
        presets = []
        for i in range(5):
            preset = preset_service.create_preset(
                f"Preset {i}",
                sample_verification_config,
                description=f"Preset number {i}",
            )
            presets.append(preset)

        # Verify all exist
        all_presets = preset_service.list_presets()
        assert len(all_presets) == 5

        # Update some
        preset_service.update_preset(presets[0]["id"], name="Updated 0")
        preset_service.update_preset(presets[2]["id"], description="New desc for 2")

        # Delete some
        preset_service.delete_preset(presets[1]["id"])
        preset_service.delete_preset(presets[4]["id"])

        # Verify final state
        remaining = preset_service.list_presets()
        assert len(remaining) == 3
        assert presets[0]["id"] in remaining
        assert presets[2]["id"] in remaining
        assert presets[3]["id"] in remaining
