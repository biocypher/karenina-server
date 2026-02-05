"""Integration tests for benchmark preset service.

Uses file I/O with temporary directories.
"""

import json
import tempfile
import time
from pathlib import Path
from uuid import UUID

import pytest
from karenina.schemas.verification import VerificationConfig

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
def temp_presets_dir():
    """Create a temporary presets directory for testing."""
    with tempfile.TemporaryDirectory() as temp_dir:
        temp_path = Path(temp_dir)
        yield temp_path


@pytest.fixture
def preset_service(temp_presets_dir):
    """Create a preset service with temporary presets directory."""
    return BenchmarkPresetService(presets_dir_path=temp_presets_dir)


@pytest.fixture
def empty_preset_service():
    """Create a preset service with empty directory."""
    with tempfile.TemporaryDirectory() as temp_dir:
        yield BenchmarkPresetService(presets_dir_path=Path(temp_dir))


@pytest.mark.integration
@pytest.mark.service
class TestBenchmarkPresetService:
    """Test cases for BenchmarkPresetService."""

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
        UUID(preset["id"])
        assert preset["config"]["replicate_count"] == 1

    def test_create_preset_without_description(self, preset_service, sample_verification_config):
        """Test creating a preset without description."""
        preset = preset_service.create_preset(
            name="No Description Preset",
            config=sample_verification_config,
        )

        assert preset["name"] == "No Description Preset"
        assert preset["description"] is None

    def test_create_multiple_presets(self, preset_service, sample_verification_config):
        """Test creating multiple presets."""
        preset1 = preset_service.create_preset("Preset 1", sample_verification_config)
        preset2 = preset_service.create_preset("Preset 2", sample_verification_config)
        preset3 = preset_service.create_preset("Preset 3", sample_verification_config)

        assert preset1["id"] != preset2["id"]
        assert preset2["id"] != preset3["id"]

        all_presets = preset_service.list_presets()
        assert len(all_presets) == 3

    def test_create_preset_duplicate_name(self, preset_service, sample_verification_config):
        """Test that duplicate preset names are rejected."""
        preset_service.create_preset("Duplicate Name", sample_verification_config)

        with pytest.raises(ValueError, match="already exists"):
            preset_service.create_preset("Duplicate Name", sample_verification_config)

    def test_create_preset_name_too_long(self, preset_service, sample_verification_config):
        """Test that preset names exceeding max length are rejected."""
        long_name = "x" * 101

        with pytest.raises(ValueError, match="cannot exceed 100 characters"):
            preset_service.create_preset(long_name, sample_verification_config)

    def test_get_preset(self, preset_service, sample_verification_config):
        """Test retrieving a preset by ID."""
        created = preset_service.create_preset("Test Preset", sample_verification_config)
        preset_id = created["id"]

        retrieved = preset_service.get_preset(preset_id)

        assert retrieved["id"] == preset_id
        assert retrieved["name"] == "Test Preset"

    def test_get_nonexistent_preset(self, preset_service):
        """Test that retrieving non-existent preset raises ValueError."""
        fake_id = "00000000-0000-0000-0000-000000000000"

        with pytest.raises(ValueError, match="not found"):
            preset_service.get_preset(fake_id)

    def test_list_presets_empty(self, preset_service):
        """Test listing presets when none exist."""
        presets = preset_service.list_presets()
        assert presets == {}

    def test_update_preset_name(self, preset_service, sample_verification_config):
        """Test updating a preset's name."""
        preset = preset_service.create_preset("Original Name", sample_verification_config)
        preset_id = preset["id"]

        time.sleep(0.01)

        updated = preset_service.update_preset(preset_id, name="New Name")

        assert updated["id"] == preset_id
        assert updated["name"] == "New Name"
        assert updated["updated_at"] != preset["updated_at"]

    def test_delete_preset(self, preset_service, sample_verification_config):
        """Test deleting a preset."""
        preset = preset_service.create_preset("To Delete", sample_verification_config)
        preset_id = preset["id"]

        assert preset_service.get_preset(preset_id) is not None

        preset_service.delete_preset(preset_id)

        with pytest.raises(ValueError, match="not found"):
            preset_service.get_preset(preset_id)

    def test_directory_creation(self, empty_preset_service):
        """Test that preset directory is created automatically on init."""
        assert empty_preset_service.presets_dir_path.exists()
        assert empty_preset_service.presets_dir_path.is_dir()

        presets = empty_preset_service.list_presets()
        assert presets == {}

    def test_preset_file_persistence(self, temp_presets_dir, sample_verification_config):
        """Test that presets persist as individual files across service instances."""
        service1 = BenchmarkPresetService(presets_dir_path=temp_presets_dir)
        preset = service1.create_preset("Persistent", sample_verification_config)
        preset_id = preset["id"]

        json_files = list(temp_presets_dir.glob("*.json"))
        assert len(json_files) == 1

        service2 = BenchmarkPresetService(presets_dir_path=temp_presets_dir)
        retrieved = service2.get_preset(preset_id)

        assert retrieved["name"] == "Persistent"

    def test_individual_preset_files(self, preset_service, sample_verification_config, temp_presets_dir):
        """Test that each preset creates its own JSON file."""
        preset_service.create_preset("Quick Test", sample_verification_config)
        preset_service.create_preset("Full Benchmark", sample_verification_config)
        preset_service.create_preset("Custom Config", sample_verification_config)

        json_files = sorted(temp_presets_dir.glob("*.json"))
        assert len(json_files) == 3

        for filepath in json_files:
            with open(filepath) as f:
                preset_data = json.load(f)
            assert "id" in preset_data
            assert "name" in preset_data

    def test_preset_file_deletion(self, preset_service, sample_verification_config, temp_presets_dir):
        """Test that deleting a preset removes its file."""
        preset = preset_service.create_preset("To Delete", sample_verification_config)
        preset_id = preset["id"]

        json_files = list(temp_presets_dir.glob("*.json"))
        assert len(json_files) == 1

        preset_service.delete_preset(preset_id)

        json_files = list(temp_presets_dir.glob("*.json"))
        assert len(json_files) == 0

    def test_full_crud_cycle(self, preset_service, sample_verification_config):
        """Test complete CRUD cycle for a preset."""
        created = preset_service.create_preset(
            "CRUD Test",
            sample_verification_config,
            description="Testing CRUD operations",
        )
        preset_id = created["id"]

        retrieved = preset_service.get_preset(preset_id)
        assert retrieved["name"] == "CRUD Test"

        updated = preset_service.update_preset(preset_id, name="Updated CRUD Test")
        assert updated["name"] == "Updated CRUD Test"

        preset_service.delete_preset(preset_id)
        with pytest.raises(ValueError):
            preset_service.get_preset(preset_id)
