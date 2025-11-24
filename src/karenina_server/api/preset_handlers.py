"""API handlers for benchmark configuration preset management."""
# ruff: noqa: B904  # Intentionally suppress exception chaining for security

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from karenina.schemas.workflow.verification import VerificationConfig
from pydantic import BaseModel, Field, field_validator

from ..services.preset_service import BenchmarkPresetService

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize service
preset_service = BenchmarkPresetService()


def sanitize_error_message(error: Exception) -> str:
    """Sanitize error messages to prevent information disclosure.

    Args:
        error: The exception to sanitize

    Returns:
        Safe error message for API response
    """
    error_str = str(error)

    # Don't expose file paths in error messages
    if "Path outside allowed directories" in error_str:
        return "Invalid file path specified"

    # Don't expose internal file system errors
    if any(keyword in error_str.lower() for keyword in ["permission denied", "no such file", "directory"]):
        return "File system operation failed"

    # Generic fallback for unexpected errors
    if len(error_str) > 200:
        return "Preset operation failed"

    return error_str


class PresetSummary(BaseModel):
    """Summary information about a preset."""

    id: str
    name: str
    description: str | None
    created_at: str
    updated_at: str
    summary: dict[str, Any] = Field(
        default_factory=dict,
        description="Summary statistics (model counts, enabled features, etc.)",
    )


class PresetDetail(BaseModel):
    """Full preset including configuration."""

    id: str
    name: str
    description: str | None
    config: dict[str, Any]  # VerificationConfig as dict
    created_at: str
    updated_at: str


class CreatePresetRequest(BaseModel):
    """Request model for creating a preset."""

    name: str = Field(..., min_length=1, max_length=100, description="Preset name")
    description: str | None = Field(None, max_length=500, description="Optional description")
    config: VerificationConfig = Field(..., description="Verification configuration")


class UpdatePresetRequest(BaseModel):
    """Request model for updating a preset."""

    name: str | None = Field(None, min_length=1, max_length=100, description="New preset name")
    description: str | None = Field(None, max_length=500, description="New description (empty string to clear)")
    config: VerificationConfig | None = Field(None, description="New verification configuration")

    @field_validator("name")
    @classmethod
    def validate_name_not_empty(_cls, v: str | None) -> str | None:  # noqa: N805
        """Validate that name is not just whitespace if provided."""
        if v is not None and len(v.strip()) == 0:
            raise ValueError("Preset name cannot be empty or whitespace")
        return v


def _generate_preset_summary(config_dict: dict[str, Any]) -> dict[str, Any]:
    """Generate summary statistics for a preset configuration.

    Args:
        config_dict: VerificationConfig as dictionary

    Returns:
        Dictionary with summary statistics
    """
    summary: dict[str, Any] = {
        "answering_model_count": len(config_dict.get("answering_models", [])),
        "parsing_model_count": len(config_dict.get("parsing_models", [])),
        "replicate_count": config_dict.get("replicate_count", 1),
        "enabled_features": [],
        "interfaces": [],
    }

    # Total model count
    summary["total_model_count"] = summary["answering_model_count"] + summary["parsing_model_count"]

    # Collect enabled features
    if config_dict.get("rubric_enabled"):
        summary["enabled_features"].append("rubric")
    if config_dict.get("abstention_enabled"):
        summary["enabled_features"].append("abstention")
    if config_dict.get("deep_judgment_enabled"):
        summary["enabled_features"].append("deep_judgment")
    if config_dict.get("deep_judgment_search_enabled"):
        summary["enabled_features"].append("deep_judgment_search")

    # Check few-shot
    few_shot_config = config_dict.get("few_shot_config")
    if few_shot_config and few_shot_config.get("enabled"):
        summary["enabled_features"].append("few_shot")

    # Collect unique interfaces
    all_models = config_dict.get("answering_models", []) + config_dict.get("parsing_models", [])
    interfaces = set()
    for model in all_models:
        if isinstance(model, dict) and "interface" in model:
            interfaces.add(model["interface"])
    summary["interfaces"] = sorted(interfaces)

    return summary


@router.get("/presets", response_model=dict[str, list[PresetSummary]])
async def list_presets() -> dict[str, list[PresetSummary]]:
    """Get all presets with summary information."""
    try:
        presets_dict = preset_service.list_presets()

        # Convert to list of summaries
        presets_list = []
        for _preset_id, preset_data in presets_dict.items():
            # Generate summary from config
            summary = _generate_preset_summary(preset_data.get("config", {}))

            preset_summary = PresetSummary(
                id=preset_data["id"],
                name=preset_data["name"],
                description=preset_data.get("description"),
                created_at=preset_data["created_at"],
                updated_at=preset_data["updated_at"],
                summary=summary,
            )
            presets_list.append(preset_summary)

        # Sort by name
        presets_list.sort(key=lambda p: p.name.lower())

        return {"presets": presets_list}

    except Exception as e:
        logger.error(f"Error listing presets: {e}")
        raise HTTPException(status_code=500, detail=sanitize_error_message(e))


@router.get("/presets/{preset_id}", response_model=dict[str, PresetDetail])
async def get_preset(preset_id: str) -> dict[str, PresetDetail]:
    """Get a specific preset by ID."""
    try:
        preset_data = preset_service.get_preset(preset_id)

        preset_detail = PresetDetail(
            id=preset_data["id"],
            name=preset_data["name"],
            description=preset_data.get("description"),
            config=preset_data["config"],
            created_at=preset_data["created_at"],
            updated_at=preset_data["updated_at"],
        )

        return {"preset": preset_detail}

    except ValueError as e:
        # Preset not found
        raise HTTPException(status_code=404, detail=sanitize_error_message(e))
    except Exception as e:
        logger.error(f"Error getting preset {preset_id}: {e}")
        raise HTTPException(status_code=500, detail=sanitize_error_message(e))


@router.post("/presets", response_model=dict[str, Any], status_code=201)
async def create_preset(request: CreatePresetRequest) -> dict[str, Any]:
    """Create a new preset."""
    try:
        preset_data = preset_service.create_preset(
            name=request.name,
            config=request.config,
            description=request.description,
        )

        return {
            "message": f"Preset '{request.name}' created successfully",
            "preset": preset_data,
        }

    except ValueError as e:
        # Validation error (duplicate name, etc.)
        raise HTTPException(status_code=400, detail=sanitize_error_message(e))
    except Exception as e:
        logger.error(f"Error creating preset: {e}")
        raise HTTPException(status_code=500, detail=sanitize_error_message(e))


@router.put("/presets/{preset_id}", response_model=dict[str, Any])
async def update_preset(preset_id: str, request: UpdatePresetRequest) -> dict[str, Any]:
    """Update an existing preset."""
    try:
        # Check if at least one field is provided
        if request.name is None and request.config is None and request.description is None:
            raise ValueError("At least one field (name, config, or description) must be provided for update")

        preset_data = preset_service.update_preset(
            preset_id=preset_id,
            name=request.name,
            config=request.config,
            description=request.description,
        )

        return {
            "message": f"Preset '{preset_data['name']}' updated successfully",
            "preset": preset_data,
        }

    except ValueError as e:
        # Not found or validation error
        error_str = str(e)
        if "not found" in error_str.lower():
            raise HTTPException(status_code=404, detail=sanitize_error_message(e))
        raise HTTPException(status_code=400, detail=sanitize_error_message(e))
    except Exception as e:
        logger.error(f"Error updating preset {preset_id}: {e}")
        raise HTTPException(status_code=500, detail=sanitize_error_message(e))


@router.delete("/presets/{preset_id}", response_model=dict[str, str])
async def delete_preset(preset_id: str) -> dict[str, str]:
    """Delete a preset."""
    try:
        preset_service.delete_preset(preset_id)
        return {"message": "Preset deleted successfully"}

    except ValueError as e:
        # Preset not found
        raise HTTPException(status_code=404, detail=sanitize_error_message(e))
    except Exception as e:
        logger.error(f"Error deleting preset {preset_id}: {e}")
        raise HTTPException(status_code=500, detail=sanitize_error_message(e))


@router.get("/presets-status")
async def get_presets_file_status() -> dict[str, Any]:
    """Get status information about the presets file."""
    try:
        return preset_service.get_directory_status()
    except Exception as e:
        logger.error(f"Error getting presets file status: {e}")
        raise HTTPException(status_code=500, detail=sanitize_error_message(e))
