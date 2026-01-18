"""API handlers for configuration management."""
# ruff: noqa: B904  # Intentionally suppress exception chaining for security

import logging
from typing import Any

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, Field, field_validator

from ..services.config_service import ConfigurationService
from ..services.defaults_service import DefaultsService

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize services
config_service = ConfigurationService()
defaults_service = DefaultsService()


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
    if len(error_str) > 100:
        return "Configuration operation failed"

    return error_str


class EnvVarUpdate(BaseModel):
    """Model for updating an environment variable."""

    key: str = Field(
        ...,
        min_length=1,
        max_length=100,
        pattern=r"^[A-Z_][A-Z0-9_]*$",
        description="Environment variable key (uppercase, underscores allowed)",
    )
    value: str = Field(..., max_length=1000, description="Environment variable value")


class EnvVarBulkUpdate(BaseModel):
    """Model for bulk updating environment variables."""

    variables: list[EnvVarUpdate] = Field(..., description="List of environment variables to update (max 50)")

    @field_validator("variables")
    @classmethod
    def validate_variables_length(_cls, v: list[EnvVarUpdate]) -> list[EnvVarUpdate]:  # noqa: N805
        """Validate that the list doesn't exceed 50 items."""
        if len(v) > 50:
            raise ValueError("Maximum 50 variables allowed in bulk update")
        return v


class EnvFileUpdate(BaseModel):
    """Model for updating entire .env file contents."""

    content: str = Field(..., max_length=50000, description="Complete .env file contents (max 50KB)")


class ConfigStatus(BaseModel):
    """Model for configuration status response."""

    variables: dict[str, str]


class DefaultConfig(BaseModel):
    """Model for default LLM configuration."""

    default_interface: str = "langchain"  # langchain, openrouter, or openai_endpoint
    default_provider: str = "anthropic"  # for langchain
    default_model: str = "claude-haiku-4-5"  # default model
    default_endpoint_base_url: str | None = None  # for openai_endpoint interface


# =============================================================================
# V2 API Endpoints (RESTful)
# =============================================================================


@router.get("/v2/config/env-vars", response_model=dict[str, str])
async def get_env_vars_v2() -> dict[str, str]:
    """Get current environment variables - masked (v2 endpoint)."""
    try:
        return config_service.read_env_vars(mask_secrets=True)
    except Exception as e:
        logger.error(f"Error reading environment variables: {e}")
        raise HTTPException(status_code=500, detail=sanitize_error_message(e))


@router.get("/v2/config/env-vars/unmasked", response_model=dict[str, str])
async def get_env_vars_unmasked_v2() -> dict[str, str]:
    """Get current environment variables - unmasked (v2 endpoint)."""
    try:
        return config_service.read_env_vars(mask_secrets=False)
    except Exception as e:
        logger.error(f"Error reading environment variables: {e}")
        raise HTTPException(status_code=500, detail=sanitize_error_message(e))


@router.put("/v2/config/env-vars")
async def update_env_var_v2(update: EnvVarUpdate) -> dict[str, str]:
    """Update a single environment variable (v2 endpoint)."""
    try:
        config_service.update_env_var(update.key, update.value)
        return {"message": f"Successfully updated {update.key}"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=sanitize_error_message(e))
    except Exception as e:
        logger.error(f"Error updating environment variable: {e}")
        raise HTTPException(status_code=500, detail=sanitize_error_message(e))


@router.put("/v2/config/env-vars/bulk")
async def update_env_vars_bulk_v2(update: EnvVarBulkUpdate) -> dict[str, str | list[str]]:
    """Update multiple environment variables atomically (v2 endpoint)."""
    try:
        # Convert to list of tuples for the service method
        updates = [(var.key, var.value) for var in update.variables]

        # Perform atomic bulk update
        config_service.update_env_vars_bulk(updates)

        updated_keys = [var.key for var in update.variables]
        return {"message": "All variables updated successfully", "updated": updated_keys}

    except ValueError as e:
        raise HTTPException(status_code=400, detail=sanitize_error_message(e))
    except Exception as e:
        logger.error(f"Error during bulk update: {e}")
        raise HTTPException(status_code=500, detail=sanitize_error_message(e))


@router.delete("/v2/config/env-vars/{key}")
async def delete_env_var_v2(key: str) -> dict[str, str]:
    """Remove an environment variable (v2 endpoint)."""
    try:
        config_service.remove_env_var(key)
        return {"message": f"Successfully removed {key}"}
    except Exception as e:
        logger.error(f"Error removing environment variable: {e}")
        raise HTTPException(status_code=500, detail=sanitize_error_message(e))


@router.get("/v2/config/env-file")
async def get_env_file_contents_v2() -> dict[str, str]:
    """Get raw .env file contents (v2 endpoint)."""
    try:
        content = config_service.get_env_file_contents()
        return {"content": content}
    except Exception as e:
        logger.error(f"Error reading .env file: {e}")
        raise HTTPException(status_code=500, detail=sanitize_error_message(e))


@router.put("/v2/config/env-file")
async def update_env_file_v2(update: EnvFileUpdate) -> dict[str, str]:
    """Update the entire .env file contents (v2 endpoint)."""
    try:
        config_service.update_env_file_contents(update.content)
        return {"message": "Successfully updated .env file"}
    except Exception as e:
        logger.error(f"Error updating .env file: {e}")
        raise HTTPException(status_code=500, detail=sanitize_error_message(e))


@router.get("/v2/config/defaults", response_model=DefaultConfig)
async def get_default_config_v2() -> DefaultConfig:
    """Get default LLM configuration settings (v2 endpoint)."""
    try:
        defaults = defaults_service.get_defaults()
        # These fields are guaranteed to exist and be strings due to fallback defaults
        interface = defaults["default_interface"]
        provider = defaults["default_provider"]
        model = defaults["default_model"]
        assert isinstance(interface, str)
        assert isinstance(provider, str)
        assert isinstance(model, str)

        return DefaultConfig(
            default_interface=interface,
            default_provider=provider,
            default_model=model,
            default_endpoint_base_url=defaults.get("default_endpoint_base_url"),
        )
    except Exception as e:
        logger.error(f"Error getting default configuration: {e}")
        raise HTTPException(status_code=500, detail=sanitize_error_message(e))


@router.put("/v2/config/defaults")
async def update_default_config_v2(config: DefaultConfig) -> dict[str, str | dict[str, str | None]]:
    """Update default LLM configuration settings (v2 endpoint)."""
    try:
        defaults_dict = {
            "default_interface": config.default_interface,
            "default_provider": config.default_provider,
            "default_model": config.default_model,
            "default_endpoint_base_url": config.default_endpoint_base_url,
        }

        defaults_service.save_defaults(defaults_dict)

        return {"message": "Default configuration saved successfully", "config": defaults_dict}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=sanitize_error_message(e))
    except Exception as e:
        logger.error(f"Error updating default configuration: {e}")
        raise HTTPException(status_code=500, detail=sanitize_error_message(e))


@router.get("/v2/config/defaults/status")
async def get_defaults_file_status_v2() -> dict[str, Any]:
    """Get status information about the defaults file (v2 endpoint)."""
    try:
        return defaults_service.get_file_status()
    except Exception as e:
        logger.error(f"Error getting defaults file status: {e}")
        raise HTTPException(status_code=500, detail=sanitize_error_message(e))


@router.post("/v2/config/defaults/reset")
async def reset_defaults_v2() -> dict[str, str]:
    """Reset defaults to fallback values (v2 endpoint)."""
    try:
        defaults_service.reset_to_fallback()
        return {"message": "Defaults reset to fallback values"}
    except Exception as e:
        logger.error(f"Error resetting defaults: {e}")
        raise HTTPException(status_code=500, detail=sanitize_error_message(e))
