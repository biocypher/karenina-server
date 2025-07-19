"""API handlers for configuration management."""

import logging

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel

from ..services.config_service import ConfigurationService
from ..services.defaults_service import DefaultsService

logger = logging.getLogger(__name__)
router = APIRouter()

# Initialize services
config_service = ConfigurationService()
defaults_service = DefaultsService()


class EnvVarUpdate(BaseModel):
    """Model for updating an environment variable."""

    key: str
    value: str


class EnvVarBulkUpdate(BaseModel):
    """Model for bulk updating environment variables."""

    variables: list[EnvVarUpdate]


class EnvFileUpdate(BaseModel):
    """Model for updating entire .env file contents."""

    content: str


class ConfigStatus(BaseModel):
    """Model for configuration status response."""

    variables: dict[str, str]


class DefaultConfig(BaseModel):
    """Model for default LLM configuration."""

    default_interface: str = "langchain"  # langchain or openrouter
    default_provider: str = "google_genai"  # for langchain
    default_model: str = "gemini-pro"  # default model


@router.get("/env-vars", response_model=dict[str, str])
async def get_env_vars():
    """Get current environment variables (masked for security)."""
    try:
        return config_service.read_env_vars(mask_secrets=True)
    except Exception as e:
        logger.error(f"Error reading environment variables: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/env-vars/unmasked", response_model=dict[str, str])
async def get_env_vars_unmasked():
    """Get current environment variables without masking (for show/hide functionality)."""
    try:
        return config_service.read_env_vars(mask_secrets=False)
    except Exception as e:
        logger.error(f"Error reading unmasked environment variables: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/env-file")
async def get_env_file_contents():
    """Get raw .env file contents."""
    try:
        content = config_service.get_env_file_contents()
        return {"content": content}
    except Exception as e:
        logger.error(f"Error reading .env file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/env-vars")
async def update_env_var(update: EnvVarUpdate):
    """Update a single environment variable."""
    try:
        config_service.update_env_var(update.key, update.value)
        return {"message": f"Successfully updated {update.key}"}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating environment variable: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/env-vars/bulk")
async def update_env_vars_bulk(update: EnvVarBulkUpdate):
    """Update multiple environment variables at once."""
    errors = []
    updated = []

    for var in update.variables:
        try:
            config_service.update_env_var(var.key, var.value)
            updated.append(var.key)
        except Exception as e:
            errors.append({"key": var.key, "error": str(e)})

    if errors:
        return {"message": "Partial update completed", "updated": updated, "errors": errors}

    return {"message": "All variables updated successfully", "updated": updated}


@router.put("/env-file")
async def update_env_file(update: EnvFileUpdate):
    """Update the entire .env file contents."""
    try:
        config_service.update_env_file_contents(update.content)
        return {"message": "Successfully updated .env file"}
    except Exception as e:
        logger.error(f"Error updating .env file: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/env-vars/{key}")
async def delete_env_var(key: str):
    """Remove an environment variable."""
    try:
        config_service.remove_env_var(key)
        return {"message": f"Successfully removed {key}"}
    except Exception as e:
        logger.error(f"Error removing environment variable: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status", response_model=ConfigStatus)
async def get_config_status():
    """Get configuration status."""
    try:
        # Get masked environment variables
        variables = config_service.read_env_vars(mask_secrets=True)

        return ConfigStatus(variables=variables)
    except Exception as e:
        logger.error(f"Error getting configuration status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/defaults", response_model=DefaultConfig)
async def get_default_config():
    """Get default LLM configuration settings."""
    try:
        defaults = defaults_service.get_defaults()
        return DefaultConfig(
            default_interface=defaults["default_interface"],
            default_provider=defaults["default_provider"],
            default_model=defaults["default_model"],
        )
    except Exception as e:
        logger.error(f"Error getting default configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.put("/defaults")
async def update_default_config(config: DefaultConfig):
    """Update default LLM configuration settings."""
    try:
        defaults_dict = {
            "default_interface": config.default_interface,
            "default_provider": config.default_provider,
            "default_model": config.default_model,
        }

        defaults_service.save_defaults(defaults_dict)

        return {"message": "Default configuration saved successfully", "config": defaults_dict}
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))
    except Exception as e:
        logger.error(f"Error updating default configuration: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/defaults/status")
async def get_defaults_file_status():
    """Get status information about the defaults file."""
    try:
        return defaults_service.get_file_status()
    except Exception as e:
        logger.error(f"Error getting defaults file status: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@router.post("/defaults/reset")
async def reset_defaults():
    """Reset defaults to fallback values."""
    try:
        defaults_service.reset_to_fallback()
        return {"message": "Defaults reset to fallback values"}
    except Exception as e:
        logger.error(f"Error resetting defaults: {e}")
        raise HTTPException(status_code=500, detail=str(e))
