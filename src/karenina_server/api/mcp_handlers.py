"""MCP (Model Context Protocol) validation API handlers."""

import logging
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel

try:
    import karenina.infrastructure.llm  # noqa: F401 - Test if LLM module is available

    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

logger = logging.getLogger(__name__)


class MCPPresetSaveRequest(BaseModel):
    """Request model for saving MCP presets."""

    name: str
    url: str
    tools: list[str] | None = None


class MCPPresetDeleteRequest(BaseModel):
    """Request model for deleting MCP presets."""

    name: str


def register_mcp_routes(app: Any, MCPValidationRequest: Any, MCPValidationResponse: Any) -> None:
    """Register MCP validation-related routes."""

    @app.get("/api/get-mcp-preset-configs")  # type: ignore[misc]
    async def get_mcp_preset_configs() -> dict[str, Any]:
        """Get MCP preset configurations from mcp_presets directory.

        The directory location can be configured via MCP_PRESETS_DIR environment variable.
        Defaults to mcp_presets/ in the current working directory.
        """
        try:
            from ..services.mcp_preset_service import MCPPresetService

            service = MCPPresetService()
            all_presets = service.list_all_presets()

            return {"presets": all_presets}

        except Exception as e:
            logger.error(f"Error getting MCP preset configurations: {e}")
            return {"presets": {}, "error": f"Failed to get MCP presets: {str(e)}"}

    @app.post("/api/validate-mcp-server", response_model=MCPValidationResponse)  # type: ignore[misc]
    async def validate_mcp_server_endpoint(request: MCPValidationRequest) -> MCPValidationResponse:
        """Validate an MCP server and return available tools."""
        # Import LLM_AVAILABLE from server to maintain compatibility with tests
        from .. import server

        if not getattr(server, "LLM_AVAILABLE", LLM_AVAILABLE):
            raise HTTPException(status_code=503, detail="LLM functionality not available")

        try:
            from karenina.infrastructure.llm.mcp_utils import create_mcp_client_and_tools

            # Create MCP URLs dict for the single server
            mcp_urls_dict = {request.server_name: request.server_url}

            # Get MCP client and tools (no filtering for validation)
            _, tools = await create_mcp_client_and_tools(mcp_urls_dict, tool_filter=None)

            # Extract tool information
            tool_list = []
            for tool in tools:
                tool_name = getattr(tool, "name", "Unknown")
                tool_description = getattr(tool, "description", None)
                tool_list.append({"name": tool_name, "description": tool_description})

            return MCPValidationResponse(success=True, tools=tool_list, error=None)

        except Exception as e:
            return MCPValidationResponse(success=False, tools=None, error=f"Failed to validate MCP server: {e!s}")

    @app.post("/api/save-mcp-preset")  # type: ignore[misc]
    async def save_mcp_preset(request: MCPPresetSaveRequest) -> dict[str, Any]:
        """Save a new or update existing MCP preset."""
        try:
            from ..services.mcp_preset_service import MCPPresetService

            service = MCPPresetService()
            preset = service.save_preset(
                name=request.name,
                url=request.url,
                tools=request.tools,
            )

            logger.info(f"Saved MCP preset '{request.name}'")
            return {"success": True, "preset": preset}

        except ValueError as e:
            logger.warning(f"Validation error saving MCP preset: {e}")
            raise HTTPException(status_code=400, detail=str(e)) from e
        except Exception as e:
            logger.error(f"Error saving MCP preset: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to save MCP preset: {str(e)}") from e

    @app.post("/api/delete-mcp-preset")  # type: ignore[misc]
    async def delete_mcp_preset(request: MCPPresetDeleteRequest) -> dict[str, Any]:
        """Delete a user-saved MCP preset."""
        try:
            from ..services.mcp_preset_service import MCPPresetService

            service = MCPPresetService()
            service.delete_preset(name=request.name)

            logger.info(f"Deleted MCP preset '{request.name}'")
            return {"success": True}

        except ValueError as e:
            logger.warning(f"Error deleting MCP preset: {e}")
            raise HTTPException(status_code=404, detail=str(e)) from e
        except Exception as e:
            logger.error(f"Error deleting MCP preset: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to delete MCP preset: {str(e)}") from e
