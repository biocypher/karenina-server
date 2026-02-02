"""MCP (Model Context Protocol) validation API handlers."""

import logging
from typing import Any

from fastapi import HTTPException
from pydantic import BaseModel

try:
    import karenina.adapters.langchain.mcp  # noqa: F401 - Test if MCP module is available

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

    # =============================================================================
    # V2 RESTful Routes
    # =============================================================================

    @app.get("/api/v2/mcp/presets")  # type: ignore[misc]
    async def get_mcp_presets_v2() -> dict[str, Any]:
        """V2: Get all MCP preset configurations.

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

    @app.post("/api/v2/mcp/servers/validation", response_model=MCPValidationResponse)  # type: ignore[misc]
    async def validate_mcp_server_v2(request: MCPValidationRequest) -> MCPValidationResponse:
        """V2: Validate an MCP server and return available tools."""
        # Import LLM_AVAILABLE from server to maintain compatibility with tests
        from .. import server

        if not getattr(server, "LLM_AVAILABLE", LLM_AVAILABLE):
            raise HTTPException(status_code=503, detail="LLM functionality not available")

        try:
            from karenina.adapters.langchain.mcp import create_mcp_client_and_tools

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

    @app.put("/api/v2/mcp/presets/{name}")  # type: ignore[misc]
    async def save_mcp_preset_v2(name: str, request: MCPPresetSaveRequest) -> dict[str, Any]:
        """V2: Create or update an MCP preset.

        Uses PUT since presets are identified by name and can be created or updated.
        """
        try:
            from ..services.mcp_preset_service import MCPPresetService

            # Ensure name in URL matches request body (or override body with URL name)
            request.name = name

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

    @app.delete("/api/v2/mcp/presets/{name}")  # type: ignore[misc]
    async def delete_mcp_preset_v2(name: str) -> dict[str, Any]:
        """V2: Delete an MCP preset by name.

        Uses DELETE method and name in URL path instead of request body.
        """
        try:
            from ..services.mcp_preset_service import MCPPresetService

            service = MCPPresetService()
            service.delete_preset(name=name)

            logger.info(f"Deleted MCP preset '{name}'")
            return {"success": True}

        except ValueError as e:
            logger.warning(f"Error deleting MCP preset: {e}")
            raise HTTPException(status_code=404, detail=str(e)) from e
        except Exception as e:
            logger.error(f"Error deleting MCP preset: {e}")
            raise HTTPException(status_code=500, detail=f"Failed to delete MCP preset: {str(e)}") from e

    # =============================================================================
    # V2 RESTful Routes
    # =============================================================================
    # The following routes provide RESTful noun-based naming conventions.
    # They delegate to the v1 handlers above for consistent behavior.
    #
    # V1 → V2 Route Mapping:
    #   GET  /api/get-mcp-preset-configs     → GET    /api/v2/mcp/presets
    #   POST /api/validate-mcp-server        → POST   /api/v2/mcp/servers/validation
    #   POST /api/save-mcp-preset            → PUT    /api/v2/mcp/presets/{name}
    #   POST /api/delete-mcp-preset          → DELETE /api/v2/mcp/presets/{name}
    # =============================================================================

    @app.get("/api/v2/mcp/presets")  # type: ignore[misc]
    async def get_mcp_presets_v2() -> dict[str, Any]:
        """V2: Get all MCP preset configurations.

        RESTful equivalent of GET /api/get-mcp-preset-configs
        """
        return await get_mcp_preset_configs()  # type: ignore[no-any-return]

    @app.post("/api/v2/mcp/servers/validation", response_model=MCPValidationResponse)  # type: ignore[misc]
    async def validate_mcp_server_v2(request: MCPValidationRequest) -> MCPValidationResponse:
        """V2: Validate an MCP server and return available tools.

        RESTful equivalent of POST /api/validate-mcp-server
        """
        return await validate_mcp_server_endpoint(request)

    @app.put("/api/v2/mcp/presets/{name}")  # type: ignore[misc]
    async def save_mcp_preset_v2(name: str, request: MCPPresetSaveRequest) -> dict[str, Any]:
        """V2: Create or update an MCP preset.

        RESTful equivalent of POST /api/save-mcp-preset
        Uses PUT since presets are identified by name and can be created or updated.
        """
        # Ensure name in URL matches request body (or override body with URL name)
        request.name = name
        return await save_mcp_preset(request)  # type: ignore[no-any-return]

    @app.delete("/api/v2/mcp/presets/{name}")  # type: ignore[misc]
    async def delete_mcp_preset_v2(name: str) -> dict[str, Any]:
        """V2: Delete an MCP preset by name.

        RESTful equivalent of POST /api/delete-mcp-preset
        Uses DELETE method and name in URL path instead of request body.
        """
        request = MCPPresetDeleteRequest(name=name)
        return await delete_mcp_preset(request)  # type: ignore[no-any-return]
