"""MCP (Model Context Protocol) validation API handlers."""

import json
import logging
import os
from typing import Any

from fastapi import HTTPException

try:
    import karenina.infrastructure.llm  # noqa: F401 - Test if LLM module is available

    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False

logger = logging.getLogger(__name__)


def register_mcp_routes(app: Any, MCPValidationRequest: Any, MCPValidationResponse: Any) -> None:
    """Register MCP validation-related routes."""

    @app.get("/api/get-mcp-preset-configs")  # type: ignore[misc]
    async def get_mcp_preset_configs() -> dict[str, Any]:
        """Get MCP preset configurations from MCP_CONFIG environment variable."""
        try:
            mcp_config_str = os.environ.get("MCP_CONFIG")
            if not mcp_config_str:
                return {"presets": {}}

            # Parse the JSON string
            try:
                mcp_config = json.loads(mcp_config_str)
            except json.JSONDecodeError as e:
                logger.warning(f"Invalid JSON in MCP_CONFIG environment variable: {e}")
                return {"presets": {}, "error": "Invalid JSON in MCP_CONFIG"}

            # Validate the configuration structure
            if not isinstance(mcp_config, dict):
                logger.warning("MCP_CONFIG must be a dictionary")
                return {"presets": {}, "error": "MCP_CONFIG must be a dictionary"}

            validated_presets = {}
            for server_name, config in mcp_config.items():
                if not isinstance(config, dict):
                    logger.warning(f"Skipping invalid config for server '{server_name}': not a dictionary")
                    continue

                if "url" not in config:
                    logger.warning(f"Skipping server '{server_name}': missing required 'url' field")
                    continue

                # Build validated configuration
                preset_config = {"name": server_name, "url": config["url"]}

                # Add tools if specified
                if "tools" in config:
                    if isinstance(config["tools"], list):
                        preset_config["tools"] = config["tools"]
                    else:
                        logger.warning(f"Invalid 'tools' for server '{server_name}': must be a list")

                validated_presets[server_name] = preset_config

            return {"presets": validated_presets}

        except Exception as e:
            logger.error(f"Error processing MCP preset configurations: {e}")
            return {"presets": {}, "error": f"Failed to process MCP_CONFIG: {str(e)}"}

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
