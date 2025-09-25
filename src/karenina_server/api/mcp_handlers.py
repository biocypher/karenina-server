"""MCP (Model Context Protocol) validation API handlers."""

from typing import Any

from fastapi import HTTPException

try:
    import karenina.llm  # noqa: F401 - Test if LLM module is available

    LLM_AVAILABLE = True
except ImportError:
    LLM_AVAILABLE = False


def register_mcp_routes(app: Any, MCPValidationRequest: Any, MCPValidationResponse: Any) -> None:
    """Register MCP validation-related routes."""

    @app.post("/api/validate-mcp-server", response_model=MCPValidationResponse)  # type: ignore[misc]
    async def validate_mcp_server_endpoint(request: MCPValidationRequest) -> MCPValidationResponse:
        """Validate an MCP server and return available tools."""
        # Import LLM_AVAILABLE from server to maintain compatibility with tests
        from .. import server

        if not getattr(server, "LLM_AVAILABLE", LLM_AVAILABLE):
            raise HTTPException(status_code=503, detail="LLM functionality not available")

        try:
            from karenina.llm.mcp_utils import create_mcp_client_and_tools

            # Create MCP URLs dict for the single server
            mcp_urls_dict = {request.server_name: request.server_url}

            # Get MCP client and tools (no filtering for validation)
            _, tools = await create_mcp_client_and_tools(mcp_urls_dict, tool_filter=None)

            # Extract tool information
            tool_list = []
            for tool in tools:
                tool_name = getattr(tool, "name", "Unknown")
                tool_description = getattr(tool, "description", None)
                tool_list.append({
                    "name": tool_name,
                    "description": tool_description
                })

            return MCPValidationResponse(
                success=True,
                tools=tool_list,
                error=None
            )

        except Exception as e:
            return MCPValidationResponse(
                success=False,
                tools=None,
                error=f"Failed to validate MCP server: {e!s}"
            )