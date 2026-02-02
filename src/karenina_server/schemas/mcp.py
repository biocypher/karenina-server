"""MCP (Model Context Protocol) Pydantic models for the Karenina API."""

from pydantic import BaseModel


class MCPTool(BaseModel):
    """Representation of an MCP tool."""

    name: str
    description: str | None = None


class MCPValidationRequest(BaseModel):
    """Request for validating an MCP server connection."""

    server_name: str
    server_url: str


class MCPValidationResponse(BaseModel):
    """Response for MCP validation endpoint."""

    success: bool
    tools: list[MCPTool] | None = None
    error: str | None = None
