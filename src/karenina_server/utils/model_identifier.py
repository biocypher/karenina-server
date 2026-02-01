"""Model identifier for type-safe model comparison keys."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ModelIdentifier:
    """Immutable identifier for a model configuration.

    Used as a hashable key for grouping verification results by model.
    Replaces string concatenation pattern (e.g., "model|[servers]").

    Attributes:
        answering_model: The model name/identifier.
        interface: The adapter interface (e.g., 'langchain', 'openrouter', 'claude_agent_sdk').
        mcp_servers: Sorted tuple of MCP server names for consistent hashing.
    """

    answering_model: str
    interface: str = "langchain"
    mcp_servers: tuple[str, ...] = ()

    @classmethod
    def from_config(cls, config: dict[str, Any]) -> ModelIdentifier:
        """Create identifier from a model configuration dict.

        Args:
            config: Dict with 'answering_model', optional 'interface', and optional 'mcp_config' keys.

        Returns:
            ModelIdentifier instance.
        """
        answering_model = config.get("answering_model", "")
        interface = config.get("interface", "langchain")
        mcp_config_str = str(config.get("mcp_config", "[]"))

        try:
            mcp_servers = json.loads(mcp_config_str)
            if not isinstance(mcp_servers, list):
                mcp_servers = []
        except Exception:
            mcp_servers = []

        return cls(
            answering_model=answering_model,
            interface=interface,
            mcp_servers=tuple(sorted(mcp_servers)),
        )

    @property
    def display_name(self) -> str:
        """Human-readable display name for the model.

        Returns:
            Formatted string like "interface:model-name +[server1, server2]".
        """
        base = f"{self.interface}:{self.answering_model}"
        if self.mcp_servers:
            return f"{base} +[{', '.join(self.mcp_servers)}]"
        return base

    def matches_servers(self, servers: list[str] | None) -> bool:
        """Check if this identifier matches the given MCP servers.

        Args:
            servers: List of MCP server names to compare.

        Returns:
            True if sorted servers match this identifier's servers.
        """
        if servers is None:
            servers = []
        return self.mcp_servers == tuple(sorted(servers))
