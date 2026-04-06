"""ollama-mcp-bridge — Security-first bridge connecting Ollama models to MCP tool servers.

Copyright 2026 BJ Gilbert
Licensed under the Apache License, Version 2.0

Public API:

    from ollama_mcp_bridge import Bridge, BridgeConfig

    async with Bridge.from_config("bridge.toml") as bridge:
        result = await bridge.run("List my memories", model="llama3.1:8b")
"""

from .bridge import Bridge
from .config import BridgeConfig, SecurityConfig, ServerConfig, load_config
from .errors import (
    BridgeError,
    ConfigError,
    ConfirmationDeniedError,
    LoopError,
    MaxTurnsError,
    MCPConnectionError,
    MCPError,
    MCPTimeoutError,
    MCPToolError,
    OllamaConnectionError,
    OllamaError,
    OllamaModelError,
    OllamaResponseError,
    ParameterRejectedError,
    RateLimitError,
    SecurityError,
    StuckModelError,
    ToolBlockedError,
    ToolIntegrityError,
)
from .types import (
    ActionClass,
    AuditEntry,
    BridgeResult,
    ServerHealth,
    StreamEvent,
    StreamEventType,
    ToolCallRecord,
)

__version__ = "0.1.0"

__all__ = [
    # Core
    "Bridge",
    # Config
    "BridgeConfig",
    "SecurityConfig",
    "ServerConfig",
    "load_config",
    # Results
    "BridgeResult",
    "StreamEvent",
    "StreamEventType",
    "ToolCallRecord",
    "AuditEntry",
    "ActionClass",
    "ServerHealth",
    # Errors
    "BridgeError",
    "ConfigError",
    "OllamaError",
    "OllamaConnectionError",
    "OllamaModelError",
    "OllamaResponseError",
    "MCPError",
    "MCPConnectionError",
    "MCPToolError",
    "MCPTimeoutError",
    "SecurityError",
    "ToolBlockedError",
    "ParameterRejectedError",
    "RateLimitError",
    "ConfirmationDeniedError",
    "ToolIntegrityError",
    "LoopError",
    "MaxTurnsError",
    "StuckModelError",
]
