"""Exception hierarchy for ollama-mcp-bridge.

All exceptions inherit from BridgeError. Consumer code can catch BridgeError
for a broad handler, or specific subclasses for fine-grained control.

Categories:
- OllamaError: Ollama API / model issues
- MCPError: MCP server / tool execution issues
- SecurityError: security policy enforcement
- LoopError: agent loop orchestration issues
"""

from __future__ import annotations


class BridgeError(Exception):
    """Base exception for all bridge errors."""

    pass


# --- Transport: Ollama ---


class OllamaError(BridgeError):
    """Base for Ollama-related errors."""

    pass


class OllamaConnectionError(OllamaError):
    """Cannot reach Ollama server."""

    pass


class OllamaModelError(OllamaError):
    """Model not found, not loaded, or unavailable."""

    pass


class OllamaResponseError(OllamaError):
    """Malformed or unexpected response from Ollama."""

    pass


# --- Transport: MCP ---


class MCPError(BridgeError):
    """Base for MCP-related errors."""

    pass


class MCPConnectionError(MCPError):
    """MCP server won't start or connection lost."""

    pass


class MCPToolError(MCPError):
    """Tool execution failed on MCP server side."""

    def __init__(self, message: str, safe_message: str | None = None):
        super().__init__(message)
        self.safe_message = safe_message or message


class MCPTimeoutError(MCPError):
    """Tool call timed out."""

    pass


# --- Security ---


class SecurityError(BridgeError):
    """Base for security enforcement errors."""

    pass


class ToolBlockedError(SecurityError):
    """Tool call blocked by security policy (SR-4 allowlist or SR-1 sanitization)."""

    def __init__(self, message: str, reason: str = ""):
        super().__init__(message)
        self.reason = reason or message


class ParameterRejectedError(SecurityError):
    """Parameter validation failed (SR-5)."""

    def __init__(self, message: str, errors: list[str] | None = None):
        super().__init__(message)
        self.validation_errors = errors or []


class RateLimitError(SecurityError):
    """Rate limit exceeded (SR-9)."""

    def __init__(self, message: str, retry_after_seconds: float = 0.0):
        super().__init__(message)
        self.retry_after_seconds = retry_after_seconds


class ConfirmationDeniedError(SecurityError):
    """User denied destructive action confirmation (SR-7)."""

    pass


class ToolIntegrityError(SecurityError):
    """Tool definition hash mismatch — possible rug pull (SR-2)."""

    def __init__(self, message: str, tool_name: str = "", server: str = ""):
        super().__init__(message)
        self.tool_name = tool_name
        self.server = server


class NoApprovedToolsError(SecurityError):
    """No tools available — pending tools require approval before use."""

    def __init__(self, message: str, pending_count: int = 0):
        super().__init__(message)
        self.pending_count = pending_count


# --- Loop ---


class LoopError(BridgeError):
    """Base for agent loop errors."""

    pass


class MaxTurnsError(LoopError):
    """Hit the maximum turn limit."""

    pass


class StuckModelError(LoopError):
    """Model is not making progress (empty responses, repeated failures)."""

    pass


# --- Configuration ---


class ConfigError(BridgeError):
    """Invalid bridge configuration."""

    pass
