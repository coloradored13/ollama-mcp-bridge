"""Bridge — the single public entry point for the ollama-mcp-bridge library.

Usage:
    async with Bridge.from_config("bridge.toml") as bridge:
        result = await bridge.run("List my memories", model="llama3.1:8b")
        print(result.content)
        print(result.tool_calls)   # what tools were called
        print(result.audit_log)    # full security audit trail

DESIGN: Bridge is the ONLY public class. All internal components (SecurityGateway,
MCPClientManager, AgentLoop, ToolTranslator, OllamaClient, AuditLogger) are
encapsulated. Consumers interact through Bridge + config types + result types.

This encapsulation is a security feature: consumers cannot accidentally bypass
SecurityGateway by calling MCPClientManager directly. The internal wiring is:

    Bridge
      -> AgentLoop (orchestrates conversation turns)
           -> OllamaClient (sends prompts to model, receives tool calls)
           -> SecurityGateway (validates and executes tool calls)
                -> MCPClientManager (actually calls MCP servers — ONLY via SecurityGateway)
           -> ToolTranslator (converts between MCP and Ollama schema formats)

SecurityGateway.execute_tool() is the only path from "model wants to call a tool"
to "tool is actually called". There is no shortcut.

SYNC WRAPPER: run_sync() uses asyncio.run() for convenience in scripts. It will
raise RuntimeError if called from an existing event loop (Jupyter, FastAPI).
Use the async API (bridge.run()) in those contexts.
"""

from __future__ import annotations

import asyncio
import importlib.metadata
import logging
import uuid
from pathlib import Path
from typing import Any, AsyncIterator, Awaitable, Callable

from .audit import AuditLogger
from .config import BridgeConfig, load_config
from .errors import BridgeError, ConfigError, NoApprovedToolsError
from .loop import AgentLoop
from .mcp_client import MCPClientManager
from .ollama_client import OllamaClient
from .security import ApprovalCallback, ConfirmationCallback, SecurityGateway
from .translator import ToolTranslator
from .types import (
    AuditEntry,
    AuditEventType,
    BridgeResult,
    PendingToolApproval,
    ServerHealth,
    StreamEvent,
    ToolSchema,
)

logger = logging.getLogger(__name__)

try:
    _BRIDGE_VERSION = importlib.metadata.version("ollama-mcp-bridge")
except importlib.metadata.PackageNotFoundError:
    _BRIDGE_VERSION = "unknown"


class Bridge:
    """Security-first bridge connecting Ollama models to MCP tool servers.

    Bridge is the consumer-facing API. All internal components (SecurityGateway,
    MCPClientManager, AgentLoop, etc.) are encapsulated.

    Lifecycle:
    1. Create via Bridge(config) or Bridge.from_config(path)
    2. Use as async context manager: async with bridge:
    3. Call bridge.run() for single requests
    4. Context manager handles connect/disconnect
    """

    def __init__(self, config: BridgeConfig):
        self._config = config
        self._session_id = str(uuid.uuid4())[:8]

        # --- Internal component wiring ---
        # This is the composition root where the dependency graph is assembled.
        # The wiring enforces the security architecture:

        # Transport layer (Layer 1): raw I/O with Ollama and MCP servers
        self._ollama = OllamaClient(host=config.ollama_host)
        self._mcp = MCPClientManager(config.servers)

        # Security layer (Layer 3): OWNS MCPClientManager.
        # SecurityGateway takes MCPClientManager as a constructor arg and becomes
        # its sole consumer. This is the key architectural constraint — no other
        # component can reach MCP servers except through SecurityGateway.
        self._audit = AuditLogger(
            audit_file=config.logging.audit_file,
            session_id=self._session_id,
        )
        self._security = SecurityGateway(self._mcp, config, self._audit)

        # Translation layer (Layer 2): converts between MCP and Ollama formats
        self._translator = ToolTranslator()

        # Orchestration layer (Layer 4): drives the multi-turn conversation loop.
        # AgentLoop receives SecurityGateway (not MCPClientManager) — it can only
        # execute tools through the security pipeline.
        self._loop = AgentLoop(
            ollama=self._ollama,
            security=self._security,
            translator=self._translator,
            max_turns=config.security.max_turns,
            max_turns_hard_cap=config.security.max_turns_hard_cap,
        )

        self._connected = False

    @classmethod
    def from_config(cls, path: str | Path) -> "Bridge":
        """Create a Bridge from a TOML config file."""
        config = load_config(path)
        return cls(config)

    @classmethod
    def from_config_sync(cls, path: str | Path) -> "Bridge":
        """Create a Bridge from a TOML config file (sync).

        Note: sync wrapper uses asyncio.run() internally.
        Raises RuntimeError if called from an existing event loop
        (e.g., Jupyter, FastAPI). Use async API in those contexts.
        """
        config = load_config(path)
        return cls(config)

    def set_confirmation_callback(self, callback: ConfirmationCallback) -> None:
        """Set callback for destructive action confirmation.

        Callback signature:
            async def confirm(server: str, tool: str, action_class: str,
                            args: dict) -> bool
        """
        self._security.set_confirmation_callback(callback)

    def set_approval_callback(self, callback: ApprovalCallback) -> None:
        """Set callback for first-run tool approval.

        Callback receives all pending tools at once (batch-capable).
        Signature: async (list[PendingToolApproval]) -> dict[str, bool]
        where keys are "server:tool_name" and values are approve/deny.
        """
        self._security.set_approval_callback(callback)

    async def __aenter__(self) -> "Bridge":
        """Connect to all configured MCP servers."""
        await self._connect()
        return self

    async def __aexit__(self, *args: Any) -> None:
        """Disconnect from all MCP servers."""
        await self._disconnect()

    async def _connect(self) -> None:
        """Connect to servers and scan tools."""
        if self._connected:
            return

        self._audit.log_event(AuditEventType.SESSION_START)

        # Check Ollama health
        if not await self._ollama.check_health():
            raise BridgeError(
                f"Cannot reach Ollama at {self._config.ollama_host}. "
                "Is Ollama running?"
            )

        # Connect to MCP servers and scan tools
        await self._security.connect_and_scan()
        self._connected = True

        approved = self._security.get_approved_tools()
        pending = self._security.get_pending_tools()
        logger.info(
            "Bridge connected: %d approved tools, %d pending approval, across %d servers",
            len(approved),
            len(pending),
            len(self._config.servers),
        )

        if pending:
            logger.warning(
                "%d tool(s) require first-run approval before they can be used. "
                "Set an approval callback via bridge.set_approval_callback() "
                "before connecting, or configure auto_approve_first_seen=true.",
                len(pending),
            )

    async def _disconnect(self) -> None:
        """Disconnect and clean up."""
        if not self._connected:
            return

        self._audit.log_event(AuditEventType.SESSION_END)
        await self._security.disconnect_all()
        self._connected = False

    async def run(
        self,
        prompt: str,
        model: str,
        system_prompt: str | None = None,
        max_turns: int | None = None,
        on_event: Callable[[StreamEvent], Awaitable[None]] | None = None,
    ) -> BridgeResult:
        """Run a prompt through the bridge.

        Args:
            prompt: User prompt text.
            model: Ollama model name (e.g., 'llama3.1:8b').
            system_prompt: Optional system prompt.
            max_turns: Override max turns from config.
            on_event: Optional streaming event callback.

        Returns:
            BridgeResult with model response, tool call records, and audit log.

        Raises:
            BridgeError: For infrastructure failures (Ollama down, config invalid).
            Never raises from tool execution failures — those are captured in result.
        """
        if not self._connected:
            await self._connect()

        # Check for pending-but-no-approved condition
        approved = self._security.get_approved_tools()
        pending = self._security.get_pending_tools()
        if not approved and pending:
            raise NoApprovedToolsError(
                f"No approved tools available. {len(pending)} tool(s) are pending "
                "first-run approval. Set an approval callback via "
                "bridge.set_approval_callback() before connecting, or "
                "configure auto_approve_first_seen=true in [security].",
                pending_count=len(pending),
            )

        # Override max_turns if specified
        if max_turns is not None:
            loop = AgentLoop(
                ollama=self._ollama,
                security=self._security,
                translator=self._translator,
                max_turns=max_turns,
                max_turns_hard_cap=self._config.security.max_turns_hard_cap,
            )
        else:
            loop = self._loop

        # Generate a per-invocation trace_id (uuid4 — random, no internal state leak).
        # This correlates all ToolCallRecords produced during this Bridge.run() call.
        trace_id = str(uuid.uuid4())

        result = await loop.execute(
            prompt=prompt,
            model=model,
            system_prompt=system_prompt,
            on_event=on_event,
            trace_id=trace_id,
        )

        # Attach audit log and provenance metadata to result
        result.audit_log = self._audit.get_session_entries()
        result.bridge_version = _BRIDGE_VERSION

        return result

    def run_sync(
        self,
        prompt: str,
        model: str,
        **kwargs: Any,
    ) -> BridgeResult:
        """Synchronous wrapper for run().

        Uses asyncio.run() internally — creates a new event loop including
        full connect/disconnect lifecycle. For multiple queries, use the
        async API (bridge.run()) to avoid reconnecting to MCP servers
        on every call.

        Raises RuntimeError if called from an existing event loop
        (Jupyter, async frameworks).
        """
        async def _run() -> BridgeResult:
            async with self:
                return await self.run(prompt, model, **kwargs)

        return asyncio.run(_run())

    async def run_stream(
        self,
        prompt: str,
        model: str,
        system_prompt: str | None = None,
    ) -> AsyncIterator[StreamEvent]:
        """Stream events from a prompt execution.

        Yields StreamEvent objects for text, tool calls, results, etc.
        """
        if not self._connected:
            await self._connect()

        async for event in self._loop.execute_stream(
            prompt=prompt,
            model=model,
            system_prompt=system_prompt,
        ):
            yield event

    async def list_available_tools(self) -> dict[str, list[str]]:
        """List approved tools by server."""
        if not self._connected:
            await self._connect()

        result: dict[str, list[str]] = {}
        for server, tools in self._security.get_approved_tools_by_server().items():
            result[server] = [t.name for t in tools]
        return result

    async def list_discovered_tools(self) -> dict[str, list[ToolSchema]]:
        """List all discovered tools by server, including unapproved.

        Returns the raw ToolSchema for each tool discovered during connect_and_scan(),
        regardless of approval state. Useful for displaying what's available and
        letting the user decide what to approve.
        """
        if not self._connected:
            await self._connect()
        return self._security.get_discovered_tools_by_server()

    async def list_pending_tool_approvals(self) -> list[PendingToolApproval]:
        """List tools awaiting first-run approval.

        Returns PendingToolApproval objects with enough context (description,
        input_schema, sanitization results) for a human to make an informed decision.
        """
        if not self._connected:
            await self._connect()
        return self._security.get_pending_tools()

    async def approve_tool(self, server: str, tool_name: str) -> None:
        """Approve a pending or integrity-blocked tool, making it callable.

        Use after connect_and_scan() to resolve individual tools without a
        batch callback. Also handles re-approval after rug-pull detection.

        Raises ToolBlockedError if the tool is not in an approvable state.
        """
        if not self._connected:
            await self._connect()
        self._security.approve_tool(server, tool_name)

    async def deny_tool(self, server: str, tool_name: str) -> None:
        """Deny a pending tool, preventing it from being used this session.

        Records the denied hash in the registry so the system remembers
        this definition was rejected.

        Raises ToolBlockedError if the tool is not in a deniable state.
        """
        if not self._connected:
            await self._connect()
        self._security.deny_tool(server, tool_name)

    async def get_server_health(self) -> dict[str, ServerHealth]:
        """Check health of all configured servers."""
        health: dict[str, ServerHealth] = {}
        for name in self._config.servers:
            health[name] = await self._mcp.health_check(name)
        return health
