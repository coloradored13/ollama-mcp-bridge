"""MCP client manager — multi-server stdio connection lifecycle.

Wraps the official MCP Python SDK (modelcontextprotocol/python-sdk) to manage
concurrent connections to multiple MCP servers over stdio transport.

TRANSPORT CHOICE: stdio (not SSE or HTTP) because:
    - All connections are local (same machine) — no network needed.
    - stdio has zero network attack surface (unlike SSE/HTTP).
    - SSE is deprecated by the MCP project.
    - Each MCP server runs as a subprocess communicating via stdin/stdout pipes.

LIFECYCLE COMPLEXITY: The MCP SDK's stdio_client() is an async context manager
that spawns an anyio task group internally. This means each connection must live
inside its own async context — you can't just store a ClientSession in a dict.

Solution: Each ServerConnection owns its own AsyncExitStack. Connecting means
entering the stdio_client context via stack.enter_async_context(), which keeps
the context alive until disconnect. Disconnecting closes that server's stack,
which triggers the MCP shutdown sequence (close stdin -> SIGTERM -> SIGKILL).
This design was validated against jonigl/ollama-mcp-bridge which uses the same
pattern successfully.

SECURITY BOUNDARY: MCPClientManager is a pure transport layer. It has NO
security logic — no allowlists, no validation, no sanitization. Security is
enforced by SecurityGateway, which owns MCPClientManager and is its sole caller.
AgentLoop never touches this class directly.
"""

from __future__ import annotations

import asyncio
import logging
from contextlib import AsyncExitStack
from dataclasses import dataclass, field
from typing import Any

from mcp.client.session import ClientSession
from mcp.client.stdio import StdioServerParameters, stdio_client

from .config import ServerConfig
from .errors import MCPConnectionError, MCPTimeoutError, MCPToolError
from .types import ServerHealth, ToolSchema

logger = logging.getLogger(__name__)

# Default timeout for tool calls (seconds)
DEFAULT_TOOL_TIMEOUT = 30.0
# Maximum reconnect attempts
MAX_RECONNECT_ATTEMPTS = 3
# Connection timeout
CONNECTION_TIMEOUT = 15.0


@dataclass
class ServerConnection:
    """Represents a live connection to a single MCP server.

    Each connection maintains its own independent lifecycle:
    - stack: AsyncExitStack that owns the stdio_client context and ClientSession.
        Closing the stack triggers the full MCP shutdown sequence for this server
        without affecting other server connections.
    - session: The MCP ClientSession used for list_tools() and call_tool().
    - tools: Cached tool schemas fetched at connect time. These are the RAW
        schemas from the server — they haven't been security-scanned yet.
        SecurityGateway scans them during connect_and_scan().
    """

    name: str
    config: ServerConfig
    stack: AsyncExitStack = field(default_factory=AsyncExitStack)
    session: ClientSession | None = None
    connected: bool = False
    tools: list[ToolSchema] = field(default_factory=list)


class MCPClientManager:
    """Manages concurrent connections to multiple MCP servers.

    Lifecycle:
    - Servers configured at init (from BridgeConfig.servers)
    - Connections established lazily (first tool call or explicit connect)
    - Per-server AsyncExitStack for independent cleanup
    - disconnect_all() on Bridge exit
    """

    def __init__(self, server_configs: dict[str, ServerConfig]):
        self._configs = server_configs
        self._connections: dict[str, ServerConnection] = {}
        self._max_servers = 10

    async def connect(self, name: str) -> None:
        """Connect to a named MCP server.

        Establishes stdio transport, creates ClientSession, initializes protocol.

        Raises:
            MCPConnectionError: If server cannot be started or initialized.
        """
        if name in self._connections and self._connections[name].connected:
            return

        config = self._configs.get(name)
        if not config:
            raise MCPConnectionError(f"No configuration for server '{name}'")

        if len(self._connections) >= self._max_servers:
            raise MCPConnectionError(f"Maximum concurrent servers ({self._max_servers}) reached")

        conn = ServerConnection(name=name, config=config)

        try:
            params = StdioServerParameters(
                command=config.command,
                args=config.args,
                env=config.env,
            )

            # Step 1: Enter stdio_client context.
            # This spawns the MCP server as a subprocess and creates stdin/stdout
            # pipes for JSON-RPC communication. The context manager keeps the
            # subprocess alive — exiting it kills the server.
            # We use enter_async_context() to transfer ownership to the per-server
            # AsyncExitStack, so the connection lives beyond this method.
            read_stream, write_stream = await asyncio.wait_for(
                conn.stack.enter_async_context(stdio_client(params)),
                timeout=CONNECTION_TIMEOUT,
            )

            # Step 2: Create ClientSession over the stdio streams.
            # ClientSession handles the JSON-RPC protocol: framing, request/response
            # matching, capability negotiation. initialize() performs the MCP handshake
            # (protocol version, server capabilities, etc.).
            session = await conn.stack.enter_async_context(ClientSession(read_stream, write_stream))
            await asyncio.wait_for(session.initialize(), timeout=CONNECTION_TIMEOUT)

            conn.session = session
            conn.connected = True

            # Fetch available tools
            tools_result = await session.list_tools()
            conn.tools = [
                ToolSchema(
                    server=name,
                    name=t.name,
                    description=t.description or "",
                    input_schema=t.inputSchema if t.inputSchema else {},
                )
                for t in tools_result.tools
            ]

            self._connections[name] = conn
            logger.info("Connected to MCP server '%s' (%d tools)", name, len(conn.tools))

        except asyncio.TimeoutError:
            await conn.stack.aclose()
            raise MCPConnectionError(
                f"Timeout connecting to server '{name}' "
                f"(command: {config.command} {' '.join(config.args)})"
            )
        except Exception as e:
            await conn.stack.aclose()
            raise MCPConnectionError(f"Failed to connect to server '{name}': {e}") from e

    async def disconnect(self, name: str) -> None:
        """Disconnect from a named MCP server."""
        conn = self._connections.pop(name, None)
        if conn:
            try:
                await conn.stack.aclose()
            except Exception as e:
                logger.warning("Error disconnecting from '%s': %s", name, e)
            conn.connected = False
            logger.info("Disconnected from MCP server '%s'", name)

    async def disconnect_all(self) -> None:
        """Disconnect from all MCP servers."""
        names = list(self._connections.keys())
        for name in names:
            await self.disconnect(name)

    async def _ensure_connected(self, name: str) -> ServerConnection:
        """Ensure server is connected, connecting lazily if needed."""
        if name not in self._connections or not self._connections[name].connected:
            await self.connect(name)
        conn = self._connections.get(name)
        if not conn or not conn.connected or not conn.session:
            raise MCPConnectionError(f"Server '{name}' is not connected")
        return conn

    async def list_tools(self, name: str) -> list[ToolSchema]:
        """List tools available from a specific server."""
        conn = await self._ensure_connected(name)
        return conn.tools

    async def list_all_tools(self) -> dict[str, list[ToolSchema]]:
        """List tools from all configured servers (connecting as needed)."""
        result: dict[str, list[ToolSchema]] = {}
        for name in self._configs:
            try:
                conn = await self._ensure_connected(name)
                result[name] = conn.tools
            except MCPConnectionError as e:
                logger.error("Cannot connect to '%s': %s", name, e)
                result[name] = []
        return result

    async def call_tool(
        self,
        server: str,
        tool: str,
        args: dict[str, Any],
        timeout: float = DEFAULT_TOOL_TIMEOUT,
    ) -> str:
        """Call a tool on an MCP server.

        Args:
            server: Server name.
            tool: Tool name (bare, not namespaced).
            args: Tool arguments (already validated by SecurityGateway).
            timeout: Maximum seconds to wait for tool result.

        Returns:
            Tool result as a string.

        Raises:
            MCPConnectionError: Server not connected.
            MCPToolError: Tool execution failed.
            MCPTimeoutError: Tool call timed out.
        """
        conn = await self._ensure_connected(server)

        try:
            result = await asyncio.wait_for(
                conn.session.call_tool(tool, args),
                timeout=timeout,
            )

            # Extract text content from result
            if result.isError:
                error_text = ""
                for content in result.content:
                    if hasattr(content, "text"):
                        error_text += content.text
                raise MCPToolError(
                    f"Tool '{tool}' on '{server}' returned error: {error_text}",
                    safe_message=error_text[:200],
                )

            # Concatenate text content
            text_parts = []
            for content in result.content:
                if hasattr(content, "text"):
                    text_parts.append(content.text)
            return "\n".join(text_parts) if text_parts else ""

        except asyncio.TimeoutError:
            raise MCPTimeoutError(f"Tool '{tool}' on '{server}' timed out after {timeout}s")
        except (MCPToolError, MCPTimeoutError):
            raise
        except Exception as e:
            raise MCPToolError(
                f"Error calling '{tool}' on '{server}': {e}",
                safe_message=str(e)[:200],
            ) from e

    async def health_check(self, name: str) -> ServerHealth:
        """Check health of a specific server connection."""
        conn = self._connections.get(name)
        if not conn or not conn.connected:
            return ServerHealth(name=name, connected=False)

        try:
            # Ping by listing tools (lightweight operation)
            await conn.session.list_tools()
            return ServerHealth(
                name=name,
                connected=True,
                tools_count=len(conn.tools),
            )
        except Exception as e:
            return ServerHealth(name=name, connected=False, error=str(e))
