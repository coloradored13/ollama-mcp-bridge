"""Minimal MCP test server for E2E testing.

Exposes three tools over stdio transport:
  - echo: Returns input text (READ, safe)
  - add: Adds two numbers (READ, safe)
  - get_secret: Returns a fake secret (READ, for testing result handling)

Run directly: python tests/fixtures/test_mcp_server.py
The bridge launches this as a subprocess via stdio transport.
"""

import asyncio
import json

from mcp.server import Server
from mcp.server.stdio import stdio_server
from mcp.types import TextContent, Tool


server = Server("test-tools")


@server.list_tools()
async def list_tools() -> list[Tool]:
    return [
        Tool(
            name="echo",
            description="Echo back the input text.",
            inputSchema={
                "type": "object",
                "properties": {
                    "text": {
                        "type": "string",
                        "description": "Text to echo back",
                    },
                },
                "required": ["text"],
            },
        ),
        Tool(
            name="add",
            description="Add two numbers and return the sum.",
            inputSchema={
                "type": "object",
                "properties": {
                    "a": {"type": "number", "description": "First number"},
                    "b": {"type": "number", "description": "Second number"},
                },
                "required": ["a", "b"],
            },
        ),
        Tool(
            name="get_secret",
            description="Retrieve a stored secret value by name.",
            inputSchema={
                "type": "object",
                "properties": {
                    "name": {
                        "type": "string",
                        "description": "Name of the secret to retrieve",
                    },
                },
                "required": ["name"],
            },
        ),
    ]


@server.call_tool()
async def call_tool(name: str, arguments: dict) -> list[TextContent]:
    if name == "echo":
        text = arguments.get("text", "")
        return [TextContent(type="text", text=text)]

    elif name == "add":
        a = arguments.get("a", 0)
        b = arguments.get("b", 0)
        result = a + b
        return [TextContent(type="text", text=json.dumps({"sum": result}))]

    elif name == "get_secret":
        secret_name = arguments.get("name", "")
        # Returns a fake secret for testing result handling
        secrets = {
            "api_key": "sk-test-1234567890abcdef",
            "password": "hunter2",
            "token": "ghp_fake_token_for_testing",
        }
        value = secrets.get(secret_name, f"unknown secret: {secret_name}")
        return [TextContent(type="text", text=value)]

    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
