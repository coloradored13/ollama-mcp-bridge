"""Minimal MCP test server for E2E testing.

Exposes tools over stdio transport:
  - echo: Returns input text (READ, safe)
  - add: Adds two numbers (READ, safe)
  - get_secret: Returns a fake secret (READ, for testing result handling)
  - flaky_tool: Fails on certain inputs (for fault injection testing)
  - slow_tool: Sleeps before responding (for timeout testing)

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
        Tool(
            name="flaky_tool",
            description="A tool that fails when input contains 'fail'. Use for testing error handling.",
            inputSchema={
                "type": "object",
                "properties": {
                    "input": {
                        "type": "string",
                        "description": "Input text. If it contains 'fail', the tool will error.",
                    },
                },
                "required": ["input"],
            },
        ),
        Tool(
            name="slow_tool",
            description="A tool that takes a while to respond. Use for testing patience.",
            inputSchema={
                "type": "object",
                "properties": {
                    "seconds": {
                        "type": "number",
                        "description": "How many seconds to wait before responding.",
                    },
                },
                "required": ["seconds"],
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

    elif name == "flaky_tool":
        input_text = arguments.get("input", "")
        if "fail" in input_text.lower():
            raise Exception(f"Tool failed on input: {input_text}")
        return [TextContent(type="text", text=f"flaky_tool succeeded: {input_text}")]

    elif name == "slow_tool":
        seconds = min(arguments.get("seconds", 1), 30)  # cap at 30s
        await asyncio.sleep(seconds)
        return [TextContent(type="text", text=f"slow_tool completed after {seconds}s")]

    else:
        return [TextContent(type="text", text=f"Unknown tool: {name}")]


async def main():
    async with stdio_server() as (read, write):
        await server.run(read, write, server.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
