"""Bidirectional schema translation between MCP and Ollama tool formats.

MCP tools use JSON Schema for their input definitions. Ollama's /api/chat
endpoint expects tools in OpenAI-compatible format (type: "function", nested
under "function" key). The translation is near-identity — the main structural
difference is the wrapper format.

Additional transformations:
- NAMESPACE PREFIXING: Multiple MCP servers can expose tools with the same name.
  We prefix with "server__tool" (double underscore) when presenting to Ollama.
  The prefix is stripped before calling MCP (which uses bare tool names).
- $ref RESOLUTION: Some MCP tools use JSON Schema $ref for type reuse.
  Small models (4B-8B) don't handle $ref well, so we inline the referenced
  types before presenting to the model.
- anyOf/oneOf FLATTENING: Small models also struggle with union types.
  We flatten these to the most permissive type (string) with a description
  noting what types are actually accepted.

This module has NO security logic. It receives ApprovedTool objects (already
security-scanned) and converts their format. Security decisions happen in
SecurityGateway before tools reach this layer.
"""

from __future__ import annotations

import copy
import logging
from typing import Any

from .types import ApprovedTool, OllamaToolCall

logger = logging.getLogger(__name__)

NAMESPACE_SEP = "__"


def _resolve_refs(schema: dict[str, Any]) -> dict[str, Any]:
    """Resolve simple $ref references within a JSON Schema.

    Handles the common case of $ref pointing to #/$defs/TypeName.
    Does NOT handle remote $ref or deeply nested circular refs.
    """
    if "$defs" not in schema and "definitions" not in schema:
        return schema

    defs = schema.get("$defs", schema.get("definitions", {}))
    resolved = copy.deepcopy(schema)

    def _walk(node: Any) -> Any:
        if isinstance(node, dict):
            if "$ref" in node:
                ref_path = node["$ref"]
                if ref_path.startswith("#/$defs/") or ref_path.startswith("#/definitions/"):
                    ref_name = ref_path.split("/")[-1]
                    if ref_name in defs:
                        return _walk(copy.deepcopy(defs[ref_name]))
                logger.warning("Cannot resolve $ref: %s — passing through", ref_path)
                return node
            return {k: _walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_walk(item) for item in node]
        return node

    result = _walk(resolved)
    # Remove $defs from top level after resolution
    result.pop("$defs", None)
    result.pop("definitions", None)
    return result


def _flatten_union_types(schema: dict[str, Any]) -> dict[str, Any]:
    """Flatten anyOf/oneOf to most permissive type with description.

    Small models (4B-8B) may not handle anyOf/oneOf well.
    """
    result = copy.deepcopy(schema)

    def _walk(node: Any) -> Any:
        if isinstance(node, dict):
            for union_key in ("anyOf", "oneOf"):
                if union_key in node:
                    variants = node[union_key]
                    types = []
                    for v in variants:
                        if isinstance(v, dict) and "type" in v:
                            types.append(v["type"])
                    if types:
                        # Use string as most permissive
                        flattened = {"type": "string"}
                        desc_parts = []
                        if "description" in node:
                            desc_parts.append(node["description"])
                        desc_parts.append(f"Accepts: {', '.join(types)}")
                        flattened["description"] = ". ".join(desc_parts)
                        return flattened
            return {k: _walk(v) for k, v in node.items()}
        if isinstance(node, list):
            return [_walk(item) for item in node]
        return node

    return _walk(result)


class ToolTranslator:
    """Bidirectional translator between MCP and Ollama tool formats."""

    def to_ollama_tools(self, approved_tools: list[ApprovedTool]) -> list[dict[str, Any]]:
        """Convert approved MCP tools to Ollama tools format.

        Returns list of dicts suitable for the /api/chat tools parameter.
        """
        ollama_tools = []
        for tool in approved_tools:
            schema = _resolve_refs(tool.input_schema)
            schema = _flatten_union_types(schema)

            ollama_tool = {
                "type": "function",
                "function": {
                    "name": tool.namespaced_name,
                    "description": tool.description,
                    "parameters": schema,
                },
            }
            ollama_tools.append(ollama_tool)
        return ollama_tools

    def parse_tool_call(
        self, tool_call: OllamaToolCall, approved_tools: list[ApprovedTool]
    ) -> tuple[str, str, dict[str, Any]] | None:
        """Parse an Ollama tool call into (server, tool_name, args).

        Returns None if the tool call doesn't match any approved tool.
        """
        # Try namespaced match first
        for tool in approved_tools:
            if tool.namespaced_name == tool_call.function_name:
                return tool.server, tool.name, tool_call.arguments

        # Try bare name match (model may drop namespace)
        for tool in approved_tools:
            if tool.name == tool_call.function_name:
                return tool.server, tool.name, tool_call.arguments

        return None

    @staticmethod
    def to_tool_result_message(tool_name: str, content: str) -> dict[str, Any]:
        """Format a tool result as an Ollama message for the conversation.

        Args:
            tool_name: The namespaced tool name (e.g., "server__tool") matching
                what was sent to Ollama in the assistant message's tool_calls.
            content: The tool execution result or error message.
        """
        return {
            "role": "tool",
            "name": tool_name,
            "content": content,
        }

    @staticmethod
    def to_ollama_messages(
        system_prompt: str | None,
        user_prompt: str,
    ) -> list[dict[str, Any]]:
        """Build initial message list for Ollama chat."""
        messages: list[dict[str, Any]] = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": user_prompt})
        return messages
