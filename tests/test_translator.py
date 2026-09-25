"""Tests for translator.py — schema translation."""

from ollama_mcp_bridge.translator import ToolTranslator, _flatten_union_types, _resolve_refs
from ollama_mcp_bridge.types import ApprovedTool, OllamaToolCall


class TestResolveRefs:
    def test_no_refs_passthrough(self):
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        assert _resolve_refs(schema) == schema

    def test_simple_ref_resolution(self):
        schema = {
            "type": "object",
            "properties": {
                "item": {"$ref": "#/$defs/Item"},
            },
            "$defs": {
                "Item": {"type": "object", "properties": {"name": {"type": "string"}}},
            },
        }
        resolved = _resolve_refs(schema)
        assert "$ref" not in str(resolved)
        assert resolved["properties"]["item"]["type"] == "object"

    def test_definitions_key(self):
        """Also handle 'definitions' (older JSON Schema)."""
        schema = {
            "type": "object",
            "properties": {
                "item": {"$ref": "#/definitions/Item"},
            },
            "definitions": {
                "Item": {"type": "string"},
            },
        }
        resolved = _resolve_refs(schema)
        assert resolved["properties"]["item"]["type"] == "string"

    def test_removes_defs_after_resolution(self):
        schema = {
            "type": "object",
            "$defs": {"X": {"type": "string"}},
            "properties": {"x": {"$ref": "#/$defs/X"}},
        }
        resolved = _resolve_refs(schema)
        assert "$defs" not in resolved


class TestFlattenUnionTypes:
    def test_anyof_flattened(self):
        schema = {
            "type": "object",
            "properties": {
                "value": {
                    "anyOf": [
                        {"type": "string"},
                        {"type": "integer"},
                    ]
                }
            },
        }
        result = _flatten_union_types(schema)
        assert result["properties"]["value"]["type"] == "string"
        assert "Accepts:" in result["properties"]["value"]["description"]

    def test_no_union_passthrough(self):
        schema = {"type": "object", "properties": {"x": {"type": "string"}}}
        assert _flatten_union_types(schema) == schema


class TestToolTranslator:
    def test_to_ollama_tools(self, sample_approved_tool: ApprovedTool):
        translator = ToolTranslator()
        tools = translator.to_ollama_tools([sample_approved_tool])

        assert len(tools) == 1
        tool = tools[0]
        assert tool["type"] == "function"
        assert tool["function"]["name"] == "sigma-mem__store_memory"
        assert "properties" in tool["function"]["parameters"]

    def test_namespace_format(self, multiple_tools: list[ApprovedTool]):
        translator = ToolTranslator()
        tools = translator.to_ollama_tools(multiple_tools)

        names = [t["function"]["name"] for t in tools]
        assert "sigma-mem__recall" in names
        assert "sigma-mem__store_memory" in names
        assert "files__write_file" in names

    def test_parse_tool_call_namespaced(self, multiple_tools: list[ApprovedTool]):
        translator = ToolTranslator()
        tc = OllamaToolCall(
            function_name="sigma-mem__recall",
            arguments={"query": "test"},
        )
        result = translator.parse_tool_call(tc, multiple_tools)
        assert result is not None
        server, name, args = result
        assert server == "sigma-mem"
        assert name == "recall"
        assert args == {"query": "test"}

    def test_parse_tool_call_bare_name(self, multiple_tools: list[ApprovedTool]):
        """Model drops namespace — still matches."""
        translator = ToolTranslator()
        tc = OllamaToolCall(function_name="recall", arguments={})
        result = translator.parse_tool_call(tc, multiple_tools)
        assert result is not None
        assert result[0] == "sigma-mem"

    def test_parse_unknown_tool(self, multiple_tools: list[ApprovedTool]):
        translator = ToolTranslator()
        tc = OllamaToolCall(function_name="nonexistent_tool", arguments={})
        assert translator.parse_tool_call(tc, multiple_tools) is None

    def test_to_tool_result_message(self):
        msg = ToolTranslator.to_tool_result_message("sigma-mem__recall", "result text")
        assert msg["role"] == "tool"
        assert msg["name"] == "sigma-mem__recall"
        assert msg["content"] == "result text"

    def test_to_ollama_messages_with_system(self):
        msgs = ToolTranslator.to_ollama_messages("You are helpful.", "Hello")
        assert len(msgs) == 2
        assert msgs[0]["role"] == "system"
        assert msgs[1]["role"] == "user"

    def test_to_ollama_messages_no_system(self):
        msgs = ToolTranslator.to_ollama_messages(None, "Hello")
        assert len(msgs) == 1
        assert msgs[0]["role"] == "user"
