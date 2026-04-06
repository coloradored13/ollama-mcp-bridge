"""Tests for types.py — shared data types."""

from ollama_mcp_bridge.types import (
    ActionClass,
    ApprovedTool,
    OllamaToolCall,
    ToolSchema,
)


class TestToolSchema:
    def test_definition_hash_deterministic(self, sample_tool_schema: ToolSchema):
        """Same schema should always produce same hash."""
        h1 = sample_tool_schema.definition_hash
        h2 = sample_tool_schema.definition_hash
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_definition_hash_changes_on_modification(self):
        """Different schemas should produce different hashes."""
        t1 = ToolSchema(
            server="s", name="t", description="original",
            input_schema={"type": "object"},
        )
        t2 = ToolSchema(
            server="s", name="t", description="modified",
            input_schema={"type": "object"},
        )
        assert t1.definition_hash != t2.definition_hash

    def test_raw_definition_canonical(self):
        """Raw definition should be JSON with sorted keys, no spaces."""
        t = ToolSchema(
            server="s", name="tool", description="desc",
            input_schema={"type": "object", "properties": {}},
        )
        raw = t.raw_definition
        assert '"description":"desc"' in raw
        assert '"name":"tool"' in raw

    def test_frozen_model(self, sample_tool_schema: ToolSchema):
        """ToolSchema should be immutable."""
        import pytest
        with pytest.raises(Exception):
            sample_tool_schema.name = "changed"


class TestApprovedTool:
    def test_namespaced_name(self, sample_approved_tool: ApprovedTool):
        assert sample_approved_tool.namespaced_name == "sigma-mem__store_memory"

    def test_namespaced_name_format(self):
        tool = ApprovedTool(
            server="files", name="read_file", description="Read",
            input_schema={}, classification=ActionClass.READ,
            definition_hash="x",
        )
        assert tool.namespaced_name == "files__read_file"


class TestOllamaToolCall:
    def test_server_extraction(self):
        tc = OllamaToolCall(
            function_name="sigma-mem__store_memory",
            arguments={},
        )
        assert tc.server == "sigma-mem"
        assert tc.tool_name == "store_memory"

    def test_bare_name_no_server(self):
        tc = OllamaToolCall(
            function_name="store_memory",
            arguments={},
        )
        assert tc.server is None
        assert tc.tool_name == "store_memory"

    def test_arguments_preserved(self):
        args = {"key": "test", "value": "hello"}
        tc = OllamaToolCall(function_name="t", arguments=args)
        assert tc.arguments == args
