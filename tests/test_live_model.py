"""Live model tests — real Ollama + real MCP, single tool calls.

Tests that a real model can discover, call, and process results from
real tools through the full security pipeline. Each test makes one
tool call (~20-30s inference per test).

Run: pytest tests/test_live_model.py
"""

from __future__ import annotations

import pytest

from tests.helpers import requires_ollama


@requires_ollama
class TestModelToolExecution:
    """Verify real model calls real tools through the security pipeline."""

    @pytest.mark.asyncio
    async def test_model_calls_echo_tool(self, live_bridge, ollama_model):
        """Real model discovers echo tool and calls it."""
        bridge = await live_bridge()

        result = await bridge.run(
            'Use the echo tool to echo back exactly "hello from e2e test".',
            model=ollama_model,
            system_prompt=(
                "You have access to tools. When asked to echo something, "
                "use the echo tool. Respond only with the tool result."
            ),
        )

        assert not result.truncated, "Conversation truncated — result is incomplete"
        echo_calls = [tc for tc in result.tool_calls if tc.tool_name == "echo"]
        assert len(echo_calls) >= 1
        assert not echo_calls[0].blocked

    @pytest.mark.asyncio
    async def test_model_calls_add_tool(self, live_bridge, ollama_model):
        """Real model uses add tool for arithmetic."""
        bridge = await live_bridge()

        result = await bridge.run(
            "What is 17 + 25? Use the add tool to compute this.",
            model=ollama_model,
            system_prompt=(
                "You have access to an add tool. Use it to compute sums. "
                "Report the result from the tool."
            ),
        )

        assert not result.truncated, "Conversation truncated — result is incomplete"
        add_calls = [tc for tc in result.tool_calls if tc.tool_name == "add"]
        assert len(add_calls) >= 1
        assert not add_calls[0].blocked
        assert "42" in result.content

    @pytest.mark.asyncio
    async def test_echo_result_relayed_to_user(self, live_bridge, ollama_model):
        """Model relays the exact tool result back to the user."""
        bridge = await live_bridge()

        result = await bridge.run(
            'Use the echo tool to echo "BRIDGE_MARKER_7x9q". '
            "Then tell me exactly what the tool returned.",
            model=ollama_model,
            system_prompt=(
                "Use the echo tool when asked. Report the exact text "
                "the tool returned, without modification."
            ),
        )

        assert not result.truncated, "Conversation truncated — result is incomplete"
        echo_calls = [tc for tc in result.tool_calls if tc.tool_name == "echo" and not tc.blocked]
        assert len(echo_calls) >= 1
        assert "BRIDGE_MARKER_7x9q" in echo_calls[0].result_summary
        assert "BRIDGE_MARKER_7x9q" in result.content

    @pytest.mark.asyncio
    async def test_execution_has_provenance(self, live_bridge, ollama_model):
        """Tool execution through real pipeline produces audit entries."""
        bridge = await live_bridge()

        result = await bridge.run(
            'Echo "provenance test" using the echo tool.',
            model=ollama_model,
            system_prompt="Use the echo tool when asked to echo something.",
        )

        assert not result.truncated, "Conversation truncated — result is incomplete"
        assert len(result.tool_calls) >= 1
        tool_call_entries = [
            e for e in result.audit_log if e.event_type.value == "tool_call"
        ]
        assert len(tool_call_entries) >= 1


@requires_ollama
class TestModelSecurityPipeline:
    """Verify security enforcement with real model."""

    @pytest.mark.asyncio
    async def test_unapproved_tool_not_callable(self, live_bridge, ollama_model):
        """Model cannot call tools that are pending approval."""
        async def approve_echo_only(pending):
            return {p.key: (p.name == "echo") for p in pending}

        bridge = await live_bridge(
            auto_approve=False, require_approval=True,
            approval_callback=approve_echo_only,
        )

        result = await bridge.run(
            "Use the add tool to compute 1 + 1.",
            model=ollama_model,
            system_prompt="Use the add tool for arithmetic.",
        )

        assert not result.truncated, "Conversation truncated — result is incomplete"
        add_calls = [tc for tc in result.tool_calls if tc.tool_name == "add"]
        if add_calls:
            assert add_calls[0].blocked

    @pytest.mark.asyncio
    async def test_result_sanitization_on_real_output(self, live_bridge, ollama_model):
        """Real tool results pass through the sanitization pipeline."""
        bridge = await live_bridge()

        result = await bridge.run(
            'Use the echo tool to echo "safe content here".',
            model=ollama_model,
            system_prompt="Use the echo tool. Just report what it returns.",
        )

        assert not result.truncated, "Conversation truncated — result is incomplete"
        for tc in result.tool_calls:
            if tc.tool_name == "echo" and not tc.blocked:
                assert "QUARANTINED" not in tc.result_summary

    @pytest.mark.asyncio
    async def test_conversation_turns_tracked(self, live_bridge, ollama_model):
        """BridgeResult.turns reflects actual conversation turns."""
        bridge = await live_bridge()

        result = await bridge.run(
            "Use the add tool to compute 1 + 1.",
            model=ollama_model,
            system_prompt="Use the add tool for math.",
        )

        assert not result.truncated, "Conversation truncated — result is incomplete"
        if result.tool_calls:
            assert result.turns >= 2, (
                f"With tool calls, expected >= 2 turns, got {result.turns}"
            )
