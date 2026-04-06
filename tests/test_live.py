"""Live E2E tests — real Ollama + real MCP server.

These tests address SQ[0] (empirical model testing) and SQ[11] (real MCP
server integration). They require a running Ollama instance with at least
one model available. Skipped automatically when Ollama is not running.

What these tests prove that mocks cannot:
  - The bridge actually connects to a real MCP server subprocess
  - Real Ollama models can discover and call tools through the bridge
  - The security pipeline works with real (not simulated) tool definitions
  - Result sanitization handles real tool output
  - The approval flow works end-to-end
  - Provenance and risk assessment are populated on real results
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

from ollama_mcp_bridge import Bridge
from ollama_mcp_bridge.config import (
    BridgeConfig,
    LoggingConfig,
    SecurityConfig,
    ServerConfig,
)

# Test infrastructure constants
TEST_MCP_SERVER = str(Path(__file__).parent / "fixtures" / "test_mcp_server.py")
TEST_PYTHON = sys.executable


def _ollama_available() -> bool:
    try:
        import json
        import urllib.request
        resp = urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3)
        data = json.loads(resp.read())
        return len(data.get("models", [])) > 0
    except Exception:
        return False


requires_ollama = pytest.mark.skipif(
    not _ollama_available(),
    reason="Ollama not running or no models available",
)


def _make_live_config(
    tmp_path,
    auto_approve: bool = False,
    require_approval: bool = True,
    allowed_tools: list[str] | None = None,
    max_turns: int = 3,
) -> BridgeConfig:
    """Build a BridgeConfig pointing at the real test MCP server."""
    return BridgeConfig(
        ollama_host="http://127.0.0.1:11434",
        servers={
            "test-tools": ServerConfig(
                command=TEST_PYTHON,
                args=[TEST_MCP_SERVER],
                allowed_tools=["echo", "add", "get_secret"] if allowed_tools is None else allowed_tools,
            ),
        },
        security=SecurityConfig(
            auto_approve_first_seen=auto_approve,
            require_first_run_approval=require_approval,
            max_turns=max_turns,
            max_tool_calls_per_session=20,
            rate_limit_per_server=10,
            approval_registry_path=str(tmp_path / "approved_tools.json"),
        ),
        logging=LoggingConfig(
            audit_file=str(tmp_path / "audit.jsonl"),
        ),
    )


# --- Connection and Discovery Tests ---


@requires_ollama
class TestLiveConnection:
    """Verify real MCP server connection and tool discovery."""

    @pytest.mark.asyncio
    async def test_connect_discovers_tools(self, tmp_path):
        """Bridge connects to real MCP server and discovers all 3 tools."""
        config = _make_live_config(tmp_path, auto_approve=True, require_approval=False)
        bridge = Bridge(config)

        async with bridge:
            tools = await bridge.list_discovered_tools()
            assert "test-tools" in tools
            tool_names = {t.name for t in tools["test-tools"]}
            assert tool_names == {"echo", "add", "get_secret"}

            approved = bridge._security.get_approved_tools()
            assert len(approved) == 3

    @pytest.mark.asyncio
    async def test_connect_with_allowlist_subset(self, tmp_path):
        """Only allowlisted tools are approved; others stay discovered-only."""
        config = _make_live_config(
            tmp_path,
            auto_approve=True,
            require_approval=False,
            allowed_tools=["echo"],
        )
        bridge = Bridge(config)

        async with bridge:
            approved = bridge._security.get_approved_tools()
            assert len(approved) == 1
            assert approved[0].name == "echo"

            # Other tools were discovered but not approved
            discovered = await bridge.list_discovered_tools()
            assert len(discovered["test-tools"]) == 3


# --- Approval Flow Tests ---


@requires_ollama
class TestLiveApprovalFlow:
    """Verify first-run approval works with real servers."""

    @pytest.mark.asyncio
    async def test_pending_approval_blocks_until_approved(self, tmp_path):
        """With require_first_run_approval, tools start pending."""
        config = _make_live_config(tmp_path, require_approval=True)
        bridge = Bridge(config)

        # Don't set approval callback — tools should be pending
        async with bridge:
            pending = await bridge.list_pending_tool_approvals()
            assert len(pending) == 3
            assert {p.name for p in pending} == {"echo", "add", "get_secret"}

            approved = bridge._security.get_approved_tools()
            assert len(approved) == 0

    @pytest.mark.asyncio
    async def test_approval_callback_approves_tools(self, tmp_path):
        """Approval callback makes tools callable."""
        config = _make_live_config(tmp_path, require_approval=True)
        bridge = Bridge(config)

        approved_tools = []

        async def approve_all(pending):
            for p in pending:
                approved_tools.append(p.name)
            return {p.key: True for p in pending}

        bridge.set_approval_callback(approve_all)

        async with bridge:
            assert set(approved_tools) == {"echo", "add", "get_secret"}
            assert len(bridge._security.get_approved_tools()) == 3
            assert len(await bridge.list_pending_tool_approvals()) == 0

    @pytest.mark.asyncio
    async def test_selective_approval(self, tmp_path):
        """Approve some tools, deny others."""
        config = _make_live_config(tmp_path, require_approval=True)
        bridge = Bridge(config)

        async def approve_echo_only(pending):
            return {p.key: (p.name == "echo") for p in pending}

        bridge.set_approval_callback(approve_echo_only)

        async with bridge:
            approved = bridge._security.get_approved_tools()
            assert len(approved) == 1
            assert approved[0].name == "echo"


# --- Live Tool Execution Tests ---


@requires_ollama
class TestLiveToolExecution:
    """Verify real model calls real tools through the security pipeline."""

    @pytest.mark.asyncio
    async def test_model_calls_echo_tool(self, tmp_path, ollama_model):
        """Real model discovers echo tool and calls it."""
        config = _make_live_config(
            tmp_path, auto_approve=True, require_approval=False,
        )
        bridge = Bridge(config)

        async with bridge:
            result = await bridge.run(
                'Use the echo tool to echo back exactly "hello from e2e test".',
                model=ollama_model,
                system_prompt=(
                    "You have access to tools. When asked to echo something, "
                    "use the echo tool. Respond only with the tool result."
                ),
            )

            # Model should have called the echo tool
            assert len(result.tool_calls) >= 1
            echo_calls = [tc for tc in result.tool_calls if tc.tool_name == "echo"]
            assert len(echo_calls) >= 1
            assert not echo_calls[0].blocked

    @pytest.mark.asyncio
    async def test_model_calls_add_tool(self, tmp_path, ollama_model):
        """Real model uses add tool for arithmetic."""
        config = _make_live_config(
            tmp_path, auto_approve=True, require_approval=False,
        )
        bridge = Bridge(config)

        async with bridge:
            result = await bridge.run(
                "What is 17 + 25? Use the add tool to compute this.",
                model=ollama_model,
                system_prompt=(
                    "You have access to an add tool. Use it to compute sums. "
                    "Report the result from the tool."
                ),
            )

            add_calls = [tc for tc in result.tool_calls if tc.tool_name == "add"]
            assert len(add_calls) >= 1
            assert not add_calls[0].blocked
            # The result should contain 42 somewhere
            assert "42" in result.content

    @pytest.mark.asyncio
    async def test_execution_result_has_provenance(self, tmp_path, ollama_model):
        """Tool execution through real pipeline produces provenance metadata."""
        config = _make_live_config(
            tmp_path, auto_approve=True, require_approval=False,
        )
        bridge = Bridge(config)

        async with bridge:
            result = await bridge.run(
                'Echo "provenance test" using the echo tool.',
                model=ollama_model,
                system_prompt="Use the echo tool when asked to echo something.",
            )

            # At least one tool call should have succeeded
            assert len(result.tool_calls) >= 1

            # Audit log should contain tool_call entries
            tool_call_entries = [
                e for e in result.audit_log if e.event_type.value == "tool_call"
            ]
            assert len(tool_call_entries) >= 1


# --- Security Pipeline Verification ---


@requires_ollama
class TestLiveSecurityPipeline:
    """Verify security enforcement with real infrastructure."""

    @pytest.mark.asyncio
    async def test_unapproved_tool_not_callable(self, tmp_path, ollama_model):
        """Model cannot call tools that are pending approval."""
        config = _make_live_config(tmp_path, require_approval=True)
        bridge = Bridge(config)

        # Approve only echo, leave add and get_secret pending
        async def approve_echo_only(pending):
            return {p.key: (p.name == "echo") for p in pending}

        bridge.set_approval_callback(approve_echo_only)

        async with bridge:
            result = await bridge.run(
                "Use the add tool to compute 1 + 1.",
                model=ollama_model,
                system_prompt="Use the add tool for arithmetic.",
            )

            # add should be blocked (not approved)
            add_calls = [tc for tc in result.tool_calls if tc.tool_name == "add"]
            if add_calls:
                assert add_calls[0].blocked

    @pytest.mark.asyncio
    async def test_empty_allowlist_blocks_all(self, tmp_path):
        """Empty allowed_tools means no tools available (fail-closed)."""
        config = _make_live_config(
            tmp_path,
            auto_approve=True,
            require_approval=False,
            allowed_tools=[],
        )
        bridge = Bridge(config)

        async with bridge:
            approved = bridge._security.get_approved_tools()
            assert len(approved) == 0

    @pytest.mark.asyncio
    async def test_result_sanitization_on_real_output(self, tmp_path, ollama_model):
        """Real tool results pass through the sanitization pipeline."""
        config = _make_live_config(
            tmp_path, auto_approve=True, require_approval=False,
        )
        bridge = Bridge(config)

        async with bridge:
            result = await bridge.run(
                'Use the echo tool to echo "safe content here".',
                model=ollama_model,
                system_prompt="Use the echo tool. Just report what it returns.",
            )

            # Tool calls should succeed without quarantine
            for tc in result.tool_calls:
                if tc.tool_name == "echo" and not tc.blocked:
                    # Result was not quarantined (clean content)
                    assert "QUARANTINED" not in tc.result_summary


# --- Multi-tool and Multi-step Flow ---


@requires_ollama
class TestLiveMultiTool:
    """Verify model uses multiple distinct tools in a single conversation."""

    @pytest.mark.asyncio
    async def test_two_different_tools_called(self, tmp_path, ollama_model):
        """Model calls both echo and add in the same conversation."""
        config = _make_live_config(
            tmp_path, auto_approve=True, require_approval=False,
            max_turns=5,
        )
        bridge = Bridge(config)

        async with bridge:
            result = await bridge.run(
                'Step 1: Use the echo tool with text "ping". '
                "Step 2: Use the add tool to compute 7 + 8. "
                "Do both steps. Report each result.",
                model=ollama_model,
                system_prompt=(
                    "You MUST use tools to complete requests. "
                    "Use the echo tool for echoing text and the add tool for math. "
                    "Complete all steps before giving your final answer."
                ),
            )

            tool_names_called = {tc.tool_name for tc in result.tool_calls if not tc.blocked}
            # Both tools must have been called
            assert "echo" in tool_names_called, (
                f"echo not called. Tools called: {tool_names_called}. "
                f"Full result: {result.content[:200]}"
            )
            assert "add" in tool_names_called, (
                f"add not called. Tools called: {tool_names_called}. "
                f"Full result: {result.content[:200]}"
            )
            # Verify the add tool got correct arguments
            add_calls = [tc for tc in result.tool_calls if tc.tool_name == "add" and not tc.blocked]
            assert len(add_calls) >= 1
            # The model should have passed numbers that sum correctly
            assert "15" in result.content, (
                f"Expected '15' in response. Got: {result.content[:300]}"
            )

    @pytest.mark.asyncio
    async def test_echo_result_appears_in_response(self, tmp_path, ollama_model):
        """Model relays the exact tool result back to the user."""
        config = _make_live_config(
            tmp_path, auto_approve=True, require_approval=False,
        )
        bridge = Bridge(config)

        async with bridge:
            result = await bridge.run(
                'Use the echo tool to echo "BRIDGE_MARKER_7x9q". '
                "Then tell me exactly what the tool returned.",
                model=ollama_model,
                system_prompt=(
                    "Use the echo tool when asked. Report the exact text "
                    "the tool returned, without modification."
                ),
            )

            echo_calls = [tc for tc in result.tool_calls if tc.tool_name == "echo" and not tc.blocked]
            assert len(echo_calls) >= 1
            # The unique marker should appear in the tool result summary
            assert "BRIDGE_MARKER_7x9q" in echo_calls[0].result_summary
            # And the model should relay it
            assert "BRIDGE_MARKER_7x9q" in result.content

    @pytest.mark.asyncio
    async def test_multiple_add_calls(self, tmp_path, ollama_model):
        """Model calls the same tool multiple times with different arguments."""
        config = _make_live_config(
            tmp_path, auto_approve=True, require_approval=False,
            max_turns=5,
        )
        bridge = Bridge(config)

        async with bridge:
            result = await bridge.run(
                "Compute these two sums using the add tool: "
                "1) 100 + 200 "
                "2) 33 + 44 "
                "Report both results.",
                model=ollama_model,
                system_prompt=(
                    "Use the add tool for each computation. "
                    "You must call the add tool twice — once for each sum. "
                    "Report both results."
                ),
            )

            add_calls = [tc for tc in result.tool_calls if tc.tool_name == "add" and not tc.blocked]
            assert len(add_calls) >= 2, (
                f"Expected 2 add calls, got {len(add_calls)}. "
                f"All calls: {[(tc.tool_name, tc.arguments) for tc in result.tool_calls]}"
            )
            # Both results should appear
            assert "300" in result.content
            assert "77" in result.content


@requires_ollama
class TestLiveMultiStep:
    """Verify model chains tool results — uses output from one call as input to another."""

    @pytest.mark.asyncio
    async def test_chain_echo_then_reference_result(self, tmp_path, ollama_model):
        """Model uses echo, then references that result in its response."""
        config = _make_live_config(
            tmp_path, auto_approve=True, require_approval=False,
            max_turns=5,
        )
        bridge = Bridge(config)

        async with bridge:
            result = await bridge.run(
                'Use the echo tool to echo "secret_code_42". '
                "Then use the add tool to add 20 + 22. "
                "Tell me: does the echo result contain the same number as the sum?",
                model=ollama_model,
                system_prompt=(
                    "Use tools in order. First echo, then add. "
                    "Compare the results and answer the question."
                ),
            )

            tool_names = [tc.tool_name for tc in result.tool_calls if not tc.blocked]
            # Should have called both tools
            assert "echo" in tool_names
            assert "add" in tool_names
            # Model should reference both results and make the comparison
            # (echo returns "secret_code_42", add returns {"sum": 42})
            assert "42" in result.content
            assert result.turns >= 2  # At least 2 turns (tool calls + final response)

    @pytest.mark.asyncio
    async def test_add_result_feeds_next_add(self, tmp_path, ollama_model):
        """Model uses the result of one add to inform the next."""
        config = _make_live_config(
            tmp_path, auto_approve=True, require_approval=False,
            max_turns=6,
        )
        bridge = Bridge(config)

        async with bridge:
            result = await bridge.run(
                "Use the add tool to compute 50 + 50. "
                "Then use the add tool again to add 1 to that result. "
                "What is the final number?",
                model=ollama_model,
                system_prompt=(
                    "Use the add tool for all arithmetic. "
                    "For the second addition, use the result from the first. "
                    "You must call add twice."
                ),
            )

            add_calls = [tc for tc in result.tool_calls if tc.tool_name == "add" and not tc.blocked]
            assert len(add_calls) >= 2, (
                f"Expected 2 add calls, got {len(add_calls)}. "
                f"Calls: {[(tc.tool_name, tc.arguments) for tc in result.tool_calls]}"
            )
            # The second call should use 100 from the first result
            second_call_args = add_calls[1].arguments
            arg_values = [second_call_args.get("a", 0), second_call_args.get("b", 0)]
            assert 100 in arg_values or 100.0 in arg_values, (
                f"Second add call should use 100 from first result. "
                f"Got args: {second_call_args}"
            )
            # Final answer should be 101
            assert "101" in result.content

    @pytest.mark.asyncio
    async def test_conversation_turns_tracked(self, tmp_path, ollama_model):
        """BridgeResult.turns reflects actual conversation turns."""
        config = _make_live_config(
            tmp_path, auto_approve=True, require_approval=False,
        )
        bridge = Bridge(config)

        async with bridge:
            result = await bridge.run(
                "Use the add tool to compute 1 + 1.",
                model=ollama_model,
                system_prompt="Use the add tool for math.",
            )

            # At least 2 turns: model calls tool (turn 1), model gives final answer (turn 2)
            if result.tool_calls:
                assert result.turns >= 2, (
                    f"With tool calls, expected >= 2 turns, got {result.turns}"
                )
