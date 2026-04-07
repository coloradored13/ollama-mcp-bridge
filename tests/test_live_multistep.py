"""Live multi-step and fault tolerance tests — real Ollama + real MCP.

Tests that require multiple tool calls per conversation: multi-tool use,
result chaining, and error recovery. These are the slowest tests (~30-60s
each) because they involve multiple inference rounds.

Run: pytest tests/test_live_multistep.py
"""

from __future__ import annotations

import pytest

from tests.helpers import requires_ollama


@requires_ollama
class TestMultiToolUse:
    """Verify model uses multiple distinct tools in a single conversation."""

    @pytest.mark.asyncio
    async def test_two_different_tools_called(self, live_bridge, ollama_model):
        """Model calls both echo and add in the same conversation."""
        bridge = await live_bridge(max_turns=5)

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

        assert not result.truncated, "Conversation truncated — result is incomplete"
        tool_names_called = {tc.tool_name for tc in result.tool_calls if not tc.blocked}
        assert "echo" in tool_names_called, (
            f"echo not called. Tools called: {tool_names_called}. "
            f"Full result: {result.content[:200]}"
        )
        assert "add" in tool_names_called, (
            f"add not called. Tools called: {tool_names_called}. "
            f"Full result: {result.content[:200]}"
        )
        assert "15" in result.content, (
            f"Expected '15' in response. Got: {result.content[:300]}"
        )

    @pytest.mark.asyncio
    async def test_multiple_add_calls(self, live_bridge, ollama_model):
        """Model calls the same tool multiple times with different arguments."""
        bridge = await live_bridge(max_turns=5)

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

        assert not result.truncated, "Conversation truncated — result is incomplete"
        add_calls = [tc for tc in result.tool_calls if tc.tool_name == "add" and not tc.blocked]
        assert len(add_calls) >= 2, (
            f"Expected 2 add calls, got {len(add_calls)}. "
            f"All calls: {[(tc.tool_name, tc.arguments) for tc in result.tool_calls]}"
        )
        assert "300" in result.content
        assert "77" in result.content


@requires_ollama
class TestResultChaining:
    """Verify model chains tool results — uses output from one call as input to another."""

    @pytest.mark.asyncio
    async def test_chain_echo_then_reference_result(self, live_bridge, ollama_model):
        """Model uses echo, then references that result in its response."""
        bridge = await live_bridge(max_turns=5)

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

        assert not result.truncated, "Conversation truncated — result is incomplete"
        tool_names = [tc.tool_name for tc in result.tool_calls if not tc.blocked]
        assert "echo" in tool_names
        assert "add" in tool_names
        assert "42" in result.content
        assert result.turns >= 2

    @pytest.mark.asyncio
    async def test_add_result_feeds_next_add(self, live_bridge, ollama_model):
        """Model uses the result of one add to inform the next."""
        bridge = await live_bridge(max_turns=6)

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

        assert not result.truncated, "Conversation truncated — result is incomplete"
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
        assert "101" in result.content


@requires_ollama
class TestFaultTolerance:
    """Verify the bridge and model handle tool failures gracefully."""

    @pytest.mark.asyncio
    async def test_tool_error_fed_back_to_model(self, live_bridge, ollama_model):
        """Tool that raises an error — model receives error and responds."""
        bridge = await live_bridge(max_turns=5)

        result = await bridge.run(
            'Use the flaky_tool with input "please fail". '
            "Report what happened.",
            model=ollama_model,
            system_prompt=(
                "Use flaky_tool when asked. If the tool returns an error, "
                "tell the user what went wrong."
            ),
        )

        flaky_calls = [tc for tc in result.tool_calls if tc.tool_name == "flaky_tool"]
        assert len(flaky_calls) >= 1
        assert "ERROR" in flaky_calls[0].result_summary or "fail" in flaky_calls[0].result_summary.lower()
        assert result.content != ""
        assert not result.truncated

    @pytest.mark.asyncio
    async def test_tool_error_then_successful_tool(self, live_bridge, ollama_model):
        """Tool fails mid-chain, model recovers and uses a different tool."""
        bridge = await live_bridge(max_turns=5)

        result = await bridge.run(
            'First try flaky_tool with input "fail please". '
            "It will fail — that's expected. "
            "Then use the add tool to compute 5 + 5 instead. "
            "Report the add result.",
            model=ollama_model,
            system_prompt=(
                "Follow the steps in order. The first tool will fail. "
                "When it does, move on to the add tool. "
                "Report the add tool result."
            ),
        )

        assert not result.truncated, "Conversation truncated — result is incomplete"
        tool_names = [tc.tool_name for tc in result.tool_calls]
        assert "flaky_tool" in tool_names, (
            f"flaky_tool not attempted. Calls: {tool_names}"
        )
        assert "add" in tool_names, (
            f"Model didn't recover to add tool. Calls: {tool_names}"
        )
        assert "10" in result.content

    @pytest.mark.asyncio
    async def test_flaky_tool_succeeds_on_safe_input(self, live_bridge, ollama_model):
        """flaky_tool works fine when input doesn't contain 'fail'."""
        bridge = await live_bridge()

        result = await bridge.run(
            'Use flaky_tool with input "hello world". Report the result.',
            model=ollama_model,
            system_prompt="Use flaky_tool when asked. Report what it returns.",
        )

        assert not result.truncated, "Conversation truncated — result is incomplete"
        flaky_calls = [
            tc for tc in result.tool_calls
            if tc.tool_name == "flaky_tool" and not tc.blocked
        ]
        assert len(flaky_calls) >= 1
        assert "succeeded" in flaky_calls[0].result_summary.lower()

    @pytest.mark.asyncio
    async def test_error_captured_in_tool_records(self, live_bridge, ollama_model):
        """Tool errors are captured in BridgeResult.tool_calls for inspection."""
        bridge = await live_bridge()

        result = await bridge.run(
            'Use flaky_tool with input "fail now".',
            model=ollama_model,
            system_prompt="Use flaky_tool when asked.",
        )

        assert not result.truncated, "Conversation truncated — result is incomplete"
        flaky_calls = [tc for tc in result.tool_calls if tc.tool_name == "flaky_tool"]
        assert len(flaky_calls) >= 1
        assert "ERROR" in flaky_calls[0].result_summary

    @pytest.mark.asyncio
    async def test_conversation_completes_despite_errors(self, live_bridge, ollama_model):
        """Multiple errors don't crash the bridge — conversation completes."""
        bridge = await live_bridge(max_turns=6)

        result = await bridge.run(
            'Try flaky_tool with "fail1", then flaky_tool with "fail2". '
            "Both will error. Then use echo with 'still alive'. "
            "Report what happened.",
            model=ollama_model,
            system_prompt=(
                "Follow each step. Tools may fail — that is expected. "
                "After failures, try the echo tool. Report all outcomes."
            ),
        )

        assert result.content != ""
        assert len(result.tool_calls) >= 1
        assert not result.truncated or result.turns >= 3

    @pytest.mark.asyncio
    async def test_blocked_tool_then_fallback(self, live_bridge, ollama_model):
        """Model uses an approved tool when others aren't available."""
        bridge = await live_bridge(allowed_tools=["echo", "add"], max_turns=5)

        result = await bridge.run(
            "Try to use the add tool to compute 3 + 4.",
            model=ollama_model,
            system_prompt="Use the add tool for arithmetic. Report the result.",
        )

        assert not result.truncated, "Conversation truncated — result is incomplete"
        add_calls = [tc for tc in result.tool_calls if tc.tool_name == "add" and not tc.blocked]
        assert len(add_calls) >= 1
        assert "7" in result.content
