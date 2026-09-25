"""Live MCP server tests — real subprocess, no model needed.

Tests connection, discovery, approval flow, and fail-closed behavior
against a real MCP server subprocess. No Ollama model inference, so
these run in ~1s each.

Run: pytest tests/test_live_mcp.py
"""

from __future__ import annotations

import pytest

from tests.helpers import requires_ollama


@requires_ollama
class TestMCPConnection:
    """Verify real MCP server connection and tool discovery."""

    @pytest.mark.asyncio
    async def test_connect_discovers_tools(self, live_bridge):
        """Bridge connects to real MCP server and discovers all tools."""
        bridge = await live_bridge()

        tools = await bridge.list_discovered_tools()
        assert "test-tools" in tools
        tool_names = {t.name for t in tools["test-tools"]}
        assert tool_names == {"echo", "add", "get_secret", "flaky_tool", "slow_tool"}

        approved = bridge._security.get_approved_tools()
        assert len(approved) == 5

    @pytest.mark.asyncio
    async def test_connect_with_allowlist_subset(self, live_bridge):
        """Only allowlisted tools are approved; others stay discovered-only."""
        bridge = await live_bridge(allowed_tools=["echo"])

        approved = bridge._security.get_approved_tools()
        assert len(approved) == 1
        assert approved[0].name == "echo"

        discovered = await bridge.list_discovered_tools()
        assert len(discovered["test-tools"]) == 5

    @pytest.mark.asyncio
    async def test_empty_allowlist_blocks_all(self, live_bridge):
        """Empty allowed_tools means no tools available (fail-closed)."""
        bridge = await live_bridge(allowed_tools=[])

        approved = bridge._security.get_approved_tools()
        assert len(approved) == 0


@requires_ollama
class TestMCPApprovalFlow:
    """Verify first-run approval works with real MCP servers."""

    @pytest.mark.asyncio
    async def test_pending_approval_blocks_until_approved(self, live_bridge):
        """With require_first_run_approval, tools start pending."""
        bridge = await live_bridge(auto_approve=False, require_approval=True)

        pending = await bridge.list_pending_tool_approvals()
        assert len(pending) == 5
        assert {p.name for p in pending} == {
            "echo",
            "add",
            "get_secret",
            "flaky_tool",
            "slow_tool",
        }
        assert len(bridge._security.get_approved_tools()) == 0

    @pytest.mark.asyncio
    async def test_approval_callback_approves_tools(self, live_bridge):
        """Approval callback makes tools callable."""
        approved_names = []

        async def approve_all(pending):
            for p in pending:
                approved_names.append(p.name)
            return {p.key: True for p in pending}

        bridge = await live_bridge(
            auto_approve=False,
            require_approval=True,
            approval_callback=approve_all,
        )

        assert set(approved_names) == {
            "echo",
            "add",
            "get_secret",
            "flaky_tool",
            "slow_tool",
        }
        assert len(bridge._security.get_approved_tools()) == 5
        assert len(await bridge.list_pending_tool_approvals()) == 0

    @pytest.mark.asyncio
    async def test_selective_approval(self, live_bridge):
        """Approve some tools, deny others."""

        async def approve_echo_only(pending):
            return {p.key: (p.name == "echo") for p in pending}

        bridge = await live_bridge(
            auto_approve=False,
            require_approval=True,
            approval_callback=approve_echo_only,
        )

        approved = bridge._security.get_approved_tools()
        assert len(approved) == 1
        assert approved[0].name == "echo"
