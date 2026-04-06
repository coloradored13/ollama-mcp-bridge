"""Tests for audit.py — structured audit logging and secret protection."""

from __future__ import annotations

import json

import pytest

from ollama_mcp_bridge.audit import AuditLogger, _summarize_params
from ollama_mcp_bridge.types import ActionClass, AuditEventType


class TestSummarizeParams:
    """Structural summary must never expose raw parameter values."""

    def test_string_shows_length_not_value(self):
        summary = _summarize_params({"query": "find my secrets"})
        assert "query:str(15)" in summary
        assert "find my secrets" not in summary

    def test_secret_keys_redacted(self):
        params = {
            "password": "hunter2",
            "api_key": "sk-abc123",
            "auth_token": "bearer-xyz",
            "data": "safe value",
        }
        summary = _summarize_params(params)
        assert "hunter2" not in summary
        assert "sk-abc123" not in summary
        assert "bearer-xyz" not in summary
        assert "password:[REDACTED]" in summary
        assert "api_key:[REDACTED]" in summary
        assert "auth_token:[REDACTED]" in summary
        assert "data:str(10)" in summary

    def test_secret_key_variations(self):
        """Covers common naming patterns for secrets."""
        for key in ["PASSWORD", "Api_Key", "private_key", "credential", "SECRET", "apikey"]:
            summary = _summarize_params({key: "sensitive"})
            assert "[REDACTED]" in summary, f"Key '{key}' was not redacted"

    def test_type_summaries(self):
        params = {
            "name": "test",
            "count": 42,
            "ratio": 3.14,
            "active": True,
            "tags": ["a", "b", "c"],
            "meta": {"x": 1},
            "empty": None,
        }
        summary = _summarize_params(params)
        assert "name:str(4)" in summary
        assert "count:int" in summary
        assert "ratio:float" in summary
        assert "active:bool" in summary
        assert "tags:list(3)" in summary
        assert "meta:dict(1)" in summary
        assert "empty:null" in summary

    def test_empty_params(self):
        assert _summarize_params({}) == "{}"


class TestAuditLoggerSecrets:
    """Verify secrets never reach disk through the audit log."""

    def test_password_not_in_audit_file(self, tmp_path):
        audit = AuditLogger(
            audit_file=str(tmp_path / "audit.jsonl"),
            session_id="test",
        )

        audit.log_tool_call(
            server="test",
            tool="login",
            action_class=ActionClass.WRITE,
            params={"username": "admin", "password": "hunter2"},
        )
        audit.flush()

        content = (tmp_path / "audit.jsonl").read_text()
        assert "hunter2" not in content
        assert "password:[REDACTED]" in content

    def test_api_key_not_in_audit_file(self, tmp_path):
        audit = AuditLogger(
            audit_file=str(tmp_path / "audit.jsonl"),
            session_id="test",
        )

        audit.log_tool_call(
            server="test",
            tool="call_api",
            action_class=ActionClass.READ,
            params={"url": "https://api.example.com", "api_key": "sk-secret-key-123"},
        )
        audit.flush()

        content = (tmp_path / "audit.jsonl").read_text()
        assert "sk-secret-key-123" not in content
        assert "api_key:[REDACTED]" in content
        # Non-secret values should show structural info
        assert "url:str(" in content

    def test_params_hash_still_present(self, tmp_path):
        """Hash is computed from full params (including secrets) for correlation."""
        audit = AuditLogger(
            audit_file=str(tmp_path / "audit.jsonl"),
            session_id="test",
        )

        audit.log_tool_call(
            server="test",
            tool="login",
            action_class=ActionClass.WRITE,
            params={"password": "hunter2"},
        )
        audit.flush()

        content = (tmp_path / "audit.jsonl").read_text()
        entry = json.loads(content.strip())
        assert len(entry["params_hash"]) == 64  # SHA-256 hex


class TestAuditSessionRetention:
    """get_session_entries() must return ALL entries, not just unflushed buffer."""

    def test_entries_survive_flush(self, tmp_path):
        audit = AuditLogger(
            audit_file=str(tmp_path / "audit.jsonl"),
            session_id="test",
        )

        audit.log_tool_call(
            server="s1", tool="t1", action_class=ActionClass.READ, params={"a": "1"},
        )
        audit.flush()
        audit.log_tool_call(
            server="s2", tool="t2", action_class=ActionClass.WRITE, params={"b": "2"},
        )

        entries = audit.get_session_entries()
        assert len(entries) == 2
        assert entries[0].tool_name == "t1"
        assert entries[1].tool_name == "t2"

    def test_tiny_buffer_limit_retains_all(self, tmp_path):
        """Even with buffer_limit=1, all entries are available in session."""
        audit = AuditLogger(
            audit_file=str(tmp_path / "audit.jsonl"),
            session_id="test",
        )
        audit._buffer_limit = 1  # force flush after every entry

        for i in range(5):
            audit.log_tool_call(
                server="s", tool=f"t{i}", action_class=ActionClass.READ, params={},
            )

        entries = audit.get_session_entries()
        assert len(entries) == 5
        assert [e.tool_name for e in entries] == ["t0", "t1", "t2", "t3", "t4"]

    def test_close_does_not_lose_entries(self, tmp_path):
        """close() flushes to disk but session entries remain available."""
        audit = AuditLogger(
            audit_file=str(tmp_path / "audit.jsonl"),
            session_id="test",
        )

        audit.log_tool_call(
            server="s", tool="t", action_class=ActionClass.READ, params={"x": "y"},
        )
        audit.close()

        entries = audit.get_session_entries()
        assert len(entries) == 1


class TestAuditEnrichedFields:
    """PR5 fields (approval_mode, definition_hash, confirmation_outcome) must serialize to disk."""

    def test_enriched_fields_written_to_jsonl(self, tmp_path):
        """New fields survive the full write path: log_event → buffer → flush → disk."""
        audit = AuditLogger(
            audit_file=str(tmp_path / "audit.jsonl"),
            session_id="test",
        )

        audit.log_event(
            AuditEventType.TOOL_FIRST_APPROVED,
            server="sigma-mem",
            tool="recall",
            reason="User approved first-seen tool",
            approval_mode="first_run_explicit",
            definition_hash="abc123def456",
        )
        audit.log_event(
            AuditEventType.TOOL_TIMEOUT,
            server="files",
            tool="delete",
            reason="Confirmation timed out",
            confirmation_outcome="TIMEOUT",
        )
        audit.flush()

        lines = (tmp_path / "audit.jsonl").read_text().strip().split("\n")
        assert len(lines) == 2

        approval = json.loads(lines[0])
        assert approval["approval_mode"] == "first_run_explicit"
        assert approval["definition_hash"] == "abc123def456"
        assert approval["event_type"] == "tool_first_approved"

        timeout = json.loads(lines[1])
        assert timeout["confirmation_outcome"] == "TIMEOUT"
        assert timeout["event_type"] == "tool_timeout"

    def test_enriched_fields_default_empty(self, tmp_path):
        """Unenriched events still serialize — new fields default to empty string."""
        audit = AuditLogger(
            audit_file=str(tmp_path / "audit.jsonl"),
            session_id="test",
        )

        audit.log_event(AuditEventType.SESSION_START)
        audit.flush()

        content = (tmp_path / "audit.jsonl").read_text().strip()
        entry = json.loads(content)
        assert entry["approval_mode"] == ""
        assert entry["definition_hash"] == ""
        assert entry["confirmation_outcome"] == ""

    def test_definition_hash_not_a_secret(self, tmp_path):
        """definition_hash is a SHA-256 hex digest — safe to write to audit log."""
        audit = AuditLogger(
            audit_file=str(tmp_path / "audit.jsonl"),
            session_id="test",
        )

        hash_val = "a" * 64  # SHA-256 length
        audit.log_event(
            AuditEventType.TOOL_FIRST_APPROVED,
            server="s",
            tool="t",
            definition_hash=hash_val,
        )
        audit.flush()

        content = (tmp_path / "audit.jsonl").read_text()
        assert hash_val in content
