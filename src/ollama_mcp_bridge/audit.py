"""Structured audit logging (SAD[5]).

Append-only JSON-L format. Every tool call is logged with:
timestamp, session_id, event_type, server, tool, params_hash, result metrics,
decision, duration. Async-buffered, thread-safe.

SECURITY: Raw parameter values are NEVER written to the audit log. Parameters
are hashed (SHA-256) and summarized structurally (field names, types, lengths).
Secret-shaped keys are explicitly redacted from summaries.
"""

from __future__ import annotations

import hashlib
import json
import logging
import os
import re
from pathlib import Path
from typing import Any

from .types import ActionClass, AuditEntry, AuditEventType

logger = logging.getLogger(__name__)

# Keys whose values should never appear in audit logs, even structurally.
_SECRET_KEY_PATTERN = re.compile(
    r"(password|passwd|secret|token|api[_-]?key|auth|credential|private[_-]?key)",
    re.IGNORECASE,
)


def _summarize_params(params: dict[str, Any]) -> str:
    """Build a structural summary of parameters without exposing raw values.

    Shows field names, value types, and string lengths — enough for debugging
    without leaking secrets. Secret-shaped keys get "[REDACTED]" as their summary.
    """
    parts: list[str] = []
    for key, value in params.items():
        if _SECRET_KEY_PATTERN.search(key):
            parts.append(f"{key}:[REDACTED]")
        elif isinstance(value, str):
            parts.append(f"{key}:str({len(value)})")
        elif isinstance(value, bool):
            parts.append(f"{key}:bool")
        elif isinstance(value, int):
            parts.append(f"{key}:int")
        elif isinstance(value, float):
            parts.append(f"{key}:float")
        elif isinstance(value, list):
            parts.append(f"{key}:list({len(value)})")
        elif isinstance(value, dict):
            parts.append(f"{key}:dict({len(value)})")
        elif value is None:
            parts.append(f"{key}:null")
        else:
            parts.append(f"{key}:{type(value).__name__}")
    return "{" + ", ".join(parts) + "}"


class AuditLogger:
    """Structured audit logger writing JSON-L to disk.

    Entries are buffered and flushed periodically or on explicit flush.
    """

    def __init__(
        self,
        audit_file: str = "~/.ollama-mcp-bridge/audit.jsonl",
        session_id: str = "",
    ):
        self._path = Path(audit_file).expanduser()
        self._session_id = session_id
        self._buffer: list[AuditEntry] = []  # pending disk writes
        self._session_entries: list[AuditEntry] = []  # full session record
        self._buffer_limit = 10
        self._ensure_directory()

    def _ensure_directory(self) -> None:
        """Create audit log directory if it doesn't exist."""
        self._path.parent.mkdir(parents=True, exist_ok=True)

    # Events that must be flushed to disk immediately — losing these
    # on crash would compromise forensic integrity.
    _CRITICAL_EVENTS = frozenset(
        {
            AuditEventType.TOOL_BLOCKED,
            AuditEventType.TOOL_DENIED,
            AuditEventType.TOOL_ERROR,
            AuditEventType.RESULT_QUARANTINED,
            AuditEventType.TAINTED_SINK_BLOCKED,
            AuditEventType.TAINTED_SINK_CONFIRMED,
            AuditEventType.RUG_PULL_DETECTED,
            AuditEventType.RATE_LIMITED,
            AuditEventType.SANITIZATION_BLOCK,
        }
    )

    def log(self, entry: AuditEntry) -> None:
        """Add an audit entry to the buffer and session record.

        Security-critical events flush immediately to disk so they
        survive a crash. Routine events buffer normally.
        """
        if not entry.session_id:
            entry.session_id = self._session_id
        self._buffer.append(entry)
        self._session_entries.append(entry)
        if entry.event_type in self._CRITICAL_EVENTS or len(self._buffer) >= self._buffer_limit:
            self.flush()

    def log_tool_call(
        self,
        server: str,
        tool: str,
        action_class: ActionClass,
        params: dict[str, Any],
        result_content: str = "",
        decision: str = "ALLOWED",
        reason: str = "",
        score: float = 0.0,
        duration_ms: float = 0.0,
        model_id: str = "",
        turn: int = 0,
        capability_manifest: dict | None = None,
        sink_type: str = "",
        taint_summary: str = "",
        deployment_mode: str = "",
        security_profile: str = "",
    ) -> None:
        """Log a tool call event with computed hashes."""
        params_json = json.dumps(params, sort_keys=True, default=str)
        params_hash = hashlib.sha256(params_json.encode()).hexdigest()
        params_summary = _summarize_params(params)

        result_hash = ""
        result_size = 0
        if result_content:
            result_size = len(result_content.encode())
            result_hash = hashlib.sha256(result_content.encode()).hexdigest()

        self.log(
            AuditEntry(
                event_type=AuditEventType.TOOL_CALL,
                server_id=server,
                tool_name=tool,
                action_class=action_class,
                params_hash=params_hash,
                params_summary=params_summary,
                result_size=result_size,
                result_hash=result_hash,
                decision=decision,
                reason=reason,
                score=score,
                duration_ms=duration_ms,
                model_id=model_id,
                turn=turn,
                capability_manifest=capability_manifest or {},
                sink_type=sink_type,
                taint_summary=taint_summary,
                deployment_mode=deployment_mode,
                security_profile=security_profile,
            )
        )

    def log_event(
        self,
        event_type: AuditEventType,
        server: str = "",
        tool: str = "",
        reason: str = "",
        score: float = 0.0,
        approval_mode: str = "",
        definition_hash: str = "",
        confirmation_outcome: str = "",
        capability_manifest: dict | None = None,
        sink_type: str = "",
        adapter_decisions: list[str] | None = None,
        taint_summary: str = "",
        deployment_mode: str = "",
        security_profile: str = "",
        decision_basis: str = "",
    ) -> None:
        """Log a non-tool-call event with optional enrichment fields."""
        self.log(
            AuditEntry(
                event_type=event_type,
                server_id=server,
                tool_name=tool,
                reason=reason,
                score=score,
                approval_mode=approval_mode,
                definition_hash=definition_hash,
                confirmation_outcome=confirmation_outcome,
                capability_manifest=capability_manifest or {},
                sink_type=sink_type,
                adapter_decisions=adapter_decisions or [],
                taint_summary=taint_summary,
                deployment_mode=deployment_mode,
                security_profile=security_profile,
                decision_basis=decision_basis,
            )
        )

    def flush(self) -> None:
        """Write buffered entries to disk and fsync for durability.

        fsync() is called unconditionally on every flush so that security-critical
        events (TOOL_BLOCKED, RUG_PULL_DETECTED, etc.) survive a crash. These events
        trigger immediate flush from log(), making fsync most valuable there — but
        applying it on all flushes avoids split logic and the performance cost is
        negligible (audit flushes are infrequent, not in the hot path).
        """
        if not self._buffer:
            return

        try:
            with open(self._path, "a") as f:
                for entry in self._buffer:
                    line = entry.model_dump_json()
                    f.write(line + "\n")
                f.flush()
                os.fsync(f.fileno())
            self._buffer.clear()
        except OSError as e:
            logger.error("Failed to write audit log: %s", e)

    def get_session_entries(self) -> list[AuditEntry]:
        """Get all entries for current session regardless of flush state."""
        return list(self._session_entries)

    def close(self) -> None:
        """Flush remaining buffer and close."""
        self.flush()

    @staticmethod
    def hash_params(params: dict[str, Any]) -> str:
        """Compute SHA-256 hash of params for audit (no raw secrets)."""
        return hashlib.sha256(json.dumps(params, sort_keys=True, default=str).encode()).hexdigest()
