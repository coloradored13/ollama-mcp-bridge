"""Shared data types for ollama-mcp-bridge.

This module is the contract surface — all internal modules import types from here.
Types are organized into four groups:

1. **Transport types**: Raw data from Ollama and MCP servers (untrusted input).
2. **Security types**: Results of security processing (sanitization, validation, gating).
3. **Audit types**: Structured logging entries for forensic review.
4. **Consumer types**: What Bridge.run() returns to the caller.

The separation between ToolSchema (raw, untrusted) and ApprovedTool (scanned, approved)
is a key security boundary. Code that receives an ApprovedTool can trust that security
scanning has occurred. Code that receives a ToolSchema must assume it could be malicious.
"""

from __future__ import annotations

import hashlib
import ipaddress
import json
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from urllib.parse import urlparse

from pydantic import BaseModel, ConfigDict, Field


# --- Transport types ---


class ToolSchema(BaseModel):
    """Raw tool schema as received from an MCP server — UNTRUSTED.

    This represents a tool definition before any security processing. The description,
    parameter names, enum values, and defaults could all contain malicious instructions
    (tool poisoning attack — see Invariant Labs disclosure, CyberArk research).

    Frozen (immutable) so the original definition is preserved for hash comparison.
    The hash is used for rug-pull detection: if a server changes a tool definition
    after initial approval, the hash won't match and the tool is blocked.
    """

    model_config = ConfigDict(frozen=True)

    server: str
    name: str
    description: str
    input_schema: dict[str, Any]

    @property
    def raw_definition(self) -> str:
        """Canonical JSON serialization for hash-based integrity checking.

        Uses sorted keys and compact separators so the same logical definition
        always produces the same string, regardless of Python dict ordering.
        Only includes fields that define the tool's behavior — server name is
        excluded because the same tool on a different server is a different thing.
        """
        return json.dumps(
            {"name": self.name, "description": self.description, "input_schema": self.input_schema},
            sort_keys=True,
            separators=(",", ":"),
        )

    @property
    def definition_hash(self) -> str:
        """SHA-256 hash for rug-pull detection.

        Stored at approval time. Rechecked on every reconnect. If the hash
        changes, the tool definition was modified after approval — this is
        the "rug pull" attack where a trusted server swaps a tool's behavior.
        """
        return hashlib.sha256(self.raw_definition.encode()).hexdigest()


class SourceType(str, Enum):
    """Origin of content flowing through the bridge.

    Determines baseline trust assumptions. Content from different sources
    carries different risk profiles — a tool_result from a web scraper is
    fundamentally different from a user message, even though both arrive
    as strings. The SinkPolicyEngine (PR 7) uses source type to decide
    what untrusted content is allowed to influence.
    """

    USER = "user"
    SYSTEM = "system"
    DEVELOPER_POLICY = "developer_policy"
    TOOL_RESULT = "tool_result"
    DOCUMENT = "document"
    WEBPAGE = "webpage"
    EMAIL = "email"
    MEMORY = "memory"
    UNKNOWN = "unknown"


class TrustLevel(str, Enum):
    """Trust classification of content origin.

    TRUSTED: System-generated or developer-policy content. Can issue instructions.
    USER_CONTROLLED: Direct user input. Trusted for intent but may contain errors.
    THIRD_PARTY: Content from external tools, documents, web. Must not be treated
        as instructions — this is the core "separate data from instructions" principle.
    UNKNOWN: Origin cannot be determined. Treated as third_party for security.
    """

    TRUSTED = "trusted"
    USER_CONTROLLED = "user_controlled"
    THIRD_PARTY = "third_party"
    UNKNOWN = "unknown"


class ActionClass(str, Enum):
    """Tool action classification — determines which security gate applies.

    READ: auto-approved, no confirmation needed.
    WRITE: auto-approved by default (configurable to require confirmation).
    DESTRUCTIVE: requires explicit human confirmation before execution.
        Timeout on confirmation defaults to denied (fail-closed).

    Classification is set in bridge.toml per-tool via destructive_tools list.
    Unclassified tools default to WRITE (not READ) as a security precaution.
    """

    READ = "READ"
    WRITE = "WRITE"
    DESTRUCTIVE = "DESTRUCTIVE"


class ApprovalMode(str, Enum):
    """How a tool was approved — tracks the trust provenance of each registry entry.

    FIRST_RUN_EXPLICIT: Human approved via approval callback during first-run scan.
    AUTO_APPROVED: auto_approve_first_seen=True or require_first_run_approval=False.
    REAPPROVED: Re-approved after rug-pull detection (hash changed, user re-confirmed).
    LEGACY: Migrated from old flat-hash registry format (pre-PR3). No metadata available.
    """

    FIRST_RUN_EXPLICIT = "first_run_explicit"
    AUTO_APPROVED = "auto_approved"
    REAPPROVED = "reapproved"
    LEGACY = "legacy"


class RegistryEntry(BaseModel):
    """Structured approval record for a single tool in the registry.

    Replaces the old flat {key: hash} format with rich metadata that
    answers: who approved this, when, how, and was it ever denied?
    """

    server: str
    tool_name: str
    approved_hash: str
    approved_at: datetime | None = None
    approval_mode: ApprovalMode = ApprovalMode.LEGACY
    classification: str = ""  # READ/WRITE/DESTRUCTIVE — informational
    notes: str | None = None
    last_seen_at: datetime | None = None
    denied_hashes: list[str] = Field(default_factory=list)
    capabilities: dict[str, Any] = Field(default_factory=dict)  # capability manifest snapshot at approval


class ToolState(str, Enum):
    """State of a tool in the first-run approval pipeline.

    Lifecycle:
      DISCOVERED → ALLOWLISTED → [sanitize] → PENDING_FIRST_APPROVAL → APPROVED / DENIED_BY_USER
      DISCOVERED → ALLOWLISTED → [sanitize] → BLOCKED_SANITIZATION  (terminal)
      DISCOVERED → ALLOWLISTED → [integrity] → BLOCKED_INTEGRITY    (terminal)
      DISCOVERED → ALLOWLISTED → [hash match] → APPROVED            (auto, skip pending)
    """

    DISCOVERED = "DISCOVERED"
    ALLOWLISTED = "ALLOWLISTED"
    PENDING_FIRST_APPROVAL = "PENDING_FIRST_APPROVAL"
    APPROVED = "APPROVED"
    BLOCKED_SANITIZATION = "BLOCKED_SANITIZATION"
    BLOCKED_INTEGRITY = "BLOCKED_INTEGRITY"
    DENIED_BY_USER = "DENIED_BY_USER"


class CapabilitySource(str, Enum):
    """How a tool's capability manifest was determined.

    CONFIG: Operator explicitly declared capabilities in bridge.toml.
        Highest trust — operator knows exactly what the tool does.
    MCP_DECLARED: MCP server provided capability metadata (future).
        Medium trust — server may misrepresent capabilities.
    INFERRED: Bridge inferred capabilities from tool name/description/schema.
        Lowest trust — conservative heuristic, may be wrong in either direction.
    """

    CONFIG = "config"
    MCP_DECLARED = "mcp_declared"
    INFERRED = "inferred"


class ToolCapabilityManifest(BaseModel):
    """Explicit capability metadata for a tool — replaces name-pattern inference.

    Each boolean flag declares a specific dangerous capability. The sink policy
    engine uses these flags to classify sinks instead of relying on tool name
    patterns (which are unreliable — a tool named "update_record" might actually
    send HTTP requests).

    Source of truth (priority order):
    1. Explicit config overrides ([capabilities.<server>.<tool>] in bridge.toml)
    2. MCP-side declarations (future — when MCP protocol supports it)
    3. Conservative bridge inference (fallback heuristic from name/description/schema)

    All flags default to False. Conservative inference should set flags True when
    uncertain — false positives (blocking a safe tool) are better than false
    negatives (allowing a dangerous tool unchecked).
    """

    model_config = ConfigDict(frozen=True)

    network_access: bool = False
    outbound_data_transfer: bool = False
    filesystem_read: bool = False
    filesystem_write: bool = False
    filesystem_delete: bool = False
    memory_write: bool = False
    external_messaging: bool = False
    code_execution: bool = False
    credential_access: bool = False
    user_identity_impact: bool = False
    destructive: bool = False
    high_consequence: bool = False
    source: CapabilitySource = CapabilitySource.INFERRED

    @property
    def is_dangerous(self) -> bool:
        """True if any high-risk capability flag is set."""
        return any([
            self.outbound_data_transfer,
            self.filesystem_delete,
            self.external_messaging,
            self.code_execution,
            self.credential_access,
            self.user_identity_impact,
            self.destructive,
            self.high_consequence,
        ])

    @property
    def has_outbound_capability(self) -> bool:
        """True if tool can send data externally."""
        return self.network_access or self.outbound_data_transfer or self.external_messaging

    @property
    def has_filesystem_capability(self) -> bool:
        """True if tool can interact with the filesystem."""
        return self.filesystem_read or self.filesystem_write or self.filesystem_delete

    def to_audit_dict(self) -> dict[str, Any]:
        """Compact representation for audit log entries."""
        # Only include True flags to keep audit entries concise
        flags = {
            k: v for k, v in self.model_dump().items()
            if v is True and k != "source"
        }
        flags["source"] = self.source.value
        return flags


class DestinationMatchResult(BaseModel):
    """Result of matching a URL against a DestinationPolicy."""

    matched: bool = False
    policy_host: str = ""
    checked_url: str = ""
    failure_reason: str = ""  # empty if matched


class DestinationPolicy(BaseModel):
    """Typed destination constraint for outbound tool calls.

    Replaces coarse domain-only allowlists with rich, per-field validation:
    scheme, host, port, path, query, IP literal controls, redirect controls.

    Configured per server+tool via [[destinations.<server>.<tool>]] TOML sections.
    Multiple policies per tool are supported (array of tables). A URL is allowed
    if it matches ANY one of the configured policies for the tool.
    """

    model_config = ConfigDict(frozen=True)

    scheme: str = "https"
    host: str  # required — the target hostname
    port: int | None = None  # None = any port allowed for the scheme
    path_prefixes: list[str] = Field(default_factory=list)  # empty = any path
    query_constraints: dict[str, str] = Field(default_factory=dict)
    allow_subdomains: bool = False
    allow_ip_literals: bool = False
    allow_private_ranges: bool = False
    allow_redirects: bool = False
    allowed_methods: list[str] = Field(default_factory=list)  # empty = any method
    max_payload_bytes: int = 65536

    def matches(self, url: str) -> DestinationMatchResult:
        """Check whether a URL satisfies all constraints in this policy.

        Validates scheme, host, IP literal status, private range, port,
        and path prefix in order. Returns a structured result with the
        specific failure reason if the URL does not match.
        """
        base = DestinationMatchResult(policy_host=self.host, checked_url=url[:200])

        try:
            parsed = urlparse(url)
        except Exception:
            return DestinationMatchResult(
                policy_host=self.host, checked_url=url[:200],
                failure_reason="malformed URL",
            )

        hostname = parsed.hostname
        if not hostname:
            return DestinationMatchResult(
                policy_host=self.host, checked_url=url[:200],
                failure_reason="URL has no hostname",
            )

        # 1. Scheme
        if parsed.scheme.lower() != self.scheme.lower():
            return DestinationMatchResult(
                **{**base.model_dump(), "failure_reason": (
                    f"scheme '{parsed.scheme}' does not match "
                    f"required '{self.scheme}'"
                )},
            )

        # 2. IP literal check (before host comparison)
        is_ip = False
        try:
            addr = ipaddress.ip_address(hostname)
            is_ip = True
        except ValueError:
            pass

        if is_ip and not self.allow_ip_literals:
            return DestinationMatchResult(
                **{**base.model_dump(), "failure_reason": (
                    f"IP literal '{hostname}' not allowed "
                    f"(allow_ip_literals=False)"
                )},
            )

        # 3. Private range check
        if is_ip and not self.allow_private_ranges:
            if addr.is_private or addr.is_loopback or addr.is_link_local:
                return DestinationMatchResult(
                    **{**base.model_dump(), "failure_reason": (
                        f"private/loopback IP '{hostname}' not allowed "
                        f"(allow_private_ranges=False)"
                    )},
                )

        # 4. Host match
        if not is_ip:
            hostname_lower = hostname.lower()
            policy_host_lower = self.host.lower()
            if hostname_lower == policy_host_lower:
                pass  # exact match
            elif self.allow_subdomains and hostname_lower.endswith(
                f".{policy_host_lower}"
            ):
                pass  # subdomain match
            else:
                return DestinationMatchResult(
                    **{**base.model_dump(), "failure_reason": (
                        f"host '{hostname}' does not match "
                        f"policy host '{self.host}'"
                        f"{' (subdomains not allowed)' if not self.allow_subdomains else ''}"
                    )},
                )
        else:
            # IP literal that passed the allow check — still must match host
            if hostname != self.host:
                return DestinationMatchResult(
                    **{**base.model_dump(), "failure_reason": (
                        f"IP '{hostname}' does not match policy host '{self.host}'"
                    )},
                )

        # 5. Port
        if self.port is not None:
            url_port = parsed.port
            if url_port is None:
                # Use default port for scheme
                url_port = 443 if self.scheme.lower() == "https" else 80
            if url_port != self.port:
                return DestinationMatchResult(
                    **{**base.model_dump(), "failure_reason": (
                        f"port {url_port} does not match required port {self.port}"
                    )},
                )

        # 6. Path prefixes
        if self.path_prefixes:
            path = parsed.path or "/"
            if not any(path.startswith(prefix) for prefix in self.path_prefixes):
                return DestinationMatchResult(
                    **{**base.model_dump(), "failure_reason": (
                        f"path '{path}' does not match any allowed prefix "
                        f"({', '.join(self.path_prefixes)})"
                    )},
                )

        return DestinationMatchResult(
            matched=True, policy_host=self.host, checked_url=url[:200],
        )


class ApprovedTool(BaseModel):
    """Tool that has passed the full security ingestion pipeline.

    An ApprovedTool has been:
    1. Checked against the server's allowlist (default-deny)
    2. Scanned by all 7 sanitization detectors for poisoning
    3. Verified against its stored hash (rug-pull detection)
    4. Classified as READ/WRITE/DESTRUCTIVE
    5. Assigned a capability manifest (config > MCP > inference)

    Receiving an ApprovedTool means security scanning HAS occurred.
    This is the type boundary between "untrusted MCP data" and
    "validated tool ready for use". Only ApprovedTools are presented
    to the Ollama model and only ApprovedTools can be executed.
    """

    model_config = ConfigDict(frozen=True)

    server: str
    name: str
    description: str  # may have been sanitized (suspicious substrings removed)
    input_schema: dict[str, Any]
    classification: ActionClass = ActionClass.WRITE
    definition_hash: str  # stored for ongoing integrity checks
    capabilities: ToolCapabilityManifest = Field(default_factory=ToolCapabilityManifest)

    @property
    def namespaced_name(self) -> str:
        """Server-namespaced tool name using __ separator.

        Multiple MCP servers can expose tools with the same name (e.g., both
        sigma-mem and files could have "search"). Namespacing prevents collisions
        when presenting tools to the model: "sigma-mem__search" vs "files__search".
        The __ separator is stripped before calling MCP (MCP uses bare names).
        """
        return f"{self.server}__{self.name}"


class OllamaToolCall(BaseModel):
    """Tool call extracted from an Ollama model response — UNTRUSTED.

    Everything in this object was generated by the model. The function_name
    could be hallucinated (not a real tool). The arguments could contain
    injection attempts (path traversal, shell metacharacters). Nothing here
    should be passed to MCP without validation by SecurityGateway.
    """

    function_name: str  # may be namespaced (server__tool) or bare (model dropped prefix)
    arguments: dict[str, Any]  # model-generated — validated before MCP call

    @property
    def server(self) -> str | None:
        """Extract server name from namespaced function name."""
        if "__" in self.function_name:
            return self.function_name.split("__", 1)[0]
        return None

    @property
    def tool_name(self) -> str:
        """Extract tool name, stripping server namespace."""
        if "__" in self.function_name:
            return self.function_name.split("__", 1)[1]
        return self.function_name


# --- Security types ---


class SanitizationDecision(str, Enum):
    """Three-tier decision from the tool description sanitization pipeline.

    PASS (score < 40): No suspicious patterns detected. Tool is safe to present to model.
    WARN (score 40-69): Suspicious patterns found but below blocking threshold.
        Tool is approved with a warning logged to audit. May be a false positive
        (e.g., a legitimate tool that mentions URLs in its description).
    BLOCK (score >= 70): High-confidence malicious content detected. Tool is rejected
        entirely and never presented to the model. User can override in config.

    Thresholds are configurable in SecurityConfig. The scoring system uses the
    maximum score across all detectors (not average) — a single high-confidence
    detection is enough to block, even if other detectors see nothing.
    """

    PASS = "PASS"
    WARN = "WARN"
    BLOCK = "BLOCK"


class SanitizationResult(BaseModel):
    """Result of sanitizing a tool definition."""

    decision: SanitizationDecision
    score: float = 0.0
    triggered_rules: list[str] = Field(default_factory=list)
    sanitized_description: str = ""
    original_description: str = ""


class ValidationResult(BaseModel):
    """Result of parameter validation (SAD[3])."""

    valid: bool
    validated_params: dict[str, Any] = Field(default_factory=dict)
    errors: list[str] = Field(default_factory=list)


class ConfirmationOutcome(str, Enum):
    """Outcome of a human confirmation request — forensic-grade distinction.

    CONFIRMED: Human explicitly approved the action.
    DENIED: Human explicitly denied the action.
    TIMEOUT: Confirmation request timed out with no response.
    NO_CALLBACK: No confirmation callback was registered (fail-closed).

    The distinction between DENIED and TIMEOUT matters for forensics:
    a timeout might be a UX issue (user didn't see the prompt), while
    a denial is a deliberate decision. Different root causes need
    different responses.
    """

    CONFIRMED = "CONFIRMED"
    DENIED = "DENIED"
    TIMEOUT = "TIMEOUT"
    NO_CALLBACK = "NO_CALLBACK"


class ContentProvenance(BaseModel):
    """Metadata tracking where content originated and its trust implications.

    Attached to every non-system content object flowing through the bridge.
    Enables the security pipeline to make source-aware decisions: a URL in a
    tool result from a web search tool (third_party) is treated differently
    than a URL the user typed directly (user_controlled).

    The can_issue_instructions field is the key security property: only TRUSTED
    sources should have this set to True. Third-party content with instructions
    is the definition of prompt injection.
    """

    source_type: SourceType = SourceType.UNKNOWN
    trust_level: TrustLevel = TrustLevel.UNKNOWN
    origin_id: str = ""
    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    can_issue_instructions: bool = False
    can_contain_sensitive_data: bool = False


class SemanticRiskAssessment(BaseModel):
    """Structured risk assessment of a content item.

    Produced by SemanticRiskAssessor — replaces the binary pass/block decision
    with a structured output that downstream components (SinkPolicyEngine in PR 7)
    can reason about. Each boolean flag maps to a specific attack pattern.

    The overall_risk_score is 0.0-1.0 (normalized from detector scores).
    Individual flags indicate which specific attack patterns were detected.
    The explanation field provides human-readable reasoning.
    raw_signals lists the specific detector matches for forensic review.
    """

    overall_risk_score: float = 0.0
    attempts_instruction_override: bool = False
    attempts_tool_routing: bool = False
    attempts_permission_escalation: bool = False
    attempts_exfiltration: bool = False
    requests_sensitive_data: bool = False
    proposes_external_destination: bool = False
    contains_social_pressure: bool = False
    contains_urgency_manipulation: bool = False
    contains_hidden_or_obfuscated_instructions: bool = False
    explanation: str = ""
    raw_signals: list[str] = Field(default_factory=list)


class TaintState(BaseModel):
    """Tracks whether tool call arguments were influenced by untrusted content.

    Computed by TaintTracker: compares values in tool call arguments (URLs,
    domains, emails, IPs) against values extracted from previous tool results.
    If a match is found, the arguments are "tainted" — they were influenced by
    content the system didn't originate and shouldn't blindly trust.

    The SinkPolicyEngine uses TaintState to decide whether to allow, gate, or
    block the tool call based on what kind of sink the target tool is.
    """

    tainted: bool = False
    taint_sources: list[str] = Field(default_factory=list)
    taint_reasons: list[str] = Field(default_factory=list)
    affected_fields: list[str] = Field(default_factory=list)
    confidence: float = 0.0


class InfluenceType(str, Enum):
    """Classification of how untrusted content influenced tool call args.

    DIRECT_VALUE_MATCH: Exact or near-exact value propagation (existing taint).
    DERIVED_*: Value was transformed but still traceable to untrusted origin.
    """

    DIRECT_VALUE_MATCH = "direct_value_match"
    DERIVED_URL_REUSE = "derived_url_reuse"
    DERIVED_PROTOCOL_CHANGE = "derived_protocol_change"
    DERIVED_EMAIL_DOMAIN = "derived_email_domain"
    DERIVED_HOSTNAME_IN_URL = "derived_hostname_in_url"


class InfluenceEvidence(BaseModel):
    """Single piece of evidence linking a tool call arg to an untrusted source."""

    influence_type: InfluenceType
    tracked_value: str = ""
    arg_value: str = ""
    origin_id: str = ""
    confidence: float = 0.0


class InfluenceState(TaintState):
    """Richer taint state with derivation tracking (PR 13).

    Extends TaintState so all existing consumers (SinkPolicyEngine, SecurityGateway)
    work unchanged. New fields provide structured evidence of how untrusted content
    influenced tool call arguments — direct copy vs. derived transformation.
    """

    direct_value_match: bool = False
    derived_from_untrusted_value: bool = False
    destination_influenced: bool = False
    evidence: list[InfluenceEvidence] = Field(default_factory=list)


class SinkDecision(str, Enum):
    """Policy decision for a tool call evaluated by SinkPolicyEngine.

    ALLOW: Proceed normally — no taint concern or non-sensitive sink.
    ALLOW_WITH_NOTICE: Tainted but low-risk sink. Logged to audit, not blocked.
    REQUIRE_CONFIRMATION: Tainted + moderately sensitive sink. Human must confirm.
    BLOCK: Tainted + sensitive sink (outbound/memory write). Blocked outright.
    """

    ALLOW = "ALLOW"
    ALLOW_WITH_NOTICE = "ALLOW_WITH_NOTICE"
    REQUIRE_CONFIRMATION = "REQUIRE_CONFIRMATION"
    BLOCK = "BLOCK"


class GateDecision(str, Enum):
    """Action gate decision (SAD[6])."""

    APPROVED = "APPROVED"
    NEEDS_CONFIRMATION = "NEEDS_CONFIRMATION"
    DENIED = "DENIED"


class ResultSanitizationTier(str, Enum):
    """Four-tier classification for tool result sanitization.

    Tool results are the second major injection vector (after tool descriptions).
    A web search tool might return attacker-controlled HTML. A file read tool might
    return a file containing "SYSTEM: ignore previous instructions". The model treats
    tool results as context and may follow embedded instructions.

    CLEAN: No suspicious patterns. Result is prepended with a provenance tag
        ("[TOOL RESULT -- EXTERNAL DATA]") as a best-effort signal to the model
        that this content is external and untrusted.
    ANNOTATED: Minor suspicious content. Warning appended but content preserved.
    REDACTED: Suspicious patterns (role prefixes, instruction language) stripped.
        Remaining clean content is passed through with a notice appended.
    QUARANTINED: Heavy injection attempt (3+ role prefixes or 2+ instruction
        patterns). Full result is blocked — model sees only a placeholder message.
        The original unfiltered result is written to the audit log for human review.
    """

    CLEAN = "CLEAN"
    ANNOTATED = "ANNOTATED"
    REDACTED = "REDACTED"
    QUARANTINED = "QUARANTINED"


class ExecutionResult(BaseModel):
    """Result of SecurityGateway.execute_tool() — the output of the atomic pipeline.

    By the time code receives an ExecutionResult, the full security pipeline has run:
    permission check, parameter validation, action gating, MCP execution, result
    sanitization, semantic risk assessment, and audit logging. The content is safe
    to inject into the model's conversation context (though the provenance tag
    reminds the model it's external).

    The provenance and risk_assessment fields enable downstream components
    (SinkPolicyEngine in PR 7) to make source-aware decisions about what the
    model can do with this content.
    """

    content: str  # sanitized content — safe to return to model
    is_error: bool = False
    sanitization_tier: ResultSanitizationTier = ResultSanitizationTier.CLEAN
    server: str = ""
    tool_name: str = ""
    duration_ms: float = 0.0
    provenance: ContentProvenance | None = None
    risk_assessment: SemanticRiskAssessment | None = None


class PendingToolApproval(BaseModel):
    """A tool awaiting first-run human approval.

    Presented to the approval callback with enough context for a human
    to make an informed approve/deny decision.
    """

    model_config = ConfigDict(frozen=True)

    server: str
    name: str
    description: str
    input_schema: dict[str, Any]
    definition_hash: str
    sanitization_result: SanitizationResult

    @property
    def key(self) -> str:
        """Unique key for this tool in callback response dict."""
        return f"{self.server}:{self.name}"


class ScanResult(BaseModel):
    """Structured result of SecurityGateway.connect_and_scan().

    Gives the caller a complete picture of what happened during scan:
    which tools were approved, which are pending, which were blocked.
    """

    approved: dict[str, list[ApprovedTool]] = Field(default_factory=dict)
    pending: list[PendingToolApproval] = Field(default_factory=list)
    blocked_sanitization: list[tuple[str, str]] = Field(default_factory=list)
    blocked_integrity: list[tuple[str, str]] = Field(default_factory=list)
    denied: list[tuple[str, str]] = Field(default_factory=list)

    @property
    def total_approved(self) -> int:
        return sum(len(t) for t in self.approved.values())

    @property
    def has_pending(self) -> bool:
        return len(self.pending) > 0


# --- Audit types ---


class AuditEventType(str, Enum):
    """Types of events recorded in audit log."""

    TOOL_CALL = "tool_call"
    TOOL_BLOCKED = "tool_blocked"
    TOOL_CONFIRMED = "tool_confirmed"
    TOOL_DENIED = "tool_denied"
    TOOL_ERROR = "tool_error"
    RATE_LIMITED = "rate_limited"
    RUG_PULL_DETECTED = "rug_pull_detected"
    SANITIZATION_WARN = "sanitization_warn"
    SANITIZATION_BLOCK = "sanitization_block"
    RESULT_QUARANTINED = "result_quarantined"
    TOOL_TIMEOUT = "tool_timeout"
    TOOL_PENDING_APPROVAL = "tool_pending_approval"
    TOOL_FIRST_APPROVED = "tool_first_approved"
    TOOL_FIRST_DENIED = "tool_first_denied"
    TOOL_REAPPROVAL_REQUIRED = "tool_reapproval_required"
    TAINTED_SINK_DETECTED = "tainted_sink_detected"
    TAINTED_SINK_BLOCKED = "tainted_sink_blocked"
    TAINTED_SINK_CONFIRMED = "tainted_sink_confirmed"
    SESSION_START = "session_start"
    SESSION_END = "session_end"


class AuditEntry(BaseModel):
    """Single entry in the structured audit log.

    Every tool call, security decision, and lifecycle event is logged here.
    The audit log enables forensic review of what a model did during a session.
    This is critical because local models have no built-in safety training —
    the audit log is the only record of model behavior.

    Security note: params_hash stores a SHA-256 of parameters, NOT the raw params.
    This prevents secrets (API keys, passwords) that might appear in tool arguments
    from being written to the audit log in plaintext. params_summary is truncated
    to 200 chars as a readable hint without full exposure.
    """

    timestamp: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    session_id: str = ""
    event_type: AuditEventType
    server_id: str = ""
    tool_name: str = ""
    action_class: ActionClass | None = None
    params_hash: str = ""  # SHA-256 of params — never raw params (may contain secrets)
    params_summary: str = ""  # structural summary (field names, types, lengths) — no raw values
    result_size: int = 0
    result_hash: str = ""
    decision: str = ""  # ALLOWED, BLOCKED, DENIED, etc.
    reason: str = ""  # why the decision was made
    score: float = 0.0  # sanitization score if applicable
    duration_ms: float = 0.0
    model_id: str = ""
    turn: int = 0  # which turn of the multi-turn loop
    approval_mode: str = ""  # ApprovalMode value for approval events
    definition_hash: str = ""  # tool definition hash for approval/integrity events
    confirmation_outcome: str = ""  # ConfirmationOutcome value for gate events


# --- Consumer-facing types ---


class ToolCallRecord(BaseModel):
    """Record of a single tool call for consumer inspection."""

    server: str
    tool_name: str
    arguments: dict[str, Any]  # sanitized — no secrets
    result_summary: str = ""  # first 200 chars of result
    duration_ms: float = 0.0
    blocked: bool = False
    block_reason: str = ""


class BridgeResult(BaseModel):
    """Result of Bridge.run() — returned to consumer."""

    content: str  # final model response text
    tool_calls: list[ToolCallRecord] = Field(default_factory=list)
    audit_log: list[AuditEntry] = Field(default_factory=list)
    model: str = ""
    turns: int = 0
    truncated: bool = False  # True if max_turns reached


class StreamEventType(str, Enum):
    """Types of streaming events."""

    TEXT = "text"
    TOOL_CALL = "tool_call"
    TOOL_RESULT = "tool_result"
    CONFIRMATION_NEEDED = "confirmation_needed"
    ERROR = "error"
    DONE = "done"


class StreamEvent(BaseModel):
    """Single event in streaming mode."""

    type: StreamEventType
    content: str | None = None
    tool: str | None = None
    server: str | None = None
    error: str | None = None


# --- Server health ---


class ServerHealth(BaseModel):
    """Health status of an MCP server connection."""

    name: str
    connected: bool = False
    tools_count: int = 0
    last_call_ms: float | None = None
    error: str | None = None
