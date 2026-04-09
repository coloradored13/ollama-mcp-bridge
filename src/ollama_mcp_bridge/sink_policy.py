"""Taint tracking and sink policy engine for source-to-sink security.

This module is the core of PR 7's "protect sinks, not just sources" architecture.
Instead of only scanning content for suspicious patterns (PR 6), it tracks whether
tool call arguments were influenced by untrusted content and blocks sensitive
actions when they are.

THREAT MODEL:
    1. Attacker controls content returned by an MCP tool (e.g., a web scraper
       returns HTML with an embedded URL).
    2. The model incorporates that URL into a subsequent tool call argument
       (e.g., calls a "send_email" tool with the attacker's URL in the body).
    3. Without taint tracking, the bridge would execute this — the model is
       just following instructions it found in "data."
    4. With taint tracking, the bridge detects that the URL originated from
       an untrusted tool result and blocks the outbound action.

COMPONENTS:
    TaintTracker: Stores values extracted from tool results. Computes taint
        state by matching tool call arguments against stored values.
    SinkPolicyEngine: Given taint state + tool classification + config,
        produces a SinkDecision (ALLOW / ALLOW_WITH_NOTICE / REQUIRE_CONFIRMATION / BLOCK).
"""

from __future__ import annotations

import ipaddress
import logging
import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any
from urllib.parse import urlparse

from .config import SecurityConfig
from .types import (
    ActionClass,
    ApprovedTool,
    ContentProvenance,
    DestinationPolicy,
    InfluenceEvidence,
    InfluenceState,
    InfluenceType,
    SemanticRiskAssessment,
    SinkDecision,
    TaintState,
    TrustLevel,
)

logger = logging.getLogger(__name__)


# --- Value extraction patterns ---

_URL_PATTERN = re.compile(r"https?://[^\s\"'<>]+", re.IGNORECASE)
_EMAIL_PATTERN = re.compile(r"[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}")
_IP_PATTERN = re.compile(r"\b\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\b")
# Bare hostname: requires at least one dot and a 2+ char TLD-like suffix.
# Anchored with \b to avoid matching inside URLs (which are caught by _URL_PATTERN first).
_HOSTNAME_PATTERN = re.compile(
    r"\b([a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?"
    r"(?:\.[a-zA-Z0-9](?:[a-zA-Z0-9-]{0,61}[a-zA-Z0-9])?)*"
    r"\.[a-zA-Z]{2,})\b"
)
# host:port — matches IP:port or hostname:port patterns.
_HOST_PORT_PATTERN = re.compile(r"\b([a-zA-Z0-9][a-zA-Z0-9._-]*:\d{1,5})\b")
# Arg field names that indicate outbound destination intent.
_DESTINATION_FIELD_NAMES = frozenset({
    "host", "hostname", "server", "endpoint", "uri", "url",
    "base_url", "webhook_url", "webhook", "callback_url",
    "destination", "target", "target_url", "recipient",
    "address", "remote", "remote_host",
})

# Memory-write tool name patterns — uses (?:^|_) and (?:_|$) instead of \b
# because tool names use underscores (store_memory), and \b treats _ as a word char.
# Excludes generic terms like "add" (could be math) — only storage-specific verbs.
_MEMORY_WRITE_PATTERNS = re.compile(
    r"(?:^|_)(store|write|save|create|insert|put|set|remember|memorize|persist)(?:_|$)",
    re.IGNORECASE,
)


class SinkType(str, Enum):
    """Classification of what kind of sink a tool call represents."""

    OUTBOUND = "outbound"
    DESTRUCTIVE = "destructive"
    MEMORY_WRITE = "memory_write"
    GENERAL_WRITE = "general_write"
    READ = "read"


@dataclass
class ExtractedValue:
    """A value extracted from text for taint tracking."""

    value: str
    kind: str  # "url", "domain", "email", "ip"


@dataclass
class TrackedResult:
    """A stored tool result with extracted values for future taint matching."""

    origin_id: str  # "server:tool_name"
    values: list[ExtractedValue] = field(default_factory=list)
    risk_score: float = 0.0


class TaintTracker:
    """Tracks tool result values and detects taint propagation to tool call args.

    Records values (URLs, domains, emails, IPs) from each tool result.
    When a new tool call comes in, checks if its arguments contain any of
    those values — indicating the model is passing untrusted content to a tool.

    Capped at max_results to prevent unbounded growth in long sessions.
    Oldest results are dropped first (FIFO).
    """

    def __init__(self, max_results: int = 50) -> None:
        self._results: list[TrackedResult] = []
        self._max_results = max_results

    def record_result(
        self,
        content: str,
        origin_id: str,
        provenance: ContentProvenance | None = None,
        risk_assessment: SemanticRiskAssessment | None = None,
    ) -> None:
        """Extract and store values from a tool result for future taint matching.

        Only tracks results from untrusted sources (THIRD_PARTY, UNKNOWN).
        TRUSTED sources (system, developer_policy) are excluded.
        """
        if provenance and provenance.trust_level == TrustLevel.TRUSTED:
            return

        values = _extract_values(content)
        if not values:
            return

        risk_score = risk_assessment.overall_risk_score if risk_assessment else 0.0
        self._results.append(TrackedResult(
            origin_id=origin_id,
            values=values,
            risk_score=risk_score,
        ))
        # Drop oldest if over capacity
        if len(self._results) > self._max_results:
            self._results = self._results[-self._max_results:]

    def compute_taint(self, args: dict[str, Any]) -> InfluenceState:
        """Check if tool call arguments contain values from previous tool results.

        Walks the argument dict recursively, extracts values, and matches
        against all stored result values. Returns InfluenceState (IS-A TaintState)
        with structured evidence of direct and derived influence.
        """
        if not self._results:
            return InfluenceState()

        arg_entries = _extract_values_from_args(args)
        if not arg_entries:
            return InfluenceState()

        taint_sources: list[str] = []
        taint_reasons: list[str] = []
        affected_fields: list[str] = []
        evidence: list[InfluenceEvidence] = []
        max_confidence = 0.0

        for arg_field, arg_extracted in arg_entries:
            for tracked in self._results:
                for tracked_val in tracked.values:
                    # Try exact match first
                    confidence = _match_confidence(arg_extracted, tracked_val)
                    influence_type: InfluenceType | None = (
                        InfluenceType.DIRECT_VALUE_MATCH if confidence > 0 else None
                    )

                    # Try derived match if no exact match
                    if confidence == 0:
                        confidence, influence_type = _derived_match_confidence(
                            arg_extracted, tracked_val,
                        )

                    if confidence > 0 and influence_type is not None:
                        taint_sources.append(tracked.origin_id)
                        taint_reasons.append(
                            f"{arg_extracted.kind}:{arg_extracted.value[:80]} "
                            f"from {tracked.origin_id}"
                        )
                        affected_fields.append(arg_field)
                        evidence.append(InfluenceEvidence(
                            influence_type=influence_type,
                            tracked_value=tracked_val.value[:80],
                            arg_value=arg_extracted.value[:80],
                            origin_id=tracked.origin_id,
                            confidence=confidence,
                        ))
                        # Amplify confidence if source was risky
                        effective = min(confidence + tracked.risk_score * 0.1, 1.0)
                        max_confidence = max(max_confidence, effective)

        if not evidence:
            return InfluenceState()

        has_direct = any(
            e.influence_type == InfluenceType.DIRECT_VALUE_MATCH for e in evidence
        )
        has_derived = any(
            e.influence_type != InfluenceType.DIRECT_VALUE_MATCH for e in evidence
        )
        destination_types = {
            InfluenceType.DERIVED_URL_REUSE,
            InfluenceType.DERIVED_PROTOCOL_CHANGE,
            InfluenceType.DERIVED_HOSTNAME_IN_URL,
            InfluenceType.DERIVED_EMAIL_DOMAIN,
        }
        has_destination = any(
            e.influence_type in destination_types for e in evidence
        )

        return InfluenceState(
            tainted=True,
            taint_sources=sorted(set(taint_sources)),
            taint_reasons=taint_reasons,
            affected_fields=sorted(set(affected_fields)),
            confidence=round(max_confidence, 2),
            direct_value_match=has_direct,
            derived_from_untrusted_value=has_derived,
            destination_influenced=has_destination,
            evidence=evidence,
        )

    def clear(self) -> None:
        """Clear all tracked results (e.g., on session reset)."""
        self._results.clear()


class SinkPolicyEngine:
    """Evaluates sink policy based on taint state and tool classification.

    Default policies (all configurable):
        tainted + outbound  → BLOCK
        tainted + destructive → REQUIRE_CONFIRMATION
        tainted + memory_write → BLOCK
        tainted + general_write → ALLOW_WITH_NOTICE
        tainted + read → ALLOW
        not tainted → ALLOW
    """

    def evaluate(
        self,
        tool: ApprovedTool,
        args: dict[str, Any],
        taint_state: TaintState,
        config: SecurityConfig,
        destination_policies: list[DestinationPolicy] | None = None,
    ) -> SinkDecision:
        """Evaluate sink policy for a tool call.

        Returns a SinkDecision that the SecurityGateway acts on.
        """
        if not taint_state.tainted:
            return SinkDecision.ALLOW

        sink_type = self._classify_sink(tool, args)

        if sink_type == SinkType.READ:
            return SinkDecision.ALLOW

        if sink_type == SinkType.OUTBOUND:
            # Check destination policies first (PR 11), fall back to domain list
            if destination_policies:
                if self._all_destinations_allowed(args, destination_policies):
                    return SinkDecision.ALLOW_WITH_NOTICE
            elif self._all_domains_allowed(args, config.allowed_outbound_domains):
                return SinkDecision.ALLOW_WITH_NOTICE

            # No policy match — check require flag
            if config.require_destination_policy_for_outbound and not destination_policies:
                return SinkDecision.BLOCK

            if config.block_tainted_exfiltration:
                return SinkDecision.BLOCK
            if config.tainted_sink_requires_confirmation:
                return SinkDecision.REQUIRE_CONFIRMATION
            return SinkDecision.ALLOW_WITH_NOTICE

        if sink_type == SinkType.MEMORY_WRITE:
            if not config.allow_memory_writes_from_third_party_content:
                return SinkDecision.BLOCK
            return SinkDecision.ALLOW_WITH_NOTICE

        if sink_type == SinkType.DESTRUCTIVE:
            if config.block_tainted_destructive_write:
                if config.tainted_sink_requires_confirmation:
                    return SinkDecision.REQUIRE_CONFIRMATION
                return SinkDecision.BLOCK
            return SinkDecision.ALLOW_WITH_NOTICE

        # GENERAL_WRITE
        return SinkDecision.ALLOW_WITH_NOTICE

    def _classify_sink(self, tool: ApprovedTool, args: dict[str, Any]) -> SinkType:
        """Determine what kind of sink this tool call represents.

        Priority order:
        1. Argument-based outbound detection (ALWAYS runs — catches exfiltration
           through non-outbound tools, e.g., a delete_file tool with a URL arg)
        2. Capability manifest (typed, explicit)
        3. ActionClass classification (config-driven)
        4. Tool name patterns (last resort)
        """
        # 1. Argument-based outbound detection — always runs regardless of manifest.
        # A tool with URLs/emails in args is suspicious even if its manifest says
        # filesystem_delete. The model may be trying to exfiltrate through a
        # non-outbound tool.
        if _args_contain_outbound_indicators(args):
            return SinkType.OUTBOUND

        # 2. Capability manifest — typed and explicit
        caps = tool.capabilities

        if caps.has_outbound_capability:
            return SinkType.OUTBOUND

        if caps.memory_write:
            return SinkType.MEMORY_WRITE

        if caps.destructive or caps.filesystem_delete:
            return SinkType.DESTRUCTIVE

        # 3. ActionClass — config-driven classification
        if tool.classification == ActionClass.READ:
            return SinkType.READ

        # 4. Tool name patterns — last resort fallback
        if _is_memory_write_tool(tool.name):
            return SinkType.MEMORY_WRITE

        if tool.classification == ActionClass.DESTRUCTIVE:
            return SinkType.DESTRUCTIVE

        return SinkType.GENERAL_WRITE

    def _all_domains_allowed(
        self, args: dict[str, Any], allowed_domains: list[str],
    ) -> bool:
        """Check if all domains in args are in the allowed list."""
        if not allowed_domains:
            return False

        domains = _extract_domains_from_args(args)
        if not domains:
            return False

        return all(
            any(domain == allowed or domain.endswith(f".{allowed}")
                for allowed in allowed_domains)
            for domain in domains
        )

    def _all_destinations_allowed(
        self, args: dict[str, Any], policies: list[DestinationPolicy],
    ) -> bool:
        """Check if all URLs in args match at least one destination policy."""
        urls = _extract_urls_from_args(args)
        if not urls:
            return False  # no URLs to validate = can't confirm allowed

        return all(
            any(policy.matches(url).matched for policy in policies)
            for url in urls
        )


# --- Module-level helpers ---


def _extract_values(text: str) -> list[ExtractedValue]:
    """Extract trackable values (URLs, domains, emails, IPs) from text."""
    values: list[ExtractedValue] = []
    seen: set[str] = set()

    for match in _URL_PATTERN.finditer(text):
        url = match.group().rstrip(".,;:)]}\"'")
        if url not in seen:
            seen.add(url)
            values.append(ExtractedValue(value=url, kind="url"))
            # Also extract domain
            try:
                domain = urlparse(url).hostname
                if domain and domain not in seen:
                    seen.add(domain)
                    values.append(ExtractedValue(value=domain, kind="domain"))
            except Exception:
                pass

    for match in _EMAIL_PATTERN.finditer(text):
        email = match.group()
        if email not in seen:
            seen.add(email)
            values.append(ExtractedValue(value=email, kind="email"))

    for match in _IP_PATTERN.finditer(text):
        ip = match.group()
        # Validate as a real IP address (rejects 999.999.999.999, semver-like strings)
        try:
            addr = ipaddress.ip_address(ip)
        except ValueError:
            continue
        # Skip loopback and unspecified
        if addr.is_loopback or addr.is_unspecified:
            continue
        if ip not in seen:
            seen.add(ip)
            values.append(ExtractedValue(value=ip, kind="ip"))

    # Host:port patterns (e.g., "10.0.0.1:8080", "api.example.com:443").
    # Checked before bare hostnames to avoid partial matches.
    for match in _HOST_PORT_PATTERN.finditer(text):
        hp = match.group(1)
        if hp not in seen:
            seen.add(hp)
            values.append(ExtractedValue(value=hp, kind="host_port"))

    # Bare hostnames (e.g., "api.example.com" without scheme).
    # Runs after URL/domain/IP extraction so already-seen values are skipped.
    for match in _HOSTNAME_PATTERN.finditer(text):
        hostname = match.group(1)
        if hostname not in seen:
            seen.add(hostname)
            values.append(ExtractedValue(value=hostname, kind="hostname"))

    return values


def _extract_values_from_args(
    args: dict[str, Any], prefix: str = "",
) -> list[tuple[str, ExtractedValue]]:
    """Recursively walk args dict and extract trackable values from strings."""
    results: list[tuple[str, ExtractedValue]] = []

    for key, value in args.items():
        field_name = f"{prefix}.{key}" if prefix else key

        if isinstance(value, str):
            for ev in _extract_values(value):
                results.append((field_name, ev))
        elif isinstance(value, dict):
            results.extend(_extract_values_from_args(value, prefix=field_name))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                item_prefix = f"{field_name}[{i}]"
                if isinstance(item, str):
                    for ev in _extract_values(item):
                        results.append((item_prefix, ev))
                elif isinstance(item, dict):
                    results.extend(_extract_values_from_args(item, prefix=item_prefix))

    return results


def _match_confidence(arg_val: ExtractedValue, tracked_val: ExtractedValue) -> float:
    """Compute match confidence between an arg value and a tracked result value.

    Returns 0.0 for no match, up to 0.9 for exact URL match.
    """
    # Exact match on same kind
    if arg_val.kind == tracked_val.kind and arg_val.value == tracked_val.value:
        if arg_val.kind == "url":
            return 0.9
        if arg_val.kind == "email":
            return 0.9
        if arg_val.kind == "ip":
            return 0.85
        return 0.8  # domain exact match

    # URL arg matches tracked domain (the URL contains the domain)
    if arg_val.kind == "url" and tracked_val.kind == "domain":
        try:
            arg_domain = urlparse(arg_val.value).hostname
            if arg_domain and (
                arg_domain == tracked_val.value
                or arg_domain.endswith(f".{tracked_val.value}")
            ):
                return 0.7
        except Exception:
            pass

    # Domain arg matches tracked URL's domain
    if arg_val.kind == "domain" and tracked_val.kind == "url":
        try:
            tracked_domain = urlparse(tracked_val.value).hostname
            if tracked_domain and (
                arg_val.value == tracked_domain
                or arg_val.value.endswith(f".{tracked_domain}")
            ):
                return 0.7
        except Exception:
            pass

    return 0.0


def _derived_match_confidence(
    arg_val: ExtractedValue, tracked_val: ExtractedValue,
) -> tuple[float, InfluenceType | None]:
    """Check for derived (non-exact) taint relationships.

    Called when _match_confidence returns 0.0. Detects transformed values
    that are still traceable to untrusted origins: URL path changes, protocol
    swaps, email domain reuse, IP/hostname in host:port patterns.
    """
    # Same host, different path or protocol (URL → URL)
    if arg_val.kind == "url" and tracked_val.kind == "url":
        try:
            arg_p = urlparse(arg_val.value)
            tracked_p = urlparse(tracked_val.value)
            if arg_p.hostname and tracked_p.hostname and (
                arg_p.hostname.lower() == tracked_p.hostname.lower()
            ):
                if arg_p.scheme.lower() != tracked_p.scheme.lower():
                    return 0.65, InfluenceType.DERIVED_PROTOCOL_CHANGE
                if arg_p.path != tracked_p.path:
                    return 0.6, InfluenceType.DERIVED_URL_REUSE
        except Exception:
            pass

    # Email domain reuse (different local part, same domain)
    if arg_val.kind == "email" and tracked_val.kind == "email":
        arg_parts = arg_val.value.split("@", 1)
        tracked_parts = tracked_val.value.split("@", 1)
        if len(arg_parts) == 2 and len(tracked_parts) == 2:
            if arg_parts[1].lower() == tracked_parts[1].lower():
                return 0.5, InfluenceType.DERIVED_EMAIL_DOMAIN

    # Tracked IP appears in host:port arg
    if arg_val.kind == "host_port" and tracked_val.kind == "ip":
        host_part = arg_val.value.rsplit(":", 1)[0]
        if host_part == tracked_val.value:
            return 0.8, InfluenceType.DERIVED_HOSTNAME_IN_URL

    # Tracked hostname/domain appears in host:port arg
    if arg_val.kind == "host_port" and tracked_val.kind in ("domain", "hostname"):
        host_part = arg_val.value.rsplit(":", 1)[0]
        if (host_part.lower() == tracked_val.value.lower()
                or host_part.lower().endswith(f".{tracked_val.value.lower()}")):
            return 0.7, InfluenceType.DERIVED_HOSTNAME_IN_URL

    # Hostname cross-kind matching (PR 12 kinds)
    if arg_val.kind == "hostname" and tracked_val.kind in ("hostname", "domain"):
        if arg_val.value.lower() == tracked_val.value.lower():
            return 0.75, InfluenceType.DIRECT_VALUE_MATCH

    if arg_val.kind == "domain" and tracked_val.kind == "hostname":
        if arg_val.value.lower() == tracked_val.value.lower():
            return 0.75, InfluenceType.DIRECT_VALUE_MATCH

    return 0.0, None


def _args_contain_outbound_indicators(args: dict[str, Any]) -> bool:
    """Check if tool arguments contain outbound indicators.

    Detects:
    - URLs, emails (original)
    - IPs, bare hostnames, host:port patterns (PR 12)
    - Destination-indicating field names with non-empty string values (PR 12)
    """
    entries = _extract_values_from_args(args)
    if any(ev.kind in ("url", "email", "ip", "hostname", "host_port") for _, ev in entries):
        return True
    return _args_contain_destination_fields(args)


def _args_contain_destination_fields(args: dict[str, Any]) -> bool:
    """Check if arg field names indicate outbound destination intent.

    Catches patterns like send_data(host="evil.com", port=443) where
    the values alone aren't URLs but the field names reveal outbound intent.
    Only matches when the value is a non-empty string (int port alone
    is not sufficient).
    """
    for key, value in args.items():
        if key.lower() in _DESTINATION_FIELD_NAMES and isinstance(value, str) and value:
            return True
    return False


def _extract_domains_from_args(args: dict[str, Any]) -> list[str]:
    """Extract all unique domains from URLs in tool arguments."""
    domains: list[str] = []
    entries = _extract_values_from_args(args)
    for _, ev in entries:
        if ev.kind == "domain":
            domains.append(ev.value)
        elif ev.kind == "url":
            try:
                host = urlparse(ev.value).hostname
                if host:
                    domains.append(host)
            except Exception:
                pass
    return list(set(domains))


def _extract_urls_from_args(args: dict[str, Any]) -> list[str]:
    """Extract all URL strings from tool arguments."""
    entries = _extract_values_from_args(args)
    return [ev.value for _, ev in entries if ev.kind == "url"]


def _is_memory_write_tool(tool_name: str) -> bool:
    """Check if a tool name suggests a memory/storage write operation."""
    return bool(_MEMORY_WRITE_PATTERNS.search(tool_name))
