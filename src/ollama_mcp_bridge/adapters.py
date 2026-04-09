"""Safe argument adapters for capability narrowing.

These adapters validate tool call arguments PROACTIVELY — regardless of
whether taint was detected. While the sink policy (PR 7) is reactive (blocks
tainted args), adapters catch structurally unsafe args even when generated
from scratch by the model.

DESIGN:
    - Opt-in: each adapter activates only when its config field is set.
      Empty config = adapter inactive = no restriction.
    - Adapters run AFTER sink policy, BEFORE MCP call.
    - Errors are returned as strings. The orchestrator raises
      ParameterRejectedError so the model gets the error + schema hint
      and can self-correct.
    - Adapters reuse value extraction from sink_policy.py and risk
      assessment from security.py.

ADAPTERS:
    SafeURL: URLs in args must match allowed_outbound_domains.
    SafePath: File paths must stay within allowed_path_roots; no traversal.
    SafeRecipient: Email addresses must be in approved_recipients.
    SafeMemoryWriteCandidate: Memory-write tool args must not contain
        instruction-like patterns (assessed via SemanticRiskAssessor).
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any
from urllib.parse import urlparse

from .config import SecurityConfig
from .sink_policy import _extract_values_from_args, _is_memory_write_tool
from .types import ApprovedTool, DestinationPolicy, PathPolicy, RecipientPolicy, normalize_and_validate_ip

logger = logging.getLogger(__name__)


# --- Path extraction ---

# Matches Unix-style absolute paths, relative paths with ./ or ../, and ~ paths.
# Requires at least one slash to distinguish from plain words.
_PATH_PATTERN = re.compile(
    r"(?:^|\s)((?:/[^\s\"']+)|(?:\./[^\s\"']+)|(?:\.\./[^\s\"']+)|(?:~/[^\s\"']+))",
)


def _extract_paths_from_args(
    args: dict[str, Any], prefix: str = "",
) -> list[tuple[str, str]]:
    """Recursively extract file path strings from arguments.

    Returns list of (field_name, path_value) tuples.
    """
    results: list[tuple[str, str]] = []

    for key, value in args.items():
        field_name = f"{prefix}.{key}" if prefix else key

        if isinstance(value, str):
            for match in _PATH_PATTERN.finditer(value):
                results.append((field_name, match.group(1)))
            # Also check if the entire value looks like a path
            if value.startswith(("/", "./", "../", "~/")):
                results.append((field_name, value))
        elif isinstance(value, dict):
            results.extend(_extract_paths_from_args(value, prefix=field_name))
        elif isinstance(value, list):
            for i, item in enumerate(value):
                item_prefix = f"{field_name}[{i}]"
                if isinstance(item, str):
                    if item.startswith(("/", "./", "../", "~/")):
                        results.append((item_prefix, item))
                elif isinstance(item, dict):
                    results.extend(
                        _extract_paths_from_args(item, prefix=item_prefix)
                    )

    # Deduplicate while preserving order
    seen: set[tuple[str, str]] = set()
    deduped: list[tuple[str, str]] = []
    for entry in results:
        if entry not in seen:
            seen.add(entry)
            deduped.append(entry)
    return deduped


# --- Adapters ---


class SafeURL:
    """Validates URLs in tool arguments against destination policies or domain list.

    Active when: destination_policies are provided, OR allowed_outbound_domains
    is non-empty, OR require_destination_policy_for_outbound is True.

    When destination policies exist, validates each URL against the full policy
    (scheme, host, port, path, IP controls). Falls back to domain-only checking
    when only allowed_outbound_domains is configured.
    """

    name = "safe_url"

    def check(
        self,
        tool: ApprovedTool,
        args: dict[str, Any],
        config: SecurityConfig,
        destination_policies: list[DestinationPolicy] | None = None,
    ) -> list[str]:
        # Destination policies take precedence (PR 11 path)
        if destination_policies:
            errors = self._check_destination_policies(args, destination_policies)
            # Also enforce policy against raw host/IP/host:port args (non-URL fields).
            # These bypass _check_destination_policies which only scans kind="url" entries.
            errors.extend(self._check_raw_host_args(args, destination_policies))
            return errors

        # Require flag with no policies — block outbound tools with URLs
        if config.require_destination_policy_for_outbound:
            entries = _extract_values_from_args(args)
            urls = [ev for _, ev in entries if ev.kind == "url"]
            if urls and tool.capabilities.has_outbound_capability:
                return [
                    f"[{self.name}] Tool '{tool.name}' has outbound capability "
                    f"but no destination policy configured "
                    f"(require_destination_policy_for_outbound=True)"
                ]

        # Legacy path: use allowed_outbound_domains
        if not config.allowed_outbound_domains:
            return []

        errors: list[str] = []
        entries = _extract_values_from_args(args)

        for field_name, ev in entries:
            if ev.kind != "url":
                continue

            try:
                host = urlparse(ev.value).hostname
            except Exception:
                errors.append(
                    f"[{self.name}] Field '{field_name}': malformed URL '{ev.value[:80]}'"
                )
                continue

            if not host:
                errors.append(
                    f"[{self.name}] Field '{field_name}': URL has no hostname"
                )
                continue

            if not any(
                host == domain or host.endswith(f".{domain}")
                for domain in config.allowed_outbound_domains
            ):
                errors.append(
                    f"[{self.name}] Field '{field_name}': domain '{host}' "
                    f"not in allowed_outbound_domains "
                    f"({', '.join(config.allowed_outbound_domains)})"
                )

        return errors

    def _check_destination_policies(
        self,
        args: dict[str, Any],
        policies: list[DestinationPolicy],
    ) -> list[str]:
        """Validate all URLs in args against destination policies."""
        errors: list[str] = []
        entries = _extract_values_from_args(args)

        for field_name, ev in entries:
            if ev.kind != "url":
                continue

            match_results = [p.matches(ev.value) for p in policies]
            if not any(r.matched for r in match_results):
                # Pick the most informative failure reason
                reasons = [r.failure_reason for r in match_results if r.failure_reason]
                reason_hint = f" ({reasons[0]})" if reasons else ""
                errors.append(
                    f"[{self.name}] Field '{field_name}': URL '{ev.value[:80]}' "
                    f"does not match any destination policy{reason_hint}"
                )

        return errors

    def _check_raw_host_args(
        self,
        args: dict[str, Any],
        policies: list[DestinationPolicy],
    ) -> list[str]:
        """Validate raw host/IP args against destination policies.

        Closes the non-URL bypass (BS[2]): tool args may contain destination hosts
        as plain "host" strings or "host:port" patterns that are not URLs and thus
        bypass _check_destination_policies which only scans kind="url" entries.

        Checks ExtractedValue entries with kind in {"ip", "hostname", "host_port"}.
        For each, synthesizes a minimal https:// URL and runs it against the policies.
        IP entries are also checked via normalize_and_validate_ip to catch encoded forms.

        Fields that already contain URLs are skipped — the URL check covers those.
        This prevents double-reporting when a URL string is parsed and both a url
        and a host_port entry are extracted from the same field value.
        """
        errors: list[str] = []
        entries = _extract_values_from_args(args)

        # Fields that already had URL entries — skip them in the raw host check
        # to avoid double-reporting (URL check covers host extraction from those).
        url_fields: set[str] = {fn for fn, ev in entries if ev.kind == "url"}

        for field_name, ev in entries:
            if ev.kind not in ("ip", "hostname", "host_port"):
                continue
            # Skip if this field already had a URL — url check covers it
            if field_name in url_fields:
                continue

            # Synthesize a URL to run through DestinationPolicy.matches()
            raw = ev.value
            if ev.kind == "host_port":
                # "10.0.0.1:8080" — use as-is after stripping port for IP check
                host_part = raw.rsplit(":", 1)[0] if ":" in raw else raw
                synthetic_url = f"https://{raw}/"
            else:
                host_part = raw
                synthetic_url = f"https://{raw}/"

            # For IP entries: apply normalize_and_validate_ip to catch encoded forms.
            # If normalized form is private/loopback and no policy allows it, block.
            if ev.kind == "ip":
                addr = normalize_and_validate_ip(host_part)
                if addr is not None and (addr.is_private or addr.is_loopback or addr.is_link_local):
                    # Check if any policy explicitly allows private ranges
                    if not any(p.allow_private_ranges or p.allow_ip_literals for p in policies):
                        errors.append(
                            f"[{self.name}] Field '{field_name}': raw IP arg "
                            f"'{raw[:80]}' resolves to private/loopback address "
                            f"({addr}) — no destination policy allows private ranges"
                        )
                        continue

            match_results = [p.matches(synthetic_url) for p in policies]
            if not any(r.matched for r in match_results):
                reasons = [r.failure_reason for r in match_results if r.failure_reason]
                reason_hint = f" ({reasons[0]})" if reasons else ""
                errors.append(
                    f"[{self.name}] Field '{field_name}': raw host arg "
                    f"'{raw[:80]}' does not match any destination policy{reason_hint}"
                )

        return errors


class SafePath:
    """Validates file paths in tool arguments against path policies.

    Active when: a PathPolicy is provided, OR allowed_path_roots is non-empty.

    When a PathPolicy is provided, validates each path against the full policy
    (allowed roots, relative path control, symlink resolution, extension
    filtering, read/write/delete constraints). Falls back to simple root-based
    checking when only allowed_path_roots is configured.
    """

    name = "safe_path"

    def check(
        self,
        tool: ApprovedTool,
        args: dict[str, Any],
        config: SecurityConfig,
        path_policy: PathPolicy | None = None,
    ) -> list[str]:
        # PathPolicy takes precedence (PR 14 path)
        if path_policy:
            return self._check_path_policy(tool, args, path_policy)

        # Legacy path: use allowed_path_roots
        if not config.allowed_path_roots:
            return []

        errors: list[str] = []
        path_entries = _extract_paths_from_args(args)

        # Normalize allowed roots
        normalized_roots = [
            os.path.normpath(os.path.expanduser(r))
            for r in config.allowed_path_roots
        ]

        for field_name, raw_path in path_entries:
            # Normalize the candidate path
            normalized = os.path.normpath(os.path.expanduser(raw_path))

            # Check if it falls under any allowed root
            if not any(
                normalized == root or normalized.startswith(root + os.sep)
                for root in normalized_roots
            ):
                errors.append(
                    f"[{self.name}] Field '{field_name}': path '{raw_path}' "
                    f"is outside allowed roots "
                    f"({', '.join(config.allowed_path_roots)})"
                )

        return errors

    def _check_path_policy(
        self,
        tool: ApprovedTool,
        args: dict[str, Any],
        policy: PathPolicy,
    ) -> list[str]:
        """Validate all paths in args against a PathPolicy."""
        errors: list[str] = []
        path_entries = _extract_paths_from_args(args)

        for field_name, raw_path in path_entries:
            result = policy.validate_path(raw_path, tool.capabilities)
            if not result.matched:
                errors.append(
                    f"[{self.name}] Field '{field_name}': {result.failure_reason}"
                )

        return errors


class SafeRecipient:
    """Validates email recipients in tool arguments against recipient policies.

    Active when: a RecipientPolicy is provided, OR approved_recipients is non-empty.

    When a RecipientPolicy is provided, validates each email against the full
    policy (exact addresses, domain-level approval, identity groups, internal-only
    mode). Falls back to flat list checking when only approved_recipients is configured.
    """

    name = "safe_recipient"

    def check(
        self,
        tool: ApprovedTool,
        args: dict[str, Any],
        config: SecurityConfig,
        recipient_policy: RecipientPolicy | None = None,
    ) -> list[str]:
        # RecipientPolicy takes precedence (PR 15 path)
        if recipient_policy:
            return self._check_recipient_policy(tool, args, recipient_policy)

        # Legacy path: use approved_recipients
        if not config.approved_recipients:
            return []

        errors: list[str] = []
        entries = _extract_values_from_args(args)

        # Normalize approved list to lowercase for comparison
        approved_lower = {r.lower() for r in config.approved_recipients}

        for field_name, ev in entries:
            if ev.kind != "email":
                continue

            if ev.value.lower() not in approved_lower:
                errors.append(
                    f"[{self.name}] Field '{field_name}': recipient "
                    f"'{ev.value}' not in approved_recipients"
                )

        return errors

    def _check_recipient_policy(
        self,
        tool: ApprovedTool,
        args: dict[str, Any],
        policy: RecipientPolicy,
    ) -> list[str]:
        """Validate all email recipients in args against a RecipientPolicy."""
        errors: list[str] = []
        entries = _extract_values_from_args(args)

        for field_name, ev in entries:
            if ev.kind != "email":
                continue

            # Internal-only mode: must match policy, no exceptions
            if policy.internal_only and not policy.has_any_policy:
                errors.append(
                    f"[{self.name}] Field '{field_name}': internal_only mode "
                    f"is active but no approved addresses/domains configured"
                )
                continue

            result = policy.validate_recipient(ev.value)
            if not result.matched:
                errors.append(
                    f"[{self.name}] Field '{field_name}': {result.failure_reason}"
                )

        return errors


class SafeMemoryWriteCandidate:
    """Validates content in memory-write tool arguments for instruction patterns.

    Active when: tool name matches memory-write patterns (always, for such tools).
    Checks: string arguments are assessed by SemanticRiskAssessor. If any
    argument has an overall_risk_score above the block threshold (normalized
    to 0-1 scale from config's 0-100 sanitization_block_threshold), the
    write is rejected.

    This catches the case where a model writes poisoned content to memory
    that a future session would read and follow as instructions.
    """

    name = "safe_memory_write"

    def __init__(self) -> None:
        # Lazy import to avoid circular dependency (security → adapters → security)
        from .security import SemanticRiskAssessor

        self._assessor = SemanticRiskAssessor()

    def check(
        self,
        tool: ApprovedTool,
        args: dict[str, Any],
        config: SecurityConfig,
    ) -> list[str]:
        # CapabilitySource-aware dispatch (Q6): prefer explicit manifest over name heuristic.
        # If the operator declared capabilities in config, trust that declaration.
        # Only fall back to name-pattern inference if capabilities were not explicitly set.
        is_memory_write = tool.capabilities.memory_write
        if not is_memory_write:
            # Capability not in manifest (INFERRED source or not set) — check name heuristic
            is_memory_write = _is_memory_write_tool(tool.name)
        if not is_memory_write:
            return []

        errors: list[str] = []
        # Normalize threshold from 0-100 config scale to 0.0-1.0
        threshold = config.sanitization_block_threshold / 100.0

        for key, value in args.items():
            if not isinstance(value, str) or len(value) < 20:
                continue

            assessment = self._assessor.assess(value)
            if assessment.overall_risk_score >= threshold:
                errors.append(
                    f"[{self.name}] Field '{key}': content contains "
                    f"instruction-like patterns "
                    f"(risk={assessment.overall_risk_score:.2f}, "
                    f"signals={assessment.raw_signals})"
                )

        return errors


# --- Orchestrator ---

# Lazy-initialized to avoid circular import at module load time.
_ADAPTERS: list[SafeURL | SafePath | SafeRecipient | SafeMemoryWriteCandidate] | None = None


def _get_adapters() -> list[SafeURL | SafePath | SafeRecipient | SafeMemoryWriteCandidate]:
    global _ADAPTERS
    if _ADAPTERS is None:
        _ADAPTERS = [SafeURL(), SafePath(), SafeRecipient(), SafeMemoryWriteCandidate()]
    return _ADAPTERS


def run_adapters(
    tool: ApprovedTool,
    args: dict[str, Any],
    config: SecurityConfig,
    destination_policies: list[DestinationPolicy] | None = None,
    path_policy: PathPolicy | None = None,
    recipient_policy: RecipientPolicy | None = None,
) -> list[str]:
    """Run all safe adapters against a tool call's arguments.

    Returns a list of error strings. Empty list means all adapters passed.
    Called by SecurityGateway.execute_tool() between sink policy and rate limiting.
    """
    errors: list[str] = []
    for adapter in _get_adapters():
        if isinstance(adapter, SafeURL):
            errors.extend(adapter.check(tool, args, config, destination_policies))
        elif isinstance(adapter, SafePath):
            errors.extend(adapter.check(tool, args, config, path_policy))
        elif isinstance(adapter, SafeRecipient):
            errors.extend(adapter.check(tool, args, config, recipient_policy))
        else:
            errors.extend(adapter.check(tool, args, config))
    return errors
