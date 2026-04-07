"""Conservative capability inference engine for tool capability manifests.

When an operator doesn't configure explicit capabilities in bridge.toml, the bridge
needs a fallback to infer what a tool can do from its name, description, and input
schema. This module provides that fallback.

DESIGN PRINCIPLE: Conservative inference. False positives (flagging a safe tool as
dangerous) are strictly better than false negatives (letting a dangerous tool through
unflagged). When uncertain, set the flag True.

This replaces the name-pattern approach in sink_policy._is_memory_write_tool() with
a structured manifest that covers 12 capability dimensions. The sink policy engine
can then use the manifest instead of ad-hoc pattern matching.

All patterns are compiled at module level for performance. The inference function is
pure — no state, no side effects, no I/O.
"""

from __future__ import annotations

import re
from typing import Any

from .types import CapabilitySource, ToolCapabilityManifest, ToolSchema


# --- Compiled pattern sets (module-level for performance) ---
# Word boundary strategy: (?:^|(?<=_)|\b) for leading edge, (?=_|$|\b) for trailing.
# This handles both underscore-separated tool names (store_memory) and space-separated
# description words. Note: \b inside [] is backspace, NOT a word boundary — never use [\b_].

# Shorthand for the boundary anchors used by every pattern.
_L = r"(?:^|(?<=_)|\b)"  # left/leading boundary
_R = r"(?=_|$|\b)"  # right/trailing boundary

_NETWORK_PATTERNS = re.compile(
    _L + r"(send|post|fetch|request|https?|api|webhook|upload|download|push"
    r"|notify|publish|forward|relay|curl|wget)" + _R,
    re.IGNORECASE,
)

_NETWORK_SCHEMA_FIELDS = re.compile(
    r"^(url|endpoint|host|uri|base_url|webhook|webhook_url|api_url|api_endpoint"
    r"|remote_url|server_url|destination_url|callback_url|target_url)$",
    re.IGNORECASE,
)

_FS_READ_PATTERNS = re.compile(
    _L + r"(read|open|load|cat|head|tail|list_files|list_dir|ls|find"
    r"|glob|stat|get_file|read_file|read_dir|list_directory|get_content"
    r"|read_content|file_info|dir_list)" + _R,
    re.IGNORECASE,
)

_FS_WRITE_PATTERNS = re.compile(
    _L + r"(write|save|create_file|write_file|append|overwrite|touch"
    r"|mkdir|copy|move|rename|write_content|save_file|create_dir"
    r"|make_dir|put_file)" + _R,
    re.IGNORECASE,
)

_FS_DELETE_PATTERNS = re.compile(
    _L + r"(delete|remove|rm|rmdir|unlink|truncate|wipe|purge"
    r"|delete_file|remove_file|delete_dir|remove_dir|clean)" + _R,
    re.IGNORECASE,
)

# Memory-write: storage-specific verbs scoped to memory/knowledge/note contexts.
# Two-pass: first check for storage verbs, then check for memory context.
_MEMORY_WRITE_VERBS = re.compile(
    r"(?:^|(?<=_))(store|write|save|create|insert|put|set|remember|memorize|persist"
    r"|update|upsert|add|log|record)(?=_|$)",
    re.IGNORECASE,
)

_MEMORY_CONTEXT = re.compile(
    _L + r"(memory|memories|knowledge|note|notes|memo|memos|remember"
    r"|memorize|brain|context|history|journal|diary|recall|store_memory"
    r"|knowledge_base|kb)" + _R,
    re.IGNORECASE,
)

_EXTERNAL_MESSAGING_PATTERNS = re.compile(
    _L + r"(send|send_email|send_message|post_comment|notify|slack|email"
    r"|sms|chat|message|tweet|dm|direct_message|post_message"
    r"|send_notification|send_sms|send_slack|post_tweet|compose)" + _R,
    re.IGNORECASE,
)

_CODE_EXECUTION_PATTERNS = re.compile(
    _L + r"(exec|execute|eval|run_code|shell|bash|script|compile"
    r"|interpret|subprocess|spawn|system|command|run_script|run_command"
    r"|execute_code|eval_code|run_shell|invoke)" + _R,
    re.IGNORECASE,
)

_CREDENTIAL_PATTERNS = re.compile(
    _L + r"(password|secret|token|api_key|credential|auth|certificate"
    r"|private_key|access_key|secret_key|passphrase|oauth|jwt"
    r"|bearer|login|signin|sign_in)" + _R,
    re.IGNORECASE,
)

_CREDENTIAL_SCHEMA_FIELDS = re.compile(
    r"^(password|secret|token|api_key|credential|auth_token|access_token"
    r"|private_key|secret_key|passphrase|bearer_token|jwt|oauth_token"
    r"|credentials|auth|key|client_secret|client_id)$",
    re.IGNORECASE,
)

_USER_IDENTITY_PATTERNS = re.compile(
    _L + r"(create_user|delete_user|change_role|change_permission"
    r"|grant|revoke_access|impersonate|sudo|add_user|remove_user"
    r"|set_role|set_permission|assign_role|modify_user|update_role"
    r"|elevate|escalate|promote_user|demote_user)" + _R,
    re.IGNORECASE,
)

_DESTRUCTIVE_PATTERNS = re.compile(
    _L + r"(drop|destroy|format|reset|factory_reset|wipe|nuke"
    r"|truncate_table|delete_all|clear_all|force|drop_table|drop_database"
    r"|force_delete|hard_reset|erase|obliterate)" + _R,
    re.IGNORECASE,
)

# High-value targets: if a delete/remove operation targets these, escalate to destructive.
_DESTRUCTIVE_TARGET_PATTERNS = re.compile(
    _L + r"(database|db|table|schema|collection|index|volume|cluster"
    r"|namespace|partition|bucket|queue|stack|registry|vault)" + _R,
    re.IGNORECASE,
)

_HIGH_CONSEQUENCE_PATTERNS = re.compile(
    _L + r"(deploy|publish_production|release|migrate_database"
    r"|alter_schema|drop|destroy|format|reset|factory_reset|wipe|nuke"
    r"|truncate_table|delete_all|clear_all|force|production|migrate"
    r"|rollback|schema_change|push_live|go_live)" + _R,
    re.IGNORECASE,
)


def infer_capabilities(tool: ToolSchema) -> ToolCapabilityManifest:
    """Infer a tool's capability manifest from its name, description, and input schema.

    This is the lowest-trust fallback — used only when the operator hasn't configured
    explicit capabilities in bridge.toml. The inference is deliberately conservative:
    flags are set True when there's any evidence the tool might have that capability.

    Args:
        tool: Raw tool schema as received from an MCP server. The name, description,
              and input_schema fields are all analyzed.

    Returns:
        A frozen ToolCapabilityManifest with source=INFERRED and boolean flags set
        based on heuristic pattern matching. All flags default to False and are only
        set True when evidence is found.

    The function is pure: no state, no side effects, deterministic output for any
    given input.
    """
    # Build the text corpus to scan: tool name + description
    name_lower = tool.name.lower()
    desc_lower = tool.description.lower() if tool.description else ""
    text = f"{name_lower} {desc_lower}"

    # Extract input_schema property names for schema-field checks
    schema_fields = _extract_schema_fields(tool.input_schema)

    # --- Evaluate each capability dimension ---

    network = (
        bool(_NETWORK_PATTERNS.search(text))
        or _any_field_matches(schema_fields, _NETWORK_SCHEMA_FIELDS)
    )

    outbound = network  # conservative: any network access implies potential outbound

    fs_read = bool(_FS_READ_PATTERNS.search(text))

    fs_write = bool(_FS_WRITE_PATTERNS.search(text))

    fs_delete = bool(_FS_DELETE_PATTERNS.search(text))

    memory = _check_memory_write(name_lower, desc_lower, text)

    messaging = bool(_EXTERNAL_MESSAGING_PATTERNS.search(text))

    code_exec = bool(_CODE_EXECUTION_PATTERNS.search(text))

    credential = (
        bool(_CREDENTIAL_PATTERNS.search(text))
        or _any_field_matches(schema_fields, _CREDENTIAL_SCHEMA_FIELDS)
    )

    identity = bool(_USER_IDENTITY_PATTERNS.search(text))

    destructive = bool(_DESTRUCTIVE_PATTERNS.search(text))

    # Conservative escalation: delete/remove targeting a high-value resource is destructive
    if fs_delete and bool(_DESTRUCTIVE_TARGET_PATTERNS.search(text)):
        destructive = True

    high_consequence = bool(_HIGH_CONSEQUENCE_PATTERNS.search(text)) or destructive

    # Conservative cross-flagging: messaging implies network + outbound
    if messaging:
        network = True
        outbound = True

    # Conservative cross-flagging: code execution could do anything
    if code_exec:
        network = True
        fs_read = True
        fs_write = True

    return ToolCapabilityManifest(
        network_access=network,
        outbound_data_transfer=outbound,
        filesystem_read=fs_read,
        filesystem_write=fs_write,
        filesystem_delete=fs_delete,
        memory_write=memory,
        external_messaging=messaging,
        code_execution=code_exec,
        credential_access=credential,
        user_identity_impact=identity,
        destructive=destructive,
        high_consequence=high_consequence,
        source=CapabilitySource.INFERRED,
    )


def _extract_schema_fields(input_schema: dict[str, Any]) -> list[str]:
    """Extract property names from a JSON Schema input_schema.

    Walks one level into the 'properties' dict. Returns lowercase field names.
    Handles missing or malformed schemas gracefully.
    """
    if not input_schema:
        return []

    properties = input_schema.get("properties", {})
    if not isinstance(properties, dict):
        return []

    return [k.lower() for k in properties]


def _any_field_matches(fields: list[str], pattern: re.Pattern[str]) -> bool:
    """Check if any schema field name matches a compiled pattern."""
    return any(pattern.match(field) for field in fields)


def _check_memory_write(name_lower: str, desc_lower: str, text: str) -> bool:
    """Check for memory-write capability using two-pass heuristic.

    A tool is flagged as memory_write if:
    1. Its name contains a storage verb AND a memory-context word, OR
    2. Its description mentions memory/knowledge/note context AND a storage verb.

    This avoids false positives on tools like "create_file" (storage verb but
    filesystem context, not memory context).
    """
    has_verb_in_name = bool(_MEMORY_WRITE_VERBS.search(name_lower))
    has_context_in_name = bool(_MEMORY_CONTEXT.search(name_lower))

    # Strong signal: both verb and context in the tool name
    if has_verb_in_name and has_context_in_name:
        return True

    # Weaker signal: verb in name + context in description
    if has_verb_in_name and bool(_MEMORY_CONTEXT.search(desc_lower)):
        return True

    # Direct name match for common memory tools
    memory_tool_names = {
        "store_memory", "save_memory", "write_memory", "create_memory",
        "remember", "memorize", "persist_memory", "log_memory",
        "store_note", "save_note", "create_note", "add_note",
        "update_knowledge", "store_knowledge", "save_knowledge",
    }
    if name_lower in memory_tool_names:
        return True

    return False
