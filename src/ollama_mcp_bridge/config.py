"""Configuration loading and validation for ollama-mcp-bridge.

Configuration is the user's primary interface for expressing security policy.
The bridge uses a default-deny model: nothing is allowed unless explicitly
configured. Key security-relevant config points:

    allowed_tools: Per-server whitelist. If non-empty, ONLY listed tools can
        be called. Wildcard ("*") is explicitly rejected — forces users to
        enumerate tools they trust.
    destructive_tools: Per-server list of tools requiring human confirmation.
        These are classified as DESTRUCTIVE and gated behind the ActionGate.
    enabled_detectors: Which sanitization detectors run at tool ingestion time.
        All 7 enabled by default. Can be narrowed if false positives are an issue.

TOML format chosen over YAML (indentation footguns, security history with
deserialization attacks) and JSON (no comments, verbose). Python 3.11+ has
tomllib in stdlib, so no external dependency needed for parsing.

All config models use extra="forbid" to catch typos early — a misspelled
field name raises an error at load time rather than silently being ignored.
"""

from __future__ import annotations

import logging
import tomllib
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .errors import ConfigError
from .types import ActionClass

logger = logging.getLogger(__name__)


class ServerConfig(BaseModel):
    """Configuration for a single MCP server."""

    model_config = ConfigDict(extra="forbid")

    command: str
    args: list[str] = Field(default_factory=list)
    env: dict[str, str] | None = None
    allowed_tools: list[str] = Field(default_factory=list)
    read_tools: list[str] = Field(default_factory=list)
    destructive_tools: list[str] = Field(default_factory=list)
    max_calls_per_minute: int = 30
    max_result_bytes: int = 65536

    @field_validator("allowed_tools")
    @classmethod
    def reject_wildcard_allowlist(cls, v: list[str]) -> list[str]:
        if "*" in v:
            raise ValueError(
                "Wildcard '*' not allowed in allowed_tools. "
                "Explicitly list each tool for security (SR-4)."
            )
        return v

    @field_validator("command")
    @classmethod
    def command_must_be_nonempty(cls, v: str) -> str:
        if not v.strip():
            raise ValueError("Server command must not be empty")
        return v


class SecurityConfig(BaseModel):
    """Security configuration for the bridge."""

    model_config = ConfigDict(extra="forbid")

    require_confirmation_for_destructive: bool = True
    confirmation_timeout_seconds: float = 60.0
    max_result_bytes: int = 65536
    rate_limit_per_server: int = 30
    max_turns: int = 10
    max_turns_hard_cap: int = 50
    max_tool_calls_per_session: int = 100
    enabled_detectors: list[str] = Field(
        default_factory=lambda: [
            "instruction_language",
            "cross_tool_reference",
            "exfiltration_pattern",
            "privilege_escalation",
            "length_anomaly",
            "role_impersonation",
            "encoding_obfuscation",
        ]
    )
    sanitization_block_threshold: float = 70.0
    sanitization_warn_threshold: float = 40.0
    require_first_run_approval: bool = True
    auto_approve_first_seen: bool = False
    approval_registry_path: str = "~/.ollama-mcp-bridge/approved_tools.json"
    # Sink policy — taint tracking and source-to-sink enforcement
    tainted_sink_requires_confirmation: bool = True
    block_tainted_exfiltration: bool = True
    block_tainted_destructive_write: bool = True
    allowed_outbound_domains: list[str] = Field(default_factory=list)
    allow_memory_writes_from_third_party_content: bool = False
    # Capability narrowing — safe adapters (opt-in, empty = disabled)
    allowed_path_roots: list[str] = Field(default_factory=list)
    approved_recipients: list[str] = Field(default_factory=list)

    @model_validator(mode="after")
    def max_turns_within_hard_cap(self) -> "SecurityConfig":
        if self.max_turns > self.max_turns_hard_cap:
            raise ValueError(
                f"max_turns ({self.max_turns}) cannot exceed "
                f"max_turns_hard_cap ({self.max_turns_hard_cap})"
            )
        return self

    @model_validator(mode="after")
    def warn_auto_approve_overrides(self) -> "SecurityConfig":
        if self.auto_approve_first_seen and self.require_first_run_approval:
            logger.warning(
                "auto_approve_first_seen=True overrides require_first_run_approval — "
                "all first-seen tools will be auto-approved."
            )
        return self


class LoggingConfig(BaseModel):
    """Logging and audit configuration."""

    model_config = ConfigDict(extra="forbid")

    audit_file: str = "~/.ollama-mcp-bridge/audit.jsonl"
    level: str = "INFO"
    # rotation_days: not implemented — audit file grows indefinitely.
    # Manual rotation: move/truncate the audit file between sessions.


class BridgeConfig(BaseModel):
    """Top-level bridge configuration."""

    model_config = ConfigDict(extra="forbid")

    ollama_host: str = "http://127.0.0.1:11434"
    servers: dict[str, ServerConfig] = Field(default_factory=dict)
    security: SecurityConfig = Field(default_factory=SecurityConfig)
    logging: LoggingConfig = Field(default_factory=LoggingConfig)

    @field_validator("ollama_host")
    @classmethod
    def ollama_host_must_be_localhost(cls, v: str) -> str:
        """Warn if ollama_host is not localhost — remote Ollama exposes model API."""
        if v and "127.0.0.1" not in v and "localhost" not in v:
            logger.warning(
                "ollama_host '%s' is not localhost — the Ollama API will be "
                "accessed over the network. Ensure this is intentional.",
                v,
            )
        return v

    @field_validator("servers")
    @classmethod
    def validate_server_names(cls, v: dict[str, ServerConfig]) -> dict[str, ServerConfig]:
        for name in v:
            if not name.replace("-", "").replace("_", "").isalnum():
                raise ValueError(
                    f"Server name '{name}' must be alphanumeric with hyphens/underscores only"
                )
        return v

    @model_validator(mode="after")
    def check_destructive_in_allowed(self) -> "BridgeConfig":
        """Destructive tools must also be in allowed_tools if allowed_tools is non-empty."""
        for name, server in self.servers.items():
            if not server.allowed_tools:
                # Fail-closed: empty allowlist means no tools available
                logger.info(
                    "Server '%s' has no allowed_tools configured — no tools from "
                    "this server will be available (fail-closed). Add tools to "
                    "allowed_tools to enable them.",
                    name,
                )
                if server.destructive_tools:
                    logger.warning(
                        "Server '%s' has destructive_tools configured but "
                        "allowed_tools is empty — destructive_tools will have "
                        "no effect. This is likely a misconfiguration.",
                        name,
                    )
            else:
                for tool in server.destructive_tools:
                    if tool not in server.allowed_tools:
                        raise ValueError(
                            f"Server '{name}': destructive tool '{tool}' "
                            f"must also be in allowed_tools"
                        )
        return self

    def get_tool_classification(self, server: str, tool: str) -> ActionClass:
        """Get the action classification for a specific tool."""
        server_config = self.servers.get(server)
        if not server_config:
            return ActionClass.WRITE
        if tool in server_config.destructive_tools:
            return ActionClass.DESTRUCTIVE
        if tool in server_config.read_tools:
            return ActionClass.READ
        return ActionClass.WRITE

    def is_tool_allowed(self, server: str, tool: str) -> bool:
        """Check if a tool is in the allowlist for its server."""
        server_config = self.servers.get(server)
        if not server_config:
            return False
        if not server_config.allowed_tools:
            # Empty allowed_tools means NO tools from this server are allowed (fail-closed)
            return False
        return tool in server_config.allowed_tools


def load_config(path: str | Path) -> BridgeConfig:
    """Load bridge configuration from a TOML file.

    Args:
        path: Path to the TOML configuration file.

    Returns:
        Validated BridgeConfig.

    Raises:
        ConfigError: If file not found, invalid TOML, or validation fails.
    """
    path = Path(path).expanduser()
    if not path.exists():
        raise ConfigError(f"Configuration file not found: {path}")

    try:
        with open(path, "rb") as f:
            raw = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(f"Invalid TOML in {path}: {e}") from e

    # TOML uses [bridge] as top-level section
    bridge_raw = raw.get("bridge", {})
    servers_raw = raw.get("servers", {})
    security_raw = raw.get("security", {})
    logging_raw = raw.get("logging", {})

    try:
        servers = {name: ServerConfig(**cfg) for name, cfg in servers_raw.items()}
        security = SecurityConfig(**security_raw) if security_raw else SecurityConfig()
        logging_cfg = LoggingConfig(**logging_raw) if logging_raw else LoggingConfig()

        return BridgeConfig(
            ollama_host=bridge_raw.get("ollama_host", "http://127.0.0.1:11434"),
            servers=servers,
            security=security,
            logging=logging_cfg,
        )
    except Exception as e:
        raise ConfigError(f"Configuration validation failed: {e}") from e
