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
from enum import Enum
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from .errors import ConfigError
from .types import (
    ActionClass,
    CapabilitySource,
    DestinationPolicy,
    PathPolicy,
    RecipientPolicy,
    ToolCapabilityManifest,
)

logger = logging.getLogger(__name__)


class SecurityProfile(str, Enum):
    """Security profile controlling baseline enforcement behavior.

    COMPAT: Backward-compatible defaults. No additional enforcement beyond
        what was configured explicitly. Useful for migration from pre-profile configs.
    STANDARD: Current defaults. All security features available but opt-in.
        Suitable for development and trusted-server deployments.
    HARDENED: Proactive security enforcement. First-run approval required,
        auto-approve forbidden, adapters auto-activate for capable tools.
        Suitable for production deployments with untrusted tools.
    HIGH_CONSEQUENCE: Maximum enforcement. Explicit capability manifest required
        for dangerous tools, destination/path/recipient policies required for
        capable tools. Inferred-only manifests blocked for dangerous tools.
        Suitable for high-stakes deployments where a single miss is bad.

    Profile is a config field, not a runtime toggle. Changing profile requires restart.
    """

    COMPAT = "compat"
    STANDARD = "standard"
    HARDENED = "hardened"
    HIGH_CONSEQUENCE = "high_consequence"


class DeploymentMode(str, Enum):
    """Deployment environment declaration.

    The bridge cannot prove the OS is sandboxed, but it can require the
    operator to declare the mode and make that declaration visible in
    logs and audit headers.

    LOCAL_DEV: Development machine. Minimal deployment checks.
    SANDBOXED: Sandboxed environment (container, VM, restricted user).
    HIGH_CONSEQUENCE: High-stakes deployment. Startup fails if required
        deployment assertions are not set.
    """

    LOCAL_DEV = "local_dev"
    SANDBOXED = "sandboxed"
    HIGH_CONSEQUENCE = "high_consequence"


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
    require_destination_policy_for_outbound: bool = False
    allow_memory_writes_from_third_party_content: bool = False
    # Capability narrowing — safe adapters (opt-in, empty = disabled)
    allowed_path_roots: list[str] = Field(default_factory=list)
    approved_recipients: list[str] = Field(default_factory=list)
    # Security profile — controls baseline enforcement behavior (PR 16)
    security_profile: SecurityProfile = SecurityProfile.STANDARD
    # Deployment guardrails (PR 17)
    deployment_mode: DeploymentMode = DeploymentMode.LOCAL_DEV
    require_network_egress_controls: bool = False
    require_filesystem_sandbox: bool = False
    require_secret_scoping: bool = False

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

    @model_validator(mode="after")
    def enforce_profile_requirements(self) -> "SecurityConfig":
        """Enforce security profile constraints on config values."""
        if self.security_profile == SecurityProfile.HARDENED:
            if not self.require_first_run_approval:
                raise ValueError(
                    "security_profile='hardened' requires require_first_run_approval=True"
                )
            if self.auto_approve_first_seen:
                raise ValueError("security_profile='hardened' forbids auto_approve_first_seen=True")
        elif self.security_profile == SecurityProfile.HIGH_CONSEQUENCE:
            if not self.require_first_run_approval:
                raise ValueError(
                    "security_profile='high_consequence' requires require_first_run_approval=True"
                )
            if self.auto_approve_first_seen:
                raise ValueError(
                    "security_profile='high_consequence' forbids auto_approve_first_seen=True"
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
    capabilities: dict[str, dict[str, ToolCapabilityManifest]] = Field(default_factory=dict)
    destinations: dict[str, dict[str, list[DestinationPolicy]]] = Field(default_factory=dict)
    paths: dict[str, dict[str, PathPolicy]] = Field(default_factory=dict)
    recipients: dict[str, dict[str, RecipientPolicy]] = Field(default_factory=dict)

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

    def get_capability_manifest(self, server: str, tool: str) -> ToolCapabilityManifest | None:
        """Look up explicit capability manifest from config, or None if not declared."""
        server_caps = self.capabilities.get(server)
        if not server_caps:
            return None
        return server_caps.get(tool)

    def get_destination_policies(self, server: str, tool: str) -> list[DestinationPolicy]:
        """Look up destination policies for a server+tool pair.

        Returns policies from:
        1. Tool-specific [[destinations.<server>.<tool>]] config
        2. Global policies converted from allowed_outbound_domains (fallback)
        """
        server_dests = self.destinations.get(server, {})
        tool_policies = server_dests.get(tool, [])

        # Also include global policies (backward-compat from allowed_outbound_domains)
        global_policies = self.destinations.get("_global", {}).get("_all", [])

        return tool_policies + global_policies

    def get_path_policy(self, server: str, tool: str) -> PathPolicy | None:
        """Look up path policy for a server+tool pair.

        Returns tool-specific policy, then server-wide policy (_all), then
        global policy (_global._all), or None if no policy configured.
        """
        server_paths = self.paths.get(server, {})
        policy = server_paths.get(tool)
        if policy:
            return policy

        # Server-wide fallback
        policy = server_paths.get("_all")
        if policy:
            return policy

        # Global fallback (from allowed_path_roots backward compat)
        global_paths = self.paths.get("_global", {})
        return global_paths.get("_all")

    def get_recipient_policy(self, server: str, tool: str) -> RecipientPolicy | None:
        """Look up recipient policy for a server+tool pair.

        Returns tool-specific policy, then server-wide policy (_all), then
        global policy (_global._all), or None if no policy configured.
        """
        server_recipients = self.recipients.get(server, {})
        policy = server_recipients.get(tool)
        if policy:
            return policy

        # Server-wide fallback
        policy = server_recipients.get("_all")
        if policy:
            return policy

        # Global fallback (from approved_recipients backward compat)
        global_recipients = self.recipients.get("_global", {})
        return global_recipients.get("_all")


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
    capabilities_raw = raw.get("capabilities", {})
    destinations_raw = raw.get("destinations", {})
    paths_raw = raw.get("paths", {})
    recipients_raw = raw.get("recipients", {})

    try:
        servers = {name: ServerConfig(**cfg) for name, cfg in servers_raw.items()}
        security = SecurityConfig(**security_raw) if security_raw else SecurityConfig()
        logging_cfg = LoggingConfig(**logging_raw) if logging_raw else LoggingConfig()

        # Parse [capabilities.<server>.<tool>] sections
        capabilities: dict[str, dict[str, ToolCapabilityManifest]] = {}
        for server_name, tools in capabilities_raw.items():
            if not isinstance(tools, dict):
                raise ConfigError(f"capabilities.{server_name} must be a table of tool configs")
            capabilities[server_name] = {}
            for tool_name, cap_fields in tools.items():
                if not isinstance(cap_fields, dict):
                    raise ConfigError(f"capabilities.{server_name}.{tool_name} must be a table")
                capabilities[server_name][tool_name] = ToolCapabilityManifest(
                    source=CapabilitySource.CONFIG,
                    **cap_fields,
                )

        # Parse [[destinations.<server>.<tool>]] sections
        destinations: dict[str, dict[str, list[DestinationPolicy]]] = {}
        for server_name, tools in destinations_raw.items():
            if not isinstance(tools, dict):
                raise ConfigError(f"destinations.{server_name} must be a table of tool configs")
            destinations[server_name] = {}
            for tool_name, policy_data in tools.items():
                if isinstance(policy_data, list):
                    # [[destinations.server.tool]] — array of tables
                    destinations[server_name][tool_name] = [
                        DestinationPolicy(**p) for p in policy_data
                    ]
                elif isinstance(policy_data, dict):
                    # [destinations.server.tool] — single table
                    destinations[server_name][tool_name] = [DestinationPolicy(**policy_data)]
                else:
                    raise ConfigError(
                        f"destinations.{server_name}.{tool_name} must be a table or array of tables"
                    )

        # Parse [paths.<server>.<tool>] sections
        paths: dict[str, dict[str, PathPolicy]] = {}
        for server_name, tools in paths_raw.items():
            if not isinstance(tools, dict):
                raise ConfigError(f"paths.{server_name} must be a table of tool configs")
            paths[server_name] = {}
            for tool_name, path_fields in tools.items():
                if not isinstance(path_fields, dict):
                    raise ConfigError(f"paths.{server_name}.{tool_name} must be a table")
                paths[server_name][tool_name] = PathPolicy(**path_fields)

        # Parse [recipients.<server>.<tool>] sections
        recipients: dict[str, dict[str, RecipientPolicy]] = {}
        for server_name, tools in recipients_raw.items():
            if not isinstance(tools, dict):
                raise ConfigError(f"recipients.{server_name} must be a table of tool configs")
            recipients[server_name] = {}
            for tool_name, recip_fields in tools.items():
                if not isinstance(recip_fields, dict):
                    raise ConfigError(f"recipients.{server_name}.{tool_name} must be a table")
                recipients[server_name][tool_name] = RecipientPolicy(**recip_fields)

        # Auto-convert allowed_path_roots to global path policy
        if security.allowed_path_roots:
            logger.info(
                "Converting allowed_path_roots to path policy "
                "(backward compatibility). Consider migrating to "
                "[paths.<server>.<tool>] for finer control."
            )
            paths.setdefault("_global", {})["_all"] = PathPolicy(
                allowed_roots=security.allowed_path_roots,
                allow_relative_paths=False,
                normalize_symlinks=True,
            )

        # Auto-convert approved_recipients to global recipient policy
        if security.approved_recipients:
            logger.info(
                "Converting approved_recipients to recipient policy "
                "(backward compatibility). Consider migrating to "
                "[recipients.<server>.<tool>] for finer control."
            )
            recipients.setdefault("_global", {})["_all"] = RecipientPolicy(
                approved_addresses=security.approved_recipients,
            )

        # Auto-convert allowed_outbound_domains to global destination policies
        if security.allowed_outbound_domains:
            logger.info(
                "Converting allowed_outbound_domains to destination policies "
                "(backward compatibility). Consider migrating to "
                "[[destinations.<server>.<tool>]] for finer control."
            )
            global_policies: list[DestinationPolicy] = []
            for domain in security.allowed_outbound_domains:
                # Generate both https and http to match current scheme-agnostic behavior
                for scheme in ("https", "http"):
                    global_policies.append(
                        DestinationPolicy(
                            host=domain,
                            scheme=scheme,
                            allow_subdomains=True,
                            allow_ip_literals=False,
                            allow_private_ranges=False,
                            allow_redirects=False,
                        )
                    )
            destinations.setdefault("_global", {})["_all"] = global_policies

        return BridgeConfig(
            ollama_host=bridge_raw.get("ollama_host", "http://127.0.0.1:11434"),
            servers=servers,
            security=security,
            logging=logging_cfg,
            capabilities=capabilities,
            destinations=destinations,
            paths=paths,
            recipients=recipients,
        )
    except ConfigError:
        raise
    except Exception as e:
        raise ConfigError(f"Configuration validation failed: {e}") from e
