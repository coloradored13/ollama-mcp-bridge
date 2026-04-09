"""Tests for config.py — configuration loading and validation."""

import pytest
import tempfile
from pathlib import Path

from ollama_mcp_bridge.config import (
    BridgeConfig,
    DeploymentMode,
    SecurityConfig,
    SecurityProfile,
    ServerConfig,
    load_config,
)
from ollama_mcp_bridge.errors import ConfigError
from ollama_mcp_bridge.types import ActionClass


class TestServerConfig:
    def test_valid_config(self):
        cfg = ServerConfig(
            command="python",
            args=["-m", "test"],
            allowed_tools=["tool1", "tool2"],
        )
        assert cfg.command == "python"
        assert cfg.allowed_tools == ["tool1", "tool2"]

    def test_reject_wildcard_allowlist(self):
        with pytest.raises(ValueError, match="Wildcard"):
            ServerConfig(command="python", allowed_tools=["*"])

    def test_empty_command_rejected(self):
        with pytest.raises(ValueError, match="must not be empty"):
            ServerConfig(command="  ")

    def test_extra_fields_rejected(self):
        with pytest.raises(Exception):
            ServerConfig(command="python", unknown_field="value")

    def test_default_rate_limit(self):
        cfg = ServerConfig(command="python")
        assert cfg.max_calls_per_minute == 30


class TestSecurityConfig:
    def test_defaults(self):
        cfg = SecurityConfig()
        assert cfg.require_confirmation_for_destructive is True
        assert cfg.max_turns == 10
        assert cfg.max_turns_hard_cap == 50
        assert len(cfg.enabled_detectors) == 7

    def test_max_turns_within_hard_cap(self):
        with pytest.raises(ValueError, match="cannot exceed"):
            SecurityConfig(max_turns=100, max_turns_hard_cap=50)

    def test_max_turns_validation_field_order_independent(self):
        """Validation works regardless of which field is set first in kwargs."""
        # hard_cap explicitly lower than max_turns — must fail even if
        # hard_cap is parsed after max_turns in the model definition
        with pytest.raises(ValueError, match="cannot exceed"):
            SecurityConfig(max_turns_hard_cap=5, max_turns=10)

    def test_custom_detectors(self):
        cfg = SecurityConfig(
            enabled_detectors=["instruction_language", "exfiltration_pattern"]
        )
        assert len(cfg.enabled_detectors) == 2

    def test_default_require_first_run_approval(self):
        cfg = SecurityConfig()
        assert cfg.require_first_run_approval is True

    def test_default_auto_approve_first_seen(self):
        cfg = SecurityConfig()
        assert cfg.auto_approve_first_seen is False

    def test_auto_approve_warns_when_require_also_true(self, caplog):
        """auto_approve_first_seen=True with require_first_run_approval=True logs warning."""
        import logging
        with caplog.at_level(logging.WARNING):
            SecurityConfig(auto_approve_first_seen=True, require_first_run_approval=True)
        assert "auto_approve_first_seen=True overrides" in caplog.text


class TestBridgeConfig:
    def test_valid_server_names(self):
        cfg = BridgeConfig(
            servers={
                "sigma-mem": ServerConfig(command="python"),
                "file_server": ServerConfig(command="npx"),
            }
        )
        assert len(cfg.servers) == 2

    def test_invalid_server_name(self):
        with pytest.raises(ValueError, match="alphanumeric"):
            BridgeConfig(
                servers={"bad name!": ServerConfig(command="python")}
            )

    def test_destructive_must_be_in_allowed(self):
        with pytest.raises(ValueError, match="must also be in allowed_tools"):
            BridgeConfig(
                servers={
                    "s": ServerConfig(
                        command="python",
                        allowed_tools=["read"],
                        destructive_tools=["delete"],
                    )
                }
            )

    def test_destructive_allowed_when_allowlist_empty(self):
        """Destructive tools with empty allowlist is a no-op (no tools allowed)."""
        cfg = BridgeConfig(
            servers={
                "s": ServerConfig(
                    command="python",
                    allowed_tools=[],
                    destructive_tools=["delete"],
                )
            }
        )
        # Config is accepted but destructive_tools have no effect — no tools are allowed
        assert cfg.servers["s"].destructive_tools == ["delete"]
        assert cfg.is_tool_allowed("s", "delete") is False

    def test_tool_classification(self):
        cfg = BridgeConfig(
            servers={
                "s": ServerConfig(
                    command="python",
                    destructive_tools=["delete_file"],
                )
            }
        )
        assert cfg.get_tool_classification("s", "delete_file") == ActionClass.DESTRUCTIVE
        assert cfg.get_tool_classification("s", "read_file") == ActionClass.WRITE

    def test_is_tool_allowed_empty_allowlist(self):
        """Empty allowlist means no tools allowed (fail-closed)."""
        cfg = BridgeConfig(
            servers={"s": ServerConfig(command="python", allowed_tools=[])}
        )
        assert cfg.is_tool_allowed("s", "anything") is False

    def test_empty_allowlist_blocks_all_tools(self):
        """Empty allowlist blocks every tool name, not just one."""
        cfg = BridgeConfig(
            servers={"s": ServerConfig(command="python", allowed_tools=[])}
        )
        for tool in ["read", "write", "delete", "list", "execute"]:
            assert cfg.is_tool_allowed("s", tool) is False

    def test_is_tool_allowed_explicit_list(self):
        cfg = BridgeConfig(
            servers={
                "s": ServerConfig(
                    command="python",
                    allowed_tools=["read", "write"],
                )
            }
        )
        assert cfg.is_tool_allowed("s", "read") is True
        assert cfg.is_tool_allowed("s", "delete") is False

    def test_unknown_server_not_allowed(self):
        cfg = BridgeConfig()
        assert cfg.is_tool_allowed("nonexistent", "tool") is False


class TestLoadConfig:
    def test_load_valid_toml(self, tmp_path: Path):
        toml_content = """
[bridge]
ollama_host = "http://127.0.0.1:11434"

[security]
max_turns = 5

[servers.test]
command = "echo"
args = ["hello"]
allowed_tools = ["tool1"]
"""
        config_file = tmp_path / "test.toml"
        config_file.write_text(toml_content)

        cfg = load_config(config_file)
        assert cfg.ollama_host == "http://127.0.0.1:11434"
        assert cfg.security.max_turns == 5
        assert "test" in cfg.servers
        assert cfg.servers["test"].allowed_tools == ["tool1"]

    def test_load_file_not_found(self):
        with pytest.raises(ConfigError, match="not found"):
            load_config("/nonexistent/path.toml")

    def test_load_invalid_toml(self, tmp_path: Path):
        config_file = tmp_path / "bad.toml"
        config_file.write_text("this is not valid toml [[[")

        with pytest.raises(ConfigError, match="Invalid TOML"):
            load_config(config_file)

    def test_load_minimal_config(self, tmp_path: Path):
        """Empty config should use all defaults."""
        config_file = tmp_path / "minimal.toml"
        config_file.write_text("")

        cfg = load_config(config_file)
        assert cfg.ollama_host == "http://127.0.0.1:11434"
        assert len(cfg.servers) == 0


# --- Destination Policy Config tests ---


from ollama_mcp_bridge.types import DestinationPolicy


class TestDestinationPolicyConfig:
    def test_parse_destination_from_toml(self, tmp_path: Path):
        """[[destinations.server.tool]] TOML sections parse correctly."""
        toml_content = """\
[servers.webhooks]
command = "echo"
allowed_tools = ["send_event"]

[[destinations.webhooks.send_event]]
scheme = "https"
host = "hooks.internal"
path_prefixes = ["/agent-events"]
"""
        config_file = tmp_path / "dest.toml"
        config_file.write_text(toml_content)

        cfg = load_config(config_file)
        policies = cfg.get_destination_policies("webhooks", "send_event")
        assert len(policies) == 1
        assert policies[0].host == "hooks.internal"
        assert policies[0].scheme == "https"
        assert policies[0].path_prefixes == ["/agent-events"]
        assert policies[0].max_payload_bytes == 65536  # default value

    def test_max_payload_bytes_nondefault_raises(self, tmp_path: Path):
        """Non-default max_payload_bytes raises ConfigError (not yet enforced)."""
        toml_content = """\
[servers.webhooks]
command = "echo"
allowed_tools = ["send_event"]

[[destinations.webhooks.send_event]]
host = "hooks.internal"
max_payload_bytes = 32768
"""
        config_file = tmp_path / "dest.toml"
        config_file.write_text(toml_content)

        with pytest.raises((ConfigError, Exception), match="not yet enforced"):
            load_config(config_file)

    def test_multiple_policies_per_tool(self, tmp_path: Path):
        """Multiple [[destinations.server.tool]] entries produce a list."""
        toml_content = """\
[servers.api]
command = "echo"
allowed_tools = ["call"]

[[destinations.api.call]]
host = "api.example.com"
path_prefixes = ["/v1/"]

[[destinations.api.call]]
host = "api.partner.com"
path_prefixes = ["/webhook"]
"""
        config_file = tmp_path / "multi.toml"
        config_file.write_text(toml_content)

        cfg = load_config(config_file)
        policies = cfg.get_destination_policies("api", "call")
        assert len(policies) == 2
        hosts = {p.host for p in policies}
        assert hosts == {"api.example.com", "api.partner.com"}

    def test_single_table_destination(self, tmp_path: Path):
        """[destinations.server.tool] (non-array) also works."""
        toml_content = """\
[servers.api]
command = "echo"
allowed_tools = ["fetch"]

[destinations.api.fetch]
host = "api.example.com"
"""
        config_file = tmp_path / "single.toml"
        config_file.write_text(toml_content)

        cfg = load_config(config_file)
        policies = cfg.get_destination_policies("api", "fetch")
        assert len(policies) == 1
        assert policies[0].host == "api.example.com"

    def test_allowed_outbound_domains_auto_converts(self, tmp_path: Path):
        """allowed_outbound_domains produces global destination policies."""
        toml_content = """\
[security]
allowed_outbound_domains = ["example.com", "api.co"]
"""
        config_file = tmp_path / "compat.toml"
        config_file.write_text(toml_content)

        cfg = load_config(config_file)
        # Global policies available for any server/tool
        policies = cfg.get_destination_policies("any-server", "any-tool")
        assert len(policies) == 4  # 2 domains x 2 schemes (https + http)
        hosts = {p.host for p in policies}
        assert hosts == {"example.com", "api.co"}
        schemes = {p.scheme for p in policies}
        assert schemes == {"https", "http"}
        # All should allow subdomains (backward compat)
        assert all(p.allow_subdomains for p in policies)

    def test_get_destination_policies_tool_specific_plus_global(self, tmp_path: Path):
        """Tool-specific policies are combined with global fallback."""
        toml_content = """\
[servers.api]
command = "echo"
allowed_tools = ["send"]

[[destinations.api.send]]
host = "specific.com"

[security]
allowed_outbound_domains = ["global.com"]
"""
        config_file = tmp_path / "combined.toml"
        config_file.write_text(toml_content)

        cfg = load_config(config_file)
        policies = cfg.get_destination_policies("api", "send")
        # 1 tool-specific + 2 global (https + http)
        assert len(policies) == 3
        hosts = {p.host for p in policies}
        assert "specific.com" in hosts
        assert "global.com" in hosts

    def test_get_destination_policies_no_config(self):
        """No destination config returns empty list."""
        cfg = BridgeConfig()
        assert cfg.get_destination_policies("server", "tool") == []

    def test_require_destination_policy_field(self):
        """SecurityConfig accepts require_destination_policy_for_outbound."""
        cfg = SecurityConfig(require_destination_policy_for_outbound=True)
        assert cfg.require_destination_policy_for_outbound is True

    def test_require_destination_policy_default_false(self):
        cfg = SecurityConfig()
        assert cfg.require_destination_policy_for_outbound is False


# --- PathPolicy config tests ---


class TestPathPolicyConfig:
    def test_parse_path_policy_from_toml(self, tmp_path: Path):
        """[paths.server.tool] TOML sections parse correctly."""
        toml_content = """\
[servers.files]
command = "echo"
allowed_tools = ["read_file"]

[paths.files.read_file]
allowed_roots = ["/tmp/sandbox", "~/safe"]
allow_relative_paths = false
normalize_symlinks = true
extensions_allowlist = [".txt", ".md"]
"""
        config_file = tmp_path / "paths.toml"
        config_file.write_text(toml_content)

        cfg = load_config(config_file)
        policy = cfg.get_path_policy("files", "read_file")
        assert policy is not None
        assert policy.allowed_roots == ["/tmp/sandbox", "~/safe"]
        assert policy.allow_relative_paths is False
        assert policy.normalize_symlinks is True
        assert policy.extensions_allowlist == [".txt", ".md"]

    def test_path_policy_read_only(self, tmp_path: Path):
        toml_content = """\
[servers.files]
command = "echo"
allowed_tools = ["read_file"]

[paths.files.read_file]
allowed_roots = ["/data"]
read_only = true
"""
        config_file = tmp_path / "readonly.toml"
        config_file.write_text(toml_content)

        cfg = load_config(config_file)
        policy = cfg.get_path_policy("files", "read_file")
        assert policy is not None
        assert policy.read_only is True

    def test_allowed_path_roots_auto_converts(self, tmp_path: Path):
        """allowed_path_roots produces global path policy."""
        toml_content = """\
[security]
allowed_path_roots = ["/tmp/sandbox", "/home/user/safe"]
"""
        config_file = tmp_path / "compat.toml"
        config_file.write_text(toml_content)

        cfg = load_config(config_file)
        policy = cfg.get_path_policy("any-server", "any-tool")
        assert policy is not None
        assert policy.allowed_roots == ["/tmp/sandbox", "/home/user/safe"]
        assert policy.allow_relative_paths is False
        assert policy.normalize_symlinks is True

    def test_get_path_policy_tool_specific(self, tmp_path: Path):
        """Tool-specific path policy takes precedence."""
        toml_content = """\
[servers.files]
command = "echo"
allowed_tools = ["write_file"]

[paths.files.write_file]
allowed_roots = ["/specific"]

[security]
allowed_path_roots = ["/global"]
"""
        config_file = tmp_path / "specific.toml"
        config_file.write_text(toml_content)

        cfg = load_config(config_file)
        policy = cfg.get_path_policy("files", "write_file")
        assert policy is not None
        assert policy.allowed_roots == ["/specific"]

    def test_get_path_policy_server_wide_fallback(self, tmp_path: Path):
        """Server-wide _all policy used when no tool-specific policy."""
        toml_content = """\
[servers.files]
command = "echo"
allowed_tools = ["read_file", "write_file"]

[paths.files._all]
allowed_roots = ["/server-wide"]
"""
        config_file = tmp_path / "serverwide.toml"
        config_file.write_text(toml_content)

        cfg = load_config(config_file)
        policy = cfg.get_path_policy("files", "read_file")
        assert policy is not None
        assert policy.allowed_roots == ["/server-wide"]

    def test_get_path_policy_no_config(self):
        """No path config returns None."""
        cfg = BridgeConfig()
        assert cfg.get_path_policy("server", "tool") is None

    def test_path_policy_delete_allowed(self, tmp_path: Path):
        toml_content = """\
[servers.files]
command = "echo"
allowed_tools = ["delete_file"]

[paths.files.delete_file]
allowed_roots = ["/tmp/trash"]
delete_allowed = true
"""
        config_file = tmp_path / "delete.toml"
        config_file.write_text(toml_content)

        cfg = load_config(config_file)
        policy = cfg.get_path_policy("files", "delete_file")
        assert policy is not None
        assert policy.delete_allowed is True


# --- RecipientPolicy config tests ---


class TestRecipientPolicyConfig:
    def test_parse_recipient_policy_from_toml(self, tmp_path: Path):
        """[recipients.server.tool] TOML sections parse correctly."""
        toml_content = """\
[servers.email]
command = "echo"
allowed_tools = ["send_email"]

[recipients.email.send_email]
approved_addresses = ["admin@co.com", "ops@co.com"]
approved_domains = ["internal.corp"]
internal_only = true
"""
        config_file = tmp_path / "recip.toml"
        config_file.write_text(toml_content)

        cfg = load_config(config_file)
        policy = cfg.get_recipient_policy("email", "send_email")
        assert policy is not None
        assert policy.approved_addresses == ["admin@co.com", "ops@co.com"]
        assert policy.approved_domains == ["internal.corp"]
        assert policy.internal_only is True

    def test_recipient_identity_groups(self, tmp_path: Path):
        toml_content = """\
[servers.email]
command = "echo"
allowed_tools = ["send_email"]

[recipients.email.send_email]
approved_addresses = []

[recipients.email.send_email.identity_groups]
engineering = ["alice@co.com", "bob@co.com"]
leadership = ["ceo@co.com"]
"""
        config_file = tmp_path / "groups.toml"
        config_file.write_text(toml_content)

        cfg = load_config(config_file)
        policy = cfg.get_recipient_policy("email", "send_email")
        assert policy is not None
        assert "engineering" in policy.identity_groups
        assert "alice@co.com" in policy.identity_groups["engineering"]
        assert "leadership" in policy.identity_groups

    def test_approved_recipients_auto_converts(self, tmp_path: Path):
        """approved_recipients produces global recipient policy."""
        toml_content = """\
[security]
approved_recipients = ["admin@co.com", "ops@co.com"]
"""
        config_file = tmp_path / "compat.toml"
        config_file.write_text(toml_content)

        cfg = load_config(config_file)
        policy = cfg.get_recipient_policy("any-server", "any-tool")
        assert policy is not None
        assert policy.approved_addresses == ["admin@co.com", "ops@co.com"]

    def test_get_recipient_policy_tool_specific(self, tmp_path: Path):
        """Tool-specific recipient policy takes precedence."""
        toml_content = """\
[servers.email]
command = "echo"
allowed_tools = ["send_email"]

[recipients.email.send_email]
approved_addresses = ["specific@co.com"]

[security]
approved_recipients = ["global@co.com"]
"""
        config_file = tmp_path / "specific.toml"
        config_file.write_text(toml_content)

        cfg = load_config(config_file)
        policy = cfg.get_recipient_policy("email", "send_email")
        assert policy is not None
        assert policy.approved_addresses == ["specific@co.com"]

    def test_get_recipient_policy_server_wide_fallback(self, tmp_path: Path):
        toml_content = """\
[servers.email]
command = "echo"
allowed_tools = ["send_email", "send_notification"]

[recipients.email._all]
approved_domains = ["internal.corp"]
"""
        config_file = tmp_path / "serverwide.toml"
        config_file.write_text(toml_content)

        cfg = load_config(config_file)
        policy = cfg.get_recipient_policy("email", "send_email")
        assert policy is not None
        assert policy.approved_domains == ["internal.corp"]

    def test_get_recipient_policy_no_config(self):
        """No recipient config returns None."""
        cfg = BridgeConfig()
        assert cfg.get_recipient_policy("server", "tool") is None


# --- SecurityProfile tests ---


class TestSecurityProfile:
    def test_enum_values(self):
        assert SecurityProfile.COMPAT == "compat"
        assert SecurityProfile.STANDARD == "standard"
        assert SecurityProfile.HARDENED == "hardened"
        assert SecurityProfile.HIGH_CONSEQUENCE == "high_consequence"

    def test_default_is_standard(self):
        cfg = SecurityConfig()
        assert cfg.security_profile == SecurityProfile.STANDARD

    def test_compat_profile_allows_anything(self):
        """Compat profile imposes no extra constraints."""
        cfg = SecurityConfig(
            security_profile="compat",
            require_first_run_approval=False,
            auto_approve_first_seen=True,
        )
        assert cfg.security_profile == SecurityProfile.COMPAT

    def test_hardened_requires_first_run_approval(self):
        with pytest.raises(Exception, match="require_first_run_approval"):
            SecurityConfig(
                security_profile="hardened",
                require_first_run_approval=False,
            )

    def test_hardened_forbids_auto_approve(self):
        with pytest.raises(Exception, match="auto_approve_first_seen"):
            SecurityConfig(
                security_profile="hardened",
                auto_approve_first_seen=True,
            )

    def test_hardened_accepts_valid_config(self):
        cfg = SecurityConfig(
            security_profile="hardened",
            require_first_run_approval=True,
            auto_approve_first_seen=False,
        )
        assert cfg.security_profile == SecurityProfile.HARDENED

    def test_high_consequence_requires_first_run_approval(self):
        with pytest.raises(Exception, match="require_first_run_approval"):
            SecurityConfig(
                security_profile="high_consequence",
                require_first_run_approval=False,
            )

    def test_high_consequence_forbids_auto_approve(self):
        with pytest.raises(Exception, match="auto_approve_first_seen"):
            SecurityConfig(
                security_profile="high_consequence",
                auto_approve_first_seen=True,
            )

    def test_high_consequence_accepts_valid_config(self):
        cfg = SecurityConfig(
            security_profile="high_consequence",
            require_first_run_approval=True,
            auto_approve_first_seen=False,
        )
        assert cfg.security_profile == SecurityProfile.HIGH_CONSEQUENCE

    def test_profile_from_toml(self, tmp_path: Path):
        toml_content = """\
[security]
security_profile = "hardened"
"""
        config_file = tmp_path / "profile.toml"
        config_file.write_text(toml_content)

        cfg = load_config(config_file)
        assert cfg.security.security_profile == SecurityProfile.HARDENED


# --- DeploymentMode tests ---


class TestDeploymentMode:
    def test_enum_values(self):
        assert DeploymentMode.LOCAL_DEV == "local_dev"
        assert DeploymentMode.SANDBOXED == "sandboxed"
        assert DeploymentMode.HIGH_CONSEQUENCE == "high_consequence"

    def test_default_is_local_dev(self):
        cfg = SecurityConfig()
        assert cfg.deployment_mode == DeploymentMode.LOCAL_DEV

    def test_deployment_fields_default(self):
        cfg = SecurityConfig()
        assert cfg.require_network_egress_controls is False
        assert cfg.require_filesystem_sandbox is False
        assert cfg.require_secret_scoping is False

    def test_deployment_from_toml(self, tmp_path: Path):
        toml_content = """\
[security]
deployment_mode = "sandboxed"
require_network_egress_controls = true
require_filesystem_sandbox = true
"""
        config_file = tmp_path / "deploy.toml"
        config_file.write_text(toml_content)

        cfg = load_config(config_file)
        assert cfg.security.deployment_mode == DeploymentMode.SANDBOXED
        assert cfg.security.require_network_egress_controls is True
        assert cfg.security.require_filesystem_sandbox is True
