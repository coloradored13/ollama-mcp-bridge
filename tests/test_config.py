"""Tests for config.py — configuration loading and validation."""

import pytest
import tempfile
from pathlib import Path

from ollama_mcp_bridge.config import (
    BridgeConfig,
    SecurityConfig,
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
        """Destructive tools OK when allowed_tools is empty (all allowed)."""
        cfg = BridgeConfig(
            servers={
                "s": ServerConfig(
                    command="python",
                    allowed_tools=[],
                    destructive_tools=["delete"],
                )
            }
        )
        assert cfg.servers["s"].destructive_tools == ["delete"]

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
        """Empty allowlist means all tools allowed."""
        cfg = BridgeConfig(
            servers={"s": ServerConfig(command="python", allowed_tools=[])}
        )
        assert cfg.is_tool_allowed("s", "anything") is True

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
