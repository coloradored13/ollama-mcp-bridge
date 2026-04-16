"""Tests for capabilities.py — conservative capability inference engine."""

import pytest

from ollama_mcp_bridge.capabilities import _extract_schema_fields, infer_capabilities
from ollama_mcp_bridge.types import CapabilitySource, ToolSchema

# --- Helpers ---


def _make_tool(
    name: str = "test_tool",
    description: str = "A test tool",
    input_schema: dict | None = None,
) -> ToolSchema:
    """Build a ToolSchema for testing."""
    return ToolSchema(
        server="test-server",
        name=name,
        description=description,
        input_schema=input_schema or {"type": "object", "properties": {}},
    )


# --- Source is always INFERRED ---


class TestSourceIsAlwaysInferred:
    """Every inferred manifest must have source=INFERRED."""

    def test_safe_tool_source(self):
        result = infer_capabilities(_make_tool("get_time", "Returns current time"))
        assert result.source == CapabilitySource.INFERRED

    def test_dangerous_tool_source(self):
        result = infer_capabilities(_make_tool("delete_all_data", "Deletes everything"))
        assert result.source == CapabilitySource.INFERRED

    def test_empty_tool_source(self):
        result = infer_capabilities(_make_tool("x", ""))
        assert result.source == CapabilitySource.INFERRED


# --- Safe read-only tool ---


class TestSafeReadOnlyTool:
    """A clearly benign tool should have most/all flags False."""

    def test_get_time_all_false(self):
        result = infer_capabilities(_make_tool("get_time", "Returns the current time"))
        assert not result.network_access
        assert not result.outbound_data_transfer
        assert not result.filesystem_read
        assert not result.filesystem_write
        assert not result.filesystem_delete
        assert not result.memory_write
        assert not result.external_messaging
        assert not result.code_execution
        assert not result.credential_access
        assert not result.user_identity_impact
        assert not result.destructive
        assert not result.high_consequence
        assert not result.is_dangerous

    def test_calculate_sum(self):
        result = infer_capabilities(_make_tool("calculate_sum", "Add two numbers"))
        assert not result.is_dangerous
        assert not result.has_outbound_capability
        assert not result.has_filesystem_capability

    def test_read_file_only_fs_read(self):
        """A read_file tool should flag filesystem_read but nothing dangerous."""
        result = infer_capabilities(_make_tool("read_file", "Read a file from disk"))
        assert result.filesystem_read
        assert not result.filesystem_write
        assert not result.filesystem_delete
        assert not result.is_dangerous


# --- Network / outbound tools ---


class TestNetworkAccess:
    """Tools that send data externally should flag network + outbound."""

    def test_send_email(self):
        result = infer_capabilities(
            _make_tool(
                "send_email",
                "Send an email to a recipient",
            )
        )
        assert result.network_access
        assert result.outbound_data_transfer
        assert result.external_messaging
        assert result.has_outbound_capability

    def test_fetch_url(self):
        result = infer_capabilities(_make_tool("fetch_url", "Fetch a URL"))
        assert result.network_access
        assert result.outbound_data_transfer

    def test_post_webhook(self):
        result = infer_capabilities(
            _make_tool(
                "post_webhook",
                "Post data to a webhook",
            )
        )
        assert result.network_access
        assert result.outbound_data_transfer

    def test_upload_file(self):
        result = infer_capabilities(
            _make_tool(
                "upload_file",
                "Upload a file to the server",
            )
        )
        assert result.network_access
        assert result.outbound_data_transfer

    def test_download_tool(self):
        result = infer_capabilities(
            _make_tool(
                "download_artifact",
                "Download build artifact",
            )
        )
        assert result.network_access


# --- Schema-field based detection ---


class TestSchemaFieldDetection:
    """Tools with revealing field names in input_schema should be flagged."""

    def test_url_field_triggers_network(self):
        """A benign-named tool with a 'url' field should flag network access."""
        schema = {
            "type": "object",
            "properties": {"url": {"type": "string"}, "format": {"type": "string"}},
        }
        result = infer_capabilities(
            _make_tool(
                "process_data",
                "Process some data",
                schema,
            )
        )
        assert result.network_access
        assert result.outbound_data_transfer

    def test_endpoint_field_triggers_network(self):
        schema = {
            "type": "object",
            "properties": {"endpoint": {"type": "string"}},
        }
        result = infer_capabilities(
            _make_tool(
                "run_task",
                "Run a generic task",
                schema,
            )
        )
        assert result.network_access

    def test_webhook_field_triggers_network(self):
        schema = {
            "type": "object",
            "properties": {"webhook": {"type": "string"}},
        }
        result = infer_capabilities(
            _make_tool(
                "notify_complete",
                "Notify when complete",
                schema,
            )
        )
        assert result.network_access

    def test_password_field_triggers_credential(self):
        """A tool with 'password' in schema should flag credential_access."""
        schema = {
            "type": "object",
            "properties": {
                "username": {"type": "string"},
                "password": {"type": "string"},
            },
        }
        result = infer_capabilities(
            _make_tool(
                "authenticate_user",
                "Log in a user",
                schema,
            )
        )
        assert result.credential_access

    def test_api_key_field_triggers_credential(self):
        schema = {
            "type": "object",
            "properties": {"api_key": {"type": "string"}},
        }
        result = infer_capabilities(
            _make_tool(
                "configure_service",
                "Configure a service",
                schema,
            )
        )
        assert result.credential_access

    def test_secret_field_triggers_credential(self):
        schema = {
            "type": "object",
            "properties": {"client_secret": {"type": "string"}},
        }
        result = infer_capabilities(
            _make_tool(
                "setup_oauth",
                "Set up OAuth flow",
                schema,
            )
        )
        assert result.credential_access

    def test_benign_fields_no_flags(self):
        """Schema with only benign field names should not trigger anything."""
        schema = {
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "limit": {"type": "integer"},
                "offset": {"type": "integer"},
            },
        }
        result = infer_capabilities(
            _make_tool(
                "search_records",
                "Search the database",
                schema,
            )
        )
        assert not result.network_access
        assert not result.credential_access


# --- Filesystem tools ---


class TestFilesystemCapabilities:
    def test_write_file(self):
        result = infer_capabilities(
            _make_tool(
                "write_file",
                "Write content to a file",
            )
        )
        assert result.filesystem_write
        assert not result.filesystem_delete

    def test_save_document(self):
        result = infer_capabilities(
            _make_tool(
                "save_document",
                "Save a document to disk",
            )
        )
        assert result.filesystem_write

    def test_delete_file(self):
        result = infer_capabilities(
            _make_tool(
                "delete_file",
                "Delete a file from disk",
            )
        )
        assert result.filesystem_delete
        assert result.is_dangerous

    def test_list_directory(self):
        result = infer_capabilities(
            _make_tool(
                "list_directory",
                "List files in a directory",
            )
        )
        assert result.filesystem_read
        assert not result.filesystem_write

    def test_mkdir_flags_write(self):
        result = infer_capabilities(
            _make_tool(
                "mkdir",
                "Create a directory",
            )
        )
        assert result.filesystem_write

    def test_rm_flags_delete(self):
        result = infer_capabilities(
            _make_tool(
                "rm",
                "Remove a file",
            )
        )
        assert result.filesystem_delete


# --- Memory write ---


class TestMemoryWrite:
    def test_store_memory(self):
        result = infer_capabilities(
            _make_tool(
                "store_memory",
                "Store a memory for later recall",
            )
        )
        assert result.memory_write

    def test_save_note(self):
        result = infer_capabilities(
            _make_tool(
                "save_note",
                "Save a note to the knowledge base",
            )
        )
        assert result.memory_write

    def test_create_knowledge(self):
        result = infer_capabilities(
            _make_tool(
                "create_knowledge",
                "Create a knowledge entry",
            )
        )
        assert result.memory_write

    def test_remember(self):
        result = infer_capabilities(
            _make_tool(
                "remember",
                "Remember this information",
            )
        )
        assert result.memory_write

    def test_update_knowledge(self):
        result = infer_capabilities(
            _make_tool(
                "update_knowledge",
                "Update the knowledge base",
            )
        )
        assert result.memory_write

    def test_verb_in_name_context_in_description(self):
        """Storage verb in name + memory context in description = memory_write."""
        result = infer_capabilities(
            _make_tool(
                "store_entry",
                "Store an entry in the knowledge base",
            )
        )
        assert result.memory_write

    def test_generic_create_not_memory(self):
        """'create_file' has a storage verb but filesystem context, not memory."""
        result = infer_capabilities(
            _make_tool(
                "create_file",
                "Create a new file on disk",
            )
        )
        assert not result.memory_write
        assert result.filesystem_write


# --- Destructive / delete ---


class TestDestructiveCapabilities:
    def test_delete_database(self):
        result = infer_capabilities(
            _make_tool(
                "delete_database",
                "Delete an entire database",
            )
        )
        assert result.destructive
        assert result.filesystem_delete
        assert result.high_consequence

    def test_drop_table(self):
        result = infer_capabilities(
            _make_tool(
                "drop_table",
                "Drop a database table",
            )
        )
        assert result.destructive
        assert result.high_consequence

    def test_factory_reset(self):
        result = infer_capabilities(
            _make_tool(
                "factory_reset",
                "Reset to factory defaults",
            )
        )
        assert result.destructive
        assert result.high_consequence

    def test_nuke_everything(self):
        result = infer_capabilities(
            _make_tool(
                "nuke_cache",
                "Completely destroy the cache",
            )
        )
        assert result.destructive


# --- High consequence ---


class TestHighConsequence:
    def test_deploy_production(self):
        result = infer_capabilities(
            _make_tool(
                "deploy_production",
                "Deploy to production environment",
            )
        )
        assert result.high_consequence

    def test_migrate_database(self):
        result = infer_capabilities(
            _make_tool(
                "migrate_database",
                "Run database migration",
            )
        )
        assert result.high_consequence

    def test_release_version(self):
        result = infer_capabilities(
            _make_tool(
                "release_version",
                "Release a new version",
            )
        )
        assert result.high_consequence

    def test_destructive_implies_high_consequence(self):
        """Any destructive tool is automatically high_consequence."""
        result = infer_capabilities(
            _make_tool(
                "wipe_data",
                "Wipe all data",
            )
        )
        assert result.destructive
        assert result.high_consequence


# --- External messaging ---


class TestExternalMessaging:
    def test_send_message(self):
        result = infer_capabilities(
            _make_tool(
                "send_message",
                "Send a message to a user",
            )
        )
        assert result.external_messaging
        assert result.network_access  # conservative cross-flag
        assert result.outbound_data_transfer

    def test_post_comment(self):
        result = infer_capabilities(
            _make_tool(
                "post_comment",
                "Post a comment on an issue",
            )
        )
        assert result.external_messaging

    def test_slack_notify(self):
        result = infer_capabilities(
            _make_tool(
                "slack_notify",
                "Send a Slack notification",
            )
        )
        assert result.external_messaging
        assert result.network_access

    def test_send_sms(self):
        result = infer_capabilities(
            _make_tool(
                "send_sms",
                "Send an SMS message",
            )
        )
        assert result.external_messaging


# --- Code execution ---


class TestCodeExecution:
    def test_run_code(self):
        result = infer_capabilities(
            _make_tool(
                "run_code",
                "Execute arbitrary code",
            )
        )
        assert result.code_execution
        # Conservative: code exec implies filesystem + network
        assert result.network_access
        assert result.filesystem_read
        assert result.filesystem_write

    def test_bash_command(self):
        result = infer_capabilities(
            _make_tool(
                "bash",
                "Run a bash command",
            )
        )
        assert result.code_execution

    def test_execute_script(self):
        result = infer_capabilities(
            _make_tool(
                "execute_script",
                "Execute a script file",
            )
        )
        assert result.code_execution

    def test_eval(self):
        result = infer_capabilities(
            _make_tool(
                "eval",
                "Evaluate an expression",
            )
        )
        assert result.code_execution


# --- Credential access ---


class TestCredentialAccess:
    def test_get_password(self):
        result = infer_capabilities(
            _make_tool(
                "get_password",
                "Retrieve a stored password",
            )
        )
        assert result.credential_access

    def test_rotate_token(self):
        result = infer_capabilities(
            _make_tool(
                "rotate_token",
                "Rotate an API token",
            )
        )
        assert result.credential_access

    def test_auth_in_name(self):
        result = infer_capabilities(
            _make_tool(
                "auth_login",
                "Authenticate and log in",
            )
        )
        assert result.credential_access


# --- User identity impact ---


class TestUserIdentityImpact:
    def test_create_user(self):
        result = infer_capabilities(
            _make_tool(
                "create_user",
                "Create a new user account",
            )
        )
        assert result.user_identity_impact

    def test_delete_user(self):
        result = infer_capabilities(
            _make_tool(
                "delete_user",
                "Delete a user account",
            )
        )
        assert result.user_identity_impact
        assert result.filesystem_delete  # "delete" also triggers fs_delete

    def test_change_role(self):
        result = infer_capabilities(
            _make_tool(
                "change_role",
                "Change a user's role",
            )
        )
        assert result.user_identity_impact

    def test_grant_access(self):
        result = infer_capabilities(
            _make_tool(
                "grant_access",
                "Grant access to a resource",
            )
        )
        assert result.user_identity_impact

    def test_sudo(self):
        result = infer_capabilities(
            _make_tool(
                "sudo_exec",
                "Execute with elevated privileges",
            )
        )
        assert result.user_identity_impact


# --- Conservative inference (ambiguous cases) ---


class TestConservativeInference:
    """Ambiguous tools should be over-flagged rather than under-flagged."""

    def test_send_in_name_flags_multiple(self):
        """'send' should flag network + outbound + messaging."""
        result = infer_capabilities(
            _make_tool(
                "send_report",
                "Send the report",
            )
        )
        assert result.network_access
        assert result.outbound_data_transfer
        assert result.external_messaging

    def test_description_triggers_even_if_name_benign(self):
        """A benign name with a revealing description should still flag."""
        result = infer_capabilities(
            _make_tool(
                "process",
                "Fetch data from the remote API and post results",
            )
        )
        assert result.network_access

    def test_exec_flags_broad_capabilities(self):
        """Code execution conservatively implies filesystem + network."""
        result = infer_capabilities(
            _make_tool(
                "exec_command",
                "Execute a system command",
            )
        )
        assert result.code_execution
        assert result.network_access
        assert result.filesystem_read
        assert result.filesystem_write

    def test_multiple_capabilities_stack(self):
        """A tool that matches multiple patterns gets all flags."""
        result = infer_capabilities(
            _make_tool(
                "send_email_with_attachment",
                "Upload file and send email notification to webhook endpoint",
                {
                    "type": "object",
                    "properties": {
                        "url": {"type": "string"},
                        "password": {"type": "string"},
                        "body": {"type": "string"},
                    },
                },
            )
        )
        assert result.network_access
        assert result.outbound_data_transfer
        assert result.external_messaging
        assert result.credential_access


# --- Edge cases ---


class TestEdgeCases:
    def test_empty_description(self):
        """Empty description should not crash, only name is analyzed."""
        result = infer_capabilities(_make_tool("read_file", ""))
        assert result.filesystem_read

    def test_empty_schema(self):
        """Empty input_schema should not crash."""
        result = infer_capabilities(
            _make_tool(
                "get_time",
                "Get current time",
                {},
            )
        )
        assert not result.is_dangerous

    def test_none_properties_in_schema(self):
        """Schema with no 'properties' key should be handled gracefully."""
        result = infer_capabilities(
            _make_tool(
                "get_info",
                "Get info",
                {"type": "object"},
            )
        )
        assert not result.is_dangerous

    def test_malformed_properties_in_schema(self):
        """Non-dict properties should be handled gracefully."""
        result = infer_capabilities(
            _make_tool(
                "get_info",
                "Get info",
                {"type": "object", "properties": "invalid"},
            )
        )
        assert not result.is_dangerous

    def test_single_char_name(self):
        """Extremely short tool names shouldn't crash."""
        result = infer_capabilities(_make_tool("x", "y"))
        assert result.source == CapabilitySource.INFERRED

    def test_underscore_boundaries(self):
        """Patterns should match on underscore boundaries in tool names."""
        result = infer_capabilities(
            _make_tool(
                "my_fetch_data",
                "Fetches data from a source",
            )
        )
        assert result.network_access

    def test_manifest_is_frozen(self):
        """Returned manifest should be immutable (frozen pydantic model)."""
        result = infer_capabilities(_make_tool("get_time", "Get time"))
        with pytest.raises(Exception):
            result.network_access = True  # type: ignore[misc]


# --- Helper function tests ---


class TestExtractSchemaFields:
    def test_extracts_property_names(self):
        schema = {
            "type": "object",
            "properties": {
                "URL": {"type": "string"},
                "Query": {"type": "string"},
            },
        }
        fields = _extract_schema_fields(schema)
        assert "url" in fields
        assert "query" in fields

    def test_empty_schema(self):
        assert _extract_schema_fields({}) == []

    def test_no_properties_key(self):
        assert _extract_schema_fields({"type": "object"}) == []

    def test_invalid_properties_type(self):
        assert _extract_schema_fields({"properties": [1, 2, 3]}) == []


# --- is_dangerous and convenience properties ---


class TestManifestProperties:
    def test_safe_tool_not_dangerous(self):
        result = infer_capabilities(_make_tool("get_time", "Get the current time"))
        assert not result.is_dangerous

    def test_outbound_is_dangerous(self):
        result = infer_capabilities(_make_tool("send_data", "Send data externally"))
        assert result.is_dangerous

    def test_has_outbound_capability(self):
        result = infer_capabilities(_make_tool("post_webhook", "Post to webhook"))
        assert result.has_outbound_capability

    def test_has_filesystem_capability(self):
        result = infer_capabilities(_make_tool("read_file", "Read a file"))
        assert result.has_filesystem_capability
        assert not result.is_dangerous  # read-only is not dangerous
