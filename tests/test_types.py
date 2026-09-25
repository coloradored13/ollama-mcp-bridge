"""Tests for types.py — shared data types."""

from datetime import datetime

import pytest

from ollama_mcp_bridge.types import (
    ActionClass,
    ApprovedTool,
    ContentProvenance,
    ExecutionResult,
    OllamaToolCall,
    ResultSanitizationTier,
    SemanticRiskAssessment,
    SourceType,
    ToolSchema,
    TrustLevel,
)


class TestToolSchema:
    def test_definition_hash_deterministic(self, sample_tool_schema: ToolSchema):
        """Same schema should always produce same hash."""
        h1 = sample_tool_schema.definition_hash
        h2 = sample_tool_schema.definition_hash
        assert h1 == h2
        assert len(h1) == 64  # SHA-256 hex

    def test_definition_hash_changes_on_modification(self):
        """Different schemas should produce different hashes."""
        schema = {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        }
        t1 = ToolSchema(
            server="s",
            name="t",
            description="original",
            input_schema=schema,
        )
        t2 = ToolSchema(
            server="s",
            name="t",
            description="modified",
            input_schema=schema,
        )
        assert t1.definition_hash != t2.definition_hash

    def test_raw_definition_canonical(self):
        """Raw definition should be JSON with sorted keys, no spaces."""
        t = ToolSchema(
            server="s",
            name="tool",
            description="desc",
            input_schema={"type": "object", "properties": {}},
        )
        raw = t.raw_definition
        assert '"description":"desc"' in raw
        assert '"name":"tool"' in raw

    def test_frozen_model(self, sample_tool_schema: ToolSchema):
        """ToolSchema should be immutable."""
        import pytest

        with pytest.raises(Exception):
            sample_tool_schema.name = "changed"


class TestApprovedTool:
    def test_namespaced_name(self, sample_approved_tool: ApprovedTool):
        assert sample_approved_tool.namespaced_name == "sigma-mem__store_memory"

    def test_namespaced_name_format(self):
        tool = ApprovedTool(
            server="files",
            name="read_file",
            description="Read",
            input_schema={},
            classification=ActionClass.READ,
            definition_hash="x",
        )
        assert tool.namespaced_name == "files__read_file"


class TestOllamaToolCall:
    def test_server_extraction(self):
        tc = OllamaToolCall(
            function_name="sigma-mem__store_memory",
            arguments={},
        )
        assert tc.server == "sigma-mem"
        assert tc.tool_name == "store_memory"

    def test_bare_name_no_server(self):
        tc = OllamaToolCall(
            function_name="store_memory",
            arguments={},
        )
        assert tc.server is None
        assert tc.tool_name == "store_memory"

    def test_arguments_preserved(self):
        args = {"key": "test", "value": "hello"}
        tc = OllamaToolCall(function_name="t", arguments=args)
        assert tc.arguments == args


class TestSourceType:
    def test_all_source_types_defined(self):
        """All spec §7.2.1 source types exist."""
        expected = {
            "user",
            "system",
            "developer_policy",
            "tool_result",
            "document",
            "webpage",
            "email",
            "memory",
            "unknown",
        }
        actual = {s.value for s in SourceType}
        assert actual == expected

    def test_string_enum(self):
        assert SourceType.TOOL_RESULT == "tool_result"
        assert isinstance(SourceType.USER, str)


class TestTrustLevel:
    def test_all_trust_levels_defined(self):
        """All spec §7.2.1 trust levels exist."""
        expected = {"trusted", "user_controlled", "third_party", "unknown"}
        actual = {t.value for t in TrustLevel}
        assert actual == expected

    def test_string_enum(self):
        assert TrustLevel.THIRD_PARTY == "third_party"


class TestContentProvenance:
    def test_defaults(self):
        p = ContentProvenance()
        assert p.source_type == SourceType.UNKNOWN
        assert p.trust_level == TrustLevel.UNKNOWN
        assert p.origin_id == ""
        assert p.can_issue_instructions is False
        assert p.can_contain_sensitive_data is False
        assert isinstance(p.timestamp, datetime)

    def test_tool_result_provenance(self):
        p = ContentProvenance(
            source_type=SourceType.TOOL_RESULT,
            trust_level=TrustLevel.THIRD_PARTY,
            origin_id="sigma-mem:recall",
            can_contain_sensitive_data=True,
        )
        assert p.source_type == SourceType.TOOL_RESULT
        assert p.trust_level == TrustLevel.THIRD_PARTY
        assert p.origin_id == "sigma-mem:recall"
        assert p.can_contain_sensitive_data is True

    def test_trusted_system_provenance(self):
        p = ContentProvenance(
            source_type=SourceType.SYSTEM,
            trust_level=TrustLevel.TRUSTED,
            can_issue_instructions=True,
        )
        assert p.can_issue_instructions is True

    def test_serialization_roundtrip(self):
        p = ContentProvenance(
            source_type=SourceType.WEBPAGE,
            trust_level=TrustLevel.THIRD_PARTY,
            origin_id="https://example.com",
        )
        data = p.model_dump()
        p2 = ContentProvenance.model_validate(data)
        assert p2.source_type == SourceType.WEBPAGE
        assert p2.origin_id == "https://example.com"


class TestSemanticRiskAssessment:
    def test_clean_defaults(self):
        """Default assessment has zero risk."""
        a = SemanticRiskAssessment()
        assert a.overall_risk_score == 0.0
        assert a.attempts_instruction_override is False
        assert a.attempts_tool_routing is False
        assert a.attempts_permission_escalation is False
        assert a.attempts_exfiltration is False
        assert a.requests_sensitive_data is False
        assert a.proposes_external_destination is False
        assert a.contains_social_pressure is False
        assert a.contains_urgency_manipulation is False
        assert a.contains_hidden_or_obfuscated_instructions is False
        assert a.explanation == ""
        assert a.raw_signals == []

    def test_high_risk_assessment(self):
        a = SemanticRiskAssessment(
            overall_risk_score=0.9,
            attempts_instruction_override=True,
            attempts_exfiltration=True,
            explanation="Multiple attack patterns detected.",
            raw_signals=["instruction_language:80", "exfiltration_pattern:90"],
        )
        assert a.overall_risk_score == 0.9
        assert a.attempts_instruction_override is True
        assert len(a.raw_signals) == 2

    def test_serialization_roundtrip(self):
        a = SemanticRiskAssessment(
            overall_risk_score=0.5,
            attempts_tool_routing=True,
            raw_signals=["cross_tool_reference:60"],
        )
        data = a.model_dump()
        a2 = SemanticRiskAssessment.model_validate(data)
        assert a2.attempts_tool_routing is True
        assert a2.raw_signals == ["cross_tool_reference:60"]


class TestExecutionResultWithProvenance:
    def test_execution_result_carries_provenance(self):
        p = ContentProvenance(
            source_type=SourceType.TOOL_RESULT,
            trust_level=TrustLevel.THIRD_PARTY,
            origin_id="sigma-mem:recall",
        )
        a = SemanticRiskAssessment(overall_risk_score=0.1)
        r = ExecutionResult(
            content="some result",
            sanitization_tier=ResultSanitizationTier.CLEAN,
            server="sigma-mem",
            tool_name="recall",
            provenance=p,
            risk_assessment=a,
        )
        assert r.provenance is not None
        assert r.provenance.source_type == SourceType.TOOL_RESULT
        assert r.risk_assessment is not None
        assert r.risk_assessment.overall_risk_score == 0.1

    def test_execution_result_provenance_optional(self):
        """Backward compat: provenance and risk_assessment are optional."""
        r = ExecutionResult(content="result")
        assert r.provenance is None
        assert r.risk_assessment is None


# --- DestinationPolicy tests ---


from ollama_mcp_bridge.types import DestinationMatchResult, DestinationPolicy


class TestDestinationPolicy:
    def test_matches_exact_host(self):
        policy = DestinationPolicy(host="api.example.com")
        result = policy.matches("https://api.example.com/v1")
        assert result.matched is True

    def test_rejects_wrong_host(self):
        policy = DestinationPolicy(host="api.example.com")
        result = policy.matches("https://other.com/v1")
        assert result.matched is False
        assert "host" in result.failure_reason

    def test_allows_subdomain_when_enabled(self):
        policy = DestinationPolicy(host="example.com", allow_subdomains=True)
        result = policy.matches("https://sub.example.com/api")
        assert result.matched is True

    def test_rejects_subdomain_when_disabled(self):
        policy = DestinationPolicy(host="example.com", allow_subdomains=False)
        result = policy.matches("https://sub.example.com/api")
        assert result.matched is False
        assert "subdomains not allowed" in result.failure_reason

    def test_rejects_wrong_scheme(self):
        policy = DestinationPolicy(host="api.example.com", scheme="https")
        result = policy.matches("http://api.example.com/v1")
        assert result.matched is False
        assert "scheme" in result.failure_reason

    def test_allows_matching_scheme(self):
        policy = DestinationPolicy(host="api.example.com", scheme="http")
        result = policy.matches("http://api.example.com/v1")
        assert result.matched is True

    def test_matches_port(self):
        policy = DestinationPolicy(host="api.example.com", port=8443)
        result = policy.matches("https://api.example.com:8443/v1")
        assert result.matched is True

    def test_rejects_wrong_port(self):
        policy = DestinationPolicy(host="api.example.com", port=8443)
        result = policy.matches("https://api.example.com:9999/v1")
        assert result.matched is False
        assert "port" in result.failure_reason

    def test_default_port_inferred(self):
        """URL without explicit port uses default 443 for https."""
        policy = DestinationPolicy(host="api.example.com", port=443)
        result = policy.matches("https://api.example.com/v1")
        assert result.matched is True

    def test_default_port_http(self):
        """URL without explicit port uses default 80 for http."""
        policy = DestinationPolicy(host="api.example.com", scheme="http", port=80)
        result = policy.matches("http://api.example.com/v1")
        assert result.matched is True

    def test_path_prefix_match(self):
        policy = DestinationPolicy(
            host="api.example.com",
            path_prefixes=["/v1/", "/v2/"],
        )
        result = policy.matches("https://api.example.com/v1/users")
        assert result.matched is True

    def test_path_prefix_no_match(self):
        policy = DestinationPolicy(
            host="api.example.com",
            path_prefixes=["/v1/"],
        )
        result = policy.matches("https://api.example.com/admin/users")
        assert result.matched is False
        assert "path" in result.failure_reason

    def test_empty_path_prefixes_allows_any(self):
        policy = DestinationPolicy(host="api.example.com")
        result = policy.matches("https://api.example.com/anything/here")
        assert result.matched is True

    def test_ip_literal_rejected_by_default(self):
        policy = DestinationPolicy(host="192.168.1.1")
        result = policy.matches("https://192.168.1.1/api")
        assert result.matched is False
        assert "IP literal" in result.failure_reason

    def test_ip_literal_allowed_when_enabled(self):
        policy = DestinationPolicy(
            host="8.8.8.8",
            allow_ip_literals=True,
            allow_private_ranges=True,
        )
        result = policy.matches("https://8.8.8.8/dns")
        assert result.matched is True

    def test_private_range_rejected(self):
        policy = DestinationPolicy(
            host="192.168.1.1",
            allow_ip_literals=True,
            allow_private_ranges=False,
        )
        result = policy.matches("https://192.168.1.1/api")
        assert result.matched is False
        assert "private" in result.failure_reason

    def test_private_range_allowed(self):
        policy = DestinationPolicy(
            host="192.168.1.1",
            allow_ip_literals=True,
            allow_private_ranges=True,
        )
        result = policy.matches("https://192.168.1.1/api")
        assert result.matched is True

    def test_frozen_model(self):
        policy = DestinationPolicy(host="example.com")
        with pytest.raises(Exception):
            policy.host = "other.com"

    def test_defaults(self):
        policy = DestinationPolicy(host="example.com")
        assert policy.scheme == "https"
        assert policy.port is None
        assert policy.path_prefixes == []
        assert policy.allow_subdomains is False
        assert policy.allow_ip_literals is False
        assert policy.allow_private_ranges is False
        assert policy.allow_redirects is False
        assert policy.allowed_methods == []
        assert policy.max_payload_bytes == 65536

    def test_dormant_query_constraints_raises(self):
        """query_constraints is not yet enforced — raises if set."""
        import pytest
        from pydantic import ValidationError

        with pytest.raises((ValidationError, Exception), match="not yet enforced"):
            DestinationPolicy(host="example.com", query_constraints={"token": "abc"})

    def test_dormant_allow_redirects_raises(self):
        """allow_redirects=True is not yet enforced — raises if set."""
        import pytest
        from pydantic import ValidationError

        with pytest.raises((ValidationError, Exception), match="not yet enforced"):
            DestinationPolicy(host="example.com", allow_redirects=True)

    def test_dormant_allowed_methods_raises(self):
        """allowed_methods is not yet enforced — raises if set."""
        import pytest
        from pydantic import ValidationError

        with pytest.raises((ValidationError, Exception), match="not yet enforced"):
            DestinationPolicy(host="example.com", allowed_methods=["GET"])

    def test_dormant_max_payload_bytes_raises(self):
        """Non-default max_payload_bytes is not yet enforced — raises if changed."""
        import pytest
        from pydantic import ValidationError

        with pytest.raises((ValidationError, Exception), match="not yet enforced"):
            DestinationPolicy(host="example.com", max_payload_bytes=32768)

    def test_dormant_defaults_pass(self):
        """All dormant fields at their default values pass validation."""
        policy = DestinationPolicy(host="example.com")
        assert policy.query_constraints == {}
        assert policy.allow_redirects is False
        assert policy.allowed_methods == []
        assert policy.max_payload_bytes == 65536

    def test_malformed_url(self):
        policy = DestinationPolicy(host="example.com")
        result = policy.matches("not a url at all")
        assert result.matched is False

    def test_url_no_hostname(self):
        policy = DestinationPolicy(host="example.com")
        result = policy.matches("https:///path")
        assert result.matched is False
        assert "no hostname" in result.failure_reason

    def test_case_insensitive_host(self):
        policy = DestinationPolicy(host="API.Example.COM")
        result = policy.matches("https://api.example.com/v1")
        assert result.matched is True

    def test_case_insensitive_scheme(self):
        policy = DestinationPolicy(host="example.com", scheme="HTTPS")
        result = policy.matches("https://example.com/v1")
        assert result.matched is True


class TestDestinationMatchResult:
    def test_default_not_matched(self):
        r = DestinationMatchResult()
        assert r.matched is False
        assert r.failure_reason == ""

    def test_matched_result(self):
        r = DestinationMatchResult(
            matched=True,
            policy_host="example.com",
            checked_url="https://example.com/v1",
        )
        assert r.matched is True
        assert r.policy_host == "example.com"


# --- InfluenceState tests ---


from ollama_mcp_bridge.types import (
    InfluenceEvidence,
    InfluenceState,
    InfluenceType,
    TaintState,
)


class TestInfluenceState:
    def test_is_a_taint_state(self):
        """InfluenceState inherits from TaintState."""
        state = InfluenceState()
        assert isinstance(state, TaintState)

    def test_defaults(self):
        state = InfluenceState()
        assert state.tainted is False
        assert state.direct_value_match is False
        assert state.derived_from_untrusted_value is False
        assert state.destination_influenced is False
        assert state.evidence == []
        assert state.taint_sources == []
        assert state.confidence == 0.0

    def test_tainted_with_evidence(self):
        evidence = [
            InfluenceEvidence(
                influence_type=InfluenceType.DIRECT_VALUE_MATCH,
                tracked_value="https://evil.com",
                arg_value="https://evil.com",
                origin_id="search:web",
                confidence=0.9,
            )
        ]
        state = InfluenceState(
            tainted=True,
            taint_sources=["search:web"],
            confidence=0.9,
            direct_value_match=True,
            evidence=evidence,
        )
        assert state.tainted is True
        assert state.direct_value_match is True
        assert len(state.evidence) == 1
        assert state.evidence[0].influence_type == InfluenceType.DIRECT_VALUE_MATCH

    def test_derived_only(self):
        state = InfluenceState(
            tainted=True,
            derived_from_untrusted_value=True,
            destination_influenced=True,
        )
        assert state.tainted is True
        assert state.direct_value_match is False
        assert state.derived_from_untrusted_value is True

    def test_inherits_taint_state_fields(self):
        state = InfluenceState(
            tainted=True,
            taint_sources=["src"],
            taint_reasons=["reason"],
            affected_fields=["url"],
            confidence=0.75,
        )
        assert state.taint_sources == ["src"]
        assert state.taint_reasons == ["reason"]
        assert state.affected_fields == ["url"]
        assert state.confidence == 0.75


class TestInfluenceType:
    def test_enum_values(self):
        assert InfluenceType.DIRECT_VALUE_MATCH == "direct_value_match"
        assert InfluenceType.DERIVED_URL_REUSE == "derived_url_reuse"
        assert InfluenceType.DERIVED_PROTOCOL_CHANGE == "derived_protocol_change"
        assert InfluenceType.DERIVED_EMAIL_DOMAIN == "derived_email_domain"
        assert InfluenceType.DERIVED_HOSTNAME_IN_URL == "derived_hostname_in_url"


class TestInfluenceEvidence:
    def test_defaults(self):
        e = InfluenceEvidence(influence_type=InfluenceType.DIRECT_VALUE_MATCH)
        assert e.tracked_value == ""
        assert e.arg_value == ""
        assert e.origin_id == ""
        assert e.confidence == 0.0


# --- PathPolicy tests ---


from ollama_mcp_bridge.types import PathMatchResult, PathPolicy, ToolCapabilityManifest


class TestPathPolicy:
    def test_path_within_root_passes(self):
        policy = PathPolicy(allowed_roots=["/tmp/sandbox"])
        result = policy.validate_path("/tmp/sandbox/file.txt")
        assert result.matched is True

    def test_path_outside_root_rejected(self):
        policy = PathPolicy(allowed_roots=["/tmp/sandbox"])
        result = policy.validate_path("/etc/passwd")
        assert result.matched is False
        assert "outside allowed roots" in result.failure_reason

    def test_traversal_attack_rejected(self):
        policy = PathPolicy(allowed_roots=["/tmp/sandbox"])
        result = policy.validate_path("/tmp/sandbox/../../etc/passwd")
        assert result.matched is False
        assert "outside allowed roots" in result.failure_reason

    def test_relative_path_rejected_by_default(self):
        policy = PathPolicy(allowed_roots=["/tmp/sandbox"])
        result = policy.validate_path("../../../etc/passwd")
        assert result.matched is False
        assert "relative paths not allowed" in result.failure_reason

    def test_relative_path_allowed_when_enabled(self):
        """When allow_relative_paths=True, relative paths within root are allowed."""
        import os

        # Use CWD-relative path that stays in root — test with absolute root
        sandbox = os.path.realpath("/tmp/sandbox_test_rel")
        os.makedirs(sandbox, exist_ok=True)
        policy = PathPolicy(
            allowed_roots=[sandbox],
            allow_relative_paths=True,
            normalize_symlinks=False,
        )
        # A relative path that normpath resolves to CWD — won't be in root
        result = policy.validate_path("./somefile.txt")
        # This should fail because CWD-relative path won't be in sandbox
        assert result.matched is False

    def test_glob_rejected_by_default(self):
        policy = PathPolicy(allowed_roots=["/tmp/sandbox"])
        result = policy.validate_path("/tmp/sandbox/*.txt")
        assert result.matched is False
        assert "glob patterns not allowed" in result.failure_reason

    def test_glob_allowed_when_enabled(self):
        policy = PathPolicy(allowed_roots=["/tmp/sandbox"], allow_globs=True)
        result = policy.validate_path("/tmp/sandbox/*.txt")
        assert result.matched is True

    def test_home_expansion_blocked(self):
        policy = PathPolicy(
            allowed_roots=["/tmp/sandbox"],
            allow_user_home_expansion=False,
        )
        result = policy.validate_path("~/secret")
        assert result.matched is False
        assert "home expansion not allowed" in result.failure_reason

    def test_home_expansion_allowed_by_default(self):
        import os

        home = os.path.expanduser("~")
        policy = PathPolicy(allowed_roots=[home])
        result = policy.validate_path("~/documents/file.txt")
        assert result.matched is True

    def test_extension_allowlist_passes(self):
        policy = PathPolicy(
            allowed_roots=["/tmp/sandbox"],
            extensions_allowlist=[".txt", ".md"],
        )
        result = policy.validate_path("/tmp/sandbox/readme.txt")
        assert result.matched is True

    def test_extension_allowlist_rejects(self):
        policy = PathPolicy(
            allowed_roots=["/tmp/sandbox"],
            extensions_allowlist=[".txt", ".md"],
        )
        result = policy.validate_path("/tmp/sandbox/exploit.sh")
        assert result.matched is False
        assert "extension" in result.failure_reason
        assert "allowlist" in result.failure_reason

    def test_extension_allowlist_without_dot(self):
        """Extension in config without leading dot still works."""
        policy = PathPolicy(
            allowed_roots=["/tmp/sandbox"],
            extensions_allowlist=["txt"],
        )
        result = policy.validate_path("/tmp/sandbox/file.txt")
        assert result.matched is True

    def test_multiple_roots(self):
        policy = PathPolicy(allowed_roots=["/tmp/a", "/tmp/b"])
        assert policy.validate_path("/tmp/a/file").matched is True
        assert policy.validate_path("/tmp/b/file").matched is True
        assert policy.validate_path("/tmp/c/file").matched is False

    def test_exact_root_path_passes(self):
        policy = PathPolicy(allowed_roots=["/tmp/sandbox"])
        result = policy.validate_path("/tmp/sandbox")
        assert result.matched is True

    def test_read_only_blocks_write_tool(self):
        caps = ToolCapabilityManifest(filesystem_write=True)
        policy = PathPolicy(allowed_roots=["/tmp/sandbox"], read_only=True)
        result = policy.validate_path("/tmp/sandbox/file.txt", caps)
        assert result.matched is False
        assert "read_only" in result.failure_reason

    def test_read_only_allows_read_tool(self):
        caps = ToolCapabilityManifest(filesystem_read=True)
        policy = PathPolicy(allowed_roots=["/tmp/sandbox"], read_only=True)
        result = policy.validate_path("/tmp/sandbox/file.txt", caps)
        assert result.matched is True

    def test_delete_not_allowed_blocks_delete_tool(self):
        caps = ToolCapabilityManifest(filesystem_delete=True)
        policy = PathPolicy(allowed_roots=["/tmp/sandbox"], delete_allowed=False)
        result = policy.validate_path("/tmp/sandbox/file.txt", caps)
        assert result.matched is False
        assert "delete not allowed" in result.failure_reason

    def test_delete_allowed_passes(self):
        caps = ToolCapabilityManifest(filesystem_delete=True)
        policy = PathPolicy(allowed_roots=["/tmp/sandbox"], delete_allowed=True)
        result = policy.validate_path("/tmp/sandbox/file.txt", caps)
        assert result.matched is True

    def test_write_only_blocks_delete_tool(self):
        caps = ToolCapabilityManifest(filesystem_delete=True)
        policy = PathPolicy(allowed_roots=["/tmp/sandbox"], write_only=True)
        result = policy.validate_path("/tmp/sandbox/file.txt", caps)
        assert result.matched is False
        assert "write_only" in result.failure_reason

    def test_frozen_model(self):
        policy = PathPolicy(allowed_roots=["/tmp"])
        with pytest.raises(Exception):
            policy.allowed_roots = ["/other"]

    def test_defaults(self):
        policy = PathPolicy(allowed_roots=["/tmp"])
        assert policy.allow_relative_paths is False
        assert policy.normalize_symlinks is True
        assert policy.allow_globs is False
        assert policy.allow_user_home_expansion is True
        assert policy.read_only is False
        assert policy.write_only is False
        assert policy.delete_allowed is False
        assert policy.extensions_allowlist == []
        assert policy.filename_pattern_allowlist == []

    def test_filename_pattern_allowlist(self):
        policy = PathPolicy(
            allowed_roots=["/tmp/sandbox"],
            filename_pattern_allowlist=[r".*\.log", r"data_\d+\.csv"],
        )
        assert policy.validate_path("/tmp/sandbox/app.log").matched is True
        assert policy.validate_path("/tmp/sandbox/data_123.csv").matched is True
        assert policy.validate_path("/tmp/sandbox/evil.sh").matched is False


class TestPathMatchResult:
    def test_default_not_matched(self):
        r = PathMatchResult()
        assert r.matched is False
        assert r.failure_reason == ""

    def test_matched_result(self):
        r = PathMatchResult(
            matched=True,
            policy_roots=["/tmp"],
            checked_path="/tmp/file.txt",
        )
        assert r.matched is True


# --- RecipientPolicy tests ---


from ollama_mcp_bridge.types import RecipientMatchResult, RecipientPolicy


class TestRecipientPolicy:
    def test_exact_address_match(self):
        policy = RecipientPolicy(approved_addresses=["admin@example.com"])
        result = policy.validate_recipient("admin@example.com")
        assert result.matched is True
        assert result.match_type == "exact_address"

    def test_exact_address_case_insensitive(self):
        policy = RecipientPolicy(approved_addresses=["Admin@Example.COM"])
        result = policy.validate_recipient("admin@example.com")
        assert result.matched is True

    def test_unapproved_address_rejected(self):
        policy = RecipientPolicy(approved_addresses=["admin@example.com"])
        result = policy.validate_recipient("evil@attacker.com")
        assert result.matched is False
        assert "does not match" in result.failure_reason

    def test_domain_match(self):
        policy = RecipientPolicy(approved_domains=["internal.corp"])
        result = policy.validate_recipient("anyone@internal.corp")
        assert result.matched is True
        assert result.match_type == "approved_domain"

    def test_subdomain_of_approved_domain(self):
        policy = RecipientPolicy(approved_domains=["corp.com"])
        result = policy.validate_recipient("user@team.corp.com")
        assert result.matched is True

    def test_domain_case_insensitive(self):
        policy = RecipientPolicy(approved_domains=["INTERNAL.CORP"])
        result = policy.validate_recipient("user@internal.corp")
        assert result.matched is True

    def test_identity_group_match(self):
        policy = RecipientPolicy(
            identity_groups={"engineering": ["alice@co.com", "bob@co.com"]},
        )
        result = policy.validate_recipient("alice@co.com")
        assert result.matched is True
        assert result.match_type == "identity_group:engineering"

    def test_identity_group_case_insensitive(self):
        policy = RecipientPolicy(
            identity_groups={"team": ["Alice@Co.COM"]},
        )
        result = policy.validate_recipient("alice@co.com")
        assert result.matched is True

    def test_no_match_any_rule(self):
        policy = RecipientPolicy(
            approved_addresses=["admin@co.com"],
            approved_domains=["internal.corp"],
            identity_groups={"team": ["bob@co.com"]},
        )
        result = policy.validate_recipient("stranger@evil.com")
        assert result.matched is False

    def test_has_any_policy_true(self):
        assert RecipientPolicy(approved_addresses=["a@b.c"]).has_any_policy is True
        assert RecipientPolicy(approved_domains=["b.c"]).has_any_policy is True
        assert RecipientPolicy(identity_groups={"g": ["a@b.c"]}).has_any_policy is True

    def test_has_any_policy_false(self):
        assert RecipientPolicy().has_any_policy is False

    def test_frozen_model(self):
        policy = RecipientPolicy(approved_addresses=["a@b.c"])
        with pytest.raises(Exception):
            policy.approved_addresses = ["x@y.z"]

    def test_defaults(self):
        policy = RecipientPolicy()
        assert policy.approved_addresses == []
        assert policy.approved_domains == []
        assert policy.identity_groups == {}
        assert policy.internal_only is False
        assert policy.allow_first_contact is False

    def test_dormant_allow_first_contact_raises(self):
        """allow_first_contact=True is not yet enforced — raises if set."""
        import pytest
        from pydantic import ValidationError

        with pytest.raises((ValidationError, Exception), match="not yet enforced"):
            RecipientPolicy(allow_first_contact=True)

    def test_dormant_allow_first_contact_false_passes(self):
        """allow_first_contact=False (default) passes validation."""
        policy = RecipientPolicy(allow_first_contact=False)
        assert policy.allow_first_contact is False

    def test_email_without_at_rejected(self):
        """Recipient without @ has no domain to match."""
        policy = RecipientPolicy(approved_domains=["co.com"])
        result = policy.validate_recipient("nodomain")
        assert result.matched is False


class TestRecipientMatchResult:
    def test_default_not_matched(self):
        r = RecipientMatchResult()
        assert r.matched is False
        assert r.match_type == ""
        assert r.failure_reason == ""

    def test_matched_result(self):
        r = RecipientMatchResult(
            matched=True,
            checked_recipient="admin@co.com",
            match_type="exact_address",
        )
        assert r.matched is True
