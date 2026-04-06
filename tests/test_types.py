"""Tests for types.py — shared data types."""

from datetime import datetime

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
            server="s", name="t", description="original",
            input_schema=schema,
        )
        t2 = ToolSchema(
            server="s", name="t", description="modified",
            input_schema=schema,
        )
        assert t1.definition_hash != t2.definition_hash

    def test_raw_definition_canonical(self):
        """Raw definition should be JSON with sorted keys, no spaces."""
        t = ToolSchema(
            server="s", name="tool", description="desc",
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
            server="files", name="read_file", description="Read",
            input_schema={}, classification=ActionClass.READ,
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
        expected = {"user", "system", "developer_policy", "tool_result",
                    "document", "webpage", "email", "memory", "unknown"}
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
