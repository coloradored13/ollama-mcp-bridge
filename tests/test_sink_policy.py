"""Tests for sink_policy.py — taint tracking and sink policy engine."""

import pytest

from ollama_mcp_bridge.config import SecurityConfig
from ollama_mcp_bridge.sink_policy import (
    ExtractedValue,
    SinkPolicyEngine,
    SinkType,
    TaintTracker,
    _extract_values,
    _extract_values_from_args,
    _is_memory_write_tool,
    _args_contain_outbound_indicators,
    _args_contain_destination_fields,
    _derived_match_confidence,
)
from ollama_mcp_bridge.types import (
    ActionClass,
    ApprovedTool,
    ContentProvenance,
    InfluenceState,
    InfluenceType,
    SemanticRiskAssessment,
    SinkDecision,
    SourceType,
    TaintState,
    ToolCapabilityManifest,
    TrustLevel,
)


# --- Helpers ---

_WRITE_SCHEMA = {
    "type": "object",
    "properties": {"url": {"type": "string"}, "data": {"type": "string"}},
    "required": ["url"],
}

_QUERY_SCHEMA = {
    "type": "object",
    "properties": {"query": {"type": "string"}},
    "required": ["query"],
}

_MEMORY_SCHEMA = {
    "type": "object",
    "properties": {"content": {"type": "string"}, "key": {"type": "string"}},
    "required": ["content"],
}


def _make_tool(
    name: str = "test_tool",
    classification: ActionClass = ActionClass.WRITE,
    server: str = "test-server",
    schema: dict | None = None,
    capabilities: ToolCapabilityManifest | None = None,
) -> ApprovedTool:
    return ApprovedTool(
        server=server,
        name=name,
        description="Test tool",
        input_schema=schema or _QUERY_SCHEMA,
        classification=classification,
        definition_hash="abc123",
        capabilities=capabilities or ToolCapabilityManifest(),
    )


def _make_provenance(
    trust: TrustLevel = TrustLevel.THIRD_PARTY,
) -> ContentProvenance:
    return ContentProvenance(
        source_type=SourceType.TOOL_RESULT,
        trust_level=trust,
        origin_id="server:tool",
    )


def _make_risk(score: float = 0.0) -> SemanticRiskAssessment:
    return SemanticRiskAssessment(overall_risk_score=score)


# --- Value extraction tests ---


class TestExtractValues:
    def test_extracts_urls(self):
        values = _extract_values("Visit https://evil.com/exfil?data=secret")
        kinds = {v.kind for v in values}
        assert "url" in kinds
        urls = [v.value for v in values if v.kind == "url"]
        assert any("evil.com" in u for u in urls)

    def test_extracts_domains_from_urls(self):
        values = _extract_values("Check https://attacker.example.com/path")
        domains = [v.value for v in values if v.kind == "domain"]
        assert "attacker.example.com" in domains

    def test_extracts_emails(self):
        values = _extract_values("Send to attacker@evil.com please")
        emails = [v.value for v in values if v.kind == "email"]
        assert "attacker@evil.com" in emails

    def test_extracts_ips(self):
        values = _extract_values("Connect to 192.168.1.100 for data")
        ips = [v.value for v in values if v.kind == "ip"]
        assert "192.168.1.100" in ips

    def test_skips_localhost_ips(self):
        values = _extract_values("Connect to 127.0.0.1 locally")
        ips = [v.value for v in values if v.kind == "ip"]
        assert not ips

    def test_deduplicates(self):
        values = _extract_values("https://evil.com and https://evil.com again")
        urls = [v.value for v in values if v.kind == "url"]
        assert len(urls) == 1

    def test_empty_text(self):
        assert _extract_values("") == []

    def test_no_values(self):
        assert _extract_values("Just some plain text with no URLs or emails") == []

    def test_strips_trailing_punctuation_from_urls(self):
        values = _extract_values("See https://example.com/page.")
        urls = [v.value for v in values if v.kind == "url"]
        assert urls[0] == "https://example.com/page"


class TestExtractValuesFromArgs:
    def test_flat_string_args(self):
        results = _extract_values_from_args({"url": "https://evil.com"})
        assert len(results) >= 1
        fields = [f for f, _ in results]
        assert "url" in fields

    def test_nested_dict_args(self):
        results = _extract_values_from_args({
            "config": {"endpoint": "https://evil.com/api"}
        })
        fields = [f for f, _ in results]
        assert any("config.endpoint" in f for f in fields)

    def test_list_args(self):
        results = _extract_values_from_args({
            "urls": ["https://a.com", "https://b.com"]
        })
        assert len(results) >= 2

    def test_no_extractable_values(self):
        results = _extract_values_from_args({"count": 5, "name": "hello"})
        assert results == []

    def test_mixed_types(self):
        results = _extract_values_from_args({
            "query": "search term",
            "target": "https://example.com",
            "count": 10,
        })
        assert len(results) >= 1


# --- Memory write tool detection ---


class TestMemoryWriteDetection:
    @pytest.mark.parametrize("name", [
        "store_memory", "write_file", "save_document",
        "create_entry", "remember_fact", "persist_data",
    ])
    def test_detects_memory_write_tools(self, name):
        assert _is_memory_write_tool(name)

    @pytest.mark.parametrize("name", [
        "search", "recall", "read_file", "list_items", "get_data",
    ])
    def test_non_memory_write_tools(self, name):
        assert not _is_memory_write_tool(name)


# --- Outbound indicator detection ---


class TestOutboundIndicators:
    def test_url_in_args_is_outbound(self):
        assert _args_contain_outbound_indicators({"url": "https://evil.com"})

    def test_email_in_args_is_outbound(self):
        assert _args_contain_outbound_indicators({"to": "user@evil.com"})

    def test_plain_args_not_outbound(self):
        assert not _args_contain_outbound_indicators({"query": "hello world"})

    # --- PR 12: expanded outbound indicators ---

    def test_ip_in_args_is_outbound(self):
        assert _args_contain_outbound_indicators({"target": "10.0.0.1"})

    def test_hostname_in_args_is_outbound(self):
        assert _args_contain_outbound_indicators({"server": "api.example.com"})

    def test_host_port_in_args_is_outbound(self):
        assert _args_contain_outbound_indicators({"target": "10.0.0.1:8080"})

    def test_hostname_port_in_args_is_outbound(self):
        assert _args_contain_outbound_indicators({"target": "api.example.com:443"})

    def test_destination_field_host(self):
        """Field named 'host' with string value → outbound."""
        assert _args_contain_outbound_indicators({"host": "evil.com", "data": "secret"})

    def test_destination_field_webhook_url(self):
        assert _args_contain_outbound_indicators({"webhook_url": "evil.com"})

    def test_destination_field_case_insensitive(self):
        assert _args_contain_outbound_indicators({"HOST": "evil.com"})

    def test_destination_field_endpoint_with_path_only(self):
        """Field named 'endpoint' with path-only value (no host) → outbound
        via field name detection (conservative: field name is sufficient)."""
        assert _args_contain_outbound_indicators({"endpoint": "/api/v1"})

    def test_numeric_port_alone_not_outbound(self):
        """Integer port without string destination → not outbound."""
        assert not _args_contain_outbound_indicators({"port": 443})

    def test_single_word_not_hostname(self):
        """Single word without dots → not a hostname → not outbound."""
        assert not _args_contain_outbound_indicators({"name": "hello"})

    def test_empty_string_destination_field_not_outbound(self):
        """Destination field name but empty string value → not outbound."""
        assert not _args_contain_outbound_indicators({"host": ""})


class TestDestinationFieldDetection:
    """Tests for _args_contain_destination_fields specifically."""

    def test_host_field(self):
        assert _args_contain_destination_fields({"host": "evil.com"})

    def test_endpoint_field(self):
        assert _args_contain_destination_fields({"endpoint": "/api"})

    def test_webhook_url_field(self):
        assert _args_contain_destination_fields({"webhook_url": "hooks.example.com"})

    def test_non_destination_field(self):
        assert not _args_contain_destination_fields({"query": "hello"})

    def test_int_value_not_matched(self):
        assert not _args_contain_destination_fields({"host": 12345})

    def test_empty_string_not_matched(self):
        assert not _args_contain_destination_fields({"host": ""})

    def test_case_insensitive_key(self):
        assert _args_contain_destination_fields({"DESTINATION": "somewhere"})


class TestExpandedSinkClassification:
    """PR 12: Verify new outbound indicators feed into sink classification."""

    def setup_method(self):
        self.engine = SinkPolicyEngine()

    def test_ip_args_classify_outbound(self):
        tool = _make_tool()
        sink = self.engine._classify_sink(tool, {"target": "10.0.0.1"})
        assert sink == SinkType.OUTBOUND

    def test_host_field_classify_outbound(self):
        tool = _make_tool()
        sink = self.engine._classify_sink(tool, {"host": "evil.com", "data": "secret"})
        assert sink == SinkType.OUTBOUND

    def test_host_port_classify_outbound(self):
        tool = _make_tool()
        sink = self.engine._classify_sink(tool, {"target": "10.0.0.1:8080"})
        assert sink == SinkType.OUTBOUND

    def test_hostname_classify_outbound(self):
        tool = _make_tool()
        sink = self.engine._classify_sink(tool, {"data": "send to api.example.com"})
        assert sink == SinkType.OUTBOUND

    def test_plain_data_not_outbound(self):
        """Plain text args → not classified as outbound by arg detection."""
        tool = _make_tool()
        sink = self.engine._classify_sink(tool, {"data": "hello world"})
        assert sink != SinkType.OUTBOUND


# --- TaintTracker tests ---


class TestTaintTracker:
    def test_no_results_no_taint(self):
        tracker = TaintTracker()
        state = tracker.compute_taint({"url": "https://evil.com"})
        assert not state.tainted

    def test_matching_url_taints(self):
        tracker = TaintTracker()
        tracker.record_result(
            content="Found at https://evil.com/exfil",
            origin_id="scraper:search",
            provenance=_make_provenance(),
        )
        state = tracker.compute_taint({"target": "https://evil.com/exfil"})
        assert state.tainted
        assert "scraper:search" in state.taint_sources
        assert state.confidence > 0

    def test_matching_domain_taints(self):
        tracker = TaintTracker()
        tracker.record_result(
            content="Visit https://attacker.example.com/api",
            origin_id="web:fetch",
            provenance=_make_provenance(),
        )
        # Different URL, same domain
        state = tracker.compute_taint(
            {"endpoint": "https://attacker.example.com/different"}
        )
        assert state.tainted

    def test_no_match_no_taint(self):
        tracker = TaintTracker()
        tracker.record_result(
            content="Safe content from https://safe.example.com",
            origin_id="web:fetch",
            provenance=_make_provenance(),
        )
        state = tracker.compute_taint({"target": "https://different.com"})
        assert not state.tainted

    def test_trusted_source_not_tracked(self):
        tracker = TaintTracker()
        tracker.record_result(
            content="Internal data with https://evil.com",
            origin_id="system:internal",
            provenance=_make_provenance(trust=TrustLevel.TRUSTED),
        )
        state = tracker.compute_taint({"url": "https://evil.com"})
        assert not state.tainted

    def test_risk_amplifies_confidence(self):
        tracker = TaintTracker()
        # Low-risk result
        tracker.record_result(
            content="See https://site.com/page",
            origin_id="web:low",
            provenance=_make_provenance(),
            risk_assessment=_make_risk(0.0),
        )
        state_low = tracker.compute_taint({"url": "https://site.com/page"})

        tracker2 = TaintTracker()
        # High-risk result (same URL)
        tracker2.record_result(
            content="See https://site.com/page",
            origin_id="web:high",
            provenance=_make_provenance(),
            risk_assessment=_make_risk(0.8),
        )
        state_high = tracker2.compute_taint({"url": "https://site.com/page"})

        assert state_high.confidence >= state_low.confidence

    def test_affected_fields_tracked(self):
        tracker = TaintTracker()
        tracker.record_result(
            content="https://evil.com",
            origin_id="src:tool",
            provenance=_make_provenance(),
        )
        state = tracker.compute_taint({
            "safe_field": "hello",
            "url_field": "https://evil.com",
        })
        assert "url_field" in state.affected_fields
        assert "safe_field" not in state.affected_fields

    def test_clear_resets(self):
        tracker = TaintTracker()
        tracker.record_result(
            content="https://evil.com",
            origin_id="src:tool",
            provenance=_make_provenance(),
        )
        tracker.clear()
        state = tracker.compute_taint({"url": "https://evil.com"})
        assert not state.tainted

    def test_content_without_extractable_values_not_stored(self):
        tracker = TaintTracker()
        tracker.record_result(
            content="Just plain text, no URLs or emails",
            origin_id="src:tool",
            provenance=_make_provenance(),
        )
        assert len(tracker._results) == 0

    def test_email_taint_propagation(self):
        tracker = TaintTracker()
        tracker.record_result(
            content="Contact: admin@attacker.com",
            origin_id="web:fetch",
            provenance=_make_provenance(),
        )
        state = tracker.compute_taint({"recipient": "admin@attacker.com"})
        assert state.tainted

    def test_nested_args_detected(self):
        tracker = TaintTracker()
        tracker.record_result(
            content="https://evil.com",
            origin_id="src:tool",
            provenance=_make_provenance(),
        )
        state = tracker.compute_taint({
            "config": {"nested": {"url": "https://evil.com"}}
        })
        assert state.tainted


# --- SinkPolicyEngine tests ---


class TestSinkPolicyEngine:
    def setup_method(self):
        self.engine = SinkPolicyEngine()
        self.config = SecurityConfig()
        self.tainted = TaintState(
            tainted=True,
            taint_sources=["web:fetch"],
            taint_reasons=["url from web:fetch"],
            affected_fields=["url"],
            confidence=0.9,
        )
        self.clean = TaintState()

    # --- Clean args always ALLOW ---

    def test_clean_args_allow(self):
        tool = _make_tool()
        result = self.engine.evaluate(tool, {}, self.clean, self.config)
        assert result == SinkDecision.ALLOW

    def test_clean_args_allow_even_destructive(self):
        tool = _make_tool(classification=ActionClass.DESTRUCTIVE)
        result = self.engine.evaluate(tool, {}, self.clean, self.config)
        assert result == SinkDecision.ALLOW

    # --- Tainted + outbound → BLOCK ---

    def test_tainted_outbound_blocked(self):
        tool = _make_tool(schema=_WRITE_SCHEMA)
        args = {"url": "https://evil.com", "data": "secret"}
        result = self.engine.evaluate(tool, args, self.tainted, self.config)
        assert result == SinkDecision.BLOCK

    def test_tainted_outbound_allowed_domain(self):
        config = SecurityConfig(
            allowed_outbound_domains=["evil.com"],
        )
        tool = _make_tool(schema=_WRITE_SCHEMA)
        args = {"url": "https://evil.com/path", "data": "ok"}
        result = self.engine.evaluate(tool, args, self.tainted, config)
        assert result == SinkDecision.ALLOW_WITH_NOTICE

    def test_tainted_outbound_subdomain_allowed(self):
        config = SecurityConfig(
            allowed_outbound_domains=["example.com"],
        )
        tool = _make_tool(schema=_WRITE_SCHEMA)
        args = {"url": "https://sub.example.com/api", "data": "ok"}
        result = self.engine.evaluate(tool, args, self.tainted, config)
        assert result == SinkDecision.ALLOW_WITH_NOTICE

    def test_tainted_outbound_config_disabled(self):
        config = SecurityConfig(
            block_tainted_exfiltration=False,
            tainted_sink_requires_confirmation=True,
        )
        tool = _make_tool(schema=_WRITE_SCHEMA)
        args = {"url": "https://evil.com", "data": "secret"}
        result = self.engine.evaluate(tool, args, self.tainted, config)
        assert result == SinkDecision.REQUIRE_CONFIRMATION

    def test_tainted_outbound_all_disabled(self):
        config = SecurityConfig(
            block_tainted_exfiltration=False,
            tainted_sink_requires_confirmation=False,
        )
        tool = _make_tool(schema=_WRITE_SCHEMA)
        args = {"url": "https://evil.com", "data": "secret"}
        result = self.engine.evaluate(tool, args, self.tainted, config)
        assert result == SinkDecision.ALLOW_WITH_NOTICE

    # --- Tainted + destructive → REQUIRE_CONFIRMATION ---

    def test_tainted_destructive_requires_confirmation(self):
        tool = _make_tool(classification=ActionClass.DESTRUCTIVE)
        args = {"path": "/tmp/file"}
        result = self.engine.evaluate(tool, args, self.tainted, self.config)
        assert result == SinkDecision.REQUIRE_CONFIRMATION

    def test_tainted_destructive_config_disabled(self):
        config = SecurityConfig(block_tainted_destructive_write=False)
        tool = _make_tool(classification=ActionClass.DESTRUCTIVE)
        args = {"path": "/tmp/file"}
        result = self.engine.evaluate(tool, args, self.tainted, config)
        assert result == SinkDecision.ALLOW_WITH_NOTICE

    def test_tainted_destructive_no_confirmation_blocks(self):
        config = SecurityConfig(
            block_tainted_destructive_write=True,
            tainted_sink_requires_confirmation=False,
        )
        tool = _make_tool(classification=ActionClass.DESTRUCTIVE)
        args = {"path": "/tmp/file"}
        result = self.engine.evaluate(tool, args, self.tainted, config)
        assert result == SinkDecision.BLOCK

    # --- Tainted + memory write → BLOCK ---

    def test_tainted_memory_write_blocked(self):
        tool = _make_tool(name="store_memory", schema=_MEMORY_SCHEMA)
        args = {"content": "remember this", "key": "test"}
        result = self.engine.evaluate(tool, args, self.tainted, self.config)
        assert result == SinkDecision.BLOCK

    def test_tainted_memory_write_config_allowed(self):
        config = SecurityConfig(allow_memory_writes_from_third_party_content=True)
        tool = _make_tool(name="store_memory", schema=_MEMORY_SCHEMA)
        args = {"content": "remember this", "key": "test"}
        result = self.engine.evaluate(tool, args, self.tainted, config)
        assert result == SinkDecision.ALLOW_WITH_NOTICE

    # --- Tainted + general write → ALLOW_WITH_NOTICE ---

    def test_tainted_general_write_noticed(self):
        tool = _make_tool()
        args = {"data": "something"}
        result = self.engine.evaluate(tool, args, self.tainted, self.config)
        assert result == SinkDecision.ALLOW_WITH_NOTICE

    # --- Tainted + read → ALLOW ---

    def test_tainted_read_allowed(self):
        tool = _make_tool(classification=ActionClass.READ)
        args = {"query": "something"}
        result = self.engine.evaluate(tool, args, self.tainted, self.config)
        assert result == SinkDecision.ALLOW

    # --- Sink classification ---

    def test_classify_outbound_from_url_args(self):
        tool = _make_tool()
        sink = self.engine._classify_sink(
            tool, {"url": "https://example.com"}
        )
        assert sink == SinkType.OUTBOUND

    def test_classify_outbound_from_email_args(self):
        tool = _make_tool()
        sink = self.engine._classify_sink(
            tool, {"recipient": "user@example.com"}
        )
        assert sink == SinkType.OUTBOUND

    def test_classify_memory_write(self):
        tool = _make_tool(name="store_memory")
        sink = self.engine._classify_sink(tool, {"data": "hello"})
        assert sink == SinkType.MEMORY_WRITE

    def test_classify_destructive(self):
        tool = _make_tool(classification=ActionClass.DESTRUCTIVE)
        sink = self.engine._classify_sink(tool, {"path": "/tmp"})
        assert sink == SinkType.DESTRUCTIVE

    def test_classify_read(self):
        tool = _make_tool(classification=ActionClass.READ)
        sink = self.engine._classify_sink(tool, {"query": "search"})
        assert sink == SinkType.READ

    def test_classify_general_write(self):
        tool = _make_tool()
        sink = self.engine._classify_sink(tool, {"data": "hello"})
        assert sink == SinkType.GENERAL_WRITE

    # --- Outbound overrides other classifications ---

    def test_destructive_with_url_is_outbound(self):
        """Outbound detection takes priority over destructive classification."""
        tool = _make_tool(classification=ActionClass.DESTRUCTIVE)
        sink = self.engine._classify_sink(
            tool, {"endpoint": "https://evil.com"}
        )
        assert sink == SinkType.OUTBOUND

    # --- Manifest-based classification ---

    def test_manifest_outbound_data_transfer(self):
        """Tool with outbound_data_transfer=True → OUTBOUND even with no URLs in args."""
        tool = _make_tool(
            capabilities=ToolCapabilityManifest(outbound_data_transfer=True),
        )
        sink = self.engine._classify_sink(tool, {"data": "plain text"})
        assert sink == SinkType.OUTBOUND

    def test_manifest_network_access(self):
        """Tool with network_access=True → OUTBOUND via has_outbound_capability."""
        tool = _make_tool(
            capabilities=ToolCapabilityManifest(network_access=True),
        )
        sink = self.engine._classify_sink(tool, {"query": "search term"})
        assert sink == SinkType.OUTBOUND

    def test_manifest_external_messaging(self):
        """Tool with external_messaging=True → OUTBOUND."""
        tool = _make_tool(
            capabilities=ToolCapabilityManifest(external_messaging=True),
        )
        sink = self.engine._classify_sink(tool, {"message": "hello"})
        assert sink == SinkType.OUTBOUND

    def test_manifest_memory_write(self):
        """Tool with memory_write=True → MEMORY_WRITE even if name doesn't match patterns."""
        tool = _make_tool(
            name="update_record",  # not in _MEMORY_WRITE_PATTERNS
            capabilities=ToolCapabilityManifest(memory_write=True),
        )
        sink = self.engine._classify_sink(tool, {"data": "hello"})
        assert sink == SinkType.MEMORY_WRITE

    def test_manifest_destructive(self):
        """Tool with destructive=True in manifest → DESTRUCTIVE."""
        tool = _make_tool(
            classification=ActionClass.WRITE,  # not DESTRUCTIVE by ActionClass
            capabilities=ToolCapabilityManifest(destructive=True),
        )
        sink = self.engine._classify_sink(tool, {"path": "/tmp/file"})
        assert sink == SinkType.DESTRUCTIVE

    def test_manifest_filesystem_delete(self):
        """Tool with filesystem_delete=True → DESTRUCTIVE."""
        tool = _make_tool(
            capabilities=ToolCapabilityManifest(filesystem_delete=True),
        )
        sink = self.engine._classify_sink(tool, {"path": "/tmp/file"})
        assert sink == SinkType.DESTRUCTIVE

    def test_manifest_default_falls_back_to_heuristics(self):
        """Tool with default manifest (all False) → falls back to arg/name heuristics."""
        # Default manifest, URL in args → OUTBOUND via arg inspection fallback
        tool = _make_tool()
        sink = self.engine._classify_sink(
            tool, {"url": "https://example.com"}
        )
        assert sink == SinkType.OUTBOUND

        # Default manifest, memory-write name → MEMORY_WRITE via name pattern fallback
        tool2 = _make_tool(name="store_memory")
        sink2 = self.engine._classify_sink(tool2, {"data": "hello"})
        assert sink2 == SinkType.MEMORY_WRITE

        # Default manifest, no indicators → GENERAL_WRITE
        tool3 = _make_tool()
        sink3 = self.engine._classify_sink(tool3, {"data": "hello"})
        assert sink3 == SinkType.GENERAL_WRITE

    def test_manifest_outbound_overrides_read_classification(self):
        """Manifest takes precedence: tool with outbound cap but READ classification → OUTBOUND."""
        tool = _make_tool(
            classification=ActionClass.READ,
            capabilities=ToolCapabilityManifest(outbound_data_transfer=True),
        )
        sink = self.engine._classify_sink(tool, {"query": "search"})
        assert sink == SinkType.OUTBOUND

    def test_manifest_outbound_priority_over_memory_write(self):
        """Outbound in manifest takes priority over memory_write in manifest."""
        tool = _make_tool(
            capabilities=ToolCapabilityManifest(
                outbound_data_transfer=True,
                memory_write=True,
            ),
        )
        sink = self.engine._classify_sink(tool, {"data": "hello"})
        assert sink == SinkType.OUTBOUND

    def test_manifest_memory_write_priority_over_destructive(self):
        """Memory write in manifest takes priority over destructive in manifest."""
        tool = _make_tool(
            capabilities=ToolCapabilityManifest(
                memory_write=True,
                destructive=True,
            ),
        )
        sink = self.engine._classify_sink(tool, {"data": "hello"})
        assert sink == SinkType.MEMORY_WRITE


# --- Destination Policy Sink Policy tests ---


from ollama_mcp_bridge.types import DestinationPolicy


class TestDestinationPolicySinkPolicy:
    """Tests for PR 11 destination policy integration in SinkPolicyEngine."""

    def setup_method(self):
        self.engine = SinkPolicyEngine()
        self.tainted = TaintState(
            tainted=True,
            taint_sources=["test:web_search"],
            taint_reasons=["url from search"],
            confidence=0.9,
        )
        self.config = SecurityConfig()

    def test_tainted_outbound_destination_policy_match(self):
        """URL matching a destination policy returns ALLOW_WITH_NOTICE."""
        tool = _make_tool(schema=_WRITE_SCHEMA)
        args = {"url": "https://api.example.com/v1/data", "data": "ok"}
        policies = [DestinationPolicy(
            host="api.example.com", path_prefixes=["/v1/"],
        )]
        result = self.engine.evaluate(
            tool, args, self.tainted, self.config,
            destination_policies=policies,
        )
        assert result == SinkDecision.ALLOW_WITH_NOTICE

    def test_tainted_outbound_destination_policy_no_match(self):
        """URL not matching any destination policy falls through to block."""
        tool = _make_tool(schema=_WRITE_SCHEMA)
        args = {"url": "https://evil.com/exfil", "data": "secret"}
        policies = [DestinationPolicy(host="api.example.com")]
        result = self.engine.evaluate(
            tool, args, self.tainted, self.config,
            destination_policies=policies,
        )
        assert result == SinkDecision.BLOCK

    def test_destination_policy_checked_before_domain_list(self):
        """When destination policies exist, domain list is not consulted."""
        config = SecurityConfig(allowed_outbound_domains=["evil.com"])
        tool = _make_tool(schema=_WRITE_SCHEMA)
        args = {"url": "https://evil.com/exfil", "data": "secret"}
        # Policy restricts to different host — should block even though domain list allows evil.com
        policies = [DestinationPolicy(host="safe.example.com")]
        result = self.engine.evaluate(
            tool, args, self.tainted, config,
            destination_policies=policies,
        )
        assert result == SinkDecision.BLOCK

    def test_require_flag_blocks_when_no_policies(self):
        """require_destination_policy_for_outbound=True blocks when no policies exist."""
        config = SecurityConfig(
            require_destination_policy_for_outbound=True,
            block_tainted_exfiltration=False,
        )
        tool = _make_tool(schema=_WRITE_SCHEMA)
        args = {"url": "https://any.com/data", "data": "secret"}
        result = self.engine.evaluate(
            tool, args, self.tainted, config,
            destination_policies=None,
        )
        assert result == SinkDecision.BLOCK

    def test_backward_compat_domain_list_still_works(self):
        """When no destination_policies passed, allowed_outbound_domains path runs."""
        config = SecurityConfig(allowed_outbound_domains=["example.com"])
        tool = _make_tool(schema=_WRITE_SCHEMA)
        args = {"url": "https://example.com/api", "data": "ok"}
        result = self.engine.evaluate(
            tool, args, self.tainted, config,
            destination_policies=None,
        )
        assert result == SinkDecision.ALLOW_WITH_NOTICE

    def test_multiple_policies_any_match_sufficient(self):
        """URL matching ANY one of multiple policies is allowed."""
        tool = _make_tool(schema=_WRITE_SCHEMA)
        args = {"url": "https://partner.com/webhook", "data": "event"}
        policies = [
            DestinationPolicy(host="internal.com"),
            DestinationPolicy(host="partner.com"),
        ]
        result = self.engine.evaluate(
            tool, args, self.tainted, self.config,
            destination_policies=policies,
        )
        assert result == SinkDecision.ALLOW_WITH_NOTICE

    def test_not_tainted_still_allows_with_policies(self):
        """Non-tainted calls always ALLOW regardless of policies."""
        clean = TaintState()
        tool = _make_tool(schema=_WRITE_SCHEMA)
        args = {"url": "https://evil.com/exfil"}
        policies = [DestinationPolicy(host="safe.com")]
        result = self.engine.evaluate(
            tool, args, clean, self.config,
            destination_policies=policies,
        )
        assert result == SinkDecision.ALLOW


# --- PR 13: Derived taint / InfluenceState tests ---


class TestDerivedMatchConfidence:
    """Tests for _derived_match_confidence — non-exact taint relationships."""

    def test_same_host_different_path(self):
        """URL reuse: same host, different path → 0.6."""
        arg = ExtractedValue(value="https://evil.com/api/v2", kind="url")
        tracked = ExtractedValue(value="https://evil.com/api/v1", kind="url")
        conf, itype = _derived_match_confidence(arg, tracked)
        assert conf == 0.6
        assert itype == InfluenceType.DERIVED_URL_REUSE

    def test_protocol_change(self):
        """Protocol swap: https→http on same host → 0.65."""
        arg = ExtractedValue(value="http://evil.com/api", kind="url")
        tracked = ExtractedValue(value="https://evil.com/api", kind="url")
        conf, itype = _derived_match_confidence(arg, tracked)
        assert conf == 0.65
        assert itype == InfluenceType.DERIVED_PROTOCOL_CHANGE

    def test_email_domain_reuse(self):
        """Different local part, same domain → 0.5."""
        arg = ExtractedValue(value="admin@evil.com", kind="email")
        tracked = ExtractedValue(value="user@evil.com", kind="email")
        conf, itype = _derived_match_confidence(arg, tracked)
        assert conf == 0.5
        assert itype == InfluenceType.DERIVED_EMAIL_DOMAIN

    def test_ip_in_host_port(self):
        """Tracked IP appears in host:port arg → 0.8."""
        arg = ExtractedValue(value="10.0.0.1:8080", kind="host_port")
        tracked = ExtractedValue(value="10.0.0.1", kind="ip")
        conf, itype = _derived_match_confidence(arg, tracked)
        assert conf == 0.8
        assert itype == InfluenceType.DERIVED_HOSTNAME_IN_URL

    def test_hostname_in_host_port(self):
        """Tracked hostname in host:port arg → 0.7."""
        arg = ExtractedValue(value="evil.com:443", kind="host_port")
        tracked = ExtractedValue(value="evil.com", kind="domain")
        conf, itype = _derived_match_confidence(arg, tracked)
        assert conf == 0.7
        assert itype == InfluenceType.DERIVED_HOSTNAME_IN_URL

    def test_hostname_cross_kind(self):
        """Hostname matches domain → 0.75."""
        arg = ExtractedValue(value="evil.com", kind="hostname")
        tracked = ExtractedValue(value="evil.com", kind="domain")
        conf, itype = _derived_match_confidence(arg, tracked)
        assert conf == 0.75
        assert itype == InfluenceType.DIRECT_VALUE_MATCH

    def test_no_derived_match(self):
        """Completely different values → 0.0."""
        arg = ExtractedValue(value="https://safe.com/api", kind="url")
        tracked = ExtractedValue(value="https://evil.com/exfil", kind="url")
        conf, itype = _derived_match_confidence(arg, tracked)
        assert conf == 0.0
        assert itype is None

    def test_same_email_exact_not_derived(self):
        """Exact same email → handled by _match_confidence, not derived."""
        arg = ExtractedValue(value="user@evil.com", kind="email")
        tracked = ExtractedValue(value="user@evil.com", kind="email")
        # _derived_match_confidence should still catch it via domain reuse
        conf, itype = _derived_match_confidence(arg, tracked)
        assert conf == 0.5  # domain matches even though full email also matches


class TestComputeTaintInfluenceState:
    """Tests that compute_taint returns InfluenceState with evidence."""

    def test_returns_influence_state(self):
        tracker = TaintTracker()
        tracker.record_result(
            content="Found at https://evil.com/api/v1",
            origin_id="scraper:search",
            provenance=_make_provenance(),
        )
        state = tracker.compute_taint({"url": "https://evil.com/api/v1"})
        assert isinstance(state, InfluenceState)
        assert isinstance(state, TaintState)  # IS-A

    def test_direct_match_evidence(self):
        tracker = TaintTracker()
        tracker.record_result(
            content="Found at https://evil.com/exfil",
            origin_id="scraper:search",
            provenance=_make_provenance(),
        )
        state = tracker.compute_taint({"target": "https://evil.com/exfil"})
        assert state.tainted is True
        assert state.direct_value_match is True
        assert state.derived_from_untrusted_value is False
        assert len(state.evidence) >= 1
        assert state.evidence[0].influence_type == InfluenceType.DIRECT_VALUE_MATCH

    def test_derived_url_reuse_evidence(self):
        """Same host, different path → derived taint detected."""
        tracker = TaintTracker()
        tracker.record_result(
            content="Found at https://evil.com/api/v1",
            origin_id="scraper:search",
            provenance=_make_provenance(),
        )
        state = tracker.compute_taint({"target": "https://evil.com/api/v2"})
        assert state.tainted is True
        assert state.derived_from_untrusted_value is True
        assert state.destination_influenced is True
        derived = [e for e in state.evidence
                   if e.influence_type == InfluenceType.DERIVED_URL_REUSE]
        assert len(derived) >= 1

    def test_protocol_change_evidence(self):
        """https→http on same host → derived protocol change."""
        tracker = TaintTracker()
        tracker.record_result(
            content="Found at https://evil.com/data",
            origin_id="web:fetch",
            provenance=_make_provenance(),
        )
        state = tracker.compute_taint({"target": "http://evil.com/data"})
        assert state.tainted is True
        assert state.destination_influenced is True
        protocol = [e for e in state.evidence
                    if e.influence_type == InfluenceType.DERIVED_PROTOCOL_CHANGE]
        assert len(protocol) >= 1

    def test_no_taint_returns_empty_influence_state(self):
        tracker = TaintTracker()
        state = tracker.compute_taint({"query": "hello"})
        assert isinstance(state, InfluenceState)
        assert state.tainted is False
        assert state.evidence == []

    def test_mixed_direct_and_derived(self):
        """Both direct and derived matches produce correct flags."""
        tracker = TaintTracker()
        tracker.record_result(
            content="Visit https://evil.com/original and user@evil.com",
            origin_id="web:fetch",
            provenance=_make_provenance(),
        )
        state = tracker.compute_taint({
            "url": "https://evil.com/original",  # direct
            "email": "admin@evil.com",  # derived (domain reuse)
        })
        assert state.tainted is True
        assert state.direct_value_match is True
        assert state.derived_from_untrusted_value is True
