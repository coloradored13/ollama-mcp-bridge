"""Tests for sink_policy.py — taint tracking and sink policy engine."""

import pytest

from ollama_mcp_bridge.config import SecurityConfig
from ollama_mcp_bridge.sink_policy import (
    SinkPolicyEngine,
    SinkType,
    TaintTracker,
    _extract_values,
    _extract_values_from_args,
    _is_memory_write_tool,
    _args_contain_outbound_indicators,
)
from ollama_mcp_bridge.types import (
    ActionClass,
    ApprovedTool,
    ContentProvenance,
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
