"""Test individual security layers in isolation, at their exact boundaries.

Mutation testing (mutmut 3.8) showed two layered defenses where one layer
masked the other in tests:

- Nesting depth: the schema check (`_measure_depth`) and the deep scan
  (`_deep_scan_values`) both emit "nesting depth" errors, and the only test
  used depth 8 against a limit of 5. The schema check could be deleted, or
  shifted off by one, with every test still passing.
- Rate limiting: only the session-wide cap was tested. Recording timestamps
  under the wrong key disabled the per-server sliding window with every test
  still passing.

Each test here passes on current code and fails on those mutants.
"""

import pytest

from ollama_mcp_bridge.config import SecurityConfig
from ollama_mcp_bridge.errors import RateLimitError
from ollama_mcp_bridge.security import ParameterValidator, RateLimiter


def _nest(levels: int) -> dict:
    """A dict nested `levels` deep: _nest(1) == {"level": 1}."""
    d: dict = {"level": 1}
    for _ in range(levels - 1):
        d = {"nested": d}
    return d


class TestSchemaDepthCheck:
    """The schema-level check on its own, without the deep scan."""

    @pytest.mark.parametrize(
        ("obj", "expected"),
        [
            ({}, 0),
            ({"a": 1}, 1),
            ([[1]], 2),
            (_nest(5), 5),
        ],
    )
    def test_measure_depth_exact_values(self, obj, expected):
        assert ParameterValidator._measure_depth(obj) == expected

    def test_allows_exactly_the_limit(self):
        limit = ParameterValidator.MAX_NESTING_DEPTH
        errors = ParameterValidator()._check_param_security(
            "data", _nest(limit), {"type": "object"}
        )
        assert errors == []

    def test_fires_one_past_the_limit(self):
        limit = ParameterValidator.MAX_NESTING_DEPTH
        errors = ParameterValidator()._check_param_security(
            "data", _nest(limit + 1), {"type": "object"}
        )
        assert errors == [f"Parameter 'data': nesting depth {limit + 1} exceeds maximum ({limit})"]


class TestPerServerRateLimit:
    """Layer 3 (per-server sliding window), isolated from layers 1 and 2."""

    def _limiter(self) -> RateLimiter:
        # Session cap set high so layer 1 can't be what fires.
        return RateLimiter(SecurityConfig(rate_limit_per_server=3, max_tool_calls_per_session=100))

    def test_trips_at_the_per_server_limit(self):
        rl = self._limiter()
        for tool in ["a", "b", "c"]:  # distinct tools, so the per-tool cap can't fire
            rl.check("files", tool)
            rl.record_call("files", tool)

        with pytest.raises(RateLimitError, match=r"Rate limit for server 'files' \(3/min\)"):
            rl.check("files", "d")

    def test_other_servers_are_unaffected(self):
        rl = self._limiter()
        for tool in ["a", "b", "c"]:
            rl.record_call("files", tool)

        rl.check("web", "a")  # must not raise
