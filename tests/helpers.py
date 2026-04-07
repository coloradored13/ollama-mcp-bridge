"""Shared constants and markers for live/adversarial tests.

Importable from test files (unlike conftest.py). Fixtures stay in conftest.
"""

from __future__ import annotations

import json
import sys
import urllib.request
from pathlib import Path

import pytest

# --- Paths ---

TEST_MCP_SERVER = str(Path(__file__).parent / "fixtures" / "test_mcp_server.py")
ADVERSARIAL_MCP_SERVER = str(Path(__file__).parent / "fixtures" / "adversarial_mcp_server.py")
TEST_PYTHON = sys.executable


# --- Ollama detection ---


def _ollama_available() -> bool:
    try:
        resp = urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3)
        data = json.loads(resp.read())
        return len(data.get("models", [])) > 0
    except Exception:
        return False


requires_ollama = pytest.mark.skipif(
    not _ollama_available(),
    reason="Ollama not running or no models available",
)
