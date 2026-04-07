#!/usr/bin/env python3
"""Run E2E tests tier-by-tier and generate reports.

Produces:
  - tests/results/e2e-report.md  (human-readable)
  - tests/results/e2e-results.xml (JUnit XML)

Each tier runs in its own pytest invocation for clean subprocess
isolation. The script detects available Ollama models and records
the environment in the report.

Usage:
  python tests/run_e2e_report.py           # run all tiers
  python tests/run_e2e_report.py --fast    # skip model tiers (MCP + pipeline only)
"""

from __future__ import annotations

import json
import platform
import subprocess
import sys
import urllib.request
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path

REPO_ROOT = Path(__file__).parent.parent
RESULTS_DIR = Path(__file__).parent / "results"
PYTHON = sys.executable


# --- Test tiers ---

TIERS = [
    {
        "name": "Unit Tests",
        "id": "unit",
        "path": "tests/",
        "ignore": [
            "tests/test_live_mcp.py",
            "tests/test_live_model.py",
            "tests/test_live_multistep.py",
            "tests/test_adversarial.py",
        ],
        "requires_model": False,
    },
    {
        "name": "MCP Connection & Approval",
        "id": "live_mcp",
        "path": "tests/test_live_mcp.py",
        "requires_model": False,
    },
    {
        "name": "Adversarial Pipeline",
        "id": "adversarial_pipeline",
        "path": "tests/test_adversarial.py::TestAdversarialResultSanitization",
        "requires_model": False,
    },
    {
        "name": "Model Single-Tool",
        "id": "live_model",
        "path": "tests/test_live_model.py",
        "requires_model": True,
    },
    {
        "name": "Multi-Step & Fault Tolerance",
        "id": "live_multistep",
        "path": "tests/test_live_multistep.py",
        "requires_model": True,
    },
    {
        "name": "Adversarial Model-in-the-Loop",
        "id": "adversarial_model",
        "path": "tests/test_adversarial.py::TestAdversarialWithModel",
        "requires_model": True,
    },
]


@dataclass
class TestResult:
    name: str
    passed: bool
    duration: float = 0.0
    failure_message: str = ""


@dataclass
class TierResult:
    name: str
    tier_id: str
    tests: list[TestResult] = field(default_factory=list)
    duration: float = 0.0
    passed: int = 0
    failed: int = 0
    skipped: int = 0
    error: str = ""


def get_environment() -> dict:
    """Gather environment info for the report."""
    env = {
        "python": platform.python_version(),
        "platform": f"{platform.system()} {platform.release()}",
        "machine": platform.machine(),
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "ollama_available": False,
        "ollama_models": [],
        "ollama_model_used": "",
    }

    try:
        resp = urllib.request.urlopen("http://127.0.0.1:11434/api/tags", timeout=3)
        data = json.loads(resp.read())
        models = data.get("models", [])
        if models:
            env["ollama_available"] = True
            env["ollama_models"] = [m["name"] for m in models]
            # Smallest model (what tests use)
            models.sort(key=lambda m: m.get("size", float("inf")))
            env["ollama_model_used"] = models[0]["name"]
    except Exception:
        pass

    try:
        resp = urllib.request.urlopen("http://127.0.0.1:11434/api/version", timeout=3)
        data = json.loads(resp.read())
        env["ollama_version"] = data.get("version", "unknown")
    except Exception:
        env["ollama_version"] = "unknown"

    return env


def run_tier(tier: dict, xml_path: Path) -> TierResult:
    """Run a single test tier and parse results from JUnit XML."""
    result = TierResult(name=tier["name"], tier_id=tier["id"])

    cmd = [
        PYTHON, "-m", "pytest",
        tier["path"],
        f"--junit-xml={xml_path}",
        "-v", "--tb=short",
    ]

    # Add ignore flags for unit tier
    for ignore in tier.get("ignore", []):
        cmd.extend(["--ignore", ignore])

    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        cwd=str(REPO_ROOT),
        env={**__import__("os").environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        timeout=600,
    )

    # Parse JUnit XML
    if xml_path.exists():
        try:
            tree = ET.parse(xml_path)
            root = tree.getroot()

            for suite in root.iter("testsuite"):
                result.duration = float(suite.get("time", 0))
                result.passed = int(suite.get("tests", 0)) - int(suite.get("failures", 0)) - int(suite.get("errors", 0)) - int(suite.get("skipped", 0))
                result.failed = int(suite.get("failures", 0)) + int(suite.get("errors", 0))
                result.skipped = int(suite.get("skipped", 0))

            for testcase in root.iter("testcase"):
                test_name = testcase.get("name", "unknown")
                test_time = float(testcase.get("time", 0))

                failure = testcase.find("failure")
                skipped_el = testcase.find("skipped")

                if failure is not None:
                    result.tests.append(TestResult(
                        name=test_name,
                        passed=False,
                        duration=test_time,
                        failure_message=failure.get("message", "")[:200],
                    ))
                elif skipped_el is not None:
                    continue  # Don't include skipped in report
                else:
                    result.tests.append(TestResult(
                        name=test_name,
                        passed=True,
                        duration=test_time,
                    ))
        except ET.ParseError:
            result.error = "Failed to parse JUnit XML"
    else:
        result.error = f"No XML output (exit code {proc.returncode})"
        if proc.stderr:
            result.error += f"\n{proc.stderr[:500]}"

    return result


def generate_markdown(env: dict, tiers: list[TierResult]) -> str:
    """Generate human-readable markdown report."""
    lines = []
    now = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    total_passed = sum(t.passed for t in tiers)
    total_failed = sum(t.failed for t in tiers)
    total_tests = total_passed + total_failed
    total_time = sum(t.duration for t in tiers)
    status = "PASS" if total_failed == 0 else "FAIL"

    lines.append("# E2E Test Report")
    lines.append("")
    lines.append(f"**Status:** {status} | **Tests:** {total_passed}/{total_tests} passed | **Time:** {total_time:.0f}s")
    lines.append(f"**Generated:** {now}")
    lines.append("")
    lines.append("## Environment")
    lines.append("")
    lines.append(f"- **Python:** {env['python']}")
    lines.append(f"- **Platform:** {env['platform']} ({env['machine']})")
    if env["ollama_available"]:
        lines.append(f"- **Ollama:** v{env.get('ollama_version', 'unknown')}")
        lines.append(f"- **Model used:** {env['ollama_model_used']}")
        lines.append(f"- **Models available:** {', '.join(env['ollama_models'])}")
    else:
        lines.append("- **Ollama:** not available (model tests skipped)")
    lines.append("")

    for tier in tiers:
        tier_status = "PASS" if tier.failed == 0 and not tier.error else "FAIL"
        tier_total = tier.passed + tier.failed
        lines.append(f"## {tier.name}")
        lines.append("")
        lines.append(f"**{tier_status}** | {tier.passed}/{tier_total} passed | {tier.duration:.1f}s")
        lines.append("")

        if tier.error:
            lines.append(f"> Error: {tier.error}")
            lines.append("")

        # Collapse large passing tiers (unit tests) to summary only
        show_table = tier.failed > 0 or len(tier.tests) <= 50
        if tier.tests and show_table:
            lines.append("| Test | Result | Time |")
            lines.append("|------|--------|------|")
            for test in tier.tests:
                mark = "PASS" if test.passed else "FAIL"
                lines.append(f"| {test.name} | {mark} | {test.duration:.1f}s |")
                if not test.passed and test.failure_message:
                    lines.append(f"| | {test.failure_message[:100]} | |")
            lines.append("")
        elif tier.tests and not show_table:
            lines.append(f"*{len(tier.tests)} tests — table omitted (all passed in <1s each)*")
            lines.append("")

    lines.append("---")
    lines.append(f"*Report generated by `tests/run_e2e_report.py`*")
    lines.append("")

    return "\n".join(lines)


def merge_junit_xml(tier_xmls: list[Path], output: Path) -> None:
    """Merge per-tier JUnit XML files into a single report."""
    root = ET.Element("testsuites")

    for xml_path in tier_xmls:
        if not xml_path.exists():
            continue
        try:
            tree = ET.parse(xml_path)
            for suite in tree.getroot().iter("testsuite"):
                root.append(suite)
        except ET.ParseError:
            continue

    tree = ET.ElementTree(root)
    ET.indent(tree, space="  ")
    tree.write(str(output), encoding="unicode", xml_declaration=True)


def main():
    fast_only = "--fast" in sys.argv
    tier_filter = None
    merge_only = "--merge" in sys.argv
    for arg in sys.argv[1:]:
        if arg.startswith("--tier="):
            tier_filter = arg.split("=", 1)[1]

    print("=" * 60)
    print("E2E Test Report Generator")
    print("=" * 60)

    env = get_environment()
    print(f"\nEnvironment:")
    print(f"  Python: {env['python']}")
    print(f"  Platform: {env['platform']}")
    if env["ollama_available"]:
        print(f"  Ollama: v{env.get('ollama_version', 'unknown')}")
        print(f"  Model: {env['ollama_model_used']}")
    else:
        print("  Ollama: not available")

    RESULTS_DIR.mkdir(exist_ok=True)
    tier_xmls: list[Path] = []
    tier_results: list[TierResult] = []

    if merge_only:
        print("\n--- Merge mode: reading existing tier XMLs ---")
    else:
        for tier in TIERS:
            if tier_filter and tier["id"] != tier_filter:
                continue

            if fast_only and tier["requires_model"]:
                print(f"\n--- Skipping {tier['name']} (--fast) ---")
                continue

            if tier["requires_model"] and not env["ollama_available"]:
                print(f"\n--- Skipping {tier['name']} (no Ollama) ---")
                continue

            print(f"\n--- {tier['name']} ---")
            xml_path = RESULTS_DIR / f"{tier['id']}.xml"

            result = run_tier(tier, xml_path)
            tier_results.append(result)

            status = "PASS" if result.failed == 0 else "FAIL"
            print(f"  {status}: {result.passed}/{result.passed + result.failed} passed in {result.duration:.1f}s")

            if result.failed > 0:
                for test in result.tests:
                    if not test.passed:
                        print(f"    FAIL: {test.name}")
                        if test.failure_message:
                            print(f"          {test.failure_message[:100]}")

    # In merge mode or after running tiers, read all existing XMLs
    if merge_only or tier_filter:
        tier_results = []
        for tier in TIERS:
            xml_path = RESULTS_DIR / f"{tier['id']}.xml"
            if xml_path.exists():
                tier_xmls.append(xml_path)
                result = TierResult(name=tier["name"], tier_id=tier["id"])
                try:
                    tree = ET.parse(xml_path)
                    root = tree.getroot()
                    for suite in root.iter("testsuite"):
                        result.duration = float(suite.get("time", 0))
                        result.passed = int(suite.get("tests", 0)) - int(suite.get("failures", 0)) - int(suite.get("errors", 0)) - int(suite.get("skipped", 0))
                        result.failed = int(suite.get("failures", 0)) + int(suite.get("errors", 0))
                    for testcase in root.iter("testcase"):
                        test_name = testcase.get("name", "unknown")
                        test_time = float(testcase.get("time", 0))
                        failure = testcase.find("failure")
                        if failure is not None:
                            result.tests.append(TestResult(name=test_name, passed=False, duration=test_time, failure_message=failure.get("message", "")[:200]))
                        elif testcase.find("skipped") is None:
                            result.tests.append(TestResult(name=test_name, passed=True, duration=test_time))
                except ET.ParseError:
                    result.error = "Failed to parse XML"
                tier_results.append(result)

    # Collect all tier XMLs for merge
    for tier in TIERS:
        xml_path = RESULTS_DIR / f"{tier['id']}.xml"
        if xml_path.exists() and xml_path not in tier_xmls:
            tier_xmls.append(xml_path)

    # Generate reports
    md_report = generate_markdown(env, tier_results)
    md_path = RESULTS_DIR / "e2e-report.md"
    md_path.write_text(md_report)
    print(f"\nMarkdown report: {md_path}")

    merged_xml_path = RESULTS_DIR / "e2e-results.xml"
    merge_junit_xml(tier_xmls, merged_xml_path)
    print(f"JUnit XML report: {merged_xml_path}")

    total_failed = sum(t.failed for t in tier_results)
    total_passed = sum(t.passed for t in tier_results)
    print(f"\n{'=' * 60}")
    print(f"Total: {total_passed}/{total_passed + total_failed} passed")
    print(f"{'=' * 60}")

    return 1 if total_failed > 0 else 0


if __name__ == "__main__":
    sys.exit(main())
