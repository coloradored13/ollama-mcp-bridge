# ollama-mcp-bridge Sequenced Implementation Plan

## Purpose
This document turns the security redesign spec into a practical delivery plan with:
- implementation sequence,
- PR scope,
- dependencies,
- acceptance tests,
- rollout guidance,
- and “do not proceed until” gates.

This plan assumes the current `main` branch already includes:
- tool-result contract fixes,
- real streaming behavior,
- secure audit summaries,
- full session audit retention,
- improved nested parameter validation,
- and the current first-seen auto-approval model.  
See current implementation in:
- `translator.py`
- `loop.py`
- `audit.py`
- `config.py`
- `security.py`  
fileciteturn30file0L1-L1 fileciteturn31file0L1-L1 fileciteturn32file0L1-L1 fileciteturn33file0L1-L1 fileciteturn28file0L1-L1

---

# Delivery Strategy

## Recommended order
1. Fail-closed allowlist semantics
2. First-run approval state machine
3. Approval registry redesign
4. Discovery and approval APIs
5. Audit/event fidelity improvements
6. Semantic-defense foundation
7. Sink policy engine
8. Capability narrowing
9. End-to-end integration and adversarial evals

## Why this order
You should not build semantic taint and sink policies on top of an approval model that still auto-approves first-seen tools or treats empty allowlists as allow-all. Fix the trust boundary first, then add smarter defenses on top of it.

---

# PR 1 — Fail-Closed Allowlist Semantics

## Goal
Change `allowed_tools = []` from “allow all” to “allow none”.

## Why first
This is the most important default-behavior correction. Right now current `main` still treats an empty allowlist as allow-all. fileciteturn33file0L1-L1

## Scope
### Files
- `src/ollama_mcp_bridge/config.py`
- `src/ollama_mcp_bridge/security.py`
- `bridge.toml`
- tests covering config and connect/scan behavior

### Changes
- Change `BridgeConfig.is_tool_allowed()` so empty allowlists return `False`.
- Update warnings/messages to reflect fail-closed semantics.
- Make discovery still possible internally even when no tools are allowed.
- Ensure zero tools from that server enter `_approved_tools`.

### Acceptance tests
- Empty allowlist exposes no callable tools.
- Discovered tools are still listed or made available for user review.
- A server with `allowed_tools=[]` yields no approved tools after `connect_and_scan()`.
- A server with explicit tools in `allowed_tools` still approves only matching tools.

## Do not proceed until
- No code path can expose tools to the model from an empty allowlist.
- Tests verify the new semantics clearly.

---

# PR 2 — First-Run Approval State Machine

## Goal
Add pending first-run approval instead of automatic approval of first-seen allowlisted tools.

## Why second
This is the core trust-model correction. Current `connect_and_scan()` still auto-approves first-seen allowlisted tools. fileciteturn28file0L1-L1

## Scope
### Files
- `src/ollama_mcp_bridge/security.py`
- `src/ollama_mcp_bridge/types.py`
- potentially `src/ollama_mcp_bridge/bridge.py`
- tests for pending/approved/denied state transitions

### Changes
- Introduce states:
  - `DISCOVERED`
  - `ALLOWLISTED`
  - `PENDING_FIRST_APPROVAL`
  - `APPROVED`
  - `BLOCKED_SANITIZATION`
  - `BLOCKED_INTEGRITY`
  - `DENIED_BY_USER`
- Update `connect_and_scan()` to:
  - discover tools,
  - sanitize them,
  - check integrity,
  - mark first-seen allowlisted tools as pending approval by default,
  - avoid calling `registry.approve()` automatically for unknown hashes.
- Add config flags:
  - `require_first_run_approval = true`
  - `auto_approve_first_seen = false`

### Acceptance tests
- First-seen allowlisted tool is pending, not approved.
- Pending tool is not callable.
- Pending tool is not shown in approved-tool list.
- If `auto_approve_first_seen = true`, legacy behavior is still available.
- Known matching hash still auto-approves silently.
- Hash mismatch still blocks and requires re-approval.

## Do not proceed until
- First-seen allowlisted tools cannot be called before approval.
- No unknown hash is silently persisted as approved under default config.

---

# PR 3 — Approval Registry Redesign

## Goal
Make the approval registry represent explicit runtime trust, not just first-seen hashes.

## Why third
Once first-run approval exists, the registry model has to reflect it.

## Scope
### Files
- `src/ollama_mcp_bridge/security.py`
- `src/ollama_mcp_bridge/types.py`
- registry storage helpers / JSON persistence logic
- tests for registry migration and integrity behavior

### Changes
- Expand registry entry model to include:
  - server
  - tool name
  - approved hash
  - approved timestamp
  - approval mode
  - classification
  - optional notes
  - optional last-seen timestamp
- Optionally track denied hashes.
- Add backward-compatible migration from old `{server:tool -> hash}` format.

### Acceptance tests
- Old registry format migrates cleanly.
- Newly approved tools store structured records.
- Re-approval updates the stored hash and metadata.
- Previously denied hash can be recognized and surfaced.
- Registry corruption still fails safely.

## Do not proceed until
- Approval state is persisted in a format that distinguishes explicit approval from silent discovery.

---

# PR 4 — Discovery and Approval APIs

## Goal
Expose discovered tools and pending approvals cleanly to callers.

## Why fourth
Once tools can be pending instead of approved, the bridge needs a first-class way to show and manage that state.

## Scope
### Files
- `src/ollama_mcp_bridge/security.py`
- `src/ollama_mcp_bridge/bridge.py`
- `src/ollama_mcp_bridge/types.py`
- tests for public API behavior

### Changes
Add public APIs:
- `Bridge.list_discovered_tools()`
- `Bridge.list_pending_tool_approvals()`
- `Bridge.approve_tool(server, tool_name)`
- `Bridge.deny_tool(server, tool_name)`

Internal APIs:
- `SecurityGateway.get_discovered_tools_by_server()`
- `SecurityGateway.get_pending_approvals()`
- `SecurityGateway.approve_tool(...)`
- `SecurityGateway.deny_tool(...)`

### Acceptance tests
- Pending tools are visible through the public API.
- Approval changes state and makes the tool callable.
- Denial leaves the tool unavailable.
- Discovery output includes unallowlisted tools separately from pending allowlisted tools.
- Public APIs fail clearly when server/tool names are unknown.

## Do not proceed until
- The bridge has a usable operator flow for reviewing pending tools.

---

# PR 5 — Audit Fidelity and Approval Event Logging

## Goal
Improve forensic accuracy and record the new approval workflow.

## Why fifth
The redesigned approval flow should be fully auditable.

## Scope
### Files
- `src/ollama_mcp_bridge/audit.py`
- `src/ollama_mcp_bridge/security.py`
- `src/ollama_mcp_bridge/types.py`
- audit tests

### Changes
Add audit events:
- `tool_pending_first_approval`
- `tool_first_approved`
- `tool_first_denied`
- `integrity_reapproval_required`

Also fix confirmation audit fidelity:
- distinguish `timeout` from `user_denied`
- distinguish `no_callback` from both

Current `main` still logs timeout/false as user denial in the destructive confirmation path. fileciteturn28file0L1-L1

### Acceptance tests
- Pending approval is logged.
- Explicit approval is logged.
- Explicit denial is logged.
- Timeout is logged as timeout, not as user denial.
- No secret values leak into new approval-related audit fields.

## Do not proceed until
- Audit logs accurately reflect human actions and timeouts.

---

# PR 6 — Semantic Defense Foundation

## Goal
Introduce provenance and structured semantic risk assessment.

## Why sixth
This is the first step beyond lexical tripwires. It should come after the trust boundary is fixed.

## Scope
### Files
- `src/ollama_mcp_bridge/types.py`
- `src/ollama_mcp_bridge/security.py`
- possibly `src/ollama_mcp_bridge/loop.py`
- new semantic-risk module(s)

### Changes
Add types:
- `ContentProvenance`
- `SemanticRiskAssessment`

Add semantic-risk interface:
- `SemanticRiskAssessor.assess(content, provenance) -> SemanticRiskAssessment`

Initial implementation can be:
- interface + stub backend,
- or a simple internal model/plugin wrapper,
- but must produce structured output, not only pass/block.

### Acceptance tests
- Tool results can be wrapped with provenance.
- Semantic assessor returns structured fields, not just a boolean.
- Semantic-risk results are attached to content objects or execution context.
- Existing lexical sanitization still works alongside semantic assessment.

## Do not proceed until
- The system can represent provenance and semantic-risk state in structured form.

---

# PR 7 — Taint Tracking and Sink Policy Engine

## Goal
Track untrusted influence and enforce source-to-sink policy.

## Why seventh
This is the real shift from “spot suspicious text” to “protect sensitive actions”.

## Scope
### Files
- new `sink_policy.py` or similar
- `src/ollama_mcp_bridge/security.py`
- `src/ollama_mcp_bridge/loop.py`
- `src/ollama_mcp_bridge/types.py`
- tests for tainted actions

### Changes
Add:
- `TaintState`
- `SinkDecision`
- `SinkPolicyEngine`

Policy examples:
- tainted + outbound transmission => block by default
- tainted + destructive write => require confirmation or block
- tainted + new external domain => block unless explicitly allowed
- tainted + memory write => block by default

### Acceptance tests
- Third-party content proposing a URL taints outbound arguments.
- Tainted outbound transmission is blocked by default.
- Tainted destructive write requires confirmation.
- Clean user-requested actions with no taint still work.

## Do not proceed until
- Sensitive sinks are no longer controlled solely by model-generated arguments.

---

# PR 8 — Capability Narrowing

## Goal
Reduce the danger of free-form model-supplied arguments.

## Why eighth
Semantic defense is much stronger when high-risk arguments are narrowed or adapted before reaching tools.

## Scope
### Files
- adapters/validators for URLs, paths, recipients, memory writes
- `src/ollama_mcp_bridge/security.py`
- tool execution path
- tests for safe adapters

### Changes
Implement safe adapters:
- `SafeURL`
- `SafePath`
- `SafeRecipient`
- `SafeMemoryWriteCandidate`

Examples:
- only approved domains may be used for outbound network operations
- paths must remain inside approved roots
- recipients must come from user input, known context, or approved contacts
- memory writes from third-party content must be blocked or explicitly approved

### Acceptance tests
- Arbitrary URLs suggested by tool output are blocked.
- Arbitrary file paths are blocked.
- Arbitrary recipients suggested by third-party content are blocked.
- Safe adapted inputs pass cleanly.

## Do not proceed until
- High-risk sinks have narrowed inputs instead of raw free-form arguments.

---

# PR 9 — Live End-to-End Tests and Adversarial Eval Harness

## Goal
Prove the new model in real runtime conditions.

## Why last
You want the architecture in place before building the heavier end-to-end test matrix.

## Scope
### Files
- live integration tests
- test MCP server fixture(s)
- eval harness scripts/config
- CI docs or local test docs

### Changes
Add at least one live test with:
- a real Ollama model
- a minimal MCP test server
- first-run approval required
- one successful approved tool flow
- one blocked pending tool
- one rug-pull reapproval case
- one tainted outbound proposal blocked

Add adversarial eval cases for:
- indirect prompt injection
- exfiltration suggestions
- recommendation manipulation
- hidden instruction paraphrases
- multi-turn persistence attempts

### Acceptance tests
- Live test passes under documented environment.
- Eval harness produces measurable pass/fail signals.
- Regressions are detectable across future PRs.

## Do not proceed until
- The system has at least one real runtime proof path beyond mocks.

---

# Cross-PR Dependencies

## Hard dependencies
- PR 2 depends on PR 1.
- PR 3 depends on PR 2.
- PR 4 depends on PR 2 and PR 3.
- PR 5 depends on PR 2 through PR 4.
- PR 7 depends on PR 6.
- PR 8 depends on PR 7.
- PR 9 depends on all prior PRs.

## Recommended merge discipline
- Keep each PR narrowly scoped.
- Avoid mixing approval-model changes with semantic-defense changes.
- Do not combine PR 6, 7, and 8 into one mega-PR.

---

# Suggested Milestones

## Milestone A — Trust Boundary Correction
Includes:
- PR 1
- PR 2
- PR 3
- PR 4
- PR 5

Outcome:
- fail-closed allowlists
- no first-run auto-approval by default
- explicit approval workflow
- correct audit trail

## Milestone B — Semantic Defense Foundation
Includes:
- PR 6
- PR 7

Outcome:
- provenance-aware content handling
- taint-aware sink policy

## Milestone C — High-Risk Capability Hardening
Includes:
- PR 8

Outcome:
- narrowed high-risk arguments
- safer outbound, file, recipient, and memory behavior

## Milestone D — Runtime Proof and Regression Defense
Includes:
- PR 9

Outcome:
- live confidence
- adversarial regression detection

---

# Recommended Acceptance Gate for Release

Before calling the bridge “security-first” again, require all of the following:

1. Empty allowlist allows no tools.
2. First-seen allowlisted tools require explicit approval by default.
3. Approval registry stores explicit trust state for runtime definitions.
4. Audit logs distinguish denial vs timeout.
5. Untrusted content provenance is tracked.
6. At least one tainted-sink policy is enforced in code.
7. At least one live end-to-end test exists.

---

# Recommended Immediate Next Step

Start with **PR 1** and **PR 2**.

Those two PRs correct the biggest trust-model problems:
- empty allowlist is no longer permissive,
- first-seen allowlisted tools are no longer silently trusted.

Everything else should build on top of that.
