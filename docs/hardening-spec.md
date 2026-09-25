# ollama-mcp-bridge “.999” Hardening Spec

## Status
Draft

## Purpose
This spec defines the next-level hardening pass for `ollama-mcp-bridge`.

The goal is **not** to claim perfection. The goal is to move from:

- **credible v1 security middleware**

to:

- **high-consequence-ready security middleware**, where the bridge can be trusted as a serious boundary in front of local models **when paired with disciplined deployment controls**.

This spec is written for the standard:

> “The model is untrusted. A single miss can be very bad.  
> We need the system to be strong, usable, and fail-closed under realistic operator error and adversarial pressure.”

---

# 1. Executive Summary

The current bridge is already strong:

- fail-closed allowlists,
- explicit first-run approval,
- structured approval registry,
- rug-pull detection,
- taint tracking,
- sink policy,
- capability narrowing adapters,
- adversarial and live E2E coverage.

That is good enough to call the project a **real security boundary**.

But it is not yet “.999” hardening, because several important controls still depend on:

- heuristic sink classification,
- opt-in adapter configuration,
- name-pattern inference,
- domain-level trust that is broader than some high-risk use cases can tolerate,
- and deployment choices outside the bridge.

This spec closes those gaps.

---

# 2. Design Goal

## Target posture
The target is:

- **fail closed by default**
- **typed capability constraints**
- **destination-aware enforcement**
- **stronger source-to-sink controls**
- **less reliance on heuristic naming**
- **less room for dangerous misconfiguration**
- **stronger deployment and release gates**

## What this does not mean
It does **not** mean:
- “perfect prompt injection defense”
- “formal proof of safety”
- “safe even if the operator grants absurdly broad tool power”

It means:
- the bridge should be hard to trick,
- hard to misconfigure dangerously,
- and hard to deploy unsafely without knowing it.

---

# 3. Current-State Strengths

The current implementation already provides:

- atomic execution through `SecurityGateway.execute_tool()`
- fail-closed empty allowlists
- explicit first-run approval
- structured registry entries with approval modes and denied hashes
- taint tracking via extracted values from tool results
- sink policy for tainted source-to-sink flows
- capability narrowing via safe adapters
- structured audit logging
- live MCP tests, live model tests, and adversarial tests

That remains the foundation. This spec builds on it rather than replacing it.

---

# 4. Hard Truths / Remaining Gaps

## 4.1 Heuristic sink classification is still too soft
Current sink policy classifies outbound primarily by URLs/emails in args and memory-write primarily by tool-name patterns.

That is good for v1, but not strong enough for high-consequence use.

### Risk
- outbound-by-IP or host-style args may be missed
- dangerous tools may be misclassified if their names are misleading
- a powerful write-capable tool may be treated too permissively if its metadata is underspecified

## 4.2 Capability narrowing is opt-in
Current safe adapters activate only when their config fields are populated.

### Risk
- a hardened deployment can accidentally run with broad capability exposure because the operator left narrowing config empty
- “secure codebase” becomes weaker than “secure deployment”

## 4.3 Domain-level allowlists are too coarse for some sinks
Allowing an entire domain is sometimes too broad for:
- multi-tenant APIs
- open redirects
- path-sensitive systems
- webhook-style endpoints
- exfiltration via trusted-but-overbroad domains

## 4.4 Path safety is still extraction-based
Current path narrowing uses path extraction heuristics.

### Risk
- bare relative filenames may be treated inconsistently
- tools with path semantics embedded in multiple args may not be constrained as strongly as they should be

## 4.5 Taint tracking is strong but not fully semantic
Current taint tracking is based on extracted values and reuse in later args.

### Risk
- model-derived or transformed malicious destinations may evade direct value matching
- semantic influence can outlive direct string reuse

## 4.6 Dangerous defaults still depend on operator discipline
Even with a strong bridge, catastrophic-risk deployments still require:
- scoped MCP tools
- sandboxing
- constrained network access
- narrow secrets
- narrow filesystem scope

The bridge should do more to demand or validate these conditions.

---

# 5. Hardening Principles

1. **Fail closed, especially in hardened mode**
2. **Prefer typed capability metadata over name inference**
3. **Protect destinations at least as strongly as content**
4. **Treat dangerous tool classes as dangerous even when args look clean**
5. **Reduce operator footguns**
6. **Make unsafe deployment obvious**
7. **Require proof, not just implementation claims**

---

# 6. New Top-Level Concept: Security Profiles

Introduce explicit security profiles.

## Profiles
- `compat`
- `standard`
- `hardened`
- `high_consequence`

## Behavior
### `compat`
- preserves current backward-compatible behavior where reasonable

### `standard`
- today’s secure default posture

### `hardened`
- stronger fail-closed rules
- stricter adapter activation
- more mandatory capability metadata
- more blocking, less notice-only behavior

### `high_consequence`
- intended for the “town burns” class of deployments
- requires stronger config completeness
- disallows broad or ambiguous capability exposure
- enforces external containment expectations

## Recommendation
Add to `[security]`:
- `security_profile = "standard"` by default

For serious deployments:
- `security_profile = "high_consequence"`

---

# 7. PR 10 — Typed Tool Capability Manifest

## Goal
Stop relying so heavily on tool-name patterns and implicit classification.

## Required change
Each tool should carry explicit capability metadata in addition to `ActionClass`.

## New type
`ToolCapabilityManifest`

Fields:
- `network_access: bool`
- `outbound_data_transfer: bool`
- `filesystem_read: bool`
- `filesystem_write: bool`
- `filesystem_delete: bool`
- `memory_write: bool`
- `external_messaging: bool`
- `code_execution: bool`
- `credential_access: bool`
- `user_identity_impact: bool`
- `destructive: bool`
- `high_consequence: bool`

## Source of truth
Capability metadata should come from:
1. explicit config overrides first
2. MCP-side declarations if ever supported
3. conservative bridge inference only as fallback

## Policy
In `high_consequence` profile:
- fallback inference alone must **not** be enough for dangerous tools
- dangerous tools without explicit capability metadata should be blocked from approval

## Acceptance criteria
- no high-risk tool can enter approved state without explicit capability metadata in hardened/high-consequence mode
- sink policy uses capability manifest, not only tool name or arg shape
- audit log records capability metadata at approval time

---

# 8. PR 11 — Destination Policy Engine

## Goal
Replace coarse domain-level trust with richer destination controls.

## New concept
`DestinationPolicy`

Fields:
- `scheme`
- `host`
- `port`
- `path_prefixes`
- `query_constraints`
- `allow_subdomains`
- `allow_ip_literals`
- `allow_private_ranges`
- `allow_redirects`
- `allowed_methods`
- `max_payload_bytes`

## Replace / extend
Current:
- `allowed_outbound_domains`

New:
- `allowed_destinations = [...]`

## Examples
### Simple
Allow:
- `https://api.example.com/v1/ingest`
but not:
- `https://api.example.com/other`
- `https://sub.api.example.com/`
- `http://api.example.com/`

### Strict internal
Allow:
- `https://hooks.company.internal/agent-events`
Disallow:
- any IP literal
- any redirect
- any non-HTTPS scheme

## High-consequence rules
- raw domain-only trust is not sufficient
- IP literals blocked by default
- redirects blocked by default
- private-address destinations require explicit opt-in
- path constraints required for outbound-capable tools

## Acceptance criteria
- outbound-capable tools cannot run in `high_consequence` mode with only domain-wide allowlists
- destination policy is enforced both in sink policy and safe adapters

---

# 9. PR 12 — Outbound Sink Detection Hardening

## Goal
Close outbound classification blind spots.

## Required changes
Outbound detection must include:
- URLs
- emails
- IPs
- hostnames
- webhook-like args
- endpoint / uri / host / recipient / destination patterns
- structured destination fields (`host`, `port`, `scheme`, `path`, `base_url`, etc.)

## Rule
If a tool has network or external messaging capability, it should be treated as an outbound sink even if current args do not obviously include a URL.

This is important because dangerous tools should not be “safe” merely because the destination is implicit or assembled downstream.

## Acceptance criteria
- IP-based exfiltration paths are classified as outbound
- host+port style args are classified as outbound
- outbound-capable tools are treated as outbound sinks even with partial destination args

---

# 10. PR 13 — Stronger Taint Model

## Goal
Move from value propagation only toward value + derivation + influence tracking.

## New type
`InfluenceState`

Fields:
- `direct_value_match: bool`
- `derived_from_untrusted_value: bool`
- `destination_influenced_by_untrusted_content: bool`
- `content_field_influenced_by_untrusted_content: bool`
- `confidence`
- `evidence`

## New rules
### Direct taint
Keep current exact and domain-level propagation.

### Derived taint
Add signals for:
- partial URL reuse
- hostname reuse with changed path
- protocol changes on same host
- copied email local-part/domain recombination
- user-visible “first result / first link / recommended target” style influence patterns

### Accumulation
Persist influence across multiple sequential tool results more explicitly.

## Note
This should still be conservative. If you cannot prove clean provenance for a sensitive destination, default to treating it as influenced in hardened mode.

## Acceptance criteria
- obvious transformed attacker destinations still taint outbound attempts
- multi-step destination derivation still reaches sink policy
- false positives remain acceptable for hardened/high-consequence modes

---

# 11. PR 14 — Path Capability Hardening

## Goal
Replace extraction-based path protection with typed path controls.

## New concept
`PathPolicy`

Fields:
- `allowed_roots`
- `allow_relative_paths`
- `normalize_symlinks`
- `allow_globs`
- `allow_user_home_expansion`
- `read_only`
- `write_only`
- `delete_allowed`
- `extensions_allowlist`
- `filename_pattern_allowlist`

## Required changes
- tool capability manifest must say whether a tool accepts path-bearing args
- path-bearing args should be declared explicitly where possible
- bare relative filenames should not bypass validation
- path validation should operate on normalized candidate paths, not only extracted slash-patterns

## High-consequence rules
- relative paths disabled by default
- symlink resolution must stay within approved roots
- write/delete tools require explicit path policy
- generic filesystem-capable tools without path policy cannot be approved

## Acceptance criteria
- bare filenames are handled deterministically
- symlink escape attempts are blocked
- tools cannot operate on paths outside declared policy even if args are structurally “clean”

---

# 12. PR 15 — Stronger Recipient and Identity Controls

## Goal
Tighten external messaging and user-impacting sinks.

## New controls
### Approved recipients
Keep, but extend to support:
- exact addresses
- approved domains
- named identity groups
- internal-only mode

### Human-targeting tools
Any tool that can:
- send email
- send chat messages
- post comments
- open tickets
- message third parties
should carry `external_messaging=true` in its capability manifest.

## Policy
In hardened/high-consequence mode:
- external messaging always requires explicit recipient policy
- any tainted recipient or tainted body content requires confirmation or block
- first-contact recipients should be blocked by default

## Acceptance criteria
- the model cannot message arbitrary people just because it generated an email address
- third-party content cannot route communication to novel recipients without explicit approval

---

# 13. PR 16 — Hardened Mode Defaults

## Goal
Reduce configuration-dependent footguns.

## New behavior by profile

### In `standard`
Current behavior is mostly acceptable.

### In `hardened`
- first-run approval required
- auto-approve first-seen forbidden
- empty allowlist means none
- outbound adapters active if outbound-capable tools exist
- path adapters active if path-capable tools exist
- recipient adapters active if external-messaging tools exist

### In `high_consequence`
- explicit capability manifest required for dangerous tools
- destination policy required for outbound-capable tools
- path policy required for filesystem tools
- recipient policy required for external messaging tools
- memory-write tools disabled unless explicitly allowed
- broad fallback heuristics cannot be the final authority

## Acceptance criteria
- high-risk tool categories cannot be used with incomplete safety policy in `high_consequence` mode
- the bridge refuses startup or scan approval when required hardening metadata is missing

---

# 14. PR 17 — Deployment Guardrails

## Goal
Make unsafe deployment harder.

## New startup checks
Add a deployment validation phase that warns or errors on:
- outbound-capable tool with no destination policy
- filesystem-write tool with no path policy
- external-messaging tool with no recipient policy
- destructive tool not explicitly classified
- tool with credential access in non-sandboxed mode
- audit path disabled or unwritable
- registry path shared unsafely across environments

## New config
- `deployment_mode = "local_dev" | "sandboxed" | "high_consequence"`
- `require_network_egress_controls = true`
- `require_filesystem_sandbox = true`
- `require_secret_scoping = true`

## Important
The bridge cannot prove the OS is sandboxed, but it can require the operator to declare the mode and make that declaration visible in logs and audit headers.

In high-consequence mode:
- startup should fail if required deployment assertions are not set

---

# 15. PR 18 — Audit and Forensic Completeness Hardening

## Goal
Make post-incident analysis airtight.

## Add to audit
- capability manifest snapshot
- sink type used for decision
- destination policy match result
- adapter decisions per adapter
- taint evidence summary
- influence evidence summary
- deployment mode
- security profile
- whether the action was allowed due to explicit allow rule vs generic policy

## New invariant
Every exit path from `execute_tool()` must produce:
- an audit event
- a decision reason
- sink policy result if sink policy was evaluated
- adapter result if adapters were evaluated

## Acceptance criteria
- audit meta-tests prove monotonic audit growth for every outcome
- forensic replay can explain why a risky action was allowed, blocked, or gated

---

# 16. PR 19 — High-Consequence E2E and Red-Team Suite

## Goal
Stress the bridge specifically at the “one miss is bad” level.

## New adversarial cases
- IP-based exfiltration instead of URL-based exfiltration
- destination split across host + port + path args
- transformed malicious destination reuse
- open-redirect on an allowed domain
- path escalation via relative filenames
- symlink escape attempts
- first-contact recipient generation
- poisoned memory write attempts
- outbound-capable tool with missing capability manifest
- misclassified powerful tool should fail approval in high-consequence mode

## Model-in-the-loop cases
- model tries to reconstruct destination rather than copy exact string
- model combines two benign tool results into a dangerous sink argument
- model tries to write poisoned memory then retrieve it later
- model tries to use a “safe” allowed domain in a dangerous path

## Acceptance criteria
- red-team suite passes in hardened/high-consequence profiles
- regressions block release

---

# 17. PR 20 — Release Gate for “High-Consequence Ready”

## Goal
Define a release standard stronger than “tests pass.”

## Required gate
A release can be labeled high-consequence-ready only if all are true:

1. capability manifests exist for dangerous tools
2. destination policies exist for outbound-capable tools
3. path policies exist for filesystem tools
4. recipient policy exists for messaging tools
5. first-run approval is required
6. no auto-approval in hardened/high-consequence modes
7. audit completeness invariants pass
8. adversarial sink tests pass
9. live model + adversarial MCP tests pass
10. operator-facing deployment validation passes

## Optional scorecard
Produce a generated hardening report:
- tool inventory
- dangerous capabilities
- policy completeness
- remaining heuristics
- deployment assertions
- adversarial suite status

---

# 18. Recommended Config Additions

Add to `[security]`:

- `security_profile = "standard"`
- `deployment_mode = "local_dev"`
- `require_explicit_capability_metadata = false`
- `require_destination_policy_for_outbound = false`
- `require_path_policy_for_filesystem = false`
- `require_recipient_policy_for_messaging = false`
- `require_network_egress_controls = false`
- `require_filesystem_sandbox = false`
- `require_secret_scoping = false`

Add new sections:

## `[capabilities.<server>.<tool>]`
Example:
```toml
[capabilities.files.write_document]
filesystem_write = true
filesystem_read = false
filesystem_delete = false
outbound_data_transfer = false
memory_write = false
external_messaging = false
destructive = false
high_consequence = true
```

## `[destinations.<server>.<tool>]`
Example:
```toml
[[destinations.webhooks.send_event]]
scheme = "https"
host = "hooks.company.internal"
allow_subdomains = false
allow_ip_literals = false
allow_redirects = false
path_prefixes = ["/agent-events"]
allowed_methods = ["POST"]
max_payload_bytes = 65536
```

## `[paths.<server>.<tool>]`
Example:
```toml
[paths.files.write_document]
allowed_roots = ["/srv/agent-safe"]
allow_relative_paths = false
normalize_symlinks = true
read_only = false
write_only = true
delete_allowed = false
extensions_allowlist = [".txt", ".md", ".json"]
```

## `[recipients.<server>.<tool>]`
Example:
```toml
[recipients.mail.send_email]
approved_addresses = ["ops@example.com", "alerts@example.com"]
approved_domains = ["example.com"]
internal_only = true
```

---

# 19. Suggested Rollout Plan

## Phase A — Typed capability model
- PR 10
- PR 16

## Phase B — Stronger sink and destination controls
- PR 11
- PR 12
- PR 13

## Phase C — Filesystem and messaging hardening
- PR 14
- PR 15

## Phase D — Operational hardening
- PR 17
- PR 18

## Phase E — “.999” validation
- PR 19
- PR 20

---

# 20. Brutal Success Criteria

I would personally stop saying “it’s strong, but...” and start saying “this is about as serious as you can reasonably make this class of system” when all of the below are true:

1. dangerous tools require explicit capability metadata
2. outbound-capable tools require explicit destination policy
3. path-capable tools require explicit path policy
4. messaging tools require explicit recipient policy
5. outbound sink detection covers URLs, emails, IPs, host/port forms, and implicit-capability tools
6. transformed attacker destinations still taint sink decisions
7. high-consequence mode refuses incomplete or ambiguous safety policy
8. audit logs can fully explain every allow/block/gate decision
9. adversarial model-in-the-loop tests cover transformed, split, and indirect sink attacks
10. the deployment checklist requires external containment, not just bridge-level logic

---

# 21. Final Summary

The current bridge is already a strong v1.

This spec is the pass that moves it toward “.999” by:

- reducing heuristic reliance,
- reducing config footguns,
- making dangerous capability policy explicit,
- tightening destination and path controls,
- strengthening taint into influence-aware source-to-sink protection,
- and raising the release bar from “works securely” to “hard to deploy unsafely.”

That is the next level.
