# ollama-mcp-bridge

Security-first bridge connecting local [Ollama](https://ollama.com) models to [MCP](https://modelcontextprotocol.io) tool servers.

Local LLMs have no built-in safety training for tool use. They will follow injected instructions, call unauthorized tools, and pass malicious parameters. This bridge is the security layer between the model and the real world.

## What it does

Gives Ollama models (llama, gemma, nemotron, etc.) access to MCP tools with an 11-step security pipeline on every tool call:

1. **Resolve** - is this an approved tool?
2. **Validate** - JSON Schema + shell injection / path traversal checks
3. **Action gate** - human confirmation for destructive actions
4. **Sink policy** - block if arguments were influenced by untrusted tool results
5. **Safe adapters** - validate URLs, paths, recipients, memory writes against config
6. **Rate limit** - per-server and per-session caps
7. **Execute** - call the MCP server
8. **Sanitize** - strip prompt injection from results before returning to model
9. **Taint track** - record result values for future taint detection
10. **Rate record** - update call counts
11. **Audit** - append-only JSON-L log of every decision

Every exit path (success, block, error, timeout) produces an audit entry. No silent failures.

## What it defends against

- Tool description poisoning (malicious instructions in tool metadata)
- Parameter injection (path traversal, shell metacharacters)
- Prompt injection via tool results (role impersonation, instruction override)
- Rug pull attacks (server changes tool definition after approval)
- Exfiltration via tainted arguments (URLs from tool results reused in outbound calls)
- Resource exhaustion (infinite tool call loops)

## Install

```
pip install -e .
```

Requires Python 3.11+, a running Ollama instance, and at least one MCP server.

## Usage

```python
from ollama_mcp_bridge import Bridge

async with Bridge.from_config("bridge.toml") as bridge:
    result = await bridge.run(
        "List my memories",
        model="llama3.1:8b",
    )
    print(result.content)
```

## Configuration

TOML config. See `bridge.toml` for a full example.

```toml
[bridge]
ollama_host = "http://127.0.0.1:11434"

[servers.sigma-mem]
command = "python"
args = ["-m", "sigma_mem.mcp_server"]
allowed_tools = ["store_memory", "recall", "search_memory"]
destructive_tools = []

[security]
require_first_run_approval = true
auto_approve_first_seen = false
# Capability narrowing (opt-in, empty = disabled)
allowed_outbound_domains = []
allowed_path_roots = []
approved_recipients = []
```

Key defaults:
- Empty `allowed_tools` = no tools available (fail-closed)
- First-seen tools require explicit approval before use
- Tainted outbound calls are blocked
- Memory writes from third-party content are blocked
- All 7 sanitization detectors enabled

## Tests

```
pip install -e ".[dev]"
pytest tests/                        # unit + live MCP (384 tests, ~3s)
pytest tests/test_adversarial.py     # adversarial eval (requires Ollama)
```

The test suite includes adversarial MCP servers that return injection payloads, and live tests with real Ollama models verifying the security pipeline blocks tainted tool calls.

## Architecture

5-layer stack:

| Layer | Component | Role |
|-------|-----------|------|
| 5 | Bridge | Public API |
| 4 | AgentLoop | Multi-turn orchestration |
| 3 | SecurityGateway | All policy enforcement (this is the point) |
| 2 | ToolTranslator | Schema conversion |
| 1 | Transport | Ollama + MCP clients |

SecurityGateway owns the MCP client. There is no code path from model intent to tool execution that bypasses security.

## Limitations

- Sanitization is pattern-based, not semantic. Sophisticated paraphrased injection may evade detection.
- Taint tracking matches extracted values (URLs, domains, emails, IPs). A model that paraphrases or transforms a tainted value won't be caught.
- Safe adapters are opt-in. Without configuration, only taint-based sink policy and sanitization are active.
- Small local models are unpredictable. The bridge constrains what they can *do*, not what they *think*.
- No reconnect re-scan. If an MCP server restarts, cached tool definitions aren't refreshed until the bridge restarts.
- Audit log rotation is not implemented. The audit file grows indefinitely; rotate manually between sessions.

## Hardening checklist

The bridge architecture is defense-in-depth, but the security level depends on how it's configured. For deployments where a breach has real consequences:

**Required:**

- [ ] `require_first_run_approval = true` (default) — never auto-trust first-seen tools
- [ ] `auto_approve_first_seen = false` (default) — no silent approval
- [ ] Every server has an explicit `allowed_tools` list — no empty allowlists on servers you intend to use
- [ ] Destructive tools are listed in `destructive_tools` — requires human confirmation
- [ ] Read-only tools are listed in `read_tools` — correct audit classification

**Strongly recommended:**

- [ ] `allowed_outbound_domains` populated — blocks URLs not on the list
- [ ] `allowed_path_roots` populated — constrains file access to specific directories
- [ ] `approved_recipients` populated — blocks unknown email recipients
- [ ] Adversarial test suite passes against your MCP servers, not just the test fixtures
- [ ] Audit log reviewed after first real session — verify the pipeline is logging what you expect

**Operational:**

- [ ] Rotate audit log between sessions (not automated)
- [ ] Review `bridge.toml` after adding new MCP servers — new tools need classification
- [ ] Run `pytest tests/test_adversarial.py` as part of release discipline, not one-time proof
- [ ] Monitor for MCP server restarts — bridge doesn't re-scan tool definitions automatically

Without this posture, the bridge is well-built but too configurable to be called maximally safe. With it, the remaining risk is "the MCP server itself is malicious" — which is outside the bridge's threat model.

## License

Apache 2.0
