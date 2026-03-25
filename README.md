<p align="center">
  <img src="https://trevec.dev/trevec-icon.png" alt="Trevec" width="80" />
</p>

<h1 align="center">Trevec</h1>

<p align="center">
  <strong>Persistent memory for AI agents and coding tools.</strong><br/>
  No API key. No cloud. Sub-50ms retrieval.
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License" /></a>
  <a href="https://github.com/trevec-dev/trevec/actions/workflows/ci.yml"><img src="https://github.com/trevec-dev/trevec/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="https://pypi.org/project/trevec/"><img src="https://img.shields.io/pypi/v/trevec?color=blue&label=PyPI" alt="PyPI" /></a>
  <a href="https://www.npmjs.com/package/trevec"><img src="https://img.shields.io/npm/v/trevec?color=blue&label=npm" alt="npm" /></a>
  <a href="https://crates.io/crates/trevec"><img src="https://img.shields.io/crates/v/trevec?color=blue&label=crates.io" alt="crates.io" /></a>
</p>

<p align="center">
  <a href="https://playground.trevec.dev"><strong>Try the Playground</strong></a> · <a href="https://docs.trevec.dev">Docs</a> · <a href="#use-case-1-mcp-server-for-your-ide">MCP Server</a> · <a href="#use-case-2-sdk-for-your-ai-apps">SDK</a>
</p>

---

> **New:** Try Trevec instantly in your browser at **[playground.trevec.dev](https://playground.trevec.dev)** — no install required.

---

Trevec gives AI agents **fast, structured memory** — for code, conversations, documents, or anything. It runs 100% on your machine, retrieves context in under 50ms, and costs nothing per query.

Two ways to use it:

| | **MCP Server** | **SDK** |
|---|---|---|
| **For** | Developers using AI coding tools | Developers building AI apps |
| **How** | One command installs into your IDE | `pip install trevec` or `npm install trevec` |
| **What it does** | Your AI assistant understands your entire codebase and remembers past conversations | Your AI agents get persistent memory with per-user isolation |
| **Works with** | Claude Code, Cursor, Windsurf, Codex, Zed, VS Code | Any Python or Node.js app — LangChain, CrewAI, custom agents |

---

## Use Case 1: MCP Server for Your IDE

**Problem:** AI coding tools read files one at a time, burn tokens re-discovering your codebase, and forget everything between sessions.

**Solution:** Trevec indexes your codebase into a structural graph and serves it to your AI assistant via MCP. One tool call replaces 6+ file reads.

```
┌──────────────────────────────────────────────────────────┐
│              Your IDE                                     │
│   Claude Code · Cursor · Windsurf · Codex · Zed          │
└────────────────────────┬─────────────────────────────────┘
                         │  MCP Protocol
                         ▼
┌──────────────────────────────────────────────────────────┐
│                   Trevec MCP Server                       │
│                                                           │
│   "How does auth work?"                                   │
│         │                                                 │
│         ▼                                                 │
│   ┌───────────┐    ┌──────────┐    ┌───────────────────┐ │
│   │  Parse    │───▶│  Search  │───▶│  Assemble Context │ │
│   │  17 langs │    │  <50ms   │    │  with graph edges │ │
│   └───────────┘    └──────────┘    └───────────────────┘ │
│         │                                                 │
│         ▼                                                 │
│   Compact context bundle with file paths,                 │
│   line ranges, and cross-file relationships               │
│                                                           │
│   + Episodic memory (remembers past sessions)             │
└──────────────────────────────────────────────────────────┘
```

### Quick Start

```bash
# Install
curl -fsSL dl.trevec.dev/install.sh | sh    # macOS/Linux
irm dl.trevec.dev/install.ps1 | iex         # Windows

# Setup (one-time — configures all your IDEs)
trevec mcp setup

# Index a project
cd your-project
trevec init     # done — your AI assistant now has full codebase context
```

### What You Get

| Without Trevec | With Trevec |
|---|---|
| 6+ tool calls to understand a feature | **1 tool call** |
| ~30s waiting for file reads | **~2s** |
| ~32K tokens burned per question | **~4K tokens** (87% savings) |
| AI forgets everything between sessions | **Persistent memory** across sessions |

> At 4 queries/hour on Claude Opus, that's **~$26/day saved** per developer.

### Available MCP Tools

| Tool | What It Does |
|------|-------------|
| `get_context` | Ask a question, get relevant source code with file paths and relationships |
| `search_code` | Find functions, classes, or patterns across your codebase |
| `read_file_topology` | See who calls a function, what it imports, what inherits from it |
| `remember_turn` | Save debugging sessions, architecture decisions, code reviews to memory |
| `recall_history` | Search past conversations — "did we discuss this before?" |
| `get_file_history` | See prior discussions and decisions about a specific file |
| `repo_summary` | Get a high-level overview of your codebase |
| `batch_context` | Run multiple queries in a single call |

---

## Use Case 2: SDK for Your AI Apps

**Problem:** Your AI agents have no memory. Every conversation starts from scratch. Cloud memory SDKs add latency, cost, and vendor lock-in.

**Solution:** Trevec gives your agents persistent, per-user memory that runs locally — like SQLite for agent memory.

```
┌──────────────────────────────────────────────────────────┐
│              Your AI Application                          │
│   Support bot · Tutor · Health app · Coding agent         │
└────────────────────────┬─────────────────────────────────┘
                         │  SDK (Python / Node.js / Rust)
                         ▼
┌──────────────────────────────────────────────────────────┐
│                    Trevec Engine                           │
│                                                           │
│   tv.add("Allergic to penicillin", user_id="patient_1")  │
│   tv.search("medication allergies", user_id="patient_1") │
│         │                                                 │
│         ▼                                                 │
│   ┌─────────────────────────────────────────────────────┐ │
│   │  Per-user isolation · Semantic search · <50ms       │ │
│   │  Persistent storage · Works offline · Zero cost     │ │
│   └─────────────────────────────────────────────────────┘ │
│                                                           │
│   Also indexes code repos for coding agents               │
└──────────────────────────────────────────────────────────┘
```

### Python

```bash
pip install trevec
```

```python
from trevec import Trevec

tv = Trevec("my-app")

# Add memories — isolated per user
tv.add("Allergic to penicillin", user_id="patient_1")
tv.add("Prefers morning appointments", user_id="patient_1")

# Semantic search
results = tv.search("medication allergies", user_id="patient_1")
# → [{"memory": "Allergic to penicillin", "score": 0.94, ...}]

# Works with conversation messages too
tv.add_messages([
    {"role": "user", "content": "I just moved to Seattle"},
    {"role": "assistant", "content": "I'll remember you're in Seattle!"}
], user_id="user_42")
```

### Node.js

```bash
npm install trevec
```

```javascript
import { Trevec } from "trevec";

const tv = new Trevec("my-app");

tv.add("Customer prefers email over phone", { userId: "cust_1" });
const results = tv.search("contact preferences", { userId: "cust_1" });
```

### For Coding Agents

```python
# Index a codebase and query it programmatically
tv = Trevec.for_repo("/path/to/project")
tv.index()
result = tv.query("How does the payment flow work?", budget=4096)
```

### Why Not a Cloud SDK?

| | Cloud Memory SDKs | Trevec |
|---|---|---|
| **API key** | Required | Not needed |
| **Data location** | Their servers | Your machine |
| **Latency** | 200-700ms | **<50ms** |
| **Cost per query** | $0.01-0.05 | **$0** |
| **Works offline** | No | **Yes** |
| **User isolation** | Varies | **Built-in** |

---

## Supported Languages

Rust · Python · JavaScript · TypeScript · Go · Java · C · C++ · C# · Ruby · Swift · Bash · Lua · Zig · JSON · HTML · CSS

## Building from Source

```bash
git clone https://github.com/trevec-dev/trevec.git
cd trevec
cargo build --release
```

### Project Structure

```
trevec/
├── crates/
│   ├── trevec-core          # Data model and graph types
│   ├── trevec-parse         # Language-aware parsing (17 languages)
│   ├── trevec-index         # Indexing, graph building, memory store
│   ├── trevec-retrieve      # Search and context assembly
│   ├── trevec-brain         # Optional async LLM enrichment
│   ├── trevec-sdk           # Unified Rust SDK
│   ├── trevec-python        # Python bindings (PyO3)
│   ├── trevec-node          # Node.js bindings (napi-rs)
│   └── trevec-cli           # CLI + MCP server
├── sdks/
│   ├── node/                # npm package
│   ├── node-native/         # npm native bindings
│   └── python/              # PyPI package
└── fixtures/                # Test fixtures
```

### Running Tests

```bash
cargo test --workspace
```

## Contributing

We welcome contributions! Please see [CONTRIBUTING.md](CONTRIBUTING.md) for guidelines.

## License

Apache 2.0 — see [LICENSE](LICENSE).

---

<p align="center">
  <a href="https://playground.trevec.dev"><strong>Try the Playground</strong></a> · <a href="https://docs.trevec.dev">Read the Docs</a> · <a href="https://trevec.dev">trevec.dev</a>
</p>
