<p align="center">
  <img src="https://trevec.dev/trevec-icon.png" alt="Trevec" width="80" />
</p>

<h1 align="center">Trevec</h1>

<p align="center">
  <strong>The memory layer for AI agents. For code, conversations, documents, or anything.</strong><br/>
  No API key. No cloud. Sub-50ms.
</p>

<p align="center">
  <a href="LICENSE"><img src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" alt="License" /></a>
  <a href="https://github.com/trevec-dev/trevec/actions/workflows/ci.yml"><img src="https://github.com/trevec-dev/trevec/actions/workflows/ci.yml/badge.svg" alt="CI" /></a>
  <a href="https://pypi.org/project/trevec/"><img src="https://img.shields.io/pypi/v/trevec?color=blue&label=PyPI" alt="PyPI" /></a>
  <a href="https://www.npmjs.com/package/trevec"><img src="https://img.shields.io/npm/v/trevec?color=blue&label=npm" alt="npm" /></a>
  <a href="https://crates.io/crates/trevec"><img src="https://img.shields.io/crates/v/trevec?color=blue&label=crates.io" alt="crates.io" /></a>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> · <a href="#why-trevec">Why Trevec</a> · <a href="#mcp-server">MCP Server</a> · <a href="#sdk">SDK</a> · <a href="https://docs.trevec.dev">Docs</a> · <a href="https://playground.trevec.dev">Playground</a>
</p>

---

Trevec is a **local-first, Rust-powered context engine** that gives AI agents structured memory and code understanding. Use it as an **MCP server** for your IDE, as an **SDK** in your AI apps, or both — same engine, same memory.

```
┌─────────────────────────────────────────────────────────────────┐
│                        Your AI Agent                            │
│           (Claude Code, Cursor, Codex, your own app)            │
└──────────────────────────┬──────────────────────────────────────┘
                           │
              ┌────────────┴────────────┐
              │                         │
        MCP Protocol               SDK (direct)
        (stdio transport)          Python · Node.js · Rust
              │                         │
              └────────────┬────────────┘
                           │
┌──────────────────────────▼──────────────────────────────────────┐
│                      Trevec Engine                              │
│                                                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │  Language-   │  │   Hybrid     │  │   Universal Context    │ │
│  │  Aware       │  │   Search     │  │       Graph            │ │
│  │  AST Parser  │  │  <50ms P95   │  │  5 domains · 23 edges  │ │
│  └─────────────┘  └──────────────┘  └────────────────────────┘ │
│                                                                 │
│  ┌─────────────┐  ┌──────────────┐  ┌────────────────────────┐ │
│  │  Episodic   │  │    Brain     │  │  Graph-Aware           │ │
│  │  Memory     │  │  (async LLM) │  │  Context Assembly      │ │
│  └─────────────┘  └──────────────┘  └────────────────────────┘ │
└─────────────────────────────────────────────────────────────────┘
            100% local · Zero data egress · Apache 2.0
```

## Quick Start

**Install the CLI:**

```bash
# macOS / Linux
curl -fsSL dl.trevec.dev/install.sh | sh

# Windows
irm dl.trevec.dev/install.ps1 | iex

# Or build from source
cargo install trevec
```

**Set up MCP (one-time) and index your project:**

```bash
trevec mcp setup          # configures Claude Code, Cursor, Codex, etc.
cd your-project
trevec init               # indexes codebase, writes IDE rules — done
```

**Or use the SDK:**

```python
pip install trevec
```

```python
from trevec import Trevec

tv = Trevec("my-app")
tv.add("User prefers dark mode and metric units", user_id="user_42")
results = tv.search("what theme does the user like?", user_id="user_42")
```

## Why Trevec

### vs. Traditional RAG

Traditional RAG treats code like a PDF — flat text chunks with keyword matching. Trevec understands code as a **structural graph**.

```
Traditional RAG                          Trevec
─────────────                            ──────
Flat text chunks                         AST-aware nodes (functions, classes, modules)
Keyword matching only                    Structural graph traversal
Minutes to re-index                      Millisecond incremental updates
Siloed file reads                        Cross-module relationship mapping
Cloud APIs required                      100% local (zero egress)
```

### vs. Cloud Memory SDKs

Cloud memory solutions add latency, cost, and vendor lock-in to every operation. Trevec runs locally — like **SQLite for agent memory**.

| | Cloud SDKs | Trevec |
|---|---|---|
| **API key required** | Yes | No |
| **Data location** | Their servers | Your machine |
| **Retrieval latency** | 200-700ms | **<50ms** |
| **Cost per operation** | $0.01-0.05 | **$0** |
| **Works offline** | No | **Yes** |
| **User isolation** | Varies | **Built-in** |
| **Code understanding** | No | **Yes** |

### Benchmarks

Real-world test: *"How does the planner agent work?"* on a 50K LoC codebase.

| Metric | Trevec | Traditional | |
|--------|--------|-------------|---|
| **Tool calls** | 1 | 6+ | 6x fewer |
| **Time to answer** | ~2s | ~30s | 15x faster |
| **Tokens consumed** | ~4K | ~32K | 87% fewer |
| **Files read manually** | 0 | 5+ | Fully automated |

> At 4 queries/hour over an 8-hour workday on Claude Opus, that's **~$26/day saved** per developer.

## How It Works

```
 Query: "How does authentication work?"
                    │
                    ▼
 ┌──────────────────────────────────┐
 │  1. PARSE                        │   Language-aware AST extraction
 │                                  │   Functions, classes, modules → nodes
 │  auth.rs ──→ [login()] [verify()]│   Imports, calls, inheritance → edges
 │  user.rs ──→ [User] [Session]   │   17 languages supported
 └──────────────┬───────────────────┘
                │
                ▼
 ┌──────────────────────────────────┐
 │  2. RETRIEVE                     │   Hybrid search across the graph
 │                                  │   Ranked by structure + meaning
 │  login() ──── 0.92              │   All local, sub-50ms
 │  verify() ─── 0.87              │
 │  Session ──── 0.71              │
 └──────────────┬───────────────────┘
                │
                ▼
 ┌──────────────────────────────────┐
 │  3. ASSEMBLE                     │   Graph-aware context expansion
 │                                  │   Follows imports, calls, inheritance
 │  login() + verify() + Session   │   Budget-managed output
 │  + their imports & callers      │   File paths, line ranges, citations
 └──────────────────────────────────┘
                │
                ▼
      Compact, structured context
      ready for any AI agent
```

## MCP Server

Trevec runs as an [MCP](https://modelcontextprotocol.io/) server, giving AI assistants structured access to your codebase and memory.

**Works with:** Claude Code · Cursor · Windsurf · Zed · VS Code · Claude Desktop · Codex

```bash
trevec mcp setup    # one-time global config for all IDEs
```

This writes a simple config — no hardcoded paths:

```json
{
  "mcpServers": {
    "trevec": {
      "command": "trevec",
      "args": ["serve"]
    }
  }
}
```

### MCP Tools

| Tool | Description |
|------|-------------|
| **`get_context`** | Primary retrieval — returns relevant source code with file paths, line ranges, and relationships |
| **`search_code`** | Quick symbol lookup — find functions, classes, patterns across the codebase |
| **`read_file_topology`** | Inspect structure — who calls a function, what a module imports, what inherits from a base |
| **`remember_turn`** | Save to memory — debugging sessions, architecture decisions, code reviews |
| **`recall_history`** | Search memory — find past discussions relevant to current work |
| **`summarize_period`** | Recap — bullet-point summary of work within a time range |
| **`get_file_history`** | File context — prior discussions, bugs, and decisions for a specific file |
| **`repo_summary`** | Codebase overview — languages, modules, entry points, conventions |
| **`neighbor_signatures`** | Dependency surface — type signatures of imports for a set of files |
| **`batch_context`** | Multi-query — run up to 10 context queries in a single call |
| **`reindex`** | Refresh — re-index when results seem stale or after major changes |

## SDK

**One SDK, every use case.** Use Trevec as persistent memory in your AI apps — customer support, tutoring, health, fintech, or any domain where agents need to remember.

### Python

```bash
pip install trevec
```

```python
from trevec import Trevec

# Persistent memory (data survives restarts)
tv = Trevec("my-app")

# Add memories with user isolation
tv.add("Allergic to penicillin", user_id="patient_1")
tv.add("Prefers morning appointments", user_id="patient_1")

# Semantic search
results = tv.search("medication allergies", user_id="patient_1")
# → [{"memory": "Allergic to penicillin", "score": 0.94, ...}]

# Mem0-compatible message format
tv.add_messages([
    {"role": "user", "content": "I just moved to Seattle"},
    {"role": "assistant", "content": "I'll remember you're in Seattle!"}
], user_id="user_42")

# Code context (index a repo, then query)
tv = Trevec.for_repo("/path/to/project")
tv.index()
result = tv.query("How does authentication work?", budget=4096)
```

### Node.js

```bash
npm install trevec
```

```javascript
import { Trevec } from "trevec";

const tv = new Trevec("my-app");

// Add and search
tv.add("Customer prefers email over phone", { userId: "cust_1" });
const results = tv.search("contact preferences", { userId: "cust_1" });

// Code context
const repo = Trevec.forRepo("/path/to/project");
repo.index();
const context = repo.query("How does the payment flow work?");
```

### Rust

```rust
use trevec_sdk::TrevecEngine;

let engine = TrevecEngine::new("my-app", Default::default())?;
engine.add("User prefers dark mode", "user_1", None)?;
let results = engine.search("theme preferences", Some("user_1"), 10);
```

### Use Cases

| Use Case | How |
|----------|-----|
| **Customer support** | Recall every order, complaint, preference per customer |
| **AI tutors** | Remember learning style, progress, weak areas per student |
| **Health apps** | Track allergies, conditions, medications — all local, HIPAA-friendly |
| **FinTech** | Store client risk profiles, investment history, compliance notes |
| **Coding agents** | Index codebase + remember debugging sessions and architecture decisions |
| **IDE integration** | Connect via MCP — Claude Code, Cursor, Windsurf, Codex |

## Universal Context Graph

Trevec doesn't just index code. The **Universal Context Graph** unifies code, conversations, documents, structured data, and AI observations into a single queryable graph.

```
┌─────────┐     ┌──────────────┐     ┌───────────┐
│  Code   │────▶│ Conversation │────▶│ Document  │
│         │     │              │     │           │
│ Function│     │  Message     │     │  Section  │
│ Class   │◀────│  Decision    │     │  Fact     │
│ Module  │     │  Preference  │     │  Citation │
└─────────┘     └──────────────┘     └───────────┘
     │                                     │
     ▼                                     ▼
┌─────────┐                         ┌───────────┐
│Structured│                        │Observation│
│         │                         │           │
│ Entity  │                         │ Reflection│
│ Record  │                         │ Pattern   │
│ Event   │                         │ Change    │
└─────────┘                         └───────────┘
```

**5 domains** · **26 node kinds** · **23 edge types** — a unified graph across code, conversations, documents, and more.

## Brain

The **Brain** is Trevec's optional async intelligence layer. It enriches the graph with LLM-generated insights without ever blocking retrieval.

| Worker | What It Does |
|--------|-------------|
| **Intent Summarizer** | Generates structured summaries of code — purpose, inputs, outputs, side effects |
| **Entity Resolver** | Deduplicates entities across domains |
| **Link Predictor** | Predicts missing relationships from usage patterns |
| **Observation Agent** | Watches code changes, generates observations and reflections |

```python
tv = Trevec("my-app", brain=True)
tv.index()
tv.process_brain()   # enrich nodes with LLM summaries
stats = tv.brain_stats()
# → {"nodes_enriched": 142, "cache_hits": 89, "llm_calls": 53, ...}
```

The Brain is **always optional** — Trevec works great without it. Enrichment never blocks the critical path.

## Playground

Try Trevec without installing anything at **[playground.trevec.dev](https://playground.trevec.dev)**.

## Supported Languages

Rust · Python · JavaScript · TypeScript · Go · Java · C · C++ · C# · Ruby · Swift · Bash · Lua · Zig · JSON · HTML · CSS

17 languages with more coming.

## CLI Commands

| Command | Description |
|---------|-------------|
| `trevec init` | Initialize, index, and write IDE rules for a repository |
| `trevec index` | Re-index the codebase (incremental) |
| `trevec ask <query>` | Query the index from the terminal |
| `trevec serve` | Start the MCP server (stdio transport) |
| `trevec watch` | Watch for file changes and re-index automatically |
| `trevec mcp setup` | One-time: configure MCP for all supported IDEs |
| `trevec mcp doctor` | Check MCP setup health |
| `trevec inspect` | Inspect the index (node/edge counts, details) |
| `trevec memory` | Manage episodic memory |
| `trevec projects` | List indexed projects |

## Configuration

After `trevec init`, edit `.trevec/config.toml`:

```toml
[index]
exclude = ["vendor/**", "*.generated.*"]

[retrieval]
anchors = 5       # number of seed results for context assembly
budget = 4096     # token budget for context output
```

All settings are optional with sensible defaults.

## Building from Source

```bash
git clone https://github.com/trevec-dev/trevec.git
cd trevec
cargo build --release
# Binary: target/release/trevec
```

### Project Structure

```
trevec/
├── crates/
│   ├── trevec-core        # Data model, Universal Context Graph types
│   ├── trevec-parse       # Language-aware AST extraction, domain parsers
│   ├── trevec-index       # Indexing pipeline, graph building, memory store
│   ├── trevec-retrieve    # Hybrid search, ranking, context expansion
│   ├── trevec-brain       # Async LLM enrichment (optional)
│   ├── trevec-sdk         # Unified Rust SDK (TrevecEngine)
│   ├── trevec-python      # Python bindings (PyO3)
│   ├── trevec-node        # Node.js bindings (napi-rs)
│   └── trevec-cli         # CLI + MCP server
├── sdks/
│   ├── node/              # npm package (TypeScript wrapper)
│   ├── node-native/       # npm native package (napi-rs binary)
│   └── python/            # PyPI package wrapper
└── fixtures/              # Test fixtures
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
  Built with Rust. Runs on your machine. Your code stays yours.
</p>
