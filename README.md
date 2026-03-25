# Trevec

**Local-first context infrastructure for AI-assisted development.**

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache%202.0-blue.svg)](LICENSE)
[![CI](https://github.com/trevec/trevec/actions/workflows/ci.yml/badge.svg)](https://github.com/trevec/trevec/actions/workflows/ci.yml)

Trevec replaces chunk-based RAG with deterministic code structure plus fast hybrid retrieval (BM25 + local embeddings). It parses your codebase into a structural graph using Tree-sitter, indexes it in LanceDB, and serves accurate context to AI tools via the Model Context Protocol (MCP).

## Quick Start

```sh
curl -fsSL dl.trevec.dev/install.sh | sh   # install
trevec mcp setup                            # one-time: configure your IDE
cd your-project
trevec init                                 # index + rules, ready to go
```

Or build from source:

```sh
cargo install trevec
trevec mcp setup
cd your-project
trevec init
```

## Commands

| Command | Description |
|---------|-------------|
| `trevec init` | Initialize, index, and write IDE rules for a repository |
| `trevec init --no-index` | Just create `.trevec/` config files (customize before indexing) |
| `trevec index` | Index the codebase (incremental on re-runs) |
| `trevec ask <query>` | Query the index for relevant context |
| `trevec inspect` | Inspect the index (node/edge counts, node details) |
| `trevec serve` | Start the MCP server over stdio |
| `trevec watch` | Watch for file changes and re-index automatically |
| `trevec mcp setup` | One-time: configure MCP integration globally for all IDEs |
| `trevec mcp remove` | Remove global MCP configuration |
| `trevec mcp doctor` | Check MCP setup health |

## MCP Integration

Trevec runs as an MCP server, providing tools to AI assistants:

- **`get_context`** - Full retrieval with graph expansion under a token budget
- **`search_code`** - Lightweight ranked search across the codebase
- **`read_file_topology`** - Graph introspection for a specific code node
- **`remember_turn`** / **`recall_history`** - Episodic memory across sessions

### Global Setup (recommended)

```sh
trevec mcp setup        # configures all IDEs at once
```

This writes global MCP configs for Claude Desktop, Claude Code, Cursor, and Codex. The config is simple — no hardcoded paths:

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

Trevec's `serve` command automatically detects the repository from the working directory set by the IDE.

### Per-Client Setup

```sh
trevec mcp setup --client cursor       # only Cursor
trevec mcp setup --client claude-code   # only Claude Code
```

To remove: `trevec mcp remove`

## How It Works

Trevec uses a three-layer architecture:

1. **Structural layer (deterministic)** - Parses files into ASTs via Tree-sitter, extracts logical units (functions, methods, classes, modules), and builds a directed graph of relationships (imports, calls, contains).

2. **Retrieval layer (hybrid)** - BM25 full-text search over code-derived text fields plus vector search over local embeddings. Results merged via Reciprocal Rank Fusion (RRF).

3. **Assembly layer (deterministic)** - Selects anchor nodes from merged retrieval, expands the graph neighborhood under a token budget, and formats output as a context bundle with file paths and spans.

## Configuration

After `trevec init`, edit `.trevec/config.toml`:

```toml
[index]
# Glob patterns to exclude from indexing (in addition to .gitignore)
exclude = ["vendor/**", "*.generated.*"]

[retrieval]
# Number of anchor nodes for context assembly
anchors = 5
# Token budget for context assembly
budget = 4096

[embeddings]
# Model name for local embeddings
model = "BAAI/bge-small-en-v1.5"
```

All settings are optional and fall back to sensible defaults.

## Supported Languages

Rust, Python, JavaScript, TypeScript, Go, Java, C, C++, Ruby, Bash, JSON, HTML, CSS, C#, Lua, Zig, Swift.

Trevec recognizes the following extensions: `.rs`, `.py`, `.pyi`, `.pyw`, `.js`, `.mjs`, `.cjs`, `.ts`, `.tsx`, `.go`, `.java`, `.c`, `.h`, `.cpp`, `.cc`, `.cxx`, `.hpp`, `.hxx`, `.hh`, `.rb`, `.sh`, `.bash`, `.json`, `.html`, `.htm`, `.css`, `.cs`, `.lua`, `.zig`, `.swift`.

## Building from Source

```sh
git clone https://github.com/trevec/trevec.git
cd trevec
cargo build --release
```

The binary will be at `target/release/trevec`.

### Running Tests

```sh
cargo test --workspace
```

Three tests are ignored by default (they require downloading the embedding model). To run them:

```sh
cargo test --workspace -- --ignored
```

## License

Apache 2.0
