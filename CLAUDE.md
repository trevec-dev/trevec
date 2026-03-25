# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Trevec (trevec.dev) is a local-first, high-performance context infrastructure for AI-assisted software development. It replaces chunk-based RAG with deterministic code structure plus fast hybrid retrieval (BM25 + local embeddings), delivering accurate context retrieval for developer tools and AI agents.

**Primary users:** Engineers using Claude Desktop, Cursor, Zed, and MCP-capable agents.

**Core thesis:** Code is a structured graph. Deterministic structure plus fast hybrid recall beats chunk-based RAG for code. Speed is the viral lever.

## Architecture: Reflex vs Brain

The system is built around a strict separation:

- **Reflex (critical path, always-on):** Tree-sitter AST extraction, BM25 full-text search, fast local embeddings over code-derived text, deterministic graph expansion (callers/callees/imports), token-budgeted context assembly. This must work well on its own.
- **Brain (optional, async enrichment):** LLM summaries, deeper embeddings, episodic memory, rule extraction. Must never block indexing or retrieval.

**Non-negotiable:** Trevec must work well with Brain disabled. Enrichment must never block the Reflex path.

## Three-Layer Architecture

1. **Structural layer (deterministic):** Parse files into ASTs via Tree-sitter, extract logical units (functions, methods, classes, modules), build a directed graph of relationships (imports, calls, contains, implements, inherits) with confidence levels.

2. **Retrieval layer (hybrid):** BM25 over code-derived text fields + vector search over compact local embeddings (signature + identifiers + doc comments). Results merged via Reciprocal Rank Fusion (RRF).

3. **Assembly layer (deterministic):** Select anchor nodes from merged retrieval, expand graph neighborhood under a token budget (preferring high-confidence edges), format output as a context bundle with file paths and spans for citations.

## Data Model

- **CodeNode:** id, kind, file_path, span, signature, doc_comment, identifiers, bm25_text, symbol_vec (embedding), code_vec (optional), ast_hash, intent_summary (optional Brain output).
- **Edge:** src_id, dst_id, edge_type (import/call/contain/implement/inherit), confidence (certain/likely/unknown).
- **ContextBundle:** bundle_id, query, anchor_node_ids, included_node_ids with spans, redaction_stats, explainability metadata.

## Key Technology Choices

- **Tree-sitter** for AST parsing and language-specific queries
- **LanceDB** for vector store + BM25 index with metadata filtering
- **MCP server** over stdio transport (tools: get_context, search_code, read_file_topology)
- **Local embeddings** — CPU-friendly, small model; no cloud dependency
- **In-memory graph** with periodic serialization for edge/relationship storage

## Ingestion Pipeline

1. Walk repo respecting .gitignore + Trevec excludes
2. Parse with Tree-sitter using language-specific queries
3. Extract CodeNodes with stable IDs and accurate spans
4. Build bm25_text from code-derived signals: `file_path + signature + identifiers + doc_comment`
5. Compute symbol_vec embedding and upsert into LanceDB
6. Build edges, update graph, persist snapshots
7. Queue optional Brain enrichment (non-blocking)

**Incremental updates:** Track file hash and per-node ast_hash; only update affected nodes/edges. Use Tree-sitter edit+reparse to minimize work.

**Dynamic languages:** Store edges with lower confidence when resolution is ambiguous. Prefer same-file/same-module resolution first.

## Retrieval Pipeline

1. Run BM25 + vector search in parallel
2. Merge with RRF
3. Select top 3-5 anchor nodes
4. Expand graph neighborhood under token budget (prefer certain > likely > unknown edges)
5. Build context bundle with citations
6. Return structured context via MCP tools

## CLI Commands (Planned)

- `trevec init` — initialize Trevec for a repository
- `trevec index` — index the codebase
- `trevec ask` — query the codebase

## Performance Targets

- Time-to-Magic: < 60 seconds from install to first correct answer
- Retrieval P95: < 50ms on 100k LoC
- Indexing: 100k LoC under 5 seconds on a modern laptop
- Zero network egress by default

## Design Constraints

- All config changes must be opt-in and reversible
- Logs go to stderr to preserve MCP stdout protocol
- No source code content in telemetry (opt-in only)
- Enrichment prompts treat repo content as untrusted data
- Secrets scanner and redaction before returning context (planned)

## Reference Documentation

Detailed design docs are in the parent directory (`../`):
- `Trevec - V1.md` — full V1 specification
- `Trevec_Implementation_and_Virality_Plan.md` — implementation plan, data model details, and metrics
- `Trevec_Proposal.md` — executive summary, risks, and go/no-go gates

<!-- trevec:rules:start -->

## Trevec MCP Tools

Use these MCP tools to retrieve precise, graph-aware code context instead of reading files manually.

### get_context
Retrieves relevant code context for a natural-language query. Returns relevant code nodes with file paths, spans, and related context. **Use this as your primary tool for understanding code.**

### search_code
Hybrid search over indexed code nodes. Returns ranked results with file paths and signatures. Use for targeted symbol or keyword lookup.

### read_file_topology
Returns the structural topology of a file: all code nodes (functions, classes, methods) with their relationships (calls, imports, contains). Use to understand file structure before making changes.

### remember_turn
Records a conversation turn into episodic memory. Call this when the user shares important context, decisions, or preferences that should persist across sessions.

### recall_history
Searches episodic memory for past conversation context. Use when the user references previous discussions or when historical context would help answer a question.

### repo_summary
Returns a high-level overview of the repository: languages, file/node/edge counts, top-level modules, entry points (most-called functions), hotspots (most-connected nodes), and detected conventions. Use for onboarding or getting a quick sense of a codebase.

### neighbor_signatures
Given a list of file paths, returns the external API surface those files depend on — imported symbols from other files with their signatures. Use to understand what a file interacts with before modifying it.

### batch_context
Runs multiple `get_context` queries in a single call. Each query can have its own budget and anchor count. Use to reduce round-trips when you need context for several related questions.

### Guidelines
- Prefer `get_context` over reading raw files — it returns only the relevant code with graph context.
- Use `search_code` for quick symbol lookups (function names, class names, error messages).
- Use `read_file_topology` before modifying a file to understand its structure and dependencies.
- Use `repo_summary` for onboarding or to get a quick overview of the codebase structure.
- Use `neighbor_signatures` to discover imports/dependencies of specific files before editing.
- Use `batch_context` when you need context for multiple queries — saves round-trips.
- Call `remember_turn` for important decisions, preferences, or context the user shares.
- Call `recall_history` when the user says "we discussed", "last time", or references prior work.

<!-- trevec:rules:end -->
