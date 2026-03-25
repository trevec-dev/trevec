use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU32, Ordering};
use std::sync::Arc;
use std::time::{Duration, Instant};

use notify::{Event, EventKind, RecursiveMode, Watcher};
use rmcp::handler::server::router::tool::ToolRouter;
use rmcp::handler::server::wrapper::Parameters;
use rmcp::model::*;
use rmcp::{tool, tool_handler, tool_router, ErrorData as McpError, ServerHandler, ServiceExt};
use serde::{Deserialize, Serialize};
use tokio::sync::RwLock;

use trevec_core::config::MemoryConfig;
use trevec_core::model::{CodeNode, Confidence, Edge, EdgeType, MemoryEvent, QueryStats};
use trevec_core::{TokenBudget, TrevecConfig};
use trevec_index::embedder::Embedder;
use trevec_index::graph::CodeGraph;
use trevec_index::memory;
use trevec_index::memory_store::MemoryStore;
use trevec_index::store::Store;
use trevec_retrieve::bundle::assemble_bundle;
use trevec_retrieve::expander::expand_graph;
use trevec_retrieve::search::{
    apply_file_path_boost, apply_literal_boost, apply_test_file_penalty,
    apply_test_fixture_penalty, extract_file_paths_from_query, filter_noncode_files, rrf_merge,
    RankedResult,
};

// ---------------------------------------------------------------------------
// Tool parameter structs
// ---------------------------------------------------------------------------

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct GetContextParams {
    /// The natural-language query describing what context you need
    #[schemars(description = "The search query describing what context you need")]
    #[serde(alias = "search", alias = "q", alias = "prompt")]
    pub query: String,

    /// Token budget for context assembly (from config or 4096)
    #[schemars(description = "Token budget for context assembly (default from config)")]
    pub budget: Option<u32>,

    /// Number of anchor nodes to select (from config or 5)
    #[schemars(description = "Number of anchor nodes to select (default from config)")]
    pub anchors: Option<u32>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SearchCodeParams {
    /// The search query
    #[schemars(description = "The search query to find code")]
    #[serde(alias = "search", alias = "q", alias = "prompt")]
    pub query: String,

    /// Maximum number of results to return (default: 20)
    #[serde(default = "default_limit")]
    #[schemars(description = "Maximum number of results to return (default: 20)")]
    pub limit: Option<u32>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct ReadFileTopologyParams {
    /// The node ID, symbol name, or file path to inspect
    #[schemars(description = "A node ID, symbol name, or file path (e.g. 'src/auth.rs')")]
    #[serde(alias = "path", alias = "file_path", alias = "file")]
    pub node_id: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct RememberTurnParams {
    /// The role of this turn: "user" or "assistant"
    #[schemars(description = "Role: user, assistant, system, or tool")]
    pub role: String,

    /// The content of this turn
    #[schemars(description = "The content text to remember")]
    pub content: String,

    /// Session identifier (defaults to a generated one)
    #[schemars(description = "Session ID for grouping turns")]
    pub session_id: Option<String>,

    /// Importance score 0-100 (default: 50)
    #[schemars(description = "Importance score 0-100 (default: 50)")]
    pub importance: Option<i32>,

    /// Pin this event to exempt it from garbage collection
    #[schemars(description = "Pin to exempt from GC (default: false)")]
    pub pinned: Option<bool>,

    /// File paths discussed or affected in this turn
    #[schemars(description = "File paths touched in this turn")]
    pub files_touched: Option<Vec<String>>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct RecallHistoryParams {
    /// The search query
    #[schemars(description = "Search query for memory recall")]
    #[serde(alias = "search", alias = "q", alias = "prompt")]
    pub query: String,

    /// Only return events from the last N days
    #[schemars(description = "Limit to events within this many days")]
    #[serde(alias = "days", alias = "time_range")]
    pub time_range_days: Option<u32>,

    /// Maximum number of results (default: 20)
    #[serde(default = "default_limit")]
    #[schemars(description = "Max results to return (default: 20)")]
    pub limit: Option<u32>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct SummarizePeriodParams {
    /// Start timestamp (unix seconds)
    #[schemars(description = "Start time as unix seconds")]
    #[serde(alias = "start", alias = "from")]
    pub start_ts: i64,

    /// End timestamp (unix seconds)
    #[schemars(description = "End time as unix seconds")]
    #[serde(alias = "end", alias = "to")]
    pub end_ts: i64,

    /// Optional focus query to filter events
    #[schemars(description = "Optional query to filter events")]
    #[serde(alias = "query", alias = "q")]
    pub focus: Option<String>,

    /// Max bullet points to return (default: 5)
    #[schemars(description = "Maximum bullet points (default: 5)")]
    #[serde(alias = "limit", alias = "max")]
    pub max_bullets: Option<u32>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct GetFileHistoryParams {
    /// The file path to look up history for
    #[schemars(description = "File path to look up discussion history")]
    #[serde(alias = "path", alias = "file")]
    pub file_path: String,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct RepoSummaryParams {
    /// Maximum tokens for the summary output (default: 500)
    #[schemars(description = "Maximum tokens for the summary output (default: 500)")]
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct NeighborSignaturesParams {
    /// Files this worker owns (repo-relative paths)
    #[schemars(description = "Files this worker owns (repo-relative paths)")]
    #[serde(alias = "files", alias = "owned_files")]
    pub target_files: Vec<String>,

    /// Maximum tokens for the signature output (default: 500)
    #[schemars(description = "Max tokens for signatures (default: 500)")]
    pub max_tokens: Option<u32>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct BatchContextParams {
    /// Array of context queries to execute in batch
    #[schemars(description = "Array of context queries to execute in batch")]
    pub queries: Vec<BatchQuery>,
}

#[derive(Debug, Deserialize, schemars::JsonSchema)]
pub struct BatchQuery {
    /// The natural-language query
    #[schemars(description = "The search query describing what context you need")]
    pub query: String,

    /// Token budget for this query (default from config)
    #[schemars(description = "Token budget for context assembly (default from config)")]
    pub budget: Option<u32>,

    /// Number of anchor nodes (default from config)
    #[schemars(description = "Number of anchor nodes to select (default from config)")]
    pub anchors: Option<u32>,
}

fn default_limit() -> Option<u32> {
    Some(20)
}

/// Log the full error chain and return a sanitized MCP error to the client.
fn user_facing_error<E>(context: &str, err: E) -> McpError
where
    E: std::fmt::Display + std::fmt::Debug,
{
    tracing::error!("{context}: {err}");
    tracing::error!("{context} details: {err:#?}");
    McpError::internal_error(
        format!("{context}. Check server logs for details."),
        None,
    )
}

/// Append a trevec repo marker as a **separate** text content block so the
/// preceding block (typically serialized JSON) stays parseable by strict
/// consumers.  The marker is an invisible HTML comment that persists in
/// Cursor's state.vscdb so the extractor can attribute the composer session
/// to this repo.
fn tag_result_with_repo(mut result: CallToolResult, repo_id: &str) -> CallToolResult {
    let marker = format!("<!-- trevec:repo_id:{repo_id} -->");
    result.content.push(Content::text(marker));
    result
}

// ---------------------------------------------------------------------------
// Response types (serialized to JSON for MCP responses)
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct SearchResultEntry {
    node_id: String,
    name: String,
    kind: String,
    file_path: String,
    signature: String,
    score: f64,
}

#[derive(Debug, Serialize)]
struct TopologyResponse {
    node: TopologyNode,
    outgoing: Vec<TopologyEdge>,
    incoming: Vec<TopologyEdge>,
}

#[derive(Debug, Serialize)]
struct TopologyNode {
    id: String,
    name: String,
    kind: String,
    file_path: String,
    signature: String,
    span: TopologySpan,
}

#[derive(Debug, Serialize)]
struct TopologySpan {
    start_line: usize,
    end_line: usize,
    start_byte: usize,
    end_byte: usize,
}

#[derive(Debug, Serialize)]
struct TopologyEdge {
    node_id: String,
    name: String,
    kind: String,
    edge_type: String,
    confidence: String,
}

#[derive(Debug, Serialize)]
struct RememberResponse {
    event_id: String,
    stored: bool,
    redactions: usize,
}

#[derive(Debug, Serialize)]
struct MemoryResultEntry {
    event_id: String,
    source: String,
    session_id: String,
    role: String,
    created_at: i64,
    content_snippet: String,
    files_touched: Vec<String>,
    score: f64,
}

#[derive(Debug, Serialize)]
struct SummaryBullet {
    timestamp: i64,
    source: String,
    content: String,
}

// ---------------------------------------------------------------------------
// Response types for new swarm tools
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize)]
struct RepoSummaryResponse {
    languages: HashMap<String, usize>,
    total_files: usize,
    total_nodes: usize,
    total_edges: usize,
    top_level_nodes: Vec<SummaryNode>,
    entry_points: Vec<SummaryNode>,
    hotspots: Vec<SummaryNode>,
    conventions: Vec<String>,
    estimated_tokens: usize,
}

#[derive(Debug, Serialize)]
struct SummaryNode {
    name: String,
    kind: String,
    file_path: String,
    signature: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    incoming_calls: Option<usize>,
    #[serde(skip_serializing_if = "Option::is_none")]
    total_connections: Option<usize>,
}

#[derive(Debug, Serialize)]
struct NeighborSignaturesResponse {
    files: Vec<FileImports>,
    estimated_tokens: usize,
}

#[derive(Debug, Serialize)]
struct FileImports {
    file_path: String,
    external_imports: Vec<ExternalSignature>,
}

#[derive(Debug, Serialize)]
struct ExternalSignature {
    source_file: String,
    name: String,
    signature: String,
    kind: String,
}

#[derive(Debug, Serialize)]
struct BatchContextResult {
    query: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    bundle: Option<serde_json::Value>,
    #[serde(skip_serializing_if = "Option::is_none")]
    error: Option<String>,
}

// ---------------------------------------------------------------------------
// TrevecServer
// ---------------------------------------------------------------------------

#[derive(Clone)]
pub struct TrevecServer {
    nodes_map: Arc<RwLock<HashMap<String, CodeNode>>>,
    graph: Arc<RwLock<CodeGraph>>,
    store: Arc<Store>,
    embedder: Arc<std::sync::Mutex<Embedder>>,
    repo_path: Arc<PathBuf>,
    repo_id: Arc<String>,
    data_dir: Arc<PathBuf>,
    default_budget: u32,
    default_anchors: u32,
    config: Arc<TrevecConfig>,
    memory_store: Option<Arc<RwLock<MemoryStore>>>,
    memory_enabled: bool,
    memory_config: Arc<MemoryConfig>,
    /// Tracks when the last incremental memory sync ran (cooldown).
    last_memory_sync: Arc<RwLock<Instant>>,
    /// Monotonic counter for unique remember_turn event IDs within a process.
    remember_counter: Arc<AtomicU32>,
    /// Nanosecond-precision startup timestamp mixed into remember_turn IDs so
    /// the same session_id + turn_index never collides across server restarts.
    session_nonce: String,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl TrevecServer {
    #[allow(clippy::too_many_arguments)]
    pub fn new(
        nodes_map: HashMap<String, CodeNode>,
        graph: CodeGraph,
        store: Store,
        embedder: Embedder,
        repo_path: PathBuf,
        data_dir: PathBuf,
        config: &TrevecConfig,
        memory_store: Option<MemoryStore>,
    ) -> Self {
        let memory_enabled = config.memory.enabled;
        let repo_id = blake3::hash(repo_path.to_string_lossy().as_bytes())
            .to_hex()[..32]
            .to_string();
        Self {
            nodes_map: Arc::new(RwLock::new(nodes_map)),
            graph: Arc::new(RwLock::new(graph)),
            store: Arc::new(store),
            embedder: Arc::new(std::sync::Mutex::new(embedder)),
            repo_path: Arc::new(repo_path),
            repo_id: Arc::new(repo_id),
            data_dir: Arc::new(data_dir),
            default_budget: config.retrieval.budget as u32,
            default_anchors: config.retrieval.anchors as u32,
            config: Arc::new(config.clone()),
            memory_store: memory_store.map(|ms| Arc::new(RwLock::new(ms))),
            memory_enabled,
            memory_config: Arc::new(config.memory.clone()),
            last_memory_sync: Arc::new(RwLock::new(Instant::now())),
            remember_counter: Arc::new(AtomicU32::new(0)),
            session_nonce: {
                let d = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default();
                format!("{}.{}", d.as_secs(), d.subsec_nanos())
            },
            tool_router: Self::tool_router(),
        }
    }

    /// Run an incremental memory sync if the cooldown (60 s) has elapsed.
    ///
    /// Extraction runs on tokio's blocking thread pool via `spawn_blocking`
    /// so that synchronous SQLite / file I/O in the Cursor / Claude Code /
    /// Codex extractors never stalls the async runtime.
    async fn maybe_sync_memory(&self) {
        const COOLDOWN_SECS: u64 = 60;

        if !self.memory_enabled {
            return;
        }

        // Fast-path: cooldown still active.
        {
            let last = self.last_memory_sync.read().await;
            if last.elapsed() < Duration::from_secs(COOLDOWN_SECS) {
                return;
            }
        }

        let Some(mem_arc) = self.memory_store.clone() else {
            return;
        };

        let repo_path = self.repo_path.clone();
        let data_dir = self.data_dir.clone();
        let config = self.memory_config.clone();
        let embedder = self.embedder.clone();
        let graph_arc = self.graph.clone();
        let nodes_arc = self.nodes_map.clone();

        let handle = tokio::runtime::Handle::current();
        let result = tokio::task::spawn_blocking(move || {
            handle.block_on(async {
                // Build file_node_map from in-memory nodes (avoids re-reading nodes.json).
                let file_node_map = {
                    let nodes = nodes_arc.read().await;
                    let mut map: HashMap<String, Vec<String>> = HashMap::new();
                    for node in nodes.values() {
                        map.entry(node.file_path.clone())
                            .or_default()
                            .push(node.id.clone());
                    }
                    map
                };

                let mut emb_guard = if config.semantic {
                    Some(embedder.lock().expect("embedder lock poisoned"))
                } else {
                    None
                };
                let emb_ref = emb_guard.as_deref_mut();

                let mut store = mem_arc.write().await;
                let mut graph = graph_arc.write().await;

                memory::ingest_memory(
                    &repo_path,
                    &data_dir,
                    &config,
                    emb_ref,
                    &mut store,
                    &mut graph,
                    &file_node_map,
                )
                .await
            })
        })
        .await;

        match result {
            Ok(Ok(stats)) => {
                if stats.events_ingested > 0 {
                    tracing::info!(
                        "Incremental memory sync: {} ingested",
                        stats.events_ingested
                    );
                    // Persist graph with any new Discussed edges.
                    let graph = self.graph.read().await;
                    let graph_path = self.data_dir.join("graph.bin");
                    if let Err(e) = graph.save(&graph_path) {
                        tracing::warn!("Failed to save graph after memory sync: {e}");
                    }
                }
            }
            Ok(Err(e)) => tracing::warn!("Incremental memory sync failed: {e}"),
            Err(e) => tracing::warn!("Memory sync task panicked: {e}"),
        }

        // Update cooldown regardless of success/failure to avoid retry storms.
        *self.last_memory_sync.write().await = Instant::now();
    }

    #[tool(
        description = "Retrieve rich code context for a question about this codebase. This is the PRIMARY tool for understanding code — use it when the user asks how something works, wants to understand a feature, or needs context before making changes. Returns relevant source code with file paths, line ranges, and related functions/classes/imports. Prefer this over search_code when deep understanding is needed."
    )]
    async fn get_context(
        &self,
        Parameters(params): Parameters<GetContextParams>,
    ) -> Result<CallToolResult, McpError> {
        let query_text = params.query.trim();
        if query_text.is_empty() {
            return Err(McpError::invalid_params("query must not be empty", None));
        }
        let budget = params.budget.unwrap_or(self.default_budget) as usize;
        if budget == 0 {
            return Err(McpError::invalid_params("budget must be greater than 0", None));
        }
        let anchors = params.anchors.unwrap_or(self.default_anchors) as usize;
        if anchors == 0 {
            return Err(McpError::invalid_params("anchors must be greater than 0", None));
        }

        let bundle = self
            .execute_context_query(query_text, budget, anchors)
            .await?;

        let json = serde_json::to_string_pretty(&bundle)
            .map_err(|e| user_facing_error("Failed to serialize bundle", e))?;

        Ok(tag_result_with_repo(
            CallToolResult::success(vec![Content::text(json)]),
            &self.repo_id,
        ))
    }

    #[tool(
        description = "Search for code symbols in the codebase. Use this for quick lookups — finding a specific function by name, checking if a class exists, or listing matches for a pattern. Returns a ranked list of matches (name, kind, file path, signature) without source code. Faster than get_context but less detailed. Use get_context instead when the user needs to understand how code works or needs surrounding context."
    )]
    async fn search_code(
        &self,
        Parameters(params): Parameters<SearchCodeParams>,
    ) -> Result<CallToolResult, McpError> {
        let query_text = params.query.trim();
        if query_text.is_empty() {
            return Err(McpError::invalid_params("query must not be empty", None));
        }
        let limit = params.limit.unwrap_or(20) as usize;
        if limit == 0 {
            return Err(McpError::invalid_params("limit must be greater than 0", None));
        }

        let start = Instant::now();

        // Embed the query
        let query_vec = self
            .embedder
            .lock()
            .expect("embedder lock poisoned")
            .embed(query_text)
            .map_err(|e| user_facing_error("Failed to embed query", e))?;

        // Run parallel searches (no lock needed — LanceDB handles concurrency)
        let (fts_results, vector_results) = tokio::join!(
            self.store.search_fts(query_text, limit),
            self.store.search_vector(&query_vec, limit),
        );

        let fts_results = fts_results
            .map_err(|e| user_facing_error("FTS search failed", e))?;
        let vector_results = vector_results
            .map_err(|e| user_facing_error("Vector search failed", e))?;

        // Acquire read lock for merge + response building
        let nodes_map = self.nodes_map.read().await;

        // RRF merge
        let mut merged = rrf_merge(&fts_results, &vector_results, 60);

        // Pipeline: filter → literal boost → file path boost → test penalty
        filter_noncode_files(&mut merged, &nodes_map);
        apply_literal_boost(&mut merged, &nodes_map, query_text);
        let extracted_paths = extract_file_paths_from_query(query_text);
        if !extracted_paths.is_empty() {
            apply_file_path_boost(&mut merged, &nodes_map, &extracted_paths, 0.05);
        }
        apply_test_file_penalty(
            &mut merged,
            &nodes_map,
            self.config.retrieval.test_file_penalty,
            &self.config.retrieval.penalty_paths,
        );
        apply_test_fixture_penalty(&mut merged, &nodes_map);

        // Build response entries
        let entries: Vec<SearchResultEntry> = merged
            .iter()
            .take(limit)
            .filter_map(|r| {
                let node = nodes_map.get(&r.node_id)?;
                Some(SearchResultEntry {
                    node_id: r.node_id.clone(),
                    name: node.name.clone(),
                    kind: node.kind.to_string(),
                    file_path: node.file_path.clone(),
                    signature: node.signature.clone(),
                    score: r.score,
                })
            })
            .collect();

        let elapsed_ms = start.elapsed().as_millis() as u64;
        crate::telemetry::capture("mcp_search", serde_json::json!({
            "retrieval_ms": elapsed_ms,
            "results_count": entries.len(),
        }));

        let json = serde_json::to_string_pretty(&entries)
            .map_err(|e| user_facing_error("Failed to serialize results", e))?;

        Ok(tag_result_with_repo(
            CallToolResult::success(vec![Content::text(json)]),
            &self.repo_id,
        ))
    }

    #[tool(
        description = "Inspect the structure and relationships of a specific code element. Use this to explore: who calls a function, what a module imports, what classes inherit from a base, or what a file contains. Accepts a symbol name or file path. Returns the element's details plus all its relationships with other code. Use this after get_context or search_code when you need to trace a specific dependency chain."
    )]
    async fn read_file_topology(
        &self,
        Parameters(params): Parameters<ReadFileTopologyParams>,
    ) -> Result<CallToolResult, McpError> {
        let node_id = params.node_id.trim();
        if node_id.is_empty() {
            return Err(McpError::invalid_params("node_id must not be empty", None));
        }

        // Acquire read locks
        let nodes_map = self.nodes_map.read().await;
        let graph = self.graph.read().await;

        // Find the node by ID, name, or file path
        let normalized_path = normalize_to_repo_relative(node_id, &self.repo_path);
        let node = nodes_map
            .get(node_id)
            .or_else(|| {
                nodes_map
                    .values()
                    .find(|n| n.name == node_id)
            })
            .or_else(|| {
                // Try matching as a file path — return the first module-level
                // node, or the first node in that file.
                let mut file_nodes: Vec<&CodeNode> = nodes_map
                    .values()
                    .filter(|n| n.file_path == normalized_path)
                    .collect();
                file_nodes.sort_by_key(|n| n.span.start_line);
                file_nodes.into_iter().find(|n| {
                    matches!(n.kind.to_string().as_str(), "module" | "class")
                }).or_else(|| {
                    nodes_map.values().find(|n| n.file_path == normalized_path)
                })
            })
            .ok_or_else(|| {
                McpError::invalid_params(
                    format!("Node '{}' not found. Try a node ID from search_code, a function name, or a repo-relative file path.", node_id),
                    None,
                )
            })?;

        let outgoing = graph.outgoing(&node.id);
        let incoming = graph.incoming(&node.id);

        let response = TopologyResponse {
            node: TopologyNode {
                id: node.id.clone(),
                name: node.name.clone(),
                kind: node.kind.to_string(),
                file_path: node.file_path.clone(),
                signature: node.signature.clone(),
                span: TopologySpan {
                    start_line: node.span.start_line,
                    end_line: node.span.end_line,
                    start_byte: node.span.start_byte,
                    end_byte: node.span.end_byte,
                },
            },
            outgoing: outgoing
                .iter()
                .map(|(neighbor_id, edge_type, confidence)| {
                    let neighbor = nodes_map.get(neighbor_id);
                    TopologyEdge {
                        node_id: neighbor_id.clone(),
                        name: neighbor.map(|n| n.name.clone()).unwrap_or_default(),
                        kind: neighbor
                            .map(|n| n.kind.to_string())
                            .unwrap_or_default(),
                        edge_type: edge_type.to_string(),
                        confidence: confidence.to_string(),
                    }
                })
                .collect(),
            incoming: incoming
                .iter()
                .map(|(neighbor_id, edge_type, confidence)| {
                    let neighbor = nodes_map.get(neighbor_id);
                    TopologyEdge {
                        node_id: neighbor_id.clone(),
                        name: neighbor.map(|n| n.name.clone()).unwrap_or_default(),
                        kind: neighbor
                            .map(|n| n.kind.to_string())
                            .unwrap_or_default(),
                        edge_type: edge_type.to_string(),
                        confidence: confidence.to_string(),
                    }
                })
                .collect(),
        };

        let json = serde_json::to_string_pretty(&response)
            .map_err(|e| user_facing_error("Failed to serialize topology", e))?;

        Ok(tag_result_with_repo(
            CallToolResult::success(vec![Content::text(json)]),
            &self.repo_id,
        ))
    }

    #[tool(
        description = "Save a conversation turn to memory for future recall. Call this after completing significant work — debugging sessions, architecture decisions, code reviews, or important discussions. Include files_touched for any files discussed or modified. Higher importance (70-100) preserves memory longer; lower (10-30) may be cleaned up sooner."
    )]
    async fn remember_turn(
        &self,
        Parameters(params): Parameters<RememberTurnParams>,
    ) -> Result<CallToolResult, McpError> {
        if !self.memory_enabled {
            return Err(McpError::internal_error(
                "Memory is disabled. Enable in .trevec/config.toml",
                None,
            ));
        }
        let Some(mem_store) = &self.memory_store else {
            return Err(McpError::internal_error("Memory store not available", None));
        };

        let content = params.content.trim();
        if content.is_empty() {
            return Err(McpError::invalid_params("content must not be empty", None));
        }

        let (scrubbed, redactions) = memory::scrub::scrub(content);
        let now_dur = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default();
        let now = now_dur.as_secs() as i64;

        let session_id = params.session_id.unwrap_or_else(|| {
            format!("mcp_{}.{}", now_dur.as_secs(), now_dur.subsec_nanos())
        });

        let turn_index = self.remember_counter.fetch_add(1, Ordering::Relaxed);

        let repo_id = (*self.repo_id).clone();

        let id_input = format!(
            "trevec_tool_calls|{}|{}|{}",
            session_id, turn_index, self.session_nonce
        );
        let id = blake3::hash(id_input.as_bytes()).to_hex()[..32].to_string();
        let content_hash = blake3::hash(content.as_bytes()).to_hex()[..32].to_string();

        let files_touched = params.files_touched.unwrap_or_default();
        let bm25_text = format!(
            "trevec_tool_calls {} {}",
            files_touched.join(" "),
            scrubbed
        );

        let mut event = MemoryEvent {
            id: id.clone(),
            repo_id,
            source: "trevec_tool_calls".to_string(),
            session_id,
            turn_index,
            role: params.role,
            event_type: "turn".to_string(),
            content_redacted: scrubbed,
            content_hash,
            created_at: now,
            importance: params.importance.unwrap_or(50),
            pinned: params.pinned.unwrap_or(false),
            files_touched: files_touched.clone(),
            tool_calls: vec![],
            bm25_text,
            symbol_vec: None,
        };

        // Embed if possible
        if let Ok(vec) = self.embedder.lock().expect("embedder lock poisoned").embed(&event.bm25_text) {
            event.symbol_vec = Some(vec);
        }

        // Upsert
        {
            let mut store = mem_store.write().await;
            store
                .upsert_events(&[event])
                .await
                .map_err(|e| user_facing_error("Failed to store memory event", e))?;
        }

        // Create Discussed edges and persist graph to disk
        if !files_touched.is_empty() {
            let nodes_map = self.nodes_map.read().await;
            let mut graph = self.graph.write().await;
            let mut edges_added = false;
            for file in &files_touched {
                // Normalize to repo-relative path for matching
                let rel = normalize_to_repo_relative(file, &self.repo_path);
                for node in nodes_map.values() {
                    if node.file_path == rel {
                        graph.add_edge(&Edge {
                            src_id: id.clone(),
                            dst_id: node.id.clone(),
                            edge_type: EdgeType::Discussed,
                            confidence: Confidence::Likely,
                        });
                        edges_added = true;
                    }
                }
            }
            if edges_added {
                let graph_path = self.data_dir.join("graph.bin");
                if let Err(e) = graph.save(&graph_path) {
                    tracing::warn!("Failed to persist graph after remember_turn: {e}");
                }
            }
        }

        let response = RememberResponse {
            event_id: id,
            stored: true,
            redactions,
        };
        let json = serde_json::to_string_pretty(&response)
            .map_err(|e| user_facing_error("Failed to serialize response", e))?;

        Ok(tag_result_with_repo(
            CallToolResult::success(vec![Content::text(json)]),
            &self.repo_id,
        ))
    }

    #[tool(
        description = "Search memory for past discussions relevant to a query. Use this to check whether a topic, bug, feature, or design decision was discussed in previous sessions. Returns matching events ordered by relevance with recent results ranked higher. Supports time_range_days to narrow results (e.g., last 7 days). Use this before starting work on a topic to recover prior context, or when the user asks 'did we discuss X before?'"
    )]
    async fn recall_history(
        &self,
        Parameters(params): Parameters<RecallHistoryParams>,
    ) -> Result<CallToolResult, McpError> {
        if !self.memory_enabled {
            return Err(McpError::internal_error(
                "Memory is disabled. Enable in .trevec/config.toml",
                None,
            ));
        }
        let Some(mem_store) = &self.memory_store else {
            return Err(McpError::internal_error("Memory store not available", None));
        };

        // Incremental sync from Cursor/Claude Code/Codex (spawn_blocking, 60s cooldown).
        self.maybe_sync_memory().await;

        let query = params.query.trim();
        if query.is_empty() {
            return Err(McpError::invalid_params("query must not be empty", None));
        }
        let limit = params.limit.unwrap_or(20) as usize;
        let search_limit = limit * 3;

        let store = mem_store.read().await;

        // Ensure we see the latest data written by other processes (e.g.
        // events stored from Cursor while Claude Code is querying).
        store.refresh().await;

        // FTS search
        let fts_results = store
            .search_fts_for_repo(query, search_limit, Some(&self.repo_id))
            .await
            .map_err(|e| user_facing_error("Memory FTS search failed", e))?;

        // Vector search (if embedder available)
        let embed_result = self.embedder.lock().expect("embedder lock poisoned").embed(query);
        let vector_results = match embed_result {
            Ok(vec) => store
                .search_vector_for_repo(&vec, search_limit, Some(&self.repo_id))
                .await
                .map_err(|e| user_facing_error("Memory vector search failed", e))?,
            Err(_) => vec![],
        };

        // RRF merge
        let mut merged = rrf_merge(&fts_results, &vector_results, 60);

        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        // Literal-match scan: fetch ALL recent events and inject any
        // whose content literally contains the query.  This catches
        // marker-style queries (e.g. TREVEC_E2E_...) that FTS buries
        // under broad-term hits.  Event tables are small (< 1k rows)
        // so this is cheap.
        let query_lc = query.to_ascii_lowercase();
        let all_events = store
            .get_events_in_range_for_repo(
                now - 90 * 86400,
                now + 86400,
                1000,
                Some(&self.repo_id),
            )
            .await
            .unwrap_or_default();

        let merged_ids: HashSet<String> =
            merged.iter().map(|r| r.node_id.clone()).collect();

        for event in &all_events {
            if merged_ids.contains(&event.id) {
                continue;
            }
            let content_lc = event.content_redacted.to_ascii_lowercase();
            let bm25_lc = event.bm25_text.to_ascii_lowercase();
            if content_lc.contains(&query_lc) || bm25_lc.contains(&query_lc) {
                merged.push(RankedResult {
                    node_id: event.id.clone(),
                    score: 2.0, // high base score for literal match
                    rank: 0,
                });
            }
        }

        // Build event map for scoring from all_events + any extras
        let event_ids: Vec<String> = merged.iter().map(|r| r.node_id.clone()).collect();
        let fetched_events = store
            .get_events_for_repo(&event_ids, Some(&self.repo_id))
            .await
            .map_err(|e| user_facing_error("Failed to retrieve memory events", e))?;

        let event_map: HashMap<String, &MemoryEvent> =
            fetched_events.iter().map(|e| (e.id.clone(), e)).collect();

        // Apply recency boost and time filtering
        let time_cutoff = params
            .time_range_days
            .map(|days| now - (days as i64 * 86400));

        for result in &mut merged {
            if let Some(event) = event_map.get(&result.node_id) {
                let days_ago = ((now - event.created_at) as f64) / 86400.0;
                let recency_boost = 1.0 / (1.0 + days_ago * 0.1);
                result.score += recency_boost * 0.01;

                // Literal content boost
                let content_lc = event.content_redacted.to_ascii_lowercase();
                if content_lc.contains(&query_lc) {
                    result.score += 1.0;
                } else {
                    for part in query_lc.split_whitespace() {
                        if part.len() < 3 {
                            continue;
                        }
                        if content_lc.contains(part) {
                            result.score += 0.05;
                        }
                    }
                }
            }
        }

        // Filter by time range and re-sort
        if let Some(cutoff) = time_cutoff {
            merged.retain(|r| {
                event_map
                    .get(&r.node_id)
                    .map(|e| e.created_at >= cutoff)
                    .unwrap_or(false)
            });
        }

        merged.sort_by(|a, b| {
            b.score
                .partial_cmp(&a.score)
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Build response
        let entries: Vec<MemoryResultEntry> = merged
            .iter()
            .take(limit)
            .filter_map(|r| {
                let event = event_map.get(&r.node_id)?;
                Some(MemoryResultEntry {
                    event_id: event.id.clone(),
                    source: event.source.clone(),
                    session_id: event.session_id.clone(),
                    role: event.role.clone(),
                    created_at: event.created_at,
                    content_snippet: event.content_redacted.clone(),
                    files_touched: event.files_touched.clone(),
                    score: r.score,
                })
            })
            .collect();

        let json = serde_json::to_string_pretty(&entries)
            .map_err(|e| user_facing_error("Failed to serialize results", e))?;

        Ok(tag_result_with_repo(
            CallToolResult::success(vec![Content::text(json)]),
            &self.repo_id,
        ))
    }

    #[tool(
        description = "Generate a bullet-point summary of memory events within a time range. Use this when the user asks for a recap of recent work, a daily/weekly summary, or 'what did we do last week?' Requires start_ts and end_ts as Unix timestamps. Optionally filter by a focus query to summarize only events matching a specific topic. Returns up to max_bullets (default 5) concise bullet points covering the key events."
    )]
    async fn summarize_period(
        &self,
        Parameters(params): Parameters<SummarizePeriodParams>,
    ) -> Result<CallToolResult, McpError> {
        if !self.memory_enabled {
            return Err(McpError::internal_error(
                "Memory is disabled. Enable in .trevec/config.toml",
                None,
            ));
        }
        let Some(mem_store) = &self.memory_store else {
            return Err(McpError::internal_error("Memory store not available", None));
        };

        // Incremental sync from Cursor/Claude Code/Codex (spawn_blocking, 60s cooldown).
        self.maybe_sync_memory().await;

        let max_bullets = params.max_bullets.unwrap_or(5) as usize;
        let store = mem_store.read().await;
        store.refresh().await;

        // If a focus query is provided, use FTS to filter. Otherwise, fetch
        // all events in the time range directly (avoids the "*" FTS issue).
        let in_range: Vec<MemoryEvent> = if let Some(ref focus) = params.focus {
            let results = store
                .search_fts_for_repo(focus, max_bullets * 4, Some(&self.repo_id))
                .await
                .map_err(|e| user_facing_error("Summary search failed", e))?;

            let event_ids: Vec<String> = results.iter().map(|r| r.node_id.clone()).collect();
            let events = store
                .get_events_for_repo(&event_ids, Some(&self.repo_id))
                .await
                .map_err(|e| user_facing_error("Failed to retrieve events", e))?;

            let mut filtered: Vec<MemoryEvent> = events
                .into_iter()
                .filter(|e| e.created_at >= params.start_ts && e.created_at <= params.end_ts)
                .collect();
            filtered.sort_by(|a, b| b.created_at.cmp(&a.created_at));
            filtered
        } else {
            store
                .get_events_in_range_for_repo(
                    params.start_ts,
                    params.end_ts,
                    max_bullets * 4,
                    Some(&self.repo_id),
                )
                .await
                .map_err(|e| user_facing_error("Failed to retrieve events in range", e))?
        };

        let bullets: Vec<SummaryBullet> = in_range
            .iter()
            .take(max_bullets)
            .map(|e| SummaryBullet {
                timestamp: e.created_at,
                source: e.source.clone(),
                content: e.content_redacted.clone(),
            })
            .collect();

        let json = serde_json::to_string_pretty(&bullets)
            .map_err(|e| user_facing_error("Failed to serialize summary", e))?;

        Ok(tag_result_with_repo(
            CallToolResult::success(vec![Content::text(json)]),
            &self.repo_id,
        ))
    }

    #[tool(
        description = "Retrieve the discussion history for a specific file. Use this before modifying a file to see what was previously discussed — prior bugs, design decisions, or review comments. Returns memory events linked to the given file, ordered by recency. Accepts a file path relative to the repository root (e.g., 'src/auth.rs')."
    )]
    async fn get_file_history(
        &self,
        Parameters(params): Parameters<GetFileHistoryParams>,
    ) -> Result<CallToolResult, McpError> {
        if !self.memory_enabled {
            return Err(McpError::internal_error(
                "Memory is disabled. Enable in .trevec/config.toml",
                None,
            ));
        }
        let Some(mem_store) = &self.memory_store else {
            return Err(McpError::internal_error("Memory store not available", None));
        };

        // Incremental sync from Cursor/Claude Code/Codex (spawn_blocking, 60s cooldown).
        self.maybe_sync_memory().await;

        let file_path = params.file_path.trim();
        if file_path.is_empty() {
            return Err(McpError::invalid_params("file_path must not be empty", None));
        }

        let nodes_map = self.nodes_map.read().await;
        let graph = self.graph.read().await;
        let store = mem_store.read().await;
        store.refresh().await;

        // Normalize to repo-relative for matching
        let normalized = normalize_to_repo_relative(file_path, &self.repo_path);

        // Find code nodes for this file
        let file_node_ids: Vec<&String> = nodes_map
            .values()
            .filter(|n| n.file_path == normalized)
            .map(|n| &n.id)
            .collect();

        // Collect memory event IDs linked via Discussed/Triggered edges
        let mut event_ids: Vec<String> = Vec::new();
        for node_id in &file_node_ids {
            let incoming = graph.incoming(node_id);
            for (src_id, edge_type, _) in &incoming {
                if matches!(edge_type, EdgeType::Discussed | EdgeType::Triggered)
                    && !event_ids.contains(src_id)
                {
                    event_ids.push(src_id.clone());
                }
            }
        }

        if event_ids.is_empty() {
            let json = serde_json::to_string_pretty::<Vec<MemoryResultEntry>>(&vec![])
                .map_err(|e| user_facing_error("Failed to serialize", e))?;
            return Ok(tag_result_with_repo(
                CallToolResult::success(vec![Content::text(json)]),
                &self.repo_id,
            ));
        }

        // Retrieve events
        let events = store
            .get_events_for_repo(&event_ids, Some(&self.repo_id))
            .await
            .map_err(|e| user_facing_error("Failed to retrieve events", e))?;

        let mut entries: Vec<MemoryResultEntry> = events
            .iter()
            .map(|e| MemoryResultEntry {
                event_id: e.id.clone(),
                source: e.source.clone(),
                session_id: e.session_id.clone(),
                role: e.role.clone(),
                created_at: e.created_at,
                content_snippet: e.content_redacted.clone(),
                files_touched: e.files_touched.clone(),
                score: 0.0,
            })
            .collect();

        // Sort by recency
        entries.sort_by(|a, b| b.created_at.cmp(&a.created_at));

        let json = serde_json::to_string_pretty(&entries)
            .map_err(|e| user_facing_error("Failed to serialize results", e))?;

        Ok(tag_result_with_repo(
            CallToolResult::success(vec![Content::text(json)]),
            &self.repo_id,
        ))
    }

    #[tool(
        description = "Re-index the repository to refresh the code database. Only call this when search results seem stale or after the user reports making significant code changes. Takes a few seconds to complete. Returns indexing stats (files parsed, nodes extracted, time taken). The server automatically watches for file changes, so manual reindex is rarely needed."
    )]
    async fn reindex(&self) -> Result<CallToolResult, McpError> {
        let repo_path = self.repo_path.clone();
        let data_dir = self.data_dir.clone();
        let config = self.config.clone();
        let nodes_map = self.nodes_map.clone();
        let graph = self.graph.clone();

        let handle = tokio::runtime::Handle::current();
        let result = tokio::task::spawn_blocking(move || {
            handle.block_on(async {
                // Run the indexing pipeline
                let stats = trevec_index::ingest::ingest_with_config(
                    &repo_path,
                    &data_dir,
                    false,
                    &config.index.exclude,
                    Some(&config.embeddings.model),
                    None,
                )
                .await
                .map_err(|e| user_facing_error("Re-index failed", e))?;

                // Reload nodes from disk
                let nodes_path = data_dir.join("nodes.json");
                let json = std::fs::read_to_string(&nodes_path)
                    .map_err(|e| user_facing_error("Failed to read nodes after reindex", e))?;
                let all_nodes: Vec<CodeNode> = serde_json::from_str(&json)
                    .map_err(|e| user_facing_error("Failed to parse nodes after reindex", e))?;
                let map: HashMap<String, CodeNode> =
                    all_nodes.iter().map(|n| (n.id.clone(), n.clone())).collect();
                *nodes_map.write().await = map;

                // Reload graph from disk
                let graph_path = data_dir.join("graph.bin");
                if let Ok(g) = CodeGraph::load(&graph_path) {
                    *graph.write().await = g;
                }

                Ok::<_, McpError>(stats)
            })
        })
        .await
        .map_err(|e| user_facing_error("Reindex task panicked", e))?;

        let stats = result?;

        // Record reindex stats
        {
            let stats_path = self.data_dir.join("stats.json");
            let mut qs = QueryStats::load(&stats_path);
            qs.record_reindex(stats.files_parsed, stats.total_ms as u128);
            if let Err(e) = qs.save(&stats_path) {
                tracing::warn!("Failed to save reindex stats: {e}");
            }
        }

        crate::telemetry::capture("mcp_reindex", serde_json::json!({
            "files_parsed": stats.files_parsed,
            "total_ms": stats.total_ms,
        }));

        let summary = serde_json::json!({
            "files_parsed": stats.files_parsed,
            "files_unchanged": stats.files_unchanged,
            "nodes_extracted": stats.nodes_extracted,
            "total_ms": stats.total_ms,
        });

        Ok(tag_result_with_repo(
            CallToolResult::success(vec![Content::text(
                serde_json::to_string_pretty(&summary)
                    .map_err(|e| user_facing_error("Failed to serialize stats", e))?,
            )]),
            &self.repo_id,
        ))
    }

    #[tool(
        description = "Get a compact codebase overview for planning. Returns language breakdown, \
        top-level modules/classes, entry points (most-called functions), dependency hotspots, \
        and detected conventions. No query needed — assembles from code structure. \
        Use this before planning tasks to understand project layout."
    )]
    async fn repo_summary(
        &self,
        Parameters(params): Parameters<RepoSummaryParams>,
    ) -> Result<CallToolResult, McpError> {
        let max_tokens = params.max_tokens.unwrap_or(500) as usize;
        if max_tokens == 0 {
            return Err(McpError::invalid_params(
                "max_tokens must be greater than 0",
                None,
            ));
        }

        let nodes_map = self.nodes_map.read().await;
        let graph = self.graph.read().await;

        // --- Language breakdown ---
        let mut languages: HashMap<String, HashSet<String>> = HashMap::new();
        for node in nodes_map.values() {
            let ext = node
                .file_path
                .rsplit('.')
                .next()
                .unwrap_or("")
                .to_string();
            let lang = extension_to_language(&ext);
            languages
                .entry(lang)
                .or_default()
                .insert(node.file_path.clone());
        }
        let lang_counts: HashMap<String, usize> = languages
            .iter()
            .map(|(lang, files)| (lang.clone(), files.len()))
            .collect();

        let unique_files: HashSet<&str> = nodes_map.values().map(|n| n.file_path.as_str()).collect();

        // --- Top-level nodes (modules, classes, structs, traits, enums) ---
        let mut top_level: Vec<&CodeNode> = nodes_map
            .values()
            .filter(|n| {
                matches!(
                    n.kind,
                    trevec_core::model::NodeKind::Module
                        | trevec_core::model::NodeKind::Class
                        | trevec_core::model::NodeKind::Struct
                        | trevec_core::model::NodeKind::Trait
                        | trevec_core::model::NodeKind::Enum
                        | trevec_core::model::NodeKind::Interface
                )
            })
            .collect();
        top_level.sort_by_key(|n| (&n.file_path, n.span.start_line));

        // --- Entry points (most incoming Call edges) ---
        let mut call_counts: Vec<(&CodeNode, usize)> = nodes_map
            .values()
            .map(|n| {
                let count = graph.incoming_count(&n.id, Some(EdgeType::Call));
                (n, count)
            })
            .filter(|(_, count)| *count > 0)
            .collect();
        call_counts.sort_by(|a, b| b.1.cmp(&a.1));

        // --- Hotspots (most total connections) ---
        let mut connection_counts: Vec<(&CodeNode, usize)> = nodes_map
            .values()
            .map(|n| {
                let count = graph.total_connections(&n.id);
                (n, count)
            })
            .filter(|(_, count)| *count > 0)
            .collect();
        connection_counts.sort_by(|a, b| b.1.cmp(&a.1));

        // --- Conventions detection ---
        let file_paths: Vec<&str> = unique_files.iter().copied().collect();
        let conventions = detect_conventions(&file_paths);

        // --- Token-budgeted assembly ---
        // Fixed overhead: stats (~50 tokens) + conventions (~50 tokens) = ~100 tokens
        let mut remaining = max_tokens.saturating_sub(100);

        let mut entry_points = Vec::new();
        for (n, count) in call_counts.iter().take(10) {
            if remaining <= 20 {
                break;
            }
            let tok = n.signature.len() / 4 + 10;
            remaining = remaining.saturating_sub(tok);
            entry_points.push(SummaryNode {
                name: n.name.clone(),
                kind: n.kind.to_string(),
                file_path: n.file_path.clone(),
                signature: n.signature.clone(),
                incoming_calls: Some(*count),
                total_connections: None,
            });
        }

        let mut top_level_nodes = Vec::new();
        for n in top_level.iter().take(20) {
            if remaining <= 20 {
                break;
            }
            let tok = n.signature.len() / 4 + 10;
            remaining = remaining.saturating_sub(tok);
            top_level_nodes.push(SummaryNode {
                name: n.name.clone(),
                kind: n.kind.to_string(),
                file_path: n.file_path.clone(),
                signature: n.signature.clone(),
                incoming_calls: None,
                total_connections: None,
            });
        }

        let mut hotspots = Vec::new();
        for (n, count) in connection_counts.iter().take(5) {
            if remaining <= 20 {
                break;
            }
            let tok = n.signature.len() / 4 + 10;
            remaining = remaining.saturating_sub(tok);
            hotspots.push(SummaryNode {
                name: n.name.clone(),
                kind: n.kind.to_string(),
                file_path: n.file_path.clone(),
                signature: n.signature.clone(),
                incoming_calls: None,
                total_connections: Some(*count),
            });
        }

        let estimated_tokens = max_tokens.saturating_sub(remaining);

        let response = RepoSummaryResponse {
            languages: lang_counts,
            total_files: unique_files.len(),
            total_nodes: nodes_map.len(),
            total_edges: graph.edge_count(),
            top_level_nodes,
            entry_points,
            hotspots,
            conventions,
            estimated_tokens,
        };

        let json = serde_json::to_string_pretty(&response)
            .map_err(|e| user_facing_error("Failed to serialize repo summary", e))?;

        Ok(tag_result_with_repo(
            CallToolResult::success(vec![Content::text(json)]),
            &self.repo_id,
        ))
    }

    #[tool(
        description = "Given a set of owned files, return type signatures of their import targets \
        that are outside the ownership set. Use this when a worker needs to understand the \
        interfaces of code it depends on without reading full source files. Returns signatures \
        grouped by source file."
    )]
    async fn neighbor_signatures(
        &self,
        Parameters(params): Parameters<NeighborSignaturesParams>,
    ) -> Result<CallToolResult, McpError> {
        if params.target_files.is_empty() {
            return Err(McpError::invalid_params(
                "target_files must not be empty",
                None,
            ));
        }
        let max_tokens = params.max_tokens.unwrap_or(500) as usize;

        let nodes_map = self.nodes_map.read().await;
        let graph = self.graph.read().await;

        // Normalize target files
        let owned_paths: HashSet<String> = params
            .target_files
            .iter()
            .map(|f| normalize_to_repo_relative(f, &self.repo_path))
            .collect();

        // Find nodes belonging to owned files
        let owned_node_ids: Vec<&str> = nodes_map
            .values()
            .filter(|n| owned_paths.contains(&n.file_path))
            .map(|n| n.id.as_str())
            .collect();

        // Follow outgoing Import edges to find external dependencies
        let mut external_by_file: HashMap<String, Vec<ExternalSignature>> = HashMap::new();
        let mut seen_ids: HashSet<String> = HashSet::new();
        let mut total_tokens: usize = 0;

        for owned_id in &owned_node_ids {
            let outgoing = graph.outgoing(&owned_id.to_string());
            for (target_id, edge_type, _) in outgoing {
                if edge_type != EdgeType::Import {
                    continue;
                }
                if seen_ids.contains(&target_id) {
                    continue;
                }
                let Some(target_node) = nodes_map.get(&target_id) else {
                    continue;
                };
                // Skip if target is in the owned set
                if owned_paths.contains(&target_node.file_path) {
                    continue;
                }
                let sig_tokens = target_node.signature.len() / 4 + 5;
                if total_tokens + sig_tokens > max_tokens {
                    break;
                }
                total_tokens += sig_tokens;
                seen_ids.insert(target_id);

                external_by_file
                    .entry(target_node.file_path.clone())
                    .or_default()
                    .push(ExternalSignature {
                        source_file: target_node.file_path.clone(),
                        name: target_node.name.clone(),
                        signature: target_node.signature.clone(),
                        kind: target_node.kind.to_string(),
                    });
            }
        }

        let files: Vec<FileImports> = external_by_file
            .into_iter()
            .map(|(file_path, sigs)| FileImports {
                file_path,
                external_imports: sigs,
            })
            .collect();

        let response = NeighborSignaturesResponse {
            files,
            estimated_tokens: total_tokens,
        };

        let json = serde_json::to_string_pretty(&response)
            .map_err(|e| user_facing_error("Failed to serialize neighbor signatures", e))?;

        Ok(tag_result_with_repo(
            CallToolResult::success(vec![Content::text(json)]),
            &self.repo_id,
        ))
    }

    #[tool(
        description = "Execute multiple context queries in a single call. Use this when a swarm \
        of workers each need context for different tasks — it amortizes the connection overhead. \
        Maximum 10 queries per batch. Each query returns an independent context bundle."
    )]
    async fn batch_context(
        &self,
        Parameters(params): Parameters<BatchContextParams>,
    ) -> Result<CallToolResult, McpError> {
        if params.queries.is_empty() {
            return Err(McpError::invalid_params(
                "queries must not be empty",
                None,
            ));
        }
        if params.queries.len() > 10 {
            return Err(McpError::invalid_params(
                "maximum 10 queries per batch",
                None,
            ));
        }

        let mut results: Vec<BatchContextResult> = Vec::with_capacity(params.queries.len());

        for bq in &params.queries {
            let query_text = bq.query.trim();
            if query_text.is_empty() {
                results.push(BatchContextResult {
                    query: bq.query.clone(),
                    bundle: None,
                    error: Some("query must not be empty".to_string()),
                });
                continue;
            }

            let budget = bq.budget.unwrap_or(self.default_budget) as usize;
            let anchors = bq.anchors.unwrap_or(self.default_anchors) as usize;

            match self.execute_context_query(query_text, budget, anchors).await {
                Ok(bundle) => {
                    let val = serde_json::to_value(&bundle).ok();
                    results.push(BatchContextResult {
                        query: bq.query.clone(),
                        bundle: val,
                        error: None,
                    });
                }
                Err(e) => {
                    results.push(BatchContextResult {
                        query: bq.query.clone(),
                        bundle: None,
                        error: Some(format!("{}", e)),
                    });
                }
            }
        }

        let json = serde_json::to_string_pretty(&results)
            .map_err(|e| user_facing_error("Failed to serialize batch results", e))?;

        Ok(tag_result_with_repo(
            CallToolResult::success(vec![Content::text(json)]),
            &self.repo_id,
        ))
    }
}

// ---------------------------------------------------------------------------
// Private helpers for TrevecServer
// ---------------------------------------------------------------------------

impl TrevecServer {
    /// Core context retrieval logic shared by `get_context` and `batch_context`.
    async fn execute_context_query(
        &self,
        query_text: &str,
        budget: usize,
        anchors: usize,
    ) -> Result<trevec_core::model::ContextBundle, McpError> {
        let start = Instant::now();

        let query_vec = self
            .embedder
            .lock()
            .expect("embedder lock poisoned")
            .embed(query_text)
            .map_err(|e| user_facing_error("Failed to embed query", e))?;

        let search_limit = anchors * 8;
        let (fts_results, vector_results) = tokio::join!(
            self.store.search_fts(query_text, search_limit),
            self.store.search_vector(&query_vec, search_limit),
        );

        let fts_results =
            fts_results.map_err(|e| user_facing_error("FTS search failed", e))?;
        let vector_results =
            vector_results.map_err(|e| user_facing_error("Vector search failed", e))?;

        let nodes_map = self.nodes_map.read().await;
        let graph_guard = self.graph.read().await;

        // RRF merge
        let mut merged = rrf_merge(&fts_results, &vector_results, 60);

        // Pipeline: filter → literal boost → file path boost → test penalty
        filter_noncode_files(&mut merged, &nodes_map);
        apply_literal_boost(&mut merged, &nodes_map, query_text);
        let extracted_paths = extract_file_paths_from_query(query_text);
        if !extracted_paths.is_empty() {
            apply_file_path_boost(&mut merged, &nodes_map, &extracted_paths, 0.05);
        }
        apply_test_file_penalty(
            &mut merged,
            &nodes_map,
            self.config.retrieval.test_file_penalty,
            &self.config.retrieval.penalty_paths,
        );
        apply_test_fixture_penalty(&mut merged, &nodes_map);

        let anchor_ids: Vec<String> = merged
            .iter()
            .take(anchors)
            .map(|r| r.node_id.clone())
            .collect();

        let token_fn = |id: &String| -> usize {
            nodes_map
                .get(id)
                .map(|n| n.span.estimated_tokens())
                .unwrap_or(0)
        };

        let mut token_budget = TokenBudget::new(budget);
        let included_ids =
            expand_graph(&graph_guard, &anchor_ids, &mut token_budget, &token_fn, 3);

        let mut bundle = assemble_bundle(
            query_text,
            &anchor_ids,
            &included_ids,
            &nodes_map,
            &self.repo_path,
        )
        .map_err(|e| user_facing_error("Failed to assemble context bundle", e))?;

        let elapsed_ms = start.elapsed().as_millis() as u64;
        bundle.retrieval_ms = Some(elapsed_ms);

        // Record stats
        {
            let stats_path = self.data_dir.join("stats.json");
            let mut stats = QueryStats::load(&stats_path);
            stats.record_query(bundle.total_estimated_tokens, bundle.total_source_file_tokens);
            if let Err(e) = stats.save(&stats_path) {
                tracing::warn!("Failed to save query stats: {e}");
            }
        }

        // Telemetry
        {
            let unique_files: HashSet<&str> = bundle.included_nodes.iter().map(|n| n.file_path.as_str()).collect();
            let tokens_saved = bundle.total_source_file_tokens.saturating_sub(bundle.total_estimated_tokens);
            let savings_pct = if bundle.total_source_file_tokens > 0 {
                (tokens_saved as f64 / bundle.total_source_file_tokens as f64) * 100.0
            } else {
                0.0
            };
            crate::telemetry::capture("mcp_query", serde_json::json!({
                "retrieval_ms": elapsed_ms,
                "tokens_served": bundle.total_estimated_tokens,
                "tokens_saved": tokens_saved,
                "savings_pct": savings_pct as u64,
                "functions_found": bundle.included_nodes.len(),
                "files_count": unique_files.len(),
            }));
        }

        Ok(bundle)
    }
}

/// Map file extension to a human-readable language name.
fn extension_to_language(ext: &str) -> String {
    match ext {
        "rs" => "Rust",
        "py" => "Python",
        "js" => "JavaScript",
        "ts" => "TypeScript",
        "tsx" => "TypeScript",
        "jsx" => "JavaScript",
        "go" => "Go",
        "java" => "Java",
        "c" | "h" => "C",
        "cpp" | "cc" | "cxx" | "hpp" => "C++",
        "cs" => "C#",
        "rb" => "Ruby",
        "swift" => "Swift",
        "kt" | "kts" => "Kotlin",
        "lua" => "Lua",
        "zig" => "Zig",
        "sh" | "bash" => "Shell",
        "html" | "htm" => "HTML",
        "css" | "scss" | "less" => "CSS",
        "json" => "JSON",
        "yaml" | "yml" => "YAML",
        "toml" => "TOML",
        "md" | "markdown" => "Markdown",
        other => return other.to_string(),
    }
    .to_string()
}

/// Detect common project conventions from file paths.
fn detect_conventions(file_paths: &[&str]) -> Vec<String> {
    let mut conventions = Vec::new();

    let has_pattern = |pat: &str| file_paths.iter().any(|p| p.contains(pat));

    if file_paths.iter().any(|p| p.contains("src/app/") && p.ends_with("page.tsx")) {
        conventions.push("Next.js App Router (src/app/**/page.tsx)".to_string());
    } else if file_paths.iter().any(|p| p.contains("app/") && p.ends_with("page.tsx")) {
        conventions.push("Next.js App Router (app/**/page.tsx)".to_string());
    }
    if has_pattern("src/pages/") {
        conventions.push("Next.js Pages Router (src/pages/)".to_string());
    }
    if file_paths.iter().any(|p| p.contains("api/") && (p.ends_with("route.ts") || p.ends_with("route.js"))) {
        conventions.push("Next.js API routes (api/**/route.ts)".to_string());
    }
    if has_pattern("components/") {
        conventions.push("Component-based architecture (components/)".to_string());
    }
    if has_pattern("src/lib/") || has_pattern("lib/") {
        conventions.push("Library/utilities pattern (lib/)".to_string());
    }
    if has_pattern("tests/") || has_pattern("__tests__/") || has_pattern("_test.go") || has_pattern("_test.rs") {
        conventions.push("Test suite present".to_string());
    }
    if has_pattern("models/") || has_pattern("schemas/") {
        conventions.push("Data models/schemas pattern".to_string());
    }
    if file_paths.iter().any(|p| *p == "Cargo.toml" || p.starts_with("crates/")) {
        conventions.push("Rust workspace".to_string());
    }
    if has_pattern("prisma/schema.prisma") {
        conventions.push("Prisma ORM (prisma/schema.prisma)".to_string());
    }

    conventions
}

#[tool_handler]
impl ServerHandler for TrevecServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05,
            capabilities: ServerCapabilities::builder().enable_tools().build(),
            server_info: Implementation {
                name: "trevec".to_string(),
                title: Some("Trevec Code Context Server".to_string()),
                version: env!("CARGO_PKG_VERSION").to_string(),
                icons: None,
                website_url: None,
            },
            instructions: Some(
                "Trevec provides code-aware context retrieval for this repository. It understands \
                 code structure (functions, classes, methods, imports, relationships) and supports \
                 memory of past discussions.\n\n\
                 Workflow for code questions: Use get_context as the primary tool — it returns \
                 rich context with source code and related code elements. Use search_code only \
                 for quick lookups when you need to find a specific symbol or check if something \
                 exists. Use read_file_topology to inspect an element's relationships (callers, \
                 callees, imports).\n\n\
                 Workflow for onboarding/overview: Use repo_summary to get a high-level overview \
                 of the repository including languages, entry points, hotspots, and conventions. \
                 Use neighbor_signatures to discover the external API surface that specific files \
                 depend on (imports and their signatures from other files).\n\n\
                 Workflow for batch operations: Use batch_context to retrieve context for multiple \
                 queries in a single call, reducing round-trips when you need context for several \
                 related questions.\n\n\
                 Workflow for memory: Call remember_turn after completing significant work to \
                 preserve context for future sessions. Use recall_history to check if a topic \
                 was discussed before. Use get_file_history before modifying a file to see \
                 prior discussions about it.\n\n\
                 Constraints: All tools are local-only (no network). get_context results may \
                 not include all matches. reindex takes a few seconds; only call it if results \
                 seem stale."
                    .to_string(),
            ),
        }
    }
}

const DEBOUNCE_MS: u64 = 800;

/// Normalize a file path to repo-relative form for matching against node.file_path.
/// Strips the repo root prefix and leading slashes so absolute extracted paths
/// like "/Users/alice/dev/src/auth.rs" become "src/auth.rs".
fn normalize_to_repo_relative(file_path: &str, repo_path: &Path) -> String {
    let repo_str = repo_path.to_string_lossy();
    let stripped = file_path
        .strip_prefix(repo_str.as_ref())
        .or_else(|| file_path.strip_prefix(&format!("{}/", repo_str)))
        .unwrap_or(file_path);
    stripped.trim_start_matches('/').to_string()
}

fn is_relevant_event(event: &Event, data_dir: &Path) -> bool {
    match event.kind {
        EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_) => {}
        _ => return false,
    }

    let data_dir_canonical = data_dir.canonicalize().unwrap_or_else(|_| data_dir.to_path_buf());
    event.paths.iter().any(|p| {
        let p_canonical = p.canonicalize().unwrap_or_else(|_| p.clone());
        !p_canonical.starts_with(&data_dir_canonical)
    })
}

/// Run the MCP server over stdio transport with background file watching.
/// Read-only: requires `trevec init` to have been run first.
/// If the repo is not indexed, starts in degraded mode where tools return
/// a helpful hint instead of failing.
pub async fn run(repo_path: PathBuf, data_dir: PathBuf) -> anyhow::Result<()> {
    // Validate path exists and is a directory before any side effects.
    if !repo_path.exists() || !repo_path.is_dir() {
        anyhow::bail!(
            "Path '{}' does not exist or is not a directory",
            repo_path.display()
        );
    }
    let repo_path = repo_path.canonicalize().unwrap_or(repo_path);

    // Guard: refuse to serve from the home directory (could accidentally index everything).
    if let Ok(home) = std::env::var("HOME") {
        let home_path = PathBuf::from(&home);
        let home_canonical = home_path.canonicalize().unwrap_or(home_path);
        if repo_path == home_canonical {
            anyhow::bail!(
                "Refusing to serve from home directory ({}). \
                 Run `trevec init` inside a specific project directory instead.",
                repo_path.display()
            );
        }
    }

    if !repo_path.join(".git").exists() {
        eprintln!(
            "Warning: '{}' does not appear to be a git repository (no .git directory)",
            repo_path.display()
        );
    }

    let data_dir: PathBuf = if data_dir == Path::new(".trevec") {
        repo_path.join(".trevec")
    } else {
        data_dir
    };

    // Check if the repo is indexed. If not, start in degraded mode.
    let nodes_path = data_dir.join("nodes.json");
    let is_ready = data_dir.exists() && nodes_path.exists();

    if !is_ready {
        tracing::warn!(
            "Repository not indexed — starting in degraded mode. \
             Run `trevec init` in your terminal to set up this project."
        );
        return run_degraded(repo_path, data_dir).await;
    }

    let config = TrevecConfig::load(&data_dir);

    // Load nodes
    let nodes_json = std::fs::read_to_string(&nodes_path)
        .map_err(|_| anyhow::anyhow!("Failed to read nodes.json. Run `trevec init` to re-index."))?;
    let all_nodes: Vec<CodeNode> =
        serde_json::from_str(&nodes_json).map_err(|e| anyhow::anyhow!("Failed to parse nodes: {e}"))?;

    let nodes_map: HashMap<String, CodeNode> =
        all_nodes.iter().map(|n| (n.id.clone(), n.clone())).collect();

    let total_files: HashSet<&str> = all_nodes.iter().map(|n| n.file_path.as_str()).collect();
    let total_files_count = total_files.len();
    let total_nodes_count = nodes_map.len();
    tracing::info!("Loaded {} nodes", total_nodes_count);

    // Load graph
    let graph_path = data_dir.join("graph.bin");
    let graph =
        CodeGraph::load(&graph_path).map_err(|e| anyhow::anyhow!("Failed to load graph: {e}"))?;

    tracing::info!(
        "Loaded graph: {} nodes, {} edges",
        graph.node_count(),
        graph.edge_count()
    );

    // Open store
    let lance_dir = data_dir.join("lance");
    let store = Store::open(lance_dir.to_str().unwrap())
        .await
        .map_err(|e| anyhow::anyhow!("Failed to open store: {e}"))?;

    tracing::info!("Opened LanceDB store");

    // Initialize embedder
    let embedder = Embedder::new_with_model(
        Some(&config.embeddings.model),
        false,
        Some(data_dir.join("models")),
        None,
    )
        .map_err(|e| anyhow::anyhow!("Failed to initialize embedder: {e}"))?;

    tracing::info!("Embedder ready");

    // Open memory store if enabled
    let memory_store = if config.memory.enabled {
        match MemoryStore::open(lance_dir.to_str().unwrap()).await {
            Ok(ms) => {
                tracing::info!("Memory store opened");
                Some(ms)
            }
            Err(e) => {
                tracing::warn!("Failed to open memory store: {e}");
                None
            }
        }
    } else {
        None
    };

    // Create server
    let server = TrevecServer::new(
        nodes_map,
        graph,
        store,
        embedder,
        repo_path.clone(),
        data_dir.clone(),
        &config,
        memory_store,
    );

    // Spawn background file watcher for automatic re-indexing
    let (tx, mut rx) = tokio::sync::mpsc::unbounded_channel();
    let watcher_repo = repo_path.clone();
    let mut watcher = notify::recommended_watcher(move |res: Result<Event, notify::Error>| {
        if let Ok(event) = res {
            let _ = tx.send(event);
        }
    })
    .map_err(|e| anyhow::anyhow!("Failed to create file watcher: {e}"))?;

    watcher
        .watch(&watcher_repo, RecursiveMode::Recursive)
        .map_err(|e| anyhow::anyhow!("Failed to watch repository: {e}"))?;

    let watcher_nodes = server.nodes_map.clone();
    let watcher_graph = server.graph.clone();
    let watcher_data_dir = data_dir.clone();
    let watcher_config = config.clone();

    tokio::spawn(async move {
        let _watcher = watcher; // Keep watcher alive for the duration of this task
        let mut last_reindex = Instant::now();

        loop {
            // Wait for the next event
            let event = match rx.recv().await {
                Some(e) => e,
                None => break,
            };

            if !is_relevant_event(&event, &watcher_data_dir) {
                continue;
            }

            // Debounce: wait for events to settle
            tokio::time::sleep(Duration::from_millis(DEBOUNCE_MS)).await;
            while rx.try_recv().is_ok() {}

            if last_reindex.elapsed() < Duration::from_millis(DEBOUNCE_MS) {
                continue;
            }

            tracing::info!("File changes detected, re-indexing...");
            match trevec_index::ingest::ingest_with_config(
                &repo_path,
                &watcher_data_dir,
                false,
                &watcher_config.index.exclude,
                Some(&watcher_config.embeddings.model),
                None,
            )
            .await
            {
                Ok(stats) => {
                    tracing::info!(
                        "Re-index complete: {} parsed, {} unchanged ({}ms)",
                        stats.files_parsed,
                        stats.files_unchanged,
                        stats.total_ms
                    );

                    // Reload nodes from disk
                    let nodes_path = watcher_data_dir.join("nodes.json");
                    if let Ok(json) = std::fs::read_to_string(&nodes_path) {
                        if let Ok(nodes) = serde_json::from_str::<Vec<CodeNode>>(&json) {
                            let map: HashMap<String, CodeNode> =
                                nodes.iter().map(|n| (n.id.clone(), n.clone())).collect();
                            tracing::info!("Reloaded {} nodes into server", map.len());
                            *watcher_nodes.write().await = map;
                        }
                    }

                    // Reload graph from disk
                    let graph_path = watcher_data_dir.join("graph.bin");
                    if let Ok(g) = CodeGraph::load(&graph_path) {
                        tracing::info!(
                            "Reloaded graph: {} nodes, {} edges",
                            g.node_count(),
                            g.edge_count()
                        );
                        *watcher_graph.write().await = g;
                    }
                }
                Err(e) => {
                    tracing::error!("Background re-index failed: {e:#}");
                }
            }

            last_reindex = Instant::now();
        }
    });

    // Memory sync strategy:
    // - Incremental sync on-demand in recall_history / summarize_period /
    //   get_file_history via maybe_sync_memory() (spawn_blocking, 60s cooldown).
    // - store.refresh() in each tool for cross-process visibility.

    crate::telemetry::maybe_show_first_run_notice();
    crate::telemetry::capture("mcp_serve_start", serde_json::json!({
        "total_nodes": total_nodes_count,
        "total_files": total_files_count,
    }));

    tracing::info!("Starting MCP server on stdio (with background file watcher)...");

    let service = server
        .serve(rmcp::transport::stdio())
        .await
        .map_err(|e| anyhow::anyhow!("Failed to start MCP server: {e}"))?;

    service
        .waiting()
        .await
        .map_err(|e| anyhow::anyhow!("MCP server error: {e}"))?;

    tracing::info!("MCP server shut down");
    Ok(())
}

/// Degraded-mode MCP server: starts and stays alive but every tool returns
/// a helpful message asking the user to run `trevec init`.
async fn run_degraded(repo_path: PathBuf, _data_dir: PathBuf) -> anyhow::Result<()> {
    let server = DegradedServer {
        repo_path: Arc::new(repo_path),
        tool_router: DegradedServer::tool_router(),
    };

    tracing::info!("Starting MCP server in degraded mode (not indexed)...");

    let service = server
        .serve(rmcp::transport::stdio())
        .await
        .map_err(|e| anyhow::anyhow!("Failed to start MCP server: {e}"))?;

    service
        .waiting()
        .await
        .map_err(|e| anyhow::anyhow!("MCP server error: {e}"))?;

    tracing::info!("MCP server shut down (degraded)");
    Ok(())
}

// ---------------------------------------------------------------------------
// Degraded-mode server (not indexed)
// ---------------------------------------------------------------------------

const NOT_INDEXED_MSG: &str = "This repository is not indexed yet. \
Run `trevec init` in your terminal to set up Trevec for this project.";

#[derive(Clone)]
struct DegradedServer {
    repo_path: Arc<PathBuf>,
    tool_router: ToolRouter<Self>,
}

#[tool_router]
impl DegradedServer {
    #[tool(
        name = "get_context",
        description = "Retrieve rich code context for a question about this codebase."
    )]
    async fn get_context(
        &self,
        _params: Parameters<GetContextParams>,
    ) -> Result<CallToolResult, McpError> {
        Ok(CallToolResult::success(vec![Content::text(NOT_INDEXED_MSG)]))
    }

    #[tool(
        name = "search_code",
        description = "Search for code symbols in the codebase."
    )]
    async fn search_code(
        &self,
        _params: Parameters<SearchCodeParams>,
    ) -> Result<CallToolResult, McpError> {
        Ok(CallToolResult::success(vec![Content::text(NOT_INDEXED_MSG)]))
    }

    #[tool(
        name = "read_file_topology",
        description = "Inspect the structure and relationships of a specific code element."
    )]
    async fn read_file_topology(
        &self,
        _params: Parameters<ReadFileTopologyParams>,
    ) -> Result<CallToolResult, McpError> {
        Ok(CallToolResult::success(vec![Content::text(NOT_INDEXED_MSG)]))
    }

    #[tool(
        name = "remember_turn",
        description = "Save a conversation turn to memory for future recall."
    )]
    async fn remember_turn(
        &self,
        _params: Parameters<RememberTurnParams>,
    ) -> Result<CallToolResult, McpError> {
        Ok(CallToolResult::success(vec![Content::text(NOT_INDEXED_MSG)]))
    }

    #[tool(
        name = "recall_history",
        description = "Search memory for past discussions relevant to a query."
    )]
    async fn recall_history(
        &self,
        _params: Parameters<RecallHistoryParams>,
    ) -> Result<CallToolResult, McpError> {
        Ok(CallToolResult::success(vec![Content::text(NOT_INDEXED_MSG)]))
    }

    #[tool(
        name = "summarize_period",
        description = "Generate a summary of memory events within a time range."
    )]
    async fn summarize_period(
        &self,
        _params: Parameters<SummarizePeriodParams>,
    ) -> Result<CallToolResult, McpError> {
        Ok(CallToolResult::success(vec![Content::text(NOT_INDEXED_MSG)]))
    }

    #[tool(
        name = "get_file_history",
        description = "Retrieve the discussion history for a specific file."
    )]
    async fn get_file_history(
        &self,
        _params: Parameters<GetFileHistoryParams>,
    ) -> Result<CallToolResult, McpError> {
        Ok(CallToolResult::success(vec![Content::text(NOT_INDEXED_MSG)]))
    }

    #[tool(
        name = "reindex",
        description = "Re-index the repository to refresh the code database."
    )]
    async fn reindex(&self) -> Result<CallToolResult, McpError> {
        Ok(CallToolResult::success(vec![Content::text(NOT_INDEXED_MSG)]))
    }

    #[tool(
        name = "repo_summary",
        description = "Get a compact codebase overview for planning."
    )]
    async fn repo_summary(
        &self,
        _params: Parameters<RepoSummaryParams>,
    ) -> Result<CallToolResult, McpError> {
        Ok(CallToolResult::success(vec![Content::text(NOT_INDEXED_MSG)]))
    }

    #[tool(
        name = "neighbor_signatures",
        description = "Get type signatures of import targets outside the ownership set."
    )]
    async fn neighbor_signatures(
        &self,
        _params: Parameters<NeighborSignaturesParams>,
    ) -> Result<CallToolResult, McpError> {
        Ok(CallToolResult::success(vec![Content::text(NOT_INDEXED_MSG)]))
    }

    #[tool(
        name = "batch_context",
        description = "Execute multiple context queries in a single call."
    )]
    async fn batch_context(
        &self,
        _params: Parameters<BatchContextParams>,
    ) -> Result<CallToolResult, McpError> {
        Ok(CallToolResult::success(vec![Content::text(NOT_INDEXED_MSG)]))
    }
}

#[tool_handler]
impl ServerHandler for DegradedServer {
    fn get_info(&self) -> ServerInfo {
        ServerInfo {
            protocol_version: ProtocolVersion::V_2024_11_05,
            capabilities: ServerCapabilities {
                tools: Some(ToolsCapability {
                    list_changed: None,
                }),
                ..Default::default()
            },
            server_info: Implementation {
                name: "trevec".to_string(),
                title: Some("Trevec Code Context Server".to_string()),
                version: env!("CARGO_PKG_VERSION").to_string(),
                icons: None,
                website_url: None,
            },
            instructions: Some(format!(
                "Trevec is not indexed for this repository ({}). \
                 Ask the user to run `trevec init` in their terminal.",
                self.repo_path.display()
            )),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_context_params_defaults() {
        let json = r#"{"query": "authentication"}"#;
        let params: GetContextParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.query, "authentication");
        assert_eq!(params.budget, None);
        assert_eq!(params.anchors, None);
    }

    #[test]
    fn test_get_context_params_custom() {
        let json = r#"{"query": "login flow", "budget": 8192, "anchors": 10}"#;
        let params: GetContextParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.query, "login flow");
        assert_eq!(params.budget, Some(8192));
        assert_eq!(params.anchors, Some(10));
    }

    #[test]
    fn test_search_code_params_defaults() {
        let json = r#"{"query": "database"}"#;
        let params: SearchCodeParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.query, "database");
        assert_eq!(params.limit, Some(20));
    }

    #[test]
    fn test_search_code_params_custom_limit() {
        let json = r#"{"query": "database", "limit": 5}"#;
        let params: SearchCodeParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.limit, Some(5));
    }

    #[test]
    fn test_topology_response_serialization() {
        let response = TopologyResponse {
            node: TopologyNode {
                id: "abc123".to_string(),
                name: "authenticate".to_string(),
                kind: "function".to_string(),
                file_path: "src/auth.rs".to_string(),
                signature: "pub fn authenticate(user: &str, pass: &str) -> bool".to_string(),
                span: TopologySpan {
                    start_line: 10,
                    end_line: 25,
                    start_byte: 200,
                    end_byte: 600,
                },
            },
            outgoing: vec![TopologyEdge {
                node_id: "def456".to_string(),
                name: "verify_hash".to_string(),
                kind: "function".to_string(),
                edge_type: "call".to_string(),
                confidence: "certain".to_string(),
            }],
            incoming: vec![],
        };

        let json = serde_json::to_string_pretty(&response).unwrap();
        assert!(json.contains("authenticate"));
        assert!(json.contains("verify_hash"));
        assert!(json.contains("outgoing"));
        assert!(json.contains("incoming"));

        // Verify it round-trips as valid JSON
        let parsed: serde_json::Value = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed["node"]["id"], "abc123");
        assert_eq!(parsed["outgoing"][0]["edge_type"], "call");
    }

    #[test]
    fn test_read_file_topology_params() {
        let json = r#"{"node_id": "abc123"}"#;
        let params: ReadFileTopologyParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.node_id, "abc123");
    }

    #[test]
    fn test_normalize_to_repo_relative_absolute_path() {
        let repo = Path::new("/Users/alice/dev/backend");
        assert_eq!(
            normalize_to_repo_relative("/Users/alice/dev/backend/src/auth.rs", repo),
            "src/auth.rs"
        );
    }

    #[test]
    fn test_normalize_to_repo_relative_with_trailing_slash() {
        let repo = Path::new("/Users/alice/dev/backend");
        // Path has separator after repo root
        assert_eq!(
            normalize_to_repo_relative("/Users/alice/dev/backend/src/main.rs", repo),
            "src/main.rs"
        );
    }

    #[test]
    fn test_normalize_to_repo_relative_already_relative() {
        let repo = Path::new("/Users/alice/dev/backend");
        assert_eq!(
            normalize_to_repo_relative("src/auth.rs", repo),
            "src/auth.rs"
        );
    }

    #[test]
    fn test_normalize_to_repo_relative_different_repo() {
        let repo = Path::new("/Users/alice/dev/backend");
        // Path from a different repo — strip_prefix fails, but leading / is trimmed
        assert_eq!(
            normalize_to_repo_relative("/Users/bob/other/src/lib.rs", repo),
            "Users/bob/other/src/lib.rs"
        );
    }

    #[tokio::test]
    async fn test_run_rejects_nonexistent_path() {
        let result = run(
            PathBuf::from("/tmp/trevec_does_not_exist_xyz"),
            PathBuf::from(".trevec"),
        )
        .await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("does not exist"), "got: {err}");
    }

    #[tokio::test]
    async fn test_run_rejects_file_as_path() {
        let tmp = tempfile::tempdir().unwrap();
        let file_path = tmp.path().join("not-a-dir.txt");
        std::fs::write(&file_path, "hello").unwrap();

        let result = run(file_path, PathBuf::from(".trevec")).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("not a directory"), "got: {err}");
    }

    // ── RememberTurnParams tests ──────────────────────────────────────

    #[test]
    fn test_remember_turn_params_required_fields() {
        let json = r#"{"role": "user", "content": "Fixed the auth bug"}"#;
        let params: RememberTurnParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.role, "user");
        assert_eq!(params.content, "Fixed the auth bug");
        assert_eq!(params.session_id, None);
        assert_eq!(params.importance, None);
        assert_eq!(params.pinned, None);
        assert_eq!(params.files_touched, None);
    }

    #[test]
    fn test_remember_turn_params_all_fields() {
        let json = r#"{
            "role": "assistant",
            "content": "Refactored the login flow",
            "session_id": "sess_123",
            "importance": 80,
            "pinned": true,
            "files_touched": ["src/auth.rs", "src/login.rs"]
        }"#;
        let params: RememberTurnParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.role, "assistant");
        assert_eq!(params.session_id, Some("sess_123".to_string()));
        assert_eq!(params.importance, Some(80));
        assert_eq!(params.pinned, Some(true));
        assert_eq!(
            params.files_touched,
            Some(vec!["src/auth.rs".to_string(), "src/login.rs".to_string()])
        );
    }

    #[test]
    fn test_remember_turn_params_missing_required() {
        // Missing "content"
        let json = r#"{"role": "user"}"#;
        let result: Result<RememberTurnParams, _> = serde_json::from_str(json);
        assert!(result.is_err());
    }

    // ── RecallHistoryParams tests ─────────────────────────────────────

    #[test]
    fn test_recall_history_params_defaults() {
        let json = r#"{"query": "auth bug"}"#;
        let params: RecallHistoryParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.query, "auth bug");
        assert_eq!(params.time_range_days, None);
        assert_eq!(params.limit, Some(20)); // default
    }

    #[test]
    fn test_recall_history_params_with_options() {
        let json = r#"{"query": "login", "time_range_days": 7, "limit": 5}"#;
        let params: RecallHistoryParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.query, "login");
        assert_eq!(params.time_range_days, Some(7));
        assert_eq!(params.limit, Some(5));
    }

    #[test]
    fn test_recall_history_params_aliases() {
        // "search" alias for "query"
        let json = r#"{"search": "debug session"}"#;
        let params: RecallHistoryParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.query, "debug session");

        // "days" alias for "time_range_days"
        let json = r#"{"q": "test", "days": 30}"#;
        let params: RecallHistoryParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.query, "test");
        assert_eq!(params.time_range_days, Some(30));
    }

    // ── SummarizePeriodParams tests ───────────────────────────────────

    #[test]
    fn test_summarize_period_params_required() {
        let json = r#"{"start_ts": 1700000000, "end_ts": 1700100000}"#;
        let params: SummarizePeriodParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.start_ts, 1700000000);
        assert_eq!(params.end_ts, 1700100000);
        assert_eq!(params.focus, None);
        assert_eq!(params.max_bullets, None);
    }

    #[test]
    fn test_summarize_period_params_all_fields() {
        let json = r#"{
            "start_ts": 1700000000,
            "end_ts": 1700100000,
            "focus": "authentication",
            "max_bullets": 3
        }"#;
        let params: SummarizePeriodParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.focus, Some("authentication".to_string()));
        assert_eq!(params.max_bullets, Some(3));
    }

    #[test]
    fn test_summarize_period_params_aliases() {
        let json = r#"{"start": 100, "end": 200, "query": "deploy", "limit": 10}"#;
        let params: SummarizePeriodParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.start_ts, 100);
        assert_eq!(params.end_ts, 200);
        assert_eq!(params.focus, Some("deploy".to_string()));
        assert_eq!(params.max_bullets, Some(10));
    }

    // ── GetFileHistoryParams tests ────────────────────────────────────

    #[test]
    fn test_get_file_history_params() {
        let json = r#"{"file_path": "src/auth.rs"}"#;
        let params: GetFileHistoryParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.file_path, "src/auth.rs");
    }

    #[test]
    fn test_get_file_history_params_aliases() {
        let json = r#"{"path": "src/main.rs"}"#;
        let params: GetFileHistoryParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.file_path, "src/main.rs");

        let json = r#"{"file": "lib.rs"}"#;
        let params: GetFileHistoryParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.file_path, "lib.rs");
    }

    // ── RepoSummaryParams tests ──────────────────────────────────────

    #[test]
    fn test_repo_summary_params_defaults() {
        let json = r#"{}"#;
        let params: RepoSummaryParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.max_tokens, None);
    }

    #[test]
    fn test_repo_summary_params_custom() {
        let json = r#"{"max_tokens": 1000}"#;
        let params: RepoSummaryParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.max_tokens, Some(1000));
    }

    // ── NeighborSignaturesParams tests ───────────────────────────────

    #[test]
    fn test_neighbor_signatures_params() {
        let json = r#"{"target_files": ["src/auth.ts", "src/login.ts"]}"#;
        let params: NeighborSignaturesParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.target_files.len(), 2);
        assert_eq!(params.target_files[0], "src/auth.ts");
        assert_eq!(params.max_tokens, None);
    }

    #[test]
    fn test_neighbor_signatures_params_aliases() {
        let json = r#"{"files": ["src/main.rs"], "max_tokens": 200}"#;
        let params: NeighborSignaturesParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.target_files, vec!["src/main.rs"]);
        assert_eq!(params.max_tokens, Some(200));

        let json = r#"{"owned_files": ["lib.rs"]}"#;
        let params: NeighborSignaturesParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.target_files, vec!["lib.rs"]);
    }

    // ── BatchContextParams tests ─────────────────────────────────────

    #[test]
    fn test_batch_context_params_single() {
        let json = r#"{"queries": [{"query": "how does auth work?"}]}"#;
        let params: BatchContextParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.queries.len(), 1);
        assert_eq!(params.queries[0].query, "how does auth work?");
        assert_eq!(params.queries[0].budget, None);
        assert_eq!(params.queries[0].anchors, None);
    }

    #[test]
    fn test_batch_context_params_multiple() {
        let json = r#"{"queries": [
            {"query": "auth flow", "budget": 2048},
            {"query": "database schema", "budget": 4096, "anchors": 3}
        ]}"#;
        let params: BatchContextParams = serde_json::from_str(json).unwrap();
        assert_eq!(params.queries.len(), 2);
        assert_eq!(params.queries[0].budget, Some(2048));
        assert_eq!(params.queries[1].anchors, Some(3));
    }

    // ── Response serialization tests ─────────────────────────────────

    #[test]
    fn test_repo_summary_response_serialization() {
        let response = RepoSummaryResponse {
            languages: HashMap::from([("Rust".to_string(), 10), ("TypeScript".to_string(), 5)]),
            total_files: 15,
            total_nodes: 120,
            total_edges: 200,
            top_level_nodes: vec![SummaryNode {
                name: "AuthModule".to_string(),
                kind: "module".to_string(),
                file_path: "src/auth.rs".to_string(),
                signature: "mod auth".to_string(),
                incoming_calls: None,
                total_connections: None,
            }],
            entry_points: vec![SummaryNode {
                name: "main".to_string(),
                kind: "function".to_string(),
                file_path: "src/main.rs".to_string(),
                signature: "fn main()".to_string(),
                incoming_calls: Some(5),
                total_connections: None,
            }],
            hotspots: vec![],
            conventions: vec!["Rust workspace".to_string()],
            estimated_tokens: 150,
        };

        let json = serde_json::to_string_pretty(&response).unwrap();
        assert!(json.contains("\"total_files\": 15"));
        assert!(json.contains("\"AuthModule\""));
        assert!(json.contains("\"incoming_calls\": 5"));
        // total_connections should be absent (skip_serializing_if)
        assert!(!json.contains("total_connections"));
    }

    #[test]
    fn test_neighbor_signatures_response_serialization() {
        let response = NeighborSignaturesResponse {
            files: vec![FileImports {
                file_path: "src/utils.ts".to_string(),
                external_imports: vec![ExternalSignature {
                    source_file: "src/utils.ts".to_string(),
                    name: "formatDate".to_string(),
                    signature: "export function formatDate(d: Date): string".to_string(),
                    kind: "function".to_string(),
                }],
            }],
            estimated_tokens: 30,
        };

        let json = serde_json::to_string_pretty(&response).unwrap();
        assert!(json.contains("formatDate"));
        assert!(json.contains("estimated_tokens"));
    }

    #[test]
    fn test_batch_context_result_serialization() {
        let result = BatchContextResult {
            query: "test query".to_string(),
            bundle: Some(serde_json::json!({"bundle_id": "abc"})),
            error: None,
        };
        let json = serde_json::to_string(&result).unwrap();
        assert!(json.contains("test query"));
        assert!(json.contains("bundle_id"));
        assert!(!json.contains("error")); // skip_serializing_if None

        let err_result = BatchContextResult {
            query: "bad".to_string(),
            bundle: None,
            error: Some("failed".to_string()),
        };
        let json = serde_json::to_string(&err_result).unwrap();
        assert!(!json.contains("bundle")); // skip_serializing_if None
        assert!(json.contains("failed"));
    }

    // ── Convention detection tests ────────────────────────────────────

    #[test]
    fn test_detect_conventions_nextjs() {
        let files = vec![
            "src/app/page.tsx",
            "src/app/layout.tsx",
            "src/app/api/auth/route.ts",
            "src/components/Button.tsx",
            "src/lib/utils.ts",
        ];
        let conventions = detect_conventions(&files);
        assert!(conventions.iter().any(|c| c.contains("Next.js App Router")));
        assert!(conventions.iter().any(|c| c.contains("Component-based")));
        assert!(conventions.iter().any(|c| c.contains("lib/")));
        assert!(conventions.iter().any(|c| c.contains("API routes")));
    }

    #[test]
    fn test_detect_conventions_rust() {
        let files = vec![
            "Cargo.toml",
            "crates/core/src/lib.rs",
            "crates/core/src/model.rs",
            "tests/integration_test.rs",
        ];
        let conventions = detect_conventions(&files);
        assert!(conventions.iter().any(|c| c.contains("Rust workspace")));
        assert!(conventions.iter().any(|c| c.contains("Test suite")));
    }

    #[test]
    fn test_detect_conventions_empty() {
        let files: Vec<&str> = vec![];
        let conventions = detect_conventions(&files);
        assert!(conventions.is_empty());
    }

    // ── Extension to language tests ──────────────────────────────────

    #[test]
    fn test_extension_to_language() {
        assert_eq!(extension_to_language("rs"), "Rust");
        assert_eq!(extension_to_language("ts"), "TypeScript");
        assert_eq!(extension_to_language("tsx"), "TypeScript");
        assert_eq!(extension_to_language("py"), "Python");
        assert_eq!(extension_to_language("go"), "Go");
        assert_eq!(extension_to_language("js"), "JavaScript");
        assert_eq!(extension_to_language("unknown_ext"), "unknown_ext");
    }
}
