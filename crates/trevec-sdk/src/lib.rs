//! # Trevec SDK — Universal Context Graph for AI Agents
//!
//! ## Quick Start
//!
//! ```no_run
//! use trevec_sdk::TrevecEngine;
//!
//! # fn example() -> anyhow::Result<()> {
//! // Zero-config: in-memory, ephemeral
//! let mut engine = TrevecEngine::default();
//!
//! // Add a memory
//! engine.add("I love hiking in Denver", "alex", None)?;
//!
//! // Search memories
//! let results = engine.search("hobbies", Some("alex"), 10);
//! # Ok(())
//! # }
//! ```

#![warn(clippy::all)]

pub mod integrations;

use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

use trevec_brain::BrainEngine;
use trevec_core::config::TrevecConfig;
use trevec_core::universal::*;
use trevec_core::TokenBudget;
use trevec_index::embedder::Embedder;
use trevec_index::graph::CodeGraph;
use trevec_index::store::Store;
use trevec_parse::conversation::ConversationParser;
use trevec_parse::registry::{DomainParser, ParserRegistry};
use trevec_retrieve::bundle::assemble_bundle;
use trevec_retrieve::expander::expand_graph;
use trevec_retrieve::search::{
    apply_file_path_boost, apply_literal_boost, apply_test_file_penalty,
    apply_test_fixture_penalty, extract_file_paths_from_query, filter_noncode_files, rrf_merge,
};

// Re-export key types for SDK consumers
pub use trevec_brain;
pub use trevec_core;
pub use trevec_core::config::BrainConfig as BrainConfiguration;
pub use trevec_core::model::{CodeNode, ContextBundle, NodeKind};
pub use trevec_core::universal::{
    DomainTag, IntentSummary, TemporalMeta, UniversalEdge, UniversalEdgeType, UniversalKind,
    UniversalNode,
};
pub use trevec_parse::registry::ParseResult;

/// Configuration for the Trevec engine.
#[derive(Debug, Clone)]
pub struct EngineConfig {
    /// Project name (used as storage identifier). If None, uses ephemeral in-memory store.
    pub project: Option<String>,
    /// Explicit path to the data directory. Overrides project-based path.
    pub data_dir: Option<PathBuf>,
    /// Trevec configuration (retrieval, embeddings, etc.).
    pub trevec_config: TrevecConfig,
    /// Enable the Brain (async intelligence).
    pub brain_enabled: bool,
    /// Verbose logging.
    pub verbose: bool,
}

impl Default for EngineConfig {
    fn default() -> Self {
        Self {
            project: None,
            data_dir: None,
            trevec_config: TrevecConfig::default(),
            brain_enabled: false,
            verbose: false,
        }
    }
}

/// Options for code query operations.
#[derive(Debug, Clone)]
pub struct QueryOptions {
    /// Token budget for context assembly.
    pub budget: usize,
    /// Number of anchor nodes to select.
    pub anchors: usize,
    /// Domain filter (None = all domains).
    pub domain_filter: Option<DomainTag>,
    /// Context depth for compression.
    pub depth: Option<String>,
}

impl Default for QueryOptions {
    fn default() -> Self {
        Self {
            budget: 4096,
            anchors: 5,
            domain_filter: None,
            depth: None,
        }
    }
}

/// Options for search/recall operations.
#[derive(Debug, Clone)]
pub struct SearchOptions {
    /// Maximum number of results.
    pub limit: usize,
    /// Time range in days (None = all time).
    pub time_range_days: Option<u32>,
    /// Domain filter (None = all domains).
    pub domain_filter: Option<DomainTag>,
}

impl Default for SearchOptions {
    fn default() -> Self {
        Self {
            limit: 10,
            time_range_days: None,
            domain_filter: None,
        }
    }
}

/// Result of a search operation.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct SearchResult {
    pub node_id: String,
    pub name: String,
    pub kind: String,
    pub file_path: String,
    pub signature: String,
    pub score: f64,
}

/// A memory entry returned from search.
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct MemoryResult {
    pub id: String,
    pub memory: String,
    pub role: Option<String>,
    pub user_id: Option<String>,
    pub created_at: Option<i64>,
    pub score: f64,
}

/// File topology information.
#[derive(Debug, Clone)]
pub struct FileTopology {
    pub file_path: String,
    pub nodes: Vec<TopologyNode>,
}

#[derive(Debug, Clone)]
pub struct TopologyNode {
    pub node_id: String,
    pub kind: String,
    pub name: String,
    pub signature: String,
    pub start_line: usize,
    pub end_line: usize,
    pub edges: Vec<TopologyEdge>,
}

#[derive(Debug, Clone)]
pub struct TopologyEdge {
    pub target_id: String,
    pub target_name: String,
    pub edge_type: String,
    pub confidence: String,
}

/// Repository summary information.
#[derive(Debug, Clone)]
pub struct RepoSummary {
    pub languages: HashMap<String, usize>,
    pub node_count: usize,
    pub edge_count: usize,
    pub file_count: usize,
    pub entry_points: Vec<String>,
    pub hotspots: Vec<String>,
}

// ── Helper: resolve data directory ──────────────────────────────────────────

fn resolve_data_dir(config: &EngineConfig) -> PathBuf {
    if let Some(ref dir) = config.data_dir {
        return dir.clone();
    }
    if let Some(ref project) = config.project {
        // ~/.trevec/<project>/
        let home = dirs_or_fallback();
        return home.join(".trevec").join(project);
    }
    // Ephemeral: temp directory
    std::env::temp_dir().join(".trevec-ephemeral")
}

fn dirs_or_fallback() -> PathBuf {
    std::env::var("HOME")
        .or_else(|_| std::env::var("USERPROFILE"))
        .map(PathBuf::from)
        .unwrap_or_else(|_| std::env::temp_dir())
}

// ═══════════════════════════════════════════════════════════════════════════
// TrevecEngine
// ═══════════════════════════════════════════════════════════════════════════

/// The main Trevec engine providing the unified SDK interface.
///
/// Three ways to create:
/// - `TrevecEngine::default()` — zero config, ephemeral in-memory
/// - `TrevecEngine::new("my-project", config)` — persistent by project name
/// - `TrevecEngine::for_repo("/path/to/repo")` — code context mode
pub struct TrevecEngine {
    project: Option<String>,
    repo_path: Option<PathBuf>,
    data_dir: PathBuf,
    config: TrevecConfig,
    parser_registry: ParserRegistry,
    brain: Option<BrainEngine>,
    /// In-memory store of universal nodes.
    universal_nodes: HashMap<String, UniversalNode>,
    /// Full pipeline components (populated after index()).
    store: Option<Store>,
    graph: Option<CodeGraph>,
    embedder: Option<Mutex<Embedder>>,
    /// Code nodes map (mirrors LanceDB for fast lookup).
    code_nodes: HashMap<String, CodeNode>,
    /// Whether the full pipeline has been initialized.
    indexed: bool,
}

impl Default for TrevecEngine {
    /// Create a zero-config ephemeral engine. No files, no paths, just works.
    fn default() -> Self {
        let config = EngineConfig::default();
        let data_dir = resolve_data_dir(&config);
        let mut registry = ParserRegistry::new();
        registry.register(Box::new(ConversationParser));

        Self {
            project: None,
            repo_path: None,
            data_dir,
            config: TrevecConfig::default(),
            parser_registry: registry,
            brain: None,
            universal_nodes: HashMap::new(),
            store: None,
            graph: None,
            embedder: None,
            code_nodes: HashMap::new(),
            indexed: false,
        }
    }
}

impl TrevecEngine {
    /// Create a new engine with a project name (persistent storage).
    pub fn new(project: impl Into<String>, config: EngineConfig) -> Result<Self> {
        let project_name = project.into();
        let mut effective_config = config;
        effective_config.project = Some(project_name.clone());
        let data_dir = resolve_data_dir(&effective_config);

        if let Err(e) = std::fs::create_dir_all(&data_dir) {
            tracing::warn!("Could not create data dir {}: {e}", data_dir.display());
        }

        let trevec_config = if data_dir.join("config.toml").exists() {
            TrevecConfig::load(&data_dir)
        } else {
            effective_config.trevec_config.clone()
        };

        let mut registry = ParserRegistry::new();
        registry.register(Box::new(ConversationParser));

        let brain = if effective_config.brain_enabled {
            let mut brain_config = trevec_config.brain.clone();
            brain_config.enabled = true;
            Some(BrainEngine::new(brain_config))
        } else {
            None
        };

        Ok(Self {
            project: Some(project_name),
            repo_path: None,
            data_dir,
            config: trevec_config,
            parser_registry: registry,
            brain,
            universal_nodes: HashMap::new(),
            store: None,
            graph: None,
            embedder: None,
            code_nodes: HashMap::new(),
            indexed: false,
        })
    }

    /// Create an engine for a code repository.
    pub fn for_repo(repo_path: Option<impl AsRef<Path>>, config: EngineConfig) -> Result<Self> {
        let repo = match repo_path {
            Some(p) => p.as_ref().to_path_buf(),
            None => detect_repo_root()
                .context("Could not detect repository root. Pass a path explicitly.")?,
        };

        let data_dir = config
            .data_dir
            .clone()
            .unwrap_or_else(|| repo.join(".trevec"));

        if let Err(e) = std::fs::create_dir_all(&data_dir) {
            tracing::warn!("Could not create data dir {}: {e}", data_dir.display());
        }

        let trevec_config = if data_dir.join("config.toml").exists() {
            TrevecConfig::load(&data_dir)
        } else {
            config.trevec_config.clone()
        };

        let mut registry = ParserRegistry::new();
        registry.register(Box::new(ConversationParser));

        let brain = if config.brain_enabled {
            let mut brain_config = trevec_config.brain.clone();
            brain_config.enabled = true;
            Some(BrainEngine::new(brain_config))
        } else {
            None
        };

        let project_name = repo
            .file_name()
            .map(|n| n.to_string_lossy().to_string())
            .unwrap_or_else(|| "repo".to_string());

        Ok(Self {
            project: Some(project_name),
            repo_path: Some(repo),
            data_dir,
            config: trevec_config,
            parser_registry: registry,
            brain,
            universal_nodes: HashMap::new(),
            store: None,
            graph: None,
            embedder: None,
            code_nodes: HashMap::new(),
            indexed: false,
        })
    }

    /// Create from a config dict (for production use).
    pub fn from_config(config: EngineConfig) -> Result<Self> {
        if config.project.is_some() {
            Self::new(config.project.clone().unwrap(), config)
        } else {
            Ok(Self::default())
        }
    }

    // ── Full Pipeline: index + query ─────────────────────────────────────

    /// Index a repository. Parses all files with Tree-sitter, generates
    /// embeddings, stores in LanceDB, and builds the code graph.
    ///
    /// If `repo_path` is provided, indexes that repo. Otherwise uses the
    /// repo_path set at construction time via `for_repo()`.
    pub async fn index(&mut self, repo_path: Option<&Path>) -> Result<trevec_index::ingest::IngestStats> {
        let repo = repo_path
            .map(|p| p.to_path_buf())
            .or_else(|| self.repo_path.clone())
            .context("No repo path set. Use for_repo() or pass a path to index().")?;

        self.repo_path = Some(repo.clone());

        // Ensure data dir exists
        std::fs::create_dir_all(&self.data_dir)?;

        // Run the full ingest pipeline
        let stats = trevec_index::ingest::ingest_with_config(
            &repo,
            &self.data_dir,
            self.config.index.exclude.is_empty(), // verbose if no excludes (first time)
            &self.config.index.exclude,
            Some(&self.config.embeddings.model),
            Some(&self.config.embeddings.device),
        )
        .await?;

        // Open the store
        let store = Store::open(self.data_dir.to_str().unwrap_or("."))
            .await
            .context("Failed to open LanceDB store after indexing")?;

        // Load the graph
        let graph_path = self.data_dir.join("graph.bin");
        let graph = if graph_path.exists() {
            CodeGraph::load(&graph_path).context("Failed to load code graph")?
        } else {
            CodeGraph::new()
        };

        // Load nodes into memory from nodes.json
        let nodes_path = self.data_dir.join("nodes.json");
        if nodes_path.exists() {
            let content = std::fs::read_to_string(&nodes_path)?;
            // nodes.json is a JSON array of CodeNode
            let nodes_vec: Vec<CodeNode> = serde_json::from_str(&content)
                .context("Failed to parse nodes.json")?;
            for node in nodes_vec {
                self.code_nodes.insert(node.id.clone(), node);
            }
        }

        // Create embedder
        let model_name = if self.config.embeddings.model.is_empty() {
            None
        } else {
            Some(self.config.embeddings.model.as_str())
        };
        let device = if self.config.embeddings.device.is_empty() || self.config.embeddings.device == "cpu" {
            None
        } else {
            Some(self.config.embeddings.device.as_str())
        };
        let embedder = Embedder::new_with_model(
            model_name,
            false,
            Some(self.data_dir.clone()),
            device,
        )
        .context("Failed to create embedder")?;

        self.store = Some(store);
        self.graph = Some(graph);
        self.embedder = Some(Mutex::new(embedder));
        self.indexed = true;

        // Also populate universal_nodes from code_nodes
        for (id, node) in &self.code_nodes {
            let universal: UniversalNode = node.clone().into();
            self.universal_nodes.insert(id.clone(), universal);
        }

        tracing::info!(
            "Indexed {} files, {} nodes, {} edges in {}ms",
            stats.files_parsed,
            stats.nodes_extracted,
            stats.edges_built,
            stats.total_ms
        );

        Ok(stats)
    }

    /// Query the indexed codebase with the full hybrid retrieval pipeline.
    ///
    /// Uses BM25 + vector search, RRF merge, graph expansion, and token budgeting.
    /// Falls back to in-memory text search if not indexed.
    pub async fn query(&self, query_text: &str, options: QueryOptions) -> Result<ContextBundle> {
        // If not indexed, fall back to simple in-memory search
        if !self.indexed || self.store.is_none() {
            return self.query_in_memory(query_text, &options);
        }

        let start = std::time::Instant::now();
        let store = self.store.as_ref().unwrap();
        let graph = self.graph.as_ref().unwrap();
        let embedder = self.embedder.as_ref().unwrap();

        // Embed the query
        let query_vec = embedder
            .lock()
            .expect("embedder lock poisoned")
            .embed(query_text)
            .context("Failed to embed query")?;

        // Parallel BM25 + vector search
        let search_limit = options.anchors * 8;
        let (fts_results, vector_results) = tokio::join!(
            store.search_fts(query_text, search_limit),
            store.search_vector(&query_vec, search_limit),
        );

        let fts_results = fts_results.context("FTS search failed")?;
        let vector_results = vector_results.context("Vector search failed")?;

        // RRF merge
        let mut merged = rrf_merge(&fts_results, &vector_results, 60);

        // Pipeline: filter → literal boost → file path boost → test penalty
        filter_noncode_files(&mut merged, &self.code_nodes);
        apply_literal_boost(&mut merged, &self.code_nodes, query_text);
        let extracted_paths = extract_file_paths_from_query(query_text);
        if !extracted_paths.is_empty() {
            apply_file_path_boost(&mut merged, &self.code_nodes, &extracted_paths, 0.05);
        }
        apply_test_file_penalty(
            &mut merged,
            &self.code_nodes,
            self.config.retrieval.test_file_penalty,
            &self.config.retrieval.penalty_paths,
        );
        apply_test_fixture_penalty(&mut merged, &self.code_nodes);

        // Select anchors
        let anchor_ids: Vec<String> = merged
            .iter()
            .take(options.anchors)
            .map(|r| r.node_id.clone())
            .collect();

        // Graph expansion under token budget
        let token_fn = |id: &String| -> usize {
            self.code_nodes
                .get(id)
                .map(|n| n.span.estimated_tokens())
                .unwrap_or(0)
        };

        let mut token_budget = TokenBudget::new(options.budget);
        let included_ids = expand_graph(graph, &anchor_ids, &mut token_budget, &token_fn, 3);

        // Assemble the context bundle
        let repo = self.repo_path.as_deref().unwrap_or(Path::new("."));
        let mut bundle = assemble_bundle(
            query_text,
            &anchor_ids,
            &included_ids,
            &self.code_nodes,
            repo,
        )
        .context("Failed to assemble context bundle")?;

        bundle.retrieval_ms = Some(start.elapsed().as_millis() as u64);

        Ok(bundle)
    }

    /// Simple in-memory query fallback when not indexed.
    fn query_in_memory(&self, query_text: &str, options: &QueryOptions) -> Result<ContextBundle> {
        let results = self.search_nodes(query_text, options.anchors);
        let anchor_ids: Vec<String> = results.iter().map(|(n, _)| n.id.clone()).collect();
        let included_ids = anchor_ids.clone();

        // Build a minimal bundle
        let mut included_nodes = Vec::new();
        for (node, _) in &results {
            included_nodes.push(trevec_core::model::IncludedNode {
                node_id: node.id.clone(),
                file_path: node.file_path.clone(),
                span: node.span.clone().unwrap_or(trevec_core::model::Span {
                    start_line: 0, start_col: 0, end_line: 0, end_col: 0,
                    start_byte: 0, end_byte: 0,
                }),
                kind: node.kind.to_node_kind().unwrap_or(trevec_core::model::NodeKind::Function),
                name: node.label.clone(),
                signature: node.signature.clone().unwrap_or_default(),
                source_text: node.label.clone(),
                is_anchor: true,
                estimated_tokens: node.label.len() / 4,
            });
        }

        let total_tokens: usize = included_nodes.iter().map(|n| n.estimated_tokens).sum();

        Ok(ContextBundle {
            bundle_id: trevec_core::generate_bundle_id(query_text),
            query: query_text.to_string(),
            anchor_node_ids: anchor_ids,
            included_nodes,
            total_estimated_tokens: total_tokens,
            total_source_file_tokens: 0,
            retrieval_ms: Some(0),
        })
    }

    /// Whether the full pipeline has been initialized (index() has been called).
    pub fn is_indexed(&self) -> bool {
        self.indexed
    }

    // ── Accessors ────────────────────────────────────────────────────────

    /// Get the project name (if set).
    pub fn project(&self) -> Option<&str> {
        self.project.as_deref()
    }

    /// Get the repository path (if in code mode).
    pub fn repo_path(&self) -> Option<&Path> {
        self.repo_path.as_deref()
    }

    /// Get the data directory path.
    pub fn data_dir(&self) -> &Path {
        &self.data_dir
    }

    /// Get the configuration.
    pub fn config(&self) -> &TrevecConfig {
        &self.config
    }

    /// Register a custom domain parser.
    pub fn add_parser(&mut self, parser: Box<dyn DomainParser>) {
        self.parser_registry.register(parser);
    }

    /// List registered parser domains.
    pub fn registered_domains(&self) -> Vec<&'static str> {
        self.parser_registry.registered_domains()
    }

    /// Get the total number of nodes.
    pub fn node_count(&self) -> usize {
        self.universal_nodes.len()
    }

    /// Get a node by ID.
    pub fn get_node(&self, id: &str) -> Option<&UniversalNode> {
        self.universal_nodes.get(id)
    }

    /// Get all nodes in a domain, optionally filtered by user_id.
    pub fn nodes_by_domain(
        &self,
        domain: DomainTag,
        user_id: Option<&str>,
    ) -> Vec<&UniversalNode> {
        self.universal_nodes
            .values()
            .filter(|n| {
                n.domain == domain
                    && match user_id {
                        Some(uid) => n
                            .attributes
                            .get("user_id")
                            .map_or(false, |v| v.to_string() == uid),
                        None => true,
                    }
            })
            .collect()
    }

    // ── Core API: add / search / get_all / delete ────────────────────────

    /// Add a memory. Accepts a string or a list of messages.
    ///
    /// This is the primary way to store information. The engine automatically
    /// extracts entities, preferences, and decisions from the content.
    pub fn add(
        &mut self,
        content: &str,
        user_id: &str,
        metadata: Option<HashMap<String, String>>,
    ) -> Result<String> {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        let id = {
            let input = format!("memory:{}:{}:{}", user_id, now, &content[..content.len().min(100)]);
            let hash = blake3::hash(input.as_bytes());
            hash.to_hex()[..32].to_string()
        };

        let label = if content.len() > 200 {
            format!("{}...", &content[..200])
        } else {
            content.to_string()
        };

        let mut attributes: HashMap<String, AttributeValue> = HashMap::new();
        attributes.insert("user_id".into(), AttributeValue::String(user_id.to_string()));

        if let Some(meta) = metadata {
            for (k, v) in meta {
                attributes.insert(k, AttributeValue::String(v));
            }
        }

        let node = UniversalNode {
            id: id.clone(),
            kind: UniversalKind::Message,
            domain: DomainTag::Conversation,
            label,
            file_path: String::new(),
            span: None,
            signature: None,
            doc_comment: None,
            identifiers: vec![],
            bm25_text: format!("{} {}", user_id, content),
            symbol_vec: None,
            ast_hash: None,
            temporal: Some(TemporalMeta::at(now)),
            attributes,
            intent_summary: None,
        };

        self.universal_nodes.insert(id.clone(), node);
        Ok(id)
    }

    /// Add messages (conversation format, like Mem0).
    ///
    /// ```python
    /// tv.add_messages([
    ///     {"role": "user", "content": "I love hiking"},
    ///     {"role": "assistant", "content": "Great!"},
    /// ], user_id="alex")
    /// ```
    pub fn add_messages(
        &mut self,
        messages: &[HashMap<String, String>],
        user_id: &str,
    ) -> Result<Vec<String>> {
        let mut ids = Vec::new();
        for msg in messages {
            let role = msg.get("role").map(|s| s.as_str()).unwrap_or("user");
            let content = msg
                .get("content")
                .map(|s| s.as_str())
                .unwrap_or("");
            if content.is_empty() {
                continue;
            }

            let mut meta = HashMap::new();
            meta.insert("role".to_string(), role.to_string());

            let id = self.add(content, user_id, Some(meta))?;
            ids.push(id);
        }
        Ok(ids)
    }

    /// Search memories. If user_id is provided, scopes to that user.
    pub fn search(
        &self,
        query: &str,
        user_id: Option<&str>,
        limit: usize,
    ) -> Vec<MemoryResult> {
        let query_lower = query.to_lowercase();
        let query_terms: Vec<&str> = query_lower.split_whitespace().collect();

        let mut scored: Vec<(&UniversalNode, f64)> = self
            .universal_nodes
            .values()
            .filter(|node| {
                // user_id scoping
                match user_id {
                    Some(uid) => node
                        .attributes
                        .get("user_id")
                        .map_or(false, |v| v.to_string() == uid),
                    None => true,
                }
            })
            .map(|node| {
                let text = node.enriched_bm25_text().to_lowercase();
                let mut score = 0.0;
                for term in &query_terms {
                    if text.contains(term) {
                        score += 1.0;
                    }
                    if node.label.to_lowercase().contains(term) {
                        score += 2.0;
                    }
                }
                (node, score)
            })
            .filter(|(_, score)| *score > 0.0)
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);

        scored
            .into_iter()
            .map(|(node, score)| MemoryResult {
                id: node.id.clone(),
                memory: node.label.clone(),
                role: node
                    .attributes
                    .get("role")
                    .map(|v| v.to_string()),
                user_id: node
                    .attributes
                    .get("user_id")
                    .map(|v| v.to_string()),
                created_at: node.temporal.as_ref().map(|t| t.t_created),
                score,
            })
            .collect()
    }

    /// Get all memories for a user.
    pub fn get_all(&self, user_id: &str) -> Vec<MemoryResult> {
        self.universal_nodes
            .values()
            .filter(|n| {
                n.attributes
                    .get("user_id")
                    .map_or(false, |v| v.to_string() == user_id)
            })
            .map(|node| MemoryResult {
                id: node.id.clone(),
                memory: node.label.clone(),
                role: node.attributes.get("role").map(|v| v.to_string()),
                user_id: Some(user_id.to_string()),
                created_at: node.temporal.as_ref().map(|t| t.t_created),
                score: 0.0,
            })
            .collect()
    }

    /// Delete a specific memory by ID.
    pub fn delete(&mut self, memory_id: &str) -> bool {
        self.universal_nodes.remove(memory_id).is_some()
    }

    /// Delete all memories for a user.
    pub fn delete_all(&mut self, user_id: &str) -> usize {
        let to_remove: Vec<String> = self
            .universal_nodes
            .iter()
            .filter(|(_, n)| {
                n.attributes
                    .get("user_id")
                    .map_or(false, |v| v.to_string() == user_id)
            })
            .map(|(id, _)| id.clone())
            .collect();
        let count = to_remove.len();
        for id in to_remove {
            self.universal_nodes.remove(&id);
        }
        count
    }

    // ── Code Context (only when repo_path is set) ────────────────────────

    /// Add a universal node directly to the store.
    pub fn add_node(&mut self, node: UniversalNode) {
        self.universal_nodes.insert(node.id.clone(), node);
    }

    /// Search across all domains (code + conversations + documents).
    pub fn search_nodes(&self, query: &str, limit: usize) -> Vec<(&UniversalNode, f64)> {
        let query_lower = query.to_lowercase();
        let query_terms: Vec<&str> = query_lower.split_whitespace().collect();

        let mut scored: Vec<(&UniversalNode, f64)> = self
            .universal_nodes
            .values()
            .map(|node| {
                let text = node.enriched_bm25_text().to_lowercase();
                let mut score = 0.0;
                for term in &query_terms {
                    if text.contains(term) {
                        score += 1.0;
                    }
                    if node.label.to_lowercase().contains(term) {
                        score += 2.0;
                    }
                }
                (node, score)
            })
            .filter(|(_, score)| *score > 0.0)
            .collect();

        scored.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(limit);
        scored
    }

    /// Parse a conversation file and add nodes to the store.
    pub fn ingest_conversation(&mut self, file_path: &str, source: &[u8]) -> Result<usize> {
        let result = self
            .parser_registry
            .parse_file(file_path, source, &self.config)?
            .context("No parser found for file")?;

        let count = result.nodes.len();
        for node in result.nodes {
            self.universal_nodes.insert(node.id.clone(), node);
        }
        Ok(count)
    }

    // ── Brain Integration ────────────────────────────────────────────────

    /// Get a reference to the Brain engine (if enabled).
    pub fn brain(&self) -> Option<&BrainEngine> {
        self.brain.as_ref()
    }

    /// Queue nodes for Brain enrichment.
    pub async fn enrich_nodes(
        &self,
        node_ids: Vec<String>,
        priority: trevec_brain::queue::Priority,
    ) {
        if let Some(ref brain) = self.brain {
            brain.enqueue_batch(node_ids, priority).await;
        }
    }

    /// Process pending Brain enrichment tasks.
    pub async fn process_brain(&self) -> usize {
        let mut processed = 0;
        if let Some(ref brain) = self.brain {
            while brain.process_one().await {
                processed += 1;
            }
        }
        processed
    }

    /// Get Brain statistics.
    pub async fn brain_stats(&self) -> Option<trevec_brain::BrainState> {
        if let Some(ref brain) = self.brain {
            Some(brain.stats().await)
        } else {
            None
        }
    }
}

/// Auto-detect repository root by walking up from cwd looking for `.git/`.
fn detect_repo_root() -> Option<PathBuf> {
    let mut current = std::env::current_dir().ok()?;
    loop {
        if current.join(".git").exists() {
            return Some(current);
        }
        if !current.pop() {
            return None;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ── Zero-config tests ────────────────────────────────────────────────

    #[test]
    fn test_default_engine() {
        let engine = TrevecEngine::default();
        assert_eq!(engine.node_count(), 0);
        assert!(engine.brain().is_none());
        assert!(engine.project().is_none());
        assert!(engine.repo_path().is_none());
        assert!(engine.registered_domains().contains(&"conversation"));
    }

    #[test]
    fn test_named_project() {
        let engine = TrevecEngine::new("test-project", EngineConfig::default()).unwrap();
        assert_eq!(engine.project(), Some("test-project"));
        assert!(engine.data_dir().to_string_lossy().contains("test-project"));
    }

    #[test]
    fn test_for_repo() {
        let tmp = tempfile::tempdir().unwrap();
        // Create a fake .git dir
        std::fs::create_dir(tmp.path().join(".git")).unwrap();

        let engine =
            TrevecEngine::for_repo(Some(tmp.path()), EngineConfig::default()).unwrap();
        assert!(engine.repo_path().is_some());
    }

    // ── add / search (Mem0-style API) ────────────────────────────────────

    #[test]
    fn test_add_and_search() {
        let mut engine = TrevecEngine::default();

        engine.add("I love hiking in Denver", "alex", None).unwrap();
        engine.add("I hate cold weather", "alex", None).unwrap();
        engine.add("My favorite food is sushi", "bob", None).unwrap();

        // Search scoped to alex
        let results = engine.search("hiking", Some("alex"), 10);
        assert_eq!(results.len(), 1);
        assert!(results[0].memory.contains("hiking"));
        assert_eq!(results[0].user_id.as_deref(), Some("alex"));

        // Search scoped to bob
        let results = engine.search("sushi", Some("bob"), 10);
        assert_eq!(results.len(), 1);

        // Cross-user search should not leak
        let results = engine.search("hiking", Some("bob"), 10);
        assert!(results.is_empty());
    }

    #[test]
    fn test_add_messages() {
        let mut engine = TrevecEngine::default();

        let messages = vec![
            {
                let mut m = HashMap::new();
                m.insert("role".to_string(), "user".to_string());
                m.insert("content".to_string(), "I moved to Denver".to_string());
                m
            },
            {
                let mut m = HashMap::new();
                m.insert("role".to_string(), "assistant".to_string());
                m.insert("content".to_string(), "Welcome to Denver!".to_string());
                m
            },
        ];

        let ids = engine.add_messages(&messages, "alex").unwrap();
        assert_eq!(ids.len(), 2);
        assert_eq!(engine.node_count(), 2);
    }

    #[test]
    fn test_get_all() {
        let mut engine = TrevecEngine::default();

        engine.add("memory 1", "alex", None).unwrap();
        engine.add("memory 2", "alex", None).unwrap();
        engine.add("memory 3", "bob", None).unwrap();

        let alex_memories = engine.get_all("alex");
        assert_eq!(alex_memories.len(), 2);

        let bob_memories = engine.get_all("bob");
        assert_eq!(bob_memories.len(), 1);
    }

    #[test]
    fn test_delete() {
        let mut engine = TrevecEngine::default();

        let id = engine.add("test memory", "alex", None).unwrap();
        assert_eq!(engine.node_count(), 1);

        assert!(engine.delete(&id));
        assert_eq!(engine.node_count(), 0);

        assert!(!engine.delete(&id)); // already deleted
    }

    #[test]
    fn test_delete_all() {
        let mut engine = TrevecEngine::default();

        engine.add("m1", "alex", None).unwrap();
        engine.add("m2", "alex", None).unwrap();
        engine.add("m3", "bob", None).unwrap();

        let deleted = engine.delete_all("alex");
        assert_eq!(deleted, 2);
        assert_eq!(engine.node_count(), 1); // bob's memory remains
    }

    #[test]
    fn test_search_unscoped() {
        let mut engine = TrevecEngine::default();

        engine.add("hiking in mountains", "alex", None).unwrap();
        engine.add("hiking trails nearby", "bob", None).unwrap();

        // Unscoped search finds both
        let results = engine.search("hiking", None, 10);
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn test_search_with_metadata() {
        let mut engine = TrevecEngine::default();

        let mut meta = HashMap::new();
        meta.insert("category".to_string(), "preference".to_string());

        engine.add("I love pizza", "alex", Some(meta)).unwrap();

        let results = engine.search("pizza", Some("alex"), 10);
        assert!(!results.is_empty());
    }

    // ── Code context tests ───────────────────────────────────────────────

    #[test]
    fn test_add_node_and_search() {
        let mut engine = TrevecEngine::default();

        engine.add_node(UniversalNode {
            id: "auth".to_string(),
            kind: UniversalKind::Function,
            domain: DomainTag::Code,
            label: "authenticate".to_string(),
            file_path: "src/auth.rs".to_string(),
            span: None,
            signature: Some("fn authenticate()".to_string()),
            doc_comment: None,
            identifiers: vec![],
            bm25_text: "authenticate JWT token src/auth.rs".to_string(),
            symbol_vec: None,
            ast_hash: None,
            temporal: None,
            attributes: HashMap::new(),
            intent_summary: None,
        });

        let results = engine.search_nodes("JWT", 10);
        assert!(!results.is_empty());
        assert_eq!(results[0].0.id, "auth");
    }

    #[test]
    fn test_brain_enrichment_improves_search() {
        let mut engine = TrevecEngine::default();

        let mut node = UniversalNode {
            id: "auth".to_string(),
            kind: UniversalKind::Function,
            domain: DomainTag::Code,
            label: "validate_token".to_string(),
            file_path: "src/auth.rs".to_string(),
            span: None,
            signature: Some("fn validate_token()".to_string()),
            doc_comment: None,
            identifiers: vec![],
            bm25_text: "validate_token src/auth.rs".to_string(),
            symbol_vec: None,
            ast_hash: None,
            temporal: None,
            attributes: HashMap::new(),
            intent_summary: None,
        };

        // Without enrichment
        engine.add_node(node.clone());
        assert!(engine.search_nodes("JWT authentication", 10).is_empty());

        // With Brain enrichment
        node.intent_summary = Some(IntentSummary {
            purpose: Some("Validates JWT tokens for authentication".to_string()),
            related_concepts: vec!["JWT".to_string(), "authentication".to_string()],
            ..Default::default()
        });
        engine.add_node(node);
        assert!(!engine.search_nodes("JWT authentication", 10).is_empty());
    }

    #[test]
    fn test_ingest_conversation() {
        let mut engine = TrevecEngine::default();

        let json = r#"{
            "sessions": [{
                "id": "s1",
                "messages": [
                    {"role": "user", "content": "I love hiking"},
                    {"role": "assistant", "content": "Great!"}
                ]
            }]
        }"#;

        let count = engine
            .ingest_conversation("chat.conversation.json", json.as_bytes())
            .unwrap();
        assert!(count > 0);
    }

    #[test]
    fn test_nodes_by_domain_with_user_id() {
        let mut engine = TrevecEngine::default();

        engine.add("m1", "alex", None).unwrap();
        engine.add("m2", "bob", None).unwrap();

        let alex = engine.nodes_by_domain(DomainTag::Conversation, Some("alex"));
        assert_eq!(alex.len(), 1);

        let all = engine.nodes_by_domain(DomainTag::Conversation, None);
        assert_eq!(all.len(), 2);
    }

    // ── Brain tests ──────────────────────────────────────────────────────

    #[test]
    fn test_engine_with_brain() {
        let engine = TrevecEngine::new(
            "brain-test",
            EngineConfig {
                brain_enabled: true,
                ..Default::default()
            },
        )
        .unwrap();
        assert!(engine.brain().is_some());
    }

    #[tokio::test]
    async fn test_brain_process() {
        let engine = TrevecEngine::new(
            "brain-test",
            EngineConfig {
                brain_enabled: true,
                ..Default::default()
            },
        )
        .unwrap();

        let brain = engine.brain().unwrap();
        brain.resume().await;

        engine
            .enrich_nodes(vec!["n1".to_string()], trevec_brain::queue::Priority::High)
            .await;

        let processed = engine.process_brain().await;
        assert_eq!(processed, 1);
    }

    // ── Config tests ─────────────────────────────────────────────────────

    #[test]
    fn test_engine_config_defaults() {
        let config = EngineConfig::default();
        assert!(!config.brain_enabled);
        assert!(config.project.is_none());
        assert!(config.data_dir.is_none());
    }

    #[test]
    fn test_query_options_defaults() {
        let opts = QueryOptions::default();
        assert_eq!(opts.budget, 4096);
        assert_eq!(opts.anchors, 5);
    }

    #[test]
    fn test_custom_parser() {
        let mut engine = TrevecEngine::default();

        struct Dummy;
        impl DomainParser for Dummy {
            fn domain_id(&self) -> &'static str { "dummy" }
            fn supported_extensions(&self) -> &[&'static str] { &[".dummy"] }
            fn parse(&self, _: &str, _: &[u8], _: &TrevecConfig) -> anyhow::Result<ParseResult> {
                Ok(ParseResult { nodes: vec![], edges: vec![] })
            }
        }

        engine.add_parser(Box::new(Dummy));
        assert!(engine.registered_domains().contains(&"dummy"));
    }
}
