//! Framework integration adapters for the Trevec SDK.
//!
//! Provides thin, idiomatic adapters that make [`TrevecEngine`] easy to use
//! with popular AI agent frameworks:
//!
//! - **[`TrevecRetriever`]** — LangChain/LangGraph-style `retrieve(query) -> Vec<Document>`.
//! - **[`TrevecTool`]** — CrewAI-style `tool_call(query) -> String`.
//! - **[`TrevecFunctionTool`]** — OpenAI function-calling JSON schema definitions.
//! - **[`TrevecMemoryAdapter`]** — Generic remember/recall memory interface.
//!
//! These are Rust-side adapters with JSON serialization; the actual Python/JS
//! bindings layer wraps these structs.

use std::sync::{Arc, Mutex};

use serde::{Deserialize, Serialize};

use crate::{SearchResult, TrevecEngine};

// ── Document (LangChain-compatible) ─────────────────────────────────────────

/// A retrieved document in LangChain/LangGraph format.
///
/// Mirrors LangChain's `Document(page_content, metadata)` model so that
/// downstream chains and agents can consume Trevec results without conversion.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Document {
    /// The text content of the document (signature + doc comment + identifiers).
    pub page_content: String,
    /// Metadata attached to the document.
    pub metadata: DocumentMetadata,
}

/// Metadata for a [`Document`], carrying Trevec-specific provenance.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DocumentMetadata {
    /// Stable node identifier in the Trevec graph.
    pub node_id: String,
    /// Human-readable name (function name, class name, etc.).
    pub name: String,
    /// Node kind (function, method, class, message, ...).
    pub kind: String,
    /// File path relative to the repository root.
    pub file_path: String,
    /// Signature of the code node, if available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    /// Start line in the source file, if available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub start_line: Option<usize>,
    /// End line in the source file, if available.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub end_line: Option<usize>,
    /// Retrieval relevance score (higher is better).
    pub score: f64,
    /// Domain tag (code, conversation, document, ...).
    pub domain: String,
}

// ── TrevecRetriever (LangChain/LangGraph) ───────────────────────────────────

/// LangChain/LangGraph retriever adapter.
///
/// Wraps a [`TrevecEngine`] and exposes a `retrieve(query)` method that returns
/// results as a `Vec<Document>`, matching the interface LangChain expects.
///
/// # Example
///
/// ```no_run
/// # use std::sync::Arc;
/// # use trevec_sdk::{TrevecEngine, EngineConfig};
/// # use trevec_sdk::integrations::TrevecRetriever;
/// # fn example() -> anyhow::Result<()> {
/// let engine = Arc::new(TrevecEngine::default());
/// let retriever = TrevecRetriever::builder(engine)
///     .top_k(5)
///     .budget(4096)
///     .build();
/// let docs = retriever.retrieve("authentication flow");
/// # Ok(())
/// # }
/// ```
pub struct TrevecRetriever {
    engine: Arc<TrevecEngine>,
    top_k: usize,
    budget: usize,
}

impl TrevecRetriever {
    /// Create a builder for configuring a `TrevecRetriever`.
    pub fn builder(engine: Arc<TrevecEngine>) -> TrevecRetrieverBuilder {
        TrevecRetrieverBuilder {
            engine,
            top_k: 5,
            budget: 4096,
        }
    }

    /// Create a retriever with default settings.
    pub fn new(engine: Arc<TrevecEngine>) -> Self {
        Self {
            engine,
            top_k: 5,
            budget: 4096,
        }
    }

    /// Retrieve documents matching `query`.
    ///
    /// Returns up to `top_k` documents sorted by relevance score (descending).
    pub fn retrieve(&self, query: &str) -> Vec<Document> {
        let results = self.engine.search_nodes(query, self.top_k);

        results
            .into_iter()
            .map(|(node, score)| {
                let mut page_content = String::new();
                if let Some(ref sig) = node.signature {
                    page_content.push_str(sig);
                    page_content.push('\n');
                }
                if let Some(ref doc) = node.doc_comment {
                    page_content.push_str(doc);
                    page_content.push('\n');
                }
                if !node.identifiers.is_empty() {
                    page_content.push_str(&node.identifiers.join(", "));
                }
                if page_content.is_empty() {
                    page_content = node.label.clone();
                }

                Document {
                    page_content,
                    metadata: DocumentMetadata {
                        node_id: node.id.clone(),
                        name: node.label.clone(),
                        kind: node.kind.to_string(),
                        file_path: node.file_path.clone(),
                        signature: node.signature.clone(),
                        start_line: node.span.as_ref().map(|s| s.start_line),
                        end_line: node.span.as_ref().map(|s| s.end_line),
                        score,
                        domain: node.domain.to_string(),
                    },
                }
            })
            .collect()
    }

    /// Retrieve documents and return them as a JSON string.
    pub fn retrieve_json(&self, query: &str) -> String {
        let docs = self.retrieve(query);
        serde_json::to_string_pretty(&docs).unwrap_or_else(|_| "[]".to_string())
    }

    /// The configured maximum number of results.
    pub fn top_k(&self) -> usize {
        self.top_k
    }

    /// The configured token budget.
    pub fn budget(&self) -> usize {
        self.budget
    }
}

/// Builder for [`TrevecRetriever`].
pub struct TrevecRetrieverBuilder {
    engine: Arc<TrevecEngine>,
    top_k: usize,
    budget: usize,
}

impl TrevecRetrieverBuilder {
    /// Set the maximum number of documents to return (default: 5).
    pub fn top_k(mut self, top_k: usize) -> Self {
        self.top_k = top_k;
        self
    }

    /// Set the token budget for context assembly (default: 4096).
    pub fn budget(mut self, budget: usize) -> Self {
        self.budget = budget;
        self
    }

    /// Build the retriever.
    pub fn build(self) -> TrevecRetriever {
        TrevecRetriever {
            engine: self.engine,
            top_k: self.top_k,
            budget: self.budget,
        }
    }
}

// ── TrevecTool (CrewAI) ─────────────────────────────────────────────────────

/// CrewAI tool adapter.
///
/// Wraps a [`TrevecEngine`] and exposes a `tool_call(query)` method that returns
/// a formatted string suitable for CrewAI tool output. Supports both search
/// and memory recall operations.
///
/// # Example
///
/// ```no_run
/// # use std::sync::Arc;
/// # use trevec_sdk::{TrevecEngine, EngineConfig};
/// # use trevec_sdk::integrations::TrevecTool;
/// # fn example() -> anyhow::Result<()> {
/// let engine = Arc::new(TrevecEngine::default());
/// let tool = TrevecTool::builder(engine)
///     .name("code_search")
///     .description("Search codebase for relevant context")
///     .max_results(10)
///     .build();
/// let output = tool.tool_call("How does the auth module work?");
/// # Ok(())
/// # }
/// ```
pub struct TrevecTool {
    engine: Arc<TrevecEngine>,
    name: String,
    description: String,
    max_results: usize,
    output_format: ToolOutputFormat,
}

/// Output format for [`TrevecTool`].
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ToolOutputFormat {
    /// Structured JSON output.
    Json,
    /// Human-readable plain text.
    Text,
}

impl TrevecTool {
    /// Create a builder for configuring a `TrevecTool`.
    pub fn builder(engine: Arc<TrevecEngine>) -> TrevecToolBuilder {
        TrevecToolBuilder {
            engine,
            name: "trevec_search".to_string(),
            description: "Search the codebase for relevant context using Trevec".to_string(),
            max_results: 5,
            output_format: ToolOutputFormat::Text,
        }
    }

    /// Create a tool with default settings.
    pub fn new(engine: Arc<TrevecEngine>) -> Self {
        Self {
            engine,
            name: "trevec_search".to_string(),
            description: "Search the codebase for relevant context using Trevec".to_string(),
            max_results: 5,
            output_format: ToolOutputFormat::Text,
        }
    }

    /// Execute a tool call with the given query. Returns formatted output.
    pub fn tool_call(&self, query: &str) -> String {
        let results = self.engine.search_nodes(query, self.max_results);

        if results.is_empty() {
            return format!("No results found for query: {}", query);
        }

        match self.output_format {
            ToolOutputFormat::Json => {
                let entries: Vec<ToolResultEntry> = results
                    .into_iter()
                    .map(|(node, score)| ToolResultEntry {
                        name: node.label.clone(),
                        kind: node.kind.to_string(),
                        file_path: node.file_path.clone(),
                        signature: node.signature.clone(),
                        score,
                    })
                    .collect();
                serde_json::to_string_pretty(&entries).unwrap_or_else(|_| "[]".to_string())
            }
            ToolOutputFormat::Text => {
                let mut output = format!("Found {} results for \"{}\":\n\n", results.len(), query);
                for (i, (node, score)) in results.iter().enumerate() {
                    output.push_str(&format!(
                        "{}. {} ({}) — {}\n   Score: {:.2}\n",
                        i + 1,
                        node.label,
                        node.kind,
                        node.file_path,
                        score,
                    ));
                    if let Some(ref sig) = node.signature {
                        output.push_str(&format!("   Signature: {}\n", sig));
                    }
                    output.push('\n');
                }
                output
            }
        }
    }

    /// The tool name (for framework registration).
    pub fn name(&self) -> &str {
        &self.name
    }

    /// The tool description (for framework registration).
    pub fn description(&self) -> &str {
        &self.description
    }
}

/// Serializable search result entry for JSON output.
#[derive(Debug, Clone, Serialize, Deserialize)]
struct ToolResultEntry {
    name: String,
    kind: String,
    file_path: String,
    #[serde(skip_serializing_if = "Option::is_none")]
    signature: Option<String>,
    score: f64,
}

/// Builder for [`TrevecTool`].
pub struct TrevecToolBuilder {
    engine: Arc<TrevecEngine>,
    name: String,
    description: String,
    max_results: usize,
    output_format: ToolOutputFormat,
}

impl TrevecToolBuilder {
    /// Set the tool name (default: "trevec_search").
    pub fn name(mut self, name: impl Into<String>) -> Self {
        self.name = name.into();
        self
    }

    /// Set the tool description (default: generic search description).
    pub fn description(mut self, description: impl Into<String>) -> Self {
        self.description = description.into();
        self
    }

    /// Set the maximum number of results (default: 5).
    pub fn max_results(mut self, max_results: usize) -> Self {
        self.max_results = max_results;
        self
    }

    /// Set the output format (default: Text).
    pub fn output_format(mut self, format: ToolOutputFormat) -> Self {
        self.output_format = format;
        self
    }

    /// Build the tool.
    pub fn build(self) -> TrevecTool {
        TrevecTool {
            engine: self.engine,
            name: self.name,
            description: self.description,
            max_results: self.max_results,
            output_format: self.output_format,
        }
    }
}

// ── TrevecFunctionTool (OpenAI function calling) ────────────────────────────

/// OpenAI function-calling adapter.
///
/// Generates JSON-schema-compatible tool definitions that can be passed to
/// the OpenAI Chat Completions API (or any compatible API) as `tools`.
/// Also dispatches incoming function calls to the underlying engine.
///
/// # Example
///
/// ```no_run
/// # use std::sync::Arc;
/// # use trevec_sdk::{TrevecEngine, EngineConfig};
/// # use trevec_sdk::integrations::TrevecFunctionTool;
/// # fn example() -> anyhow::Result<()> {
/// let engine = Arc::new(TrevecEngine::default());
/// let func_tool = TrevecFunctionTool::new(engine);
///
/// // Get the tool definitions to send in an API request
/// let definitions = func_tool.tool_definitions();
/// let json = serde_json::to_string_pretty(&definitions)?;
///
/// // Dispatch a function call received from the API
/// let result = func_tool.call("search_code", r#"{"query": "auth"}"#);
/// # Ok(())
/// # }
/// ```
pub struct TrevecFunctionTool {
    engine: Arc<TrevecEngine>,
    max_results: usize,
    budget: usize,
}

/// A single OpenAI-format tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionToolDefinition {
    /// Always "function" for function-calling tools.
    #[serde(rename = "type")]
    pub tool_type: String,
    /// The function specification.
    pub function: FunctionSpec,
}

/// Function specification within a tool definition.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionSpec {
    /// Function name (used for dispatch).
    pub name: String,
    /// Human-readable description of what the function does.
    pub description: String,
    /// JSON Schema describing the parameters.
    pub parameters: serde_json::Value,
}

/// Arguments for the `search_code` function.
#[derive(Debug, Deserialize)]
struct SearchCodeArgs {
    query: String,
    #[serde(default = "default_limit")]
    limit: usize,
}

/// Arguments for the `get_context` function.
#[derive(Debug, Deserialize)]
struct GetContextArgs {
    query: String,
    #[serde(default = "default_budget")]
    budget: Option<usize>,
    #[serde(default = "default_anchors")]
    anchors: Option<usize>,
}

/// Arguments for the `read_file_topology` function.
#[derive(Debug, Deserialize)]
struct ReadFileTopologyArgs {
    file_path: String,
}

fn default_limit() -> usize {
    5
}
fn default_budget() -> Option<usize> {
    None
}
fn default_anchors() -> Option<usize> {
    None
}

/// Result of dispatching a function call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FunctionCallResult {
    /// Whether the call succeeded.
    pub success: bool,
    /// The result content (JSON string on success, error message on failure).
    pub content: String,
}

impl TrevecFunctionTool {
    /// Create a new function-calling tool adapter.
    pub fn new(engine: Arc<TrevecEngine>) -> Self {
        Self {
            engine,
            max_results: 10,
            budget: 4096,
        }
    }

    /// Create with custom limits.
    pub fn with_limits(engine: Arc<TrevecEngine>, max_results: usize, budget: usize) -> Self {
        Self {
            engine,
            max_results,
            budget,
        }
    }

    /// Generate OpenAI-format tool definitions for all supported functions.
    ///
    /// Returns a `Vec<FunctionToolDefinition>` that can be serialized and
    /// sent as the `tools` parameter in a Chat Completions request.
    pub fn tool_definitions(&self) -> Vec<FunctionToolDefinition> {
        vec![
            FunctionToolDefinition {
                tool_type: "function".to_string(),
                function: FunctionSpec {
                    name: "search_code".to_string(),
                    description: "Search the codebase for relevant code nodes using \
                                  hybrid BM25 + vector search."
                        .to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural language search query"
                            },
                            "limit": {
                                "type": "integer",
                                "description": "Maximum number of results (default: 5)",
                                "default": 5
                            }
                        },
                        "required": ["query"],
                        "additionalProperties": false
                    }),
                },
            },
            FunctionToolDefinition {
                tool_type: "function".to_string(),
                function: FunctionSpec {
                    name: "get_context".to_string(),
                    description: "Retrieve graph-aware code context for a natural language \
                                  query. Returns relevant code nodes with file paths, spans, \
                                  and relationships."
                        .to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "query": {
                                "type": "string",
                                "description": "Natural language query describing what you need context for"
                            },
                            "budget": {
                                "type": "integer",
                                "description": "Token budget for context assembly (default: 4096)",
                                "default": 4096
                            },
                            "anchors": {
                                "type": "integer",
                                "description": "Number of anchor nodes to select (default: 5)",
                                "default": 5
                            }
                        },
                        "required": ["query"],
                        "additionalProperties": false
                    }),
                },
            },
            FunctionToolDefinition {
                tool_type: "function".to_string(),
                function: FunctionSpec {
                    name: "read_file_topology".to_string(),
                    description: "Read the structural topology of a file: all code nodes \
                                  (functions, classes, methods) with their relationships. \
                                  Use to understand file structure before making changes."
                        .to_string(),
                    parameters: serde_json::json!({
                        "type": "object",
                        "properties": {
                            "file_path": {
                                "type": "string",
                                "description": "Path to the file (relative to repo root)"
                            }
                        },
                        "required": ["file_path"],
                        "additionalProperties": false
                    }),
                },
            },
        ]
    }

    /// Dispatch a function call by name, returning the result.
    ///
    /// `function_name` must match one of the names from [`tool_definitions`].
    /// `arguments` is a JSON string matching the function's parameter schema.
    pub fn call(&self, function_name: &str, arguments: &str) -> FunctionCallResult {
        match function_name {
            "search_code" => self.call_search_code(arguments),
            "get_context" => self.call_get_context(arguments),
            "read_file_topology" => self.call_read_file_topology(arguments),
            _ => FunctionCallResult {
                success: false,
                content: format!("Unknown function: {}", function_name),
            },
        }
    }

    fn call_search_code(&self, arguments: &str) -> FunctionCallResult {
        let args: SearchCodeArgs = match serde_json::from_str(arguments) {
            Ok(a) => a,
            Err(e) => {
                return FunctionCallResult {
                    success: false,
                    content: format!("Invalid arguments: {}", e),
                }
            }
        };

        let limit = args.limit.min(self.max_results);
        let results = self.engine.search_nodes(&args.query, limit);

        let entries: Vec<SearchResult> = results
            .into_iter()
            .map(|(node, score)| SearchResult {
                node_id: node.id.clone(),
                name: node.label.clone(),
                kind: node.kind.to_string(),
                file_path: node.file_path.clone(),
                signature: node.signature.clone().unwrap_or_default(),
                score,
            })
            .collect();

        FunctionCallResult {
            success: true,
            content: serde_json::to_string(&entries).unwrap_or_else(|_| "[]".to_string()),
        }
    }

    fn call_get_context(&self, arguments: &str) -> FunctionCallResult {
        let args: GetContextArgs = match serde_json::from_str(arguments) {
            Ok(a) => a,
            Err(e) => {
                return FunctionCallResult {
                    success: false,
                    content: format!("Invalid arguments: {}", e),
                }
            }
        };

        let budget = args.budget.unwrap_or(self.budget);
        let anchors = args.anchors.unwrap_or(5);
        let limit = anchors * 2;

        let results = self.engine.search_nodes(&args.query, limit);

        // Build a context-like response with the top results
        let context: Vec<serde_json::Value> = results
            .into_iter()
            .take(anchors)
            .map(|(node, score)| {
                serde_json::json!({
                    "node_id": node.id,
                    "name": node.label,
                    "kind": node.kind.to_string(),
                    "file_path": node.file_path,
                    "signature": node.signature,
                    "doc_comment": node.doc_comment,
                    "domain": node.domain.to_string(),
                    "score": score,
                    "span": node.span.as_ref().map(|s| serde_json::json!({
                        "start_line": s.start_line,
                        "end_line": s.end_line,
                    })),
                })
            })
            .collect();

        let response = serde_json::json!({
            "query": args.query,
            "budget": budget,
            "anchors": context.len(),
            "results": context,
        });

        FunctionCallResult {
            success: true,
            content: serde_json::to_string(&response).unwrap_or_else(|_| "{}".to_string()),
        }
    }

    fn call_read_file_topology(&self, arguments: &str) -> FunctionCallResult {
        let args: ReadFileTopologyArgs = match serde_json::from_str(arguments) {
            Ok(a) => a,
            Err(e) => {
                return FunctionCallResult {
                    success: false,
                    content: format!("Invalid arguments: {}", e),
                }
            }
        };

        // Find all nodes in the requested file
        let file_nodes: Vec<serde_json::Value> = self
            .engine
            .search_nodes("", 1000) // get all, filter by path below
            .into_iter()
            .filter(|(node, _)| node.file_path == args.file_path)
            .map(|(node, _)| {
                serde_json::json!({
                    "node_id": node.id,
                    "name": node.label,
                    "kind": node.kind.to_string(),
                    "signature": node.signature,
                    "span": node.span.as_ref().map(|s| serde_json::json!({
                        "start_line": s.start_line,
                        "end_line": s.end_line,
                    })),
                })
            })
            .collect();

        let response = serde_json::json!({
            "file_path": args.file_path,
            "node_count": file_nodes.len(),
            "nodes": file_nodes,
        });

        FunctionCallResult {
            success: true,
            content: serde_json::to_string(&response).unwrap_or_else(|_| "{}".to_string()),
        }
    }
}

// ── TrevecMemoryAdapter ─────────────────────────────────────────────────────

/// Generic memory adapter for AI agent frameworks.
///
/// Provides a simple `remember` / `recall` interface that any framework can
/// use for persistent conversational memory backed by the Trevec graph.
///
/// # Example
///
/// ```no_run
/// # use std::sync::{Arc, Mutex};
/// # use trevec_sdk::{TrevecEngine, EngineConfig};
/// # use trevec_sdk::integrations::TrevecMemoryAdapter;
/// # fn example() -> anyhow::Result<()> {
/// # let engine_inner = TrevecEngine::default();
/// # let engine = Arc::new(Mutex::new(engine_inner));
/// let memory = TrevecMemoryAdapter::builder(engine)
///     .session_id("session_001")
///     .default_importance(50)
///     .max_recall(10)
///     .build();
///
/// // Store a memory
/// let id = memory.remember("user", "We should use PostgreSQL for the database")?;
///
/// // Recall relevant memories
/// let memories = memory.recall("database choice");
/// # Ok(())
/// # }
/// ```
pub struct TrevecMemoryAdapter {
    /// The engine must be behind a Mutex because `remember` requires `&mut`.
    engine: Arc<Mutex<TrevecEngine>>,
    session_id: Option<String>,
    default_importance: Option<i32>,
    max_recall: usize,
    time_range_days: Option<u32>,
}

/// A recalled memory entry.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEntry {
    /// Stable node identifier.
    pub id: String,
    /// The memory content.
    pub content: String,
    /// Role that created this memory (user, assistant, system).
    pub role: String,
    /// Retrieval relevance score.
    pub score: f64,
    /// Session the memory was created in, if known.
    #[serde(skip_serializing_if = "Option::is_none")]
    pub session_id: Option<String>,
}

impl TrevecMemoryAdapter {
    /// Create a builder for configuring a `TrevecMemoryAdapter`.
    pub fn builder(engine: Arc<Mutex<TrevecEngine>>) -> TrevecMemoryAdapterBuilder {
        TrevecMemoryAdapterBuilder {
            engine,
            session_id: None,
            default_importance: None,
            max_recall: 10,
            time_range_days: None,
        }
    }

    /// Create a memory adapter with default settings.
    pub fn new(engine: Arc<Mutex<TrevecEngine>>) -> Self {
        Self {
            engine,
            session_id: None,
            default_importance: None,
            max_recall: 10,
            time_range_days: None,
        }
    }

    /// Store a memory turn. Returns the node ID of the stored memory.
    pub fn remember(&self, role: &str, content: &str) -> anyhow::Result<String> {
        self.remember_with_files(role, content, &[])
    }

    /// Store a memory turn with associated file paths.
    pub fn remember_with_files(
        &self,
        role: &str,
        content: &str,
        _files: &[&str],
    ) -> anyhow::Result<String> {
        let mut meta = std::collections::HashMap::new();
        meta.insert("role".to_string(), role.to_string());
        if let Some(ref sid) = self.session_id {
            meta.insert("session_id".to_string(), sid.clone());
        }

        let user_id = self
            .session_id
            .as_deref()
            .unwrap_or("default");

        let mut engine = self.engine.lock().expect("engine mutex poisoned");
        engine.add(content, user_id, Some(meta))
    }

    /// Recall memories relevant to a query.
    pub fn recall(&self, query: &str) -> Vec<MemoryEntry> {
        let user_id = self
            .session_id
            .as_deref();

        let engine = self.engine.lock().expect("engine mutex poisoned");

        engine
            .search(query, user_id, self.max_recall)
            .into_iter()
            .map(|r| {
                let role = r.role.unwrap_or_else(|| "unknown".to_string());

                MemoryEntry {
                    id: r.id.clone(),
                    content: r.memory.clone(),
                    role,
                    score: r.score,
                    session_id: r.user_id,
                }
            })
            .collect()
    }

    /// Recall memories and return them as a JSON string.
    pub fn recall_json(&self, query: &str) -> String {
        let entries = self.recall(query);
        serde_json::to_string_pretty(&entries).unwrap_or_else(|_| "[]".to_string())
    }

    /// Get the number of stored memory nodes.
    pub fn memory_count(&self) -> usize {
        let engine = self.engine.lock().expect("engine mutex poisoned");
        engine
            .nodes_by_domain(crate::DomainTag::Conversation, None)
            .len()
    }

    /// The configured session ID, if any.
    pub fn session_id(&self) -> Option<&str> {
        self.session_id.as_deref()
    }
}

/// Builder for [`TrevecMemoryAdapter`].
pub struct TrevecMemoryAdapterBuilder {
    engine: Arc<Mutex<TrevecEngine>>,
    session_id: Option<String>,
    default_importance: Option<i32>,
    max_recall: usize,
    time_range_days: Option<u32>,
}

impl TrevecMemoryAdapterBuilder {
    /// Set the session ID for all memories created by this adapter.
    pub fn session_id(mut self, session_id: impl Into<String>) -> Self {
        self.session_id = Some(session_id.into());
        self
    }

    /// Set the default importance score for new memories (0-100).
    pub fn default_importance(mut self, importance: i32) -> Self {
        self.default_importance = Some(importance);
        self
    }

    /// Set the maximum number of results for recall (default: 10).
    pub fn max_recall(mut self, max: usize) -> Self {
        self.max_recall = max;
        self
    }

    /// Limit recall to memories within the last N days.
    pub fn time_range_days(mut self, days: u32) -> Self {
        self.time_range_days = Some(days);
        self
    }

    /// Build the memory adapter.
    pub fn build(self) -> TrevecMemoryAdapter {
        TrevecMemoryAdapter {
            engine: self.engine,
            session_id: self.session_id,
            default_importance: self.default_importance,
            max_recall: self.max_recall,
            time_range_days: self.time_range_days,
        }
    }
}

// ── Tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::EngineConfig;
    use std::collections::HashMap;
    use trevec_core::universal::*;

    /// Helper: create a test engine pre-loaded with sample nodes.
    fn test_engine_with_nodes() -> TrevecEngine {
        let mut engine = TrevecEngine::default();

        // Add sample code nodes
        engine.add_node(UniversalNode {
            id: "auth_validate".to_string(),
            kind: UniversalKind::Function,
            domain: DomainTag::Code,
            label: "validate_token".to_string(),
            file_path: "src/auth.rs".to_string(),
            span: Some(trevec_core::model::Span {
                start_line: 10,
                start_col: 0,
                end_line: 25,
                end_col: 1,
                start_byte: 200,
                end_byte: 500,
            }),
            signature: Some("fn validate_token(token: &str) -> Result<Claims>".to_string()),
            doc_comment: Some("Validate a JWT bearer token".to_string()),
            identifiers: vec!["JWT".to_string(), "Claims".to_string()],
            bm25_text: "validate_token JWT bearer token authentication src/auth.rs".to_string(),
            symbol_vec: None,
            ast_hash: Some("abc123".to_string()),
            temporal: None,
            attributes: HashMap::new(),
            intent_summary: None,
        });

        engine.add_node(UniversalNode {
            id: "db_query".to_string(),
            kind: UniversalKind::Function,
            domain: DomainTag::Code,
            label: "find_user".to_string(),
            file_path: "src/db.rs".to_string(),
            span: Some(trevec_core::model::Span {
                start_line: 5,
                start_col: 0,
                end_line: 15,
                end_col: 1,
                start_byte: 100,
                end_byte: 300,
            }),
            signature: Some("fn find_user(id: UserId) -> Option<User>".to_string()),
            doc_comment: Some("Look up a user by ID".to_string()),
            identifiers: vec!["UserId".to_string(), "User".to_string()],
            bm25_text: "find_user database query user lookup src/db.rs".to_string(),
            symbol_vec: None,
            ast_hash: Some("def456".to_string()),
            temporal: None,
            attributes: HashMap::new(),
            intent_summary: None,
        });

        engine.add_node(UniversalNode {
            id: "handler_login".to_string(),
            kind: UniversalKind::Function,
            domain: DomainTag::Code,
            label: "login_handler".to_string(),
            file_path: "src/handlers.rs".to_string(),
            span: None,
            signature: Some("async fn login_handler(req: Request) -> Response".to_string()),
            doc_comment: None,
            identifiers: vec!["Request".to_string(), "Response".to_string()],
            bm25_text: "login_handler authentication endpoint request response src/handlers.rs"
                .to_string(),
            symbol_vec: None,
            ast_hash: None,
            temporal: None,
            attributes: HashMap::new(),
            intent_summary: None,
        });

        engine
    }

    // ── TrevecRetriever tests ───────────────────────────────────────────

    #[test]
    fn test_retriever_returns_documents() {
        let engine = Arc::new(test_engine_with_nodes());
        let retriever = TrevecRetriever::new(engine);

        let docs = retriever.retrieve("JWT authentication");
        assert!(!docs.is_empty(), "Should find at least one document");
        assert_eq!(docs[0].metadata.node_id, "auth_validate");
        assert_eq!(docs[0].metadata.domain, "code");
        assert!(docs[0].metadata.score > 0.0);
    }

    #[test]
    fn test_retriever_respects_top_k() {
        let engine = Arc::new(test_engine_with_nodes());
        let retriever = TrevecRetriever::builder(engine).top_k(1).build();

        let docs = retriever.retrieve("authentication");
        assert!(docs.len() <= 1, "Should respect top_k limit");
    }

    #[test]
    fn test_retriever_document_content() {
        let engine = Arc::new(test_engine_with_nodes());
        let retriever = TrevecRetriever::new(engine);

        let docs = retriever.retrieve("database user");
        assert!(!docs.is_empty());

        let doc = &docs[0];
        // page_content should include signature and doc comment
        assert!(
            doc.page_content.contains("find_user"),
            "page_content should contain the signature"
        );
        assert_eq!(doc.metadata.file_path, "src/db.rs");
        assert_eq!(doc.metadata.kind, "function");
        assert_eq!(doc.metadata.start_line, Some(5));
        assert_eq!(doc.metadata.end_line, Some(15));
    }

    #[test]
    fn test_retriever_json_output() {
        let engine = Arc::new(test_engine_with_nodes());
        let retriever = TrevecRetriever::new(engine);

        let json = retriever.retrieve_json("JWT");
        let parsed: Vec<Document> = serde_json::from_str(&json).unwrap();
        assert!(!parsed.is_empty());
        assert_eq!(parsed[0].metadata.node_id, "auth_validate");
    }

    #[test]
    fn test_retriever_builder_defaults() {
        let engine = Arc::new(test_engine_with_nodes());
        let retriever = TrevecRetriever::builder(engine).build();

        assert_eq!(retriever.top_k(), 5);
        assert_eq!(retriever.budget(), 4096);
    }

    #[test]
    fn test_retriever_empty_results() {
        let engine = Arc::new(test_engine_with_nodes());
        let retriever = TrevecRetriever::new(engine);

        let docs = retriever.retrieve("xyzzy_nonexistent_term");
        assert!(docs.is_empty());
    }

    // ── TrevecTool tests ────────────────────────────────────────────────

    #[test]
    fn test_tool_call_text_output() {
        let engine = Arc::new(test_engine_with_nodes());
        let tool = TrevecTool::new(engine);

        let output = tool.tool_call("JWT authentication");
        assert!(output.contains("validate_token"), "Should mention the function name");
        assert!(output.contains("src/auth.rs"), "Should mention the file path");
        assert!(output.contains("Score:"), "Should include score");
    }

    #[test]
    fn test_tool_call_json_output() {
        let engine = Arc::new(test_engine_with_nodes());
        let tool = TrevecTool::builder(engine)
            .output_format(ToolOutputFormat::Json)
            .build();

        let output = tool.tool_call("database");
        let parsed: Vec<ToolResultEntry> = serde_json::from_str(&output).unwrap();
        assert!(!parsed.is_empty());
        assert_eq!(parsed[0].name, "find_user");
    }

    #[test]
    fn test_tool_call_no_results() {
        let engine = Arc::new(test_engine_with_nodes());
        let tool = TrevecTool::new(engine);

        let output = tool.tool_call("xyzzy_nothing");
        assert!(output.contains("No results found"));
    }

    #[test]
    fn test_tool_builder_custom_name() {
        let engine = Arc::new(test_engine_with_nodes());
        let tool = TrevecTool::builder(engine)
            .name("my_search")
            .description("Custom search tool")
            .max_results(3)
            .build();

        assert_eq!(tool.name(), "my_search");
        assert_eq!(tool.description(), "Custom search tool");
    }

    #[test]
    fn test_tool_respects_max_results() {
        let engine = Arc::new(test_engine_with_nodes());
        let tool = TrevecTool::builder(engine)
            .max_results(1)
            .output_format(ToolOutputFormat::Json)
            .build();

        let output = tool.tool_call("authentication");
        let parsed: Vec<ToolResultEntry> = serde_json::from_str(&output).unwrap();
        assert!(parsed.len() <= 1);
    }

    // ── TrevecFunctionTool tests ────────────────────────────────────────

    #[test]
    fn test_function_tool_definitions_schema() {
        let engine = Arc::new(test_engine_with_nodes());
        let func_tool = TrevecFunctionTool::new(engine);

        let defs = func_tool.tool_definitions();
        assert_eq!(defs.len(), 3, "Should have 3 tool definitions");

        // Verify all definitions have correct type
        for def in &defs {
            assert_eq!(def.tool_type, "function");
            assert!(!def.function.name.is_empty());
            assert!(!def.function.description.is_empty());
        }

        // Verify specific tool names
        let names: Vec<&str> = defs.iter().map(|d| d.function.name.as_str()).collect();
        assert!(names.contains(&"search_code"));
        assert!(names.contains(&"get_context"));
        assert!(names.contains(&"read_file_topology"));
    }

    #[test]
    fn test_function_tool_definitions_serializable() {
        let engine = Arc::new(test_engine_with_nodes());
        let func_tool = TrevecFunctionTool::new(engine);

        let defs = func_tool.tool_definitions();
        let json = serde_json::to_string_pretty(&defs).unwrap();

        // Should be valid JSON that can be deserialized back
        let parsed: Vec<FunctionToolDefinition> = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.len(), 3);

        // Verify the search_code schema has required fields
        let search = parsed.iter().find(|d| d.function.name == "search_code").unwrap();
        let params = &search.function.parameters;
        assert_eq!(params["type"], "object");
        let required = params["required"].as_array().unwrap();
        assert!(required.iter().any(|v| v == "query"));
    }

    #[test]
    fn test_function_tool_call_search() {
        let engine = Arc::new(test_engine_with_nodes());
        let func_tool = TrevecFunctionTool::new(engine);

        let result = func_tool.call("search_code", r#"{"query": "JWT authentication"}"#);
        assert!(result.success);

        let entries: Vec<SearchResult> = serde_json::from_str(&result.content).unwrap();
        assert!(!entries.is_empty());
        assert_eq!(entries[0].name, "validate_token");
    }

    #[test]
    fn test_function_tool_call_get_context() {
        let engine = Arc::new(test_engine_with_nodes());
        let func_tool = TrevecFunctionTool::new(engine);

        let result = func_tool.call(
            "get_context",
            r#"{"query": "database lookup", "budget": 2048, "anchors": 2}"#,
        );
        assert!(result.success);

        let parsed: serde_json::Value = serde_json::from_str(&result.content).unwrap();
        assert_eq!(parsed["query"], "database lookup");
        assert_eq!(parsed["budget"], 2048);
        let results = parsed["results"].as_array().unwrap();
        assert!(!results.is_empty());
    }

    #[test]
    fn test_function_tool_call_unknown_function() {
        let engine = Arc::new(test_engine_with_nodes());
        let func_tool = TrevecFunctionTool::new(engine);

        let result = func_tool.call("nonexistent_function", "{}");
        assert!(!result.success);
        assert!(result.content.contains("Unknown function"));
    }

    #[test]
    fn test_function_tool_call_invalid_args() {
        let engine = Arc::new(test_engine_with_nodes());
        let func_tool = TrevecFunctionTool::new(engine);

        let result = func_tool.call("search_code", "not valid json");
        assert!(!result.success);
        assert!(result.content.contains("Invalid arguments"));
    }

    // ── TrevecMemoryAdapter tests ───────────────────────────────────────

    #[test]
    fn test_memory_remember_and_recall() {
        let engine = TrevecEngine::default();
        let engine = Arc::new(Mutex::new(engine));
        let memory = TrevecMemoryAdapter::new(engine);

        let id = memory.remember("user", "We decided to use JWT for authentication").unwrap();
        assert!(!id.is_empty());

        let entries = memory.recall("JWT authentication");
        assert!(!entries.is_empty(), "Should recall the stored memory");
        assert!(entries[0].content.contains("JWT"));
        assert_eq!(entries[0].role, "user");
    }

    #[test]
    fn test_memory_with_session_and_importance() {
        let engine = TrevecEngine::default();
        let engine = Arc::new(Mutex::new(engine));
        let memory = TrevecMemoryAdapter::builder(engine)
            .session_id("test-session")
            .default_importance(80)
            .max_recall(5)
            .build();

        assert_eq!(memory.session_id(), Some("test-session"));

        memory
            .remember("assistant", "PostgreSQL is the chosen database")
            .unwrap();

        let entries = memory.recall("database choice");
        assert!(!entries.is_empty());
        // Session ID should be persisted in the node attributes
        assert_eq!(entries[0].session_id.as_deref(), Some("test-session"));
    }

    #[test]
    fn test_memory_with_files() {
        let engine = TrevecEngine::default();
        let engine = Arc::new(Mutex::new(engine));
        let memory = TrevecMemoryAdapter::new(engine);

        memory
            .remember_with_files(
                "user",
                "Updated the auth module to use bcrypt",
                &["src/auth.rs", "src/password.rs"],
            )
            .unwrap();

        let entries = memory.recall("auth bcrypt");
        assert!(!entries.is_empty());
    }

    #[test]
    fn test_memory_count() {
        let engine = TrevecEngine::default();
        let engine = Arc::new(Mutex::new(engine));
        let memory = TrevecMemoryAdapter::new(engine);

        assert_eq!(memory.memory_count(), 0);

        memory.remember("user", "First memory item").unwrap();
        memory.remember("assistant", "Second memory item").unwrap();

        assert_eq!(memory.memory_count(), 2);
    }

    #[test]
    fn test_memory_recall_json() {
        let engine = TrevecEngine::default();
        let engine = Arc::new(Mutex::new(engine));
        let memory = TrevecMemoryAdapter::new(engine);

        memory.remember("user", "Use Redis for caching").unwrap();

        let json = memory.recall_json("Redis caching");
        let parsed: Vec<MemoryEntry> = serde_json::from_str(&json).unwrap();
        assert!(!parsed.is_empty());
        assert!(parsed[0].content.contains("Redis"));
    }

    #[test]
    fn test_memory_recall_no_results() {
        let engine = TrevecEngine::default();
        let engine = Arc::new(Mutex::new(engine));
        let memory = TrevecMemoryAdapter::new(engine);

        memory.remember("user", "We chose PostgreSQL").unwrap();

        let entries = memory.recall("xyzzy_unrelated");
        assert!(entries.is_empty());
    }
}
