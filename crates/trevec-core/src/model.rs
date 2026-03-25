use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

/// Stable node identifier: blake3(file_path|kind|signature|start_byte), 32 hex chars.
pub type NodeId = String;

/// The kind of code structure a node represents.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum NodeKind {
    Function,
    Method,
    Class,
    Interface,
    Module,
    Struct,
    Enum,
    Trait,
    Macro,
    Type,
    DocSection,
}

impl fmt::Display for NodeKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            NodeKind::Function => write!(f, "function"),
            NodeKind::Method => write!(f, "method"),
            NodeKind::Class => write!(f, "class"),
            NodeKind::Interface => write!(f, "interface"),
            NodeKind::Module => write!(f, "module"),
            NodeKind::Struct => write!(f, "struct"),
            NodeKind::Enum => write!(f, "enum"),
            NodeKind::Trait => write!(f, "trait"),
            NodeKind::Macro => write!(f, "macro"),
            NodeKind::Type => write!(f, "type"),
            NodeKind::DocSection => write!(f, "doc_section"),
        }
    }
}

impl NodeKind {
    /// Map a tags.scm tag suffix to a NodeKind.
    pub fn from_tag_suffix(suffix: &str) -> Option<Self> {
        match suffix {
            "function" => Some(NodeKind::Function),
            "method" => Some(NodeKind::Method),
            "class" => Some(NodeKind::Class),
            "interface" => Some(NodeKind::Interface),
            "module" => Some(NodeKind::Module),
            "struct" => Some(NodeKind::Struct),
            "enum" => Some(NodeKind::Enum),
            "trait" => Some(NodeKind::Trait),
            "macro" => Some(NodeKind::Macro),
            "type" => Some(NodeKind::Type),
            _ => None,
        }
    }
}

/// The type of relationship between two nodes.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum EdgeType {
    Import,
    Call,
    Contain,
    Implement,
    Inherit,
    /// Memory event → code node: "discussed in this session"
    Discussed,
    /// Memory event → code node: "session caused change to this code"
    Triggered,
    /// Doc section → code node: "doc references this symbol"
    Reference,
}

impl fmt::Display for EdgeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            EdgeType::Import => write!(f, "import"),
            EdgeType::Call => write!(f, "call"),
            EdgeType::Contain => write!(f, "contain"),
            EdgeType::Implement => write!(f, "implement"),
            EdgeType::Inherit => write!(f, "inherit"),
            EdgeType::Discussed => write!(f, "discussed"),
            EdgeType::Triggered => write!(f, "triggered"),
            EdgeType::Reference => write!(f, "reference"),
        }
    }
}

/// Confidence level for an edge. Ordered: Certain > Likely > Unknown.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, PartialOrd, Ord, Serialize, Deserialize)]
pub enum Confidence {
    Unknown = 0,
    Likely = 1,
    Certain = 2,
}

impl fmt::Display for Confidence {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Confidence::Unknown => write!(f, "unknown"),
            Confidence::Likely => write!(f, "likely"),
            Confidence::Certain => write!(f, "certain"),
        }
    }
}

/// A source location span within a file.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Span {
    pub start_line: usize,
    pub start_col: usize,
    pub end_line: usize,
    pub end_col: usize,
    pub start_byte: usize,
    pub end_byte: usize,
}

impl Span {
    pub fn byte_length(&self) -> usize {
        self.end_byte.saturating_sub(self.start_byte)
    }

    /// Estimate tokens as byte_length / 4.
    pub fn estimated_tokens(&self) -> usize {
        self.byte_length() / 4
    }

    /// Check if this span fully contains another span.
    pub fn contains(&self, other: &Span) -> bool {
        self.start_byte <= other.start_byte && self.end_byte >= other.end_byte
    }
}

/// A code structure node extracted from the AST.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CodeNode {
    pub id: NodeId,
    pub kind: NodeKind,
    pub file_path: String,
    pub span: Span,
    pub name: String,
    pub signature: String,
    pub doc_comment: Option<String>,
    pub identifiers: Vec<String>,
    pub bm25_text: String,
    pub symbol_vec: Option<Vec<f32>>,
    pub ast_hash: String,
}

impl CodeNode {
    /// Build the BM25 text field from code-derived signals.
    ///
    /// Build the BM25 text field from code-derived signals.
    pub fn build_bm25_text(
        file_path: &str,
        _name: &str,
        signature: &str,
        identifiers: &[String],
        doc_comment: Option<&str>,
    ) -> String {
        let mut parts = vec![file_path.to_string()];
        parts.push(signature.to_string());
        if !identifiers.is_empty() {
            parts.push(identifiers.join(" "));
        }
        if let Some(doc) = doc_comment {
            parts.push(doc.to_string());
        }
        parts.join(" ")
    }

    /// Build the text used for embedding: signature + identifiers + doc_comment.
    pub fn embedding_text(&self) -> String {
        let mut parts = vec![self.signature.clone()];
        if !self.identifiers.is_empty() {
            parts.push(self.identifiers.join(" "));
        }
        if let Some(ref doc) = self.doc_comment {
            parts.push(doc.clone());
        }
        parts.join(" ")
    }
}

/// Entry in the file manifest for incremental indexing.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileManifestEntry {
    pub file_hash: String,
    pub node_ids: Vec<NodeId>,
}

/// Manifest tracking file hashes and their node IDs for incremental indexing.
pub type FileManifest = HashMap<String, FileManifestEntry>;

/// A directed edge between two code nodes.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Edge {
    pub src_id: NodeId,
    pub dst_id: NodeId,
    pub edge_type: EdgeType,
    pub confidence: Confidence,
}

/// A node included in a context bundle with its source text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct IncludedNode {
    pub node_id: NodeId,
    pub file_path: String,
    pub span: Span,
    pub kind: NodeKind,
    pub name: String,
    pub signature: String,
    pub source_text: String,
    pub is_anchor: bool,
    pub estimated_tokens: usize,
}

/// The assembled context bundle returned to the caller.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ContextBundle {
    pub bundle_id: String,
    pub query: String,
    pub anchor_node_ids: Vec<NodeId>,
    pub included_nodes: Vec<IncludedNode>,
    pub total_estimated_tokens: usize,
    /// Total tokens if the agent read every source file in full (baseline for savings calc).
    #[serde(default)]
    pub total_source_file_tokens: usize,
    /// Time taken for retrieval in milliseconds.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub retrieval_ms: Option<u64>,
}

impl ContextBundle {
    /// Build the one-liner summary: "Retrieved N functions from M files in Xms · saved ~Y tokens (Z%)"
    fn summary_line(&self) -> String {
        let unique_files: std::collections::HashSet<&str> =
            self.included_nodes.iter().map(|n| n.file_path.as_str()).collect();
        let files_count = unique_files.len();
        let functions_count = self.included_nodes.len();

        let timing = match self.retrieval_ms {
            Some(ms) => format!(" in {}ms", ms),
            None => String::new(),
        };

        let savings = if self.total_source_file_tokens > 0 {
            let saved = self
                .total_source_file_tokens
                .saturating_sub(self.total_estimated_tokens);
            let pct = (saved as f64 / self.total_source_file_tokens as f64) * 100.0;
            format!(" · saved ~{} tokens ({:.0}%)", format_tokens(saved), pct)
        } else {
            String::new()
        };

        format!(
            "Retrieved {} functions from {} files{}{}",
            functions_count, files_count, timing, savings
        )
    }

    /// Format the bundle as human-readable text with file grouping and citations.
    pub fn format_text(&self) -> String {
        let mut output = String::new();
        let summary = self.summary_line();

        output.push_str(&format!("Query: {}\n", self.query));
        output.push_str(&summary);
        output.push('\n');
        output.push_str(&"─".repeat(60));
        output.push('\n');

        // Group by file
        let mut by_file: std::collections::BTreeMap<&str, Vec<&IncludedNode>> =
            std::collections::BTreeMap::new();
        for node in &self.included_nodes {
            by_file
                .entry(&node.file_path)
                .or_default()
                .push(node);
        }

        for (file_path, mut nodes) in by_file {
            nodes.sort_by_key(|n| n.span.start_line);
            output.push_str(&format!("\n## {}\n\n", file_path));
            for node in nodes {
                output.push_str(&format!(
                    "### {} `{}` (L{}-L{})\n\n",
                    node.kind,
                    node.name,
                    node.span.start_line + 1,
                    node.span.end_line + 1,
                ));
                output.push_str("```\n");
                output.push_str(&node.source_text);
                if !node.source_text.ends_with('\n') {
                    output.push('\n');
                }
                output.push_str("```\n\n");
            }
        }

        // Footer one-liner
        output.push_str("---\n");
        output.push_str(&summary);
        output.push('\n');

        output
    }
}

/// Format a token count with comma separators.
fn format_tokens(n: usize) -> String {
    if n < 1_000 {
        n.to_string()
    } else if n < 1_000_000 {
        format!("{},{:03}", n / 1_000, n % 1_000)
    } else {
        let millions = n / 1_000_000;
        let thousands = (n % 1_000_000) / 1_000;
        let remainder = n % 1_000;
        format!("{},{:03},{:03}", millions, thousands, remainder)
    }
}

/// Cumulative stats for get_context queries, persisted per-project.
#[derive(Debug, Clone, Default, Serialize, Deserialize)]
pub struct QueryStats {
    pub total_queries: u64,
    pub total_tokens_returned: u64,
    pub total_source_file_tokens: u64,
    /// Unix timestamp of first recorded query.
    #[serde(default)]
    pub first_query_at: Option<i64>,
    /// Total number of reindex operations.
    #[serde(default)]
    pub total_reindexes: u64,
    /// Duration of the last reindex in milliseconds.
    #[serde(default)]
    pub last_reindex_ms: u64,
    /// Number of files parsed in the last reindex.
    #[serde(default)]
    pub last_reindex_files: u64,
}

impl QueryStats {
    pub fn tokens_saved(&self) -> u64 {
        self.total_source_file_tokens
            .saturating_sub(self.total_tokens_returned)
    }

    pub fn savings_percentage(&self) -> f64 {
        if self.total_source_file_tokens == 0 {
            return 0.0;
        }
        (self.tokens_saved() as f64 / self.total_source_file_tokens as f64) * 100.0
    }

    pub fn record_query(&mut self, tokens_returned: usize, source_file_tokens: usize) {
        self.total_queries += 1;
        self.total_tokens_returned += tokens_returned as u64;
        self.total_source_file_tokens += source_file_tokens as u64;
        if self.first_query_at.is_none() {
            self.first_query_at = Some(
                std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs() as i64,
            );
        }
    }

    pub fn record_reindex(&mut self, files_parsed: usize, total_ms: u128) {
        self.total_reindexes += 1;
        self.last_reindex_ms = total_ms as u64;
        self.last_reindex_files = files_parsed as u64;
    }

    pub fn merge(&mut self, other: &QueryStats) {
        self.total_queries += other.total_queries;
        self.total_tokens_returned += other.total_tokens_returned;
        self.total_source_file_tokens += other.total_source_file_tokens;
        self.total_reindexes += other.total_reindexes;
        match (self.first_query_at, other.first_query_at) {
            (Some(a), Some(b)) => self.first_query_at = Some(a.min(b)),
            (None, Some(b)) => self.first_query_at = Some(b),
            _ => {}
        }
    }

    pub fn load(path: &std::path::Path) -> Self {
        match std::fs::read_to_string(path) {
            Ok(content) => serde_json::from_str(&content).unwrap_or_default(),
            Err(_) => Self::default(),
        }
    }

    pub fn save(&self, path: &std::path::Path) -> anyhow::Result<()> {
        let json = serde_json::to_string_pretty(self)?;
        std::fs::write(path, json)?;
        Ok(())
    }
}

/// A persistent record of an AI chat turn or tool call.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MemoryEvent {
    /// Deterministic ID: blake3(source + session_id + turn_index)
    pub id: String,
    /// blake3 hash of the repo path
    pub repo_id: String,
    /// Source: "cursor", "claude_code", "codex", "manual", "trevec_tool"
    pub source: String,
    /// Session identifier from the source
    pub session_id: String,
    /// Turn index within the session
    pub turn_index: u32,
    /// Role: "user", "assistant", "system", "tool"
    pub role: String,
    /// Event type: "turn", "tool_call", "decision", "summary"
    pub event_type: String,
    /// Scrubbed text content
    pub content_redacted: String,
    /// blake3 hash of original content (for dedupe)
    pub content_hash: String,
    /// Unix epoch seconds
    pub created_at: i64,
    /// Importance score 0-100
    pub importance: i32,
    /// Whether this event is pinned (exempt from GC)
    pub pinned: bool,
    /// File paths touched in this turn
    pub files_touched: Vec<String>,
    /// Tool calls made in this turn
    pub tool_calls: Vec<String>,
    /// Text for BM25 search: source + files + tool_calls + content keywords
    pub bm25_text: String,
    /// Embedding vector (384-dim, same model as code nodes)
    pub symbol_vec: Option<Vec<f32>>,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_span_estimated_tokens() {
        let span = Span {
            start_line: 0,
            start_col: 0,
            end_line: 10,
            end_col: 0,
            start_byte: 0,
            end_byte: 400,
        };
        assert_eq!(span.estimated_tokens(), 100);
    }

    #[test]
    fn test_span_contains() {
        let outer = Span {
            start_line: 0,
            start_col: 0,
            end_line: 50,
            end_col: 0,
            start_byte: 0,
            end_byte: 1000,
        };
        let inner = Span {
            start_line: 5,
            start_col: 0,
            end_line: 10,
            end_col: 0,
            start_byte: 100,
            end_byte: 300,
        };
        assert!(outer.contains(&inner));
        assert!(!inner.contains(&outer));
    }

    #[test]
    fn test_build_bm25_text() {
        let text = CodeNode::build_bm25_text(
            "src/main.rs",
            "main",
            "fn main()",
            &["println".to_string(), "args".to_string()],
            Some("Entry point"),
        );
        assert!(text.contains("src/main.rs"));
        assert!(text.contains("fn main()"));
        assert!(text.contains("println args"));
        assert!(text.contains("Entry point"));
        // Name appears in: signature x1, file_path x1
        assert_eq!(text.matches("main").count(), 2);
    }

    #[test]
    fn test_confidence_ordering() {
        assert!(Confidence::Certain > Confidence::Likely);
        assert!(Confidence::Likely > Confidence::Unknown);
    }

    #[test]
    fn test_node_kind_from_tag_suffix() {
        assert_eq!(
            NodeKind::from_tag_suffix("function"),
            Some(NodeKind::Function)
        );
        assert_eq!(NodeKind::from_tag_suffix("class"), Some(NodeKind::Class));
        assert_eq!(NodeKind::from_tag_suffix("unknown_tag"), None);
    }

    // ── QueryStats tests ──────────────────────────────────────────────

    #[test]
    fn test_query_stats_default() {
        let stats = QueryStats::default();
        assert_eq!(stats.total_queries, 0);
        assert_eq!(stats.total_tokens_returned, 0);
        assert_eq!(stats.total_source_file_tokens, 0);
        assert_eq!(stats.first_query_at, None);
    }

    #[test]
    fn test_query_stats_record_query() {
        let mut stats = QueryStats::default();
        stats.record_query(500, 5000);
        assert_eq!(stats.total_queries, 1);
        assert_eq!(stats.total_tokens_returned, 500);
        assert_eq!(stats.total_source_file_tokens, 5000);
        assert!(stats.first_query_at.is_some());

        let first_ts = stats.first_query_at.unwrap();
        stats.record_query(300, 3000);
        assert_eq!(stats.total_queries, 2);
        assert_eq!(stats.total_tokens_returned, 800);
        assert_eq!(stats.total_source_file_tokens, 8000);
        // first_query_at should not change on subsequent queries
        assert_eq!(stats.first_query_at, Some(first_ts));
    }

    #[test]
    fn test_query_stats_tokens_saved() {
        let mut stats = QueryStats::default();
        stats.record_query(500, 5000);
        assert_eq!(stats.tokens_saved(), 4500);
    }

    #[test]
    fn test_query_stats_tokens_saved_saturates() {
        // Edge case: tokens_returned > source_file_tokens should not underflow
        let stats = QueryStats {
            total_queries: 1,
            total_tokens_returned: 10000,
            total_source_file_tokens: 5000,
            first_query_at: None,
            ..Default::default()
        };
        assert_eq!(stats.tokens_saved(), 0);
    }

    #[test]
    fn test_query_stats_savings_percentage() {
        let mut stats = QueryStats::default();
        stats.record_query(1000, 10000);
        let pct = stats.savings_percentage();
        assert!((pct - 90.0).abs() < 0.01);
    }

    #[test]
    fn test_query_stats_savings_percentage_zero_source() {
        let stats = QueryStats::default();
        assert_eq!(stats.savings_percentage(), 0.0);
    }


    #[test]
    fn test_query_stats_merge() {
        let mut a = QueryStats::default();
        a.record_query(500, 5000);

        let mut b = QueryStats::default();
        b.record_query(300, 3000);

        // Manually set b's first_query_at earlier
        b.first_query_at = Some(100);
        a.first_query_at = Some(200);

        a.merge(&b);
        assert_eq!(a.total_queries, 2);
        assert_eq!(a.total_tokens_returned, 800);
        assert_eq!(a.total_source_file_tokens, 8000);
        assert_eq!(a.first_query_at, Some(100)); // picks earlier timestamp
    }

    #[test]
    fn test_query_stats_merge_into_empty() {
        let mut a = QueryStats::default();
        let mut b = QueryStats::default();
        b.record_query(500, 5000);

        a.merge(&b);
        assert_eq!(a.total_queries, 1);
        assert_eq!(a.first_query_at, b.first_query_at);
    }

    #[test]
    fn test_query_stats_save_load_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("stats.json");

        let mut stats = QueryStats::default();
        stats.record_query(500, 5000);
        stats.record_query(300, 3000);
        stats.save(&path).unwrap();

        let loaded = QueryStats::load(&path);
        assert_eq!(loaded.total_queries, 2);
        assert_eq!(loaded.total_tokens_returned, 800);
        assert_eq!(loaded.total_source_file_tokens, 8000);
        assert_eq!(loaded.first_query_at, stats.first_query_at);
    }

    #[test]
    fn test_query_stats_load_missing_file() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("nonexistent.json");
        let stats = QueryStats::load(&path);
        assert_eq!(stats.total_queries, 0);
    }

    #[test]
    fn test_query_stats_load_malformed_json() {
        let tmp = tempfile::tempdir().unwrap();
        let path = tmp.path().join("stats.json");
        std::fs::write(&path, "not valid json {{{}").unwrap();
        let stats = QueryStats::load(&path);
        assert_eq!(stats.total_queries, 0);
    }

    // ── ContextBundle tests ───────────────────────────────────────────

    #[test]
    fn test_context_bundle_format_text() {
        let bundle = ContextBundle {
            bundle_id: "test123".to_string(),
            query: "how does auth work".to_string(),
            anchor_node_ids: vec!["node_a".to_string()],
            included_nodes: vec![
                IncludedNode {
                    node_id: "node_a".to_string(),
                    file_path: "src/auth.rs".to_string(),
                    span: Span {
                        start_line: 0,
                        start_col: 0,
                        end_line: 5,
                        end_col: 0,
                        start_byte: 0,
                        end_byte: 100,
                    },
                    kind: NodeKind::Function,
                    name: "authenticate".to_string(),
                    signature: "fn authenticate()".to_string(),
                    source_text: "fn authenticate() {\n  true\n}".to_string(),
                    is_anchor: true,
                    estimated_tokens: 25,
                },
            ],
            total_estimated_tokens: 25,
            total_source_file_tokens: 500,
            retrieval_ms: Some(12),
        };

        let text = bundle.format_text();
        assert!(text.contains("how does auth work"));
        assert!(text.contains("src/auth.rs"));
        assert!(text.contains("authenticate"));
        // Human-friendly summary line
        assert!(text.contains("Retrieved 1 functions from 1 files in 12ms"));
        assert!(text.contains("saved ~475 tokens (95%)"));
        // No [anchor] markers
        assert!(!text.contains("[anchor]"));
        // Footer one-liner
        assert!(text.contains("---\n"));
    }

    #[test]
    fn test_context_bundle_total_source_file_tokens_serialization() {
        let bundle = ContextBundle {
            bundle_id: "test".to_string(),
            query: "q".to_string(),
            anchor_node_ids: vec![],
            included_nodes: vec![],
            total_estimated_tokens: 100,
            total_source_file_tokens: 5000,
            retrieval_ms: None,
        };

        let json = serde_json::to_string(&bundle).unwrap();
        assert!(json.contains("\"total_source_file_tokens\":5000"));

        let parsed: ContextBundle = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed.total_source_file_tokens, 5000);
    }

    #[test]
    fn test_context_bundle_total_source_file_tokens_defaults_to_zero() {
        // Old bundles without the field should deserialize with default 0
        let json = r#"{
            "bundle_id": "test",
            "query": "q",
            "anchor_node_ids": [],
            "included_nodes": [],
            "total_estimated_tokens": 100
        }"#;
        let bundle: ContextBundle = serde_json::from_str(json).unwrap();
        assert_eq!(bundle.total_source_file_tokens, 0);
    }
}
