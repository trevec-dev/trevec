//! Brain integration helpers for the MCP server and CLI.
//!
//! This module provides the glue between the Brain engine and the existing
//! TrevecServer / indexing pipeline. It handles:
//! - Starting the Brain after indexing completes
//! - Pausing Brain during active queries (sleep-time compute)
//! - Feeding retrieval results back to the Brain for co-occurrence tracking
//! - Applying Brain enrichments to CodeNodes before retrieval

use std::collections::HashMap;
use std::sync::Arc;
use tokio::sync::RwLock;

use trevec_core::config::BrainConfig;
use trevec_core::model::CodeNode;
use trevec_core::universal::IntentSummary;

use crate::queue::{EnrichmentTask, Priority, TaskType};
use crate::workers::link_predictor::CoOccurrenceTracker;
use crate::BrainEngine;

/// Brain integration handle attached to the MCP server.
pub struct BrainHandle {
    engine: Arc<BrainEngine>,
    co_occurrence: Arc<RwLock<CoOccurrenceTracker>>,
    /// Nodes that have been enriched (node_id -> IntentSummary).
    enrichments: Arc<RwLock<HashMap<String, IntentSummary>>>,
}

impl BrainHandle {
    /// Create a new Brain handle from configuration.
    pub fn new(config: &BrainConfig) -> Option<Self> {
        if !config.enabled {
            return None;
        }
        let engine = Arc::new(BrainEngine::new(config.clone()));
        Some(Self {
            engine,
            co_occurrence: Arc::new(RwLock::new(CoOccurrenceTracker::new(3))),
            enrichments: Arc::new(RwLock::new(HashMap::new())),
        })
    }

    /// Called after indexing completes. Queues all nodes for enrichment.
    pub async fn on_index_complete(&self, nodes: &HashMap<String, CodeNode>) {
        self.engine.resume().await;

        // Queue public/exported symbols first (high priority)
        let mut high_priority = Vec::new();
        let mut low_priority = Vec::new();

        for (id, node) in nodes {
            if is_public_symbol(node) {
                high_priority.push(id.clone());
            } else {
                low_priority.push(id.clone());
            }
        }

        self.engine
            .enqueue_batch(high_priority, Priority::High)
            .await;
        self.engine
            .enqueue_batch(low_priority, Priority::Background)
            .await;

        tracing::info!(
            "Brain: queued {} nodes for enrichment ({} queue size)",
            nodes.len(),
            self.engine.queue_len()
        );
    }

    /// Called before a query executes. Pauses the Brain.
    pub async fn on_query_start(&self) {
        self.engine.pause().await;
    }

    /// Called after a query completes. Resumes the Brain and records co-occurrence.
    pub async fn on_query_complete(&self, result_node_ids: &[String]) {
        self.engine.resume().await;

        // Record co-occurrence for link prediction
        let mut tracker = self.co_occurrence.write().await;
        tracker.record(result_node_ids);

        // Boost priority for queried nodes (they're in the hot path)
        for id in result_node_ids {
            self.engine
                .enqueue_node(id.clone(), Priority::Critical)
                .await;
        }
    }

    /// Process pending Brain tasks. Call this periodically or during idle time.
    /// Returns the number of tasks processed.
    pub async fn tick(&self) -> usize {
        let mut processed = 0;
        // Process up to 10 tasks per tick to avoid blocking
        for _ in 0..10 {
            if !self.engine.process_one().await {
                break;
            }
            processed += 1;
        }
        processed
    }

    /// Get enriched bm25_text for a node (base + Brain intent summary).
    pub async fn enriched_bm25_text(&self, node: &CodeNode) -> String {
        let enrichments = self.enrichments.read().await;
        if let Some(summary) = enrichments.get(&node.id) {
            let extra = summary.to_bm25_text();
            if extra.is_empty() {
                node.bm25_text.clone()
            } else {
                format!("{} {}", node.bm25_text, extra)
            }
        } else {
            node.bm25_text.clone()
        }
    }

    /// Store an enrichment result for a node.
    pub async fn store_enrichment(&self, node_id: String, summary: IntentSummary) {
        let mut enrichments = self.enrichments.write().await;
        enrichments.insert(node_id, summary);
    }

    /// Get predicted edges from co-occurrence tracking.
    pub async fn predicted_edges(&self) -> Vec<crate::workers::link_predictor::PredictedEdge> {
        let tracker = self.co_occurrence.read().await;
        tracker.predicted_edges()
    }

    /// Get Brain statistics.
    pub async fn stats(&self) -> crate::BrainState {
        self.engine.stats().await
    }

    /// Get the number of stored enrichments.
    pub async fn enrichment_count(&self) -> usize {
        let enrichments = self.enrichments.read().await;
        enrichments.len()
    }

    /// Get queue length.
    pub fn queue_len(&self) -> usize {
        self.engine.queue_len()
    }
}

/// Heuristic: is this a public/exported symbol worth prioritizing?
fn is_public_symbol(node: &CodeNode) -> bool {
    let sig = &node.signature;
    // Rust: pub fn, pub struct, pub enum, pub trait
    if sig.starts_with("pub ") {
        return true;
    }
    // Python: no leading underscore (convention for public)
    if node.file_path.ends_with(".py") && !node.name.starts_with('_') {
        return true;
    }
    // JS/TS: export
    if sig.starts_with("export ") {
        return true;
    }
    // Go: uppercase first letter
    if node.file_path.ends_with(".go") {
        if let Some(first) = node.name.chars().next() {
            return first.is_uppercase();
        }
    }
    false
}

/// Queue changed files for observation (called from file watcher).
pub fn queue_observations(
    _engine: &BrainEngine,
    changed_files: &[String],
) -> Vec<EnrichmentTask> {
    changed_files
        .iter()
        .map(|file| EnrichmentTask {
            node_id: file.clone(),
            priority: Priority::High,
            task_type: TaskType::Observation,
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use trevec_core::model::{NodeKind, Span};

    fn make_node(id: &str, name: &str, sig: &str, file_path: &str) -> CodeNode {
        CodeNode {
            id: id.to_string(),
            kind: NodeKind::Function,
            file_path: file_path.to_string(),
            span: Span {
                start_line: 0,
                start_col: 0,
                end_line: 10,
                end_col: 0,
                start_byte: 0,
                end_byte: 200,
            },
            name: name.to_string(),
            signature: sig.to_string(),
            doc_comment: None,
            identifiers: vec![],
            bm25_text: format!("{} {} {}", file_path, sig, name),
            symbol_vec: None,
            ast_hash: "hash".to_string(),
        }
    }

    fn test_config() -> BrainConfig {
        BrainConfig {
            enabled: true,
            ..BrainConfig::default()
        }
    }

    #[test]
    fn test_is_public_symbol_rust() {
        let node = make_node("n1", "authenticate", "pub fn authenticate()", "src/auth.rs");
        assert!(is_public_symbol(&node));

        let node = make_node("n2", "helper", "fn helper()", "src/auth.rs");
        assert!(!is_public_symbol(&node));
    }

    #[test]
    fn test_is_public_symbol_python() {
        let node = make_node("n1", "authenticate", "def authenticate()", "auth.py");
        assert!(is_public_symbol(&node));

        let node = make_node("n2", "_helper", "def _helper()", "auth.py");
        assert!(!is_public_symbol(&node));
    }

    #[test]
    fn test_is_public_symbol_js() {
        let node =
            make_node("n1", "authenticate", "export function authenticate()", "auth.js");
        assert!(is_public_symbol(&node));
    }

    #[test]
    fn test_is_public_symbol_go() {
        let node = make_node("n1", "Authenticate", "func Authenticate()", "auth.go");
        assert!(is_public_symbol(&node));

        let node = make_node("n2", "helper", "func helper()", "auth.go");
        assert!(!is_public_symbol(&node));
    }

    #[tokio::test]
    async fn test_brain_handle_creation() {
        let handle = BrainHandle::new(&test_config());
        assert!(handle.is_some());

        let disabled = BrainHandle::new(&BrainConfig::default());
        assert!(disabled.is_none()); // default has enabled=false
    }

    #[tokio::test]
    async fn test_brain_handle_index_complete() {
        let handle = BrainHandle::new(&test_config()).unwrap();

        let mut nodes = HashMap::new();
        nodes.insert(
            "n1".to_string(),
            make_node("n1", "authenticate", "pub fn authenticate()", "src/auth.rs"),
        );
        nodes.insert(
            "n2".to_string(),
            make_node("n2", "helper", "fn helper()", "src/auth.rs"),
        );

        handle.on_index_complete(&nodes).await;
        assert!(handle.queue_len() > 0);
    }

    #[tokio::test]
    async fn test_brain_handle_query_lifecycle() {
        let handle = BrainHandle::new(&test_config()).unwrap();

        // Simulate query lifecycle
        handle.on_query_start().await;
        let stats = handle.stats().await;
        assert!(!stats.active); // paused during query

        handle
            .on_query_complete(&["node_a".to_string(), "node_b".to_string()])
            .await;
        let stats = handle.stats().await;
        assert!(stats.active); // resumed after query
    }

    #[tokio::test]
    async fn test_brain_handle_enrichment_storage() {
        let handle = BrainHandle::new(&test_config()).unwrap();

        let summary = IntentSummary {
            purpose: Some("Validates JWT tokens".to_string()),
            related_concepts: vec!["JWT".to_string(), "authentication".to_string()],
            ..Default::default()
        };

        handle
            .store_enrichment("node_1".to_string(), summary)
            .await;
        assert_eq!(handle.enrichment_count().await, 1);

        let node = make_node("node_1", "validate", "fn validate()", "src/auth.rs");
        let enriched = handle.enriched_bm25_text(&node).await;
        assert!(enriched.contains("Validates JWT tokens"));
        assert!(enriched.contains("JWT authentication"));
    }

    #[tokio::test]
    async fn test_brain_handle_unenriched_node() {
        let handle = BrainHandle::new(&test_config()).unwrap();

        let node = make_node("node_1", "validate", "fn validate()", "src/auth.rs");
        let text = handle.enriched_bm25_text(&node).await;
        // Should just be the base bm25_text
        assert_eq!(text, node.bm25_text);
    }

    #[tokio::test]
    async fn test_brain_handle_co_occurrence() {
        let handle = BrainHandle::new(&test_config()).unwrap();

        // Record co-occurrence 3 times (threshold is 3)
        for _ in 0..3 {
            handle
                .on_query_complete(&["a".to_string(), "b".to_string()])
                .await;
        }

        let predictions = handle.predicted_edges().await;
        assert!(!predictions.is_empty());
    }

    #[test]
    fn test_queue_observations() {
        let engine = BrainEngine::new(test_config());
        let tasks = queue_observations(&engine, &["src/auth.rs".to_string()]);
        assert_eq!(tasks.len(), 1);
        assert_eq!(tasks[0].task_type, TaskType::Observation);
        assert_eq!(tasks[0].priority, Priority::High);
    }
}
