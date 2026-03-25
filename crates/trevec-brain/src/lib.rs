//! # Trevec Brain: Progressive Intelligence Engine
//!
//! The Brain is the async LLM enrichment layer that makes the graph progressively
//! smarter over time without blocking the Reflex retrieval path.
//!
//! ## Architecture
//!
//! - **Reflex** (hot path): Deterministic parsing → retrieval in <50ms. Always works.
//! - **Brain** (cold path): Background LLM enrichment during idle time.
//! - **Cache**: All Brain outputs cached keyed on content hash. Never recompute.
//!
//! ## Workers
//!
//! 1. **Intent Summarizer**: Code → structured summary (purpose, inputs, outputs)
//! 2. **Entity Resolver**: Deduplicate entities (Jaro-Winkler + cosine + co-occurrence)
//! 3. **Link Predictor**: Predict missing edges (co-occurrence based)
//! 4. **Cross-Domain Linker**: Find cross-domain connections (embedding similarity)
//! 5. **Observation Agent**: Watch code changes, generate observations/reflections

#![warn(clippy::all)]

pub mod cache;
pub mod inference;
pub mod integration;
pub mod queue;
pub mod retention;
pub mod workers;

use std::sync::Arc;
use tokio::sync::RwLock;
use trevec_core::config::BrainConfig;

/// The Brain engine that manages enrichment workers.
pub struct BrainEngine {
    config: BrainConfig,
    cache: Arc<RwLock<cache::BrainCache>>,
    queue: Arc<queue::EnrichmentQueue>,
    state: Arc<RwLock<BrainState>>,
}

#[derive(Debug, Default)]
pub struct BrainState {
    /// Total nodes enriched in this session.
    pub nodes_enriched: u64,
    /// Total cache hits (avoided LLM calls).
    pub cache_hits: u64,
    /// Total LLM calls made.
    pub llm_calls: u64,
    /// Estimated cost this session ($).
    pub estimated_cost: f64,
    /// Whether the Brain is currently active (not paused).
    pub active: bool,
    /// Unix timestamp of last activity.
    pub last_activity: i64,
}

impl BrainEngine {
    /// Create a new Brain engine with the given configuration.
    pub fn new(config: BrainConfig) -> Self {
        Self {
            config: config.clone(),
            cache: Arc::new(RwLock::new(cache::BrainCache::new())),
            queue: Arc::new(queue::EnrichmentQueue::new()),
            state: Arc::new(RwLock::new(BrainState::default())),
        }
    }

    /// Whether the Brain is enabled.
    pub fn is_enabled(&self) -> bool {
        self.config.enabled
    }

    /// Get current Brain statistics.
    pub async fn stats(&self) -> BrainState {
        let state = self.state.read().await;
        BrainState {
            nodes_enriched: state.nodes_enriched,
            cache_hits: state.cache_hits,
            llm_calls: state.llm_calls,
            estimated_cost: state.estimated_cost,
            active: state.active,
            last_activity: state.last_activity,
        }
    }

    /// Pause the Brain (e.g., when user is actively querying).
    pub async fn pause(&self) {
        let mut state = self.state.write().await;
        state.active = false;
        tracing::debug!("Brain paused");
    }

    /// Resume the Brain (e.g., when user is idle).
    pub async fn resume(&self) {
        let mut state = self.state.write().await;
        state.active = true;
        state.last_activity = now_epoch();
        tracing::debug!("Brain resumed");
    }

    /// Queue a node for enrichment.
    pub async fn enqueue_node(&self, node_id: String, priority: queue::Priority) {
        self.queue.push(queue::EnrichmentTask {
            node_id,
            priority,
            task_type: queue::TaskType::IntentSummary,
        });
    }

    /// Queue multiple nodes for enrichment.
    pub async fn enqueue_batch(&self, node_ids: Vec<String>, priority: queue::Priority) {
        for id in node_ids {
            self.enqueue_node(id, priority).await;
        }
    }

    /// Process one enrichment task from the queue.
    /// Returns true if a task was processed, false if queue is empty.
    pub async fn process_one(&self) -> bool {
        if !self.is_enabled() {
            return false;
        }

        let state = self.state.read().await;
        if !state.active {
            return false;
        }
        drop(state);

        let task = match self.queue.pop() {
            Some(t) => t,
            None => return false,
        };

        // Check cache first
        let cache_key = format!("{}:{:?}", task.node_id, task.task_type);
        {
            let cache = self.cache.read().await;
            if cache.get(&cache_key).is_some() {
                let mut state = self.state.write().await;
                state.cache_hits += 1;
                return true;
            }
        }

        // Process the task (this is where LLM calls would happen)
        let result = match task.task_type {
            queue::TaskType::IntentSummary => {
                // In a real implementation, this would call the LLM
                // For now, just record that we attempted it
                tracing::debug!("Would enrich node {} with intent summary", task.node_id);
                Some("pending_enrichment".to_string())
            }
            queue::TaskType::EntityResolution => {
                tracing::debug!("Would resolve entity {}", task.node_id);
                Some("pending_resolution".to_string())
            }
            queue::TaskType::LinkPrediction => {
                tracing::debug!("Would predict links for {}", task.node_id);
                Some("pending_prediction".to_string())
            }
            queue::TaskType::Observation => {
                tracing::debug!("Would observe changes for {}", task.node_id);
                Some("pending_observation".to_string())
            }
        };

        if let Some(value) = result {
            let mut cache = self.cache.write().await;
            cache.set(cache_key, value);

            let mut state = self.state.write().await;
            state.nodes_enriched += 1;
            state.last_activity = now_epoch();
        }

        true
    }

    /// Get the enrichment queue length.
    pub fn queue_len(&self) -> usize {
        self.queue.len()
    }

    /// Get a reference to the cache for external inspection.
    pub fn cache(&self) -> &Arc<RwLock<cache::BrainCache>> {
        &self.cache
    }

    /// Get a reference to the config.
    pub fn config(&self) -> &BrainConfig {
        &self.config
    }
}

fn now_epoch() -> i64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn test_config() -> BrainConfig {
        BrainConfig {
            enabled: true,
            ..BrainConfig::default()
        }
    }

    #[tokio::test]
    async fn test_brain_engine_creation() {
        let engine = BrainEngine::new(test_config());
        assert!(engine.is_enabled());
        assert_eq!(engine.queue_len(), 0);
    }

    #[tokio::test]
    async fn test_brain_enqueue_and_process() {
        let engine = BrainEngine::new(test_config());
        engine.resume().await;

        engine
            .enqueue_node("node_1".to_string(), queue::Priority::High)
            .await;
        engine
            .enqueue_node("node_2".to_string(), queue::Priority::Low)
            .await;

        assert_eq!(engine.queue_len(), 2);

        // Process high priority first
        assert!(engine.process_one().await);
        assert_eq!(engine.queue_len(), 1);

        assert!(engine.process_one().await);
        assert_eq!(engine.queue_len(), 0);

        // Empty queue returns false
        assert!(!engine.process_one().await);
    }

    #[tokio::test]
    async fn test_brain_pause_resume() {
        let engine = BrainEngine::new(test_config());
        engine
            .enqueue_node("node_1".to_string(), queue::Priority::High)
            .await;

        // Not resumed yet, should not process
        assert!(!engine.process_one().await);

        engine.resume().await;
        assert!(engine.process_one().await);
    }

    #[tokio::test]
    async fn test_brain_cache_hit() {
        let engine = BrainEngine::new(test_config());
        engine.resume().await;

        engine
            .enqueue_node("node_1".to_string(), queue::Priority::High)
            .await;
        engine.process_one().await;

        // Enqueue same node again
        engine
            .enqueue_node("node_1".to_string(), queue::Priority::High)
            .await;
        engine.process_one().await;

        let stats = engine.stats().await;
        assert_eq!(stats.cache_hits, 1);
        assert_eq!(stats.nodes_enriched, 1);
    }

    #[tokio::test]
    async fn test_brain_stats() {
        let engine = BrainEngine::new(test_config());
        engine.resume().await;

        engine
            .enqueue_batch(
                vec![
                    "a".to_string(),
                    "b".to_string(),
                    "c".to_string(),
                ],
                queue::Priority::Medium,
            )
            .await;

        while engine.process_one().await {}

        let stats = engine.stats().await;
        assert_eq!(stats.nodes_enriched, 3);
        assert!(stats.active);
    }

    #[tokio::test]
    async fn test_brain_disabled() {
        let mut config = test_config();
        config.enabled = false;
        let engine = BrainEngine::new(config);
        engine.resume().await;

        engine
            .enqueue_node("node_1".to_string(), queue::Priority::High)
            .await;
        assert!(!engine.process_one().await); // Disabled, should not process
    }
}
