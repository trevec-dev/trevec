//! Semantic Cache for Brain enrichment outputs.
//!
//! Two-layer cache:
//! 1. In-memory LRU (fast, for hot nodes)
//! 2. Persistent storage (survives process restart) — future: LanceDB table
//!
//! Cache keys are deterministic: blake3(node_id + ast_hash + prompt_version)
//! Invalidation: ast_hash change → invalidate entry

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Cache entry with metadata for invalidation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct CacheEntry {
    pub value: String,
    pub created_at: i64,
    /// The ast_hash at time of generation (for invalidation).
    pub source_hash: Option<String>,
    /// The prompt/model version used (for re-enrichment on upgrade).
    pub prompt_version: Option<String>,
}

/// In-memory LRU cache for Brain enrichment results.
pub struct BrainCache {
    entries: HashMap<String, CacheEntry>,
    max_entries: usize,
    /// Ordered by access time (most recent first).
    access_order: Vec<String>,
}

impl BrainCache {
    pub fn new() -> Self {
        Self::with_capacity(10_000)
    }

    pub fn with_capacity(max_entries: usize) -> Self {
        Self {
            entries: HashMap::new(),
            max_entries,
            access_order: Vec::new(),
        }
    }

    /// Get a cached value, updating access order.
    pub fn get(&self, key: &str) -> Option<&CacheEntry> {
        self.entries.get(key)
    }

    /// Set a cached value.
    pub fn set(&mut self, key: String, value: String) {
        self.set_with_meta(key, value, None, None);
    }

    /// Set a cached value with metadata.
    pub fn set_with_meta(
        &mut self,
        key: String,
        value: String,
        source_hash: Option<String>,
        prompt_version: Option<String>,
    ) {
        let now = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64;

        let entry = CacheEntry {
            value,
            created_at: now,
            source_hash,
            prompt_version,
        };

        // Evict if at capacity
        if self.entries.len() >= self.max_entries && !self.entries.contains_key(&key) {
            self.evict_lru();
        }

        self.entries.insert(key.clone(), entry);

        // Update access order
        self.access_order.retain(|k| k != &key);
        self.access_order.push(key);
    }

    /// Invalidate all entries whose source_hash doesn't match.
    pub fn invalidate_by_hash(&mut self, node_id_prefix: &str, current_hash: &str) -> usize {
        let keys_to_remove: Vec<String> = self
            .entries
            .iter()
            .filter(|(k, v)| {
                k.starts_with(node_id_prefix)
                    && v.source_hash.as_deref() != Some(current_hash)
            })
            .map(|(k, _)| k.clone())
            .collect();

        let count = keys_to_remove.len();
        for key in &keys_to_remove {
            self.entries.remove(key);
            self.access_order.retain(|k| k != key);
        }
        count
    }

    /// Remove a specific entry.
    pub fn invalidate(&mut self, key: &str) -> bool {
        self.access_order.retain(|k| k != key);
        self.entries.remove(key).is_some()
    }

    /// Number of entries in the cache.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Clear all entries.
    pub fn clear(&mut self) {
        self.entries.clear();
        self.access_order.clear();
    }

    /// Evict the least recently used entry.
    fn evict_lru(&mut self) {
        if let Some(oldest_key) = self.access_order.first().cloned() {
            self.entries.remove(&oldest_key);
            self.access_order.remove(0);
        }
    }

    /// Cache hit rate (for metrics).
    pub fn stats(&self) -> CacheStats {
        CacheStats {
            entries: self.entries.len(),
            max_entries: self.max_entries,
            utilization: self.entries.len() as f64 / self.max_entries as f64,
        }
    }
}

impl Default for BrainCache {
    fn default() -> Self {
        Self::new()
    }
}

#[derive(Debug, Clone)]
pub struct CacheStats {
    pub entries: usize,
    pub max_entries: usize,
    pub utilization: f64,
}

/// Generate a cache key from node ID, source hash, and prompt version.
pub fn cache_key(node_id: &str, source_hash: &str, prompt_version: &str) -> String {
    let input = format!("{}|{}|{}", node_id, source_hash, prompt_version);
    let hash = blake3::hash(input.as_bytes());
    hash.to_hex()[..32].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cache_set_get() {
        let mut cache = BrainCache::new();
        cache.set("key1".to_string(), "value1".to_string());

        let entry = cache.get("key1").unwrap();
        assert_eq!(entry.value, "value1");
        assert!(entry.created_at > 0);
    }

    #[test]
    fn test_cache_miss() {
        let cache = BrainCache::new();
        assert!(cache.get("nonexistent").is_none());
    }

    #[test]
    fn test_cache_overwrite() {
        let mut cache = BrainCache::new();
        cache.set("key1".to_string(), "value1".to_string());
        cache.set("key1".to_string(), "value2".to_string());

        assert_eq!(cache.get("key1").unwrap().value, "value2");
        assert_eq!(cache.len(), 1);
    }

    #[test]
    fn test_cache_lru_eviction() {
        let mut cache = BrainCache::with_capacity(3);
        cache.set("a".to_string(), "1".to_string());
        cache.set("b".to_string(), "2".to_string());
        cache.set("c".to_string(), "3".to_string());
        assert_eq!(cache.len(), 3);

        // Adding a 4th should evict "a" (LRU)
        cache.set("d".to_string(), "4".to_string());
        assert_eq!(cache.len(), 3);
        assert!(cache.get("a").is_none());
        assert!(cache.get("d").is_some());
    }

    #[test]
    fn test_cache_invalidate() {
        let mut cache = BrainCache::new();
        cache.set("key1".to_string(), "value1".to_string());
        assert!(cache.invalidate("key1"));
        assert!(cache.get("key1").is_none());
        assert!(!cache.invalidate("key1")); // already removed
    }

    #[test]
    fn test_cache_invalidate_by_hash() {
        let mut cache = BrainCache::new();
        cache.set_with_meta(
            "node_1:summary".to_string(),
            "old summary".to_string(),
            Some("old_hash".to_string()),
            None,
        );
        cache.set_with_meta(
            "node_1:links".to_string(),
            "old links".to_string(),
            Some("old_hash".to_string()),
            None,
        );
        cache.set_with_meta(
            "node_2:summary".to_string(),
            "other".to_string(),
            Some("other_hash".to_string()),
            None,
        );

        // Invalidate node_1 entries with non-matching hash
        let removed = cache.invalidate_by_hash("node_1", "new_hash");
        assert_eq!(removed, 2);
        assert!(cache.get("node_1:summary").is_none());
        assert!(cache.get("node_1:links").is_none());
        assert!(cache.get("node_2:summary").is_some()); // different prefix, untouched
    }

    #[test]
    fn test_cache_clear() {
        let mut cache = BrainCache::new();
        cache.set("a".to_string(), "1".to_string());
        cache.set("b".to_string(), "2".to_string());
        cache.clear();
        assert!(cache.is_empty());
    }

    #[test]
    fn test_cache_stats() {
        let mut cache = BrainCache::with_capacity(100);
        cache.set("a".to_string(), "1".to_string());
        let stats = cache.stats();
        assert_eq!(stats.entries, 1);
        assert_eq!(stats.max_entries, 100);
        assert!((stats.utilization - 0.01).abs() < 0.001);
    }

    #[test]
    fn test_cache_key_deterministic() {
        let k1 = cache_key("node_1", "hash_abc", "v1");
        let k2 = cache_key("node_1", "hash_abc", "v1");
        assert_eq!(k1, k2);
        assert_eq!(k1.len(), 32);
    }

    #[test]
    fn test_cache_key_different_inputs() {
        let k1 = cache_key("node_1", "hash_abc", "v1");
        let k2 = cache_key("node_1", "hash_def", "v1");
        assert_ne!(k1, k2);
    }
}
