//! Worker 3: Link Predictor
//!
//! Predicts missing edges in the code graph using co-occurrence patterns.
//! Lightweight approach (no GNN needed for v0.1): if two nodes are frequently
//! retrieved together, predict an edge between them.

use std::collections::HashMap;
use trevec_core::model::NodeId;
use trevec_core::Confidence;

/// Co-occurrence counter for node pairs.
#[derive(Debug, Default)]
pub struct CoOccurrenceTracker {
    /// Maps (node_a, node_b) → co-occurrence count (sorted pair for dedup).
    counts: HashMap<(String, String), u32>,
    /// Minimum co-occurrence count to predict an edge.
    threshold: u32,
}

impl CoOccurrenceTracker {
    pub fn new(threshold: u32) -> Self {
        Self {
            counts: HashMap::new(),
            threshold,
        }
    }

    /// Record that these nodes appeared together in a retrieval result.
    pub fn record(&mut self, node_ids: &[NodeId]) {
        for i in 0..node_ids.len() {
            for j in (i + 1)..node_ids.len() {
                let pair = sorted_pair(&node_ids[i], &node_ids[j]);
                *self.counts.entry(pair).or_insert(0) += 1;
            }
        }
    }

    /// Get predicted edges above the threshold.
    pub fn predicted_edges(&self) -> Vec<PredictedEdge> {
        self.counts
            .iter()
            .filter(|(_, count)| **count >= self.threshold)
            .map(|((a, b), count)| PredictedEdge {
                src_id: a.clone(),
                dst_id: b.clone(),
                co_occurrence_count: *count,
                confidence: if *count >= self.threshold * 3 {
                    Confidence::Likely
                } else {
                    Confidence::Unknown
                },
            })
            .collect()
    }

    /// Number of tracked pairs.
    pub fn pair_count(&self) -> usize {
        self.counts.len()
    }

    /// Get co-occurrence count for a specific pair.
    pub fn get_count(&self, a: &str, b: &str) -> u32 {
        let pair = sorted_pair(a, b);
        self.counts.get(&pair).copied().unwrap_or(0)
    }

    /// Clear all tracked data.
    pub fn clear(&mut self) {
        self.counts.clear();
    }
}

/// A predicted edge from co-occurrence analysis.
#[derive(Debug, Clone)]
pub struct PredictedEdge {
    pub src_id: String,
    pub dst_id: String,
    pub co_occurrence_count: u32,
    pub confidence: Confidence,
}

/// Sort a pair of strings for consistent hashing.
fn sorted_pair(a: &str, b: &str) -> (String, String) {
    if a <= b {
        (a.to_string(), b.to_string())
    } else {
        (b.to_string(), a.to_string())
    }
}

/// Predict test-to-implementation links based on naming conventions.
pub fn predict_test_impl_links(
    test_files: &[String],
    impl_files: &[String],
) -> Vec<PredictedEdge> {
    let mut predictions = Vec::new();

    for test_file in test_files {
        let test_name = extract_base_name(test_file);
        if test_name.is_empty() {
            continue;
        }

        // Strip test prefix/suffix to get the implementation name
        let impl_name = test_name
            .strip_prefix("test_")
            .or_else(|| test_name.strip_suffix("_test"))
            .or_else(|| test_name.strip_suffix(".test"))
            .or_else(|| test_name.strip_suffix("_spec"))
            .or_else(|| test_name.strip_prefix("test"))
            .unwrap_or(&test_name);

        for impl_file in impl_files {
            let impl_base = extract_base_name(impl_file);
            if impl_base == impl_name
                || impl_base.contains(impl_name)
                || impl_name.contains(&impl_base)
            {
                predictions.push(PredictedEdge {
                    src_id: test_file.clone(),
                    dst_id: impl_file.clone(),
                    co_occurrence_count: 0,
                    confidence: Confidence::Likely,
                });
            }
        }
    }

    predictions
}

/// Extract the base name from a file path (without extension or directory).
fn extract_base_name(path: &str) -> String {
    let file_name = path
        .rsplit('/')
        .next()
        .or_else(|| path.rsplit('\\').next())
        .unwrap_or(path);

    // Remove extension
    file_name
        .rsplit_once('.')
        .map(|(name, _)| name)
        .unwrap_or(file_name)
        .to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_co_occurrence_basic() {
        let mut tracker = CoOccurrenceTracker::new(2);

        tracker.record(&["a".to_string(), "b".to_string(), "c".to_string()]);
        tracker.record(&["a".to_string(), "b".to_string()]);

        assert_eq!(tracker.get_count("a", "b"), 2);
        assert_eq!(tracker.get_count("a", "c"), 1);
        assert_eq!(tracker.get_count("b", "c"), 1);

        let predictions = tracker.predicted_edges();
        assert_eq!(predictions.len(), 1); // only (a,b) meets threshold=2
        assert_eq!(predictions[0].src_id, "a");
        assert_eq!(predictions[0].dst_id, "b");
    }

    #[test]
    fn test_co_occurrence_symmetry() {
        let mut tracker = CoOccurrenceTracker::new(1);
        tracker.record(&["x".to_string(), "y".to_string()]);

        assert_eq!(tracker.get_count("x", "y"), 1);
        assert_eq!(tracker.get_count("y", "x"), 1); // symmetric
    }

    #[test]
    fn test_co_occurrence_confidence() {
        let mut tracker = CoOccurrenceTracker::new(2);
        for _ in 0..10 {
            tracker.record(&["a".to_string(), "b".to_string()]);
        }

        let predictions = tracker.predicted_edges();
        assert!(!predictions.is_empty());
        assert_eq!(predictions[0].confidence, Confidence::Likely); // count=10 >> threshold*3=6
    }

    #[test]
    fn test_co_occurrence_empty() {
        let tracker = CoOccurrenceTracker::new(2);
        assert!(tracker.predicted_edges().is_empty());
        assert_eq!(tracker.pair_count(), 0);
    }

    #[test]
    fn test_predict_test_impl_links_python() {
        let test_files = vec!["tests/test_auth.py".to_string()];
        let impl_files = vec!["src/auth.py".to_string(), "src/db.py".to_string()];

        let links = predict_test_impl_links(&test_files, &impl_files);
        assert!(!links.is_empty());
        assert!(links.iter().any(|l| l.dst_id == "src/auth.py"));
        assert!(!links.iter().any(|l| l.dst_id == "src/db.py"));
    }

    #[test]
    fn test_predict_test_impl_links_rust() {
        let test_files = vec!["tests/auth_test.rs".to_string()];
        let impl_files = vec!["src/auth.rs".to_string()];

        let links = predict_test_impl_links(&test_files, &impl_files);
        assert!(!links.is_empty());
    }

    #[test]
    fn test_predict_test_impl_links_js() {
        let test_files = vec!["__tests__/auth.test.js".to_string()];
        let impl_files = vec!["src/auth.js".to_string()];

        let links = predict_test_impl_links(&test_files, &impl_files);
        assert!(!links.is_empty());
    }

    #[test]
    fn test_extract_base_name() {
        assert_eq!(extract_base_name("src/auth.rs"), "auth");
        assert_eq!(extract_base_name("tests/test_auth.py"), "test_auth");
        assert_eq!(extract_base_name("auth"), "auth");
    }

    #[test]
    fn test_co_occurrence_clear() {
        let mut tracker = CoOccurrenceTracker::new(1);
        tracker.record(&["a".to_string(), "b".to_string()]);
        assert_eq!(tracker.pair_count(), 1);
        tracker.clear();
        assert_eq!(tracker.pair_count(), 0);
    }
}
