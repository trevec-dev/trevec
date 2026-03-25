//! Ebbinghaus-inspired retention scoring for Brain-enriched data.
//!
//! Prevents graph bloat by decaying stale enrichments.
//! Based on research from CortexGraph and YourMemory systems.

use trevec_core::universal::RetentionScore;

/// Default half-lives for different enrichment types (in days).
pub const DEFAULT_INTENT_HALF_LIFE: f64 = 30.0;
pub const DEFAULT_OBSERVATION_HALF_LIFE: f64 = 14.0;
pub const DEFAULT_REFLECTION_HALF_LIFE: f64 = 30.0;
pub const DEFAULT_CROSS_DOMAIN_HALF_LIFE: f64 = 60.0;
pub const DEFAULT_PREDICTION_HALF_LIFE: f64 = 7.0;

/// Default thresholds.
pub const DEFAULT_PRUNE_THRESHOLD: f64 = 0.15;
pub const DEFAULT_ARCHIVE_THRESHOLD: f64 = 0.05;
pub const DEFAULT_REFRESH_LOW: f64 = 0.15;
pub const DEFAULT_REFRESH_HIGH: f64 = 0.65;
pub const DEFAULT_PROMOTE_ACCESS_COUNT: u32 = 10;
pub const DEFAULT_PROMOTE_WINDOW_DAYS: i64 = 14;

/// Retention scoring configuration.
#[derive(Debug, Clone)]
pub struct RetentionConfig {
    pub prune_threshold: f64,
    pub archive_threshold: f64,
    pub refresh_low: f64,
    pub refresh_high: f64,
    pub promote_access_count: u32,
    pub promote_window_days: i64,
}

impl Default for RetentionConfig {
    fn default() -> Self {
        Self {
            prune_threshold: DEFAULT_PRUNE_THRESHOLD,
            archive_threshold: DEFAULT_ARCHIVE_THRESHOLD,
            refresh_low: DEFAULT_REFRESH_LOW,
            refresh_high: DEFAULT_REFRESH_HIGH,
            promote_access_count: DEFAULT_PROMOTE_ACCESS_COUNT,
            promote_window_days: DEFAULT_PROMOTE_WINDOW_DAYS,
        }
    }
}

/// Action to take based on retention score.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum RetentionAction {
    /// Score is healthy, keep as-is.
    Keep,
    /// Score is in refresh zone, re-run Brain worker.
    Refresh,
    /// Score is below prune threshold, remove enrichment.
    Prune,
    /// Score is below archive threshold, move to cold storage.
    Archive,
    /// Node is permanent (frequently accessed), never decay.
    Permanent,
}

/// Compute the retention action for a score.
pub fn retention_action(score: &RetentionScore, config: &RetentionConfig) -> RetentionAction {
    if score.permanent {
        return RetentionAction::Permanent;
    }
    if score.score < config.archive_threshold {
        RetentionAction::Archive
    } else if score.score < config.prune_threshold {
        RetentionAction::Prune
    } else if score.score < config.refresh_high {
        RetentionAction::Refresh
    } else {
        RetentionAction::Keep
    }
}

/// Determine if a node should be promoted to permanent based on access patterns.
pub fn should_promote(
    access_count: u32,
    last_access: i64,
    now: i64,
    config: &RetentionConfig,
) -> bool {
    if access_count >= config.promote_access_count {
        let window_secs = config.promote_window_days * 86400;
        (now - last_access) < window_secs
    } else {
        false
    }
}

/// Batch compute retention scores for a set of nodes.
pub fn batch_score(
    nodes: &[(String, f64, u32, i64)], // (node_id, importance, access_count, last_access)
    now: i64,
    half_life_days: f64,
) -> Vec<(String, RetentionScore)> {
    nodes
        .iter()
        .map(|(id, importance, access_count, last_access)| {
            let score =
                RetentionScore::compute(*importance, *access_count, *last_access, now, half_life_days);
            (id.clone(), score)
        })
        .collect()
}

/// Statistics about retention scoring results.
#[derive(Debug, Default)]
pub struct RetentionStats {
    pub total: usize,
    pub keep: usize,
    pub refresh: usize,
    pub prune: usize,
    pub archive: usize,
    pub permanent: usize,
}

/// Batch analyze retention scores.
pub fn analyze_retention(
    scores: &[(String, RetentionScore)],
    config: &RetentionConfig,
) -> RetentionStats {
    let mut stats = RetentionStats {
        total: scores.len(),
        ..Default::default()
    };

    for (_, score) in scores {
        match retention_action(score, config) {
            RetentionAction::Keep => stats.keep += 1,
            RetentionAction::Refresh => stats.refresh += 1,
            RetentionAction::Prune => stats.prune += 1,
            RetentionAction::Archive => stats.archive += 1,
            RetentionAction::Permanent => stats.permanent += 1,
        }
    }

    stats
}

#[cfg(test)]
mod tests {
    use super::*;

    fn now() -> i64 {
        1_000_000
    }

    #[test]
    fn test_healthy_node() {
        let score = RetentionScore::compute(
            1.0, // high importance
            10,  // frequently accessed
            now() - 86400, // 1 day ago
            now(),
            30.0, // 30-day half-life
        );
        let config = RetentionConfig::default();
        assert_eq!(retention_action(&score, &config), RetentionAction::Keep);
    }

    #[test]
    fn test_stale_node_prune() {
        let score = RetentionScore::compute(
            0.3,                // low importance
            0,                  // never accessed
            now() - 86400 * 90, // 90 days ago
            now(),
            30.0,
        );
        let config = RetentionConfig::default();
        let action = retention_action(&score, &config);
        assert!(
            action == RetentionAction::Prune || action == RetentionAction::Archive,
            "Expected prune or archive, got {:?}",
            action
        );
    }

    #[test]
    fn test_permanent_node() {
        let mut score = RetentionScore::compute(0.1, 0, 0, now(), 3.0);
        score.permanent = true;
        let config = RetentionConfig::default();
        assert_eq!(retention_action(&score, &config), RetentionAction::Permanent);
    }

    #[test]
    fn test_should_promote() {
        let config = RetentionConfig::default();
        // 15 accesses, last access was recently
        assert!(should_promote(15, now() - 86400, now(), &config));
        // 5 accesses (below threshold)
        assert!(!should_promote(5, now() - 86400, now(), &config));
        // 15 accesses but last access was 30 days ago
        assert!(!should_promote(15, now() - 86400 * 30, now(), &config));
    }

    #[test]
    fn test_batch_score() {
        let nodes = vec![
            ("a".to_string(), 1.0, 5, now() - 86400),
            ("b".to_string(), 0.5, 0, now() - 86400 * 60),
            ("c".to_string(), 0.8, 20, now() - 3600),
        ];

        let scores = batch_score(&nodes, now(), 30.0);
        assert_eq!(scores.len(), 3);

        // Node c (recently accessed, high importance) should have highest score
        assert!(scores[2].1.score > scores[0].1.score);
        assert!(scores[2].1.score > scores[1].1.score);
    }

    #[test]
    fn test_analyze_retention() {
        let config = RetentionConfig::default();
        let scores = vec![
            ("healthy".to_string(), RetentionScore {
                score: 1.5,
                access_count: 10,
                last_access: now(),
                permanent: false,
            }),
            ("refresh".to_string(), RetentionScore {
                score: 0.4,
                access_count: 2,
                last_access: now() - 86400 * 10,
                permanent: false,
            }),
            ("prune".to_string(), RetentionScore {
                score: 0.10,
                access_count: 0,
                last_access: now() - 86400 * 60,
                permanent: false,
            }),
            ("perm".to_string(), RetentionScore {
                score: 0.01,
                access_count: 0,
                last_access: 0,
                permanent: true,
            }),
        ];

        let stats = analyze_retention(&scores, &config);
        assert_eq!(stats.total, 4);
        assert_eq!(stats.keep, 1);
        assert_eq!(stats.refresh, 1);
        assert_eq!(stats.permanent, 1);
        // prune has score 0.10 which is < 0.15 (prune threshold) but > 0.05 (archive)
        assert_eq!(stats.prune, 1);
    }
}
