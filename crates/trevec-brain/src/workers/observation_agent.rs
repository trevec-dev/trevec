//! Worker 5: Observation Agent
//!
//! Watches code changes and builds a running observation log.
//! Inspired by Mastra's Observational Memory (94.87% LongMemEval-s).

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use trevec_core::universal::*;

/// A structured observation about a code change.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Observation {
    /// When the observation was made (unix epoch).
    pub timestamp: i64,
    /// File that changed.
    pub file_path: String,
    /// Type of change.
    pub change_type: ChangeType,
    /// Human-readable description of the change.
    pub description: String,
    /// Impact assessment.
    pub impact: Option<String>,
    /// Priority level.
    pub priority: ObservationPriority,
    /// Related code symbols.
    pub related_symbols: Vec<String>,
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq)]
pub enum ChangeType {
    Added,
    Modified,
    Deleted,
    Renamed,
}

impl std::fmt::Display for ChangeType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Added => write!(f, "added"),
            Self::Modified => write!(f, "modified"),
            Self::Deleted => write!(f, "deleted"),
            Self::Renamed => write!(f, "renamed"),
        }
    }
}

#[derive(Debug, Clone, Copy, Serialize, Deserialize, PartialEq, Eq, PartialOrd, Ord)]
pub enum ObservationPriority {
    Low,
    Medium,
    High,
}

/// A higher-level reflection consolidating multiple observations.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ReflectionEntry {
    /// When the reflection was generated.
    pub timestamp: i64,
    /// Summary of the observations.
    pub summary: String,
    /// Pattern detected (e.g., "feature_addition", "refactoring", "bug_fix").
    pub pattern: Option<String>,
    /// Scope of the changes.
    pub scope: String,
    /// Files involved.
    pub related_files: Vec<String>,
    /// IDs of observations this reflects on.
    pub observation_ids: Vec<String>,
}

/// Detect changes between old and new sets of symbols in a file.
pub fn detect_changes(
    file_path: &str,
    old_symbols: &[String],
    new_symbols: &[String],
    timestamp: i64,
) -> Vec<Observation> {
    let mut observations = Vec::new();
    let old_set: std::collections::HashSet<&str> =
        old_symbols.iter().map(|s| s.as_str()).collect();
    let new_set: std::collections::HashSet<&str> =
        new_symbols.iter().map(|s| s.as_str()).collect();

    // New symbols (added)
    for sym in new_set.difference(&old_set) {
        observations.push(Observation {
            timestamp,
            file_path: file_path.to_string(),
            change_type: ChangeType::Added,
            description: format!("New symbol `{}` added to {}", sym, file_path),
            impact: None,
            priority: ObservationPriority::Medium,
            related_symbols: vec![sym.to_string()],
        });
    }

    // Removed symbols
    for sym in old_set.difference(&new_set) {
        observations.push(Observation {
            timestamp,
            file_path: file_path.to_string(),
            change_type: ChangeType::Deleted,
            description: format!("Symbol `{}` removed from {}", sym, file_path),
            impact: Some("Callers of this symbol may be affected".to_string()),
            priority: ObservationPriority::High,
            related_symbols: vec![sym.to_string()],
        });
    }

    // Modified (symbols that exist in both but might have changed)
    if !old_symbols.is_empty()
        && !new_symbols.is_empty()
        && old_symbols != new_symbols
        && observations.is_empty()
    {
        observations.push(Observation {
            timestamp,
            file_path: file_path.to_string(),
            change_type: ChangeType::Modified,
            description: format!("{} was modified", file_path),
            impact: None,
            priority: ObservationPriority::Low,
            related_symbols: new_symbols.to_vec(),
        });
    }

    observations
}

/// Convert an observation to a UniversalNode for graph storage.
pub fn observation_to_node(obs: &Observation, id_prefix: &str) -> UniversalNode {
    let id = {
        let input = format!(
            "{}:{}:{}:{}",
            id_prefix, obs.file_path, obs.timestamp, obs.description
        );
        let hash = blake3::hash(input.as_bytes());
        hash.to_hex()[..32].to_string()
    };

    UniversalNode {
        id,
        kind: UniversalKind::ObservationEntry,
        domain: DomainTag::Observation,
        label: obs.description.clone(),
        file_path: obs.file_path.clone(),
        span: None,
        signature: Some(format!("[{}] {}", obs.change_type, obs.file_path)),
        doc_comment: obs.impact.clone(),
        identifiers: obs.related_symbols.clone(),
        bm25_text: format!(
            "observation {} {} {} {}",
            obs.change_type,
            obs.file_path,
            obs.description,
            obs.related_symbols.join(" ")
        ),
        symbol_vec: None,
        ast_hash: None,
        temporal: Some(TemporalMeta::at(obs.timestamp)),
        attributes: {
            let mut m = HashMap::new();
            m.insert(
                "priority".into(),
                AttributeValue::String(format!("{:?}", obs.priority)),
            );
            m.insert(
                "change_type".into(),
                AttributeValue::String(obs.change_type.to_string()),
            );
            m
        },
        intent_summary: None,
    }
}

/// Convert a reflection to a UniversalNode.
pub fn reflection_to_node(reflection: &ReflectionEntry, id_prefix: &str) -> UniversalNode {
    let id = {
        let input = format!("{}:reflection:{}", id_prefix, reflection.timestamp);
        let hash = blake3::hash(input.as_bytes());
        hash.to_hex()[..32].to_string()
    };

    UniversalNode {
        id,
        kind: UniversalKind::Reflection,
        domain: DomainTag::Observation,
        label: reflection.summary.clone(),
        file_path: String::new(),
        span: None,
        signature: reflection.pattern.clone(),
        doc_comment: Some(reflection.scope.clone()),
        identifiers: reflection.related_files.clone(),
        bm25_text: format!(
            "reflection {} {} {}",
            reflection.summary,
            reflection.scope,
            reflection.related_files.join(" ")
        ),
        symbol_vec: None,
        ast_hash: None,
        temporal: Some(TemporalMeta::at(reflection.timestamp)),
        attributes: HashMap::new(),
        intent_summary: None,
    }
}

/// Generate a simple reflection from a batch of observations (rule-based, no LLM).
pub fn generate_reflection(observations: &[Observation]) -> Option<ReflectionEntry> {
    if observations.is_empty() {
        return None;
    }

    let files: Vec<String> = observations
        .iter()
        .map(|o| o.file_path.clone())
        .collect::<std::collections::HashSet<_>>()
        .into_iter()
        .collect();

    let added_count = observations
        .iter()
        .filter(|o| o.change_type == ChangeType::Added)
        .count();
    let modified_count = observations
        .iter()
        .filter(|o| o.change_type == ChangeType::Modified)
        .count();
    let deleted_count = observations
        .iter()
        .filter(|o| o.change_type == ChangeType::Deleted)
        .count();

    let pattern = if added_count > modified_count + deleted_count {
        "feature_addition"
    } else if deleted_count > added_count + modified_count {
        "cleanup"
    } else if modified_count > 0 && added_count == 0 && deleted_count == 0 {
        "refactoring"
    } else {
        "mixed_changes"
    };

    let summary = format!(
        "{} files changed: {} added, {} modified, {} removed",
        files.len(),
        added_count,
        modified_count,
        deleted_count
    );

    let scope = if files.len() == 1 {
        files[0].clone()
    } else {
        // Find common path prefix
        let common = common_prefix(&files);
        if common.is_empty() {
            "multiple modules".to_string()
        } else {
            common
        }
    };

    let latest_ts = observations
        .iter()
        .map(|o| o.timestamp)
        .max()
        .unwrap_or(0);

    Some(ReflectionEntry {
        timestamp: latest_ts,
        summary,
        pattern: Some(pattern.to_string()),
        scope,
        related_files: files,
        observation_ids: vec![], // filled by caller
    })
}

/// Find the common path prefix among file paths.
fn common_prefix(paths: &[String]) -> String {
    if paths.is_empty() {
        return String::new();
    }
    if paths.len() == 1 {
        return paths[0]
            .rsplit_once('/')
            .map(|(dir, _)| dir.to_string())
            .unwrap_or_default();
    }

    let parts: Vec<Vec<&str>> = paths.iter().map(|p| p.split('/').collect()).collect();
    let min_len = parts.iter().map(|p| p.len()).min().unwrap_or(0);

    let mut common = Vec::new();
    for i in 0..min_len {
        let segment = parts[0][i];
        if parts.iter().all(|p| p[i] == segment) {
            common.push(segment);
        } else {
            break;
        }
    }

    common.join("/")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_detect_changes_added() {
        let old = vec!["foo".to_string(), "bar".to_string()];
        let new = vec!["foo".to_string(), "bar".to_string(), "baz".to_string()];

        let obs = detect_changes("src/lib.rs", &old, &new, 1000);
        assert_eq!(obs.len(), 1);
        assert_eq!(obs[0].change_type, ChangeType::Added);
        assert!(obs[0].description.contains("baz"));
    }

    #[test]
    fn test_detect_changes_deleted() {
        let old = vec!["foo".to_string(), "bar".to_string()];
        let new = vec!["foo".to_string()];

        let obs = detect_changes("src/lib.rs", &old, &new, 1000);
        assert_eq!(obs.len(), 1);
        assert_eq!(obs[0].change_type, ChangeType::Deleted);
        assert_eq!(obs[0].priority, ObservationPriority::High);
    }

    #[test]
    fn test_detect_changes_none() {
        let syms = vec!["foo".to_string(), "bar".to_string()];
        let obs = detect_changes("src/lib.rs", &syms, &syms, 1000);
        assert!(obs.is_empty());
    }

    #[test]
    fn test_observation_to_node() {
        let obs = Observation {
            timestamp: 1000,
            file_path: "src/auth.rs".to_string(),
            change_type: ChangeType::Modified,
            description: "authenticate() signature changed".to_string(),
            impact: Some("Callers need update".to_string()),
            priority: ObservationPriority::High,
            related_symbols: vec!["authenticate".to_string()],
        };

        let node = observation_to_node(&obs, "test");
        assert_eq!(node.domain, DomainTag::Observation);
        assert_eq!(node.kind, UniversalKind::ObservationEntry);
        assert!(node.bm25_text.contains("authenticate"));
        assert!(node.temporal.is_some());
    }

    #[test]
    fn test_generate_reflection() {
        let observations = vec![
            Observation {
                timestamp: 1000,
                file_path: "src/auth.rs".to_string(),
                change_type: ChangeType::Added,
                description: "New MFA module".to_string(),
                impact: None,
                priority: ObservationPriority::Medium,
                related_symbols: vec!["mfa_verify".to_string()],
            },
            Observation {
                timestamp: 1001,
                file_path: "src/auth.rs".to_string(),
                change_type: ChangeType::Added,
                description: "New TOTP handler".to_string(),
                impact: None,
                priority: ObservationPriority::Medium,
                related_symbols: vec!["totp_handler".to_string()],
            },
        ];

        let reflection = generate_reflection(&observations).unwrap();
        assert_eq!(reflection.pattern.as_deref(), Some("feature_addition"));
        assert!(reflection.summary.contains("2 added"));
        assert!(!reflection.related_files.is_empty());
    }

    #[test]
    fn test_generate_reflection_empty() {
        assert!(generate_reflection(&[]).is_none());
    }

    #[test]
    fn test_reflection_to_node() {
        let reflection = ReflectionEntry {
            timestamp: 1000,
            summary: "Auth module getting MFA support".to_string(),
            pattern: Some("feature_addition".to_string()),
            scope: "src/auth".to_string(),
            related_files: vec!["src/auth.rs".to_string()],
            observation_ids: vec![],
        };

        let node = reflection_to_node(&reflection, "test");
        assert_eq!(node.kind, UniversalKind::Reflection);
        assert_eq!(node.domain, DomainTag::Observation);
        assert!(node.bm25_text.contains("Auth module"));
    }

    #[test]
    fn test_common_prefix() {
        let paths = vec![
            "src/auth/login.rs".to_string(),
            "src/auth/mfa.rs".to_string(),
            "src/auth/jwt.rs".to_string(),
        ];
        assert_eq!(common_prefix(&paths), "src/auth");
    }

    #[test]
    fn test_common_prefix_no_common() {
        let paths = vec!["src/auth.rs".to_string(), "lib/db.rs".to_string()];
        assert_eq!(common_prefix(&paths), "");
    }
}
