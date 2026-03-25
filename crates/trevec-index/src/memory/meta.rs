use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

/// Tracks per-source ingestion cursors for incremental memory extraction.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct MemoryMeta {
    /// Last processed rowid from Cursor's SQLite DB
    pub cursor_last_rowid: Option<i64>,
    /// Claude Code session file → byte offset (for incremental reads)
    pub claude_code_offsets: HashMap<String, u64>,
    /// Codex session file → byte offset
    pub codex_offsets: HashMap<String, u64>,
    /// Unix timestamp of last garbage collection run
    pub last_gc: Option<i64>,
    /// Total events pruned across all GC runs
    pub total_events_pruned: u64,
}

impl MemoryMeta {
    /// Load from `<data_dir>/memory_meta.json`, returning defaults if missing.
    pub fn load(data_dir: &Path) -> Self {
        let path = meta_path(data_dir);
        match std::fs::read_to_string(&path) {
            Ok(content) => serde_json::from_str(&content).unwrap_or_default(),
            Err(_) => Self::default(),
        }
    }

    /// Save to `<data_dir>/memory_meta.json` using atomic write (tmp + rename).
    pub fn save(&self, data_dir: &Path) -> Result<()> {
        let path = meta_path(data_dir);
        let tmp_path = path.with_extension("json.tmp");
        let content = serde_json::to_string_pretty(self)
            .context("Failed to serialize memory meta")?;
        std::fs::write(&tmp_path, content.as_bytes())
            .with_context(|| format!("Failed to write {}", tmp_path.display()))?;
        std::fs::rename(&tmp_path, &path)
            .with_context(|| format!("Failed to rename {} to {}", tmp_path.display(), path.display()))?;
        Ok(())
    }
}

fn meta_path(data_dir: &Path) -> PathBuf {
    data_dir.join("memory_meta.json")
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_meta_roundtrip() {
        let dir = tempfile::tempdir().unwrap();
        let mut meta = MemoryMeta::default();
        meta.cursor_last_rowid = Some(42);
        meta.claude_code_offsets
            .insert("session1.jsonl".to_string(), 1024);
        meta.codex_offsets
            .insert("codex_session.jsonl".to_string(), 2048);
        meta.last_gc = Some(1700000000);
        meta.total_events_pruned = 100;

        meta.save(dir.path()).unwrap();
        let loaded = MemoryMeta::load(dir.path());

        assert_eq!(loaded.cursor_last_rowid, Some(42));
        assert_eq!(
            loaded.claude_code_offsets.get("session1.jsonl"),
            Some(&1024)
        );
        assert_eq!(
            loaded.codex_offsets.get("codex_session.jsonl"),
            Some(&2048)
        );
        assert_eq!(loaded.last_gc, Some(1700000000));
        assert_eq!(loaded.total_events_pruned, 100);
    }

    #[test]
    fn test_meta_load_missing() {
        let dir = tempfile::tempdir().unwrap();
        let meta = MemoryMeta::load(dir.path());
        assert!(meta.cursor_last_rowid.is_none());
        assert!(meta.claude_code_offsets.is_empty());
    }

    #[test]
    fn test_meta_atomic_write() {
        let dir = tempfile::tempdir().unwrap();
        let meta = MemoryMeta {
            cursor_last_rowid: Some(99),
            ..Default::default()
        };
        meta.save(dir.path()).unwrap();

        // Verify no .tmp file remains
        let tmp = dir.path().join("memory_meta.json.tmp");
        assert!(!tmp.exists());

        // Verify actual file exists and is valid
        let path = dir.path().join("memory_meta.json");
        assert!(path.exists());
        let loaded: MemoryMeta =
            serde_json::from_str(&std::fs::read_to_string(&path).unwrap()).unwrap();
        assert_eq!(loaded.cursor_last_rowid, Some(99));
    }
}
