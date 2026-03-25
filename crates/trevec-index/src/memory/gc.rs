use anyhow::Result;
use std::time::{SystemTime, UNIX_EPOCH};
use trevec_core::config::MemoryConfig;

use super::meta::MemoryMeta;
use crate::memory_store::MemoryStore;

/// Statistics from a GC run.
#[derive(Debug, Default)]
pub struct GcStats {
    pub expired_deleted: usize,
    pub raw_expired_deleted: usize,
    pub over_limit_deleted: usize,
    pub total_deleted: usize,
}

/// Run garbage collection on the memory store.
pub async fn run_gc(
    store: &MemoryStore,
    config: &MemoryConfig,
    meta: &mut MemoryMeta,
    dry_run: bool,
) -> Result<GcStats> {
    let mut stats = GcStats::default();
    let now = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    // 1. Delete events older than retention_days (non-pinned)
    let retention_cutoff = now - (config.retention_days as i64 * 86400);
    let expired_count = store.count_expired(retention_cutoff).await?;
    stats.expired_deleted = expired_count;

    if !dry_run && expired_count > 0 {
        store.delete_expired(retention_cutoff).await?;
    }

    // 2. If raw_retention_days < retention_days, also prune raw turns
    if config.raw_retention_days < config.retention_days {
        let raw_cutoff = now - (config.raw_retention_days as i64 * 86400);
        let raw_expired = store.count_raw_expired(raw_cutoff).await?;
        stats.raw_expired_deleted = raw_expired;

        if !dry_run && raw_expired > 0 {
            store.delete_raw_expired(raw_cutoff).await?;
        }
    }

    // 3. Check event count limit
    let total_count = store.count().await?;
    if total_count > config.max_events as usize {
        let over = total_count - config.max_events as usize;
        stats.over_limit_deleted = over;

        if !dry_run {
            store.delete_oldest_nonpinned(over).await?;
        }
    }

    stats.total_deleted = stats.expired_deleted + stats.raw_expired_deleted + stats.over_limit_deleted;

    if !dry_run {
        meta.last_gc = Some(now);
        meta.total_events_pruned += stats.total_deleted as u64;
    }

    Ok(stats)
}
