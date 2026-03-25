use anyhow::{Context, Result};
use clap::Subcommand;
use std::collections::HashMap;
use std::path::PathBuf;

use trevec_core::model::CodeNode;
use trevec_core::TrevecConfig;
use trevec_index::graph::CodeGraph;
use trevec_index::memory;
use trevec_index::memory_store::MemoryStore;

#[derive(Subcommand)]
pub enum MemoryAction {
    /// Show memory event counts, disk usage, and source breakdown
    Status,
    /// Sync memory from enabled sources (extract + ingest)
    Sync {
        /// Only sync a specific source: cursor, claude-code, codex
        #[arg(long)]
        source: Option<String>,
    },
    /// Run garbage collection on memory events
    Gc {
        /// Show what would be deleted without actually deleting
        #[arg(long)]
        dry_run: bool,
    },
    /// Delete all memory data
    Wipe {
        /// Skip confirmation prompt
        #[arg(long)]
        force: bool,
    },
}

pub async fn run(
    action: MemoryAction,
    path: PathBuf,
    data_dir: PathBuf,
) -> Result<()> {
    let path = path.canonicalize().unwrap_or(path);
    let config = TrevecConfig::load(&data_dir);

    if !config.memory.enabled {
        eprintln!("Memory is disabled. Enable it in .trevec/config.toml:");
        eprintln!("  [memory]");
        eprintln!("  enabled = true");
        return Ok(());
    }

    match action {
        MemoryAction::Status => run_status(&data_dir).await,
        MemoryAction::Sync { source } => run_sync(&path, &data_dir, &config, source).await,
        MemoryAction::Gc { dry_run } => run_gc(&data_dir, &config, dry_run).await,
        MemoryAction::Wipe { force } => run_wipe(&data_dir, force).await,
    }
}

async fn run_status(data_dir: &std::path::Path) -> Result<()> {
    let lance_dir = data_dir.join("lance");
    let store = MemoryStore::open(lance_dir.to_str().unwrap())
        .await
        .context("Failed to open memory store")?;

    let count = store.count().await?;
    let meta = memory::meta::MemoryMeta::load(data_dir);

    eprintln!("Memory Status:");
    eprintln!("  Events:       {}", count);
    eprintln!(
        "  Last GC:      {}",
        meta.last_gc
            .map(format_ts)
            .unwrap_or_else(|| "never".to_string())
    );
    eprintln!("  Events pruned: {}", meta.total_events_pruned);

    // Source breakdown
    let sources_tracked = [
        ("Claude Code sessions", meta.claude_code_offsets.len()),
        ("Codex sessions", meta.codex_offsets.len()),
    ];
    for (name, tracked) in sources_tracked {
        if tracked > 0 {
            eprintln!("  {} tracked: {}", name, tracked);
        }
    }
    if let Some(rowid) = meta.cursor_last_rowid {
        eprintln!("  Cursor last rowid: {}", rowid);
    }

    // Disk size
    let lance_memory_dir = data_dir.join("lance").join(TABLE_NAME_DIR);
    if lance_memory_dir.exists() {
        let size = dir_size(&lance_memory_dir);
        eprintln!("  Memory store size: {}", format_bytes(size));
    }

    Ok(())
}

const TABLE_NAME_DIR: &str = "memory_events.lance";

async fn run_sync(
    repo_path: &std::path::Path,
    data_dir: &std::path::Path,
    config: &TrevecConfig,
    source: Option<String>,
) -> Result<()> {
    let lance_dir = data_dir.join("lance");
    let mut store = MemoryStore::open(lance_dir.to_str().unwrap())
        .await
        .context("Failed to open memory store")?;

    // Load nodes to build file→node_id map for Discussed edges
    let file_node_map = load_file_node_map(data_dir)?;

    let graph_path = data_dir.join("graph.bin");
    let mut graph = if graph_path.exists() {
        CodeGraph::load(&graph_path).unwrap_or_default()
    } else {
        CodeGraph::default()
    };

    // Build a config with only the requested source enabled
    let mem_config = if let Some(ref src) = source {
        let mut c = config.memory.clone();
        c.claude_code.enabled = src == "claude-code" || src == "claude_code";
        c.cursor.enabled = src == "cursor";
        c.codex.enabled = src == "codex";
        c
    } else {
        config.memory.clone()
    };

    // Optionally create embedder
    let mut embedder = if mem_config.semantic {
        match trevec_index::embedder::Embedder::new_with_model(
            Some(&config.embeddings.model),
            true,
            Some(data_dir.join("models")),
            None,
        ) {
            Ok(e) => Some(e),
            Err(e) => {
                eprintln!("Warning: embedder failed to load ({e}), skipping semantic");
                None
            }
        }
    } else {
        None
    };

    eprintln!("Syncing memory...");
    let stats = memory::ingest_memory(
        repo_path,
        data_dir,
        &mem_config,
        embedder.as_mut(),
        &mut store,
        &mut graph,
        &file_node_map,
    )
    .await?;

    // Save updated graph
    graph.save(&graph_path)?;

    eprintln!("Memory sync complete:");
    eprintln!("  Events ingested:  {}", stats.events_ingested);
    eprintln!("  Events skipped:   {}", stats.events_skipped_dedupe);
    eprintln!("  Redactions:       {}", stats.total_redactions);
    eprintln!("  Discussed edges:  {}", stats.discussed_edges_created);

    Ok(())
}

async fn run_gc(
    data_dir: &std::path::Path,
    config: &TrevecConfig,
    dry_run: bool,
) -> Result<()> {
    let lance_dir = data_dir.join("lance");
    let store = MemoryStore::open(lance_dir.to_str().unwrap())
        .await
        .context("Failed to open memory store")?;

    let mut meta = memory::meta::MemoryMeta::load(data_dir);

    let stats = memory::gc::run_gc(&store, &config.memory, &mut meta, dry_run).await?;

    if !dry_run {
        meta.save(data_dir)?;
    }

    let prefix = if dry_run { "Would delete" } else { "Deleted" };
    eprintln!("Memory GC {}:", if dry_run { "(dry run)" } else { "complete" });
    eprintln!("  {} expired:     {}", prefix, stats.expired_deleted);
    eprintln!("  {} raw expired: {}", prefix, stats.raw_expired_deleted);
    eprintln!("  {} over limit:  {}", prefix, stats.over_limit_deleted);
    eprintln!("  {} total:       {}", prefix, stats.total_deleted);

    Ok(())
}

async fn run_wipe(data_dir: &std::path::Path, force: bool) -> Result<()> {
    if !force {
        eprintln!("This will permanently delete all memory data.");
        eprintln!("Use --force to confirm.");
        return Ok(());
    }

    let lance_dir = data_dir.join("lance");
    let mut store = MemoryStore::open(lance_dir.to_str().unwrap())
        .await
        .context("Failed to open memory store")?;

    let count = store.count().await?;
    store.clear().await?;

    // Remove meta file
    let meta_path = data_dir.join("memory_meta.json");
    if meta_path.exists() {
        std::fs::remove_file(&meta_path)?;
    }

    eprintln!("Wiped {} memory events.", count);
    Ok(())
}

/// Build a map of file_path → [node_ids] from the nodes.json index.
fn load_file_node_map(
    data_dir: &std::path::Path,
) -> Result<HashMap<String, Vec<String>>> {
    let nodes_path = data_dir.join("nodes.json");
    let mut map: HashMap<String, Vec<String>> = HashMap::new();

    if let Ok(json) = std::fs::read_to_string(&nodes_path) {
        if let Ok(nodes) = serde_json::from_str::<Vec<CodeNode>>(&json) {
            for node in nodes {
                map.entry(node.file_path.clone())
                    .or_default()
                    .push(node.id.clone());
            }
        }
    }

    Ok(map)
}

fn format_ts(ts: i64) -> String {
    // Simple unix timestamp to human-readable
    let days_ago =
        (std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64
            - ts)
            / 86400;
    if days_ago == 0 {
        "today".to_string()
    } else if days_ago == 1 {
        "yesterday".to_string()
    } else {
        format!("{} days ago", days_ago)
    }
}

fn dir_size(path: &std::path::Path) -> u64 {
    let mut total = 0;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            if let Ok(meta) = entry.metadata() {
                if meta.is_file() {
                    total += meta.len();
                } else if meta.is_dir() {
                    total += dir_size(&entry.path());
                }
            }
        }
    }
    total
}

fn format_bytes(bytes: u64) -> String {
    if bytes < 1024 {
        format!("{} B", bytes)
    } else if bytes < 1024 * 1024 {
        format!("{:.1} KB", bytes as f64 / 1024.0)
    } else {
        format!("{:.1} MB", bytes as f64 / (1024.0 * 1024.0))
    }
}
