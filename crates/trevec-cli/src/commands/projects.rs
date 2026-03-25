use anyhow::{Context, Result};
use clap::Subcommand;
use serde::{Deserialize, Serialize};
use std::path::{Path, PathBuf};

use trevec_core::model::CodeNode;
use trevec_core::TrevecConfig;
use trevec_index::graph::CodeGraph;
use trevec_index::memory_store::MemoryStore;

// ---------------------------------------------------------------------------
// CLI subcommands
// ---------------------------------------------------------------------------

#[derive(Subcommand)]
pub enum ProjectsAction {
    /// List all tracked repositories
    List,

    /// Show detailed project info
    Show {
        /// Path to the repository
        #[arg(default_value = ".")]
        path: PathBuf,
    },

    /// Remove trevec data for a repository
    Remove {
        /// Path to the repository
        path: PathBuf,

        /// Skip confirmation
        #[arg(long)]
        force: bool,
    },
}

pub async fn run(action: ProjectsAction) -> Result<()> {
    match action {
        ProjectsAction::List => list_projects().await,
        ProjectsAction::Show { path } => show_project(path).await,
        ProjectsAction::Remove { path, force } => remove_project(path, force),
    }
}

// ---------------------------------------------------------------------------
// Registry (persisted at ~/.trevec/projects.json)
// ---------------------------------------------------------------------------

#[derive(Debug, Serialize, Deserialize, Clone)]
pub struct RegistryEntry {
    pub path: String,
    pub added_at: i64,
}

fn registry_path() -> PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| ".".to_string());
    PathBuf::from(home).join(".trevec/projects.json")
}

pub fn load_registry() -> Vec<RegistryEntry> {
    load_registry_from(&registry_path())
}

fn load_registry_from(path: &Path) -> Vec<RegistryEntry> {
    match std::fs::read_to_string(path) {
        Ok(content) => serde_json::from_str(&content).unwrap_or_default(),
        Err(_) => vec![],
    }
}

fn save_registry_to(path: &Path, entries: &[RegistryEntry]) -> Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create {}", parent.display()))?;
    }
    let json = serde_json::to_string_pretty(entries)?;
    std::fs::write(path, json)
        .with_context(|| format!("Failed to write {}", path.display()))?;
    Ok(())
}

/// Register a project path in the global registry. Called by serve and mcp setup.
pub fn register_project(repo_path: &Path) {
    register_project_in(repo_path, &registry_path());
}

fn register_project_in(repo_path: &Path, reg_path: &Path) {
    let canonical = repo_path
        .canonicalize()
        .unwrap_or_else(|_| repo_path.to_path_buf());
    let path_str = canonical.to_string_lossy().to_string();

    let mut entries = load_registry_from(reg_path);
    if entries.iter().any(|e| e.path == path_str) {
        return; // Already registered
    }

    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    entries.push(RegistryEntry {
        path: path_str,
        added_at: now,
    });

    if let Err(e) = save_registry_to(reg_path, &entries) {
        tracing::warn!("Failed to update project registry: {e}");
    }
}

/// Remove a project path from the global registry.
fn unregister_project(repo_path: &Path) {
    unregister_project_from(repo_path, &registry_path());
}

fn unregister_project_from(repo_path: &Path, reg_path: &Path) {
    let canonical = repo_path
        .canonicalize()
        .unwrap_or_else(|_| repo_path.to_path_buf());
    let path_str = canonical.to_string_lossy().to_string();

    let mut entries = load_registry_from(reg_path);
    let before = entries.len();
    entries.retain(|e| e.path != path_str);
    if entries.len() < before {
        let _ = save_registry_to(reg_path, &entries);
    }
}

// ---------------------------------------------------------------------------
// Subcommand implementations
// ---------------------------------------------------------------------------

async fn list_projects() -> Result<()> {
    let entries = load_registry();
    if entries.is_empty() {
        eprintln!("No tracked projects.");
        eprintln!("Run `trevec init` in a repository to register it.");
        return Ok(());
    }

    // Header
    eprintln!(
        "{:<50} {:<12} {:<18} MCP CLIENTS",
        "PATH", "STATUS", "LAST INDEXED"
    );

    for entry in &entries {
        let repo = PathBuf::from(&entry.path);
        let data_dir = repo.join(".trevec");

        // Status
        let nodes_path = data_dir.join("nodes.json");
        let status = if !data_dir.exists() {
            "not init"
        } else if !nodes_path.exists() {
            "not indexed"
        } else {
            "indexed"
        };

        // Last indexed (from nodes.json mtime)
        let last_indexed = if nodes_path.exists() {
            nodes_path
                .metadata()
                .ok()
                .and_then(|m| m.modified().ok())
                .map(format_relative_time)
                .unwrap_or_else(|| "\u{2014}".to_string())
        } else {
            "\u{2014}".to_string()
        };

        // MCP clients — check global first, then legacy per-repo
        let mut clients = Vec::new();
        for (key, label) in [
            ("claude", "Claude Desktop"),
            ("claude-code", "Claude Code"),
            ("cursor", "Cursor"),
            ("codex", "Codex"),
        ] {
            if crate::commands::setup::is_globally_configured(key) {
                clients.push(format!("{} (global)", label));
            } else if crate::commands::setup::is_client_configured(&repo, key) {
                clients.push(format!("{} (legacy)", label));
            }
        }
        let clients_str = if clients.is_empty() {
            "\u{2014}".to_string()
        } else {
            clients.join(", ")
        };

        // Shorten path for display
        let display_path = shorten_path(&entry.path);
        eprintln!(
            "{:<50} {:<12} {:<18} {}",
            display_path, status, last_indexed, clients_str
        );
    }

    Ok(())
}

async fn show_project(path: PathBuf) -> Result<()> {
    let path = path.canonicalize().unwrap_or(path);
    let data_dir = path.join(".trevec");

    eprintln!("Repository: {}", path.display());

    if !data_dir.exists() {
        eprintln!("Status:     not initialized");
        eprintln!("\nRun `trevec init` to initialize and index.");
        return Ok(());
    }

    let nodes_path = data_dir.join("nodes.json");
    if !nodes_path.exists() {
        eprintln!("Status:     initialized (not indexed)");
        eprintln!("\nRun `trevec index` to index.");
        return Ok(());
    }

    eprintln!("Status:     indexed");

    // Last indexed
    if let Ok(meta) = nodes_path.metadata() {
        if let Ok(modified) = meta.modified() {
            eprintln!("Last indexed: {}", format_relative_time(modified));
        }
    }

    // Load nodes for stats
    let nodes_json = std::fs::read_to_string(&nodes_path).context("Failed to read nodes")?;
    let nodes: Vec<CodeNode> = serde_json::from_str(&nodes_json).context("Failed to parse nodes")?;

    // Unique files
    let unique_files: std::collections::HashSet<&str> =
        nodes.iter().map(|n| n.file_path.as_str()).collect();

    // Load graph
    let graph_path = data_dir.join("graph.bin");
    let (graph_nodes, graph_edges) = if let Ok(graph) = CodeGraph::load(&graph_path) {
        (graph.node_count(), graph.edge_count())
    } else {
        (0, 0)
    };

    eprintln!("\nCode:");
    eprintln!("  Files indexed:  {}", unique_files.len());
    eprintln!("  Code nodes:     {}", nodes.len());
    eprintln!("  Graph nodes:    {}", graph_nodes);
    eprintln!("  Graph edges:    {}", graph_edges);

    // Storage sizes
    eprintln!("\nStorage:");
    let total_size = dir_size(&data_dir);
    eprintln!("  Total .trevec:  {}", format_bytes(total_size));

    let lance_dir = data_dir.join("lance");
    if lance_dir.exists() {
        eprintln!("  Lance DB:       {}", format_bytes(dir_size(&lance_dir)));
    }
    if graph_path.exists() {
        let size = std::fs::metadata(&graph_path).map(|m| m.len()).unwrap_or(0);
        eprintln!("  Graph:          {}", format_bytes(size));
    }
    if nodes_path.exists() {
        let size = std::fs::metadata(&nodes_path).map(|m| m.len()).unwrap_or(0);
        eprintln!("  nodes.json:     {}", format_bytes(size));
    }

    // Memory stats
    let config = TrevecConfig::load(&data_dir);
    if config.memory.enabled {
        eprintln!("\nMemory:");
        if let Ok(mem_store) = MemoryStore::open(lance_dir.to_str().unwrap_or("")).await {
            let mem_count = mem_store.count().await.unwrap_or(0);
            eprintln!("  Total events:   {}", mem_count);

            let meta = trevec_index::memory::meta::MemoryMeta::load(&data_dir);
            eprintln!("  Events pruned:  {}", meta.total_events_pruned);

            eprintln!(
                "  Retention:      {} days ({} max)",
                config.memory.retention_days,
                config.memory.max_events
            );

            if let Some(ts) = meta.last_gc {
                let now = std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs() as i64;
                let elapsed_min = (now - ts) / 60;
                let gc_interval = config.memory.gc_interval_minutes as i64;
                let next_gc = gc_interval.saturating_sub(elapsed_min);
                eprintln!("  Next GC:        ~{} min", next_gc.max(0));
            }
        }
    } else {
        eprintln!("\nMemory: disabled");
    }

    // MCP client status — check global first, then legacy per-repo
    eprintln!("\nMCP Clients:");
    for (key, label) in [
        ("claude", "Claude Desktop"),
        ("claude-code", "Claude Code"),
        ("cursor", "Cursor"),
        ("codex", "Codex"),
    ] {
        let status = if crate::commands::setup::is_globally_configured(key) {
            "global"
        } else if crate::commands::setup::is_client_configured(&path, key) {
            "legacy (per-repo)"
        } else {
            "not configured"
        };
        eprintln!("  {:<16}{}", format!("{}:", label), status);
    }

    // Rule-file status
    eprintln!("\nRule Files:");
    for (key, label) in [
        ("cursor", "Cursor (.cursor/rules/trevec.mdc)"),
        ("claude-code", "Claude Code (CLAUDE.md)"),
        ("codex", "Codex (AGENTS.md)"),
    ] {
        let present = crate::commands::rules::is_rule_present(&path, key);
        eprintln!(
            "  {}: {}",
            label,
            if present { "present" } else { "missing" }
        );
    }

    Ok(())
}

fn remove_project(path: PathBuf, force: bool) -> Result<()> {
    let path = path.canonicalize().unwrap_or(path);
    let data_dir = path.join(".trevec");

    if !data_dir.exists() {
        eprintln!("No .trevec/ directory found at {}", path.display());
        return Ok(());
    }

    if !force {
        eprintln!(
            "This will delete {} and all index data.",
            data_dir.display()
        );
        eprint!("Continue? [y/N] ");
        std::io::Write::flush(&mut std::io::stderr())?;
        let mut input = String::new();
        std::io::stdin().read_line(&mut input)?;
        if !input.trim().eq_ignore_ascii_case("y") {
            eprintln!("Cancelled.");
            return Ok(());
        }
    }

    std::fs::remove_dir_all(&data_dir)
        .with_context(|| format!("Failed to remove {}", data_dir.display()))?;
    eprintln!("Removed {}", data_dir.display());

    // Unregister from project registry
    unregister_project(&path);
    eprintln!("Removed from project registry.");

    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn format_relative_time(time: std::time::SystemTime) -> String {
    let now = std::time::SystemTime::now();
    let elapsed = now.duration_since(time).unwrap_or_default();
    let secs = elapsed.as_secs();

    if secs < 60 {
        "just now".to_string()
    } else if secs < 3600 {
        format!("{} min ago", secs / 60)
    } else if secs < 86400 {
        format!("{} hours ago", secs / 3600)
    } else {
        format!("{} days ago", secs / 86400)
    }
}

pub fn shorten_path(path: &str) -> String {
    if let Ok(home) = std::env::var("HOME") {
        if let Some(rest) = path.strip_prefix(&home) {
            return format!("~{rest}");
        }
    }
    path.to_string()
}

fn dir_size(path: &Path) -> u64 {
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

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_registry_roundtrip() {
        let tmp = tempfile::tempdir().unwrap();
        let reg_path = tmp.path().join("projects.json");

        // Empty initially
        let entries = load_registry_from(&reg_path);
        assert!(entries.is_empty());

        // Save some entries
        let entries = vec![
            RegistryEntry {
                path: "/Users/alice/dev/backend".to_string(),
                added_at: 1707300000,
            },
            RegistryEntry {
                path: "/Users/alice/dev/frontend".to_string(),
                added_at: 1707300100,
            },
        ];
        save_registry_to(&reg_path, &entries).unwrap();

        // Reload and verify
        let loaded = load_registry_from(&reg_path);
        assert_eq!(loaded.len(), 2);
        assert_eq!(loaded[0].path, "/Users/alice/dev/backend");
        assert_eq!(loaded[1].path, "/Users/alice/dev/frontend");
        assert_eq!(loaded[0].added_at, 1707300000);
    }

    #[test]
    fn test_register_project_creates_entry() {
        let tmp = tempfile::tempdir().unwrap();
        let reg_path = tmp.path().join("registry/projects.json");
        let repo = tmp.path().join("my-project");
        std::fs::create_dir_all(&repo).unwrap();

        register_project_in(&repo, &reg_path);

        let entries = load_registry_from(&reg_path);
        assert_eq!(entries.len(), 1);
        // Path should be canonical
        let canonical = repo.canonicalize().unwrap();
        assert_eq!(entries[0].path, canonical.to_string_lossy());
        assert!(entries[0].added_at > 0);
    }

    #[test]
    fn test_register_project_idempotent() {
        let tmp = tempfile::tempdir().unwrap();
        let reg_path = tmp.path().join("projects.json");
        let repo = tmp.path().join("my-project");
        std::fs::create_dir_all(&repo).unwrap();

        register_project_in(&repo, &reg_path);
        register_project_in(&repo, &reg_path);
        register_project_in(&repo, &reg_path);

        let entries = load_registry_from(&reg_path);
        assert_eq!(entries.len(), 1);
    }

    #[test]
    fn test_register_multiple_projects() {
        let tmp = tempfile::tempdir().unwrap();
        let reg_path = tmp.path().join("projects.json");

        let repo_a = tmp.path().join("repo-a");
        let repo_b = tmp.path().join("repo-b");
        std::fs::create_dir_all(&repo_a).unwrap();
        std::fs::create_dir_all(&repo_b).unwrap();

        register_project_in(&repo_a, &reg_path);
        register_project_in(&repo_b, &reg_path);

        let entries = load_registry_from(&reg_path);
        assert_eq!(entries.len(), 2);
    }

    #[test]
    fn test_unregister_project() {
        let tmp = tempfile::tempdir().unwrap();
        let reg_path = tmp.path().join("projects.json");
        let repo = tmp.path().join("my-project");
        std::fs::create_dir_all(&repo).unwrap();

        register_project_in(&repo, &reg_path);
        assert_eq!(load_registry_from(&reg_path).len(), 1);

        unregister_project_from(&repo, &reg_path);
        assert_eq!(load_registry_from(&reg_path).len(), 0);
    }

    #[test]
    fn test_unregister_nonexistent_is_noop() {
        let tmp = tempfile::tempdir().unwrap();
        let reg_path = tmp.path().join("projects.json");
        let repo_a = tmp.path().join("repo-a");
        let repo_b = tmp.path().join("repo-b");
        std::fs::create_dir_all(&repo_a).unwrap();
        std::fs::create_dir_all(&repo_b).unwrap();

        register_project_in(&repo_a, &reg_path);
        assert_eq!(load_registry_from(&reg_path).len(), 1);

        // Unregister repo_b (not registered) → no change
        unregister_project_from(&repo_b, &reg_path);
        assert_eq!(load_registry_from(&reg_path).len(), 1);
    }

    #[test]
    fn test_load_registry_malformed_json() {
        let tmp = tempfile::tempdir().unwrap();
        let reg_path = tmp.path().join("projects.json");
        std::fs::write(&reg_path, "not valid json {{{").unwrap();

        // Should fall back to empty
        let entries = load_registry_from(&reg_path);
        assert!(entries.is_empty());
    }

    #[test]
    fn test_format_relative_time() {
        let now = std::time::SystemTime::now();

        assert_eq!(format_relative_time(now), "just now");

        let five_min_ago = now - std::time::Duration::from_secs(300);
        assert_eq!(format_relative_time(five_min_ago), "5 min ago");

        let two_hours_ago = now - std::time::Duration::from_secs(7200);
        assert_eq!(format_relative_time(two_hours_ago), "2 hours ago");

        let three_days_ago = now - std::time::Duration::from_secs(259200);
        assert_eq!(format_relative_time(three_days_ago), "3 days ago");
    }

    #[test]
    fn test_shorten_path() {
        // Always test the non-home case since HOME may vary
        assert_eq!(shorten_path("/opt/data/repo"), "/opt/data/repo");
    }

    #[test]
    fn test_format_bytes() {
        assert_eq!(format_bytes(500), "500 B");
        assert_eq!(format_bytes(1536), "1.5 KB");
        assert_eq!(format_bytes(2 * 1024 * 1024), "2.0 MB");
    }
}
