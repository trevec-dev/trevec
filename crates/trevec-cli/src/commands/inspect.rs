use anyhow::{Context, Result};
use std::collections::HashMap;
use std::path::{Path, PathBuf};

use trevec_core::model::CodeNode;
use trevec_core::TrevecConfig;
use trevec_index::graph::CodeGraph;
use trevec_index::memory_store::MemoryStore;
use trevec_index::store::Store;

/// Run the inspect command for debugging the index.
pub async fn run(
    data_dir: PathBuf,
    show_stats: bool,
    node_id: Option<String>,
) -> Result<()> {
    if show_stats {
        print_stats(&data_dir).await?;
    }

    if let Some(ref id) = node_id {
        print_node(&data_dir, id)?;
    }

    if !show_stats && node_id.is_none() {
        // Default: show stats
        print_stats(&data_dir).await?;
    }

    Ok(())
}

async fn print_stats(data_dir: &Path) -> Result<()> {
    // Load nodes
    let nodes_path = data_dir.join("nodes.json");
    let nodes_json =
        std::fs::read_to_string(&nodes_path).context("No index found. Run `trevec index` first.")?;
    let nodes: Vec<CodeNode> =
        serde_json::from_str(&nodes_json).context("Failed to parse nodes")?;

    // Load graph
    let graph_path = data_dir.join("graph.bin");
    let graph = CodeGraph::load(&graph_path).context("Failed to load graph")?;

    // Load store for count
    let lance_dir = data_dir.join("lance");
    let store = Store::open(lance_dir.to_str().unwrap())
        .await
        .context("Failed to open store")?;
    let store_count = store.count().await.unwrap_or(0);

    // Count by kind
    let mut by_kind: HashMap<String, usize> = HashMap::new();
    for node in &nodes {
        *by_kind.entry(node.kind.to_string()).or_default() += 1;
    }

    // Count by language (from file extension)
    let mut by_file: HashMap<String, usize> = HashMap::new();
    for node in &nodes {
        let ext = std::path::Path::new(&node.file_path)
            .extension()
            .and_then(|e| e.to_str())
            .unwrap_or("unknown")
            .to_string();
        *by_file.entry(ext).or_default() += 1;
    }

    // Unique files
    let unique_files: std::collections::HashSet<&str> =
        nodes.iter().map(|n| n.file_path.as_str()).collect();

    eprintln!("Index Statistics:");
    eprintln!("  Nodes:       {}", nodes.len());
    eprintln!("  Store rows:  {}", store_count);
    eprintln!("  Graph nodes: {}", graph.node_count());
    eprintln!("  Graph edges: {}", graph.edge_count());
    eprintln!("  Files:       {}", unique_files.len());

    if !by_kind.is_empty() {
        eprintln!("\n  By kind:");
        let mut kinds: Vec<_> = by_kind.iter().collect();
        kinds.sort_by(|a, b| b.1.cmp(a.1));
        for (kind, count) in kinds {
            eprintln!("    {}: {}", kind, count);
        }
    }

    if !by_file.is_empty() {
        eprintln!("\n  By extension:");
        let mut exts: Vec<_> = by_file.iter().collect();
        exts.sort_by(|a, b| b.1.cmp(a.1));
        for (ext, count) in exts {
            eprintln!("    .{}: {}", ext, count);
        }
    }

    // Index size on disk
    let lance_dir = data_dir.join("lance");
    if lance_dir.exists() {
        let size = dir_size(&lance_dir);
        eprintln!("\n  Lance DB size: {}", format_bytes(size));
    }

    let graph_path = data_dir.join("graph.bin");
    if graph_path.exists() {
        let size = std::fs::metadata(&graph_path).map(|m| m.len()).unwrap_or(0);
        eprintln!("  Graph size:    {}", format_bytes(size));
    }

    // Memory stats
    let config = TrevecConfig::load(data_dir);
    if config.memory.enabled {
        let lance_dir = data_dir.join("lance");
        if let Ok(mem_store) = MemoryStore::open(lance_dir.to_str().unwrap()).await {
            let mem_count = mem_store.count().await.unwrap_or(0);
            eprintln!("\n  Memory events: {}", mem_count);

            let meta = trevec_index::memory::meta::MemoryMeta::load(data_dir);
            if let Some(ts) = meta.last_gc {
                let days = (std::time::SystemTime::now()
                    .duration_since(std::time::UNIX_EPOCH)
                    .unwrap_or_default()
                    .as_secs() as i64
                    - ts)
                    / 86400;
                eprintln!("  Last GC:       {} days ago", days);
            }
            eprintln!("  Events pruned: {}", meta.total_events_pruned);
        }
    }

    Ok(())
}

fn print_node(data_dir: &Path, node_id: &str) -> Result<()> {
    let nodes_path = data_dir.join("nodes.json");
    let nodes_json =
        std::fs::read_to_string(&nodes_path).context("No index found. Run `trevec index` first.")?;
    let nodes: Vec<CodeNode> =
        serde_json::from_str(&nodes_json).context("Failed to parse nodes")?;

    let node = nodes.iter().find(|n| n.id == node_id || n.name == node_id);

    match node {
        Some(n) => {
            eprintln!("Node: {}", n.id);
            eprintln!("  Name:       {}", n.name);
            eprintln!("  Kind:       {}", n.kind);
            eprintln!("  File:       {}", n.file_path);
            eprintln!(
                "  Span:       L{}-L{} ({}-{})",
                n.span.start_line + 1,
                n.span.end_line + 1,
                n.span.start_byte,
                n.span.end_byte
            );
            eprintln!("  Signature:  {}", n.signature);
            if let Some(ref doc) = n.doc_comment {
                eprintln!("  Doc:        {}", doc);
            }
            eprintln!("  Identifiers: {:?}", n.identifiers);
            eprintln!("  AST Hash:   {}", n.ast_hash);

            // Show graph neighbors
            let graph_path = data_dir.join("graph.bin");
            if let Ok(graph) = CodeGraph::load(&graph_path) {
                let neighbors = graph.neighbors(&n.id);
                if !neighbors.is_empty() {
                    eprintln!("\n  Graph neighbors:");
                    for (neighbor_id, edge_type, confidence) in &neighbors {
                        let neighbor_name = nodes
                            .iter()
                            .find(|nn| &nn.id == neighbor_id)
                            .map(|nn| nn.name.as_str())
                            .unwrap_or("?");
                        eprintln!(
                            "    {} {} ({}) [{}]",
                            edge_type, neighbor_name, neighbor_id, confidence
                        );
                    }
                }
            }
        }
        None => {
            eprintln!("Node '{}' not found.", node_id);
            // Show some suggestions
            let matches: Vec<_> = nodes
                .iter()
                .filter(|n| n.name.contains(node_id) || n.id.starts_with(node_id))
                .take(5)
                .collect();
            if !matches.is_empty() {
                eprintln!("Did you mean:");
                for m in matches {
                    eprintln!("  {} ({})", m.name, m.id);
                }
            }
        }
    }

    Ok(())
}

fn dir_size(path: &std::path::Path) -> u64 {
    let mut total = 0;
    if let Ok(entries) = std::fs::read_dir(path) {
        for entry in entries.flatten() {
            let metadata = entry.metadata();
            if let Ok(meta) = metadata {
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
