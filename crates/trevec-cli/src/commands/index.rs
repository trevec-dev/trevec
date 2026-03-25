use anyhow::Result;
use std::collections::HashMap;
use std::path::PathBuf;
use trevec_core::model::CodeNode;
use trevec_core::TrevecConfig;
use trevec_index::graph::CodeGraph;
use trevec_index::memory;
use trevec_index::memory_store::MemoryStore;

/// Run the indexing pipeline on a repository.
pub async fn run(path: PathBuf, data_dir: PathBuf, verbose: bool, gpu: bool) -> Result<()> {
    let path = path.canonicalize().unwrap_or(path);

    // Auto-init: create .trevec/ config files if missing (so `trevec index` works standalone).
    let effective_data_dir = if data_dir == std::path::Path::new(".trevec") {
        path.join(".trevec")
    } else {
        data_dir.clone()
    };
    if !effective_data_dir.exists() {
        eprintln!("No .trevec/ found — initializing...");
        std::fs::create_dir_all(&effective_data_dir)?;
        let _ = crate::commands::init::write_if_missing(
            &effective_data_dir.join("config.toml"),
            crate::commands::init::DEFAULT_CONFIG,
        );
        let _ = crate::commands::init::write_if_missing(
            &effective_data_dir.join(".gitignore"),
            crate::commands::init::TREVEC_GITIGNORE,
        );
        let _ = crate::commands::init::write_if_missing(
            &effective_data_dir.join("manifest.json"),
            "{}",
        );
        eprintln!("Initialized trevec at {}", effective_data_dir.display());
    }

    let config = TrevecConfig::load(&data_dir);

    let device = if gpu { Some("cuda") } else { Some(config.embeddings.device.as_str()) }
        .filter(|d| *d != "cpu");

    if gpu {
        eprintln!("GPU acceleration requested (CUDA)");
    }

    eprintln!("Indexing {}...", path.display());

    let stats =
        trevec_index::ingest::ingest_with_config(
            &path,
            &data_dir,
            verbose,
            &config.index.exclude,
            Some(&config.embeddings.model),
            device,
        )
        .await?;

    eprintln!("\nIndexing complete:");
    eprintln!("  Files discovered: {}", stats.files_discovered);
    eprintln!("  Files parsed:     {}", stats.files_parsed);
    eprintln!("  Files unchanged:  {}", stats.files_unchanged);
    eprintln!("  Files deleted:    {}", stats.files_deleted);
    eprintln!("  Files skipped:    {}", stats.files_skipped);
    eprintln!("  Nodes extracted:  {}", stats.nodes_extracted);
    eprintln!("  Nodes deleted:    {}", stats.nodes_deleted);
    eprintln!("  Edges built:      {}", stats.edges_built);

    if !stats.languages.is_empty() {
        eprintln!("  Languages:");
        let mut langs: Vec<_> = stats.languages.iter().collect();
        langs.sort_by(|a, b| b.1.cmp(a.1));
        for (lang, count) in langs {
            eprintln!("    {}: {} files", lang, count);
        }
    }

    eprintln!("\nTiming:");
    eprintln!("  Discovery:   {}ms", stats.discover_ms);
    eprintln!("  Parsing:     {}ms", stats.parse_ms);
    eprintln!("  Embeddings:  {}ms", stats.embed_ms);
    eprintln!("  Store:       {}ms", stats.store_ms);
    eprintln!("  Graph:       {}ms", stats.graph_ms);
    eprintln!("  Total:       {}ms", stats.total_ms);

    crate::telemetry::capture("cli_index", serde_json::json!({
        "files_parsed": stats.files_parsed,
        "nodes_extracted": stats.nodes_extracted,
        "total_ms": stats.total_ms,
    }));

    // Memory sync after indexing
    sync_memory(&path, &data_dir, &config).await?;

    Ok(())
}

/// Run memory sync: extract from configured sources, dedupe, embed, store.
/// Prints summary to stderr. No-op if memory is disabled.
pub(crate) async fn sync_memory(
    repo_path: &std::path::Path,
    data_dir: &std::path::Path,
    config: &TrevecConfig,
) -> Result<()> {
    if !config.memory.enabled {
        return Ok(());
    }

    eprintln!("\nSyncing memory...");
    let lance_dir = data_dir.join("lance");
    let mut mem_store = MemoryStore::open(lance_dir.to_str().unwrap()).await?;

    let file_node_map = load_file_node_map(data_dir);
    let graph_path = data_dir.join("graph.bin");
    let mut graph = if graph_path.exists() {
        CodeGraph::load(&graph_path).unwrap_or_default()
    } else {
        CodeGraph::default()
    };

    let mut embedder = if config.memory.semantic {
        trevec_index::embedder::Embedder::new_with_model(
            Some(&config.embeddings.model),
            true,
            Some(data_dir.join("models")),
            None,
        )
        .ok()
    } else {
        None
    };

    let mem_stats = memory::ingest_memory(
        repo_path,
        data_dir,
        &config.memory,
        embedder.as_mut(),
        &mut mem_store,
        &mut graph,
        &file_node_map,
    )
    .await?;

    graph.save(&graph_path)?;

    eprintln!("Memory sync:");
    eprintln!("  Events ingested: {}", mem_stats.events_ingested);
    eprintln!("  Events skipped (empty): {}", mem_stats.events_skipped_empty);
    eprintln!("  Events skipped (dupe):  {}", mem_stats.events_skipped_dedupe);
    eprintln!("  Discussed edges: {}", mem_stats.discussed_edges_created);
    if mem_stats.cursor_extracted > 0 {
        eprintln!("  Cursor turns:    {}", mem_stats.cursor_extracted);
    }
    if mem_stats.claude_code_extracted > 0 {
        eprintln!("  Claude Code turns: {}", mem_stats.claude_code_extracted);
    }
    if mem_stats.codex_extracted > 0 {
        eprintln!("  Codex turns:     {}", mem_stats.codex_extracted);
    }

    Ok(())
}

fn load_file_node_map(data_dir: &std::path::Path) -> HashMap<String, Vec<String>> {
    let nodes_path = data_dir.join("nodes.json");
    let mut map: HashMap<String, Vec<String>> = HashMap::new();
    if let Ok(json) = std::fs::read_to_string(nodes_path) {
        if let Ok(nodes) = serde_json::from_str::<Vec<CodeNode>>(&json) {
            for node in nodes {
                map.entry(node.file_path.clone())
                    .or_default()
                    .push(node.id.clone());
            }
        }
    }
    map
}
