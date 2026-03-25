use anyhow::{Context, Result};
use notify::{Event, EventKind, RecursiveMode, Watcher};
use std::collections::HashMap;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::{Duration, Instant};
use trevec_core::model::CodeNode;
use trevec_core::TrevecConfig;
use trevec_index::graph::CodeGraph;
use trevec_index::memory;
use trevec_index::memory_store::MemoryStore;

const DEBOUNCE_MS: u64 = 800;

/// Run the file watcher: initial index, then re-index on file changes.
pub async fn run(path: PathBuf, data_dir: PathBuf, verbose: bool) -> Result<()> {
    let path = path.canonicalize().unwrap_or(path);
    let config = TrevecConfig::load(&data_dir);

    eprintln!("Running initial index...");
    let stats =
        trevec_index::ingest::ingest_with_config(
            &path,
            &data_dir,
            verbose,
            &config.index.exclude,
            Some(&config.embeddings.model),
            None,
        )
        .await?;
    eprintln!(
        "Initial index: {} files, {} nodes, {} edges ({}ms)",
        stats.files_discovered, stats.nodes_extracted, stats.edges_built, stats.total_ms
    );

    // Initial memory sync
    if config.memory.enabled {
        sync_memory(&path, &data_dir, &config).await;
    }

    let (tx, rx) = mpsc::channel::<Event>();
    let mut watcher = notify::recommended_watcher(move |res: Result<Event, notify::Error>| {
        if let Ok(event) = res {
            let _ = tx.send(event);
        }
    })
    .context("Failed to create file watcher")?;

    watcher
        .watch(&path, RecursiveMode::Recursive)
        .context("Failed to watch repository")?;

    eprintln!("Watching {} for changes (Ctrl+C to stop)...", path.display());

    let mut last_reindex = Instant::now();

    loop {
        match rx.recv_timeout(Duration::from_millis(100)) {
            Ok(event) => {
                if !is_relevant_event(&event, &data_dir) {
                    continue;
                }

                // Debounce: wait until events settle
                let mut changed_paths: Vec<PathBuf> = event.paths.clone();
                let debounce_deadline = Instant::now() + Duration::from_millis(DEBOUNCE_MS);

                while Instant::now() < debounce_deadline {
                    match rx.recv_timeout(Duration::from_millis(50)) {
                        Ok(ev) if is_relevant_event(&ev, &data_dir) => {
                            changed_paths.extend(ev.paths);
                        }
                        _ => {}
                    }
                }

                // Skip if we just re-indexed very recently
                if last_reindex.elapsed() < Duration::from_millis(DEBOUNCE_MS) {
                    continue;
                }

                changed_paths.sort();
                changed_paths.dedup();

                eprintln!("\nChanges detected ({} files):", changed_paths.len());
                for p in changed_paths.iter().take(5) {
                    let display = p.strip_prefix(&path).unwrap_or(p);
                    eprintln!("  {}", display.display());
                }
                if changed_paths.len() > 5 {
                    eprintln!("  ... and {} more", changed_paths.len() - 5);
                }

                eprintln!("Re-indexing...");
                match trevec_index::ingest::ingest_with_config(
                    &path,
                    &data_dir,
                    verbose,
                    &config.index.exclude,
                    Some(&config.embeddings.model),
                    None,
                )
                .await
                {
                    Ok(stats) => {
                        eprintln!(
                            "Re-index complete: {} parsed, {} unchanged, {} deleted ({}ms)",
                            stats.files_parsed,
                            stats.files_unchanged,
                            stats.files_deleted,
                            stats.total_ms
                        );
                    }
                    Err(e) => {
                        eprintln!("Re-index failed: {e:#}");
                    }
                }

                // Memory sync after re-index
                if config.memory.enabled {
                    sync_memory(&path, &data_dir, &config).await;
                }

                last_reindex = Instant::now();
            }
            Err(mpsc::RecvTimeoutError::Timeout) => continue,
            Err(mpsc::RecvTimeoutError::Disconnected) => break,
        }
    }

    Ok(())
}

async fn sync_memory(path: &Path, data_dir: &Path, config: &TrevecConfig) {
    let lance_dir = data_dir.join("lance");
    let mut mem_store = match MemoryStore::open(lance_dir.to_str().unwrap()).await {
        Ok(s) => s,
        Err(e) => {
            eprintln!("Memory store open failed: {e}");
            return;
        }
    };

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
            false,
            Some(data_dir.join("models")),
            None,
        )
        .ok()
    } else {
        None
    };

    match memory::ingest_memory(
        path,
        data_dir,
        &config.memory,
        embedder.as_mut(),
        &mut mem_store,
        &mut graph,
        &file_node_map,
    )
    .await
    {
        Ok(stats) if stats.events_ingested > 0 => {
            if let Err(e) = graph.save(&graph_path) {
                eprintln!("Failed to save graph after memory sync: {e}");
            }
            eprintln!(
                "Memory sync: {} ingested, {} skipped",
                stats.events_ingested, stats.events_skipped_dedupe
            );
        }
        Ok(_) => {}
        Err(e) => eprintln!("Memory sync failed: {e}"),
    }
}

fn load_file_node_map(data_dir: &Path) -> HashMap<String, Vec<String>> {
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

fn is_relevant_event(event: &Event, data_dir: &Path) -> bool {
    match event.kind {
        EventKind::Create(_) | EventKind::Modify(_) | EventKind::Remove(_) => {}
        _ => return false,
    }

    // Ignore changes inside the data dir itself
    let data_dir_canonical = data_dir.canonicalize().unwrap_or_else(|_| data_dir.to_path_buf());
    event.paths.iter().any(|p| {
        let p_canonical = p.canonicalize().unwrap_or_else(|_| p.clone());
        !p_canonical.starts_with(&data_dir_canonical)
    })
}
