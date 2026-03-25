use anyhow::{Context, Result};
use std::collections::HashMap;
use std::io::BufRead;
use std::path::PathBuf;

use trevec_core::model::CodeNode;
use trevec_core::{TokenBudget, TrevecConfig};
use trevec_index::embedder::Embedder;
use trevec_index::graph::CodeGraph;
use trevec_index::memory_store::MemoryStore;
use trevec_index::store::Store;
use trevec_retrieve::bundle::assemble_bundle;
use trevec_retrieve::expander::expand_graph;
use trevec_retrieve::search::{
    apply_file_path_boost, apply_literal_boost, apply_test_file_penalty,
    apply_test_fixture_penalty, extract_file_paths_from_query, filter_noncode_files, rrf_merge,
};

/// Run a retrieval query against the index.
pub async fn run(
    query_text: String,
    repo_path: PathBuf,
    data_dir: PathBuf,
    budget: Option<usize>,
    anchors: Option<usize>,
    json_output: bool,
    verbose: bool,
    gpu: bool,
) -> Result<()> {
    let repo_path = repo_path.canonicalize().unwrap_or(repo_path);
    let config = TrevecConfig::load(&data_dir);
    let budget = budget.unwrap_or(config.retrieval.budget);
    let anchors = anchors.unwrap_or(config.retrieval.anchors);

    // Auto re-index if files changed since last index (use GPU if requested)
    let device = if gpu { Some("cuda") } else { Some(config.embeddings.device.as_str()) }
        .filter(|d| *d != "cpu");
    if trevec_index::ingest::is_index_stale(&repo_path, &data_dir, &config.index.exclude) {
        eprintln!("Index is stale, re-indexing...");
        let stats = trevec_index::ingest::ingest_with_config(
            &repo_path,
            &data_dir,
            false,
            &config.index.exclude,
            Some(&config.embeddings.model),
            device,
        )
        .await?;
        eprintln!(
            "Re-indexed: {} parsed, {} unchanged ({}ms)",
            stats.files_parsed, stats.files_unchanged, stats.total_ms
        );
    }

    // Load the node index
    let nodes_path = data_dir.join("nodes.json");
    let nodes_json =
        std::fs::read_to_string(&nodes_path).context("No index found. Run `trevec index` first.")?;
    let all_nodes: Vec<CodeNode> =
        serde_json::from_str(&nodes_json).context("Failed to parse nodes index")?;

    let nodes_map: HashMap<String, CodeNode> = all_nodes
        .iter()
        .map(|n| (n.id.clone(), n.clone()))
        .collect();

    // Load the graph
    let graph_path = data_dir.join("graph.bin");
    let graph = CodeGraph::load(&graph_path).context("Failed to load graph")?;

    // Open the store
    let lance_dir = data_dir.join("lance");
    let store = Store::open(lance_dir.to_str().unwrap())
        .await
        .context("Failed to open store")?;

    // Embed the query
    let mut embedder =
        Embedder::new_with_model(Some(&config.embeddings.model), true, Some(data_dir.join("models")), None)
            .context("Failed to initialize embedder")?;
    let query_vec = embedder
        .embed(&query_text)
        .context("Failed to embed query")?;

    // Run parallel searches (FTS + vector)
    let search_limit = anchors * 8; // Fetch more candidates for RRF
    let (fts_results, vector_results) = tokio::join!(
        store.search_fts(&query_text, search_limit),
        store.search_vector(&query_vec, search_limit),
    );

    let fts_results = fts_results.context("FTS search failed")?;
    let vector_results = vector_results.context("Vector search failed")?;

    if verbose {
        eprintln!("FTS results: {}", fts_results.len());
        for (i, r) in fts_results.iter().enumerate().take(10) {
            if let Some(node) = nodes_map.get(&r.node_id) {
                eprintln!("  {}. {} ({})", i + 1, node.name, node.file_path);
            }
        }
        eprintln!("Vector results: {}", vector_results.len());
        for (i, r) in vector_results.iter().enumerate().take(10) {
            if let Some(node) = nodes_map.get(&r.node_id) {
                eprintln!(
                    "  {}. {} ({}) score={:.4}",
                    i + 1,
                    node.name,
                    node.file_path,
                    r.score
                );
            }
        }
    }

    // RRF merge
    let mut merged = rrf_merge(&fts_results, &vector_results, 60);

    // Pipeline: filter → literal boost → file path boost → test penalty
    filter_noncode_files(&mut merged, &nodes_map);
    apply_literal_boost(&mut merged, &nodes_map, &query_text);

    // File path extraction boost (stack traces, error messages)
    let extracted_paths = extract_file_paths_from_query(&query_text);
    if !extracted_paths.is_empty() {
        if verbose {
            eprintln!("Extracted file paths: {:?}", extracted_paths);
        }
        apply_file_path_boost(&mut merged, &nodes_map, &extracted_paths, 0.05);
    }

    apply_test_file_penalty(
        &mut merged,
        &nodes_map,
        config.retrieval.test_file_penalty,
        &config.retrieval.penalty_paths,
    );
    apply_test_fixture_penalty(&mut merged, &nodes_map);

    if verbose {
        eprintln!("\nMerged ranking:");
        for (i, r) in merged.iter().enumerate().take(10) {
            if let Some(node) = nodes_map.get(&r.node_id) {
                eprintln!(
                    "  {}. {} ({}) rrf={:.6}",
                    i + 1,
                    node.name,
                    node.file_path,
                    r.score
                );
            }
        }
    }

    // Select anchor nodes
    let anchor_ids: Vec<String> = merged
        .iter()
        .take(anchors)
        .map(|r| r.node_id.clone())
        .collect();

    // Token lookup function
    let token_fn = |id: &String| -> usize {
        nodes_map
            .get(id)
            .map(|n| n.span.estimated_tokens())
            .unwrap_or(0)
    };

    // Expand graph
    let mut token_budget = TokenBudget::new(budget);
    let included_ids = expand_graph(&graph, &anchor_ids, &mut token_budget, &token_fn, 3);

    if verbose {
        eprintln!(
            "\nExpanded to {} nodes, ~{} tokens used of {} budget",
            included_ids.len(),
            token_budget.used(),
            token_budget.total()
        );
    }

    // Assemble bundle
    let bundle = assemble_bundle(&query_text, &anchor_ids, &included_ids, &nodes_map, &repo_path)?;

    // Output
    if json_output {
        let json = serde_json::to_string_pretty(&bundle)?;
        println!("{}", json);
    } else {
        print!("{}", bundle.format_text());
    }

    // Memory search (if enabled)
    if config.memory.enabled {
        let lance_dir = data_dir.join("lance");
        if let Ok(mem_store) = MemoryStore::open(lance_dir.to_str().unwrap()).await {
            let mem_fts = mem_store
                .search_fts(&query_text, 10)
                .await
                .unwrap_or_default();

            let mem_vec = if !query_vec.is_empty() {
                mem_store
                    .search_vector(&query_vec, 10)
                    .await
                    .unwrap_or_default()
            } else {
                vec![]
            };

            let mut mem_merged = trevec_retrieve::search::rrf_merge(&mem_fts, &mem_vec, 60);

            // Recency boost
            let now = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64;

            let event_ids: Vec<String> =
                mem_merged.iter().take(5).map(|r| r.node_id.clone()).collect();
            let events = mem_store.get_events(&event_ids).await.unwrap_or_default();
            let event_map: HashMap<String, _> =
                events.iter().map(|e| (e.id.clone(), e)).collect();

            for result in &mut mem_merged {
                if let Some(event) = event_map.get(&result.node_id) {
                    let days_ago = ((now - event.created_at) as f64) / 86400.0;
                    let recency_boost = 1.0 / (1.0 + days_ago * 0.1);
                    result.score += recency_boost * 0.01;
                }
            }

            mem_merged.sort_by(|a, b| {
                b.score
                    .partial_cmp(&a.score)
                    .unwrap_or(std::cmp::Ordering::Equal)
            });

            if !events.is_empty() {
                if json_output {
                    // Already output above, skip
                } else {
                    println!("\n{}", "─".repeat(60));
                    println!("# History\n");
                    for r in mem_merged.iter().take(5) {
                        if let Some(event) = event_map.get(&r.node_id) {
                            let snippet = if event.content_redacted.len() > 200 {
                                format!("{}...", &event.content_redacted[..200])
                            } else {
                                event.content_redacted.clone()
                            };
                            let days_ago = (now - event.created_at) / 86400;
                            let time_label = if days_ago == 0 {
                                "today".to_string()
                            } else if days_ago == 1 {
                                "yesterday".to_string()
                            } else {
                                format!("{} days ago", days_ago)
                            };
                            println!(
                                "- [{}] ({}, {}) {}",
                                event.source, time_label, event.session_id, snippet
                            );
                        }
                    }
                    println!();
                }
            }
        }
    }

    Ok(())
}

/// Batch mode: load model/index once, read queries from stdin, output JSON per line.
/// Each line of stdin is a query. Each line of stdout is a ContextBundle JSON.
/// Timing per query is output to stderr.
pub async fn run_batch(
    repo_path: PathBuf,
    data_dir: PathBuf,
    budget: Option<usize>,
    anchors: Option<usize>,
    verbose: bool,
    gpu: bool,
) -> Result<()> {
    let repo_path = repo_path.canonicalize().unwrap_or(repo_path);
    let config = TrevecConfig::load(&data_dir);
    let budget = budget.unwrap_or(config.retrieval.budget);
    let anchors = anchors.unwrap_or(config.retrieval.anchors);

    // Auto re-index if stale (use GPU if requested)
    let device = if gpu { Some("cuda") } else { Some(config.embeddings.device.as_str()) }
        .filter(|d| *d != "cpu");
    if trevec_index::ingest::is_index_stale(&repo_path, &data_dir, &config.index.exclude) {
        eprintln!("Index is stale, re-indexing...");
        let stats = trevec_index::ingest::ingest_with_config(
            &repo_path,
            &data_dir,
            false,
            &config.index.exclude,
            Some(&config.embeddings.model),
            device,
        )
        .await?;
        eprintln!(
            "Re-indexed: {} parsed, {} unchanged ({}ms)",
            stats.files_parsed, stats.files_unchanged, stats.total_ms
        );
    }

    // Load everything once
    let nodes_path = data_dir.join("nodes.json");
    let nodes_json =
        std::fs::read_to_string(&nodes_path).context("No index found. Run `trevec index` first.")?;
    let all_nodes: Vec<CodeNode> =
        serde_json::from_str(&nodes_json).context("Failed to parse nodes index")?;
    let nodes_map: HashMap<String, CodeNode> = all_nodes
        .iter()
        .map(|n| (n.id.clone(), n.clone()))
        .collect();

    let graph_path = data_dir.join("graph.bin");
    let graph = CodeGraph::load(&graph_path).context("Failed to load graph")?;

    let lance_dir = data_dir.join("lance");
    let store = Store::open(lance_dir.to_str().unwrap())
        .await
        .context("Failed to open store")?;

    let mut embedder =
        Embedder::new_with_model(Some(&config.embeddings.model), true, Some(data_dir.join("models")), None)
            .context("Failed to initialize embedder")?;

    eprintln!("ready");

    // Read queries from stdin, one per line
    let stdin = std::io::stdin();
    for line in stdin.lock().lines() {
        let query_text = match line {
            Ok(l) => l,
            Err(_) => break,
        };
        let query_text = query_text.trim().to_string();
        if query_text.is_empty() {
            continue;
        }

        let start = std::time::Instant::now();

        // Embed query
        let query_vec = match embedder.embed(&query_text) {
            Ok(v) => v,
            Err(e) => {
                eprintln!("error: embed failed: {e}");
                println!("null");
                continue;
            }
        };

        // Parallel search (FTS + vector)
        let search_limit = anchors * 8;
        let (fts_results, vector_results) = tokio::join!(
            store.search_fts(&query_text, search_limit),
            store.search_vector(&query_vec, search_limit),
        );

        let fts_results = match fts_results {
            Ok(r) => r,
            Err(e) => {
                eprintln!("warn: FTS search failed: {e}");
                vec![]
            }
        };
        let vector_results = match vector_results {
            Ok(r) => r,
            Err(e) => {
                eprintln!("warn: vector search failed: {e}");
                vec![]
            }
        };

        // RRF merge
        let mut merged = rrf_merge(&fts_results, &vector_results, 60);

        // Pipeline: filter → literal boost → file path boost → test penalty
        filter_noncode_files(&mut merged, &nodes_map);
        apply_literal_boost(&mut merged, &nodes_map, &query_text);

        // File path extraction boost (stack traces, error messages)
        let extracted_paths = extract_file_paths_from_query(&query_text);
        if !extracted_paths.is_empty() {
            apply_file_path_boost(&mut merged, &nodes_map, &extracted_paths, 0.05);
        }

        apply_test_file_penalty(
            &mut merged,
            &nodes_map,
            config.retrieval.test_file_penalty,
            &config.retrieval.penalty_paths,
        );
        apply_test_fixture_penalty(&mut merged, &nodes_map);

        // Select anchors
        let anchor_ids: Vec<String> = merged
            .iter()
            .take(anchors)
            .map(|r| r.node_id.clone())
            .collect();

        // Expand graph
        let token_fn = |id: &String| -> usize {
            nodes_map
                .get(id)
                .map(|n| n.span.estimated_tokens())
                .unwrap_or(0)
        };
        let mut token_budget = TokenBudget::new(budget);
        let included_ids = expand_graph(&graph, &anchor_ids, &mut token_budget, &token_fn, 3);

        // Assemble bundle
        match assemble_bundle(&query_text, &anchor_ids, &included_ids, &nodes_map, &repo_path) {
            Ok(bundle) => {
                let elapsed = start.elapsed();
                if verbose {
                    eprintln!(
                        "query: {}ms, anchors={}, nodes={}, tokens={}",
                        elapsed.as_millis(),
                        anchor_ids.len(),
                        included_ids.len(),
                        bundle.total_estimated_tokens
                    );
                }
                // Compact JSON, one line per query
                let json = serde_json::to_string(&bundle)?;
                println!("{}", json);
            }
            Err(e) => {
                eprintln!("error: assemble failed: {e}");
                println!("null");
            }
        }
    }

    Ok(())
}
