use anyhow::{Context, Result};
use rayon::prelude::*;
use std::collections::{HashMap, HashSet};
use std::path::Path;
use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
use std::sync::Arc;
use std::time::Instant;

use trevec_core::model::{CodeNode, FileManifest, FileManifestEntry};
use trevec_core::compute_file_hash;
use trevec_parse::edges::build_edges;
use trevec_parse::extract::{
    extract_from_source, extract_markdown_sections, extract_references_from_source,
    ExtractionResult, Reference,
};
use trevec_parse::languages::language_for_extension;
use trevec_parse::walker::discover_files;

use crate::embedder::Embedder;
use crate::graph::CodeGraph;
use crate::store::Store;

/// Statistics from the ingestion pipeline.
#[derive(Debug, Default)]
pub struct IngestStats {
    pub files_discovered: usize,
    pub files_parsed: usize,
    pub files_skipped: usize,
    pub files_unchanged: usize,
    pub files_deleted: usize,
    pub nodes_extracted: usize,
    pub nodes_deleted: usize,
    pub edges_built: usize,
    pub languages: HashMap<String, usize>,
    pub discover_ms: u128,
    pub parse_ms: u128,
    pub embed_ms: u128,
    pub store_ms: u128,
    pub graph_ms: u128,
    pub total_ms: u128,
}

/// Result of processing a single file during parallel parsing.
enum FileOutcome {
    /// File skipped: unsupported extension or read error.
    Skipped,
    /// File unchanged since last index — carry forward manifest entry.
    Unchanged {
        relative_path: String,
        manifest_entry: FileManifestEntry,
        references: Vec<Reference>,
    },
    /// File successfully parsed — new/changed nodes extracted.
    Parsed {
        relative_path: String,
        file_hash: String,
        stale_ids: Vec<String>,
        nodes: Vec<CodeNode>,
        references: Vec<Reference>,
        language_name: String,
    },
    /// File read succeeded but extraction failed.
    /// Carries the old manifest entry (if any) so node ID linkage is preserved
    /// for cleanup on a future successful parse.
    ParseFailed {
        relative_path: String,
        old_manifest_entry: Option<FileManifestEntry>,
    },
}

fn needs_existing_nodes(files_unchanged: usize, carried_manifest_entries: usize) -> bool {
    files_unchanged > 0 || carried_manifest_entries > 0
}

/// Load the file manifest from disk.
pub fn load_manifest(data_dir: &Path) -> FileManifest {
    let manifest_path = data_dir.join("manifest.json");
    if manifest_path.exists() {
        match std::fs::read_to_string(&manifest_path) {
            Ok(content) => serde_json::from_str(&content).unwrap_or_default(),
            Err(_) => FileManifest::new(),
        }
    } else {
        FileManifest::new()
    }
}

/// Save the file manifest to disk.
fn save_manifest(data_dir: &Path, manifest: &FileManifest) -> Result<()> {
    let manifest_path = data_dir.join("manifest.json");
    let json = serde_json::to_string_pretty(manifest)
        .context("Failed to serialize manifest")?;
    std::fs::write(&manifest_path, json)
        .context("Failed to write manifest")?;
    Ok(())
}

/// Check whether the index is stale (files changed since last index).
///
/// Uses file modification times compared against the manifest's write time
/// for a fast O(n) stat-only check. Also detects deleted files.
pub fn is_index_stale(repo_path: &Path, data_dir: &Path, extra_excludes: &[String]) -> bool {
    let manifest_path = data_dir.join("manifest.json");
    let manifest_mtime = match std::fs::metadata(&manifest_path).and_then(|m| m.modified()) {
        Ok(t) => t,
        Err(_) => return true, // No manifest = no index
    };

    let manifest = load_manifest(data_dir);
    if manifest.is_empty() {
        return true;
    }

    // Check if any discovered source file is newer than the manifest
    let files = match discover_files(repo_path, extra_excludes) {
        Ok(f) => f,
        Err(_) => return true,
    };

    for file in &files {
        if let Ok(mtime) = std::fs::metadata(file).and_then(|m| m.modified()) {
            if mtime > manifest_mtime {
                return true;
            }
        }
    }

    // Check if any manifest entry's file was deleted
    for path in manifest.keys() {
        if !repo_path.join(path).exists() {
            return true;
        }
    }

    false
}

/// Run the full ingestion pipeline on a repository with incremental support.
pub async fn ingest(
    repo_path: &Path,
    data_dir: &Path,
    verbose: bool,
) -> Result<IngestStats> {
    ingest_with_config(repo_path, data_dir, verbose, &[], None, None).await
}

/// Run the full ingestion pipeline with user-provided exclude patterns
/// and optional embedding model name.
pub async fn ingest_with_config(
    repo_path: &Path,
    data_dir: &Path,
    verbose: bool,
    extra_excludes: &[String],
    embedding_model: Option<&str>,
    device: Option<&str>,
) -> Result<IngestStats> {
    let total_start = Instant::now();
    let mut stats = IngestStats::default();

    // Ensure data dir exists
    std::fs::create_dir_all(data_dir).context("Failed to create data directory")?;

    // Load previous manifest for incremental diffing
    let old_manifest = load_manifest(data_dir);

    // 1. Discover files
    let t = Instant::now();
    let files = discover_files(repo_path, extra_excludes).context("Failed to discover files")?;
    stats.files_discovered = files.len();
    stats.discover_ms = t.elapsed().as_millis();
    if verbose {
        tracing::info!(
            "Discovered {} files in {}ms",
            files.len(),
            stats.discover_ms
        );
    }

    // 2. Parse files with incremental diffing (parallel via rayon)
    let t = Instant::now();
    let mut new_nodes: Vec<CodeNode> = Vec::new();
    let mut all_references: Vec<Reference> = Vec::new();
    let mut new_manifest: FileManifest = FileManifest::new();
    let mut stale_node_ids: Vec<String> = Vec::new();
    let mut seen_files: std::collections::HashSet<String> = std::collections::HashSet::new();
    let mut carried_manifest_entries = 0usize;

    // Progress counter for parallel parsing
    let total_files = files.len();
    let progress = Arc::new(AtomicUsize::new(0));
    let progress_done = Arc::new(AtomicBool::new(false));
    let progress_handle = if !verbose && total_files > 0 {
        let progress = Arc::clone(&progress);
        let done = Arc::clone(&progress_done);
        Some(std::thread::spawn(move || {
            while !done.load(Ordering::Relaxed) {
                let count = progress.load(Ordering::Relaxed);
                eprint!("\rParsing files... {count}/{total_files}");
                std::thread::sleep(std::time::Duration::from_millis(100));
            }
            let count = progress.load(Ordering::Relaxed);
            eprint!("\rParsing files... {count}/{total_files}\n");
        }))
    } else {
        None
    };

    // Phase A: parallel file processing — each thread creates its own Parser/Query
    let outcomes: Vec<FileOutcome> = files
        .par_iter()
        .map(|file_path| {
            progress.fetch_add(1, Ordering::Relaxed);
            let ext = match file_path.extension().and_then(|e| e.to_str()) {
                Some(e) => e,
                None => return FileOutcome::Skipped,
            };

            let source = match std::fs::read(file_path) {
                Ok(s) => s,
                Err(e) => {
                    tracing::warn!("Failed to read {}: {}", file_path.display(), e);
                    return FileOutcome::Skipped;
                }
            };

            let relative_path = file_path
                .strip_prefix(repo_path)
                .unwrap_or(file_path)
                .to_string_lossy()
                .replace('\\', "/");

            let file_hash = compute_file_hash(&source);

            // Check for markdown files (no tree-sitter needed)
            let is_markdown = ext == "md" || ext == "mdx";

            let lang_config = if is_markdown {
                None
            } else {
                match language_for_extension(ext) {
                    Some(config) => Some(config),
                    None => return FileOutcome::Skipped,
                }
            };

            // Check if unchanged
            if let Some(old_entry) = old_manifest.get(&relative_path) {
                if old_entry.file_hash == file_hash {
                    let references = if let Some(ref lc) = lang_config {
                        extract_references_from_source(&relative_path, &source, lc)
                            .unwrap_or_default()
                    } else {
                        vec![]
                    };
                    return FileOutcome::Unchanged {
                        relative_path,
                        manifest_entry: old_entry.clone(),
                        references,
                    };
                }
            }

            // Collect stale IDs from old manifest if file changed
            let stale_ids = old_manifest
                .get(&relative_path)
                .map(|e| e.node_ids.clone())
                .unwrap_or_default();

            // New or changed file — parse it
            if is_markdown {
                let ExtractionResult { nodes, references } =
                    extract_markdown_sections(&relative_path, &source);
                FileOutcome::Parsed {
                    relative_path,
                    file_hash,
                    stale_ids,
                    nodes,
                    references,
                    language_name: "markdown".to_string(),
                }
            } else {
                let lang_config = lang_config.unwrap();
                match extract_from_source(&relative_path, &source, &lang_config) {
                    Ok(ExtractionResult { nodes, references }) => FileOutcome::Parsed {
                        relative_path,
                        file_hash,
                        stale_ids,
                        nodes,
                        references,
                        language_name: lang_config.name.to_string(),
                    },
                    Err(e) => {
                        tracing::warn!("Failed to extract from {}: {}", relative_path, e);
                        FileOutcome::ParseFailed {
                            relative_path: relative_path.clone(),
                            old_manifest_entry: old_manifest.get(&relative_path).cloned(),
                        }
                    }
                }
            }
        })
        .collect();

    // Stop progress display
    progress_done.store(true, Ordering::Relaxed);
    if let Some(handle) = progress_handle {
        let _ = handle.join();
    }

    // Phase B: sequential aggregation
    for outcome in outcomes {
        match outcome {
            FileOutcome::Skipped => {
                stats.files_skipped += 1;
            }
            FileOutcome::Unchanged {
                relative_path,
                manifest_entry,
                references,
            } => {
                seen_files.insert(relative_path.clone());
                stats.files_unchanged += 1;
                new_manifest.insert(relative_path, manifest_entry);
                all_references.extend(references);
            }
            FileOutcome::Parsed {
                relative_path,
                file_hash,
                stale_ids,
                nodes,
                references,
                language_name,
            } => {
                seen_files.insert(relative_path.clone());
                *stats.languages.entry(language_name).or_default() += 1;
                stats.files_parsed += 1;
                let node_ids: Vec<String> = nodes.iter().map(|n| n.id.clone()).collect();
                new_manifest.insert(
                    relative_path,
                    FileManifestEntry {
                        file_hash,
                        node_ids,
                    },
                );
                stale_node_ids.extend(stale_ids);
                new_nodes.extend(nodes);
                all_references.extend(references);
            }
            FileOutcome::ParseFailed { relative_path, old_manifest_entry } => {
                seen_files.insert(relative_path.clone());
                stats.files_skipped += 1;
                // Carry forward old manifest entry so node ID linkage survives
                // for cleanup on a future successful parse.
                if let Some(entry) = old_manifest_entry {
                    new_manifest.insert(relative_path, entry);
                    carried_manifest_entries += 1;
                }
            }
        }
    }

    // Detect deleted files: in old manifest but not seen
    for (old_path, old_entry) in &old_manifest {
        if !seen_files.contains(old_path) {
            stats.files_deleted += 1;
            stale_node_ids.extend(old_entry.node_ids.iter().cloned());
        }
    }

    stats.nodes_deleted = stale_node_ids.len();
    stats.nodes_extracted = new_nodes.len();
    stats.parse_ms = t.elapsed().as_millis();
    if verbose {
        tracing::info!(
            "Extracted {} nodes from {} files in {}ms (unchanged: {}, deleted: {})",
            new_nodes.len(),
            stats.files_parsed,
            stats.parse_ms,
            stats.files_unchanged,
            stats.files_deleted
        );
    }

    // 3. Build edges (from new nodes + references only)
    let t = Instant::now();
    // For edge building, we need all nodes (new + unchanged).
    // Load existing nodes for unchanged files separately so we can combine
    // them with embedded new_nodes later for serialization.
    let mut existing_nodes: Vec<CodeNode> = Vec::new();
    if needs_existing_nodes(stats.files_unchanged, carried_manifest_entries) {
        let nodes_path = data_dir.join("nodes.json");
        if nodes_path.exists() {
            if let Ok(content) = std::fs::read_to_string(&nodes_path) {
                if let Ok(prev_nodes) = serde_json::from_str::<Vec<CodeNode>>(&content) {
                    let new_node_ids: std::collections::HashSet<&str> =
                        new_nodes.iter().map(|n| n.id.as_str()).collect();
                    let stale_ids: std::collections::HashSet<&str> =
                        stale_node_ids.iter().map(|s| s.as_str()).collect();
                    for node in prev_nodes {
                        if !new_node_ids.contains(node.id.as_str())
                            && !stale_ids.contains(node.id.as_str())
                        {
                            existing_nodes.push(node);
                        }
                    }
                }
            }
        }
    }

    // Combine for edge building (embeddings not needed for edges)
    let mut all_nodes_for_edges: Vec<CodeNode> = Vec::with_capacity(new_nodes.len() + existing_nodes.len());
    all_nodes_for_edges.extend(new_nodes.iter().cloned());
    all_nodes_for_edges.extend_from_slice(&existing_nodes);
    let edges = build_edges(&all_nodes_for_edges, &all_references);
    stats.edges_built = edges.len();
    let edge_ms = t.elapsed().as_millis();
    if verbose {
        tracing::info!("Built {} edges in {}ms", edges.len(), edge_ms);
    }

    // 4. Compute embeddings (only for new/changed nodes)
    // Sort texts by length before batching to minimize ONNX padding waste.
    // ONNX pads all texts in a batch to the longest text's token count;
    // sorting groups similar lengths together, reducing wasted compute by ~75%.
    let t = Instant::now();
    if !new_nodes.is_empty() {
        let mut embedder =
            Embedder::new_with_model(embedding_model, true, Some(data_dir.join("models")), device)
                .context("Failed to initialize embedder")?;

        let mut indexed_texts: Vec<(usize, String)> = new_nodes
            .iter()
            .enumerate()
            .map(|(i, n)| (i, n.embedding_text()))
            .collect();
        indexed_texts.sort_by_key(|(_, text)| text.len());

        let sorted_texts: Vec<String> = indexed_texts.iter().map(|(_, t)| t.clone()).collect();
        let sorted_embeddings = embedder
            .embed_batch(&sorted_texts)
            .context("Failed to compute embeddings")?;

        for ((orig_idx, _), embedding) in indexed_texts.into_iter().zip(sorted_embeddings.into_iter()) {
            new_nodes[orig_idx].symbol_vec = Some(embedding);
        }
    }
    stats.embed_ms = t.elapsed().as_millis();
    if verbose {
        tracing::info!(
            "Computed embeddings for {} nodes in {}ms",
            new_nodes.len(),
            stats.embed_ms
        );
    }

    // 5. Update LanceDB store
    let t = Instant::now();
    let lance_dir = data_dir.join("lance");
    let mut store = Store::open(lance_dir.to_str().unwrap())
        .await
        .context("Failed to open store")?;

    // Delete stale nodes first
    if !stale_node_ids.is_empty() {
        store
            .delete_nodes(&stale_node_ids)
            .await
            .context("Failed to delete stale nodes")?;
        if verbose {
            tracing::info!("Deleted {} stale nodes from store", stale_node_ids.len());
        }
    }

    // Upsert new/changed nodes
    if !new_nodes.is_empty() {
        store
            .upsert_nodes(&new_nodes)
            .await
            .context("Failed to upsert nodes")?;
    }
    stats.store_ms = t.elapsed().as_millis();
    if verbose {
        tracing::info!(
            "Store updated ({} upserted, {} deleted) in {}ms",
            new_nodes.len(),
            stale_node_ids.len(),
            stats.store_ms
        );
    }

    // 6. Build and persist graph (rebuild from all edges, preserving memory edges)
    let t = Instant::now();
    let graph_path = data_dir.join("graph.bin");

    // Carry forward Discussed/Triggered edges from the previous graph so
    // memory links survive reindex, but drop edges pointing to removed code nodes.
    let mut memory_edges = if graph_path.exists() {
        CodeGraph::load(&graph_path)
            .map(|old| old.extract_memory_edges())
            .unwrap_or_default()
    } else {
        vec![]
    };
    let live_code_node_ids: HashSet<&str> = all_nodes_for_edges
        .iter()
        .map(|node| node.id.as_str())
        .collect();
    let before_filter = memory_edges.len();
    memory_edges.retain(|edge| live_code_node_ids.contains(edge.dst_id.as_str()));
    let dropped_memory_edges = before_filter.saturating_sub(memory_edges.len());

    let mut graph = CodeGraph::new();
    graph.build_from_edges(&edges);
    graph.build_from_edges(&memory_edges);
    graph.save(&graph_path).context("Failed to save graph")?;
    stats.graph_ms = t.elapsed().as_millis();
    if verbose {
        tracing::info!(
            "Built graph ({} nodes, {} edges) and saved in {}ms",
            graph.node_count(),
            graph.edge_count(),
            stats.graph_ms
        );
        if dropped_memory_edges > 0 {
            tracing::info!(
                "Dropped {} stale memory edges targeting removed code nodes",
                dropped_memory_edges
            );
        }
    }

    // Save the full node list (new + unchanged) for retrieval.
    // Strip symbol_vec before writing — embeddings are already in LanceDB and
    // would bloat nodes.json (~1.5 KB per node for 384-dim f32 vectors).
    let nodes_path = data_dir.join("nodes.json");
    let all_nodes: Vec<CodeNode> = new_nodes
        .iter()
        .chain(existing_nodes.iter())
        .map(|n| {
            let mut node = n.clone();
            node.symbol_vec = None;
            node
        })
        .collect();
    let nodes_json =
        serde_json::to_string(&all_nodes).context("Failed to serialize nodes")?;
    std::fs::write(&nodes_path, nodes_json).context("Failed to write nodes file")?;

    // Save updated manifest
    save_manifest(data_dir, &new_manifest)?;

    stats.total_ms = total_start.elapsed().as_millis();
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_needs_existing_nodes_when_carrying_manifest_entries() {
        assert!(needs_existing_nodes(0, 1));
        assert!(needs_existing_nodes(1, 0));
        assert!(!needs_existing_nodes(0, 0));
    }

    /// End-to-end integration test: create a temp repo, ingest, verify outputs.
    /// Ignored by default because it requires the embedding model to be downloaded.
    #[tokio::test]
    #[ignore]
    async fn test_ingest_basic() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = tmp.path().join("repo");
        let src = repo.join("src");
        fs::create_dir_all(&src).unwrap();

        fs::write(
            src.join("main.rs"),
            b"fn main() { println!(\"hello\"); }\nfn helper() -> i32 { 42 }\n",
        )
        .unwrap();

        let data_dir = tmp.path().join(".trevec");

        let stats = ingest(&repo, &data_dir, false).await.unwrap();
        assert_eq!(stats.files_parsed, 1);
        assert!(stats.nodes_extracted >= 2);
        assert!(data_dir.join("lance").exists());
        assert!(data_dir.join("graph.bin").exists());
        assert!(data_dir.join("nodes.json").exists());
        assert!(data_dir.join("manifest.json").exists());

        // Verify manifest was created
        let manifest: FileManifest =
            serde_json::from_str(&fs::read_to_string(data_dir.join("manifest.json")).unwrap())
                .unwrap();
        assert_eq!(manifest.len(), 1);
        assert!(manifest.contains_key("src/main.rs"));

        // Second run should be fully incremental
        let stats2 = ingest(&repo, &data_dir, false).await.unwrap();
        assert_eq!(stats2.files_unchanged, 1);
        assert_eq!(stats2.files_parsed, 0);
        assert_eq!(stats2.nodes_extracted, 0);
    }
}
