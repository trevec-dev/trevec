pub mod claude_code;
pub mod codex;
pub mod cursor;
pub mod gc;
pub mod meta;
pub mod scrub;

use std::path::Path;

use anyhow::Result;
use trevec_core::config::MemoryConfig;
use trevec_core::model::{Confidence, Edge, EdgeType, MemoryEvent};

use crate::embedder::Embedder;
use crate::graph::CodeGraph;
use crate::memory_store::MemoryStore;

/// A raw turn extracted from a chat source before processing.
#[derive(Debug, Clone)]
pub struct RawTurn {
    pub source: String,
    pub session_id: String,
    pub turn_index: u32,
    pub timestamp: i64,
    pub role: String,
    pub user_prompt: String,
    pub assistant_text: String,
    pub tool_calls: Vec<String>,
    pub files_touched: Vec<String>,
}

/// Convert a RawTurn into a MemoryEvent by scrubbing secrets and computing IDs.
pub fn raw_turn_to_event(turn: &RawTurn, repo_id: &str, max_event_chars: usize) -> MemoryEvent {
    let (user_scrubbed, _) = scrub::scrub(&turn.user_prompt);
    let (assistant_scrubbed, _) = scrub::scrub(&turn.assistant_text);

    // Deterministic ID: blake3(source + session_id + turn_index)
    // repo_id is NOT included — each repo has its own LanceDB instance, so collisions are impossible.
    let id_input = format!(
        "{}|{}|{}",
        turn.source, turn.session_id, turn.turn_index
    );
    let id = blake3::hash(id_input.as_bytes()).to_hex()[..32].to_string();

    // Content hash for dedupe: blake3(user_prompt + assistant_text) on raw content
    let content_input = format!("{}{}", turn.user_prompt, turn.assistant_text);
    let content_hash = blake3::hash(content_input.as_bytes()).to_hex()[..32].to_string();

    // Build combined content (truncated)
    let combined = if user_scrubbed.is_empty() {
        assistant_scrubbed.clone()
    } else if assistant_scrubbed.is_empty() {
        user_scrubbed.clone()
    } else {
        format!("User: {}\nAssistant: {}", user_scrubbed, assistant_scrubbed)
    };
    let content_redacted = if combined.len() > max_event_chars {
        combined[..max_event_chars].to_string()
    } else {
        combined
    };

    // Build BM25 text: source + files + tool_calls + scrubbed content
    let mut bm25_parts = vec![turn.source.clone()];
    if !turn.files_touched.is_empty() {
        bm25_parts.push(turn.files_touched.join(" "));
    }
    if !turn.tool_calls.is_empty() {
        bm25_parts.push(turn.tool_calls.join(" "));
    }
    bm25_parts.push(content_redacted.clone());
    let bm25_text = bm25_parts.join(" ");

    let created_at = if turn.timestamp == 0 {
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_secs() as i64
    } else {
        turn.timestamp
    };

    MemoryEvent {
        id,
        repo_id: repo_id.to_string(),
        source: turn.source.clone(),
        session_id: turn.session_id.clone(),
        turn_index: turn.turn_index,
        role: turn.role.clone(),
        event_type: "turn".to_string(),
        content_redacted,
        content_hash,
        created_at,
        importance: 0,
        pinned: false,
        files_touched: turn.files_touched.clone(),
        tool_calls: turn.tool_calls.clone(),
        bm25_text,
        symbol_vec: None,
    }
}

/// Statistics from a memory ingest run.
#[derive(Debug, Default)]
pub struct IngestMemoryStats {
    pub events_ingested: usize,
    pub events_skipped_empty: usize,
    pub events_skipped_dedupe: usize,
    pub total_redactions: usize,
    pub discussed_edges_created: usize,
    pub cursor_extracted: usize,
    pub claude_code_extracted: usize,
    pub codex_extracted: usize,
}

/// Run the memory ingestion pipeline: extract from enabled sources, scrub, embed, store.
pub async fn ingest_memory(
    repo_path: &Path,
    data_dir: &Path,
    config: &MemoryConfig,
    embedder: Option<&mut Embedder>,
    store: &mut MemoryStore,
    graph: &mut CodeGraph,
    file_node_map: &std::collections::HashMap<String, Vec<String>>,
) -> Result<IngestMemoryStats> {
    let mut stats = IngestMemoryStats::default();
    let mut all_meta = meta::MemoryMeta::load(data_dir);

    let repo_id = blake3::hash(repo_path.to_string_lossy().as_bytes()).to_hex()[..32].to_string();
    let mut all_turns: Vec<RawTurn> = Vec::new();

    // Claude Code extractor
    if config.claude_code.enabled && config.sources.iter().any(|s| s == "claude_code") {
        let extractor = config
            .claude_code
            .projects_dir
            .as_ref()
            .map(|p| claude_code::ClaudeCodeExtractor::new(p.into()))
            .or_else(claude_code::ClaudeCodeExtractor::detect);

        if let Some(ext) = extractor {
            match ext.extract(repo_path, &mut all_meta) {
                Ok(turns) => {
                    tracing::info!("Claude Code: extracted {} turns", turns.len());
                    stats.claude_code_extracted = turns.len();
                    all_turns.extend(turns);
                }
                Err(e) => tracing::warn!("Claude Code extraction failed: {e}"),
            }
        }
    }

    // Cursor extractor
    if config.cursor.enabled && config.sources.iter().any(|s| s == "cursor") {
        let extractor = config
            .cursor
            .db_path
            .as_ref()
            .map(|p| cursor::CursorExtractor::new(p.into()))
            .or_else(cursor::CursorExtractor::detect);

        if let Some(ext) = extractor {
            match ext.extract(repo_path, &mut all_meta) {
                Ok(turns) => {
                    tracing::info!("Cursor: extracted {} turns", turns.len());
                    stats.cursor_extracted = turns.len();
                    all_turns.extend(turns);
                }
                Err(e) => tracing::warn!("Cursor extraction failed: {e}"),
            }
        }
    }

    // Codex extractor
    if config.codex.enabled && config.sources.iter().any(|s| s == "codex") {
        let extractor = config
            .codex
            .sessions_dir
            .as_ref()
            .map(|p| codex::CodexExtractor::new(p.into()))
            .or_else(codex::CodexExtractor::detect);

        if let Some(ext) = extractor {
            match ext.extract(repo_path, &mut all_meta) {
                Ok(turns) => {
                    tracing::info!("Codex: extracted {} turns", turns.len());
                    stats.codex_extracted = turns.len();
                    all_turns.extend(turns);
                }
                Err(e) => tracing::warn!("Codex extraction failed: {e}"),
            }
        }
    }

    if all_turns.is_empty() {
        all_meta.save(data_dir)?;
        return Ok(stats);
    }

    // Convert to events
    let mut events: Vec<MemoryEvent> = all_turns
        .iter()
        .map(|t| raw_turn_to_event(t, &repo_id, config.max_event_chars))
        .collect();

    // Filter out events with empty content (e.g. tool-only turns with no user/assistant text)
    let before_empty_filter = events.len();
    events.retain(|e| !e.content_redacted.trim().is_empty());
    stats.events_skipped_empty = before_empty_filter - events.len();

    // Content hash dedupe: check all content_hashes against the store globally
    let candidate_hashes: Vec<String> = events.iter().map(|e| e.content_hash.clone()).collect();
    let existing_hashes = store
        .find_existing_content_hashes_for_repo(&candidate_hashes, Some(&repo_id))
        .await
        .unwrap_or_default();

    events.retain(|e| {
        if existing_hashes.contains(&e.content_hash) {
            stats.events_skipped_dedupe += 1;
            false
        } else {
            true
        }
    });

    if events.is_empty() {
        all_meta.save(data_dir)?;
        return Ok(stats);
    }

    // Embed events if semantic is enabled
    if config.semantic {
        if let Some(emb) = embedder {
            let texts: Vec<String> = events.iter().map(|e| e.bm25_text.clone()).collect();
            match emb.embed_batch(&texts) {
                Ok(vecs) => {
                    for (event, vec) in events.iter_mut().zip(vecs.into_iter()) {
                        event.symbol_vec = Some(vec);
                    }
                }
                Err(e) => tracing::warn!("Memory embedding failed: {e}"),
            }
        }
    }

    // Upsert into store
    stats.events_ingested = events.len();
    store.upsert_events(&events).await?;

    // Create Discussed edges: event → code nodes with matching file_path.
    // Normalize extracted paths to repo-relative form before matching,
    // since extractors may produce absolute paths.
    let repo_prefix = repo_path.to_string_lossy();
    for event in &events {
        for file in &event.files_touched {
            let rel = file
                .strip_prefix(repo_prefix.as_ref())
                .or_else(|| file.strip_prefix(&format!("{}/", repo_prefix)))
                .unwrap_or(file)
                .trim_start_matches('/');
            if let Some(node_ids) = file_node_map.get(rel) {
                for node_id in node_ids {
                    let edge = Edge {
                        src_id: event.id.clone(),
                        dst_id: node_id.clone(),
                        edge_type: EdgeType::Discussed,
                        confidence: Confidence::Likely,
                    };
                    graph.add_edge(&edge);
                    stats.discussed_edges_created += 1;
                }
            }
        }
    }

    // Count redactions
    for turn in &all_turns {
        let (_, user_count) = scrub::scrub(&turn.user_prompt);
        let (_, asst_count) = scrub::scrub(&turn.assistant_text);
        stats.total_redactions += user_count + asst_count;
    }

    all_meta.save(data_dir)?;
    Ok(stats)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_raw_turn() -> RawTurn {
        RawTurn {
            source: "claude_code".to_string(),
            session_id: "sess_abc".to_string(),
            turn_index: 0,
            timestamp: 1700000000,
            role: "user".to_string(),
            user_prompt: "How does authentication work?".to_string(),
            assistant_text: "The auth module uses JWT tokens.".to_string(),
            tool_calls: vec!["Read".to_string()],
            files_touched: vec!["src/auth.rs".to_string()],
        }
    }

    #[test]
    fn test_raw_turn_to_event_id_deterministic() {
        let turn = make_raw_turn();
        let e1 = raw_turn_to_event(&turn, "repo123", 8000);
        let e2 = raw_turn_to_event(&turn, "repo123", 8000);
        assert_eq!(e1.id, e2.id);
        assert_eq!(e1.id.len(), 32);
    }

    #[test]
    fn test_raw_turn_to_event_content_hash() {
        let turn = make_raw_turn();
        let event = raw_turn_to_event(&turn, "repo123", 8000);
        assert_eq!(event.content_hash.len(), 32);

        // Different content → different hash
        let mut turn2 = turn.clone();
        turn2.user_prompt = "Different question".to_string();
        let event2 = raw_turn_to_event(&turn2, "repo123", 8000);
        assert_ne!(event.content_hash, event2.content_hash);
    }

    #[test]
    fn test_raw_turn_to_event_bm25_text() {
        let turn = make_raw_turn();
        let event = raw_turn_to_event(&turn, "repo123", 8000);
        assert!(event.bm25_text.contains("claude_code"));
        assert!(event.bm25_text.contains("src/auth.rs"));
        assert!(event.bm25_text.contains("Read"));
        assert!(event.bm25_text.contains("authentication"));
    }

    #[test]
    fn test_raw_turn_to_event_truncation() {
        let mut turn = make_raw_turn();
        turn.user_prompt = "x".repeat(10000);
        let event = raw_turn_to_event(&turn, "repo123", 100);
        assert!(event.content_redacted.len() <= 100);
    }

    #[test]
    fn test_raw_turn_to_event_scrubs_secrets() {
        let mut turn = make_raw_turn();
        turn.user_prompt = "Use key sk-abcdefghijklmnopqrstuvwxyz please".to_string();
        let event = raw_turn_to_event(&turn, "repo123", 8000);
        assert!(!event.content_redacted.contains("sk-abcdefghijklmnopqrstuvwxyz"));
        assert!(event.content_redacted.contains("<REDACTED_API_KEY>"));
    }

    #[test]
    fn test_raw_turn_to_event_fields() {
        let turn = make_raw_turn();
        let event = raw_turn_to_event(&turn, "repo123", 8000);
        assert_eq!(event.source, "claude_code");
        assert_eq!(event.session_id, "sess_abc");
        assert_eq!(event.turn_index, 0);
        assert_eq!(event.role, "user");
        assert_eq!(event.event_type, "turn");
        assert_eq!(event.repo_id, "repo123");
        assert_eq!(event.created_at, 1700000000);
        assert!(!event.pinned);
        assert_eq!(event.importance, 0);
        assert_eq!(event.files_touched, vec!["src/auth.rs"]);
        assert_eq!(event.tool_calls, vec!["Read"]);
        assert!(event.symbol_vec.is_none());
    }

    #[test]
    fn test_raw_turn_to_event_zero_timestamp_fallback() {
        let mut turn = make_raw_turn();
        turn.timestamp = 0;
        let event = raw_turn_to_event(&turn, "repo123", 8000);
        // Should have been replaced with current wall-clock time
        assert!(event.created_at > 0, "zero timestamp should be replaced with now()");
    }

    #[test]
    fn test_empty_content_event_detected() {
        // An event with empty content_redacted should be filterable
        let mut turn = make_raw_turn();
        turn.user_prompt = String::new();
        turn.assistant_text = String::new();
        let event = raw_turn_to_event(&turn, "repo123", 8000);
        assert!(
            event.content_redacted.trim().is_empty(),
            "event with blank user_prompt and assistant_text should have empty content_redacted"
        );
    }
}
