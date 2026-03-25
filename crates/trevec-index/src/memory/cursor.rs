use anyhow::{Context, Result};
use rusqlite::Connection;
use std::collections::{HashMap, HashSet};
use std::path::{Path, PathBuf};

use super::meta::MemoryMeta;
use super::RawTurn;

/// Extracts chat history from Cursor's state.vscdb SQLite database.
pub struct CursorExtractor {
    db_path: PathBuf,
}

impl CursorExtractor {
    pub fn new(db_path: PathBuf) -> Self {
        Self { db_path }
    }

    /// Auto-detect Cursor's state.vscdb path for the current platform.
    pub fn detect() -> Option<Self> {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .ok()?;
        let home = PathBuf::from(home);

        let path = if cfg!(target_os = "macos") {
            home.join("Library/Application Support/Cursor/User/globalStorage/state.vscdb")
        } else if cfg!(target_os = "linux") {
            home.join(".config/Cursor/User/globalStorage/state.vscdb")
        } else if cfg!(target_os = "windows") {
            home.join("AppData/Roaming/Cursor/User/globalStorage/state.vscdb")
        } else {
            return None;
        };

        if path.is_file() {
            Some(Self::new(path))
        } else {
            None
        }
    }

    /// Extract turns for the given repo path.
    /// Copies the database to a temp file to avoid WAL lock conflicts.
    pub fn extract(
        &self,
        repo_path: &Path,
        meta: &mut MemoryMeta,
    ) -> Result<Vec<RawTurn>> {
        // Snapshot: copy DB + WAL sidecars to avoid lock conflicts while still
        // seeing the most recent committed rows.
        let tmp_dir = tempfile::tempdir().context("Failed to create temp dir for Cursor snapshot")?;
        let snapshot_path = tmp_dir.path().join("state.vscdb");
        snapshot_sqlite_with_sidecars(&self.db_path, &snapshot_path)?;

        let conn = Connection::open_with_flags(
            &snapshot_path,
            rusqlite::OpenFlags::SQLITE_OPEN_READ_ONLY,
        )
        .context("Failed to open Cursor snapshot")?;

        let repo_str = repo_path.to_string_lossy();
        let last_rowid = meta.cursor_last_rowid.unwrap_or(0);

        let mut turns = Vec::new();
        let mut changed_composer_ids: HashSet<String> = HashSet::new();

        // Query composerData keys, only rows newer than our watermark.
        // These rows indicate changed/created composer sessions.
        let mut composer_stmt = conn
            .prepare(
                "SELECT rowid, key FROM cursorDiskKV \
                 WHERE key LIKE 'composerData:%' AND rowid > ?1 \
                 ORDER BY rowid",
            )
            .context("Failed to prepare Cursor query")?;

        let composer_rows = composer_stmt
            .query_map([last_rowid], |row| {
                let rowid: i64 = row.get(0)?;
                let key: String = row.get(1)?;
                Ok((rowid, key))
            })
            .context("Failed to query Cursor DB")?;

        let mut max_rowid = last_rowid;
        for row in composer_rows {
            let (rowid, key) = match row {
                Ok(r) => r,
                Err(_) => continue,
            };

            if rowid > max_rowid {
                max_rowid = rowid;
            }

            let composer_id = key.strip_prefix("composerData:").unwrap_or(&key).to_string();
            changed_composer_ids.insert(composer_id);
        }

        // Query bubble rows newer than watermark. Cursor v13 stores turns in
        // `bubbleId:{composer_id}:{bubble_id}` rows even when composerData lacks
        // inline `bubbles`/`bubbleIds`.
        let mut bubble_stmt = conn
            .prepare(
                "SELECT rowid, key FROM cursorDiskKV \
                 WHERE key LIKE 'bubbleId:%' AND rowid > ?1 \
                 ORDER BY rowid",
            )
            .context("Failed to prepare Cursor bubble query")?;

        let bubble_rows = bubble_stmt
            .query_map([last_rowid], |row| {
                let rowid: i64 = row.get(0)?;
                let key: String = row.get(1)?;
                Ok((rowid, key))
            })
            .context("Failed to query Cursor bubble rows")?;

        for row in bubble_rows {
            let (rowid, key) = match row {
                Ok(r) => r,
                Err(_) => continue,
            };

            if rowid > max_rowid {
                max_rowid = rowid;
            }

            let Some(rest) = key.strip_prefix("bubbleId:") else {
                continue;
            };
            let Some((composer_id, _bubble_id)) = rest.split_once(':') else {
                continue;
            };
            changed_composer_ids.insert(composer_id.to_string());
        }

        // Rebuild turns for each changed composer from current DB state.
        let mut sessions_with_composer_turns: HashSet<String> = HashSet::new();
        let mut composer_ids: Vec<String> = changed_composer_ids.into_iter().collect();
        composer_ids.sort();
        for composer_id in composer_ids {
            let composer_key = format!("composerData:{composer_id}");
            let composer_value = match conn.query_row(
                "SELECT value FROM cursorDiskKV WHERE key = ?1",
                [&composer_key],
                |row| row.get::<_, String>(0),
            ) {
                Ok(v) => v,
                Err(_) => continue,
            };

            let parsed: serde_json::Value = match serde_json::from_str(&composer_value) {
                Ok(v) => v,
                Err(_) => continue,
            };

            let extracted = extract_composer_turns(&conn, &parsed, &composer_id, &repo_str);
            if !extracted.is_empty() {
                sessions_with_composer_turns.insert(composer_id.clone());
            }
            turns.extend(extracted);
        }

        // Cursor can also persist conversation items as `agentKv:blob:*` rows.
        // Process these rows too so sessions that don't emit composer/bubble
        // changes in this window are still ingested.
        let mut agent_stmt = conn
            .prepare(
                "SELECT rowid, value FROM cursorDiskKV \
                 WHERE key LIKE 'agentKv:blob:%' AND rowid > ?1 \
                 ORDER BY rowid",
            )
            .context("Failed to prepare Cursor agentKv query")?;
        let agent_rows = agent_stmt
            .query_map([last_rowid], |row| {
                let rowid: i64 = row.get(0)?;
                let value: Vec<u8> = row.get(1)?;
                Ok((rowid, value))
            })
            .context("Failed to query Cursor agentKv rows")?;

        let mut agent_payloads: Vec<(i64, Vec<u8>)> = Vec::new();
        for row in agent_rows {
            let (rowid, value) = match row {
                Ok(r) => r,
                Err(_) => continue,
            };
            if rowid > max_rowid {
                max_rowid = rowid;
            }
            agent_payloads.push((rowid, value));
        }

        let fallback = extract_agent_kv_turns(
            &conn,
            &agent_payloads,
            &repo_str,
            &sessions_with_composer_turns,
        );
        turns.extend(fallback);

        // Always advance the watermark when we scanned new rows.  The
        // layered relevance check (marker → per-bubble paths → composer-level)
        // is the proper fix for "filtered-and-lost" turns.  Holding the
        // watermark back would cause repeated O(n) rescans of the entire
        // global tail in multi-repo Cursor DBs where some repos yield no
        // matching turns.
        if max_rowid > last_rowid {
            meta.cursor_last_rowid = Some(max_rowid);
        }

        Ok(turns)
    }
}

fn snapshot_sqlite_with_sidecars(src: &Path, dst: &Path) -> Result<()> {
    std::fs::copy(src, dst).with_context(|| {
        format!("Failed to snapshot Cursor DB from {}", src.display())
    })?;

    // Copy SQLite WAL/SHM files when present so snapshot reads include recent
    // pages that may not be checkpointed into the main DB file yet.
    for suffix in ["-wal", "-shm"] {
        let mut src_sidecar = src.as_os_str().to_os_string();
        src_sidecar.push(suffix);
        let src_sidecar = PathBuf::from(src_sidecar);
        if !src_sidecar.exists() {
            continue;
        }

        let mut dst_sidecar = dst.as_os_str().to_os_string();
        dst_sidecar.push(suffix);
        let dst_sidecar = PathBuf::from(dst_sidecar);

        std::fs::copy(&src_sidecar, &dst_sidecar).with_context(|| {
            format!(
                "Failed to snapshot Cursor sidecar from {}",
                src_sidecar.display()
            )
        })?;
    }

    Ok(())
}

fn extract_agent_kv_turns(
    conn: &Connection,
    agent_rows: &[(i64, Vec<u8>)],
    repo_str: &str,
    skip_sessions: &HashSet<String>,
) -> Vec<RawTurn> {
    let mut turns = Vec::new();
    let mut relevance_cache: HashMap<String, bool> = HashMap::new();
    let now = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs() as i64;

    // Pre-compute the marker so we can detect it in raw payload text.
    let repo_marker = compute_repo_id_marker(repo_str);
    let repo_hash = compute_repo_id_hash(repo_str);
    let repo_basename = repo_basename_lower(repo_str);

    for (rowid, raw_value) in agent_rows {
        // Agent KV stores a mix of JSON and binary blobs.
        let value = String::from_utf8_lossy(raw_value);
        if !value.trim_start().starts_with('{') {
            continue;
        }

        let parsed: serde_json::Value = match serde_json::from_str(&value) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let role = match parsed.get("role").and_then(|r| r.as_str()) {
            Some(r) => r,
            None => continue,
        };
        if role == "tool" {
            continue;
        }

        let (mut text, tool_calls) = extract_agent_kv_content(&parsed);
        if role == "user" {
            text = normalize_cursor_user_text(&text);
        }
        if text.trim().is_empty() {
            continue;
        }

        let session_id = lookup_session_for_rowid(conn, *rowid);
        if let Some(session_id) = &session_id {
            if skip_sessions.contains(session_id) {
                continue;
            }
        }

        // Check for the trevec repo marker in raw payload text — this
        // catches tool responses stored as agentKv rows, not just bubbles.
        let has_marker = value.contains(&repo_marker) || value.contains(&repo_hash);
        let has_tool_repo_signal = if let Some(base) = &repo_basename {
            let value_lc = value.to_ascii_lowercase();
            value_lc.contains(&format!("mcp_{}-trevec", base))
                || (value_lc.contains(base) && value_lc.contains("trevec"))
        } else {
            false
        };

        // Prefer composer/session repo filtering when we can resolve a
        // session, but fall back to checking whether this payload itself
        // mentions the current repo path or contains our marker.
        let payload_mentions_repo =
            has_marker || has_tool_repo_signal || json_mentions_repo(&parsed, repo_str);
        let is_relevant = if let Some(session_id) = &session_id {
            let session_relevant = *relevance_cache
                .entry(session_id.clone())
                .or_insert_with(|| is_session_relevant_to_repo(conn, session_id, repo_str));
            session_relevant || payload_mentions_repo
        } else {
            payload_mentions_repo
        };
        if !is_relevant {
            continue;
        }

        // Use SQLite rowid for a stable, monotonic turn index to avoid ID
        // collisions across incremental sync runs.
        let turn_index = if *rowid <= 0 {
            0
        } else {
            u32::try_from(*rowid).unwrap_or(u32::MAX)
        };

        let (user_prompt, assistant_text) = if role == "user" {
            (text, String::new())
        } else {
            (String::new(), text)
        };

        let session_id =
            session_id.unwrap_or_else(|| format!("cursor_row_{}", rowid));

        turns.push(RawTurn {
            source: "cursor".to_string(),
            session_id,
            turn_index,
            timestamp: now,
            role: role.to_string(),
            user_prompt,
            assistant_text,
            tool_calls,
            files_touched: Vec::new(),
        });
    }

    turns
}

fn extract_agent_kv_content(
    parsed: &serde_json::Value,
) -> (String, Vec<String>) {
    let mut text_parts: Vec<String> = Vec::new();
    let mut tool_calls: Vec<String> = Vec::new();

    if let Some(content) = parsed.get("content") {
        if let Some(content_str) = content.as_str() {
            text_parts.push(content_str.to_string());
        } else if let Some(blocks) = content.as_array() {
            for block in blocks {
                let kind = block.get("type").and_then(|t| t.as_str()).unwrap_or("");
                match kind {
                    "text" => {
                        if let Some(t) = block.get("text").and_then(|t| t.as_str()) {
                            if !t.trim().is_empty() {
                                text_parts.push(t.to_string());
                            }
                        }
                    }
                    "tool_use" => {
                        if let Some(name) = block.get("name").and_then(|n| n.as_str()) {
                            tool_calls.push(name.to_string());
                        }
                    }
                    _ => {}
                }
            }
        }
    }

    if text_parts.is_empty() {
        if let Some(t) = parsed.get("text").and_then(|t| t.as_str()) {
            text_parts.push(t.to_string());
        }
    }

    (text_parts.join("\n"), tool_calls)
}

fn is_cursor_system_metadata_text(text: &str) -> bool {
    let trimmed = text.trim_start();
    trimmed.starts_with("<open_and_recently_viewed_files>")
        || trimmed.starts_with("<user_info>")
}

fn normalize_cursor_user_text(text: &str) -> String {
    if !is_cursor_system_metadata_text(text) {
        return text.trim().to_string();
    }

    if let Some(query) = extract_tag_block(text, "user_query") {
        return query;
    }

    String::new()
}

fn extract_tag_block(text: &str, tag: &str) -> Option<String> {
    let open = format!("<{}>", tag);
    let close = format!("</{}>", tag);
    let start = text.find(&open)?;
    let end = text[start..].find(&close)?;
    let content = &text[start + open.len()..start + end];
    let trimmed = content.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_string())
    }
}

fn json_mentions_repo(value: &serde_json::Value, repo_str: &str) -> bool {
    if repo_str.is_empty() {
        return false;
    }

    match value {
        serde_json::Value::String(s) => s.contains(repo_str),
        serde_json::Value::Array(arr) => arr.iter().any(|v| json_mentions_repo(v, repo_str)),
        serde_json::Value::Object(obj) => {
            obj.values().any(|v| json_mentions_repo(v, repo_str))
        }
        _ => false,
    }
}

fn is_session_relevant_to_repo(
    conn: &Connection,
    session_id: &str,
    repo_str: &str,
) -> bool {
    let composer_key = format!("composerData:{session_id}");
    let composer_value: String = match conn.query_row(
        "SELECT value FROM cursorDiskKV WHERE key = ?1",
        [&composer_key],
        |row| row.get(0),
    ) {
        Ok(v) => v,
        Err(_) => return false,
    };
    let parsed: serde_json::Value = match serde_json::from_str(&composer_value) {
        Ok(v) => v,
        Err(_) => return false,
    };
    // Use the full layered check (marker + per-bubble + composer-level).
    is_composer_relevant_to_repo(conn, session_id, &parsed, repo_str)
}

fn lookup_session_for_rowid(conn: &Connection, rowid: i64) -> Option<String> {
    // Prefer nearest bubble row because these are tightly associated with the
    // active composer/session and tend to interleave with agentKv rows.
    if let Ok(key) = conn.query_row(
        "SELECT key FROM cursorDiskKV \
         WHERE key LIKE 'bubbleId:%' AND rowid <= ?1 \
         ORDER BY rowid DESC LIMIT 1",
        [rowid],
        |row| row.get::<_, String>(0),
    ) {
        if let Some(rest) = key.strip_prefix("bubbleId:") {
            if let Some((composer_id, _)) = rest.split_once(':') {
                return Some(composer_id.to_string());
            }
        }
    }

    // Fallback to nearest composerData row when no bubble exists.
    let key = conn
        .query_row(
            "SELECT key FROM cursorDiskKV \
             WHERE key LIKE 'composerData:%' AND rowid <= ?1 \
             ORDER BY rowid DESC LIMIT 1",
            [rowid],
            |row| row.get::<_, String>(0),
        )
        .ok()?;
    Some(key.strip_prefix("composerData:").unwrap_or(&key).to_string())
}

/// Extract turns from a composerData JSON blob (handles both v1 and v2 schema).
fn extract_composer_turns(
    conn: &Connection,
    data: &serde_json::Value,
    composer_id: &str,
    repo_str: &str,
) -> Vec<RawTurn> {
    let mut turns = Vec::new();

    // Layered relevance check (v13-compatible)
    if !is_composer_relevant_to_repo(conn, composer_id, data, repo_str) {
        return turns;
    }

    // Extract files touched from context
    let files_touched = extract_files_from_context(data);

    // Try v1 schema (inline bubbles array)
    if let Some(bubbles) = data.get("bubbles").and_then(|b| b.as_array()) {
        extract_from_bubbles(bubbles, composer_id, &files_touched, &mut turns);
        return turns;
    }

    // Try v2 schema (separate bubbleIds)
    if let Some(bubble_ids) = data.get("bubbleIds").and_then(|b| b.as_array()) {
        let mut bubbles = Vec::new();
        for bid in bubble_ids {
            if let Some(bid_str) = bid.as_str() {
                let key = format!("bubbleId:{}:{}", composer_id, bid_str);
                if let Ok(value) = conn.query_row(
                    "SELECT value FROM cursorDiskKV WHERE key = ?1",
                    [&key],
                    |row| row.get::<_, String>(0),
                ) {
                    if let Ok(bubble) = serde_json::from_str::<serde_json::Value>(&value) {
                        bubbles.push(bubble);
                    }
                }
            }
        }
        extract_from_bubbles(&bubbles, composer_id, &files_touched, &mut turns);
    }

    // Cursor v13 fallback: bubbles may only be stored as separate
    // `bubbleId:{composer_id}:{bubble_id}` rows.
    if turns.is_empty() {
        let mut bubbles = Vec::new();
        let key_prefix = format!("bubbleId:{composer_id}:%");
        if let Ok(mut stmt) = conn.prepare(
            "SELECT value FROM cursorDiskKV \
             WHERE key LIKE ?1 \
             ORDER BY rowid",
        ) {
            if let Ok(rows) = stmt.query_map([&key_prefix], |row| row.get::<_, String>(0)) {
                for row in rows {
                    let value = match row {
                        Ok(v) => v,
                        Err(_) => continue,
                    };
                    if let Ok(bubble) = serde_json::from_str::<serde_json::Value>(&value) {
                        bubbles.push(bubble);
                    }
                }
            }
        }
        if !bubbles.is_empty() {
            extract_from_bubbles(&bubbles, composer_id, &files_touched, &mut turns);
        }
    }

    turns
}

/// Extract turns from a bubbles array (either v1 inline or v2 fetched).
fn extract_from_bubbles(
    bubbles: &[serde_json::Value],
    session_id: &str,
    files_touched: &[String],
    turns: &mut Vec<RawTurn>,
) {
    let mut current_user = String::new();
    let mut turn_index: u32 = 0;

    for bubble in bubbles {
        let bubble_type = bubble.get("type").and_then(|t| t.as_u64()).unwrap_or(0);
        let text = bubble
            .get("text")
            .and_then(|t| t.as_str())
            .unwrap_or("")
            .to_string();

        let created_at = bubble
            .get("createdAt")
            .map(|t| {
                if let Some(v) = t.as_i64() {
                    v
                } else if let Some(s) = t.as_str() {
                    // Try plain integer first, then ISO 8601 / RFC 3339
                    s.parse::<i64>().unwrap_or_else(|_| parse_iso8601_timestamp(s))
                } else {
                    0
                }
            })
            .unwrap_or(0);
        // Cursor timestamps are often milliseconds. If parsing fails or is
        // missing, fall back to "now" so retention/ordering still works.
        let timestamp = if created_at == 0 {
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64
        } else if created_at > 1_000_000_000_000 {
            created_at / 1000
        } else {
            created_at
        };

        // Extract tool calls from assistant bubbles. Cursor stores this as
        // either an array or a single object depending on version/capability.
        let tool_calls: Vec<String> = if bubble_type == 2 {
            if let Some(td) = bubble.get("toolFormerData") {
                if let Some(arr) = td.as_array() {
                    arr.iter()
                        .filter_map(|t| t.get("name").and_then(|n| n.as_str()))
                        .map(|s| s.to_string())
                        .collect()
                } else if let Some(name) = td.get("name").and_then(|n| n.as_str()) {
                    vec![name.to_string()]
                } else {
                    vec![]
                }
            } else {
                vec![]
            }
        } else {
            vec![]
        };

        match bubble_type {
            1 => {
                // User message
                current_user = text;
            }
            2 => {
                // Assistant message → emit turn
                if text.trim().is_empty() && current_user.trim().is_empty() {
                    continue;
                }
                turns.push(RawTurn {
                    source: "cursor".to_string(),
                    session_id: session_id.to_string(),
                    turn_index,
                    timestamp,
                    role: "user".to_string(),
                    user_prompt: current_user.clone(),
                    assistant_text: text,
                    tool_calls,
                    files_touched: files_touched.to_vec(),
                });
                turn_index += 1;
                current_user.clear();
            }
            _ => {}
        }
    }
}

/// Compute the canonical `trevec:repo_id:{hash}` marker for a given repo path.
fn compute_repo_id_marker(repo_str: &str) -> String {
    format!("trevec:repo_id:{}", compute_repo_id_hash(repo_str))
}

fn compute_repo_id_hash(repo_str: &str) -> String {
    let hash = blake3::hash(repo_str.as_bytes()).to_hex();
    hash[..32].to_string()
}

fn repo_basename_lower(repo_str: &str) -> Option<String> {
    let base = Path::new(repo_str).file_name()?.to_str()?;
    let trimmed = base.trim();
    if trimmed.is_empty() {
        None
    } else {
        Some(trimmed.to_ascii_lowercase())
    }
}

/// Layered relevance check for a composer session.
///
/// 1. Composer-level fields — fast, no extra DB queries.
/// 2. Single pass over `bubbleId:*` rows — check raw text for trevec marker,
///    then parse JSON and check per-bubble file paths.
/// 3. Default false — no signal means don't import (avoids cross-contamination).
fn is_composer_relevant_to_repo(
    conn: &Connection,
    composer_id: &str,
    data: &serde_json::Value,
    repo_str: &str,
) -> bool {
    if repo_str.is_empty() {
        return false;
    }

    // Layer 1: Composer-level check (fast — no extra DB work).
    if is_relevant_to_repo(data, repo_str) {
        return true;
    }

    // Layer 2: Single pass over bubble rows — check trevec marker on raw
    // string first (cheap), then parse JSON for per-bubble file paths.
    let repo_marker = compute_repo_id_marker(repo_str);
    let key_prefix = format!("bubbleId:{composer_id}:%");
    if let Ok(mut stmt) = conn.prepare(
        "SELECT value FROM cursorDiskKV WHERE key LIKE ?1",
    ) {
        if let Ok(rows) = stmt.query_map([&key_prefix], |row| row.get::<_, String>(0)) {
            for row in rows {
                let value = match row {
                    Ok(v) => v,
                    Err(_) => continue,
                };
                // Marker check on raw text (avoids JSON parse cost).
                if value.contains(&repo_marker) {
                    return true;
                }
                // Per-bubble file paths — v13 stores context per bubble.
                if let Ok(bubble) = serde_json::from_str::<serde_json::Value>(&value) {
                    if bubble_mentions_repo_path(&bubble, repo_str) {
                        return true;
                    }
                }
            }
        }
    }

    // Layer 3: Default false — no association signal.
    false
}

/// Check if a bubble JSON value contains file paths under the repo.
fn bubble_mentions_repo_path(bubble: &serde_json::Value, repo_str: &str) -> bool {
    // Check bubble-level context.fileSelections
    if let Some(ctx) = bubble.get("context") {
        if let Some(sels) = ctx.get("fileSelections").and_then(|s| s.as_array()) {
            for sel in sels {
                if let Some(path) = sel.get("path").and_then(|p| p.as_str()) {
                    if path.contains(repo_str) || repo_str.contains(path) {
                        return true;
                    }
                }
                // uri field (some versions)
                if let Some(uri) = sel.get("uri").and_then(|u| u.as_str()) {
                    if uri.contains(repo_str) {
                        return true;
                    }
                }
            }
        }
    }
    // Also check bubble text content for file paths mentioning the repo
    if let Some(text) = bubble.get("text").and_then(|t| t.as_str()) {
        if text.contains(repo_str) {
            return true;
        }
    }
    false
}

/// Parse an ISO 8601 / RFC 3339 timestamp string to unix seconds.
/// Returns 0 on failure so the caller can fall back to "now".
pub(super) fn parse_iso8601_timestamp(s: &str) -> i64 {
    // Handle common formats: "2024-01-15T10:30:00Z", "2024-01-15T10:30:00.000Z",
    // "2024-01-15T10:30:00+00:00", "2024-01-15T10:30:00.000+00:00"
    //
    // Strategy: strip fractional seconds, parse date/time parts manually.
    let s = s.trim();
    if s.len() < 19 {
        return 0;
    }

    // Split at 'T' or ' '
    let (date_part, rest) = if let Some(pos) = s.find('T').or_else(|| s.find(' ')) {
        (&s[..pos], &s[pos + 1..])
    } else {
        return 0;
    };

    // Parse date: YYYY-MM-DD
    let date_parts: Vec<&str> = date_part.split('-').collect();
    if date_parts.len() != 3 {
        return 0;
    }
    let year: i64 = date_parts[0].parse().unwrap_or(0);
    let month: i64 = date_parts[1].parse().unwrap_or(0);
    let day: i64 = date_parts[2].parse().unwrap_or(0);
    if year < 1970 || !(1..=12).contains(&month) || !(1..=31).contains(&day) {
        return 0;
    }

    // Parse time: HH:MM:SS[.frac][Z|+HH:MM|-HH:MM]
    // Strip timezone suffix to get time portion
    let (time_core, tz_offset_secs) = if rest.ends_with('Z') || rest.ends_with('z') {
        (&rest[..rest.len() - 1], 0_i64)
    } else if let Some(plus_pos) = rest.rfind('+') {
        if plus_pos > 2 {
            let tz = &rest[plus_pos + 1..];
            let offset = parse_tz_offset(tz);
            (&rest[..plus_pos], offset)
        } else {
            (rest, 0)
        }
    } else if let Some(minus_pos) = rest.rfind('-') {
        if minus_pos > 2 {
            let tz = &rest[minus_pos + 1..];
            let offset = -parse_tz_offset(tz);
            (&rest[..minus_pos], offset)
        } else {
            (rest, 0)
        }
    } else {
        (rest, 0)
    };

    // Strip fractional seconds
    let time_str = if let Some(dot_pos) = time_core.find('.') {
        &time_core[..dot_pos]
    } else {
        time_core
    };

    let time_parts: Vec<&str> = time_str.split(':').collect();
    if time_parts.len() != 3 {
        return 0;
    }
    let hour: i64 = time_parts[0].parse().unwrap_or(0);
    let min: i64 = time_parts[1].parse().unwrap_or(0);
    let sec: i64 = time_parts[2].parse().unwrap_or(0);

    // Convert to unix timestamp using a simplified calculation
    // (doesn't handle leap seconds, but that's fine for our purposes)
    let days = days_since_epoch(year, month, day);
    days * 86400 + hour * 3600 + min * 60 + sec - tz_offset_secs
}

fn parse_tz_offset(tz: &str) -> i64 {
    // Parse "HH:MM" or "HHMM" or "HH"
    let digits: String = tz.chars().filter(|c| c.is_ascii_digit()).collect();
    match digits.len() {
        1..=2 => digits.parse::<i64>().unwrap_or(0) * 3600,
        3..=4 => {
            let h: i64 = digits[..2].parse().unwrap_or(0);
            let m: i64 = digits[2..].parse().unwrap_or(0);
            h * 3600 + m * 60
        }
        _ => 0,
    }
}

fn days_since_epoch(year: i64, month: i64, day: i64) -> i64 {
    // Compute days from 1970-01-01 to the given date.
    // Uses the algorithm from https://howardhinnant.github.io/date_algorithms.html
    let y = if month <= 2 { year - 1 } else { year };
    let era = if y >= 0 { y } else { y - 399 } / 400;
    let yoe = (y - era * 400) as u64;
    let m = month as u64;
    let doy = (153 * (if m > 2 { m - 3 } else { m + 9 }) + 2) / 5 + day as u64 - 1;
    let doe = yoe * 365 + yoe / 4 - yoe / 100 + doy;
    (era * 146097 + doe as i64) - 719468
}

/// Check if a composer session is relevant to the given repo path.
fn is_relevant_to_repo(data: &serde_json::Value, repo_str: &str) -> bool {
    // Check relevantFiles / context.fileSelections for matching paths
    if let Some(files) = data.get("relevantFiles").and_then(|f| f.as_array()) {
        for f in files {
            if let Some(path) = f.as_str().or_else(|| f.get("path").and_then(|p| p.as_str())) {
                if path.contains(repo_str) || repo_str.contains(path) {
                    return true;
                }
            }
        }
    }
    // Also check context
    if let Some(ctx) = data.get("context") {
        if let Some(sels) = ctx.get("fileSelections").and_then(|s| s.as_array()) {
            for sel in sels {
                if let Some(path) = sel.get("path").and_then(|p| p.as_str()) {
                    if path.contains(repo_str) || repo_str.contains(path) {
                        return true;
                    }
                }
            }
        }
    }
    false
}

/// Extract file paths from Cursor composer context.
fn extract_files_from_context(data: &serde_json::Value) -> Vec<String> {
    let mut files = Vec::new();

    if let Some(arr) = data.get("relevantFiles").and_then(|f| f.as_array()) {
        for f in arr {
            if let Some(path) = f.as_str().or_else(|| f.get("path").and_then(|p| p.as_str())) {
                if !files.contains(&path.to_string()) {
                    files.push(path.to_string());
                }
            }
        }
    }
    if let Some(ctx) = data.get("context") {
        if let Some(sels) = ctx.get("fileSelections").and_then(|s| s.as_array()) {
            for sel in sels {
                if let Some(path) = sel.get("path").and_then(|p| p.as_str()) {
                    if !files.contains(&path.to_string()) {
                        files.push(path.to_string());
                    }
                }
            }
        }
    }

    files
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_from_bubbles_v1() {
        let bubbles = vec![
            serde_json::json!({
                "type": 1,
                "text": "How does auth work?",
                "createdAt": 1700000000000_i64
            }),
            serde_json::json!({
                "type": 2,
                "text": "Auth uses JWT tokens.",
                "createdAt": 1700000001000_i64,
                "toolFormerData": [{"name": "codeblock"}]
            }),
        ];

        let mut turns = Vec::new();
        extract_from_bubbles(&bubbles, "comp_1", &["src/auth.rs".to_string()], &mut turns);

        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].source, "cursor");
        assert_eq!(turns[0].user_prompt, "How does auth work?");
        assert_eq!(turns[0].assistant_text, "Auth uses JWT tokens.");
        assert_eq!(turns[0].tool_calls, vec!["codeblock"]);
        assert_eq!(turns[0].files_touched, vec!["src/auth.rs"]);
        assert_eq!(turns[0].timestamp, 1700000001); // assistant bubble timestamp
    }

    #[test]
    fn test_is_relevant_to_repo() {
        let data = serde_json::json!({
            "relevantFiles": ["src/auth.rs", "src/main.rs"]
        });
        assert!(is_relevant_to_repo(&data, "src/auth"));

        let data_no_match = serde_json::json!({
            "relevantFiles": ["other/file.py"]
        });
        assert!(!is_relevant_to_repo(&data_no_match, "src/auth"));
    }

    #[test]
    fn test_extract_files_from_context() {
        let data = serde_json::json!({
            "relevantFiles": ["src/a.rs", "src/b.rs"],
            "context": {
                "fileSelections": [
                    {"path": "src/c.rs"}
                ]
            }
        });
        let files = extract_files_from_context(&data);
        assert_eq!(files, vec!["src/a.rs", "src/b.rs", "src/c.rs"]);
    }

    #[test]
    fn test_cursor_db_roundtrip() {
        // Create an in-memory SQLite DB with Cursor-like schema
        let conn = Connection::open_in_memory().unwrap();
        conn.execute(
            "CREATE TABLE cursorDiskKV (key TEXT PRIMARY KEY, value TEXT)",
            [],
        )
        .unwrap();

        let composer_data = serde_json::json!({
            "bubbles": [
                {"type": 1, "text": "Explain the login flow", "createdAt": 1700000000000_i64},
                {"type": 2, "text": "The login uses OAuth2.", "createdAt": 1700000001000_i64}
            ],
            "relevantFiles": ["src/login.rs"]
        });

        conn.execute(
            "INSERT INTO cursorDiskKV (key, value) VALUES (?1, ?2)",
            ["composerData:comp_1", &serde_json::to_string(&composer_data).unwrap()],
        )
        .unwrap();

        let turns = extract_composer_turns(
            &conn,
            &composer_data,
            "comp_1",
            "src/login",
        );

        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].user_prompt, "Explain the login flow");
        assert_eq!(turns[0].assistant_text, "The login uses OAuth2.");
    }

    #[test]
    fn test_cursor_watermark_incremental() {
        // Verify that extract() only fetches rows with rowid > cursor_last_rowid
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("state.vscdb");

        let conn = Connection::open(&db_path).unwrap();
        conn.execute(
            "CREATE TABLE cursorDiskKV (key TEXT PRIMARY KEY, value TEXT)",
            [],
        )
        .unwrap();

        // Insert first composer
        let data1 = serde_json::json!({
            "bubbles": [
                {"type": 1, "text": "First question", "createdAt": 1700000000000_i64},
                {"type": 2, "text": "First answer", "createdAt": 1700000001000_i64}
            ],
            "relevantFiles": ["/tmp/repo/src/a.rs"]
        });
        conn.execute(
            "INSERT INTO cursorDiskKV (key, value) VALUES (?1, ?2)",
            ["composerData:comp_1", &serde_json::to_string(&data1).unwrap()],
        )
        .unwrap();

        // Get the rowid of the first insert
        let rowid1: i64 = conn
            .query_row(
                "SELECT rowid FROM cursorDiskKV WHERE key = 'composerData:comp_1'",
                [],
                |row| row.get(0),
            )
            .unwrap();

        // Insert second composer
        let data2 = serde_json::json!({
            "bubbles": [
                {"type": 1, "text": "Second question", "createdAt": 1700000002000_i64},
                {"type": 2, "text": "Second answer", "createdAt": 1700000003000_i64}
            ],
            "relevantFiles": ["/tmp/repo/src/b.rs"]
        });
        conn.execute(
            "INSERT INTO cursorDiskKV (key, value) VALUES (?1, ?2)",
            ["composerData:comp_2", &serde_json::to_string(&data2).unwrap()],
        )
        .unwrap();
        drop(conn);

        let extractor = CursorExtractor::new(db_path);
        let repo = Path::new("/tmp/repo");

        // First extract: should get both composers
        let mut meta = MemoryMeta::default();
        let turns = extractor.extract(repo, &mut meta).unwrap();
        assert_eq!(turns.len(), 2);
        assert!(meta.cursor_last_rowid.is_some());

        // Second extract with watermark: set to rowid of first insert
        // Only the second composer (rowid > rowid1) should be returned
        let mut meta2 = MemoryMeta::default();
        meta2.cursor_last_rowid = Some(rowid1);
        let turns2 = extractor.extract(repo, &mut meta2).unwrap();
        assert_eq!(turns2.len(), 1);
        assert_eq!(turns2[0].user_prompt, "Second question");
    }

    #[test]
    fn test_extract_composer_turns_fallback_to_bubble_rows() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute(
            "CREATE TABLE cursorDiskKV (key TEXT PRIMARY KEY, value TEXT)",
            [],
        )
        .unwrap();

        // Composer has no inline bubbles/bubbleIds (Cursor v13 shape).
        let composer_data = serde_json::json!({
            "_v": 13,
            "context": { "fileSelections": [{"path": "/tmp/repo/src/auth.rs"}] }
        });
        conn.execute(
            "INSERT INTO cursorDiskKV (key, value) VALUES (?1, ?2)",
            ["composerData:comp_1", &serde_json::to_string(&composer_data).unwrap()],
        )
        .unwrap();

        let user_bubble = serde_json::json!({
            "type": 1,
            "text": "How does login work?",
            "createdAt": 1700000000000_i64
        });
        let asst_bubble = serde_json::json!({
            "type": 2,
            "text": "Login uses OAuth and JWT.",
            "createdAt": 1700000001000_i64
        });
        conn.execute(
            "INSERT INTO cursorDiskKV (key, value) VALUES (?1, ?2)",
            [
                "bubbleId:comp_1:user_1",
                &serde_json::to_string(&user_bubble).unwrap(),
            ],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO cursorDiskKV (key, value) VALUES (?1, ?2)",
            [
                "bubbleId:comp_1:assistant_1",
                &serde_json::to_string(&asst_bubble).unwrap(),
            ],
        )
        .unwrap();

        let turns = extract_composer_turns(&conn, &composer_data, "comp_1", "src/auth");
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].user_prompt, "How does login work?");
        assert_eq!(turns[0].assistant_text, "Login uses OAuth and JWT.");
        assert_eq!(turns[0].files_touched, vec!["/tmp/repo/src/auth.rs"]);
    }

    #[test]
    fn test_extract_from_bubbles_toolformer_object() {
        let bubbles = vec![
            serde_json::json!({
                "type": 1,
                "text": "Please inspect auth",
                "createdAt": 1700000000000_i64
            }),
            serde_json::json!({
                "type": 2,
                "text": "",
                "createdAt": 1700000001000_i64,
                "toolFormerData": {"name": "read_file_v2"}
            }),
        ];

        let mut turns = Vec::new();
        extract_from_bubbles(&bubbles, "comp_2", &[], &mut turns);

        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].tool_calls, vec!["read_file_v2"]);
        assert_eq!(turns[0].user_prompt, "Please inspect auth");
    }

    #[test]
    fn test_cursor_extract_detects_bubble_only_updates_after_watermark() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("state.vscdb");

        let conn = Connection::open(&db_path).unwrap();
        conn.execute(
            "CREATE TABLE cursorDiskKV (key TEXT PRIMARY KEY, value TEXT)",
            [],
        )
        .unwrap();

        let composer_data = serde_json::json!({
            "_v": 13,
            "context": { "fileSelections": [{"path": "/tmp/repo/src/auth.rs"}] }
        });
        conn.execute(
            "INSERT INTO cursorDiskKV (key, value) VALUES (?1, ?2)",
            ["composerData:comp_3", &serde_json::to_string(&composer_data).unwrap()],
        )
        .unwrap();

        let composer_rowid: i64 = conn
            .query_row(
                "SELECT rowid FROM cursorDiskKV WHERE key='composerData:comp_3'",
                [],
                |row| row.get(0),
            )
            .unwrap();

        // Add bubbles AFTER composer row.
        let user_bubble = serde_json::json!({
            "type": 1,
            "text": "Auth question",
            "createdAt": 1700000000000_i64
        });
        let asst_bubble = serde_json::json!({
            "type": 2,
            "text": "Auth answer",
            "createdAt": 1700000001000_i64
        });
        conn.execute(
            "INSERT INTO cursorDiskKV (key, value) VALUES (?1, ?2)",
            [
                "bubbleId:comp_3:user_1",
                &serde_json::to_string(&user_bubble).unwrap(),
            ],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO cursorDiskKV (key, value) VALUES (?1, ?2)",
            [
                "bubbleId:comp_3:assistant_1",
                &serde_json::to_string(&asst_bubble).unwrap(),
            ],
        )
        .unwrap();
        drop(conn);

        let extractor = CursorExtractor::new(db_path);
        let mut meta = MemoryMeta::default();
        meta.cursor_last_rowid = Some(composer_rowid);

        let turns = extractor.extract(Path::new("/tmp/repo"), &mut meta).unwrap();
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].assistant_text, "Auth answer");
    }

    #[test]
    fn test_snapshot_copies_wal_sidecars() {
        let dir = tempfile::tempdir().unwrap();
        let src = dir.path().join("state.vscdb");
        let dst = dir.path().join("snapshot.vscdb");

        std::fs::write(&src, b"db").unwrap();
        std::fs::write(format!("{}-wal", src.display()), b"wal").unwrap();
        std::fs::write(format!("{}-shm", src.display()), b"shm").unwrap();

        snapshot_sqlite_with_sidecars(&src, &dst).unwrap();

        assert_eq!(std::fs::read(&dst).unwrap(), b"db");
        assert_eq!(
            std::fs::read(format!("{}-wal", dst.display())).unwrap(),
            b"wal"
        );
        assert_eq!(
            std::fs::read(format!("{}-shm", dst.display())).unwrap(),
            b"shm"
        );
    }

    #[test]
    fn test_extract_agent_kv_turns_fallback_json() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute(
            "CREATE TABLE cursorDiskKV (key TEXT PRIMARY KEY, value BLOB)",
            [],
        )
        .unwrap();

        conn.execute(
            "INSERT INTO cursorDiskKV (key, value) VALUES (?1, ?2)",
            [
                "composerData:comp_x",
                &serde_json::to_string(&serde_json::json!({
                    "_v": 13,
                    "context": { "fileSelections": [{"path": "/tmp/repo/src/main.rs"}] }
                }))
                .unwrap(),
            ],
        )
        .unwrap();

        // Metadata-style user blob should be ignored.
        let user_meta = serde_json::json!({
            "role": "user",
            "content": [{"type": "text", "text": "<open_and_recently_viewed_files>\n..."}]
        })
        .to_string()
        .into_bytes();
        conn.execute(
            "INSERT INTO cursorDiskKV (key, value) VALUES (?1, ?2)",
            rusqlite::params!["agentKv:blob:user_1", user_meta],
        )
        .unwrap();

        // Assistant blob should be extracted.
        let assistant = serde_json::json!({
            "role": "assistant",
            "content": [{"type": "text", "text": "TREVEC_AGENTKV_MARKER"}]
        })
        .to_string()
        .into_bytes();
        conn.execute(
            "INSERT INTO cursorDiskKV (key, value) VALUES (?1, ?2)",
            rusqlite::params!["agentKv:blob:assistant_1", assistant],
        )
        .unwrap();

        let rows = vec![
            (2_i64, conn
                .query_row(
                    "SELECT value FROM cursorDiskKV WHERE key='agentKv:blob:user_1'",
                    [],
                    |row| row.get::<_, Vec<u8>>(0),
                )
                .unwrap()),
            (3_i64, conn
                .query_row(
                    "SELECT value FROM cursorDiskKV WHERE key='agentKv:blob:assistant_1'",
                    [],
                    |row| row.get::<_, Vec<u8>>(0),
                )
                .unwrap()),
        ];

        let turns = extract_agent_kv_turns(&conn, &rows, "/tmp/repo", &HashSet::new());
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].assistant_text, "TREVEC_AGENTKV_MARKER");
        assert_eq!(turns[0].source, "cursor");
        assert_eq!(turns[0].turn_index, 3);
    }

    #[test]
    fn test_extract_agent_kv_turns_user_query_from_metadata_is_kept() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute(
            "CREATE TABLE cursorDiskKV (key TEXT PRIMARY KEY, value BLOB)",
            [],
        )
        .unwrap();

        conn.execute(
            "INSERT INTO cursorDiskKV (key, value) VALUES (?1, ?2)",
            [
                "composerData:comp_user",
                &serde_json::to_string(&serde_json::json!({
                    "_v": 13,
                    "context": { "fileSelections": [{"path": "/tmp/repo/src/main.rs"}] }
                }))
                .unwrap(),
            ],
        )
        .unwrap();

        let user_with_metadata = serde_json::json!({
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": "<open_and_recently_viewed_files>\n...</open_and_recently_viewed_files>"
                },
                {
                    "type": "text",
                    "text": "<user_query>\nTREVEC_CURSOR_TEST_MARKER\n</user_query>"
                }
            ]
        })
        .to_string()
        .into_bytes();
        conn.execute(
            "INSERT INTO cursorDiskKV (key, value) VALUES (?1, ?2)",
            rusqlite::params!["agentKv:blob:user_meta_query", user_with_metadata],
        )
        .unwrap();

        let rows = vec![
            (
                2_i64,
                conn.query_row(
                    "SELECT value FROM cursorDiskKV WHERE key='agentKv:blob:user_meta_query'",
                    [],
                    |row| row.get::<_, Vec<u8>>(0),
                )
                .unwrap(),
            ),
        ];

        let turns = extract_agent_kv_turns(&conn, &rows, "/tmp/repo", &HashSet::new());
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].role, "user");
        assert_eq!(turns[0].user_prompt, "TREVEC_CURSOR_TEST_MARKER");
        assert_eq!(turns[0].assistant_text, "");
    }

    #[test]
    fn test_extract_agent_kv_turns_skips_sessions_with_composer_turns() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute(
            "CREATE TABLE cursorDiskKV (key TEXT PRIMARY KEY, value BLOB)",
            [],
        )
        .unwrap();
        conn.execute(
            "INSERT INTO cursorDiskKV (key, value) VALUES (?1, ?2)",
            [
                "composerData:comp_skip",
                &serde_json::to_string(&serde_json::json!({"_v": 13})).unwrap(),
            ],
        )
        .unwrap();

        let assistant = serde_json::json!({
            "role": "assistant",
            "content": [{"type": "text", "text": "SHOULD_NOT_INGEST"}]
        })
        .to_string()
        .into_bytes();
        conn.execute(
            "INSERT INTO cursorDiskKV (key, value) VALUES (?1, ?2)",
            rusqlite::params!["agentKv:blob:assistant_skip", assistant],
        )
        .unwrap();

        let rows = vec![
            (
                2_i64,
                conn.query_row(
                    "SELECT value FROM cursorDiskKV WHERE key='agentKv:blob:assistant_skip'",
                    [],
                    |row| row.get::<_, Vec<u8>>(0),
                )
                .unwrap(),
            ),
        ];
        let mut skip = HashSet::new();
        skip.insert("comp_skip".to_string());
        let turns = extract_agent_kv_turns(&conn, &rows, "/tmp/repo", &skip);
        assert!(turns.is_empty());
    }

    // -----------------------------------------------------------------------
    // New tests for layered relevance, ISO timestamps, watermark guard
    // -----------------------------------------------------------------------

    #[test]
    fn test_parse_iso8601_utc() {
        let ts = parse_iso8601_timestamp("2024-01-15T10:30:00Z");
        assert_eq!(ts, 1705314600);
    }

    #[test]
    fn test_parse_iso8601_with_fractional_seconds() {
        let ts = parse_iso8601_timestamp("2024-01-15T10:30:00.123Z");
        assert_eq!(ts, 1705314600);
    }

    #[test]
    fn test_parse_iso8601_with_positive_offset() {
        let ts = parse_iso8601_timestamp("2024-01-15T10:30:00+05:00");
        // 10:30 in UTC+5 = 05:30 UTC
        assert_eq!(ts, 1705314600 - 18000);
    }

    #[test]
    fn test_parse_iso8601_with_negative_offset() {
        let ts = parse_iso8601_timestamp("2024-01-15T10:30:00-03:00");
        // 10:30 in UTC-3 = 13:30 UTC
        assert_eq!(ts, 1705314600 + 10800);
    }

    #[test]
    fn test_parse_iso8601_invalid_returns_zero() {
        assert_eq!(parse_iso8601_timestamp("not-a-date"), 0);
        assert_eq!(parse_iso8601_timestamp(""), 0);
        assert_eq!(parse_iso8601_timestamp("12345"), 0); // too short for ISO
    }

    #[test]
    fn test_iso8601_timestamp_in_bubble() {
        let bubbles = vec![
            serde_json::json!({
                "type": 1,
                "text": "What is this?",
                "createdAt": "2024-01-15T10:30:00Z"
            }),
            serde_json::json!({
                "type": 2,
                "text": "It is a test.",
                "createdAt": "2024-01-15T10:30:01.500Z"
            }),
        ];

        let mut turns = Vec::new();
        extract_from_bubbles(&bubbles, "iso_comp", &[], &mut turns);

        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].timestamp, 1705314601);
    }

    #[test]
    fn test_layered_relevance_trevec_marker() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute(
            "CREATE TABLE cursorDiskKV (key TEXT PRIMARY KEY, value TEXT)",
            [],
        )
        .unwrap();

        let repo_str = "/tmp/test_repo";
        let repo_id = blake3::hash(repo_str.as_bytes()).to_hex();
        let marker = format!("trevec:repo_id:{}", &repo_id[..32]);

        // Composer with NO relevantFiles / fileSelections (would fail old check).
        let composer_data = serde_json::json!({"_v": 13});
        conn.execute(
            "INSERT INTO cursorDiskKV (key, value) VALUES (?1, ?2)",
            ["composerData:comp_marker", &serde_json::to_string(&composer_data).unwrap()],
        )
        .unwrap();

        // Bubble containing the repo marker in text.
        let bubble = serde_json::json!({
            "type": 2,
            "text": format!("Here is context\n<!-- {} -->", marker),
            "createdAt": 1700000000000_i64
        });
        conn.execute(
            "INSERT INTO cursorDiskKV (key, value) VALUES (?1, ?2)",
            [
                "bubbleId:comp_marker:asst_1",
                &serde_json::to_string(&bubble).unwrap(),
            ],
        )
        .unwrap();

        assert!(is_composer_relevant_to_repo(&conn, "comp_marker", &composer_data, repo_str));
    }

    #[test]
    fn test_layered_relevance_per_bubble_file_path() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute(
            "CREATE TABLE cursorDiskKV (key TEXT PRIMARY KEY, value TEXT)",
            [],
        )
        .unwrap();

        // Composer with empty context (v13-style).
        let composer_data = serde_json::json!({"_v": 13});
        conn.execute(
            "INSERT INTO cursorDiskKV (key, value) VALUES (?1, ?2)",
            ["composerData:comp_bubble", &serde_json::to_string(&composer_data).unwrap()],
        )
        .unwrap();

        // Bubble with per-bubble fileSelections.
        let bubble = serde_json::json!({
            "type": 1,
            "text": "Help with auth",
            "context": {
                "fileSelections": [{"path": "/tmp/my_project/src/auth.rs"}]
            }
        });
        conn.execute(
            "INSERT INTO cursorDiskKV (key, value) VALUES (?1, ?2)",
            [
                "bubbleId:comp_bubble:user_1",
                &serde_json::to_string(&bubble).unwrap(),
            ],
        )
        .unwrap();

        assert!(is_composer_relevant_to_repo(&conn, "comp_bubble", &composer_data, "/tmp/my_project"));
        assert!(!is_composer_relevant_to_repo(&conn, "comp_bubble", &composer_data, "/tmp/other_project"));
    }

    #[test]
    fn test_layered_relevance_falls_back_to_composer_level() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute(
            "CREATE TABLE cursorDiskKV (key TEXT PRIMARY KEY, value TEXT)",
            [],
        )
        .unwrap();

        // Old-style composer with relevantFiles at composer level (no separate bubbles).
        let composer_data = serde_json::json!({
            "relevantFiles": ["/tmp/my_project/src/main.rs"]
        });
        conn.execute(
            "INSERT INTO cursorDiskKV (key, value) VALUES (?1, ?2)",
            ["composerData:comp_old", &serde_json::to_string(&composer_data).unwrap()],
        )
        .unwrap();

        assert!(is_composer_relevant_to_repo(&conn, "comp_old", &composer_data, "/tmp/my_project"));
    }

    #[test]
    fn test_layered_relevance_default_false() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute(
            "CREATE TABLE cursorDiskKV (key TEXT PRIMARY KEY, value TEXT)",
            [],
        )
        .unwrap();

        // Composer with no signals at all.
        let composer_data = serde_json::json!({"_v": 13});
        conn.execute(
            "INSERT INTO cursorDiskKV (key, value) VALUES (?1, ?2)",
            ["composerData:comp_empty", &serde_json::to_string(&composer_data).unwrap()],
        )
        .unwrap();

        assert!(!is_composer_relevant_to_repo(&conn, "comp_empty", &composer_data, "/tmp/some_repo"));
    }

    #[test]
    fn test_watermark_advances_even_when_no_turns_match() {
        // Watermark must advance even when no composers match the current
        // repo.  Otherwise every sync re-scans the entire global tail in
        // multi-repo Cursor DBs — an O(n) performance regression.
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("state.vscdb");

        let conn = Connection::open(&db_path).unwrap();
        conn.execute(
            "CREATE TABLE cursorDiskKV (key TEXT PRIMARY KEY, value TEXT)",
            [],
        )
        .unwrap();

        // Insert a composer for a DIFFERENT repo — should not match.
        let data = serde_json::json!({
            "bubbles": [
                {"type": 1, "text": "Question", "createdAt": 1700000000000_i64},
                {"type": 2, "text": "Answer", "createdAt": 1700000001000_i64}
            ],
            "relevantFiles": ["/other/project/src/a.rs"]
        });
        conn.execute(
            "INSERT INTO cursorDiskKV (key, value) VALUES (?1, ?2)",
            ["composerData:comp_other", &serde_json::to_string(&data).unwrap()],
        )
        .unwrap();
        drop(conn);

        let extractor = CursorExtractor::new(db_path);
        let mut meta = MemoryMeta::default();

        let turns = extractor.extract(Path::new("/tmp/my_repo"), &mut meta).unwrap();
        assert!(turns.is_empty());
        // Watermark SHOULD have advanced past the scanned rows.
        assert!(meta.cursor_last_rowid.is_some());
    }

    #[test]
    fn test_agent_kv_marker_detection() {
        // Verify that a trevec repo marker in an agentKv payload is enough
        // to associate the row with this repo, even when no file paths
        // or composer-level context match.
        let conn = Connection::open_in_memory().unwrap();
        conn.execute(
            "CREATE TABLE cursorDiskKV (key TEXT PRIMARY KEY, value BLOB)",
            [],
        )
        .unwrap();

        // Composer with no file context at all.
        conn.execute(
            "INSERT INTO cursorDiskKV (key, value) VALUES (?1, ?2)",
            [
                "composerData:comp_marker",
                &serde_json::to_string(&serde_json::json!({"_v": 13})).unwrap(),
            ],
        )
        .unwrap();

        let repo_str = "/tmp/marker_repo";
        let marker = compute_repo_id_marker(repo_str);

        // Assistant response with the trevec marker embedded.
        let assistant = serde_json::json!({
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": format!("Here is your context.\n<!-- {} -->", marker)
            }]
        })
        .to_string()
        .into_bytes();
        conn.execute(
            "INSERT INTO cursorDiskKV (key, value) VALUES (?1, ?2)",
            rusqlite::params!["agentKv:blob:asst_marker", assistant],
        )
        .unwrap();

        let rows = vec![(
            2_i64,
            conn.query_row(
                "SELECT value FROM cursorDiskKV WHERE key='agentKv:blob:asst_marker'",
                [],
                |row| row.get::<_, Vec<u8>>(0),
            )
            .unwrap(),
        )];

        let turns = extract_agent_kv_turns(&conn, &rows, repo_str, &HashSet::new());
        assert_eq!(turns.len(), 1, "marker in agentKv should associate the row");
    }

    #[test]
    fn test_agent_kv_repo_hash_detection() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute(
            "CREATE TABLE cursorDiskKV (key TEXT PRIMARY KEY, value BLOB)",
            [],
        )
        .unwrap();

        let repo_str = "/Users/test/fastapi-master";
        let repo_hash = compute_repo_id_hash(repo_str);
        let assistant = serde_json::json!({
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": format!("Stored in memory for repo_id:{}", repo_hash)
            }]
        })
        .to_string()
        .into_bytes();
        conn.execute(
            "INSERT INTO cursorDiskKV (key, value) VALUES (?1, ?2)",
            rusqlite::params!["agentKv:blob:asst_repo_hash", assistant],
        )
        .unwrap();

        let rows = vec![(
            1_i64,
            conn.query_row(
                "SELECT value FROM cursorDiskKV WHERE key='agentKv:blob:asst_repo_hash'",
                [],
                |row| row.get::<_, Vec<u8>>(0),
            )
            .unwrap(),
        )];

        let turns = extract_agent_kv_turns(&conn, &rows, repo_str, &HashSet::new());
        assert_eq!(turns.len(), 1, "repo hash in agentKv should associate the row");
    }

    #[test]
    fn test_agent_kv_repo_tool_signal_detection() {
        let conn = Connection::open_in_memory().unwrap();
        conn.execute(
            "CREATE TABLE cursorDiskKV (key TEXT PRIMARY KEY, value BLOB)",
            [],
        )
        .unwrap();

        let repo_str = "/Users/test/fastapi-master";
        let assistant = serde_json::json!({
            "role": "assistant",
            "content": [{
                "type": "text",
                "text": "Called mcp_fastapi-master-trevec_remember_turn successfully."
            }]
        })
        .to_string()
        .into_bytes();
        conn.execute(
            "INSERT INTO cursorDiskKV (key, value) VALUES (?1, ?2)",
            rusqlite::params!["agentKv:blob:asst_repo_tool", assistant],
        )
        .unwrap();

        let rows = vec![(
            1_i64,
            conn.query_row(
                "SELECT value FROM cursorDiskKV WHERE key='agentKv:blob:asst_repo_tool'",
                [],
                |row| row.get::<_, Vec<u8>>(0),
            )
            .unwrap(),
        )];

        let turns = extract_agent_kv_turns(&conn, &rows, repo_str, &HashSet::new());
        assert_eq!(turns.len(), 1, "repo tool signal should associate the row");
    }
}
