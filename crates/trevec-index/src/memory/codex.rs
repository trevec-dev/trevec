use anyhow::{Context, Result};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use super::meta::MemoryMeta;
use super::RawTurn;

/// Extracts chat history from Codex's JSONL session files.
pub struct CodexExtractor {
    sessions_dir: PathBuf,
}

impl CodexExtractor {
    pub fn new(sessions_dir: PathBuf) -> Self {
        Self { sessions_dir }
    }

    /// Auto-detect the Codex sessions directory.
    pub fn detect() -> Option<Self> {
        let home = std::env::var("HOME")
            .or_else(|_| std::env::var("USERPROFILE"))
            .ok()?;
        let dir = PathBuf::from(home).join(".codex").join("sessions");
        if dir.is_dir() {
            Some(Self::new(dir))
        } else {
            None
        }
    }

    /// Extract turns for the given repo path, reading incrementally from offsets.
    pub fn extract(
        &self,
        repo_path: &Path,
        meta: &mut MemoryMeta,
    ) -> Result<Vec<RawTurn>> {
        if !self.sessions_dir.is_dir() {
            return Ok(vec![]);
        }

        let mut all_turns = Vec::new();
        let repo_str = repo_path.to_string_lossy();

        // Walk all JSONL files
        let entries = walk_jsonl_files(&self.sessions_dir)?;

        for path in entries {
            let file_key = path
                .strip_prefix(&self.sessions_dir)
                .unwrap_or(&path)
                .to_string_lossy()
                .to_string();

            let offset = meta.codex_offsets.get(&file_key).copied().unwrap_or(0);

            let file_len = std::fs::metadata(&path)
                .map(|m| m.len())
                .unwrap_or(0);

            if file_len <= offset {
                continue;
            }

            let turns = parse_codex_file(&path, offset, &repo_str)?;
            if !turns.is_empty() {
                meta.codex_offsets.insert(file_key, file_len);
            }
            all_turns.extend(turns);
        }

        Ok(all_turns)
    }
}

/// Recursively find all .jsonl files in a directory.
fn walk_jsonl_files(dir: &Path) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();
    if !dir.is_dir() {
        return Ok(files);
    }

    for entry in std::fs::read_dir(dir)
        .with_context(|| format!("Failed to read dir {}", dir.display()))?
    {
        let entry = entry?;
        let path = entry.path();
        if path.is_dir() {
            files.extend(walk_jsonl_files(&path)?);
        } else if path.extension().map(|e| e == "jsonl").unwrap_or(false) {
            files.push(path);
        }
    }

    Ok(files)
}

/// Parse a Codex JSONL session file from the given byte offset.
fn parse_codex_file(
    path: &Path,
    offset: u64,
    repo_str: &str,
) -> Result<Vec<RawTurn>> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open {}", path.display()))?;

    let reader = BufReader::new(file);
    let mut turns = Vec::new();
    let mut session_id = path
        .file_stem()
        .unwrap_or_default()
        .to_string_lossy()
        .to_string();
    let mut session_cwd = String::new();
    let mut current_user = String::new();
    let mut turn_index: u32 = 0;
    let mut bytes_read: u64 = 0;

    for line in reader.lines() {
        let line = line.with_context(|| format!("Failed to read from {}", path.display()))?;
        let line_bytes = line.len() as u64 + 1;

        if bytes_read + line_bytes <= offset {
            bytes_read += line_bytes;
            // Count turns for proper indexing
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&line) {
                if val.get("type").and_then(|t| t.as_str()) == Some("response_item")
                    && val.get("role").and_then(|r| r.as_str()) == Some("assistant")
                {
                    turn_index += 1;
                }
            }
            continue;
        }
        bytes_read += line_bytes;

        let val: serde_json::Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue,
        };

        let item_type = val.get("type").and_then(|t| t.as_str()).unwrap_or("");

        match item_type {
            "session_meta" => {
                if let Some(cwd) = val.get("cwd").and_then(|c| c.as_str()) {
                    session_cwd = cwd.to_string();
                }
                if let Some(sid) = val.get("session_id").and_then(|s| s.as_str()) {
                    session_id = sid.to_string();
                }
            }
            "response_item" => {
                let role = val.get("role").and_then(|r| r.as_str()).unwrap_or("");
                let text = val
                    .get("text")
                    .or_else(|| val.get("content"))
                    .and_then(|t| t.as_str())
                    .unwrap_or("")
                    .to_string();

                let timestamp = val
                    .get("timestamp")
                    .and_then(|t| t.as_i64())
                    .unwrap_or(0);

                match role {
                    "user" => {
                        current_user = text;
                    }
                    "assistant" => {
                        // Extract tool calls
                        let tool_calls: Vec<String> = val
                            .get("tool_calls")
                            .and_then(|tc| tc.as_array())
                            .map(|arr| {
                                arr.iter()
                                    .filter_map(|t| {
                                        t.get("name").and_then(|n| n.as_str()).map(|s| s.to_string())
                                    })
                                    .collect()
                            })
                            .unwrap_or_default();

                        // Check if session is for our repo
                        if !session_cwd.is_empty() && !session_cwd.contains(repo_str) {
                            continue;
                        }

                        turns.push(RawTurn {
                            source: "codex".to_string(),
                            session_id: session_id.clone(),
                            turn_index,
                            timestamp,
                            role: "user".to_string(),
                            user_prompt: current_user.clone(),
                            assistant_text: text,
                            tool_calls,
                            files_touched: vec![],
                        });

                        turn_index += 1;
                        current_user.clear();
                    }
                    _ => {}
                }
            }
            // function_call items
            "function_call" => {
                // These are tool calls that we can attach to the next assistant turn
                // For now we capture them inline with response_item
            }
            _ => {}
        }
    }

    Ok(turns)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_parse_codex_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("session1.jsonl");

        let lines = vec![
            r#"{"type":"session_meta","cwd":"/home/user/project","session_id":"sess_123"}"#,
            r#"{"type":"response_item","role":"user","text":"What is this codebase?","timestamp":1700000000}"#,
            r#"{"type":"response_item","role":"assistant","text":"This is a Rust project.","timestamp":1700000001}"#,
        ];
        std::fs::write(&path, lines.join("\n") + "\n").unwrap();

        let turns = parse_codex_file(&path, 0, "/home/user/project").unwrap();
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].source, "codex");
        assert_eq!(turns[0].session_id, "sess_123");
        assert_eq!(turns[0].user_prompt, "What is this codebase?");
        assert_eq!(turns[0].assistant_text, "This is a Rust project.");
    }

    #[test]
    fn test_parse_codex_file_wrong_repo() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("session2.jsonl");

        let lines = vec![
            r#"{"type":"session_meta","cwd":"/other/repo"}"#,
            r#"{"type":"response_item","role":"user","text":"Hello","timestamp":1700000000}"#,
            r#"{"type":"response_item","role":"assistant","text":"Hi","timestamp":1700000001}"#,
        ];
        std::fs::write(&path, lines.join("\n") + "\n").unwrap();

        let turns = parse_codex_file(&path, 0, "/home/user/project").unwrap();
        assert_eq!(turns.len(), 0); // Filtered out — different CWD
    }

    #[test]
    fn test_walk_jsonl_files() {
        let dir = tempfile::tempdir().unwrap();
        let sub = dir.path().join("sub");
        std::fs::create_dir(&sub).unwrap();

        std::fs::write(dir.path().join("a.jsonl"), "").unwrap();
        std::fs::write(sub.join("b.jsonl"), "").unwrap();
        std::fs::write(dir.path().join("c.txt"), "").unwrap();

        let files = walk_jsonl_files(dir.path()).unwrap();
        assert_eq!(files.len(), 2);
    }
}
