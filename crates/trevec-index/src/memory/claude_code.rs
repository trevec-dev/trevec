use anyhow::{Context, Result};
use std::io::{BufRead, BufReader};
use std::path::{Path, PathBuf};

use super::meta::MemoryMeta;
use super::RawTurn;

/// Extracts chat history from Claude Code's JSONL session files.
pub struct ClaudeCodeExtractor {
    projects_dir: PathBuf,
}

impl ClaudeCodeExtractor {
    pub fn new(projects_dir: PathBuf) -> Self {
        Self { projects_dir }
    }

    /// Auto-detect the Claude Code projects directory.
    pub fn detect() -> Option<Self> {
        let home = dirs_path()?;
        let dir = home.join(".claude").join("projects");
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
        let encoded = encode_repo_path(repo_path);
        let project_dir = self.projects_dir.join(&encoded);

        if !project_dir.is_dir() {
            return Ok(vec![]);
        }

        let mut all_turns = Vec::new();

        // Glob for JSONL files in the project directory
        let entries: Vec<_> = std::fs::read_dir(&project_dir)
            .with_context(|| format!("Failed to read {}", project_dir.display()))?
            .filter_map(|e| e.ok())
            .filter(|e| {
                e.path()
                    .extension()
                    .map(|ext| ext == "jsonl")
                    .unwrap_or(false)
            })
            .collect();

        for entry in entries {
            let path = entry.path();
            let file_key = path
                .file_name()
                .unwrap_or_default()
                .to_string_lossy()
                .to_string();

            let offset = meta
                .claude_code_offsets
                .get(&file_key)
                .copied()
                .unwrap_or(0);

            let file_len = std::fs::metadata(&path)
                .map(|m| m.len())
                .unwrap_or(0);

            if file_len <= offset {
                continue;
            }

            let turns = parse_jsonl_file(&path, offset, &encoded)?;
            if !turns.is_empty() {
                meta.claude_code_offsets.insert(file_key, file_len);
            }
            all_turns.extend(turns);
        }

        Ok(all_turns)
    }
}

/// Encode a repo path to Claude Code's dash-separated format.
/// e.g., `/Users/alice/dev/backend` → `-Users-alice-dev-backend`
fn encode_repo_path(path: &Path) -> String {
    let s = path.to_string_lossy();
    s.replace('/', "-")
}

/// Parse a Claude Code JSONL session file from the given byte offset.
fn parse_jsonl_file(
    path: &Path,
    offset: u64,
    session_id: &str,
) -> Result<Vec<RawTurn>> {
    let file = std::fs::File::open(path)
        .with_context(|| format!("Failed to open {}", path.display()))?;

    let reader = BufReader::new(file);
    let mut turns = Vec::new();
    let mut current_user_prompt = String::new();
    let mut current_tool_calls = Vec::new();
    let mut current_files = Vec::new();
    let mut turn_index: u32 = 0;
    let mut bytes_read: u64 = 0;

    for line in reader.lines() {
        let line = line.with_context(|| format!("Failed to read line from {}", path.display()))?;
        let line_bytes = line.len() as u64 + 1; // +1 for newline

        if bytes_read + line_bytes <= offset {
            bytes_read += line_bytes;
            // Still need to count turn indices for lines before offset
            if let Ok(val) = serde_json::from_str::<serde_json::Value>(&line) {
                if val.get("type").and_then(|t| t.as_str()) == Some("assistant") {
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

        let msg_type = val.get("type").and_then(|t| t.as_str()).unwrap_or("");

        match msg_type {
            "human" => {
                // Extract user prompt text from message.content
                current_user_prompt.clear();
                current_tool_calls.clear();
                current_files.clear();

                if let Some(message) = val.get("message") {
                    if let Some(content) = message.get("content") {
                        extract_text_content(content, &mut current_user_prompt);
                    }
                }
            }
            "assistant" => {
                let mut assistant_text = String::new();

                if let Some(message) = val.get("message") {
                    if let Some(content) = message.get("content") {
                        extract_assistant_content(
                            content,
                            &mut assistant_text,
                            &mut current_tool_calls,
                            &mut current_files,
                        );
                    }
                }

                let timestamp = val
                    .get("timestamp")
                    .and_then(|t| t.as_str())
                    .and_then(chrono_parse_or_unix)
                    .unwrap_or(0);

                turns.push(RawTurn {
                    source: "claude_code".to_string(),
                    session_id: session_id.to_string(),
                    turn_index,
                    timestamp,
                    role: "user".to_string(),
                    user_prompt: current_user_prompt.clone(),
                    assistant_text,
                    tool_calls: current_tool_calls.clone(),
                    files_touched: current_files.clone(),
                });

                turn_index += 1;
                current_user_prompt.clear();
                current_tool_calls.clear();
                current_files.clear();
            }
            _ => {}
        }
    }

    Ok(turns)
}

/// Extract text from a content value (string or array of content blocks).
fn extract_text_content(content: &serde_json::Value, out: &mut String) {
    match content {
        serde_json::Value::String(s) => {
            out.push_str(s);
        }
        serde_json::Value::Array(arr) => {
            for block in arr {
                if block.get("type").and_then(|t| t.as_str()) == Some("text") {
                    if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                        if !out.is_empty() {
                            out.push('\n');
                        }
                        out.push_str(text);
                    }
                }
            }
        }
        _ => {}
    }
}

/// Extract assistant text, tool calls, and file paths from content blocks.
fn extract_assistant_content(
    content: &serde_json::Value,
    text_out: &mut String,
    tool_calls: &mut Vec<String>,
    files: &mut Vec<String>,
) {
    let blocks = match content {
        serde_json::Value::Array(arr) => arr.as_slice(),
        serde_json::Value::String(s) => {
            text_out.push_str(s);
            return;
        }
        _ => return,
    };

    for block in blocks {
        let block_type = block.get("type").and_then(|t| t.as_str()).unwrap_or("");
        match block_type {
            "text" => {
                if let Some(text) = block.get("text").and_then(|t| t.as_str()) {
                    if !text_out.is_empty() {
                        text_out.push('\n');
                    }
                    text_out.push_str(text);
                }
            }
            "tool_use" => {
                if let Some(name) = block.get("name").and_then(|n| n.as_str()) {
                    tool_calls.push(name.to_string());
                }
                // Extract file paths from tool input
                if let Some(input) = block.get("input") {
                    extract_file_paths_from_input(input, files);
                }
            }
            _ => {}
        }
    }
}

/// Extract file paths from tool_use input objects.
fn extract_file_paths_from_input(input: &serde_json::Value, files: &mut Vec<String>) {
    // Check common fields: file_path, path, command (for bash)
    for key in &["file_path", "path"] {
        if let Some(fp) = input.get(*key).and_then(|v| v.as_str()) {
            if !files.contains(&fp.to_string()) {
                files.push(fp.to_string());
            }
        }
    }
    // For Bash commands, try to extract paths from the command string
    if let Some(cmd) = input.get("command").and_then(|v| v.as_str()) {
        // Simple heuristic: look for paths that look like file references
        for token in cmd.split_whitespace() {
            if (token.contains('/') || token.contains('.'))
                && !token.starts_with('-')
                && !token.starts_with("http")
                && !files.contains(&token.to_string())
            {
                files.push(token.to_string());
            }
        }
    }
}

/// Try to parse a timestamp string as unix seconds or ISO-8601.
fn chrono_parse_or_unix(s: &str) -> Option<i64> {
    // Try as a plain integer (unix seconds/millis)
    if let Ok(n) = s.parse::<i64>() {
        if n > 1_000_000_000_000 {
            // milliseconds → seconds
            return Some(n / 1000);
        }
        return Some(n);
    }
    // Try ISO-8601 / RFC 3339 (e.g., "2024-01-15T10:30:00Z")
    let ts = super::cursor::parse_iso8601_timestamp(s);
    if ts > 0 { Some(ts) } else { None }
}

fn dirs_path() -> Option<PathBuf> {
    #[cfg(target_os = "macos")]
    {
        std::env::var("HOME").ok().map(PathBuf::from)
    }
    #[cfg(target_os = "linux")]
    {
        std::env::var("HOME").ok().map(PathBuf::from)
    }
    #[cfg(target_os = "windows")]
    {
        std::env::var("USERPROFILE").ok().map(PathBuf::from)
    }
    #[cfg(not(any(target_os = "macos", target_os = "linux", target_os = "windows")))]
    {
        std::env::var("HOME").ok().map(PathBuf::from)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_encode_repo_path() {
        let path = Path::new("/Users/alice/dev/backend");
        assert_eq!(encode_repo_path(path), "-Users-alice-dev-backend");
    }

    #[test]
    fn test_encode_repo_path_trailing_slash() {
        let path = Path::new("/home/user/project/");
        // PathBuf normalizes trailing slashes
        assert_eq!(encode_repo_path(path), "-home-user-project-");
    }

    #[test]
    fn test_extract_text_content_string() {
        let val = serde_json::json!("Hello world");
        let mut out = String::new();
        extract_text_content(&val, &mut out);
        assert_eq!(out, "Hello world");
    }

    #[test]
    fn test_extract_text_content_array() {
        let val = serde_json::json!([
            {"type": "text", "text": "First block"},
            {"type": "text", "text": "Second block"}
        ]);
        let mut out = String::new();
        extract_text_content(&val, &mut out);
        assert!(out.contains("First block"));
        assert!(out.contains("Second block"));
    }

    #[test]
    fn test_extract_assistant_content_with_tools() {
        let val = serde_json::json!([
            {"type": "text", "text": "Let me read that file."},
            {"type": "tool_use", "name": "Read", "input": {"file_path": "src/main.rs"}}
        ]);
        let mut text = String::new();
        let mut tools = Vec::new();
        let mut files = Vec::new();
        extract_assistant_content(&val, &mut text, &mut tools, &mut files);
        assert!(text.contains("Let me read that file"));
        assert_eq!(tools, vec!["Read"]);
        assert_eq!(files, vec!["src/main.rs"]);
    }

    #[test]
    fn test_parse_jsonl_file() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.jsonl");

        let lines = vec![
            r#"{"type":"human","message":{"role":"user","content":"What does auth do?"},"timestamp":"1700000000"}"#,
            r#"{"type":"assistant","message":{"role":"assistant","content":[{"type":"text","text":"Auth handles login."},{"type":"tool_use","name":"Read","input":{"file_path":"src/auth.rs"}}]},"timestamp":"1700000001"}"#,
        ];
        std::fs::write(&path, lines.join("\n") + "\n").unwrap();

        let turns = parse_jsonl_file(&path, 0, "test_session").unwrap();
        assert_eq!(turns.len(), 1);
        assert_eq!(turns[0].source, "claude_code");
        assert_eq!(turns[0].user_prompt, "What does auth do?");
        assert!(turns[0].assistant_text.contains("Auth handles login"));
        assert_eq!(turns[0].tool_calls, vec!["Read"]);
        assert_eq!(turns[0].files_touched, vec!["src/auth.rs"]);
    }

    #[test]
    fn test_parse_jsonl_file_incremental() {
        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("test.jsonl");

        let line1 = r#"{"type":"human","message":{"role":"user","content":"First"},"timestamp":"1700000000"}"#;
        let line2 = r#"{"type":"assistant","message":{"role":"assistant","content":"Response 1"},"timestamp":"1700000001"}"#;
        let line3 = r#"{"type":"human","message":{"role":"user","content":"Second"},"timestamp":"1700000002"}"#;
        let line4 = r#"{"type":"assistant","message":{"role":"assistant","content":"Response 2"},"timestamp":"1700000003"}"#;

        let content = format!("{}\n{}\n{}\n{}\n", line1, line2, line3, line4);
        std::fs::write(&path, &content).unwrap();

        // First read: get all turns
        let all = parse_jsonl_file(&path, 0, "sess").unwrap();
        assert_eq!(all.len(), 2);

        // Compute offset after first two lines
        let offset = (line1.len() + 1 + line2.len() + 1) as u64;
        let incremental = parse_jsonl_file(&path, offset, "sess").unwrap();
        assert_eq!(incremental.len(), 1);
        assert_eq!(incremental[0].user_prompt, "Second");
    }

    #[test]
    fn test_chrono_parse_unix_seconds() {
        assert_eq!(chrono_parse_or_unix("1700000000"), Some(1700000000));
    }

    #[test]
    fn test_chrono_parse_unix_millis() {
        assert_eq!(chrono_parse_or_unix("1700000000000"), Some(1700000000));
    }
}
