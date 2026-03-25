use anyhow::{Context, Result};
use std::path::Path;

const MARKER_START: &str = "<!-- trevec:rules:start -->";
const MARKER_END: &str = "<!-- trevec:rules:end -->";

/// Full `.mdc` file for Cursor. Trevec owns the entire file.
const CURSOR_RULE: &str = r#"---
description: Trevec MCP tools for code context retrieval and episodic memory
globs:
alwaysApply: true
---

# Trevec MCP Tools

Use these MCP tools to retrieve precise, graph-aware code context instead of reading files manually.

## Tools

### get_context
Retrieves relevant code context for a natural-language query. Returns relevant code nodes with file paths, spans, and related context. **Use this as your primary tool for understanding code.**

### search_code
Hybrid search over indexed code nodes. Returns ranked results with file paths and signatures. Use for targeted symbol or keyword lookup.

### read_file_topology
Returns the structural topology of a file: all code nodes (functions, classes, methods) with their relationships (calls, imports, contains). Use to understand file structure before making changes.

### repo_summary
Returns a high-level overview of the repository: languages, file/node/edge counts, top-level modules, entry points, hotspots, and detected conventions. Use for onboarding or getting a quick sense of a codebase.

### neighbor_signatures
Given a list of file paths, returns the external API surface those files depend on — imported symbols from other files with their signatures.

### batch_context
Runs multiple `get_context` queries in a single call. Each query can have its own budget and anchor count. Reduces round-trips for multi-query workflows.

### remember_turn
Records a conversation turn into episodic memory. Call this when the user shares important context, decisions, or preferences that should persist across sessions.

### recall_history
Searches episodic memory for past conversation context. Use when the user references previous discussions or when historical context would help answer a question.

## Guidelines

- Prefer `get_context` over reading raw files — it returns only the relevant code with graph context.
- Use `search_code` for quick symbol lookups (function names, class names, error messages).
- Use `read_file_topology` before modifying a file to understand its structure and dependencies.
- Use `repo_summary` for onboarding or to get a quick overview of the codebase structure.
- Use `neighbor_signatures` to discover imports/dependencies of specific files before editing.
- Use `batch_context` when you need context for multiple queries — saves round-trips.
- Call `remember_turn` for important decisions, preferences, or context the user shares.
- Call `recall_history` when the user says "we discussed", "last time", or references prior work.
"#;

/// Markdown block for Claude Code `CLAUDE.md`. Wrapped in markers.
const CLAUDE_CODE_RULE: &str = r#"
## Trevec MCP Tools

Use these MCP tools to retrieve precise, graph-aware code context instead of reading files manually.

### get_context
Retrieves relevant code context for a natural-language query. Returns relevant code nodes with file paths, spans, and related context. **Use this as your primary tool for understanding code.**

### search_code
Hybrid search over indexed code nodes. Returns ranked results with file paths and signatures. Use for targeted symbol or keyword lookup.

### read_file_topology
Returns the structural topology of a file: all code nodes (functions, classes, methods) with their relationships (calls, imports, contains). Use to understand file structure before making changes.

### repo_summary
Returns a high-level overview of the repository: languages, file/node/edge counts, top-level modules, entry points, hotspots, and detected conventions. Use for onboarding or getting a quick sense of a codebase.

### neighbor_signatures
Given a list of file paths, returns the external API surface those files depend on — imported symbols from other files with their signatures.

### batch_context
Runs multiple `get_context` queries in a single call. Each query can have its own budget and anchor count. Reduces round-trips for multi-query workflows.

### remember_turn
Records a conversation turn into episodic memory. Call this when the user shares important context, decisions, or preferences that should persist across sessions.

### recall_history
Searches episodic memory for past conversation context. Use when the user references previous discussions or when historical context would help answer a question.

### Guidelines
- Prefer `get_context` over reading raw files — it returns only the relevant code with graph context.
- Use `search_code` for quick symbol lookups (function names, class names, error messages).
- Use `read_file_topology` before modifying a file to understand its structure and dependencies.
- Use `repo_summary` for onboarding or to get a quick overview of the codebase structure.
- Use `neighbor_signatures` to discover imports/dependencies of specific files before editing.
- Use `batch_context` when you need context for multiple queries — saves round-trips.
- Call `remember_turn` for important decisions, preferences, or context the user shares.
- Call `recall_history` when the user says "we discussed", "last time", or references prior work.
"#;

/// Markdown block for Codex `AGENTS.md`. Wrapped in markers.
const CODEX_RULE: &str = r#"
## Trevec MCP Tools

Use these MCP tools to retrieve precise, graph-aware code context instead of reading files manually.

### get_context
Retrieves relevant code context for a natural-language query. Returns relevant code nodes with file paths, spans, and related context. **Use this as your primary tool for understanding code.**

### search_code
Hybrid search over indexed code nodes. Returns ranked results with file paths and signatures. Use for targeted symbol or keyword lookup.

### read_file_topology
Returns the structural topology of a file: all code nodes (functions, classes, methods) with their relationships (calls, imports, contains). Use to understand file structure before making changes.

### repo_summary
Returns a high-level overview of the repository: languages, file/node/edge counts, top-level modules, entry points, hotspots, and detected conventions. Use for onboarding or getting a quick sense of a codebase.

### neighbor_signatures
Given a list of file paths, returns the external API surface those files depend on — imported symbols from other files with their signatures.

### batch_context
Runs multiple `get_context` queries in a single call. Each query can have its own budget and anchor count. Reduces round-trips for multi-query workflows.

### remember_turn
Records a conversation turn into episodic memory. Call this when the user shares important context, decisions, or preferences that should persist across sessions.

### recall_history
Searches episodic memory for past conversation context. Use when the user references previous discussions or when historical context would help answer a question.

### Guidelines
- Prefer `get_context` over reading raw files — it returns only the relevant code with graph context.
- Use `search_code` for quick symbol lookups (function names, class names, error messages).
- Use `read_file_topology` before modifying a file to understand its structure and dependencies.
- Use `repo_summary` for onboarding or to get a quick overview of the codebase structure.
- Use `neighbor_signatures` to discover imports/dependencies of specific files before editing.
- Use `batch_context` when you need context for multiple queries — saves round-trips.
- Call `remember_turn` for important decisions, preferences, or context the user shares.
- Call `recall_history` when the user says "we discussed", "last time", or references prior work.
"#;

/// Write the Cursor rule file at `.cursor/rules/trevec.mdc`.
/// Returns `true` if the file was written (created or overwritten).
pub fn write_cursor_rule(repo_path: &Path) -> Result<bool> {
    let rules_dir = repo_path.join(".cursor/rules");
    std::fs::create_dir_all(&rules_dir)
        .with_context(|| format!("Failed to create {}", rules_dir.display()))?;
    let rule_path = rules_dir.join("trevec.mdc");
    std::fs::write(&rule_path, CURSOR_RULE)
        .with_context(|| format!("Failed to write {}", rule_path.display()))?;
    eprintln!("  Cursor rule: {}", rule_path.display());
    Ok(true)
}

/// Insert or replace a marker-delimited block in a markdown file.
/// If the file doesn't exist, creates it. If markers already exist, replaces
/// the content between them. Otherwise appends the block.
/// Returns `true` if the file was modified.
pub fn upsert_markdown_rule(file_path: &Path, block: &str) -> Result<bool> {
    let wrapped = format!("{MARKER_START}\n{block}\n{MARKER_END}\n");

    let existing = if file_path.exists() {
        std::fs::read_to_string(file_path)
            .with_context(|| format!("Failed to read {}", file_path.display()))?
    } else {
        String::new()
    };

    let new_content = if let (Some(start), Some(end)) = (
        existing.find(MARKER_START),
        existing.find(MARKER_END),
    ) {
        let end = end + MARKER_END.len();
        // Consume trailing newline if present
        let end = if existing[end..].starts_with('\n') {
            end + 1
        } else {
            end
        };
        format!("{}{}{}", &existing[..start], wrapped, &existing[end..])
    } else if existing.is_empty() {
        wrapped
    } else {
        // Append with a blank line separator
        let separator = if existing.ends_with('\n') { "\n" } else { "\n\n" };
        format!("{existing}{separator}{wrapped}")
    };

    if let Some(parent) = file_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create {}", parent.display()))?;
    }
    std::fs::write(file_path, &new_content)
        .with_context(|| format!("Failed to write {}", file_path.display()))?;
    eprintln!("  Rule block: {}", file_path.display());
    Ok(true)
}

/// Remove the Cursor rule file.
/// Returns `true` if the file existed and was removed.
#[cfg_attr(not(test), allow(dead_code))]
pub fn remove_cursor_rule(repo_path: &Path) -> Result<bool> {
    let rule_path = repo_path.join(".cursor/rules/trevec.mdc");
    if rule_path.exists() {
        std::fs::remove_file(&rule_path)
            .with_context(|| format!("Failed to remove {}", rule_path.display()))?;
        eprintln!("  Removed: {}", rule_path.display());
        Ok(true)
    } else {
        Ok(false)
    }
}

/// Remove the marker-delimited block from a markdown file.
/// Returns `true` if the block was found and removed.
#[cfg_attr(not(test), allow(dead_code))]
pub fn remove_markdown_rule(file_path: &Path) -> Result<bool> {
    if !file_path.exists() {
        return Ok(false);
    }

    let content = std::fs::read_to_string(file_path)
        .with_context(|| format!("Failed to read {}", file_path.display()))?;

    let (Some(start), Some(end)) = (content.find(MARKER_START), content.find(MARKER_END)) else {
        return Ok(false);
    };

    let end = end + MARKER_END.len();
    // Consume trailing newline if present
    let end = if content[end..].starts_with('\n') {
        end + 1
    } else {
        end
    };

    // Also remove a leading blank line that was used as separator
    let start = if start > 0 && content[..start].ends_with('\n') {
        start - 1
    } else {
        start
    };

    let new_content = format!("{}{}", &content[..start], &content[end..]);
    std::fs::write(file_path, &new_content)
        .with_context(|| format!("Failed to write {}", file_path.display()))?;
    eprintln!("  Removed rule block: {}", file_path.display());
    Ok(true)
}

/// Check whether a rule file/block is present for the given client.
pub fn is_rule_present(repo_path: &Path, client_key: &str) -> bool {
    match client_key {
        "cursor" => repo_path.join(".cursor/rules/trevec.mdc").exists(),
        "claude-code" => {
            let path = repo_path.join("CLAUDE.md");
            path.exists()
                && std::fs::read_to_string(&path)
                    .map(|c| c.contains(MARKER_START))
                    .unwrap_or(false)
        }
        "codex" => {
            let path = repo_path.join("AGENTS.md");
            path.exists()
                && std::fs::read_to_string(&path)
                    .map(|c| c.contains(MARKER_START))
                    .unwrap_or(false)
        }
        _ => false,
    }
}

/// Write rules for the specified clients.
pub fn write_rules_for_clients(repo_path: &Path, client_names: &[&str]) -> Result<()> {
    for name in client_names {
        match *name {
            "Cursor" | "cursor" => {
                write_cursor_rule(repo_path)?;
            }
            "Claude Code" | "claude-code" => {
                upsert_markdown_rule(&repo_path.join("CLAUDE.md"), CLAUDE_CODE_RULE)?;
            }
            "Codex" | "codex" => {
                upsert_markdown_rule(&repo_path.join("AGENTS.md"), CODEX_RULE)?;
            }
            // Claude Desktop has no rule file concept
            _ => {}
        }
    }
    Ok(())
}

/// Remove rules for the specified clients.
#[cfg_attr(not(test), allow(dead_code))]
pub fn remove_rules_for_clients(repo_path: &Path, client_names: &[&str]) -> Result<()> {
    for name in client_names {
        match *name {
            "Cursor" | "cursor" => {
                remove_cursor_rule(repo_path)?;
            }
            "Claude Code" | "claude-code" => {
                remove_markdown_rule(&repo_path.join("CLAUDE.md"))?;
            }
            "Codex" | "codex" => {
                remove_markdown_rule(&repo_path.join("AGENTS.md"))?;
            }
            _ => {}
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_write_cursor_rule_creates_file() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = tmp.path();

        let written = write_cursor_rule(repo).unwrap();
        assert!(written);

        let rule_path = repo.join(".cursor/rules/trevec.mdc");
        assert!(rule_path.exists());
        let content = std::fs::read_to_string(&rule_path).unwrap();
        assert!(content.contains("get_context"));
        assert!(content.contains("alwaysApply: true"));
    }

    #[test]
    fn test_write_cursor_rule_overwrites() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = tmp.path();

        write_cursor_rule(repo).unwrap();
        // Overwrite with same content
        write_cursor_rule(repo).unwrap();

        let rule_path = repo.join(".cursor/rules/trevec.mdc");
        let content = std::fs::read_to_string(&rule_path).unwrap();
        assert!(content.contains("get_context"));
        // Only one file, no duplication
        assert_eq!(content.matches("get_context").count(), CURSOR_RULE.matches("get_context").count());
    }

    #[test]
    fn test_upsert_markdown_creates_file() {
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("CLAUDE.md");

        upsert_markdown_rule(&file, CLAUDE_CODE_RULE).unwrap();

        let content = std::fs::read_to_string(&file).unwrap();
        assert!(content.contains(MARKER_START));
        assert!(content.contains(MARKER_END));
        assert!(content.contains("get_context"));
    }

    #[test]
    fn test_upsert_markdown_preserves_existing() {
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("CLAUDE.md");

        // Write some pre-existing content
        std::fs::write(&file, "# My Project\n\nSome existing content.\n").unwrap();

        upsert_markdown_rule(&file, CLAUDE_CODE_RULE).unwrap();

        let content = std::fs::read_to_string(&file).unwrap();
        assert!(content.contains("# My Project"));
        assert!(content.contains("Some existing content."));
        assert!(content.contains(MARKER_START));
        assert!(content.contains("get_context"));
    }

    #[test]
    fn test_upsert_markdown_idempotent() {
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("CLAUDE.md");

        std::fs::write(&file, "# My Project\n").unwrap();

        upsert_markdown_rule(&file, CLAUDE_CODE_RULE).unwrap();
        upsert_markdown_rule(&file, CLAUDE_CODE_RULE).unwrap();

        let content = std::fs::read_to_string(&file).unwrap();
        // Only one marker pair after two upserts
        assert_eq!(content.matches(MARKER_START).count(), 1);
        assert_eq!(content.matches(MARKER_END).count(), 1);
        assert!(content.contains("# My Project"));
    }

    #[test]
    fn test_remove_cursor_rule() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = tmp.path();

        write_cursor_rule(repo).unwrap();
        assert!(repo.join(".cursor/rules/trevec.mdc").exists());

        let removed = remove_cursor_rule(repo).unwrap();
        assert!(removed);
        assert!(!repo.join(".cursor/rules/trevec.mdc").exists());
    }

    #[test]
    fn test_remove_cursor_rule_when_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let removed = remove_cursor_rule(tmp.path()).unwrap();
        assert!(!removed);
    }

    #[test]
    fn test_remove_markdown_rule() {
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("CLAUDE.md");

        std::fs::write(&file, "# My Project\n").unwrap();
        upsert_markdown_rule(&file, CLAUDE_CODE_RULE).unwrap();

        let removed = remove_markdown_rule(&file).unwrap();
        assert!(removed);

        let content = std::fs::read_to_string(&file).unwrap();
        assert!(!content.contains(MARKER_START));
        assert!(!content.contains("get_context"));
        assert!(content.contains("# My Project"));
    }

    #[test]
    fn test_remove_markdown_rule_when_no_block() {
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("AGENTS.md");

        std::fs::write(&file, "# Agents\n").unwrap();

        let removed = remove_markdown_rule(&file).unwrap();
        assert!(!removed);
    }

    #[test]
    fn test_remove_markdown_rule_when_no_file() {
        let tmp = tempfile::tempdir().unwrap();
        let file = tmp.path().join("AGENTS.md");

        let removed = remove_markdown_rule(&file).unwrap();
        assert!(!removed);
    }

    #[test]
    fn test_is_rule_present_cursor() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = tmp.path();

        assert!(!is_rule_present(repo, "cursor"));
        write_cursor_rule(repo).unwrap();
        assert!(is_rule_present(repo, "cursor"));
    }

    #[test]
    fn test_is_rule_present_claude_code() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = tmp.path();

        assert!(!is_rule_present(repo, "claude-code"));
        upsert_markdown_rule(&repo.join("CLAUDE.md"), CLAUDE_CODE_RULE).unwrap();
        assert!(is_rule_present(repo, "claude-code"));
    }

    #[test]
    fn test_is_rule_present_codex() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = tmp.path();

        assert!(!is_rule_present(repo, "codex"));
        upsert_markdown_rule(&repo.join("AGENTS.md"), CODEX_RULE).unwrap();
        assert!(is_rule_present(repo, "codex"));
    }

    #[test]
    fn test_write_and_remove_rules_for_clients() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = tmp.path();

        write_rules_for_clients(repo, &["Cursor", "Claude Code", "Codex"]).unwrap();
        assert!(is_rule_present(repo, "cursor"));
        assert!(is_rule_present(repo, "claude-code"));
        assert!(is_rule_present(repo, "codex"));

        remove_rules_for_clients(repo, &["Cursor", "Claude Code", "Codex"]).unwrap();
        assert!(!is_rule_present(repo, "cursor"));
        assert!(!is_rule_present(repo, "claude-code"));
        assert!(!is_rule_present(repo, "codex"));
    }
}
