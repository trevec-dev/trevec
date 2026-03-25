use anyhow::{Context, Result};
use std::path::{Path, PathBuf};

pub(crate) const DEFAULT_CONFIG: &str = r#"# Trevec configuration
# Uncomment and modify settings as needed.

# [index]
# Glob patterns to exclude from indexing (in addition to .gitignore).
# exclude = ["vendor/**", "node_modules/**", "*.generated.*"]

# [retrieval]
# Number of anchor nodes for context assembly.
# anchors = 5
# Token budget for context assembly.
# budget = 4096
# Demote test files in results (0.0 = disabled, 0.5 = default, 1.0 = full suppression).
# test_file_penalty = 0.5
# Additional path patterns to demote (substring match).
# penalty_paths = ["examples/", "fixtures/"]

# [embeddings]
# Model name for local embeddings.
# model = "BAAI/bge-small-en-v1.5"

# [memory]
# Episodic memory: persistent, searchable records of AI chat sessions.
# All memory sources are enabled by default. Set to false to disable.
# enabled = true
# sources = ["cursor", "claude_code", "codex", "trevec_tool_calls"]
# retention_days = 30
# max_events = 200000
# semantic = true
# redaction_mode = "strict"
#
# [memory.cursor]
# enabled = true
#
# [memory.claude_code]
# enabled = true
#
# [memory.codex]
# enabled = true
"#;

pub(crate) const TREVEC_GITIGNORE: &str = "# Trevec data — do not commit\n*\n";

/// Run the init command: create .trevec/ with config, gitignore, manifest,
/// then optionally index, write IDE rules, and register the project.
pub async fn run(path: PathBuf, no_index: bool, no_rules: bool) -> Result<()> {
    let path = path.canonicalize().unwrap_or(path);
    let trevec_dir = path.join(".trevec");

    let already_existed = trevec_dir.exists();
    std::fs::create_dir_all(&trevec_dir).context("Failed to create .trevec directory")?;

    let created = write_if_missing(&trevec_dir.join("config.toml"), DEFAULT_CONFIG)?;
    let gitignore_created = write_if_missing(&trevec_dir.join(".gitignore"), TREVEC_GITIGNORE)?;
    let manifest_created = write_if_missing(&trevec_dir.join("manifest.json"), "{}")?;

    if already_existed && !created && !gitignore_created && !manifest_created {
        eprintln!("Already initialized at {}", trevec_dir.display());
    } else {
        eprintln!("Initialized trevec at {}", trevec_dir.display());
        if created {
            eprintln!("  Created config.toml");
        }
        if gitignore_created {
            eprintln!("  Created .gitignore");
        }
        if manifest_created {
            eprintln!("  Created manifest.json");
        }
    }

    crate::telemetry::maybe_show_first_run_notice();

    // Index (unless --no-index)
    if !no_index {
        crate::commands::index::run(path.clone(), trevec_dir.clone(), false, false).await?;
    } else {
        eprintln!("\nRun `trevec index` when ready to index.");
    }

    // Write IDE rule files (unless --no-rules)
    if !no_rules {
        eprintln!("\nWriting IDE rules...");
        crate::commands::rules::write_rules_for_clients(
            &path,
            &["Cursor", "Claude Code", "Codex"],
        )?;
    }

    // Register in project registry
    crate::commands::projects::register_project(&path);

    crate::telemetry::capture("cli_init", serde_json::json!({
        "indexed": !no_index,
        "rules": !no_rules,
    }));

    Ok(())
}

/// Write a file only if it doesn't already exist. Returns true if created.
pub(crate) fn write_if_missing(path: &Path, content: &str) -> Result<bool> {
    if path.exists() {
        return Ok(false);
    }
    std::fs::write(path, content)
        .with_context(|| format!("Failed to write {}", path.display()))?;
    Ok(true)
}
