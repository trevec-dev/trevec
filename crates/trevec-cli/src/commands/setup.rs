use anyhow::{Context, Result};
use std::io::{self, Write};
use std::path::{Path, PathBuf};

/// Config format for an MCP target.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ConfigFormat {
    /// JSON with `mcpServers.trevec` (Claude Desktop, Cursor, Claude Code)
    Json,
    /// TOML with `[mcp_servers.trevec]` (Codex)
    Toml,
}

pub(crate) struct Target {
    pub name: &'static str,
    pub config_path: PathBuf,
    pub format: ConfigFormat,
}

/// Supported client filter values.
const VALID_CLIENTS: &[&str] = &["all", "claude", "claude-code", "cursor", "codex"];

/// Run the setup command: write global MCP configs for selected clients.
/// This is a one-time, global operation — no repo path needed.
pub async fn run(
    client: &str,
    auto_yes: bool,
) -> Result<()> {
    // Validate client parameter BEFORE any side effects.
    if !VALID_CLIENTS.contains(&client.to_lowercase().as_str()) {
        anyhow::bail!(
            "Unknown client '{client}'. Supported values: claude, claude-code, cursor, codex, all"
        );
    }

    let targets = discover_global_targets(client);
    if targets.is_empty() {
        eprintln!("No supported IDE configurations found for client '{client}'.");
        eprintln!("Supported: claude, claude-code, cursor, codex, all");
        return Ok(());
    }

    // UX summary
    let trevec_bin = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("trevec"));
    let version = env!("CARGO_PKG_VERSION");
    let client_names: Vec<&str> = targets.iter().map(|t| t.name).collect();

    eprintln!("Trevec MCP Setup (global)");
    eprintln!("  Binary:     {} (v{})", trevec_bin.display(), version);
    eprintln!("  Clients:    {}", client_names.join(", "));

    for target in &targets {
        match target.format {
            ConfigFormat::Json => install_json_config_global(target, auto_yes)?,
            ConfigFormat::Toml => install_toml_config_global(target, auto_yes)?,
        }
    }

    eprintln!("\nDone! Now run `trevec init` inside each project you want to use with Trevec.");

    crate::telemetry::capture("cli_mcp_setup", serde_json::json!({
        "clients_configured": client_names,
    }));

    Ok(())
}

/// Remove global MCP server config for the filtered clients.
pub async fn run_remove(
    client: &str,
    auto_yes: bool,
) -> Result<()> {
    // Validate client parameter BEFORE any side effects.
    if !VALID_CLIENTS.contains(&client.to_lowercase().as_str()) {
        anyhow::bail!(
            "Unknown client '{client}'. Supported values: claude, claude-code, cursor, codex, all"
        );
    }

    let targets = discover_global_targets(client);
    if targets.is_empty() {
        eprintln!("No supported IDE configurations found for client '{client}'.");
        eprintln!("Supported: claude, claude-code, cursor, codex, all");
        return Ok(());
    }

    for target in &targets {
        match target.format {
            ConfigFormat::Json => remove_json_config(target, auto_yes)?,
            ConfigFormat::Toml => remove_toml_config(target, auto_yes)?,
        }
    }

    Ok(())
}

/// Discover global config paths for MCP setup (one-time, not per-repo).
pub(crate) fn discover_global_targets(client: &str) -> Vec<Target> {
    let mut targets = Vec::new();
    let client = client.to_lowercase();

    let home = match std::env::var("HOME") {
        Ok(h) => PathBuf::from(h),
        Err(_) => return targets,
    };

    // Claude Desktop (macOS — global config)
    if client == "all" || client == "claude" {
        let claude_config = home
            .join("Library/Application Support/Claude/claude_desktop_config.json");
        if claude_config.parent().is_some_and(|p| p.exists()) {
            targets.push(Target {
                name: "Claude Desktop",
                config_path: claude_config,
                format: ConfigFormat::Json,
            });
        }
    }

    // Cursor (global ~/.cursor/mcp.json)
    if client == "all" || client == "cursor" {
        targets.push(Target {
            name: "Cursor",
            config_path: home.join(".cursor/mcp.json"),
            format: ConfigFormat::Json,
        });
    }

    // Claude Code (user scope ~/.claude.json)
    if client == "all" || client == "claude-code" {
        targets.push(Target {
            name: "Claude Code",
            config_path: home.join(".claude.json"),
            format: ConfigFormat::Json,
        });
    }

    // Codex (global ~/.codex/config.toml)
    if client == "all" || client == "codex" {
        targets.push(Target {
            name: "Codex",
            config_path: home.join(".codex/config.toml"),
            format: ConfigFormat::Toml,
        });
    }

    targets
}

/// Discover per-repo config targets. Still needed by doctor.rs and projects.rs
/// for checking legacy per-repo configs.
pub(crate) fn discover_targets(repo_path: &Path, client: &str) -> Vec<Target> {
    let mut targets = Vec::new();
    let client = client.to_lowercase();

    // Claude Desktop (macOS — global config)
    if client == "all" || client == "claude" {
        if let Ok(home) = std::env::var("HOME") {
            let claude_config = PathBuf::from(&home)
                .join("Library/Application Support/Claude/claude_desktop_config.json");
            if claude_config.parent().is_some_and(|p| p.exists()) {
                targets.push(Target {
                    name: "Claude Desktop",
                    config_path: claude_config,
                    format: ConfigFormat::Json,
                });
            }
        }
    }

    // Claude Code (project-level .mcp.json at repo root)
    if client == "all" || client == "claude-code" {
        targets.push(Target {
            name: "Claude Code",
            config_path: repo_path.join(".mcp.json"),
            format: ConfigFormat::Json,
        });
    }

    // Cursor (project-level .cursor/mcp.json)
    if client == "all" || client == "cursor" {
        targets.push(Target {
            name: "Cursor",
            config_path: repo_path.join(".cursor/mcp.json"),
            format: ConfigFormat::Json,
        });
    }

    // Codex (project-level .codex/config.toml)
    if client == "all" || client == "codex" {
        targets.push(Target {
            name: "Codex",
            config_path: repo_path.join(".codex/config.toml"),
            format: ConfigFormat::Toml,
        });
    }

    targets
}

/// Check whether a given IDE client has trevec configured **for this specific repo**.
///
/// For project-scoped configs (Cursor, Claude Code, Codex), the config lives
/// inside the repo so its mere existence with a trevec entry implies this repo.
/// For global configs (Claude Desktop), verifies that the `--path` arg in the
/// trevec entry matches `repo_path`.
pub fn is_client_configured(repo_path: &Path, client_key: &str) -> bool {
    let canonical = repo_path
        .canonicalize()
        .unwrap_or_else(|_| repo_path.to_path_buf());
    let repo_str = canonical.to_string_lossy();

    let targets = discover_targets(&canonical, client_key);
    for target in &targets {
        if !target.config_path.exists() {
            continue;
        }

        match target.format {
            ConfigFormat::Json => {
                let Ok(content) = std::fs::read_to_string(&target.config_path) else {
                    continue;
                };
                let Ok(config) = serde_json::from_str::<serde_json::Value>(&content) else {
                    continue;
                };
                let Some(entry) = config.get("mcpServers").and_then(|s| s.get("trevec")) else {
                    continue;
                };

                // Project-scoped configs live inside the repo.
                let config_canonical = target
                    .config_path
                    .canonicalize()
                    .unwrap_or_else(|_| target.config_path.clone());
                if config_canonical.starts_with(&canonical) {
                    return true;
                }

                // Global configs (Claude Desktop): check that the --path arg matches.
                if let Some(args) = entry.get("args").and_then(|a| a.as_array()) {
                    let args_strs: Vec<&str> = args.iter().filter_map(|v| v.as_str()).collect();
                    for pair in args_strs.windows(2) {
                        if pair[0] == "--path" && pair[1] == repo_str.as_ref() {
                            return true;
                        }
                    }
                }
            }
            ConfigFormat::Toml => {
                let Ok(content) = std::fs::read_to_string(&target.config_path) else {
                    continue;
                };
                let Ok(config) = content.parse::<toml::Table>() else {
                    continue;
                };
                if config
                    .get("mcp_servers")
                    .and_then(|s| s.get("trevec"))
                    .is_some()
                {
                    // Codex config is project-scoped → existence implies this repo
                    return true;
                }
            }
        }
    }
    false
}

/// Check whether trevec is configured globally for a given client.
pub fn is_globally_configured(client_key: &str) -> bool {
    let targets = discover_global_targets(client_key);
    for target in &targets {
        if !target.config_path.exists() {
            continue;
        }
        match target.format {
            ConfigFormat::Json => {
                let Ok(content) = std::fs::read_to_string(&target.config_path) else {
                    continue;
                };
                let Ok(config) = serde_json::from_str::<serde_json::Value>(&content) else {
                    continue;
                };
                if config.get("mcpServers").and_then(|s| s.get("trevec")).is_some() {
                    return true;
                }
            }
            ConfigFormat::Toml => {
                let Ok(content) = std::fs::read_to_string(&target.config_path) else {
                    continue;
                };
                let Ok(config) = content.parse::<toml::Table>() else {
                    continue;
                };
                if config.get("mcp_servers").and_then(|s| s.get("trevec")).is_some() {
                    return true;
                }
            }
        }
    }
    false
}

/// Extract the configured binary from a global config for a given client.
pub(crate) fn get_global_configured_binary(client_key: &str) -> Option<PathBuf> {
    let targets = discover_global_targets(client_key);
    for target in &targets {
        if !target.config_path.exists() {
            continue;
        }
        match target.format {
            ConfigFormat::Json => {
                let content = std::fs::read_to_string(&target.config_path).ok()?;
                let config: serde_json::Value = serde_json::from_str(&content).ok()?;
                let command = config
                    .get("mcpServers")
                    .and_then(|s| s.get("trevec"))
                    .and_then(|t| t.get("command"))
                    .and_then(|c| c.as_str())?;
                return Some(PathBuf::from(command));
            }
            ConfigFormat::Toml => {
                let content = std::fs::read_to_string(&target.config_path).ok()?;
                let config: toml::Table = content.parse().ok()?;
                let command = config
                    .get("mcp_servers")
                    .and_then(|s| s.get("trevec"))
                    .and_then(|t| t.get("command"))
                    .and_then(|c| c.as_str())?;
                return Some(PathBuf::from(command));
            }
        }
    }
    None
}

/// Extract the configured binary command from a client's per-repo MCP config.
/// Returns `None` if the config doesn't exist or has no trevec entry.
#[cfg_attr(not(test), allow(dead_code))]
pub(crate) fn get_configured_binary(repo_path: &Path, client_key: &str) -> Option<PathBuf> {
    let canonical = repo_path
        .canonicalize()
        .unwrap_or_else(|_| repo_path.to_path_buf());
    let targets = discover_targets(&canonical, client_key);

    for target in &targets {
        if !target.config_path.exists() {
            continue;
        }

        match target.format {
            ConfigFormat::Json => {
                let content = std::fs::read_to_string(&target.config_path).ok()?;
                let config: serde_json::Value = serde_json::from_str(&content).ok()?;
                let command = config
                    .get("mcpServers")
                    .and_then(|s| s.get("trevec"))
                    .and_then(|t| t.get("command"))
                    .and_then(|c| c.as_str())?;
                return Some(PathBuf::from(command));
            }
            ConfigFormat::Toml => {
                let content = std::fs::read_to_string(&target.config_path).ok()?;
                let config: toml::Table = content.parse().ok()?;
                let command = config
                    .get("mcp_servers")
                    .and_then(|s| s.get("trevec"))
                    .and_then(|t| t.get("command"))
                    .and_then(|c| c.as_str())?;
                return Some(PathBuf::from(command));
            }
        }
    }
    None
}

/// Return `"trevec"` if the binary is on PATH, else the absolute path.
fn resolve_trevec_command() -> String {
    // Check if "trevec" resolves to something on PATH via `which`
    if let Ok(output) = std::process::Command::new("which")
        .arg("trevec")
        .output()
    {
        if output.status.success() {
            return "trevec".to_string();
        }
    }
    // Fall back to absolute path of current binary
    let exe = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("trevec"));
    eprintln!(
        "  Note: 'trevec' not found on PATH. Using absolute path: {}",
        exe.display()
    );
    exe.to_string_lossy().to_string()
}

// ---------------------------------------------------------------------------
// JSON install / remove (global configs)
// ---------------------------------------------------------------------------

fn install_json_config_global(target: &Target, auto_yes: bool) -> Result<()> {
    eprintln!(
        "\n{}: {}",
        target.name,
        target.config_path.display()
    );

    // Use "trevec" if on PATH, else fall back to absolute binary path
    let trevec_bin = resolve_trevec_command();

    let trevec_entry = serde_json::json!({
        "command": trevec_bin,
        "args": ["serve"]
    });

    eprintln!("  Will add mcpServers.trevec:");
    eprintln!("    {}", serde_json::to_string_pretty(&trevec_entry).unwrap().replace('\n', "\n    "));

    if !confirm("  Continue?", auto_yes)? {
        eprintln!("  Skipped.");
        return Ok(());
    }

    // Load existing config or start fresh
    let mut config: serde_json::Value = if target.config_path.exists() {
        let backup = target.config_path.with_extension("json.bak");
        std::fs::copy(&target.config_path, &backup)
            .with_context(|| format!("Failed to back up {}", target.config_path.display()))?;
        eprintln!("  Backed up to {}", backup.display());

        let content = std::fs::read_to_string(&target.config_path)
            .with_context(|| format!("Failed to read {}", target.config_path.display()))?;
        serde_json::from_str(&content)
            .with_context(|| format!("Failed to parse {}", target.config_path.display()))?
    } else {
        serde_json::json!({})
    };

    // Merge mcpServers.trevec
    let servers = config
        .as_object_mut()
        .context("Config is not a JSON object")?
        .entry("mcpServers")
        .or_insert_with(|| serde_json::json!({}));
    servers
        .as_object_mut()
        .context("mcpServers is not a JSON object")?
        .insert("trevec".to_string(), trevec_entry);

    // Ensure parent dir exists
    if let Some(parent) = target.config_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create {}", parent.display()))?;
    }

    let output = serde_json::to_string_pretty(&config)?;
    std::fs::write(&target.config_path, output)
        .with_context(|| format!("Failed to write {}", target.config_path.display()))?;

    eprintln!("  Done.");
    Ok(())
}

fn remove_json_config(target: &Target, auto_yes: bool) -> Result<()> {
    eprintln!(
        "\n{}: {}",
        target.name,
        target.config_path.display()
    );

    if !target.config_path.exists() {
        eprintln!("  Config file does not exist, nothing to remove.");
        return Ok(());
    }

    let content = std::fs::read_to_string(&target.config_path)
        .with_context(|| format!("Failed to read {}", target.config_path.display()))?;
    let mut config: serde_json::Value = serde_json::from_str(&content)
        .with_context(|| format!("Failed to parse {}", target.config_path.display()))?;

    let removed = config
        .as_object_mut()
        .and_then(|obj| obj.get_mut("mcpServers"))
        .and_then(|servers| servers.as_object_mut())
        .and_then(|servers| servers.remove("trevec"))
        .is_some();

    if !removed {
        eprintln!("  No trevec entry found, nothing to remove.");
        return Ok(());
    }

    if !confirm("  Will remove mcpServers.trevec. Continue?", auto_yes)? {
        eprintln!("  Skipped.");
        return Ok(());
    }

    let backup = target.config_path.with_extension("json.bak");
    std::fs::copy(&target.config_path, &backup)
        .with_context(|| format!("Failed to back up {}", target.config_path.display()))?;
    eprintln!("  Backed up to {}", backup.display());

    let output = serde_json::to_string_pretty(&config)?;
    std::fs::write(&target.config_path, output)
        .with_context(|| format!("Failed to write {}", target.config_path.display()))?;

    eprintln!("  Removed.");
    Ok(())
}

// ---------------------------------------------------------------------------
// TOML install / remove (Codex)
// ---------------------------------------------------------------------------

fn install_toml_config_global(target: &Target, auto_yes: bool) -> Result<()> {
    eprintln!(
        "\n{}: {}",
        target.name,
        target.config_path.display()
    );

    let trevec_bin = resolve_trevec_command();

    eprintln!("  Will add [mcp_servers.trevec]:");
    eprintln!("    command = \"{}\"", trevec_bin);
    eprintln!("    args = [\"serve\"]");

    if !confirm("  Continue?", auto_yes)? {
        eprintln!("  Skipped.");
        return Ok(());
    }

    // Load existing config or start fresh
    let mut config: toml::Table = if target.config_path.exists() {
        let backup = target.config_path.with_extension("toml.bak");
        std::fs::copy(&target.config_path, &backup)
            .with_context(|| format!("Failed to back up {}", target.config_path.display()))?;
        eprintln!("  Backed up to {}", backup.display());

        let content = std::fs::read_to_string(&target.config_path)
            .with_context(|| format!("Failed to read {}", target.config_path.display()))?;
        content
            .parse()
            .with_context(|| format!("Failed to parse {}", target.config_path.display()))?
    } else {
        toml::Table::new()
    };

    // Build the trevec server entry
    let mut trevec_entry = toml::Table::new();
    trevec_entry.insert(
        "command".to_string(),
        toml::Value::String(trevec_bin),
    );
    trevec_entry.insert(
        "args".to_string(),
        toml::Value::Array(vec![
            toml::Value::String("serve".to_string()),
        ]),
    );

    // Merge into mcp_servers table
    let mcp_servers = config
        .entry("mcp_servers")
        .or_insert_with(|| toml::Value::Table(toml::Table::new()));
    let servers_table = mcp_servers
        .as_table_mut()
        .context("mcp_servers is not a TOML table")?;
    servers_table.insert("trevec".to_string(), toml::Value::Table(trevec_entry));

    // Ensure parent dir exists
    if let Some(parent) = target.config_path.parent() {
        std::fs::create_dir_all(parent)
            .with_context(|| format!("Failed to create {}", parent.display()))?;
    }

    let output = toml::to_string_pretty(&config)?;
    std::fs::write(&target.config_path, output)
        .with_context(|| format!("Failed to write {}", target.config_path.display()))?;

    eprintln!("  Done.");
    Ok(())
}

fn remove_toml_config(target: &Target, auto_yes: bool) -> Result<()> {
    eprintln!(
        "\n{}: {}",
        target.name,
        target.config_path.display()
    );

    if !target.config_path.exists() {
        eprintln!("  Config file does not exist, nothing to remove.");
        return Ok(());
    }

    let content = std::fs::read_to_string(&target.config_path)
        .with_context(|| format!("Failed to read {}", target.config_path.display()))?;
    let mut config: toml::Table = content
        .parse()
        .with_context(|| format!("Failed to parse {}", target.config_path.display()))?;

    let removed = config
        .get_mut("mcp_servers")
        .and_then(|s| s.as_table_mut())
        .and_then(|servers| servers.remove("trevec"))
        .is_some();

    if !removed {
        eprintln!("  No trevec entry found, nothing to remove.");
        return Ok(());
    }

    if !confirm("  Will remove [mcp_servers.trevec]. Continue?", auto_yes)? {
        eprintln!("  Skipped.");
        return Ok(());
    }

    let backup = target.config_path.with_extension("toml.bak");
    std::fs::copy(&target.config_path, &backup)
        .with_context(|| format!("Failed to back up {}", target.config_path.display()))?;
    eprintln!("  Backed up to {}", backup.display());

    let output = toml::to_string_pretty(&config)?;
    std::fs::write(&target.config_path, output)
        .with_context(|| format!("Failed to write {}", target.config_path.display()))?;

    eprintln!("  Removed.");
    Ok(())
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

fn confirm(prompt: &str, auto_yes: bool) -> Result<bool> {
    if auto_yes {
        eprintln!("{} [y/N] y (auto)", prompt);
        return Ok(true);
    }
    eprint!("{} [y/N] ", prompt);
    io::stderr().flush()?;
    let mut input = String::new();
    io::stdin().read_line(&mut input)?;
    Ok(input.trim().eq_ignore_ascii_case("y"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_valid_clients() {
        assert!(VALID_CLIENTS.contains(&"all"));
        assert!(VALID_CLIENTS.contains(&"claude"));
        assert!(VALID_CLIENTS.contains(&"claude-code"));
        assert!(VALID_CLIENTS.contains(&"cursor"));
        assert!(VALID_CLIENTS.contains(&"codex"));
        assert!(!VALID_CLIENTS.contains(&"vscode"));
    }

    #[test]
    fn test_cursor_configured_for_correct_repo() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = tmp.path().join("my-project");
        std::fs::create_dir_all(repo.join(".cursor")).unwrap();

        let config = serde_json::json!({
            "mcpServers": {
                "trevec": {
                    "command": "/usr/local/bin/trevec",
                    "args": ["serve", "--path", repo.to_string_lossy(), "--data-dir", repo.join(".trevec").to_string_lossy()]
                }
            }
        });
        std::fs::write(
            repo.join(".cursor/mcp.json"),
            serde_json::to_string_pretty(&config).unwrap(),
        )
        .unwrap();

        assert!(is_client_configured(&repo, "cursor"));
    }

    #[test]
    fn test_cursor_not_configured_without_entry() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = tmp.path().join("my-project");
        std::fs::create_dir_all(repo.join(".cursor")).unwrap();

        let config = serde_json::json!({ "mcpServers": {} });
        std::fs::write(
            repo.join(".cursor/mcp.json"),
            serde_json::to_string_pretty(&config).unwrap(),
        )
        .unwrap();

        assert!(!is_client_configured(&repo, "cursor"));
    }

    #[test]
    fn test_cursor_config_does_not_leak_to_other_repo() {
        let tmp = tempfile::tempdir().unwrap();
        let repo_a = tmp.path().join("repo-a");
        let repo_b = tmp.path().join("repo-b");
        std::fs::create_dir_all(repo_a.join(".cursor")).unwrap();
        std::fs::create_dir_all(&repo_b).unwrap();

        let config = serde_json::json!({
            "mcpServers": {
                "trevec": { "command": "trevec", "args": ["serve"] }
            }
        });
        std::fs::write(
            repo_a.join(".cursor/mcp.json"),
            serde_json::to_string_pretty(&config).unwrap(),
        )
        .unwrap();

        assert!(is_client_configured(&repo_a, "cursor"));
        assert!(!is_client_configured(&repo_b, "cursor"));
    }

    #[test]
    fn test_global_config_matches_correct_repo_path() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = tmp.path().join("my-project");
        std::fs::create_dir_all(&repo).unwrap();

        let config_path = tmp.path().join("claude_config.json");
        let config = serde_json::json!({
            "mcpServers": {
                "trevec": {
                    "command": "/usr/local/bin/trevec",
                    "args": ["serve", "--path", repo.to_string_lossy(), "--data-dir", repo.join(".trevec").to_string_lossy()]
                }
            }
        });
        std::fs::write(
            &config_path,
            serde_json::to_string_pretty(&config).unwrap(),
        )
        .unwrap();

        let content = std::fs::read_to_string(&config_path).unwrap();
        let parsed: serde_json::Value = serde_json::from_str(&content).unwrap();
        let entry = parsed.get("mcpServers").unwrap().get("trevec").unwrap();
        let args = entry.get("args").unwrap().as_array().unwrap();
        let args_strs: Vec<&str> = args.iter().filter_map(|v| v.as_str()).collect();

        let repo_str = repo.to_string_lossy();
        let has_match = args_strs
            .windows(2)
            .any(|pair| pair[0] == "--path" && pair[1] == repo_str.as_ref());
        assert!(has_match);

        let other = tmp.path().join("other-project");
        let other_str = other.to_string_lossy();
        let has_other = args_strs
            .windows(2)
            .any(|pair| pair[0] == "--path" && pair[1] == other_str.as_ref());
        assert!(!has_other);
    }

    #[test]
    fn test_discover_targets_filters_by_client() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = tmp.path();

        // "cursor" only → exactly 1 target
        let cursor_only = discover_targets(repo, "cursor");
        assert_eq!(cursor_only.len(), 1);
        assert_eq!(cursor_only[0].name, "Cursor");

        // "claude-code" only → exactly 1 target
        let cc_only = discover_targets(repo, "claude-code");
        assert_eq!(cc_only.len(), 1);
        assert_eq!(cc_only[0].name, "Claude Code");

        // "codex" only → exactly 1 target
        let codex_only = discover_targets(repo, "codex");
        assert_eq!(codex_only.len(), 1);
        assert_eq!(codex_only[0].name, "Codex");
        assert_eq!(codex_only[0].format, ConfigFormat::Toml);

        // "claude" should never produce non-Claude-Desktop targets
        let claude_only = discover_targets(repo, "claude");
        for t in &claude_only {
            assert_eq!(t.name, "Claude Desktop");
        }

        // Unknown client → empty
        let unknown = discover_targets(repo, "vscode");
        assert!(unknown.is_empty());
    }

    #[test]
    fn test_discover_targets_all_includes_four_clients() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = tmp.path();

        let all = discover_targets(repo, "all");
        let names: Vec<&str> = all.iter().map(|t| t.name).collect();

        // Always includes Claude Code, Cursor, Codex (project-scoped)
        assert!(names.contains(&"Claude Code"));
        assert!(names.contains(&"Cursor"));
        assert!(names.contains(&"Codex"));
        // Claude Desktop may or may not be present (depends on macOS + ~/Library)
    }

    #[test]
    fn test_claude_code_configured_for_repo() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = tmp.path().join("my-project");
        std::fs::create_dir_all(&repo).unwrap();

        let config = serde_json::json!({
            "mcpServers": {
                "trevec": {
                    "command": "trevec",
                    "args": ["serve", "--path", repo.to_string_lossy()]
                }
            }
        });
        std::fs::write(
            repo.join(".mcp.json"),
            serde_json::to_string_pretty(&config).unwrap(),
        )
        .unwrap();

        assert!(is_client_configured(&repo, "claude-code"));
    }

    #[test]
    fn test_claude_code_not_configured_without_file() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = tmp.path().join("my-project");
        std::fs::create_dir_all(&repo).unwrap();

        assert!(!is_client_configured(&repo, "claude-code"));
    }

    #[test]
    fn test_codex_configured_for_repo() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = tmp.path().join("my-project");
        std::fs::create_dir_all(repo.join(".codex")).unwrap();

        let toml_content = r#"
[mcp_servers.trevec]
command = "trevec"
args = ["serve"]
"#;
        std::fs::write(repo.join(".codex/config.toml"), toml_content).unwrap();

        assert!(is_client_configured(&repo, "codex"));
    }

    #[test]
    fn test_codex_not_configured_without_entry() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = tmp.path().join("my-project");
        std::fs::create_dir_all(repo.join(".codex")).unwrap();

        let toml_content = "[mcp_servers]\n";
        std::fs::write(repo.join(".codex/config.toml"), toml_content).unwrap();

        assert!(!is_client_configured(&repo, "codex"));
    }

    #[test]
    fn test_codex_not_configured_without_file() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = tmp.path().join("my-project");
        std::fs::create_dir_all(&repo).unwrap();

        assert!(!is_client_configured(&repo, "codex"));
    }

    #[tokio::test]
    async fn test_run_rejects_invalid_client() {
        let result = run("vscode", false).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Unknown client"), "got: {err}");
    }

    #[tokio::test]
    async fn test_run_remove_rejects_invalid_client() {
        let result = run_remove("vscode", false).await;
        assert!(result.is_err());
        let err = result.unwrap_err().to_string();
        assert!(err.contains("Unknown client"), "got: {err}");
    }

    #[test]
    fn test_discover_global_targets_filters_by_client() {
        // "cursor" only → exactly 1 target
        let cursor_only = discover_global_targets("cursor");
        assert_eq!(cursor_only.len(), 1);
        assert_eq!(cursor_only[0].name, "Cursor");

        // "claude-code" only → exactly 1 target
        let cc_only = discover_global_targets("claude-code");
        assert_eq!(cc_only.len(), 1);
        assert_eq!(cc_only[0].name, "Claude Code");

        // "codex" only → exactly 1 target
        let codex_only = discover_global_targets("codex");
        assert_eq!(codex_only.len(), 1);
        assert_eq!(codex_only[0].name, "Codex");
        assert_eq!(codex_only[0].format, ConfigFormat::Toml);

        // Unknown client → empty
        let unknown = discover_global_targets("vscode");
        assert!(unknown.is_empty());
    }

    #[test]
    fn test_get_configured_binary_json() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = tmp.path().join("my-project");
        std::fs::create_dir_all(&repo).unwrap();

        let config = serde_json::json!({
            "mcpServers": {
                "trevec": {
                    "command": "/usr/local/bin/trevec",
                    "args": ["serve"]
                }
            }
        });
        std::fs::write(
            repo.join(".mcp.json"),
            serde_json::to_string_pretty(&config).unwrap(),
        )
        .unwrap();

        let binary = get_configured_binary(&repo, "claude-code");
        assert_eq!(binary, Some(PathBuf::from("/usr/local/bin/trevec")));
    }

    #[test]
    fn test_get_configured_binary_toml() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = tmp.path().join("my-project");
        std::fs::create_dir_all(repo.join(".codex")).unwrap();

        let toml_content = r#"
[mcp_servers.trevec]
command = "/opt/bin/trevec"
args = ["serve"]
"#;
        std::fs::write(repo.join(".codex/config.toml"), toml_content).unwrap();

        let binary = get_configured_binary(&repo, "codex");
        assert_eq!(binary, Some(PathBuf::from("/opt/bin/trevec")));
    }

    #[test]
    fn test_get_configured_binary_missing() {
        let tmp = tempfile::tempdir().unwrap();
        let repo = tmp.path().join("my-project");
        std::fs::create_dir_all(&repo).unwrap();

        assert!(get_configured_binary(&repo, "claude-code").is_none());
    }
}
