use anyhow::Result;
use std::path::{Path, PathBuf};

use trevec_core::TrevecConfig;

/// Run the doctor command: read-only health diagnostic for MCP wiring, rules,
/// memory, and index.
pub async fn run(path: PathBuf) -> Result<()> {
    let path = path.canonicalize().unwrap_or(path);
    let data_dir = path.join(".trevec");
    let mut issues = 0u32;

    // 1. Binary info
    let current_exe = std::env::current_exe().unwrap_or_else(|_| PathBuf::from("trevec"));
    let version = env!("CARGO_PKG_VERSION");
    eprintln!("Trevec Doctor");
    eprintln!("  Binary:  {} (v{})", current_exe.display(), version);
    eprintln!("  Repo:    {}", path.display());
    eprintln!();

    // 2. Init check
    if data_dir.exists() {
        eprintln!("[ok]   .trevec/ exists");
    } else {
        eprintln!("[FAIL] .trevec/ not found — run `trevec init`");
        issues += 1;
    }

    // 3. Index readiness
    let nodes_path = data_dir.join("nodes.json");
    if nodes_path.exists() {
        let stale = nodes_path
            .metadata()
            .ok()
            .and_then(|m| m.modified().ok())
            .and_then(|t| t.elapsed().ok())
            .is_some_and(|d| d.as_secs() > 86400);
        if stale {
            eprintln!("[!!]   nodes.json exists but is >24h old — consider re-indexing");
            issues += 1;
        } else {
            eprintln!("[ok]   nodes.json exists and is recent");
        }
    } else {
        eprintln!("[FAIL] nodes.json not found — run `trevec init` or `trevec index`");
        issues += 1;
    }

    // 4. MCP wiring — check global configs
    eprintln!();
    eprintln!("MCP Wiring (global):");
    let clients = [
        ("claude", "Claude Desktop"),
        ("claude-code", "Claude Code"),
        ("cursor", "Cursor"),
        ("codex", "Codex"),
    ];

    for (key, label) in &clients {
        let globally = crate::commands::setup::is_globally_configured(key);
        if globally {
            // Check binary path match in global config
            let binary_match = check_global_binary_match(key, &current_exe);
            match binary_match {
                BinaryCheck::Match => {
                    eprintln!("[ok]   {} — configured (global), binary matches", label);
                }
                BinaryCheck::Mismatch(configured_bin) => {
                    eprintln!(
                        "[!!]   {} — configured (global), but binary mismatch (configured: {})",
                        label,
                        configured_bin.display()
                    );
                    issues += 1;
                }
                BinaryCheck::Unknown => {
                    eprintln!("[ok]   {} — configured (global)", label);
                }
            }
        } else {
            // Check for legacy per-repo config
            let legacy = crate::commands::setup::is_client_configured(&path, key);
            if legacy {
                eprintln!(
                    "[!!]   {} — legacy per-repo config found. Run `trevec mcp setup` to switch to global config.",
                    label
                );
                issues += 1;
            } else {
                eprintln!("[!!]   {} — not configured. Run `trevec mcp setup`.", label);
                issues += 1;
            }
        }
    }

    // 5. Rules per client
    eprintln!();
    eprintln!("IDE Rules:");
    let rule_clients = [
        ("cursor", "Cursor (.cursor/rules/trevec.mdc)"),
        ("claude-code", "Claude Code (CLAUDE.md)"),
        ("codex", "Codex (AGENTS.md)"),
    ];

    for (key, label) in &rule_clients {
        let present = crate::commands::rules::is_rule_present(&path, key);
        if present {
            eprintln!("[ok]   {} — present", label);
        } else {
            eprintln!("[!!]   {} — missing", label);
            issues += 1;
        }
    }

    // 6. Memory flags
    eprintln!();
    eprintln!("Memory:");
    let config = TrevecConfig::load(&data_dir);
    if config.memory.enabled {
        eprintln!("[ok]   memory.enabled = true");
    } else {
        eprintln!("[!!]   memory.enabled = false");
        issues += 1;
    }

    let extractors = [
        (config.memory.cursor.enabled, "cursor"),
        (config.memory.claude_code.enabled, "claude_code"),
        (config.memory.codex.enabled, "codex"),
    ];
    for (enabled, name) in &extractors {
        if *enabled {
            eprintln!("[ok]   memory.{}.enabled = true", name);
        } else {
            eprintln!("[!!]   memory.{}.enabled = false", name);
            issues += 1;
        }
    }

    // 7. Summary
    eprintln!();
    if issues == 0 {
        eprintln!("All checks passed.");
    } else {
        eprintln!("{} issue(s) found.", issues);
    }

    Ok(())
}

enum BinaryCheck {
    Match,
    Mismatch(PathBuf),
    Unknown,
}

fn check_global_binary_match(client_key: &str, current_exe: &Path) -> BinaryCheck {
    let Some(configured_bin) =
        crate::commands::setup::get_global_configured_binary(client_key)
    else {
        return BinaryCheck::Unknown;
    };

    // "trevec" (bare command) counts as a match if the current exe is named "trevec"
    let configured_str = configured_bin.to_string_lossy();
    if configured_str == "trevec" {
        return BinaryCheck::Match;
    }

    let current_canonical = current_exe
        .canonicalize()
        .unwrap_or_else(|_| current_exe.to_path_buf());
    let configured_canonical = configured_bin
        .canonicalize()
        .unwrap_or_else(|_| configured_bin.clone());

    if current_canonical == configured_canonical {
        BinaryCheck::Match
    } else {
        BinaryCheck::Mismatch(configured_bin)
    }
}
