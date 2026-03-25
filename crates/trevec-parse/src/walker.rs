use anyhow::{Context, Result};
use ignore::overrides::OverrideBuilder;
use ignore::WalkBuilder;
use std::path::{Path, PathBuf};

/// Patterns of files/directories to always exclude.
const DENYLIST: &[&str] = &[
    "node_modules",
    "__pycache__",
    ".git",
    "dist",
    "build",
    "target",
    ".trevec",
    "vendor",
    ".venv",
    "venv",
    ".env",
];

/// Extensions to always exclude (binary/generated files).
const DENIED_EXTENSIONS: &[&str] = &[
    "lock", "min.js", "min.css", "map", "wasm", "png", "jpg", "jpeg", "gif", "svg", "ico",
    "ttf", "woff", "woff2", "eot", "pdf", "zip", "tar", "gz", "exe", "dll", "so", "dylib",
    "pyc", "pyo", "class", "o", "obj",
];

/// Discover files in a repository, respecting .gitignore, the hardcoded denylist,
/// and any user-provided exclude glob patterns from config.
pub fn discover_files(root: &Path, extra_excludes: &[String]) -> Result<Vec<PathBuf>> {
    let mut files = Vec::new();

    let mut builder = WalkBuilder::new(root);
    builder
        .hidden(true) // skip hidden files by default
        .git_ignore(true)
        .git_global(true)
        .git_exclude(true);

    if !extra_excludes.is_empty() {
        let mut overrides = OverrideBuilder::new(root);
        for pattern in extra_excludes {
            // Negate: `!pattern` tells the override to exclude matching paths
            overrides
                .add(&format!("!{pattern}"))
                .with_context(|| format!("Invalid exclude pattern: {pattern}"))?;
        }
        let overrides = overrides.build().context("Failed to build exclude overrides")?;
        builder.overrides(overrides);
    }

    let walker = builder.build();

    for entry in walker {
        let entry = entry?;
        let path = entry.path();

        // Skip directories
        if path.is_dir() {
            continue;
        }

        // Check denylist on all path components
        if path
            .components()
            .any(|c| DENYLIST.contains(&c.as_os_str().to_str().unwrap_or("")))
        {
            continue;
        }

        // Check denied extensions
        if let Some(ext) = path.extension().and_then(|e| e.to_str()) {
            if DENIED_EXTENSIONS.contains(&ext) {
                continue;
            }
            // Check compound extensions like .min.js
            if let Some(stem) = path.file_stem().and_then(|s| s.to_str()) {
                if stem.ends_with(".min") {
                    continue;
                }
            }
        }

        files.push(path.to_path_buf());
    }

    files.sort();
    Ok(files)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn test_discover_files_basic() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();

        // Create some files
        fs::write(root.join("main.rs"), "fn main() {}").unwrap();
        fs::write(root.join("lib.py"), "def foo(): pass").unwrap();
        fs::write(root.join("data.json"), "{}").unwrap();

        let files = discover_files(root, &[]).unwrap();
        assert_eq!(files.len(), 3);
    }

    #[test]
    fn test_discover_files_respects_denylist() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();

        fs::write(root.join("main.rs"), "fn main() {}").unwrap();
        let nm = root.join("node_modules");
        fs::create_dir(&nm).unwrap();
        fs::write(nm.join("pkg.js"), "module.exports = {}").unwrap();

        let files = discover_files(root, &[]).unwrap();
        assert_eq!(files.len(), 1);
        assert!(files[0].ends_with("main.rs"));
    }

    #[test]
    fn test_discover_files_skips_binary_extensions() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();

        fs::write(root.join("main.rs"), "fn main() {}").unwrap();
        fs::write(root.join("Cargo.lock"), "").unwrap();
        fs::write(root.join("image.png"), "").unwrap();

        let files = discover_files(root, &[]).unwrap();
        assert_eq!(files.len(), 1);
    }

    #[test]
    fn test_discover_files_extra_excludes() {
        let dir = tempfile::tempdir().unwrap();
        let root = dir.path();

        fs::write(root.join("main.rs"), "fn main() {}").unwrap();
        fs::write(root.join("generated.rs"), "// auto").unwrap();
        let sub = root.join("vendor_local");
        fs::create_dir(&sub).unwrap();
        fs::write(sub.join("lib.rs"), "// vendored").unwrap();

        let excludes = vec!["generated.rs".to_string(), "vendor_local/**".to_string()];
        let files = discover_files(root, &excludes).unwrap();
        assert_eq!(files.len(), 1);
        assert!(files[0].ends_with("main.rs"));
    }
}
