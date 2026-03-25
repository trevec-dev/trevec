use tree_sitter::{Language, Query};

/// A supported language with its grammar and tags query.
pub struct LanguageConfig {
    pub name: &'static str,
    pub language: Language,
    pub query: Option<Query>,
}

fn build_config(
    name: &'static str,
    language: Language,
    tags_query: &'static str,
) -> LanguageConfig {
    let query = if tags_query.is_empty() {
        None
    } else {
        match Query::new(&language, tags_query) {
            Ok(q) => Some(q),
            Err(e) => {
                tracing::warn!("Failed to compile tags query for {name}: {e}");
                None
            }
        }
    };

    LanguageConfig {
        name,
        language,
        query,
    }
}

/// Get a language configuration from a file extension.
/// Returns None for unsupported extensions.
pub fn language_for_extension(ext: &str) -> Option<LanguageConfig> {
    match ext {
        "rs" => Some(build_config(
            "rust",
            tree_sitter_rust::LANGUAGE.into(),
            tree_sitter_rust::TAGS_QUERY,
        )),
        "py" | "pyi" | "pyw" => Some(build_config(
            "python",
            tree_sitter_python::LANGUAGE.into(),
            tree_sitter_python::TAGS_QUERY,
        )),
        "js" | "mjs" | "cjs" => Some(build_config(
            "javascript",
            tree_sitter_javascript::LANGUAGE.into(),
            tree_sitter_javascript::TAGS_QUERY,
        )),
        "ts" => Some(build_config(
            "typescript",
            tree_sitter_typescript::LANGUAGE_TYPESCRIPT.into(),
            tree_sitter_typescript::TAGS_QUERY,
        )),
        "tsx" => Some(build_config(
            "tsx",
            tree_sitter_typescript::LANGUAGE_TSX.into(),
            tree_sitter_typescript::TAGS_QUERY,
        )),
        "go" => Some(build_config(
            "go",
            tree_sitter_go::LANGUAGE.into(),
            tree_sitter_go::TAGS_QUERY,
        )),
        "java" => Some(build_config(
            "java",
            tree_sitter_java::LANGUAGE.into(),
            tree_sitter_java::TAGS_QUERY,
        )),
        "c" | "h" => Some(build_config(
            "c",
            tree_sitter_c::LANGUAGE.into(),
            tree_sitter_c::TAGS_QUERY,
        )),
        "cpp" | "cc" | "cxx" | "hpp" | "hxx" | "hh" => Some(build_config(
            "cpp",
            tree_sitter_cpp::LANGUAGE.into(),
            tree_sitter_cpp::TAGS_QUERY,
        )),
        "rb" => Some(build_config(
            "ruby",
            tree_sitter_ruby::LANGUAGE.into(),
            tree_sitter_ruby::TAGS_QUERY,
        )),
        "sh" | "bash" => Some(build_config(
            "bash",
            tree_sitter_bash::LANGUAGE.into(),
            "",
        )),
        "json" => Some(build_config(
            "json",
            tree_sitter_json::LANGUAGE.into(),
            "",
        )),
        "html" | "htm" => Some(build_config(
            "html",
            tree_sitter_html::LANGUAGE.into(),
            "",
        )),
        "css" => Some(build_config(
            "css",
            tree_sitter_css::LANGUAGE.into(),
            "",
        )),
        "cs" => Some(build_config(
            "c_sharp",
            tree_sitter_c_sharp::LANGUAGE.into(),
            "",
        )),
        "lua" => Some(build_config(
            "lua",
            tree_sitter_lua::LANGUAGE.into(),
            tree_sitter_lua::TAGS_QUERY,
        )),
        "zig" => Some(build_config(
            "zig",
            tree_sitter_zig::LANGUAGE.into(),
            "",
        )),
        "swift" => Some(build_config(
            "swift",
            tree_sitter_swift::LANGUAGE.into(),
            tree_sitter_swift::TAGS_QUERY,
        )),
        _ => None,
    }
}

/// Get the list of all supported file extensions.
pub fn supported_extensions() -> &'static [&'static str] {
    &[
        "rs", "py", "pyi", "pyw", "js", "mjs", "cjs", "ts", "tsx", "go", "java", "c", "h",
        "cpp", "cc", "cxx", "hpp", "hxx", "hh", "rb", "sh", "bash", "json", "html", "htm",
        "css", "cs", "lua", "zig", "swift", "md", "mdx",
    ]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_known_extensions() {
        assert!(language_for_extension("rs").is_some());
        assert!(language_for_extension("py").is_some());
        assert!(language_for_extension("ts").is_some());
        assert!(language_for_extension("go").is_some());
        assert!(language_for_extension("java").is_some());
    }

    #[test]
    fn test_unknown_extensions() {
        assert!(language_for_extension("xyz").is_none());
        assert!(language_for_extension("").is_none());
    }

    #[test]
    fn test_rust_has_tags_query() {
        let config = language_for_extension("rs").unwrap();
        assert!(config.query.is_some());
    }
}
