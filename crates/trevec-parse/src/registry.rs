//! Domain Parser Registry: pluggable system where domain-specific parsers register
//! their node types, edge types, and extraction logic.
//!
//! Tree-sitter AST parsing becomes one parser plugin. New domains (conversations,
//! documents, structured data) add new plugins.

use anyhow::Result;
use std::collections::HashMap;
use trevec_core::universal::{UniversalEdge, UniversalNode};
use trevec_core::TrevecConfig;

/// Result of parsing a single file/input through a domain parser.
#[derive(Debug, Clone)]
pub struct ParseResult {
    pub nodes: Vec<UniversalNode>,
    pub edges: Vec<UniversalEdge>,
}

/// Trait that all domain parsers implement.
pub trait DomainParser: Send + Sync {
    /// Unique identifier for this parser (e.g., "code", "conversation", "document").
    fn domain_id(&self) -> &'static str;

    /// File extensions this parser handles (e.g., [".rs", ".py"]).
    fn supported_extensions(&self) -> &[&'static str];

    /// Parse input bytes into universal nodes and edges.
    fn parse(
        &self,
        file_path: &str,
        source: &[u8],
        config: &TrevecConfig,
    ) -> Result<ParseResult>;

    /// Whether this parser can handle the given file (fallback for unknown extensions).
    fn can_parse(&self, file_path: &str, _first_bytes: &[u8]) -> bool {
        let path_lower = file_path.to_lowercase();
        self.supported_extensions()
            .iter()
            .any(|ext| path_lower.ends_with(ext))
    }
}

/// Registry of domain parsers, routing files to the appropriate parser.
pub struct ParserRegistry {
    parsers: Vec<Box<dyn DomainParser>>,
    extension_map: HashMap<String, usize>,
}

impl ParserRegistry {
    pub fn new() -> Self {
        Self {
            parsers: Vec::new(),
            extension_map: HashMap::new(),
        }
    }

    /// Register a domain parser. Its supported extensions are indexed for fast lookup.
    pub fn register(&mut self, parser: Box<dyn DomainParser>) {
        let idx = self.parsers.len();
        for ext in parser.supported_extensions() {
            self.extension_map
                .insert(ext.to_lowercase(), idx);
        }
        self.parsers.push(parser);
    }

    /// Find the parser for a given file path based on extension.
    pub fn parser_for_file(&self, file_path: &str) -> Option<&dyn DomainParser> {
        let path_lower = file_path.to_lowercase();

        // Try extension map first (fast path)
        for (ext, idx) in &self.extension_map {
            if path_lower.ends_with(ext) {
                return Some(self.parsers[*idx].as_ref());
            }
        }

        // Fallback: ask each parser
        for parser in &self.parsers {
            if parser.can_parse(file_path, &[]) {
                return Some(parser.as_ref());
            }
        }

        None
    }

    /// Parse a file using the appropriate domain parser.
    pub fn parse_file(
        &self,
        file_path: &str,
        source: &[u8],
        config: &TrevecConfig,
    ) -> Result<Option<ParseResult>> {
        match self.parser_for_file(file_path) {
            Some(parser) => Ok(Some(parser.parse(file_path, source, config)?)),
            None => Ok(None),
        }
    }

    /// List all registered parser domain IDs.
    pub fn registered_domains(&self) -> Vec<&'static str> {
        self.parsers.iter().map(|p| p.domain_id()).collect()
    }

    /// Number of registered parsers.
    pub fn len(&self) -> usize {
        self.parsers.len()
    }

    pub fn is_empty(&self) -> bool {
        self.parsers.is_empty()
    }
}

impl Default for ParserRegistry {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use trevec_core::universal::*;

    /// A test parser that handles .test files.
    struct TestParser;

    impl DomainParser for TestParser {
        fn domain_id(&self) -> &'static str {
            "test"
        }

        fn supported_extensions(&self) -> &[&'static str] {
            &[".test", ".tst"]
        }

        fn parse(
            &self,
            file_path: &str,
            _source: &[u8],
            _config: &TrevecConfig,
        ) -> Result<ParseResult> {
            Ok(ParseResult {
                nodes: vec![UniversalNode {
                    id: "test_node".to_string(),
                    kind: UniversalKind::Entity,
                    domain: DomainTag::Structured,
                    label: "test".to_string(),
                    file_path: file_path.to_string(),
                    span: None,
                    signature: None,
                    doc_comment: None,
                    identifiers: vec![],
                    bm25_text: "test".to_string(),
                    symbol_vec: None,
                    ast_hash: None,
                    temporal: None,
                    attributes: Default::default(),
                    intent_summary: None,
                }],
                edges: vec![],
            })
        }
    }

    #[test]
    fn test_registry_register_and_find() {
        let mut registry = ParserRegistry::new();
        registry.register(Box::new(TestParser));

        assert_eq!(registry.len(), 1);
        assert!(!registry.is_empty());
        assert_eq!(registry.registered_domains(), vec!["test"]);

        assert!(registry.parser_for_file("data.test").is_some());
        assert!(registry.parser_for_file("data.tst").is_some());
        assert!(registry.parser_for_file("data.rs").is_none());
    }

    #[test]
    fn test_registry_parse_file() {
        let mut registry = ParserRegistry::new();
        registry.register(Box::new(TestParser));
        let config = TrevecConfig::default();

        let result = registry
            .parse_file("data.test", b"content", &config)
            .unwrap();
        assert!(result.is_some());
        let pr = result.unwrap();
        assert_eq!(pr.nodes.len(), 1);
        assert_eq!(pr.nodes[0].id, "test_node");

        // Unknown extension
        let result = registry
            .parse_file("data.xyz", b"content", &config)
            .unwrap();
        assert!(result.is_none());
    }

    #[test]
    fn test_registry_case_insensitive() {
        let mut registry = ParserRegistry::new();
        registry.register(Box::new(TestParser));

        assert!(registry.parser_for_file("DATA.TEST").is_some());
        assert!(registry.parser_for_file("Data.Tst").is_some());
    }

    #[test]
    fn test_registry_empty() {
        let registry = ParserRegistry::new();
        assert!(registry.is_empty());
        assert_eq!(registry.len(), 0);
        assert!(registry.parser_for_file("anything.rs").is_none());
    }
}
