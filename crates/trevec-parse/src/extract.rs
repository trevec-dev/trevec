use anyhow::{Context, Result};
use std::collections::HashMap;
use streaming_iterator::StreamingIterator;
use tree_sitter::{Parser, QueryCursor};
use trevec_core::{
    compute_ast_hash, generate_node_id,
    model::{CodeNode, NodeKind, Span},
};

use crate::languages::LanguageConfig;

/// Extract only references (calls/imports) from a source file.
pub fn extract_references_from_source(
    file_path: &str,
    source: &[u8],
    lang_config: &LanguageConfig,
) -> Result<Vec<Reference>> {
    let mut parser = Parser::new();
    parser
        .set_language(&lang_config.language)
        .context("Failed to set language")?;

    let tree = parser
        .parse(source, None)
        .context("Failed to parse source")?;

    let Some(query) = lang_config.query.as_ref() else {
        return Ok(vec![]);
    };

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(query, tree.root_node(), source);
    let capture_names: Vec<&str> = query.capture_names().to_vec();

    let mut references = Vec::new();

    while let Some(m) = matches.next() {
        let mut name_text: Option<String> = None;
        let mut reference_capture: Option<(String, tree_sitter::Node)> = None;

        for capture in m.captures {
            let capture_name = capture_names[capture.index as usize];
            let node = capture.node;
            let text = node.utf8_text(source).unwrap_or("").to_string();

            if capture_name == "name" {
                name_text = Some(text);
            } else if capture_name.starts_with("reference.") {
                reference_capture = Some((capture_name.to_string(), node));
            }
        }

        if let Some((ref_capture_name, ref_node)) = reference_capture {
            let ref_name =
                name_text.unwrap_or_else(|| ref_node.utf8_text(source).unwrap_or("").to_string());

            if !ref_name.is_empty() {
                let ref_kind = if ref_capture_name.contains("call") {
                    ReferenceKind::Call
                } else {
                    ReferenceKind::Import
                };

                references.push(Reference {
                    name: ref_name,
                    kind: ref_kind,
                    file_path: file_path.to_string(),
                    start_byte: ref_node.start_byte(),
                    end_byte: ref_node.end_byte(),
                });
            }
        }
    }

    Ok(references)
}

/// A reference to a call or import found during extraction.
#[derive(Debug, Clone)]
pub struct Reference {
    pub name: String,
    pub kind: ReferenceKind,
    pub file_path: String,
    pub start_byte: usize,
    pub end_byte: usize,
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ReferenceKind {
    Call,
    Import,
}

/// Result of extracting nodes from a single file.
pub struct ExtractionResult {
    pub nodes: Vec<CodeNode>,
    pub references: Vec<Reference>,
}

/// Extract CodeNodes and references from a source file using tags.scm queries.
pub fn extract_from_source(
    file_path: &str,
    source: &[u8],
    lang_config: &LanguageConfig,
) -> Result<ExtractionResult> {
    let mut parser = Parser::new();
    parser
        .set_language(&lang_config.language)
        .context("Failed to set language")?;

    let tree = parser
        .parse(source, None)
        .context("Failed to parse source")?;

    let source_str = std::str::from_utf8(source).unwrap_or("");

    let Some(query) = lang_config.query.as_ref() else {
        return Ok(ExtractionResult {
            nodes: vec![],
            references: vec![],
        });
    };

    let mut cursor = QueryCursor::new();
    let mut matches = cursor.matches(query, tree.root_node(), source);

    let mut nodes = Vec::new();
    let mut references = Vec::new();

    let capture_names: Vec<&str> = query.capture_names().to_vec();

    // Use StreamingIterator - call next() which returns Option<&QueryMatch>
    while let Some(m) = matches.next() {
        let mut name_text: Option<String> = None;
        let mut definition_capture: Option<(usize, tree_sitter::Node)> = None;
        let mut reference_capture: Option<(String, tree_sitter::Node)> = None;

        for capture in m.captures {
            let capture_name = capture_names[capture.index as usize];
            let node = capture.node;
            let text = node.utf8_text(source).unwrap_or("").to_string();

            if capture_name == "name" {
                name_text = Some(text);
            } else if capture_name.starts_with("definition.") {
                definition_capture = Some((capture.index as usize, node));
            } else if capture_name.starts_with("reference.") {
                reference_capture = Some((capture_name.to_string(), node));
            }
        }

        // Handle definitions → CodeNode
        if let Some((_idx, def_node)) = definition_capture {
            let name = name_text.clone().unwrap_or_default();
            if name.is_empty() {
                continue;
            }

            // Determine the capture name to get the kind suffix
            let kind_suffix = m
                .captures
                .iter()
                .find_map(|c| {
                    let cn = capture_names[c.index as usize];
                    cn.strip_prefix("definition.")
                });

            let kind = kind_suffix
                .and_then(NodeKind::from_tag_suffix)
                .unwrap_or(NodeKind::Function);

            let span = Span {
                start_line: def_node.start_position().row,
                start_col: def_node.start_position().column,
                end_line: def_node.end_position().row,
                end_col: def_node.end_position().column,
                start_byte: def_node.start_byte(),
                end_byte: def_node.end_byte(),
            };

            let signature = extract_signature(source_str, &span);
            let doc_comment = extract_doc_comment(source_str, span.start_line);
            let identifiers = extract_identifiers(def_node, source);

            let bm25_text = CodeNode::build_bm25_text(
                file_path,
                &name,
                &signature,
                &identifiers,
                doc_comment.as_deref(),
            );

            let node_source = &source[span.start_byte..span.end_byte.min(source.len())];
            let ast_hash = compute_ast_hash(node_source);

            let id = generate_node_id(file_path, kind, &signature, span.start_byte);

            nodes.push(CodeNode {
                id,
                kind,
                file_path: file_path.to_string(),
                span,
                name: name.clone(),
                signature,
                doc_comment,
                identifiers,
                bm25_text,
                symbol_vec: None,
                ast_hash,
            });
        }

        // Handle references
        if let Some((ref_capture_name, ref_node)) = reference_capture {
            let ref_name = name_text.unwrap_or_else(|| {
                ref_node.utf8_text(source).unwrap_or("").to_string()
            });

            if !ref_name.is_empty() {
                let ref_kind = if ref_capture_name.contains("call") {
                    ReferenceKind::Call
                } else {
                    ReferenceKind::Import
                };

                references.push(Reference {
                    name: ref_name,
                    kind: ref_kind,
                    file_path: file_path.to_string(),
                    start_byte: ref_node.start_byte(),
                    end_byte: ref_node.end_byte(),
                });
            }
        }
    }

    dedup_nodes(&mut nodes);

    Ok(ExtractionResult { nodes, references })
}

/// Extract DocSection nodes from a markdown/MDX file using line-based heading detection.
/// No tree-sitter needed — headings are detected by leading `#` characters.
/// A markdown section being tracked during heading-based extraction.
struct OpenSection {
    level: usize,
    heading: String,
    start_line: usize,
    start_byte: usize,
    body_lines: Vec<String>,
}

pub fn extract_markdown_sections(file_path: &str, source: &[u8]) -> ExtractionResult {
    let source_str = match std::str::from_utf8(source) {
        Ok(s) => s,
        Err(_) => return ExtractionResult { nodes: vec![], references: vec![] },
    };

    let lines: Vec<&str> = source_str.lines().collect();
    let mut nodes = Vec::new();

    let mut stack: Vec<OpenSection> = Vec::new();
    let mut byte_offset: usize = 0;

    // Max lines per section to avoid huge nodes
    const MAX_SECTION_LINES: usize = 500;

    for (line_idx, line) in lines.iter().enumerate() {
        let line_start_byte = byte_offset;
        byte_offset += line.len() + 1; // +1 for newline

        // Detect heading: line starts with 1-6 `#` followed by space
        let trimmed = line.trim_start();
        if trimmed.starts_with('#') {
            let hashes = trimmed.bytes().take_while(|&b| b == b'#').count();
            if (1..=6).contains(&hashes) {
                let heading_text = trimmed[hashes..].trim().to_string();
                if heading_text.is_empty() {
                    continue;
                }

                // Close all sections at same or deeper level
                while let Some(open) = stack.last() {
                    if open.level >= hashes {
                        let section = stack.pop().unwrap();
                        finalize_section(
                            file_path, &section, line_idx, line_start_byte,
                            source, &mut nodes,
                        );
                    } else {
                        break;
                    }
                }

                // Open new section
                stack.push(OpenSection {
                    level: hashes,
                    heading: heading_text,
                    start_line: line_idx,
                    start_byte: line_start_byte,
                    body_lines: Vec::new(),
                });
            } else if let Some(section) = stack.last_mut() {
                if section.body_lines.len() < MAX_SECTION_LINES {
                    section.body_lines.push(line.to_string());
                }
            }
        } else if let Some(section) = stack.last_mut() {
            if section.body_lines.len() < MAX_SECTION_LINES {
                section.body_lines.push(line.to_string());
            }
        }
    }

    // Close remaining open sections (end at EOF)
    let total_lines = lines.len();
    let total_bytes = source.len();
    while let Some(section) = stack.pop() {
        finalize_section(
            file_path, &section, total_lines, total_bytes,
            source, &mut nodes,
        );
    }

    ExtractionResult { nodes, references: vec![] }
}

/// Finalize a markdown section into a CodeNode.
fn finalize_section(
    file_path: &str,
    section: &OpenSection,
    end_line: usize,
    end_byte: usize,
    source: &[u8],
    nodes: &mut Vec<CodeNode>,
) {
    // Extract first paragraph (first non-empty lines before first blank line)
    let mut first_para = Vec::new();
    let mut found_text = false;
    for line in &section.body_lines {
        let trimmed = line.trim();
        if trimmed.is_empty() {
            if found_text {
                break;
            }
        } else {
            found_text = true;
            first_para.push(trimmed.to_string());
        }
    }
    let doc_comment = if first_para.is_empty() {
        None
    } else {
        Some(first_para.join(" "))
    };

    // Extract backtick-quoted identifiers from body
    let body_text = section.body_lines.join("\n");
    let identifiers = extract_backtick_identifiers(&body_text);

    let bm25_text = CodeNode::build_bm25_text(
        file_path,
        &section.heading,
        &section.heading,
        &identifiers,
        doc_comment.as_deref(),
    );

    let section_source = &source[section.start_byte..end_byte.min(source.len())];
    let ast_hash = compute_ast_hash(section_source);
    let id = generate_node_id(file_path, NodeKind::DocSection, &section.heading, section.start_byte);

    nodes.push(CodeNode {
        id,
        kind: NodeKind::DocSection,
        file_path: file_path.to_string(),
        span: Span {
            start_line: section.start_line,
            start_col: 0,
            end_line: end_line.saturating_sub(1),
            end_col: 0,
            start_byte: section.start_byte,
            end_byte,
        },
        name: section.heading.clone(),
        signature: section.heading.clone(),
        doc_comment,
        identifiers,
        bm25_text,
        symbol_vec: None,
        ast_hash,
    });
}

/// Extract backtick-quoted identifiers from markdown text.
fn extract_backtick_identifiers(text: &str) -> Vec<String> {
    let mut identifiers = Vec::new();
    let mut seen = std::collections::HashSet::new();
    let mut chars = text.chars().peekable();

    while let Some(ch) = chars.next() {
        if ch == '`' {
            // Skip code blocks (```)
            if chars.peek() == Some(&'`') {
                // Skip until closing ``` or end
                chars.next(); // second `
                if chars.peek() == Some(&'`') {
                    chars.next(); // third `
                    // Skip until closing ```
                    let mut consecutive_backticks = 0;
                    for c in chars.by_ref() {
                        if c == '`' {
                            consecutive_backticks += 1;
                            if consecutive_backticks >= 3 {
                                break;
                            }
                        } else {
                            consecutive_backticks = 0;
                        }
                    }
                }
                continue;
            }

            // Single backtick — extract until closing backtick
            let mut ident = String::new();
            for c in chars.by_ref() {
                if c == '`' {
                    break;
                }
                if c == '\n' {
                    // Unclosed backtick — abort
                    ident.clear();
                    break;
                }
                ident.push(c);
            }

            let ident = ident.trim().to_string();
            // Only keep identifiers that look like code symbols (1-80 chars, no spaces in middle unless it's short)
            if !ident.is_empty()
                && ident.len() <= 80
                && !ident.contains(' ')
                && seen.insert(ident.clone())
            {
                identifiers.push(ident);
            }
        }
    }

    identifiers
}

/// Specificity ranking for NodeKind — higher is more specific.
fn kind_specificity(kind: NodeKind) -> u8 {
    match kind {
        NodeKind::Method => 4,
        NodeKind::Function => 3,
        NodeKind::Class => 2,
        NodeKind::Struct => 2,
        NodeKind::Trait => 2,
        NodeKind::Interface => 2,
        NodeKind::Enum => 2,
        NodeKind::Module => 1,
        NodeKind::Macro => 1,
        NodeKind::Type => 1,
        NodeKind::DocSection => 0,
    }
}

/// Remove duplicate nodes that share the same span in the same file.
/// When Tree-sitter matches both @definition.function and @definition.method
/// for the same AST node (e.g., impl-block methods in Rust), keep the most
/// specific kind.
fn dedup_nodes(nodes: &mut Vec<CodeNode>) {
    // Group by (file_path, start_byte, end_byte)
    let mut best: HashMap<(String, usize, usize), usize> = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        let key = (
            node.file_path.clone(),
            node.span.start_byte,
            node.span.end_byte,
        );
        match best.get(&key) {
            Some(&existing_idx) => {
                if kind_specificity(node.kind) > kind_specificity(nodes[existing_idx].kind) {
                    best.insert(key, i);
                }
            }
            None => {
                best.insert(key, i);
            }
        }
    }

    let keep: std::collections::HashSet<usize> = best.values().copied().collect();
    let mut i = 0;
    nodes.retain(|_| {
        let result = keep.contains(&i);
        i += 1;
        result
    });
}

/// Extract the signature: first line of the node up to `{` or newline.
fn extract_signature(source: &str, span: &Span) -> String {
    let lines: Vec<&str> = source.lines().collect();
    if span.start_line >= lines.len() {
        return String::new();
    }

    let first_line = lines[span.start_line].trim();

    // Truncate at first `{`
    let sig = if let Some(pos) = first_line.find('{') {
        first_line[..pos].trim()
    } else {
        first_line
    };

    // Also truncate at trailing `:` for Python-style defs
    let sig = if let Some(pos) = sig.rfind(':') {
        let after = sig[pos + 1..].trim();
        if after.is_empty() {
            sig[..pos].trim()
        } else {
            sig
        }
    } else {
        sig
    };

    sig.to_string()
}

/// Extract doc comment: look for comment lines immediately above the start_line.
fn extract_doc_comment(source: &str, start_line: usize) -> Option<String> {
    let lines: Vec<&str> = source.lines().collect();
    if start_line == 0 {
        return None;
    }

    let mut doc_lines = Vec::new();
    let mut line_idx = start_line.saturating_sub(1);

    loop {
        let line = lines.get(line_idx).map(|l| l.trim()).unwrap_or("");

        if line.starts_with("///")
            || line.starts_with("//!")
            || line.starts_with('#')
            || line.starts_with("/**")
            || line.starts_with("* ")
            || line.starts_with("*/")
            || line.starts_with("\"\"\"")
            || line.starts_with("'''")
        {
            let cleaned = line
                .trim_start_matches("///")
                .trim_start_matches("//!")
                .trim_start_matches("/**")
                .trim_start_matches("* ")
                .trim_start_matches("*/")
                .trim_start_matches('#')
                .trim();
            if !cleaned.is_empty() {
                doc_lines.push(cleaned.to_string());
            }
        } else if line.is_empty() {
            // Allow one blank line gap
            if line_idx > 0 && line_idx + 1 < start_line {
                break;
            }
        } else {
            break;
        }

        if line_idx == 0 {
            break;
        }
        line_idx -= 1;
    }

    if doc_lines.is_empty() {
        None
    } else {
        doc_lines.reverse();
        Some(doc_lines.join(" "))
    }
}

/// Extract identifiers from child nodes (parameter names, field names, etc.).
fn extract_identifiers(node: tree_sitter::Node, source: &[u8]) -> Vec<String> {
    let mut identifiers = Vec::new();
    let mut seen = std::collections::HashSet::new();

    collect_identifiers(node, source, &mut identifiers, &mut seen, 0);

    identifiers
}

fn collect_identifiers(
    node: tree_sitter::Node,
    source: &[u8],
    identifiers: &mut Vec<String>,
    seen: &mut std::collections::HashSet<String>,
    depth: usize,
) {
    // Don't recurse too deep
    if depth > 4 {
        return;
    }

    let kind = node.kind();

    if kind == "identifier"
        || kind == "type_identifier"
        || kind == "field_identifier"
        || kind == "property_identifier"
    {
        if let Ok(text) = node.utf8_text(source) {
            let text = text.to_string();
            if text.len() > 1 && !is_keyword(&text) && seen.insert(text.clone()) {
                identifiers.push(text);
            }
        }
    }

    for i in 0..node.child_count() {
        if let Some(child) = node.child(i) {
            let child_kind = child.kind();
            // Skip function bodies to avoid noise
            if child_kind == "block"
                || child_kind == "compound_statement"
                || child_kind == "statement_block"
                || child_kind == "function_body"
                || child_kind == "body"
            {
                continue;
            }
            collect_identifiers(child, source, identifiers, seen, depth + 1);
        }
    }
}

fn is_keyword(s: &str) -> bool {
    matches!(
        s,
        "fn" | "def"
            | "class"
            | "struct"
            | "enum"
            | "trait"
            | "impl"
            | "pub"
            | "private"
            | "protected"
            | "public"
            | "static"
            | "const"
            | "let"
            | "var"
            | "mut"
            | "self"
            | "Self"
            | "this"
            | "return"
            | "if"
            | "else"
            | "for"
            | "while"
            | "match"
            | "switch"
            | "case"
            | "break"
            | "continue"
            | "true"
            | "false"
            | "None"
            | "null"
            | "void"
            | "async"
            | "await"
            | "yield"
            | "import"
            | "from"
            | "export"
            | "default"
            | "new"
            | "delete"
            | "typeof"
            | "instanceof"
            | "in"
            | "of"
    )
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::languages::language_for_extension;

    #[test]
    fn test_extract_rust_functions() {
        let source = br#"
/// Adds two numbers together.
fn add(a: i32, b: i32) -> i32 {
    a + b
}

fn multiply(x: i32, y: i32) -> i32 {
    x * y
}
"#;
        let lang = language_for_extension("rs").unwrap();
        let result = extract_from_source("test.rs", source, &lang).unwrap();

        assert!(
            result.nodes.len() >= 2,
            "Expected at least 2 nodes, got {}",
            result.nodes.len()
        );

        let add_node = result.nodes.iter().find(|n| n.name == "add");
        assert!(add_node.is_some(), "Should find 'add' function");
        let add_node = add_node.unwrap();
        assert_eq!(add_node.kind, NodeKind::Function);
        assert!(add_node.signature.contains("fn add"));
    }

    #[test]
    fn test_extract_python_functions() {
        let source = br#"
def greet(name: str) -> str:
    """Say hello."""
    return f"Hello, {name}!"

class Calculator:
    def add(self, a, b):
        return a + b
"#;
        let lang = language_for_extension("py").unwrap();
        let result = extract_from_source("test.py", source, &lang).unwrap();

        assert!(
            !result.nodes.is_empty(),
            "Should extract some nodes from Python"
        );
    }

    #[test]
    fn test_extract_javascript_functions() {
        let source = br#"
function fetchData(url) {
    return fetch(url).then(r => r.json());
}

class ApiClient {
    constructor(baseUrl) {
        this.baseUrl = baseUrl;
    }

    async get(path) {
        return fetchData(this.baseUrl + path);
    }
}
"#;
        let lang = language_for_extension("js").unwrap();
        let result = extract_from_source("test.js", source, &lang).unwrap();

        assert!(
            !result.nodes.is_empty(),
            "Should extract some nodes from JavaScript"
        );
    }

    #[test]
    fn test_extract_deterministic_ids() {
        let source = b"fn main() { println!(\"hello\"); }";
        let lang = language_for_extension("rs").unwrap();

        let result1 = extract_from_source("test.rs", source, &lang).unwrap();
        let result2 = extract_from_source("test.rs", source, &lang).unwrap();

        assert_eq!(result1.nodes.len(), result2.nodes.len());
        for (n1, n2) in result1.nodes.iter().zip(result2.nodes.iter()) {
            assert_eq!(n1.id, n2.id, "Node IDs should be deterministic");
        }
    }

    #[test]
    fn test_extract_signature() {
        let source = "fn add(a: i32, b: i32) -> i32 {\n    a + b\n}";
        let span = Span {
            start_line: 0,
            start_col: 0,
            end_line: 2,
            end_col: 1,
            start_byte: 0,
            end_byte: source.len(),
        };
        let sig = extract_signature(source, &span);
        assert_eq!(sig, "fn add(a: i32, b: i32) -> i32");
    }

    #[test]
    fn test_extract_go_functions() {
        let source = br#"
package main

import "fmt"

// Hello greets the user by name.
func Hello(name string) string {
    return fmt.Sprintf("Hello, %s!", name)
}

func Add(a int, b int) int {
    return a + b
}
"#;
        let lang = language_for_extension("go").unwrap();
        let result = extract_from_source("test.go", source, &lang).unwrap();

        assert!(
            !result.nodes.is_empty(),
            "Should extract nodes from Go source"
        );

        let hello = result.nodes.iter().find(|n| n.name == "Hello");
        assert!(hello.is_some(), "Should find 'Hello' function in Go");
        let hello = hello.unwrap();
        assert_eq!(hello.kind, NodeKind::Function);
    }

    #[test]
    fn test_extract_typescript_functions() {
        let source = br#"
interface User {
    id: string;
    name: string;
}

function createUser(name: string): User {
    return { id: "1", name };
}

class UserService {
    private users: User[] = [];

    addUser(user: User): void {
        this.users.push(user);
    }

    getUser(id: string): User | undefined {
        return this.users.find(u => u.id === id);
    }
}
"#;
        let lang = language_for_extension("ts").unwrap();
        let result = extract_from_source("test.ts", source, &lang).unwrap();

        assert!(
            !result.nodes.is_empty(),
            "Should extract nodes from TypeScript source"
        );
    }

    #[test]
    fn test_extract_references_from_source_rust() {
        let source = br#"
fn helper() -> i32 {
    42
}

fn main() {
    let x = helper();
    println!("value: {}", x);
}
"#;
        let lang = language_for_extension("rs").unwrap();
        let refs = extract_references_from_source("test.rs", source, &lang).unwrap();

        // Should find at least the call to `helper` and macro invocation `println`
        let call_names: Vec<&str> = refs
            .iter()
            .filter(|r| r.kind == ReferenceKind::Call)
            .map(|r| r.name.as_str())
            .collect();

        assert!(
            call_names.contains(&"helper"),
            "Should find call to 'helper', got: {:?}",
            call_names
        );
        assert!(
            call_names.contains(&"println"),
            "Should find macro invocation 'println', got: {:?}",
            call_names
        );

        // All references should have the correct file_path
        for r in &refs {
            assert_eq!(r.file_path, "test.rs");
        }
    }

    #[test]
    fn test_dedup_method_vs_function() {
        // Rust impl block methods should be deduped: keep Method over Function.
        let source = br#"
struct Auth;

impl Auth {
    /// Verify a password hash.
    fn verify_password(hash: &str, password: &str) -> bool {
        hash == password
    }
}
"#;
        let lang = language_for_extension("rs").unwrap();
        let result = extract_from_source("test.rs", source, &lang).unwrap();

        let verify_nodes: Vec<_> = result
            .nodes
            .iter()
            .filter(|n| n.name == "verify_password")
            .collect();

        assert_eq!(
            verify_nodes.len(),
            1,
            "verify_password should appear exactly once after dedup, got {}",
            verify_nodes.len()
        );
        assert_eq!(
            verify_nodes[0].kind,
            NodeKind::Method,
            "Should keep Method (more specific) over Function"
        );
    }

    #[test]
    fn test_extract_markdown_basic() {
        let source = b"# Getting Started\n\nThis is the intro.\n\n## Installation\n\nRun `npm install` to get started.\n\n## Usage\n\nUse `authenticate()` to log in.\n";
        let result = extract_markdown_sections("README.md", source);

        assert_eq!(result.nodes.len(), 3, "Should extract 3 sections");

        let names: Vec<&str> = result.nodes.iter().map(|n| n.name.as_str()).collect();
        assert!(names.contains(&"Getting Started"));
        assert!(names.contains(&"Installation"));
        assert!(names.contains(&"Usage"));

        // All should be DocSection kind
        for node in &result.nodes {
            assert_eq!(node.kind, NodeKind::DocSection);
        }

        // Check doc_comment (first paragraph)
        let intro = result.nodes.iter().find(|n| n.name == "Getting Started").unwrap();
        assert_eq!(intro.doc_comment.as_deref(), Some("This is the intro."));
    }

    #[test]
    fn test_extract_markdown_identifiers() {
        let source = b"# Auth\n\nUse `authenticate()` and `AuthService` to handle login.\nAlso see `verify_token`.\n";
        let result = extract_markdown_sections("auth.md", source);

        assert_eq!(result.nodes.len(), 1);
        let node = &result.nodes[0];
        assert!(node.identifiers.contains(&"authenticate()".to_string()));
        assert!(node.identifiers.contains(&"AuthService".to_string()));
        assert!(node.identifiers.contains(&"verify_token".to_string()));
    }

    #[test]
    fn test_extract_markdown_nested_headings() {
        let source = b"# Top\n\nTop content.\n\n## Child A\n\nChild A content.\n\n### Grandchild\n\nDeep content.\n\n## Child B\n\nChild B content.\n";
        let result = extract_markdown_sections("nested.md", source);

        assert_eq!(result.nodes.len(), 4, "Should extract 4 sections: Top, Child A, Grandchild, Child B");

        // Verify spans don't overlap incorrectly
        let top = result.nodes.iter().find(|n| n.name == "Top").unwrap();
        let child_a = result.nodes.iter().find(|n| n.name == "Child A").unwrap();
        let grandchild = result.nodes.iter().find(|n| n.name == "Grandchild").unwrap();
        let child_b = result.nodes.iter().find(|n| n.name == "Child B").unwrap();

        // Top should contain all others (it spans the whole doc)
        assert!(top.span.contains(&child_a.span), "Top should contain Child A");
        assert!(top.span.contains(&grandchild.span), "Top should contain Grandchild");
        assert!(top.span.contains(&child_b.span), "Top should contain Child B");

        // Child A should contain Grandchild
        assert!(child_a.span.contains(&grandchild.span), "Child A should contain Grandchild");
    }

    #[test]
    fn test_extract_markdown_code_blocks_ignored() {
        let source = b"# Example\n\nSome text with `real_ident` here.\n\n```rust\nfn not_an_ident() {}\nlet x = `backtick`;\n```\n\nMore text.\n";
        let result = extract_markdown_sections("example.md", source);

        assert_eq!(result.nodes.len(), 1);
        let node = &result.nodes[0];
        // Should extract real_ident but not things inside code blocks
        assert!(node.identifiers.contains(&"real_ident".to_string()));
        assert!(!node.identifiers.contains(&"not_an_ident()".to_string()));
    }

    #[test]
    fn test_extract_markdown_deterministic_ids() {
        let source = b"# Hello\n\nWorld.\n";
        let result1 = extract_markdown_sections("doc.md", source);
        let result2 = extract_markdown_sections("doc.md", source);

        assert_eq!(result1.nodes.len(), result2.nodes.len());
        for (n1, n2) in result1.nodes.iter().zip(result2.nodes.iter()) {
            assert_eq!(n1.id, n2.id, "IDs should be deterministic");
            assert_eq!(n1.ast_hash, n2.ast_hash, "Hashes should be deterministic");
        }
    }

    #[test]
    fn test_extract_backtick_identifiers() {
        let text = "Use `foo()` and `BarService` but not `hello world` or empty `` ones.";
        let idents = extract_backtick_identifiers(text);
        assert!(idents.contains(&"foo()".to_string()));
        assert!(idents.contains(&"BarService".to_string()));
        // "hello world" has a space, should be excluded
        assert!(!idents.iter().any(|i| i.contains("hello")));
    }
}
