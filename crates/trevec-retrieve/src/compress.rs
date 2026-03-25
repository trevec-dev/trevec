//! Context Compression Engine: reduces token usage by 60-80% while preserving
//! retrieval quality through structural awareness.
//!
//! Three-layer compression:
//! 1. Signature-first: send signatures for non-anchor nodes
//! 2. Overlap deduplication: remove redundant nested spans
//! 3. Import chain compression: consolidate repeated imports

use std::collections::{HashMap, HashSet};
use trevec_core::model::{CodeNode, ContextBundle, IncludedNode, NodeKind, Span};

/// Controls how much detail is included in the context bundle.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ContextDepth {
    /// Level 0: File tree + module summary names only
    Summary,
    /// Level 1: Signatures + doc comments (no implementation)
    Signatures,
    /// Level 2: Full code for anchor nodes, signatures for rest (DEFAULT)
    Anchors,
    /// Level 3: Full implementation for all included nodes
    Full,
}

impl Default for ContextDepth {
    fn default() -> Self {
        Self::Anchors
    }
}

impl ContextDepth {
    pub fn from_str_loose(s: &str) -> Self {
        match s.to_lowercase().as_str() {
            "summary" | "0" => Self::Summary,
            "signatures" | "sigs" | "1" => Self::Signatures,
            "anchors" | "2" => Self::Anchors,
            "full" | "3" => Self::Full,
            _ => Self::Anchors,
        }
    }
}

/// Result of context compression.
#[derive(Debug, Clone)]
pub struct CompressedBundle {
    /// The compressed context bundle.
    pub bundle: ContextBundle,
    /// Depth level used for compression.
    pub depth: ContextDepth,
    /// Original token count before compression.
    pub original_tokens: usize,
    /// Token count after compression.
    pub compressed_tokens: usize,
    /// Percentage saved (0.0-100.0).
    pub savings_percent: f32,
    /// Node IDs that can be expanded to deeper level on demand.
    pub expandable_nodes: Vec<String>,
}

/// Compress a context bundle using the specified depth level.
pub fn compress_bundle(bundle: &ContextBundle, depth: ContextDepth) -> CompressedBundle {
    let original_tokens = bundle.total_estimated_tokens;

    let mut compressed_nodes: Vec<IncludedNode> = match depth {
        ContextDepth::Summary => compress_to_summary(&bundle.included_nodes),
        ContextDepth::Signatures => compress_to_signatures(&bundle.included_nodes),
        ContextDepth::Anchors => {
            compress_anchors_full(&bundle.included_nodes, &bundle.anchor_node_ids)
        }
        ContextDepth::Full => bundle.included_nodes.clone(), // No compression at full depth
    };

    // Always apply deduplication and import compression
    compressed_nodes = deduplicate_overlapping(compressed_nodes);
    compressed_nodes = compress_imports(compressed_nodes);

    let compressed_tokens: usize = compressed_nodes.iter().map(|n| n.estimated_tokens).sum();
    let expandable_nodes: Vec<String> = compressed_nodes
        .iter()
        .filter(|n| !n.is_anchor && n.source_text.contains("// ..."))
        .map(|n| n.node_id.clone())
        .collect();

    let savings_percent = if original_tokens > 0 {
        ((original_tokens - compressed_tokens) as f32 / original_tokens as f32) * 100.0
    } else {
        0.0
    };

    CompressedBundle {
        bundle: ContextBundle {
            bundle_id: bundle.bundle_id.clone(),
            query: bundle.query.clone(),
            anchor_node_ids: bundle.anchor_node_ids.clone(),
            included_nodes: compressed_nodes,
            total_estimated_tokens: compressed_tokens,
            total_source_file_tokens: bundle.total_source_file_tokens,
            retrieval_ms: bundle.retrieval_ms,
        },
        depth,
        original_tokens,
        compressed_tokens,
        savings_percent,
        expandable_nodes,
    }
}

/// Level 0: Compress to summary — only file paths and node names.
fn compress_to_summary(nodes: &[IncludedNode]) -> Vec<IncludedNode> {
    // Group by file, show one summary node per file
    let mut by_file: HashMap<&str, Vec<&IncludedNode>> = HashMap::new();
    for node in nodes {
        by_file.entry(&node.file_path).or_default().push(node);
    }

    let mut result = Vec::new();
    for (file_path, file_nodes) in by_file {
        let names: Vec<String> = file_nodes
            .iter()
            .map(|n| format!("{} {}", n.kind, n.name))
            .collect();
        let summary_text = format!(
            "// {}: {} nodes — {}\n",
            file_path,
            file_nodes.len(),
            names.join(", ")
        );
        let tokens = summary_text.len() / 4;

        result.push(IncludedNode {
            node_id: format!("summary:{}", file_path),
            file_path: file_path.to_string(),
            span: Span {
                start_line: 0,
                start_col: 0,
                end_line: 0,
                end_col: 0,
                start_byte: 0,
                end_byte: 0,
            },
            kind: NodeKind::Module,
            name: file_path
                .rsplit('/')
                .next()
                .unwrap_or(file_path)
                .to_string(),
            signature: names.join(", "),
            source_text: summary_text,
            is_anchor: false,
            estimated_tokens: tokens,
        });
    }
    result
}

/// Level 1: Show signatures + doc comments only, no implementation.
fn compress_to_signatures(nodes: &[IncludedNode]) -> Vec<IncludedNode> {
    nodes
        .iter()
        .map(|node| {
            let sig_text = build_signature_text(node);
            let tokens = sig_text.len() / 4;
            IncludedNode {
                node_id: node.node_id.clone(),
                file_path: node.file_path.clone(),
                span: node.span.clone(),
                kind: node.kind,
                name: node.name.clone(),
                signature: node.signature.clone(),
                source_text: sig_text,
                is_anchor: node.is_anchor,
                estimated_tokens: tokens,
            }
        })
        .collect()
}

/// Level 2: Full code for anchors, signatures for non-anchors.
fn compress_anchors_full(nodes: &[IncludedNode], anchor_ids: &[String]) -> Vec<IncludedNode> {
    let anchor_set: HashSet<&str> = anchor_ids.iter().map(|s| s.as_str()).collect();

    nodes
        .iter()
        .map(|node| {
            if anchor_set.contains(node.node_id.as_str()) || node.is_anchor {
                // Anchors keep full source
                node.clone()
            } else {
                // Non-anchors get signature only
                let sig_text = build_signature_text(node);
                let tokens = sig_text.len() / 4;
                IncludedNode {
                    node_id: node.node_id.clone(),
                    file_path: node.file_path.clone(),
                    span: node.span.clone(),
                    kind: node.kind,
                    name: node.name.clone(),
                    signature: node.signature.clone(),
                    source_text: sig_text,
                    is_anchor: false,
                    estimated_tokens: tokens,
                }
            }
        })
        .collect()
}

/// Build a signature-only representation of a node.
fn build_signature_text(node: &IncludedNode) -> String {
    let mut text = String::new();

    // Add signature
    if !node.signature.is_empty() {
        text.push_str(&node.signature);
        // Add hint that implementation is available
        text.push_str(" // ... (expandable)");
    } else {
        text.push_str(&format!("{} {}", node.kind, node.name));
        text.push_str(" // ... (expandable)");
    }

    text.push('\n');
    text
}

/// Remove nodes whose spans are fully contained within another included node.
fn deduplicate_overlapping(nodes: Vec<IncludedNode>) -> Vec<IncludedNode> {
    if nodes.len() <= 1 {
        return nodes;
    }

    let mut result = Vec::with_capacity(nodes.len());
    let mut removed: HashSet<usize> = HashSet::new();

    for i in 0..nodes.len() {
        if removed.contains(&i) {
            continue;
        }
        for j in 0..nodes.len() {
            if i == j || removed.contains(&j) {
                continue;
            }
            // If node j's span is fully within node i's span (same file)
            if nodes[i].file_path == nodes[j].file_path
                && nodes[i].span.contains(&nodes[j].span)
                && !nodes[j].is_anchor
            {
                removed.insert(j);
            }
        }
    }

    for (i, node) in nodes.into_iter().enumerate() {
        if !removed.contains(&i) {
            result.push(node);
        }
    }

    result
}

/// Compress import-heavy context by consolidating redundant imports.
fn compress_imports(nodes: Vec<IncludedNode>) -> Vec<IncludedNode> {
    // If multiple nodes from the same file are all imports/short, consolidate
    let mut by_file: HashMap<String, Vec<(usize, &IncludedNode)>> = HashMap::new();
    for (i, node) in nodes.iter().enumerate() {
        by_file
            .entry(node.file_path.clone())
            .or_default()
            .push((i, node));
    }

    let mut consolidated_indices: HashSet<usize> = HashSet::new();
    let mut extra_nodes: Vec<IncludedNode> = Vec::new();

    for (file_path, file_nodes) in &by_file {
        // Find import-like nodes (very short, < 3 lines)
        let imports: Vec<(usize, &IncludedNode)> = file_nodes
            .iter()
            .filter(|(_, n)| {
                n.source_text.lines().count() <= 3
                    && (n.source_text.contains("import ")
                        || n.source_text.contains("use ")
                        || n.source_text.contains("require(")
                        || n.source_text.contains("from "))
            })
            .copied()
            .collect();

        // If 3+ imports from same file, consolidate
        if imports.len() >= 3 {
            for (idx, _) in &imports {
                consolidated_indices.insert(*idx);
            }
            let import_text: Vec<&str> = imports
                .iter()
                .map(|(_, n)| n.source_text.as_str())
                .collect();
            let combined = format!(
                "// {} imports from {}\n{}",
                imports.len(),
                file_path,
                import_text.join("")
            );
            let tokens = combined.len() / 4;
            extra_nodes.push(IncludedNode {
                node_id: format!("imports:{}", file_path),
                file_path: file_path.clone(),
                span: Span {
                    start_line: 0,
                    start_col: 0,
                    end_line: 0,
                    end_col: 0,
                    start_byte: 0,
                    end_byte: 0,
                },
                kind: NodeKind::Module,
                name: "imports".to_string(),
                signature: format!("{} imports", imports.len()),
                source_text: combined,
                is_anchor: false,
                estimated_tokens: tokens,
            });
        }
    }

    let mut result: Vec<IncludedNode> = nodes
        .into_iter()
        .enumerate()
        .filter(|(i, _)| !consolidated_indices.contains(i))
        .map(|(_, n)| n)
        .collect();

    result.extend(extra_nodes);
    result
}

/// Estimate token savings for a given context bundle at a depth level.
pub fn estimate_savings(bundle: &ContextBundle, depth: ContextDepth) -> (usize, f32) {
    let compressed = compress_bundle(bundle, depth);
    (compressed.compressed_tokens, compressed.savings_percent)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_bundle() -> ContextBundle {
        ContextBundle {
            bundle_id: "test".to_string(),
            query: "how does auth work".to_string(),
            anchor_node_ids: vec!["anchor_1".to_string()],
            included_nodes: vec![
                IncludedNode {
                    node_id: "anchor_1".to_string(),
                    file_path: "src/auth.rs".to_string(),
                    span: Span {
                        start_line: 0,
                        start_col: 0,
                        end_line: 20,
                        end_col: 0,
                        start_byte: 0,
                        end_byte: 400,
                    },
                    kind: NodeKind::Function,
                    name: "authenticate".to_string(),
                    signature: "fn authenticate(req: &Request) -> Result<User>".to_string(),
                    source_text: "fn authenticate(req: &Request) -> Result<User> {\n    let token = req.header(\"Authorization\")?;\n    let claims = verify_jwt(token)?;\n    let user = db.find_user(claims.sub)?;\n    Ok(user)\n}\n".to_string(),
                    is_anchor: true,
                    estimated_tokens: 50,
                },
                IncludedNode {
                    node_id: "helper_1".to_string(),
                    file_path: "src/auth.rs".to_string(),
                    span: Span {
                        start_line: 25,
                        start_col: 0,
                        end_line: 40,
                        end_col: 0,
                        start_byte: 500,
                        end_byte: 800,
                    },
                    kind: NodeKind::Function,
                    name: "verify_jwt".to_string(),
                    signature: "fn verify_jwt(token: &str) -> Result<Claims>".to_string(),
                    source_text: "fn verify_jwt(token: &str) -> Result<Claims> {\n    let key = load_key()?;\n    let claims = decode(token, key)?;\n    if claims.exp < now() { return Err(\"expired\"); }\n    Ok(claims)\n}\n".to_string(),
                    is_anchor: false,
                    estimated_tokens: 40,
                },
                IncludedNode {
                    node_id: "helper_2".to_string(),
                    file_path: "src/db.rs".to_string(),
                    span: Span {
                        start_line: 10,
                        start_col: 0,
                        end_line: 25,
                        end_col: 0,
                        start_byte: 200,
                        end_byte: 500,
                    },
                    kind: NodeKind::Function,
                    name: "find_user".to_string(),
                    signature: "fn find_user(id: &str) -> Result<User>".to_string(),
                    source_text: "fn find_user(id: &str) -> Result<User> {\n    let row = conn.query(\"SELECT * FROM users WHERE id = $1\", &[id])?;\n    Ok(User::from_row(row)?)\n}\n".to_string(),
                    is_anchor: false,
                    estimated_tokens: 30,
                },
            ],
            total_estimated_tokens: 120,
            total_source_file_tokens: 5000,
            retrieval_ms: Some(45),
        }
    }

    #[test]
    fn test_compress_full_no_change() {
        let bundle = make_test_bundle();
        let compressed = compress_bundle(&bundle, ContextDepth::Full);
        // Full depth should preserve all content (minus dedup/import compression)
        assert_eq!(compressed.depth, ContextDepth::Full);
        assert_eq!(compressed.original_tokens, 120);
    }

    #[test]
    fn test_compress_anchors_reduces_tokens() {
        let bundle = make_test_bundle();
        let compressed = compress_bundle(&bundle, ContextDepth::Anchors);

        // Non-anchor nodes should be compressed to signatures
        assert!(compressed.compressed_tokens < compressed.original_tokens);
        assert!(compressed.savings_percent > 0.0);

        // Anchor should still have full source
        let anchor = compressed
            .bundle
            .included_nodes
            .iter()
            .find(|n| n.node_id == "anchor_1")
            .unwrap();
        assert!(anchor.source_text.contains("verify_jwt"));
    }

    #[test]
    fn test_compress_signatures_reduces_more() {
        let bundle = make_test_bundle();
        let anchor_compressed = compress_bundle(&bundle, ContextDepth::Anchors);
        let sig_compressed = compress_bundle(&bundle, ContextDepth::Signatures);

        // Signatures should be smaller than anchors mode
        assert!(sig_compressed.compressed_tokens <= anchor_compressed.compressed_tokens);
    }

    #[test]
    fn test_compress_summary_smallest() {
        let bundle = make_test_bundle();
        let summary_compressed = compress_bundle(&bundle, ContextDepth::Summary);

        // Summary should be the smallest
        assert!(summary_compressed.compressed_tokens < 50);
        assert!(summary_compressed.savings_percent > 50.0);
    }

    #[test]
    fn test_deduplicate_overlapping() {
        let outer = IncludedNode {
            node_id: "outer".to_string(),
            file_path: "src/mod.rs".to_string(),
            span: Span {
                start_line: 0,
                start_col: 0,
                end_line: 50,
                end_col: 0,
                start_byte: 0,
                end_byte: 1000,
            },
            kind: NodeKind::Class,
            name: "MyClass".to_string(),
            signature: "class MyClass".to_string(),
            source_text: "class MyClass { ... }".to_string(),
            is_anchor: true,
            estimated_tokens: 250,
        };

        let inner = IncludedNode {
            node_id: "inner".to_string(),
            file_path: "src/mod.rs".to_string(),
            span: Span {
                start_line: 5,
                start_col: 0,
                end_line: 10,
                end_col: 0,
                start_byte: 100,
                end_byte: 200,
            },
            kind: NodeKind::Method,
            name: "method".to_string(),
            signature: "fn method()".to_string(),
            source_text: "fn method() {}".to_string(),
            is_anchor: false,
            estimated_tokens: 10,
        };

        let result = deduplicate_overlapping(vec![outer, inner]);
        // Inner should be removed since it's contained in outer
        assert_eq!(result.len(), 1);
        assert_eq!(result[0].node_id, "outer");
    }

    #[test]
    fn test_deduplicate_keeps_different_files() {
        let node_a = IncludedNode {
            node_id: "a".to_string(),
            file_path: "src/a.rs".to_string(),
            span: Span {
                start_line: 0,
                start_col: 0,
                end_line: 50,
                end_col: 0,
                start_byte: 0,
                end_byte: 1000,
            },
            kind: NodeKind::Function,
            name: "a".to_string(),
            signature: "fn a()".to_string(),
            source_text: "fn a() {}".to_string(),
            is_anchor: true,
            estimated_tokens: 10,
        };

        let node_b = IncludedNode {
            node_id: "b".to_string(),
            file_path: "src/b.rs".to_string(),
            span: Span {
                start_line: 0,
                start_col: 0,
                end_line: 10,
                end_col: 0,
                start_byte: 0,
                end_byte: 200,
            },
            kind: NodeKind::Function,
            name: "b".to_string(),
            signature: "fn b()".to_string(),
            source_text: "fn b() {}".to_string(),
            is_anchor: false,
            estimated_tokens: 10,
        };

        let result = deduplicate_overlapping(vec![node_a, node_b]);
        assert_eq!(result.len(), 2);
    }

    #[test]
    fn test_context_depth_from_str() {
        assert_eq!(ContextDepth::from_str_loose("summary"), ContextDepth::Summary);
        assert_eq!(
            ContextDepth::from_str_loose("signatures"),
            ContextDepth::Signatures
        );
        assert_eq!(ContextDepth::from_str_loose("anchors"), ContextDepth::Anchors);
        assert_eq!(ContextDepth::from_str_loose("full"), ContextDepth::Full);
        assert_eq!(ContextDepth::from_str_loose("0"), ContextDepth::Summary);
        assert_eq!(ContextDepth::from_str_loose("3"), ContextDepth::Full);
        assert_eq!(
            ContextDepth::from_str_loose("unknown"),
            ContextDepth::Anchors
        );
    }

    #[test]
    fn test_estimate_savings() {
        let bundle = make_test_bundle();
        let (tokens, savings) = estimate_savings(&bundle, ContextDepth::Summary);
        assert!(tokens < bundle.total_estimated_tokens);
        assert!(savings > 0.0);
    }

    #[test]
    fn test_empty_bundle_compression() {
        let bundle = ContextBundle {
            bundle_id: "empty".to_string(),
            query: "test".to_string(),
            anchor_node_ids: vec![],
            included_nodes: vec![],
            total_estimated_tokens: 0,
            total_source_file_tokens: 0,
            retrieval_ms: None,
        };

        let compressed = compress_bundle(&bundle, ContextDepth::Summary);
        assert_eq!(compressed.compressed_tokens, 0);
        assert_eq!(compressed.savings_percent, 0.0);
    }

    #[test]
    fn test_compress_preserves_anchor_ids() {
        let bundle = make_test_bundle();
        let compressed = compress_bundle(&bundle, ContextDepth::Anchors);
        assert_eq!(
            compressed.bundle.anchor_node_ids,
            bundle.anchor_node_ids
        );
    }

    #[test]
    fn test_import_compression() {
        let imports: Vec<IncludedNode> = (0..5)
            .map(|i| IncludedNode {
                node_id: format!("import_{i}"),
                file_path: "src/main.rs".to_string(),
                span: Span {
                    start_line: i,
                    start_col: 0,
                    end_line: i,
                    end_col: 0,
                    start_byte: i * 30,
                    end_byte: (i + 1) * 30,
                },
                kind: NodeKind::Module,
                name: format!("import_{i}"),
                signature: format!("use crate::mod{i}"),
                source_text: format!("use crate::mod{i};\n"),
                is_anchor: false,
                estimated_tokens: 5,
            })
            .collect();

        let result = compress_imports(imports);
        // 5 imports should be consolidated into 1
        assert!(result.len() < 5);
    }
}
