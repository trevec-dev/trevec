use std::collections::HashMap;
use trevec_core::model::{CodeNode, Confidence, Edge, EdgeType};

use crate::extract::{Reference, ReferenceKind};

/// Build edges from extracted nodes and references.
pub fn build_edges(
    nodes: &[CodeNode],
    references: &[Reference],
) -> Vec<Edge> {
    let mut edges = Vec::new();

    // Build lookup maps
    let mut nodes_by_name: HashMap<&str, Vec<&CodeNode>> = HashMap::new();
    let mut nodes_by_id: HashMap<&str, &CodeNode> = HashMap::new();

    for node in nodes {
        nodes_by_name
            .entry(node.name.as_str())
            .or_default()
            .push(node);
        nodes_by_id.insert(node.id.as_str(), node);
    }

    // 1. Containment edges: if one node's span contains another in the same file
    build_containment_edges(nodes, &mut edges);

    // 2. Call edges: match reference.call names to definition names
    for reference in references.iter().filter(|r| r.kind == ReferenceKind::Call) {
        if let Some(targets) = nodes_by_name.get(reference.name.as_str()) {
            // Find the source node that contains this reference
            let source_node = find_containing_node(nodes, &reference.file_path, reference.start_byte);

            if let Some(src) = source_node {
                for target in targets {
                    // Don't create self-edges
                    if src.id == target.id {
                        continue;
                    }

                    let confidence = if src.file_path == target.file_path {
                        Confidence::Certain
                    } else if targets.len() == 1 {
                        Confidence::Likely
                    } else {
                        Confidence::Unknown
                    };

                    edges.push(Edge {
                        src_id: src.id.clone(),
                        dst_id: target.id.clone(),
                        edge_type: EdgeType::Call,
                        confidence,
                    });
                }
            }
        }
    }

    // 3. Import edges: match reference.import names to definitions
    for reference in references.iter().filter(|r| r.kind == ReferenceKind::Import) {
        if let Some(targets) = nodes_by_name.get(reference.name.as_str()) {
            // Find the source node that contains this import
            let source_node = find_containing_node(nodes, &reference.file_path, reference.start_byte);

            if let Some(src) = source_node {
                for target in targets {
                    if src.id == target.id {
                        continue;
                    }

                    let confidence = if targets.len() == 1 {
                        Confidence::Likely
                    } else {
                        Confidence::Unknown
                    };

                    edges.push(Edge {
                        src_id: src.id.clone(),
                        dst_id: target.id.clone(),
                        edge_type: EdgeType::Import,
                        confidence,
                    });
                }
            }
        }
    }

    // 4. Doc Reference edges: match DocSection identifiers to code node names
    for node in nodes.iter().filter(|n| n.kind == trevec_core::NodeKind::DocSection) {
        for identifier in &node.identifiers {
            // Strip trailing () from function-style references like `authenticate()`
            let clean_name = identifier.trim_end_matches("()");
            if clean_name.is_empty() {
                continue;
            }

            if let Some(targets) = nodes_by_name.get(clean_name) {
                // Only link to code nodes, not other doc sections
                let code_targets: Vec<&&CodeNode> = targets
                    .iter()
                    .filter(|t| t.kind != trevec_core::NodeKind::DocSection)
                    .collect();

                for target in &code_targets {
                    edges.push(Edge {
                        src_id: node.id.clone(),
                        dst_id: target.id.clone(),
                        edge_type: EdgeType::Reference,
                        confidence: if code_targets.len() == 1 {
                            Confidence::Likely
                        } else {
                            Confidence::Unknown
                        },
                    });
                }
            }
        }
    }

    // Deduplicate edges
    edges.sort_by(|a, b| {
        (&a.src_id, &a.dst_id, &a.edge_type)
            .cmp(&(&b.src_id, &b.dst_id, &b.edge_type))
    });
    edges.dedup_by(|a, b| {
        a.src_id == b.src_id && a.dst_id == b.dst_id && a.edge_type == b.edge_type
    });

    edges
}

/// Build containment edges: class contains method, module contains function, etc.
fn build_containment_edges(nodes: &[CodeNode], edges: &mut Vec<Edge>) {
    // Sort by span size (largest first) for efficient containment checking
    let mut sorted: Vec<&CodeNode> = nodes.iter().collect();
    sorted.sort_by(|a, b| {
        b.span
            .byte_length()
            .cmp(&a.span.byte_length())
    });

    for i in 0..sorted.len() {
        for j in (i + 1)..sorted.len() {
            let outer = sorted[i];
            let inner = sorted[j];

            // Must be same file
            if outer.file_path != inner.file_path {
                continue;
            }

            // Check containment
            if outer.span.contains(&inner.span) {
                // Only create contain edges for logical containment
                // (class→method, module→function, class→class, etc.)
                let is_logical_container = matches!(
                    outer.kind,
                    trevec_core::NodeKind::Class
                        | trevec_core::NodeKind::Module
                        | trevec_core::NodeKind::Struct
                        | trevec_core::NodeKind::Trait
                        | trevec_core::NodeKind::Enum
                        | trevec_core::NodeKind::Interface
                        | trevec_core::NodeKind::DocSection
                );

                if is_logical_container {
                    edges.push(Edge {
                        src_id: outer.id.clone(),
                        dst_id: inner.id.clone(),
                        edge_type: EdgeType::Contain,
                        confidence: Confidence::Certain,
                    });
                }
            }
        }
    }
}

/// Find the innermost node that contains the given byte offset in the given file.
fn find_containing_node<'a>(
    nodes: &'a [CodeNode],
    file_path: &str,
    byte_offset: usize,
) -> Option<&'a CodeNode> {
    let mut best: Option<&CodeNode> = None;
    let mut best_size = usize::MAX;

    for node in nodes {
        if node.file_path != file_path {
            continue;
        }
        if node.span.start_byte <= byte_offset && byte_offset < node.span.end_byte {
            let size = node.span.byte_length();
            if size < best_size {
                best = Some(node);
                best_size = size;
            }
        }
    }

    best
}

#[cfg(test)]
mod tests {
    use super::*;
    use trevec_core::model::{NodeKind, Span};

    fn make_node(id: &str, name: &str, kind: NodeKind, file: &str, start: usize, end: usize) -> CodeNode {
        CodeNode {
            id: id.to_string(),
            kind,
            file_path: file.to_string(),
            span: Span {
                start_line: 0,
                start_col: 0,
                end_line: 10,
                end_col: 0,
                start_byte: start,
                end_byte: end,
            },
            name: name.to_string(),
            signature: format!("{} {}", kind, name),
            doc_comment: None,
            identifiers: vec![],
            bm25_text: String::new(),
            symbol_vec: None,
            ast_hash: String::new(),
        }
    }

    #[test]
    fn test_containment_edges() {
        let nodes = vec![
            make_node("class1", "MyClass", NodeKind::Class, "test.py", 0, 200),
            make_node("method1", "do_stuff", NodeKind::Method, "test.py", 50, 150),
            make_node("func1", "helper", NodeKind::Function, "test.py", 250, 350),
        ];

        let edges = build_edges(&nodes, &[]);

        let contain_edges: Vec<_> = edges
            .iter()
            .filter(|e| e.edge_type == EdgeType::Contain)
            .collect();
        assert_eq!(contain_edges.len(), 1);
        assert_eq!(contain_edges[0].src_id, "class1");
        assert_eq!(contain_edges[0].dst_id, "method1");
        assert_eq!(contain_edges[0].confidence, Confidence::Certain);
    }

    #[test]
    fn test_call_edges() {
        let nodes = vec![
            make_node("func1", "caller", NodeKind::Function, "a.rs", 0, 100),
            make_node("func2", "callee", NodeKind::Function, "a.rs", 200, 300),
        ];

        let refs = vec![Reference {
            name: "callee".to_string(),
            kind: ReferenceKind::Call,
            file_path: "a.rs".to_string(),
            start_byte: 50,
            end_byte: 56,
        }];

        let edges = build_edges(&nodes, &refs);
        let call_edges: Vec<_> = edges
            .iter()
            .filter(|e| e.edge_type == EdgeType::Call)
            .collect();
        assert_eq!(call_edges.len(), 1);
        assert_eq!(call_edges[0].src_id, "func1");
        assert_eq!(call_edges[0].dst_id, "func2");
        assert_eq!(call_edges[0].confidence, Confidence::Certain);
    }

    #[test]
    fn test_doc_reference_edges() {
        // Doc section mentions `callee` which exists as a code node
        let mut doc_node = make_node("doc1", "API Guide", NodeKind::DocSection, "api.md", 0, 200);
        doc_node.identifiers = vec!["callee".to_string(), "unknown_func".to_string()];

        let nodes = vec![
            doc_node,
            make_node("func1", "callee", NodeKind::Function, "a.rs", 0, 100),
        ];

        let edges = build_edges(&nodes, &[]);
        let ref_edges: Vec<_> = edges
            .iter()
            .filter(|e| e.edge_type == EdgeType::Reference)
            .collect();

        assert_eq!(ref_edges.len(), 1, "Should create 1 reference edge for 'callee'");
        assert_eq!(ref_edges[0].src_id, "doc1");
        assert_eq!(ref_edges[0].dst_id, "func1");
        assert_eq!(ref_edges[0].confidence, Confidence::Likely);
    }

    #[test]
    fn test_doc_containment_edges() {
        // Parent doc section contains child doc section
        let nodes = vec![
            make_node("doc1", "Top", NodeKind::DocSection, "doc.md", 0, 500),
            make_node("doc2", "Child", NodeKind::DocSection, "doc.md", 100, 300),
        ];

        let edges = build_edges(&nodes, &[]);
        let contain_edges: Vec<_> = edges
            .iter()
            .filter(|e| e.edge_type == EdgeType::Contain)
            .collect();

        assert_eq!(contain_edges.len(), 1, "Parent DocSection should contain child");
        assert_eq!(contain_edges[0].src_id, "doc1");
        assert_eq!(contain_edges[0].dst_id, "doc2");
    }

    #[test]
    fn test_doc_reference_strips_parens() {
        // Doc mentions `authenticate()` — should match code node named `authenticate`
        let mut doc_node = make_node("doc1", "Auth", NodeKind::DocSection, "auth.md", 0, 200);
        doc_node.identifiers = vec!["authenticate()".to_string()];

        let nodes = vec![
            doc_node,
            make_node("func1", "authenticate", NodeKind::Function, "auth.rs", 0, 100),
        ];

        let edges = build_edges(&nodes, &[]);
        let ref_edges: Vec<_> = edges
            .iter()
            .filter(|e| e.edge_type == EdgeType::Reference)
            .collect();

        assert_eq!(ref_edges.len(), 1, "Should match after stripping ()");
        assert_eq!(ref_edges[0].dst_id, "func1");
    }
}
