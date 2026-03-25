use std::collections::HashMap;
use std::path::Path;

use anyhow::{Context, Result};
use trevec_core::model::{CodeNode, ContextBundle, IncludedNode, NodeId};
use trevec_core::generate_bundle_id;

/// Assemble a context bundle from selected node IDs.
pub fn assemble_bundle(
    query: &str,
    anchor_ids: &[NodeId],
    included_ids: &[NodeId],
    nodes: &HashMap<NodeId, CodeNode>,
    repo_path: &Path,
) -> Result<ContextBundle> {
    let mut included_nodes = Vec::new();
    let mut total_tokens = 0;

    let anchor_set: std::collections::HashSet<&NodeId> = anchor_ids.iter().collect();
    let mut file_cache: HashMap<String, String> = HashMap::new();

    for node_id in included_ids {
        let Some(node) = nodes.get(node_id) else {
            continue;
        };

        // Read the source text for this node, caching file contents
        let source_text = if let Some(content) = file_cache.get(&node.file_path) {
            let lines: Vec<&str> = content.lines().collect();
            let start = node.span.start_line.min(lines.len());
            let end = (node.span.end_line + 1).min(lines.len());
            lines[start..end].join("\n")
        } else {
            let file_path = repo_path.join(&node.file_path);
            if file_path.exists() {
                let content = std::fs::read_to_string(&file_path)
                    .with_context(|| format!("Failed to read {}", file_path.display()))?;
                let lines: Vec<&str> = content.lines().collect();
                let start = node.span.start_line.min(lines.len());
                let end = (node.span.end_line + 1).min(lines.len());
                let text = lines[start..end].join("\n");
                file_cache.insert(node.file_path.clone(), content);
                text
            } else {
                format!("// Source not available: {}", node.file_path)
            }
        };

        let estimated_tokens = source_text.len() / 4;
        total_tokens += estimated_tokens;

        included_nodes.push(IncludedNode {
            node_id: node_id.clone(),
            file_path: node.file_path.clone(),
            span: node.span.clone(),
            kind: node.kind,
            name: node.name.clone(),
            signature: node.signature.clone(),
            source_text,
            is_anchor: anchor_set.contains(node_id),
            estimated_tokens,
        });
    }

    // Sum full file sizes for tokens-saved calculation
    let total_source_file_tokens: usize = file_cache.values().map(|c| c.len() / 4).sum();

    Ok(ContextBundle {
        bundle_id: generate_bundle_id(query),
        query: query.to_string(),
        anchor_node_ids: anchor_ids.to_vec(),
        included_nodes,
        total_estimated_tokens: total_tokens,
        total_source_file_tokens,
        retrieval_ms: None,
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use trevec_core::model::{CodeNode, NodeKind, Span};

    fn make_test_node(id: &str, name: &str) -> CodeNode {
        CodeNode {
            id: id.to_string(),
            kind: NodeKind::Function,
            file_path: "test.rs".to_string(),
            span: Span {
                start_line: 0,
                start_col: 0,
                end_line: 2,
                end_col: 1,
                start_byte: 0,
                end_byte: 40,
            },
            name: name.to_string(),
            signature: format!("fn {}()", name),
            doc_comment: None,
            identifiers: vec![],
            bm25_text: String::new(),
            symbol_vec: None,
            ast_hash: String::new(),
        }
    }

    #[test]
    fn test_assemble_bundle_marks_anchors() {
        let node_a = make_test_node("a", "func_a");
        let node_b = make_test_node("b", "func_b");

        let mut nodes = HashMap::new();
        nodes.insert("a".to_string(), node_a);
        nodes.insert("b".to_string(), node_b);

        // Use a temp dir so files don't need to exist
        let dir = tempfile::tempdir().unwrap();
        std::fs::write(dir.path().join("test.rs"), "fn func_a() {}\nfn func_b() {}\nend").unwrap();

        let bundle = assemble_bundle(
            "test query",
            &["a".to_string()],
            &["a".to_string(), "b".to_string()],
            &nodes,
            dir.path(),
        )
        .unwrap();

        assert_eq!(bundle.included_nodes.len(), 2);

        let anchor = bundle.included_nodes.iter().find(|n| n.node_id == "a").unwrap();
        assert!(anchor.is_anchor);

        let non_anchor = bundle.included_nodes.iter().find(|n| n.node_id == "b").unwrap();
        assert!(!non_anchor.is_anchor);
    }

    #[test]
    fn test_assemble_bundle_total_source_file_tokens() {
        let node_a = make_test_node("a", "func_a");
        let node_b = make_test_node("b", "func_b");

        let mut nodes = HashMap::new();
        nodes.insert("a".to_string(), node_a);
        nodes.insert("b".to_string(), node_b);

        let dir = tempfile::tempdir().unwrap();
        let file_content = "fn func_a() {}\nfn func_b() {}\nend";
        std::fs::write(dir.path().join("test.rs"), file_content).unwrap();

        let bundle = assemble_bundle(
            "test query",
            &["a".to_string()],
            &["a".to_string(), "b".to_string()],
            &nodes,
            dir.path(),
        )
        .unwrap();

        // total_source_file_tokens = file_content.len() / 4
        // Both nodes are in the same file, so only one file in the cache
        let expected = file_content.len() / 4;
        assert_eq!(bundle.total_source_file_tokens, expected);
        assert!(bundle.total_source_file_tokens > 0);
    }

    #[test]
    fn test_assemble_bundle_multiple_files() {
        let mut node_a = make_test_node("a", "func_a");
        node_a.file_path = "file1.rs".to_string();
        let mut node_b = make_test_node("b", "func_b");
        node_b.file_path = "file2.rs".to_string();

        let mut nodes = HashMap::new();
        nodes.insert("a".to_string(), node_a);
        nodes.insert("b".to_string(), node_b);

        let dir = tempfile::tempdir().unwrap();
        let content1 = "fn func_a() {}\nfn helper() {}\nfn other() {}";
        let content2 = "fn func_b() {}\nfn more() {}";
        std::fs::write(dir.path().join("file1.rs"), content1).unwrap();
        std::fs::write(dir.path().join("file2.rs"), content2).unwrap();

        let bundle = assemble_bundle(
            "test query",
            &["a".to_string()],
            &["a".to_string(), "b".to_string()],
            &nodes,
            dir.path(),
        )
        .unwrap();

        // Both files contribute to total_source_file_tokens
        let expected = content1.len() / 4 + content2.len() / 4;
        assert_eq!(bundle.total_source_file_tokens, expected);
    }

    #[test]
    fn test_assemble_bundle_empty_included_ids() {
        let nodes = HashMap::new();
        let dir = tempfile::tempdir().unwrap();

        let bundle = assemble_bundle("test query", &[], &[], &nodes, dir.path()).unwrap();

        assert!(bundle.included_nodes.is_empty());
        assert_eq!(bundle.total_estimated_tokens, 0);
        assert_eq!(bundle.total_source_file_tokens, 0);
    }

    #[test]
    fn test_assemble_bundle_missing_file() {
        let node_a = make_test_node("a", "func_a");
        let mut nodes = HashMap::new();
        nodes.insert("a".to_string(), node_a);

        let dir = tempfile::tempdir().unwrap();
        // Don't create test.rs — file doesn't exist

        let bundle = assemble_bundle(
            "test query",
            &["a".to_string()],
            &["a".to_string()],
            &nodes,
            dir.path(),
        )
        .unwrap();

        // Should still produce a node, with fallback source text
        assert_eq!(bundle.included_nodes.len(), 1);
        assert!(bundle.included_nodes[0]
            .source_text
            .contains("Source not available"));
    }
}
