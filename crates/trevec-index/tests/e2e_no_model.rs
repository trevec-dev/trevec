//! End-to-end integration test that exercises the full pipeline without
//! requiring the fastembed model download. Nodes get zero vectors.

use std::fs;
use tempfile::tempdir;

use trevec_core::model::CodeNode;
use trevec_index::graph::CodeGraph;
use trevec_index::store::Store;
use trevec_parse::edges::build_edges;
use trevec_parse::extract::extract_from_source;
use trevec_parse::languages::language_for_extension;
use trevec_parse::walker::discover_files;

const FIXTURE_RS: &str = r#"
use std::collections::HashMap;

pub fn authenticate(user: &str, pass: &str) -> bool {
    verify_hash(user, pass)
}

fn verify_hash(user: &str, pass: &str) -> bool {
    let _ = (user, pass);
    true
}

pub struct Session {
    pub token: String,
    pub user: String,
}

impl Session {
    pub fn new(user: String) -> Self {
        Self { token: "tok".to_string(), user }
    }
}
"#;

#[tokio::test]
async fn e2e_pipeline_without_model() {
    let tmp = tempdir().unwrap();
    let repo = tmp.path().join("repo");
    let src = repo.join("src");
    fs::create_dir_all(&src).unwrap();
    fs::write(src.join("auth.rs"), FIXTURE_RS).unwrap();

    // 1. Discover files
    let files = discover_files(&repo, &[]).unwrap();
    assert_eq!(files.len(), 1, "should discover one file");

    // 2. Extract nodes from source
    let file_path = &files[0];
    let source = fs::read(file_path).unwrap();
    let relative = file_path
        .strip_prefix(&repo)
        .unwrap()
        .to_string_lossy()
        .replace('\\', "/");
    let lang_config = language_for_extension("rs").expect("Rust should be supported");
    let result = extract_from_source(&relative, &source, &lang_config).unwrap();

    assert!(
        result.nodes.len() >= 3,
        "should extract at least 3 nodes (2 functions + 1 struct + impl), got {}",
        result.nodes.len()
    );

    // 3. Build edges
    let edges = build_edges(&result.nodes, &result.references);
    // authenticate calls verify_hash, so we expect at least 1 edge
    assert!(
        !edges.is_empty(),
        "should have at least one edge (authenticate -> verify_hash)"
    );

    // 4. Graph roundtrip
    let mut graph = CodeGraph::new();
    graph.build_from_edges(&edges);
    assert!(graph.node_count() > 0);
    assert!(graph.edge_count() > 0);

    let graph_path = tmp.path().join("graph.bin");
    graph.save(&graph_path).unwrap();
    let loaded = CodeGraph::load(&graph_path).unwrap();
    assert_eq!(loaded.node_count(), graph.node_count());
    assert_eq!(loaded.edge_count(), graph.edge_count());

    // 5. Store: upsert nodes with zero vectors
    let lance_dir = tmp.path().join("lance");
    let mut store = Store::open(lance_dir.to_str().unwrap()).await.unwrap();

    // Set symbol_vec to zero vectors (384-dim like bge-small-en)
    let mut nodes: Vec<CodeNode> = result.nodes;
    for node in &mut nodes {
        node.symbol_vec = Some(vec![0.0f32; 384]);
    }

    store.upsert_nodes(&nodes).await.unwrap();

    let count = store.count().await.unwrap();
    assert_eq!(count, nodes.len(), "store row count should match node count");

    // 6. FTS search
    let fts = store.search_fts("authenticate", 5).await.unwrap();
    assert!(
        !fts.is_empty(),
        "FTS search for 'authenticate' should return results"
    );
    // The top result should be the authenticate function
    let top_node = nodes.iter().find(|n| n.id == fts[0].node_id);
    assert!(top_node.is_some(), "top FTS result should map to a known node");
}
