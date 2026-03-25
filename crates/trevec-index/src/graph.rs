use anyhow::{Context, Result};
use petgraph::graph::DiGraph;
use petgraph::graph::NodeIndex;
use petgraph::visit::EdgeRef;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fs;
use std::path::Path;
use trevec_core::model::{Confidence, Edge, EdgeType, NodeId};

/// Serializable graph data for persistence.
#[derive(Serialize, Deserialize)]
struct GraphData {
    node_ids: Vec<NodeId>,
    edges: Vec<(usize, usize, EdgeData)>,
}

#[derive(Serialize, Deserialize)]
struct EdgeData {
    edge_type: EdgeType,
    confidence: Confidence,
}

/// In-memory directed graph of code relationships.
pub struct CodeGraph {
    graph: DiGraph<NodeId, (EdgeType, Confidence)>,
    node_map: HashMap<NodeId, NodeIndex>,
}

impl CodeGraph {
    pub fn new() -> Self {
        Self {
            graph: DiGraph::new(),
            node_map: HashMap::new(),
        }
    }

    /// Ensure a node exists in the graph.
    pub fn add_node(&mut self, node_id: &NodeId) -> NodeIndex {
        if let Some(&idx) = self.node_map.get(node_id) {
            idx
        } else {
            let idx = self.graph.add_node(node_id.clone());
            self.node_map.insert(node_id.clone(), idx);
            idx
        }
    }

    /// Add an edge between two nodes.
    pub fn add_edge(&mut self, edge: &Edge) {
        let src_idx = self.add_node(&edge.src_id);
        let dst_idx = self.add_node(&edge.dst_id);
        self.graph
            .add_edge(src_idx, dst_idx, (edge.edge_type, edge.confidence));
    }

    /// Build the graph from a list of edges (nodes are created implicitly).
    pub fn build_from_edges(&mut self, edges: &[Edge]) {
        for edge in edges {
            self.add_edge(edge);
        }
    }

    /// Get all neighbors of a node (both incoming and outgoing).
    pub fn neighbors(&self, node_id: &NodeId) -> Vec<(NodeId, EdgeType, Confidence)> {
        let Some(&idx) = self.node_map.get(node_id) else {
            return vec![];
        };

        let mut result = Vec::new();

        // Outgoing edges
        for edge_ref in self.graph.edges_directed(idx, petgraph::Direction::Outgoing) {
            let target_id = &self.graph[edge_ref.target()];
            let &(edge_type, confidence) = edge_ref.weight();
            result.push((target_id.clone(), edge_type, confidence));
        }

        // Incoming edges
        for edge_ref in self.graph.edges_directed(idx, petgraph::Direction::Incoming) {
            let source_id = &self.graph[edge_ref.source()];
            let &(edge_type, confidence) = edge_ref.weight();
            result.push((source_id.clone(), edge_type, confidence));
        }

        result
    }

    /// Get outgoing neighbors of a node (callees, contained items, etc.).
    pub fn outgoing(&self, node_id: &NodeId) -> Vec<(NodeId, EdgeType, Confidence)> {
        let Some(&idx) = self.node_map.get(node_id) else {
            return vec![];
        };

        self.graph
            .edges_directed(idx, petgraph::Direction::Outgoing)
            .map(|edge_ref| {
                let target_id = &self.graph[edge_ref.target()];
                let &(edge_type, confidence) = edge_ref.weight();
                (target_id.clone(), edge_type, confidence)
            })
            .collect()
    }

    /// Get incoming neighbors of a node (callers, containers, etc.).
    pub fn incoming(&self, node_id: &NodeId) -> Vec<(NodeId, EdgeType, Confidence)> {
        let Some(&idx) = self.node_map.get(node_id) else {
            return vec![];
        };

        self.graph
            .edges_directed(idx, petgraph::Direction::Incoming)
            .map(|edge_ref| {
                let source_id = &self.graph[edge_ref.source()];
                let &(edge_type, confidence) = edge_ref.weight();
                (source_id.clone(), edge_type, confidence)
            })
            .collect()
    }

    /// Remove a node and all its edges from the graph.
    pub fn remove_node(&mut self, node_id: &NodeId) {
        if let Some(idx) = self.node_map.remove(node_id) {
            self.graph.remove_node(idx);
            // petgraph may invalidate the last node's index after removal,
            // so rebuild node_map from the graph.
            self.node_map.clear();
            for idx in self.graph.node_indices() {
                self.node_map.insert(self.graph[idx].clone(), idx);
            }
        }
    }

    /// Get the total number of nodes.
    pub fn node_count(&self) -> usize {
        self.graph.node_count()
    }

    /// Get the total number of edges.
    pub fn edge_count(&self) -> usize {
        self.graph.edge_count()
    }

    /// Serialize the graph to a file.
    pub fn save(&self, path: &Path) -> Result<()> {
        let mut node_ids = Vec::new();
        let mut idx_to_pos: HashMap<NodeIndex, usize> = HashMap::new();

        for (i, idx) in self.graph.node_indices().enumerate() {
            node_ids.push(self.graph[idx].clone());
            idx_to_pos.insert(idx, i);
        }

        let edges: Vec<(usize, usize, EdgeData)> = self
            .graph
            .edge_indices()
            .filter_map(|ei| {
                let (src, dst) = self.graph.edge_endpoints(ei)?;
                let &(edge_type, confidence) = self.graph.edge_weight(ei)?;
                Some((
                    idx_to_pos[&src],
                    idx_to_pos[&dst],
                    EdgeData {
                        edge_type,
                        confidence,
                    },
                ))
            })
            .collect();

        let data = GraphData { node_ids, edges };
        let bytes = bincode::serialize(&data).context("Failed to serialize graph")?;
        fs::write(path, bytes).context("Failed to write graph file")?;
        Ok(())
    }

    /// Deserialize a graph from a file.
    pub fn load(path: &Path) -> Result<Self> {
        let bytes = fs::read(path).context("Failed to read graph file")?;
        let data: GraphData = bincode::deserialize(&bytes).context("Failed to deserialize graph")?;

        let mut graph = DiGraph::new();
        let mut node_map = HashMap::new();

        let indices: Vec<NodeIndex> = data
            .node_ids
            .iter()
            .map(|id| {
                let idx = graph.add_node(id.clone());
                node_map.insert(id.clone(), idx);
                idx
            })
            .collect();

        for (src_pos, dst_pos, edge_data) in data.edges {
            graph.add_edge(
                indices[src_pos],
                indices[dst_pos],
                (edge_data.edge_type, edge_data.confidence),
            );
        }

        Ok(Self { graph, node_map })
    }

    /// Extract all memory-related edges (Discussed, Triggered) as `Edge` structs.
    /// Used to carry memory edges forward when rebuilding the graph during reindex.
    pub fn extract_memory_edges(&self) -> Vec<Edge> {
        let mut edges = Vec::new();
        for ei in self.graph.edge_indices() {
            if let Some((src_idx, dst_idx)) = self.graph.edge_endpoints(ei) {
                if let Some(&(edge_type, confidence)) = self.graph.edge_weight(ei) {
                    if matches!(edge_type, EdgeType::Discussed | EdgeType::Triggered) {
                        edges.push(Edge {
                            src_id: self.graph[src_idx].clone(),
                            dst_id: self.graph[dst_idx].clone(),
                            edge_type,
                            confidence,
                        });
                    }
                }
            }
        }
        edges
    }

    /// Get all node IDs in the graph.
    pub fn all_node_ids(&self) -> Vec<NodeId> {
        self.node_map.keys().cloned().collect()
    }

    /// Count incoming edges of a specific type for a node.
    /// If `filter_type` is `None`, counts all incoming edges.
    pub fn incoming_count(&self, node_id: &NodeId, filter_type: Option<EdgeType>) -> usize {
        let Some(&idx) = self.node_map.get(node_id) else {
            return 0;
        };
        self.graph
            .edges_directed(idx, petgraph::Direction::Incoming)
            .filter(|e| filter_type.is_none_or(|t| e.weight().0 == t))
            .count()
    }

    /// Count total connections (incoming + outgoing) for a node.
    pub fn total_connections(&self, node_id: &NodeId) -> usize {
        let Some(&idx) = self.node_map.get(node_id) else {
            return 0;
        };
        self.graph
            .edges_directed(idx, petgraph::Direction::Incoming)
            .count()
            + self
                .graph
                .edges_directed(idx, petgraph::Direction::Outgoing)
                .count()
    }

    /// Clear the graph.
    pub fn clear(&mut self) {
        self.graph.clear();
        self.node_map.clear();
    }
}

impl Default for CodeGraph {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_graph_basic() {
        let mut graph = CodeGraph::new();

        let edge = Edge {
            src_id: "a".to_string(),
            dst_id: "b".to_string(),
            edge_type: EdgeType::Call,
            confidence: Confidence::Certain,
        };

        graph.add_edge(&edge);

        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 1);

        let neighbors = graph.neighbors(&"a".to_string());
        assert_eq!(neighbors.len(), 1);
        assert_eq!(neighbors[0].0, "b");
        assert_eq!(neighbors[0].1, EdgeType::Call);
    }

    #[test]
    fn test_graph_roundtrip() {
        let mut graph = CodeGraph::new();
        let edges = vec![
            Edge {
                src_id: "a".to_string(),
                dst_id: "b".to_string(),
                edge_type: EdgeType::Call,
                confidence: Confidence::Certain,
            },
            Edge {
                src_id: "b".to_string(),
                dst_id: "c".to_string(),
                edge_type: EdgeType::Contain,
                confidence: Confidence::Likely,
            },
        ];
        graph.build_from_edges(&edges);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("graph.bin");

        graph.save(&path).unwrap();
        let loaded = CodeGraph::load(&path).unwrap();

        assert_eq!(loaded.node_count(), graph.node_count());
        assert_eq!(loaded.edge_count(), graph.edge_count());
    }

    #[test]
    fn test_graph_remove_node() {
        let mut graph = CodeGraph::new();
        let edges = vec![
            Edge {
                src_id: "a".to_string(),
                dst_id: "b".to_string(),
                edge_type: EdgeType::Call,
                confidence: Confidence::Certain,
            },
            Edge {
                src_id: "b".to_string(),
                dst_id: "c".to_string(),
                edge_type: EdgeType::Call,
                confidence: Confidence::Certain,
            },
        ];
        graph.build_from_edges(&edges);
        assert_eq!(graph.node_count(), 3);
        assert_eq!(graph.edge_count(), 2);

        graph.remove_node(&"b".to_string());
        assert_eq!(graph.node_count(), 2);
        assert_eq!(graph.edge_count(), 0); // Both edges involved "b"

        // "a" and "c" should still exist
        assert!(graph.node_map.contains_key("a"));
        assert!(graph.node_map.contains_key("c"));
        assert!(!graph.node_map.contains_key("b"));
    }

    #[test]
    fn test_directional_neighbors() {
        let mut graph = CodeGraph::new();
        let edge = Edge {
            src_id: "caller".to_string(),
            dst_id: "callee".to_string(),
            edge_type: EdgeType::Call,
            confidence: Confidence::Certain,
        };
        graph.add_edge(&edge);

        let outgoing = graph.outgoing(&"caller".to_string());
        assert_eq!(outgoing.len(), 1);
        assert_eq!(outgoing[0].0, "callee");

        let incoming = graph.incoming(&"callee".to_string());
        assert_eq!(incoming.len(), 1);
        assert_eq!(incoming[0].0, "caller");

        // caller has no incoming
        assert_eq!(graph.incoming(&"caller".to_string()).len(), 0);
    }

    #[test]
    fn test_extract_memory_edges_filters_by_type() {
        let mut graph = CodeGraph::new();
        let edges = vec![
            Edge {
                src_id: "a".into(),
                dst_id: "b".into(),
                edge_type: EdgeType::Call,
                confidence: Confidence::Certain,
            },
            Edge {
                src_id: "evt1".into(),
                dst_id: "b".into(),
                edge_type: EdgeType::Discussed,
                confidence: Confidence::Certain,
            },
            Edge {
                src_id: "evt2".into(),
                dst_id: "a".into(),
                edge_type: EdgeType::Triggered,
                confidence: Confidence::Likely,
            },
            Edge {
                src_id: "a".into(),
                dst_id: "c".into(),
                edge_type: EdgeType::Import,
                confidence: Confidence::Certain,
            },
        ];
        graph.build_from_edges(&edges);

        let mem_edges = graph.extract_memory_edges();
        assert_eq!(mem_edges.len(), 2);

        let types: Vec<EdgeType> = mem_edges.iter().map(|e| e.edge_type).collect();
        assert!(types.contains(&EdgeType::Discussed));
        assert!(types.contains(&EdgeType::Triggered));
        // No Call or Import edges should appear
        assert!(!types.contains(&EdgeType::Call));
        assert!(!types.contains(&EdgeType::Import));
    }

    #[test]
    fn test_extract_memory_edges_preserves_fields() {
        let mut graph = CodeGraph::new();
        let edge = Edge {
            src_id: "mem_event_42".into(),
            dst_id: "code_node_7".into(),
            edge_type: EdgeType::Discussed,
            confidence: Confidence::Likely,
        };
        graph.add_edge(&edge);

        let extracted = graph.extract_memory_edges();
        assert_eq!(extracted.len(), 1);
        assert_eq!(extracted[0].src_id, "mem_event_42");
        assert_eq!(extracted[0].dst_id, "code_node_7");
        assert_eq!(extracted[0].edge_type, EdgeType::Discussed);
        assert_eq!(extracted[0].confidence, Confidence::Likely);
    }

    #[test]
    fn test_extract_memory_edges_empty_graph() {
        let graph = CodeGraph::new();
        assert!(graph.extract_memory_edges().is_empty());
    }

    #[test]
    fn test_extract_memory_edges_no_memory_edges() {
        let mut graph = CodeGraph::new();
        let edges = vec![
            Edge {
                src_id: "a".into(),
                dst_id: "b".into(),
                edge_type: EdgeType::Call,
                confidence: Confidence::Certain,
            },
            Edge {
                src_id: "b".into(),
                dst_id: "c".into(),
                edge_type: EdgeType::Contain,
                confidence: Confidence::Certain,
            },
        ];
        graph.build_from_edges(&edges);
        assert!(graph.extract_memory_edges().is_empty());
    }

    #[test]
    fn test_memory_edges_survive_save_load() {
        let mut graph = CodeGraph::new();
        let edges = vec![
            Edge {
                src_id: "call_a".into(),
                dst_id: "call_b".into(),
                edge_type: EdgeType::Call,
                confidence: Confidence::Certain,
            },
            Edge {
                src_id: "evt1".into(),
                dst_id: "call_b".into(),
                edge_type: EdgeType::Discussed,
                confidence: Confidence::Certain,
            },
        ];
        graph.build_from_edges(&edges);

        let dir = tempfile::tempdir().unwrap();
        let path = dir.path().join("graph.bin");
        graph.save(&path).unwrap();

        let loaded = CodeGraph::load(&path).unwrap();
        let mem_edges = loaded.extract_memory_edges();
        assert_eq!(mem_edges.len(), 1);
        assert_eq!(mem_edges[0].src_id, "evt1");
        assert_eq!(mem_edges[0].edge_type, EdgeType::Discussed);
    }

    #[test]
    fn test_memory_edges_carry_forward_on_rebuild() {
        // Simulate reindex: build graph with code + memory edges,
        // extract memory edges, rebuild with new code edges, re-add memory edges
        let mut graph = CodeGraph::new();
        let initial_edges = vec![
            Edge {
                src_id: "fn_a".into(),
                dst_id: "fn_b".into(),
                edge_type: EdgeType::Call,
                confidence: Confidence::Certain,
            },
            Edge {
                src_id: "evt1".into(),
                dst_id: "fn_b".into(),
                edge_type: EdgeType::Discussed,
                confidence: Confidence::Certain,
            },
            Edge {
                src_id: "evt2".into(),
                dst_id: "fn_a".into(),
                edge_type: EdgeType::Triggered,
                confidence: Confidence::Likely,
            },
        ];
        graph.build_from_edges(&initial_edges);

        // Extract memory edges before rebuild
        let mem_edges = graph.extract_memory_edges();
        assert_eq!(mem_edges.len(), 2);

        // Rebuild with new code edges (simulating reindex — code changed)
        let new_code_edges = vec![
            Edge {
                src_id: "fn_a".into(),
                dst_id: "fn_c".into(),
                edge_type: EdgeType::Call,
                confidence: Confidence::Certain,
            },
            Edge {
                src_id: "fn_b".into(),
                dst_id: "fn_c".into(),
                edge_type: EdgeType::Import,
                confidence: Confidence::Likely,
            },
        ];
        let mut new_graph = CodeGraph::new();
        new_graph.build_from_edges(&new_code_edges);
        new_graph.build_from_edges(&mem_edges);

        // New graph has both code and memory edges
        assert_eq!(new_graph.edge_count(), 4); // 2 code + 2 memory
        let re_extracted = new_graph.extract_memory_edges();
        assert_eq!(re_extracted.len(), 2);
    }

    #[test]
    fn test_all_node_ids() {
        let mut graph = CodeGraph::new();
        let edges = vec![
            Edge {
                src_id: "a".into(),
                dst_id: "b".into(),
                edge_type: EdgeType::Call,
                confidence: Confidence::Certain,
            },
            Edge {
                src_id: "b".into(),
                dst_id: "c".into(),
                edge_type: EdgeType::Import,
                confidence: Confidence::Likely,
            },
        ];
        graph.build_from_edges(&edges);

        let mut ids = graph.all_node_ids();
        ids.sort();
        assert_eq!(ids, vec!["a", "b", "c"]);
    }

    #[test]
    fn test_all_node_ids_empty() {
        let graph = CodeGraph::new();
        assert!(graph.all_node_ids().is_empty());
    }

    #[test]
    fn test_incoming_count() {
        let mut graph = CodeGraph::new();
        let edges = vec![
            Edge {
                src_id: "a".into(),
                dst_id: "b".into(),
                edge_type: EdgeType::Call,
                confidence: Confidence::Certain,
            },
            Edge {
                src_id: "c".into(),
                dst_id: "b".into(),
                edge_type: EdgeType::Call,
                confidence: Confidence::Certain,
            },
            Edge {
                src_id: "d".into(),
                dst_id: "b".into(),
                edge_type: EdgeType::Import,
                confidence: Confidence::Likely,
            },
        ];
        graph.build_from_edges(&edges);

        // All incoming
        assert_eq!(graph.incoming_count(&"b".into(), None), 3);
        // Only Call edges
        assert_eq!(
            graph.incoming_count(&"b".into(), Some(EdgeType::Call)),
            2
        );
        // Only Import edges
        assert_eq!(
            graph.incoming_count(&"b".into(), Some(EdgeType::Import)),
            1
        );
        // Node with no incoming
        assert_eq!(graph.incoming_count(&"a".into(), None), 0);
        // Unknown node
        assert_eq!(graph.incoming_count(&"unknown".into(), None), 0);
    }

    #[test]
    fn test_total_connections() {
        let mut graph = CodeGraph::new();
        let edges = vec![
            Edge {
                src_id: "a".into(),
                dst_id: "b".into(),
                edge_type: EdgeType::Call,
                confidence: Confidence::Certain,
            },
            Edge {
                src_id: "b".into(),
                dst_id: "c".into(),
                edge_type: EdgeType::Call,
                confidence: Confidence::Certain,
            },
            Edge {
                src_id: "d".into(),
                dst_id: "b".into(),
                edge_type: EdgeType::Import,
                confidence: Confidence::Likely,
            },
        ];
        graph.build_from_edges(&edges);

        // b has 1 outgoing (b→c) + 2 incoming (a→b, d→b) = 3
        assert_eq!(graph.total_connections(&"b".into()), 3);
        // a has 1 outgoing (a→b) + 0 incoming = 1
        assert_eq!(graph.total_connections(&"a".into()), 1);
        // c has 0 outgoing + 1 incoming (b→c) = 1
        assert_eq!(graph.total_connections(&"c".into()), 1);
        // Unknown node
        assert_eq!(graph.total_connections(&"unknown".into()), 0);
    }

    #[test]
    fn test_stale_memory_edge_pruning() {
        // Simulate: memory edge points to a node that was deleted during reindex
        let mut graph = CodeGraph::new();
        let edges = vec![
            Edge {
                src_id: "fn_a".into(),
                dst_id: "fn_b".into(),
                edge_type: EdgeType::Call,
                confidence: Confidence::Certain,
            },
            Edge {
                src_id: "evt1".into(),
                dst_id: "fn_b".into(),
                edge_type: EdgeType::Discussed,
                confidence: Confidence::Certain,
            },
            Edge {
                src_id: "evt2".into(),
                dst_id: "fn_deleted".into(),
                edge_type: EdgeType::Discussed,
                confidence: Confidence::Certain,
            },
        ];
        graph.build_from_edges(&edges);

        let mut mem_edges = graph.extract_memory_edges();
        assert_eq!(mem_edges.len(), 2);

        // After reindex, fn_deleted no longer exists. Prune stale edges.
        let live_node_ids: std::collections::HashSet<&str> =
            ["fn_a", "fn_b", "fn_c"].iter().copied().collect();
        mem_edges.retain(|e| live_node_ids.contains(e.dst_id.as_str()));

        assert_eq!(mem_edges.len(), 1);
        assert_eq!(mem_edges[0].dst_id, "fn_b");
    }
}
