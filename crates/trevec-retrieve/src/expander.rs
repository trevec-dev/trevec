use std::collections::{BinaryHeap, HashSet};
use std::cmp::Ordering;

use trevec_core::model::{Confidence, NodeId};
use trevec_core::TokenBudget;
use trevec_index::graph::CodeGraph;

/// An expansion candidate ordered by confidence (higher = better).
#[derive(Debug, Eq, PartialEq)]
struct Candidate {
    node_id: NodeId,
    confidence: Confidence,
    depth: usize,
}

impl Ord for Candidate {
    fn cmp(&self, other: &Self) -> Ordering {
        // Higher confidence first, then shallower depth
        self.confidence
            .cmp(&other.confidence)
            .then_with(|| other.depth.cmp(&self.depth))
    }
}

impl PartialOrd for Candidate {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> {
        Some(self.cmp(other))
    }
}

/// Expand graph neighborhood from anchor nodes under a token budget.
/// Uses a priority queue ordered by edge confidence (Certain > Likely > Unknown).
/// Returns the set of node IDs to include in the context bundle.
pub fn expand_graph(
    graph: &CodeGraph,
    anchor_ids: &[NodeId],
    budget: &mut TokenBudget,
    node_tokens: &dyn Fn(&NodeId) -> usize,
    max_depth: usize,
) -> Vec<NodeId> {
    let mut included: Vec<NodeId> = Vec::new();
    let mut visited: HashSet<NodeId> = HashSet::new();
    let mut heap: BinaryHeap<Candidate> = BinaryHeap::new();

    // Start with anchor nodes
    for id in anchor_ids {
        let tokens = node_tokens(id);
        if budget.try_consume(tokens) {
            included.push(id.clone());
            visited.insert(id.clone());

            // Add neighbors of anchor to the heap
            for (neighbor_id, _edge_type, confidence) in graph.neighbors(id) {
                if !visited.contains(&neighbor_id) {
                    heap.push(Candidate {
                        node_id: neighbor_id,
                        confidence,
                        depth: 1,
                    });
                }
            }
        }
    }

    // Expand from the priority queue
    while let Some(candidate) = heap.pop() {
        if budget.is_exhausted() {
            break;
        }

        if visited.contains(&candidate.node_id) {
            continue;
        }

        if candidate.depth > max_depth {
            continue;
        }

        let tokens = node_tokens(&candidate.node_id);
        if !budget.try_consume(tokens) {
            continue;
        }

        visited.insert(candidate.node_id.clone());
        included.push(candidate.node_id.clone());

        // Add this node's neighbors at depth + 1
        for (neighbor_id, _edge_type, confidence) in graph.neighbors(&candidate.node_id) {
            if !visited.contains(&neighbor_id) {
                heap.push(Candidate {
                    node_id: neighbor_id,
                    confidence,
                    depth: candidate.depth + 1,
                });
            }
        }
    }

    included
}

#[cfg(test)]
mod tests {
    use super::*;
    use trevec_core::model::{Edge, EdgeType};

    #[test]
    fn test_expand_basic() {
        let mut graph = CodeGraph::new();
        graph.build_from_edges(&[
            Edge {
                src_id: "a".to_string(),
                dst_id: "b".to_string(),
                edge_type: EdgeType::Call,
                confidence: Confidence::Certain,
            },
            Edge {
                src_id: "a".to_string(),
                dst_id: "c".to_string(),
                edge_type: EdgeType::Call,
                confidence: Confidence::Likely,
            },
            Edge {
                src_id: "b".to_string(),
                dst_id: "d".to_string(),
                edge_type: EdgeType::Call,
                confidence: Confidence::Unknown,
            },
        ]);

        let mut budget = TokenBudget::new(300);
        let token_fn = |_id: &NodeId| -> usize { 100 };

        let result = expand_graph(
            &graph,
            &["a".to_string()],
            &mut budget,
            &token_fn,
            3,
        );

        // Should include a (anchor) + b (Certain) + c (Likely) = 300 tokens
        assert!(result.contains(&"a".to_string()));
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn test_expand_respects_budget() {
        let mut graph = CodeGraph::new();
        graph.build_from_edges(&[
            Edge {
                src_id: "a".to_string(),
                dst_id: "b".to_string(),
                edge_type: EdgeType::Call,
                confidence: Confidence::Certain,
            },
            Edge {
                src_id: "a".to_string(),
                dst_id: "c".to_string(),
                edge_type: EdgeType::Call,
                confidence: Confidence::Certain,
            },
        ]);

        let mut budget = TokenBudget::new(150);
        let token_fn = |_id: &NodeId| -> usize { 100 };

        let result = expand_graph(
            &graph,
            &["a".to_string()],
            &mut budget,
            &token_fn,
            3,
        );

        // Only enough budget for anchor (100) + one neighbor (100 would exceed 150)
        assert_eq!(result.len(), 1);
    }

    #[test]
    fn test_expand_prefers_certain_over_likely() {
        let mut graph = CodeGraph::new();
        graph.build_from_edges(&[
            Edge {
                src_id: "a".to_string(),
                dst_id: "likely_node".to_string(),
                edge_type: EdgeType::Call,
                confidence: Confidence::Likely,
            },
            Edge {
                src_id: "a".to_string(),
                dst_id: "certain_node".to_string(),
                edge_type: EdgeType::Call,
                confidence: Confidence::Certain,
            },
        ]);

        let mut budget = TokenBudget::new(200);
        let token_fn = |_id: &NodeId| -> usize { 100 };

        let result = expand_graph(
            &graph,
            &["a".to_string()],
            &mut budget,
            &token_fn,
            3,
        );

        // Anchor + one neighbor (budget allows 2 total)
        assert_eq!(result.len(), 2);
        // The certain node should be included (expanded first due to higher confidence)
        assert!(result.contains(&"certain_node".to_string()));
    }
}
