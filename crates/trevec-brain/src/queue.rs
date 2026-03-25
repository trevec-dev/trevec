//! Priority enrichment queue for the Brain.
//!
//! Tasks are ordered by priority:
//! 1. Critical: Recently queried nodes (hot path enrichment)
//! 2. High: Public/exported symbols (API surface)
//! 3. Medium: High-connectivity nodes (graph hotspots)
//! 4. Low: Recently changed nodes
//! 5. Background: All remaining nodes (sweep)

use std::collections::BinaryHeap;
use std::sync::Mutex;

/// Priority levels for enrichment tasks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum Priority {
    Background = 0,
    Low = 1,
    Medium = 2,
    High = 3,
    Critical = 4,
}

/// Types of enrichment tasks.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TaskType {
    IntentSummary,
    EntityResolution,
    LinkPrediction,
    Observation,
}

/// A task in the enrichment queue.
#[derive(Debug, Clone)]
pub struct EnrichmentTask {
    pub node_id: String,
    pub priority: Priority,
    pub task_type: TaskType,
}

impl PartialEq for EnrichmentTask {
    fn eq(&self, other: &Self) -> bool {
        self.priority == other.priority
    }
}

impl Eq for EnrichmentTask {}

impl PartialOrd for EnrichmentTask {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for EnrichmentTask {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.priority.cmp(&other.priority)
    }
}

/// Thread-safe priority queue for enrichment tasks.
pub struct EnrichmentQueue {
    heap: Mutex<BinaryHeap<EnrichmentTask>>,
}

impl EnrichmentQueue {
    pub fn new() -> Self {
        Self {
            heap: Mutex::new(BinaryHeap::new()),
        }
    }

    /// Push a task into the queue.
    pub fn push(&self, task: EnrichmentTask) {
        let mut heap = self.heap.lock().unwrap();
        heap.push(task);
    }

    /// Pop the highest-priority task from the queue.
    pub fn pop(&self) -> Option<EnrichmentTask> {
        let mut heap = self.heap.lock().unwrap();
        heap.pop()
    }

    /// Peek at the highest-priority task without removing it.
    pub fn peek_priority(&self) -> Option<Priority> {
        let heap = self.heap.lock().unwrap();
        heap.peek().map(|t| t.priority)
    }

    /// Number of tasks in the queue.
    pub fn len(&self) -> usize {
        let heap = self.heap.lock().unwrap();
        heap.len()
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Clear all tasks from the queue.
    pub fn clear(&self) {
        let mut heap = self.heap.lock().unwrap();
        heap.clear();
    }
}

impl Default for EnrichmentQueue {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_queue_priority_ordering() {
        let queue = EnrichmentQueue::new();

        queue.push(EnrichmentTask {
            node_id: "low".to_string(),
            priority: Priority::Low,
            task_type: TaskType::IntentSummary,
        });
        queue.push(EnrichmentTask {
            node_id: "critical".to_string(),
            priority: Priority::Critical,
            task_type: TaskType::IntentSummary,
        });
        queue.push(EnrichmentTask {
            node_id: "medium".to_string(),
            priority: Priority::Medium,
            task_type: TaskType::IntentSummary,
        });

        assert_eq!(queue.len(), 3);

        // Should pop in priority order: critical, medium, low
        assert_eq!(queue.pop().unwrap().node_id, "critical");
        assert_eq!(queue.pop().unwrap().node_id, "medium");
        assert_eq!(queue.pop().unwrap().node_id, "low");
        assert!(queue.pop().is_none());
    }

    #[test]
    fn test_queue_empty() {
        let queue = EnrichmentQueue::new();
        assert!(queue.is_empty());
        assert!(queue.pop().is_none());
        assert!(queue.peek_priority().is_none());
    }

    #[test]
    fn test_queue_clear() {
        let queue = EnrichmentQueue::new();
        queue.push(EnrichmentTask {
            node_id: "a".to_string(),
            priority: Priority::High,
            task_type: TaskType::IntentSummary,
        });
        queue.push(EnrichmentTask {
            node_id: "b".to_string(),
            priority: Priority::Low,
            task_type: TaskType::IntentSummary,
        });

        assert_eq!(queue.len(), 2);
        queue.clear();
        assert!(queue.is_empty());
    }

    #[test]
    fn test_queue_peek_priority() {
        let queue = EnrichmentQueue::new();
        queue.push(EnrichmentTask {
            node_id: "a".to_string(),
            priority: Priority::Low,
            task_type: TaskType::IntentSummary,
        });
        queue.push(EnrichmentTask {
            node_id: "b".to_string(),
            priority: Priority::High,
            task_type: TaskType::IntentSummary,
        });

        assert_eq!(queue.peek_priority(), Some(Priority::High));
    }
}
