//! Python bindings for the Trevec SDK via PyO3.
//!
//! Usage:
//! ```python
//! from trevec import Trevec
//!
//! # Zero config
//! tv = Trevec()
//! tv.add("I love hiking in Denver", user_id="alex")
//! results = tv.search("hobbies", user_id="alex")
//!
//! # Persistent project
//! tv = Trevec("my-project")
//!
//! # Code context
//! tv = Trevec.for_repo("/path/to/repo")
//! ```

use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;
use std::collections::HashMap;

use trevec_sdk::{EngineConfig, TrevecEngine};

/// Main Trevec class exposed to Python.
///
/// Three ways to create:
/// - `Trevec()` — zero config, ephemeral in-memory
/// - `Trevec("my-project")` — persistent by project name
/// - `Trevec.for_repo("/path/to/repo")` — code context mode
#[pyclass]
struct Trevec {
    engine: TrevecEngine,
    runtime: tokio::runtime::Runtime,
}

#[pymethods]
impl Trevec {
    /// Create a new Trevec instance.
    ///
    /// Args:
    ///     project: Optional project name for persistent storage.
    ///              If omitted, uses ephemeral in-memory mode.
    ///     brain: Enable the Brain async intelligence engine (default: False)
    ///
    /// Examples:
    ///     tv = Trevec()                    # ephemeral, zero config
    ///     tv = Trevec("my-project")        # persistent at ~/.trevec/my-project/
    ///     tv = Trevec("my-project", brain=True)  # with Brain enrichment
    #[new]
    #[pyo3(signature = (project=None, brain=false))]
    fn new(project: Option<&str>, brain: bool) -> PyResult<Self> {
        let engine = match project {
            Some(name) => TrevecEngine::new(
                name,
                EngineConfig {
                    brain_enabled: brain,
                    ..Default::default()
                },
            )
            .map_err(|e| PyValueError::new_err(format!("Failed to create engine: {e}")))?,
            None => {
                if brain {
                    let mut e = TrevecEngine::default();
                    // Can't enable brain on default engine without re-creating
                    // For now, just return default
                    e
                } else {
                    TrevecEngine::default()
                }
            }
        };

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| PyValueError::new_err(format!("Failed to create runtime: {e}")))?;

        Ok(Self { engine, runtime })
    }

    /// Create a Trevec instance for a code repository.
    ///
    /// Args:
    ///     repo_path: Path to the repository root (auto-detects from cwd if None)
    ///     brain: Enable the Brain async intelligence engine
    ///
    /// Examples:
    ///     tv = Trevec.for_repo("/path/to/repo")
    ///     tv = Trevec.for_repo()  # auto-detect from cwd
    #[staticmethod]
    #[pyo3(signature = (repo_path=None, brain=false))]
    fn for_repo(repo_path: Option<&str>, brain: bool) -> PyResult<Self> {
        let engine = TrevecEngine::for_repo(
            repo_path.map(std::path::Path::new),
            EngineConfig {
                brain_enabled: brain,
                ..Default::default()
            },
        )
        .map_err(|e| PyValueError::new_err(format!("Failed to create engine: {e}")))?;

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| PyValueError::new_err(format!("Failed to create runtime: {e}")))?;

        Ok(Self { engine, runtime })
    }

    /// Add a memory.
    ///
    /// Args:
    ///     content: The text to remember (string or list of message dicts)
    ///     user_id: User identifier for scoping memories
    ///     metadata: Optional dict of extra metadata
    ///
    /// Returns:
    ///     The memory ID (or list of IDs for message lists)
    ///
    /// Examples:
    ///     tv.add("I love hiking", user_id="alex")
    ///     tv.add([
    ///         {"role": "user", "content": "I moved to Denver"},
    ///         {"role": "assistant", "content": "Welcome!"},
    ///     ], user_id="alex")
    #[pyo3(signature = (content, user_id="default", metadata=None))]
    fn add(
        &mut self,
        content: &Bound<'_, PyAny>,
        user_id: &str,
        metadata: Option<HashMap<String, String>>,
    ) -> PyResult<PyObject> {
        Python::with_gil(|py| {
            // Check if content is a list (messages format)
            if let Ok(list) = content.downcast::<pyo3::types::PyList>() {
                let mut messages: Vec<HashMap<String, String>> = Vec::new();
                for item in list.iter() {
                    if let Ok(dict) = item.downcast::<pyo3::types::PyDict>() {
                        let mut msg = HashMap::new();
                        for (k, v) in dict.iter() {
                            if let (Ok(key), Ok(val)) = (k.extract::<String>(), v.extract::<String>())
                            {
                                msg.insert(key, val);
                            }
                        }
                        messages.push(msg);
                    }
                }

                let ids = self
                    .engine
                    .add_messages(&messages, user_id)
                    .map_err(|e| PyValueError::new_err(format!("Add failed: {e}")))?;

                Ok(ids.into_pyobject(py).unwrap().into_any().unbind())
            } else {
                // String content
                let text: String = content
                    .extract()
                    .map_err(|_| PyValueError::new_err("content must be a string or list of message dicts"))?;

                let id = self
                    .engine
                    .add(&text, user_id, metadata)
                    .map_err(|e| PyValueError::new_err(format!("Add failed: {e}")))?;

                Ok(id.into_pyobject(py).unwrap().into_any().unbind())
            }
        })
    }

    /// Search memories.
    ///
    /// Args:
    ///     query: Search query text
    ///     user_id: Optional user ID to scope search
    ///     limit: Maximum results (default: 10)
    ///
    /// Returns:
    ///     List of dicts with memory info and score
    #[pyo3(signature = (query, user_id=None, limit=10))]
    fn search(
        &self,
        query: &str,
        user_id: Option<&str>,
        limit: usize,
    ) -> Vec<HashMap<String, PyObject>> {
        let results = self.engine.search(query, user_id, limit);

        Python::with_gil(|py| {
            results
                .into_iter()
                .map(|r| {
                    let mut m = HashMap::new();
                    m.insert("id".into(), r.id.into_pyobject(py).unwrap().into_any().unbind());
                    m.insert("memory".into(), r.memory.into_pyobject(py).unwrap().into_any().unbind());
                    m.insert("score".into(), r.score.into_pyobject(py).unwrap().into_any().unbind());
                    if let Some(role) = r.role {
                        m.insert("role".into(), role.into_pyobject(py).unwrap().into_any().unbind());
                    }
                    if let Some(uid) = r.user_id {
                        m.insert("user_id".into(), uid.into_pyobject(py).unwrap().into_any().unbind());
                    }
                    if let Some(ts) = r.created_at {
                        m.insert("created_at".into(), ts.into_pyobject(py).unwrap().into_any().unbind());
                    }
                    m
                })
                .collect()
        })
    }

    /// Get all memories for a user.
    ///
    /// Args:
    ///     user_id: User identifier
    ///
    /// Returns:
    ///     List of memory dicts
    fn get_all(&self, user_id: &str) -> Vec<HashMap<String, PyObject>> {
        let results = self.engine.get_all(user_id);

        Python::with_gil(|py| {
            results
                .into_iter()
                .map(|r| {
                    let mut m = HashMap::new();
                    m.insert("id".into(), r.id.into_pyobject(py).unwrap().into_any().unbind());
                    m.insert("memory".into(), r.memory.into_pyobject(py).unwrap().into_any().unbind());
                    if let Some(ts) = r.created_at {
                        m.insert("created_at".into(), ts.into_pyobject(py).unwrap().into_any().unbind());
                    }
                    m
                })
                .collect()
        })
    }

    /// Delete a specific memory by ID.
    fn delete(&mut self, memory_id: &str) -> bool {
        self.engine.delete(memory_id)
    }

    /// Delete all memories for a user.
    fn delete_all(&mut self, user_id: &str) -> usize {
        self.engine.delete_all(user_id)
    }

    /// Index a code repository.
    ///
    /// Parses all files with Tree-sitter, generates embeddings,
    /// stores in LanceDB, and builds the code graph.
    ///
    /// Args:
    ///     repo_path: Path to the repository (uses for_repo path if None)
    ///
    /// Returns:
    ///     Dict with indexing stats (files_parsed, nodes_extracted, etc.)
    #[pyo3(signature = (repo_path=None))]
    fn index(&mut self, repo_path: Option<&str>) -> PyResult<HashMap<String, PyObject>> {
        let path = repo_path.map(std::path::Path::new);
        let stats = self.runtime.block_on(async {
            self.engine.index(path).await
        }).map_err(|e| PyValueError::new_err(format!("Index failed: {e}")))?;

        Python::with_gil(|py| {
            let mut m = HashMap::new();
            m.insert("files_discovered".into(), stats.files_discovered.into_pyobject(py).unwrap().into_any().unbind());
            m.insert("files_parsed".into(), stats.files_parsed.into_pyobject(py).unwrap().into_any().unbind());
            m.insert("nodes_extracted".into(), stats.nodes_extracted.into_pyobject(py).unwrap().into_any().unbind());
            m.insert("edges_built".into(), stats.edges_built.into_pyobject(py).unwrap().into_any().unbind());
            m.insert("total_ms".into(), (stats.total_ms as u64).into_pyobject(py).unwrap().into_any().unbind());
            Ok(m)
        })
    }

    /// Query the indexed codebase with the full hybrid retrieval pipeline.
    ///
    /// Uses BM25 + vector search, RRF merge, graph expansion, and token budgeting.
    /// Falls back to simple text search if not indexed.
    ///
    /// Args:
    ///     query: Natural language query
    ///     budget: Token budget for context assembly (default: 4096)
    ///     anchors: Number of anchor nodes (default: 5)
    ///
    /// Returns:
    ///     Dict with context bundle (query, included_nodes, total_tokens, etc.)
    #[pyo3(signature = (query, budget=4096, anchors=5))]
    fn query(&self, query: &str, budget: usize, anchors: usize) -> PyResult<HashMap<String, PyObject>> {
        let options = trevec_sdk::QueryOptions {
            budget,
            anchors,
            ..Default::default()
        };

        let bundle = self.runtime.block_on(async {
            self.engine.query(query, options).await
        }).map_err(|e| PyValueError::new_err(format!("Query failed: {e}")))?;

        Python::with_gil(|py| {
            let mut m = HashMap::new();
            m.insert("query".into(), bundle.query.clone().into_pyobject(py).unwrap().into_any().unbind());
            m.insert("total_tokens".into(), bundle.total_estimated_tokens.into_pyobject(py).unwrap().into_any().unbind());
            m.insert("total_source_tokens".into(), bundle.total_source_file_tokens.into_pyobject(py).unwrap().into_any().unbind());
            if let Some(ms) = bundle.retrieval_ms {
                m.insert("retrieval_ms".into(), ms.into_pyobject(py).unwrap().into_any().unbind());
            }

            // Format included nodes as list of dicts
            let nodes: Vec<HashMap<String, PyObject>> = bundle.included_nodes.iter().map(|n| {
                let mut nm = HashMap::new();
                nm.insert("file_path".into(), n.file_path.clone().into_pyobject(py).unwrap().into_any().unbind());
                nm.insert("name".into(), n.name.clone().into_pyobject(py).unwrap().into_any().unbind());
                nm.insert("kind".into(), n.kind.to_string().into_pyobject(py).unwrap().into_any().unbind());
                nm.insert("signature".into(), n.signature.clone().into_pyobject(py).unwrap().into_any().unbind());
                nm.insert("source".into(), n.source_text.clone().into_pyobject(py).unwrap().into_any().unbind());
                nm.insert("start_line".into(), (n.span.start_line + 1).into_pyobject(py).unwrap().into_any().unbind());
                nm.insert("end_line".into(), (n.span.end_line + 1).into_pyobject(py).unwrap().into_any().unbind());
                nm.insert("is_anchor".into(), n.is_anchor.into_pyobject(py).unwrap().to_owned().into_any().unbind());
                nm
            }).collect();
            m.insert("nodes".into(), nodes.into_pyobject(py).unwrap().into_any().unbind());

            // Text context (formatted)
            m.insert("context".into(), bundle.format_text().into_pyobject(py).unwrap().into_any().unbind());

            Ok(m)
        })
    }

    /// Whether the full pipeline has been initialized.
    fn is_indexed(&self) -> bool {
        self.engine.is_indexed()
    }

    /// Get the number of nodes in the graph.
    fn node_count(&self) -> usize {
        self.engine.node_count()
    }

    /// List registered parser domains.
    fn domains(&self) -> Vec<String> {
        self.engine
            .registered_domains()
            .into_iter()
            .map(|s| s.to_string())
            .collect()
    }

    /// Ingest a conversation file (JSON format).
    fn ingest_conversation(&mut self, file_path: &str, content: &str) -> PyResult<usize> {
        self.engine
            .ingest_conversation(file_path, content.as_bytes())
            .map_err(|e| PyValueError::new_err(format!("Ingest failed: {e}")))
    }

    /// Queue nodes for Brain enrichment.
    fn enrich(&mut self, node_ids: Vec<String>) -> PyResult<()> {
        self.runtime.block_on(async {
            self.engine
                .enrich_nodes(node_ids, trevec_brain::queue::Priority::High)
                .await;
        });
        Ok(())
    }

    /// Process pending Brain enrichment tasks.
    fn process_brain(&mut self) -> usize {
        self.runtime
            .block_on(async { self.engine.process_brain().await })
    }

    /// Get Brain statistics.
    fn brain_stats(&self) -> Option<HashMap<String, PyObject>> {
        let stats = self
            .runtime
            .block_on(async { self.engine.brain_stats().await })?;

        Python::with_gil(|py| {
            let mut m = HashMap::new();
            m.insert("nodes_enriched".into(), stats.nodes_enriched.into_pyobject(py).unwrap().into_any().unbind());
            m.insert("cache_hits".into(), stats.cache_hits.into_pyobject(py).unwrap().into_any().unbind());
            m.insert("llm_calls".into(), stats.llm_calls.into_pyobject(py).unwrap().into_any().unbind());
            m.insert("estimated_cost".into(), stats.estimated_cost.into_pyobject(py).unwrap().into_any().unbind());
            m.insert("active".into(), stats.active.into_pyobject(py).unwrap().to_owned().into_any().unbind());
            Some(m)
        })
    }

    fn __repr__(&self) -> String {
        let project = self.engine.project().unwrap_or("ephemeral");
        format!(
            "Trevec(project='{}', nodes={})",
            project,
            self.engine.node_count()
        )
    }
}

/// Python module definition.
#[pymodule]
fn trevec(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Trevec>()?;
    m.add("__version__", env!("CARGO_PKG_VERSION"))?;
    Ok(())
}
