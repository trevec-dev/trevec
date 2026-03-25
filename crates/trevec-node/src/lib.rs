//! Native Node.js bindings for the Trevec SDK via napi-rs.
//!
//! Usage from TypeScript/JavaScript:
//! ```typescript
//! const { Trevec } = require('trevec-native');
//!
//! const tv = new Trevec();
//! tv.add("I love hiking", "alex");
//! const results = tv.search("hobbies", "alex");
//! ```

use napi::bindgen_prelude::*;
use napi_derive::napi;
use std::collections::HashMap;

use trevec_sdk::{EngineConfig, TrevecEngine};

#[napi]
pub struct Trevec {
    engine: TrevecEngine,
    runtime: tokio::runtime::Runtime,
}

#[napi]
impl Trevec {
    /// Create a new Trevec instance.
    /// - No args: ephemeral in-memory
    /// - With project name: persistent at ~/.trevec/<project>/
    #[napi(constructor)]
    pub fn new(project: Option<String>) -> Result<Self> {
        let engine = match project {
            Some(name) => TrevecEngine::new(
                name,
                EngineConfig::default(),
            )
            .map_err(|e| Error::from_reason(format!("Failed to create engine: {e}")))?,
            None => TrevecEngine::default(),
        };

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| Error::from_reason(format!("Failed to create runtime: {e}")))?;

        Ok(Self { engine, runtime })
    }

    /// Create a Trevec instance for a code repository.
    #[napi(factory)]
    pub fn for_repo(repo_path: Option<String>) -> Result<Self> {
        let path = repo_path.as_deref().map(std::path::Path::new);
        let engine = TrevecEngine::for_repo(path, EngineConfig::default())
            .map_err(|e| Error::from_reason(format!("Failed to create engine: {e}")))?;

        let runtime = tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
            .map_err(|e| Error::from_reason(format!("Failed to create runtime: {e}")))?;

        Ok(Self { engine, runtime })
    }

    /// Add a memory.
    #[napi]
    pub fn add(
        &mut self,
        content: String,
        user_id: String,
        metadata: Option<HashMap<String, String>>,
    ) -> Result<String> {
        self.engine
            .add(&content, &user_id, metadata)
            .map_err(|e| Error::from_reason(format!("Add failed: {e}")))
    }

    /// Search memories. Scoped by userId if provided.
    #[napi]
    pub fn search(
        &self,
        query: String,
        user_id: Option<String>,
        limit: Option<u32>,
    ) -> Vec<MemoryResult> {
        let results = self
            .engine
            .search(&query, user_id.as_deref(), limit.unwrap_or(10) as usize);

        results
            .into_iter()
            .map(|r| MemoryResult {
                id: r.id,
                memory: r.memory,
                user_id: r.user_id,
                role: r.role,
                created_at: r.created_at,
                score: r.score,
            })
            .collect()
    }

    /// Get all memories for a user.
    #[napi]
    pub fn get_all(&self, user_id: String) -> Vec<MemoryResult> {
        self.engine
            .get_all(&user_id)
            .into_iter()
            .map(|r| MemoryResult {
                id: r.id,
                memory: r.memory,
                user_id: r.user_id,
                role: r.role,
                created_at: r.created_at,
                score: r.score,
            })
            .collect()
    }

    /// Delete a specific memory.
    #[napi]
    pub fn delete(&mut self, memory_id: String) -> bool {
        self.engine.delete(&memory_id)
    }

    /// Delete all memories for a user.
    #[napi]
    pub fn delete_all(&mut self, user_id: String) -> u32 {
        self.engine.delete_all(&user_id) as u32
    }

    /// Total number of memories.
    #[napi(getter)]
    pub fn node_count(&self) -> u32 {
        self.engine.node_count() as u32
    }

    /// Whether the full pipeline has been indexed.
    #[napi(getter)]
    pub fn is_indexed(&self) -> bool {
        self.engine.is_indexed()
    }

    /// Index a code repository.
    #[napi]
    pub fn index(&mut self, repo_path: Option<String>) -> Result<IndexStats> {
        let path = repo_path.as_deref().map(std::path::Path::new);
        let stats = self
            .runtime
            .block_on(async { self.engine.index(path).await })
            .map_err(|e| Error::from_reason(format!("Index failed: {e}")))?;

        Ok(IndexStats {
            files_parsed: stats.files_parsed as u32,
            nodes_extracted: stats.nodes_extracted as u32,
            edges_built: stats.edges_built as u32,
            total_ms: stats.total_ms as u32,
        })
    }

    /// Query the indexed codebase.
    #[napi]
    pub fn query(&self, query_text: String, budget: Option<u32>) -> Result<QueryResult> {
        let options = trevec_sdk::QueryOptions {
            budget: budget.unwrap_or(4096) as usize,
            ..Default::default()
        };

        let bundle = self
            .runtime
            .block_on(async { self.engine.query(&query_text, options).await })
            .map_err(|e| Error::from_reason(format!("Query failed: {e}")))?;

        Ok(QueryResult {
            query: bundle.query.clone(),
            context: bundle.format_text(),
            total_tokens: bundle.total_estimated_tokens as u32,
            retrieval_ms: bundle.retrieval_ms.unwrap_or(0) as u32,
        })
    }

    #[napi]
    pub fn to_string(&self) -> String {
        let project = self.engine.project().unwrap_or("ephemeral");
        format!("Trevec(project='{}', nodes={})", project, self.engine.node_count())
    }
}

#[napi(object)]
pub struct MemoryResult {
    pub id: String,
    pub memory: String,
    pub user_id: Option<String>,
    pub role: Option<String>,
    pub created_at: Option<i64>,
    pub score: f64,
}

#[napi(object)]
pub struct IndexStats {
    pub files_parsed: u32,
    pub nodes_extracted: u32,
    pub edges_built: u32,
    pub total_ms: u32,
}

#[napi(object)]
pub struct QueryResult {
    pub query: String,
    pub context: String,
    pub total_tokens: u32,
    pub retrieval_ms: u32,
}
