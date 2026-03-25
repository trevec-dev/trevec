use anyhow::{Context, Result};
use arrow_array::types::Float32Type;
use arrow_array::{
    Array, BooleanArray, FixedSizeListArray, Float32Array, Int32Array, Int64Array, RecordBatch,
    RecordBatchIterator, StringArray, UInt32Array,
};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lancedb::index::scalar::{FtsIndexBuilder, FullTextSearchQuery};
use lancedb::index::Index;
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use lancedb::table::Table;
use lancedb::Connection;
use std::collections::HashSet;
use std::sync::Arc;
use std::time::Duration;

use crate::embedder::EMBEDDING_DIM;
use crate::store::SearchResult;
use trevec_core::model::MemoryEvent;

const TABLE_NAME: &str = "memory_events";

/// LanceDB-backed store for MemoryEvents with FTS + vector search.
pub struct MemoryStore {
    connection: Connection,
    table: Option<Table>,
}

fn escape_sql_string(value: &str) -> String {
    value.replace('\'', "''")
}

fn is_index_already_exists_error(err_text: &str) -> bool {
    let text = err_text.to_ascii_lowercase();
    text.contains("already exists")
        || text.contains("duplicate")
        || text.contains("exists")
}

fn is_missing_repo_id_error(err_text: &str) -> bool {
    let text = err_text.to_ascii_lowercase();
    text.contains("repo_id")
        && (text.contains("missing")
            || text.contains("not found")
            || text.contains("no field")
            || text.contains("schema")
            || text.contains("column"))
}

fn is_missing_fts_index_error(err_text: &str) -> bool {
    let text = err_text.to_ascii_lowercase();
    (text.contains("fts") || text.contains("full text"))
        && text.contains("index")
        && (text.contains("not found")
            || text.contains("missing")
            || text.contains("does not exist")
            || text.contains("required"))
}

async fn ensure_fts_index(table: &Table) -> Result<()> {
    match table
        .create_index(&["bm25_text"], Index::FTS(FtsIndexBuilder::default()))
        .execute()
        .await
    {
        Ok(_) => Ok(()),
        Err(err) => {
            let err_text = err.to_string();
            if is_index_already_exists_error(&err_text) {
                Ok(())
            } else {
                Err(err).context("Failed to create FTS index on memory_events")
            }
        }
    }
}

async fn run_memory_fts_query(
    table: &Table,
    processed_query: &str,
    limit: usize,
    repo_id: Option<&str>,
) -> Result<Vec<RecordBatch>> {
    let fts_query = FullTextSearchQuery::new(processed_query.to_string());
    let mut query_builder = table.query().full_text_search(fts_query);
    if let Some(repo) = repo_id {
        let predicate = format!("repo_id = '{}'", escape_sql_string(repo));
        query_builder = query_builder.only_if(predicate);
    }

    let stream = query_builder
        .select(Select::Columns(vec!["id".to_string()]))
        .limit(limit)
        .execute()
        .await?;
    let batches: Vec<RecordBatch> = stream.try_collect().await?;
    Ok(batches)
}

async fn run_memory_vector_query(
    table: &Table,
    query_vec: &[f32],
    limit: usize,
    repo_id: Option<&str>,
) -> Result<Vec<RecordBatch>> {
    let mut query_builder = table
        .vector_search(query_vec)
        .context("Failed to build memory vector search")?
        .column("symbol_vec");
    if let Some(repo) = repo_id {
        let predicate = format!("repo_id = '{}'", escape_sql_string(repo));
        query_builder = query_builder.only_if(predicate);
    }

    let stream = query_builder.limit(limit).execute().await?;
    let batches: Vec<RecordBatch> = stream.try_collect().await?;
    Ok(batches)
}

async fn run_memory_query(
    table: &Table,
    predicate: String,
    limit: usize,
) -> Result<Vec<RecordBatch>> {
    let stream = table
        .query()
        .only_if(predicate)
        .limit(limit)
        .execute()
        .await?;
    let batches: Vec<RecordBatch> = stream.try_collect().await?;
    Ok(batches)
}

async fn run_memory_query_with_select(
    table: &Table,
    predicate: String,
    columns: Vec<String>,
    limit: usize,
) -> Result<Vec<RecordBatch>> {
    let stream = table
        .query()
        .only_if(predicate)
        .select(Select::Columns(columns))
        .limit(limit)
        .execute()
        .await?;
    let batches: Vec<RecordBatch> = stream.try_collect().await?;
    Ok(batches)
}

impl MemoryStore {
    /// Force the table to re-read the latest version from disk.
    /// Call this before queries when another process may have written data.
    pub async fn refresh(&self) {
        if let Some(table) = &self.table {
            if let Err(e) = table.checkout_latest().await {
                tracing::warn!("Failed to refresh memory table to latest version: {e}");
            }
        }
    }

    /// Open or create a memory store in the given LanceDB directory.
    pub async fn open(data_dir: &str) -> Result<Self> {
        let connection = lancedb::connect(data_dir)
            .read_consistency_interval(Duration::from_secs(0))
            .execute()
            .await
            .context("Failed to connect to LanceDB for memory store")?;

        let table_names = connection
            .table_names()
            .execute()
            .await
            .context("Failed to list tables")?;

        let table = if table_names.contains(&TABLE_NAME.to_string()) {
            let table = connection
                .open_table(TABLE_NAME)
                .execute()
                .await
                .context("Failed to open memory_events table")?;

            // Best-effort healing for legacy installs where the table existed
            // before the FTS index was added.
            if let Err(err) = ensure_fts_index(&table).await {
                tracing::warn!("Failed to ensure memory_events FTS index: {err}");
            }

            Some(table)
        } else {
            None
        };

        Ok(Self { connection, table })
    }

    /// Upsert events into the store, creating the table and indexes if needed.
    pub async fn upsert_events(&mut self, events: &[MemoryEvent]) -> Result<()> {
        if events.is_empty() {
            return Ok(());
        }

        let (batch, schema) = events_to_record_batch(events)?;

        if let Some(table) = &self.table {
            let reader: Box<dyn arrow_array::RecordBatchReader + Send> =
                Box::new(RecordBatchIterator::new(vec![Ok(batch)], schema));
            let mut merge = table.merge_insert(&["id"]);
            merge.when_matched_update_all(None);
            merge.when_not_matched_insert_all();
            merge
                .execute(reader)
                .await
                .context("Failed to upsert memory events")?;

            // Rebuild the FTS index so newly upserted rows are searchable.
            // LanceDB FTS indexes are not automatically updated on insert;
            // create_index with replace=true (the default) rebuilds from
            // scratch.
            ensure_fts_index(table).await?;
        } else {
            let reader: Box<dyn arrow_array::RecordBatchReader + Send> =
                Box::new(RecordBatchIterator::new(vec![Ok(batch)], schema));
            let table = self
                .connection
                .create_table(TABLE_NAME, reader)
                .execute()
                .await
                .context("Failed to create memory_events table")?;

            // Create FTS index on bm25_text
            ensure_fts_index(&table).await?;

            self.table = Some(table);
        }

        Ok(())
    }

    /// Search by BM25 full-text search.
    pub async fn search_fts(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        self.search_fts_for_repo(query, limit, None).await
    }

    /// Search by BM25 full-text search scoped to an optional repo_id.
    pub async fn search_fts_for_repo(
        &self,
        query: &str,
        limit: usize,
        repo_id: Option<&str>,
    ) -> Result<Vec<SearchResult>> {
        let Some(table) = &self.table else {
            return Ok(vec![]);
        };

        let processed = crate::store::preprocess_fts_query(query);
        if processed.is_empty() {
            return Ok(vec![]);
        }

        let batches = match run_memory_fts_query(table, &processed, limit, repo_id).await {
            Ok(batches) => batches,
            Err(first_err) => {
                let first_err_text = first_err.to_string();

                // If the FTS index is missing (legacy table), create it and retry once.
                if is_missing_fts_index_error(&first_err_text) {
                    if let Err(index_err) = ensure_fts_index(table).await {
                        tracing::warn!(
                            "Failed to auto-create memory_events FTS index after search error: {index_err}"
                        );
                    }

                    match run_memory_fts_query(table, &processed, limit, repo_id).await {
                        Ok(batches) => batches,
                        Err(retry_err) => {
                            let retry_text = retry_err.to_string();
                            if repo_id.is_some() && is_missing_repo_id_error(&retry_text) {
                                tracing::warn!(
                                    "Memory table missing repo_id column, falling back to unscoped FTS query"
                                );
                                run_memory_fts_query(table, &processed, limit, None)
                                    .await
                                    .context("Memory FTS search failed")?
                            } else {
                                return Err(retry_err).context("Memory FTS search failed");
                            }
                        }
                    }
                } else if repo_id.is_some() && is_missing_repo_id_error(&first_err_text) {
                    tracing::warn!(
                        "Memory table missing repo_id column, falling back to unscoped FTS query"
                    );
                    run_memory_fts_query(table, &processed, limit, None)
                        .await
                        .context("Memory FTS search failed")?
                } else {
                    return Err(first_err).context("Memory FTS search failed");
                }
            }
        };

        let mut search_results = Vec::new();
        for batch in &batches {
            let Some(ids) = batch
                .column_by_name("id")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            else {
                continue;
            };
            for i in 0..ids.len() {
                search_results.push(SearchResult {
                    node_id: ids.value(i).to_string(),
                    score: 0.0,
                    rank: search_results.len() + 1,
                });
            }
        }

        Ok(search_results)
    }

    /// Search by vector similarity.
    pub async fn search_vector(
        &self,
        query_vec: &[f32],
        limit: usize,
    ) -> Result<Vec<SearchResult>> {
        self.search_vector_for_repo(query_vec, limit, None).await
    }

    /// Search by vector similarity scoped to an optional repo_id.
    pub async fn search_vector_for_repo(
        &self,
        query_vec: &[f32],
        limit: usize,
        repo_id: Option<&str>,
    ) -> Result<Vec<SearchResult>> {
        let Some(table) = &self.table else {
            return Ok(vec![]);
        };

        let batches = match run_memory_vector_query(table, query_vec, limit, repo_id).await {
            Ok(batches) => batches,
            Err(first_err) => {
                let first_err_text = first_err.to_string();
                if repo_id.is_some() && is_missing_repo_id_error(&first_err_text) {
                    tracing::warn!(
                        "Memory table missing repo_id column, falling back to unscoped vector query"
                    );
                    run_memory_vector_query(table, query_vec, limit, None)
                        .await
                        .context("Memory vector search failed")?
                } else {
                    return Err(first_err).context("Memory vector search failed");
                }
            }
        };

        let mut search_results = Vec::new();
        for batch in &batches {
            let Some(ids) = batch
                .column_by_name("id")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            else {
                continue;
            };

            let distances = batch
                .column_by_name("_distance")
                .and_then(|c| c.as_any().downcast_ref::<Float32Array>());

            for i in 0..ids.len() {
                let score = distances
                    .filter(|d| i < d.len())
                    .map(|d| 1.0 / (1.0 + d.value(i)))
                    .unwrap_or(0.0);
                search_results.push(SearchResult {
                    node_id: ids.value(i).to_string(),
                    score,
                    rank: search_results.len() + 1,
                });
            }
        }

        Ok(search_results)
    }

    /// Delete events older than the given timestamp (non-pinned only).
    pub async fn delete_expired(&self, before_ts: i64) -> Result<()> {
        let Some(table) = &self.table else {
            return Ok(());
        };
        let predicate = format!("created_at < {} AND pinned = false", before_ts);
        table
            .delete(&predicate)
            .await
            .context("Failed to delete expired memory events")?;
        Ok(())
    }

    /// Count events that would be deleted by `delete_expired`.
    pub async fn count_expired(&self, before_ts: i64) -> Result<usize> {
        let Some(table) = &self.table else {
            return Ok(0);
        };
        let predicate = format!("created_at < {} AND pinned = false", before_ts);
        table
            .count_rows(Some(predicate))
            .await
            .context("Failed to count expired events")
    }

    /// Count raw turn events older than a cutoff (non-pinned).
    pub async fn count_raw_expired(&self, before_ts: i64) -> Result<usize> {
        let Some(table) = &self.table else {
            return Ok(0);
        };
        let predicate = format!(
            "created_at < {} AND pinned = false AND event_type = 'turn'",
            before_ts
        );
        table
            .count_rows(Some(predicate))
            .await
            .context("Failed to count raw expired events")
    }

    /// Delete raw turn events older than a cutoff (non-pinned).
    pub async fn delete_raw_expired(&self, before_ts: i64) -> Result<()> {
        let Some(table) = &self.table else {
            return Ok(());
        };
        let predicate = format!(
            "created_at < {} AND pinned = false AND event_type = 'turn'",
            before_ts
        );
        table
            .delete(&predicate)
            .await
            .context("Failed to delete raw expired events")?;
        Ok(())
    }

    /// Delete the oldest non-pinned events (for count-based GC).
    /// Fetches all non-pinned IDs + timestamps, sorts by created_at ascending,
    /// then deletes the oldest `count` events. LanceDB has no ORDER BY, so
    /// sorting is done in-memory.
    pub async fn delete_oldest_nonpinned(&self, count: usize) -> Result<()> {
        let Some(table) = &self.table else {
            return Ok(());
        };

        // Fetch all non-pinned events' IDs and timestamps.
        // LanceDB defaults to limit=10, so set a high limit.
        let results = table
            .query()
            .only_if("pinned = false")
            .select(Select::Columns(vec![
                "id".to_string(),
                "created_at".to_string(),
            ]))
            .limit(100_000)
            .execute()
            .await
            .context("Failed to query non-pinned events")?;

        let batches: Vec<RecordBatch> = results
            .try_collect()
            .await
            .context("Failed to collect non-pinned events")?;

        let mut id_ts_pairs: Vec<(String, i64)> = Vec::new();
        for batch in &batches {
            let ids = batch
                .column_by_name("id")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>());
            let timestamps = batch
                .column_by_name("created_at")
                .and_then(|c| c.as_any().downcast_ref::<Int64Array>());
            if let (Some(ids), Some(ts)) = (ids, timestamps) {
                for i in 0..ids.len() {
                    id_ts_pairs.push((ids.value(i).to_string(), ts.value(i)));
                }
            }
        }

        // Sort by created_at ascending (oldest first)
        id_ts_pairs.sort_by_key(|(_, ts)| *ts);

        let ids_to_delete: Vec<String> = id_ts_pairs
            .into_iter()
            .take(count)
            .map(|(id, _)| id)
            .collect();

        if !ids_to_delete.is_empty() {
            self.delete_events(&ids_to_delete).await?;
        }

        Ok(())
    }

    /// Delete events by their IDs.
    pub async fn delete_events(&self, event_ids: &[String]) -> Result<()> {
        let Some(table) = &self.table else {
            return Ok(());
        };
        if event_ids.is_empty() {
            return Ok(());
        }

        let escaped: Vec<String> = event_ids
            .iter()
            .map(|id| format!("'{}'", id.replace('\'', "''")))
            .collect();
        let predicate = format!("id IN ({})", escaped.join(", "));

        table
            .delete(&predicate)
            .await
            .context("Failed to delete memory events")?;

        Ok(())
    }

    /// Get the total number of events.
    pub async fn count(&self) -> Result<usize> {
        let Some(table) = &self.table else {
            return Ok(0);
        };
        table
            .count_rows(None)
            .await
            .context("Failed to count memory events")
    }

    /// Retrieve events by their IDs.
    pub async fn get_events(&self, event_ids: &[String]) -> Result<Vec<MemoryEvent>> {
        self.get_events_for_repo(event_ids, None).await
    }

    /// Retrieve events by their IDs scoped to an optional repo_id.
    pub async fn get_events_for_repo(
        &self,
        event_ids: &[String],
        repo_id: Option<&str>,
    ) -> Result<Vec<MemoryEvent>> {
        let Some(table) = &self.table else {
            return Ok(vec![]);
        };
        if event_ids.is_empty() {
            return Ok(vec![]);
        }

        let escaped: Vec<String> = event_ids
            .iter()
            .map(|id| format!("'{}'", escape_sql_string(id)))
            .collect();
        let base_predicate = format!("id IN ({})", escaped.join(", "));
        let scoped_predicate = if let Some(repo) = repo_id {
            format!(
                "{} AND repo_id = '{}'",
                base_predicate,
                escape_sql_string(repo)
            )
        } else {
            base_predicate.clone()
        };

        // Use a generous limit to account for possible duplicate rows in the
        // table; LanceDB merge-insert can leave stale versions.
        let fetch_limit = event_ids.len() * 2 + 10;
        let batches = match run_memory_query(table, scoped_predicate, fetch_limit).await {
            Ok(batches) => batches,
            Err(first_err) => {
                let first_err_text = first_err.to_string();
                if repo_id.is_some() && is_missing_repo_id_error(&first_err_text) {
                    tracing::warn!(
                        "Memory table missing repo_id column, falling back to unscoped get_events query"
                    );
                    run_memory_query(table, base_predicate, fetch_limit)
                        .await
                        .context("Failed to query events by ID")?
                } else {
                    return Err(first_err).context("Failed to query events by ID");
                }
            }
        };

        let mut events = Vec::new();
        for batch in &batches {
            events.extend(record_batch_to_events(batch)?);
        }

        Ok(events)
    }

    /// Check which content_hashes already exist in the store.
    /// Returns the subset of input hashes that are present.
    pub async fn find_existing_content_hashes(
        &self,
        hashes: &[String],
    ) -> Result<HashSet<String>> {
        self.find_existing_content_hashes_for_repo(hashes, None).await
    }

    /// Check which content_hashes already exist, optionally scoped to a repo.
    pub async fn find_existing_content_hashes_for_repo(
        &self,
        hashes: &[String],
        repo_id: Option<&str>,
    ) -> Result<HashSet<String>> {
        let Some(table) = &self.table else {
            return Ok(HashSet::new());
        };
        if hashes.is_empty() {
            return Ok(HashSet::new());
        }

        let escaped: Vec<String> = hashes
            .iter()
            .map(|h| format!("'{}'", escape_sql_string(h)))
            .collect();
        let base_predicate = format!("content_hash IN ({})", escaped.join(", "));
        let scoped_predicate = if let Some(repo) = repo_id {
            format!(
                "{} AND repo_id = '{}'",
                base_predicate,
                escape_sql_string(repo)
            )
        } else {
            base_predicate.clone()
        };

        let hash_count = hashes.len();
        let batches = match run_memory_query_with_select(
            table,
            scoped_predicate,
            vec!["content_hash".to_string()],
            hash_count,
        )
        .await
        {
            Ok(batches) => batches,
            Err(first_err) => {
                let first_err_text = first_err.to_string();
                if repo_id.is_some() && is_missing_repo_id_error(&first_err_text) {
                    tracing::warn!(
                        "Memory table missing repo_id column, falling back to unscoped content-hash query"
                    );
                    run_memory_query_with_select(
                        table,
                        base_predicate,
                        vec!["content_hash".to_string()],
                        hash_count,
                    )
                        .await
                        .context("Failed to query existing content hashes")?
                } else {
                    return Err(first_err).context("Failed to query existing content hashes");
                }
            }
        };

        let mut found = HashSet::new();
        for batch in &batches {
            if let Some(col) = batch
                .column_by_name("content_hash")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            {
                for i in 0..col.len() {
                    found.insert(col.value(i).to_string());
                }
            }
        }

        Ok(found)
    }

    /// Retrieve events within a time range, sorted by recency (in-memory sort).
    pub async fn get_events_in_range(
        &self,
        start_ts: i64,
        end_ts: i64,
        limit: usize,
    ) -> Result<Vec<MemoryEvent>> {
        self.get_events_in_range_for_repo(start_ts, end_ts, limit, None)
            .await
    }

    /// Retrieve events in a time range scoped to an optional repo_id.
    pub async fn get_events_in_range_for_repo(
        &self,
        start_ts: i64,
        end_ts: i64,
        limit: usize,
        repo_id: Option<&str>,
    ) -> Result<Vec<MemoryEvent>> {
        let Some(table) = &self.table else {
            return Ok(vec![]);
        };

        let base_predicate = format!("created_at >= {} AND created_at <= {}", start_ts, end_ts);
        let scoped_predicate = if let Some(repo) = repo_id {
            format!(
                "{} AND repo_id = '{}'",
                base_predicate,
                escape_sql_string(repo)
            )
        } else {
            base_predicate.clone()
        };

        // Fetch all matching rows (LanceDB has no ORDER BY, so we sort
        // in-memory then truncate to `limit`).  Use a high fetch limit to
        // override LanceDB's default limit of 10.
        let fetch_limit = 100_000;
        let batches = match run_memory_query(table, scoped_predicate, fetch_limit).await {
            Ok(batches) => batches,
            Err(first_err) => {
                let first_err_text = first_err.to_string();
                if repo_id.is_some() && is_missing_repo_id_error(&first_err_text) {
                    tracing::warn!(
                        "Memory table missing repo_id column, falling back to unscoped time-range query"
                    );
                    run_memory_query(table, base_predicate, fetch_limit)
                        .await
                        .context("Failed to query events by time range")?
                } else {
                    return Err(first_err).context("Failed to query events by time range");
                }
            }
        };

        let mut events = Vec::new();
        for batch in &batches {
            events.extend(record_batch_to_events(batch)?);
        }

        // Sort by created_at descending (most recent first)
        events.sort_by(|a, b| b.created_at.cmp(&a.created_at));
        events.truncate(limit);
        Ok(events)
    }

    /// Drop the memory_events table.
    pub async fn clear(&mut self) -> Result<()> {
        if self.table.is_some() {
            self.connection
                .drop_table(TABLE_NAME, &[])
                .await
                .context("Failed to drop memory_events table")?;
            self.table = None;
        }
        Ok(())
    }
}

/// Convert MemoryEvents into an Arrow RecordBatch.
fn events_to_record_batch(events: &[MemoryEvent]) -> Result<(RecordBatch, Arc<Schema>)> {
    let dim = EMBEDDING_DIM as i32;

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("repo_id", DataType::Utf8, false),
        Field::new("source", DataType::Utf8, false),
        Field::new("session_id", DataType::Utf8, false),
        Field::new("turn_index", DataType::UInt32, false),
        Field::new("role", DataType::Utf8, false),
        Field::new("event_type", DataType::Utf8, false),
        Field::new("content_redacted", DataType::Utf8, false),
        Field::new("content_hash", DataType::Utf8, false),
        Field::new("created_at", DataType::Int64, false),
        Field::new("importance", DataType::Int32, false),
        Field::new("pinned", DataType::Boolean, false),
        Field::new("files_touched", DataType::Utf8, false), // JSON string
        Field::new("tool_calls", DataType::Utf8, false),    // JSON string
        Field::new("bm25_text", DataType::Utf8, false),
        Field::new(
            "symbol_vec",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim,
            ),
            true,
        ),
    ]));

    let ids: Vec<&str> = events.iter().map(|e| e.id.as_str()).collect();
    let repo_ids: Vec<&str> = events.iter().map(|e| e.repo_id.as_str()).collect();
    let sources: Vec<&str> = events.iter().map(|e| e.source.as_str()).collect();
    let session_ids: Vec<&str> = events.iter().map(|e| e.session_id.as_str()).collect();
    let turn_indices: Vec<u32> = events.iter().map(|e| e.turn_index).collect();
    let roles: Vec<&str> = events.iter().map(|e| e.role.as_str()).collect();
    let event_types: Vec<&str> = events.iter().map(|e| e.event_type.as_str()).collect();
    let contents: Vec<&str> = events.iter().map(|e| e.content_redacted.as_str()).collect();
    let content_hashes: Vec<&str> = events.iter().map(|e| e.content_hash.as_str()).collect();
    let created_ats: Vec<i64> = events.iter().map(|e| e.created_at).collect();
    let importances: Vec<i32> = events.iter().map(|e| e.importance).collect();
    let pinneds: Vec<bool> = events.iter().map(|e| e.pinned).collect();

    let files_touched_json: Vec<String> = events
        .iter()
        .map(|e| serde_json::to_string(&e.files_touched).unwrap_or_else(|_| "[]".to_string()))
        .collect();
    let files_touched_refs: Vec<&str> = files_touched_json.iter().map(|s| s.as_str()).collect();

    let tool_calls_json: Vec<String> = events
        .iter()
        .map(|e| serde_json::to_string(&e.tool_calls).unwrap_or_else(|_| "[]".to_string()))
        .collect();
    let tool_calls_refs: Vec<&str> = tool_calls_json.iter().map(|s| s.as_str()).collect();

    let bm25_texts: Vec<&str> = events.iter().map(|e| e.bm25_text.as_str()).collect();

    // Build embedding vectors.
    // Events without embeddings are stored as NULL (not zero vectors) so
    // vector search does not treat non-embedded events as semantic matches.
    let embeddings: Vec<Option<Vec<Option<f32>>>> = events
        .iter()
        .map(|event| {
            event.symbol_vec.as_ref().and_then(|vector| {
                if vector.len() != EMBEDDING_DIM {
                    tracing::warn!(
                        "Memory event {} has embedding dim {}, expected {}; storing NULL vector",
                        event.id,
                        vector.len(),
                        EMBEDDING_DIM
                    );
                    None
                } else {
                    Some(vector.iter().map(|&value| Some(value)).collect())
                }
            })
        })
        .collect();

    let vec_array = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(embeddings, dim);

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(ids)),
            Arc::new(StringArray::from(repo_ids)),
            Arc::new(StringArray::from(sources)),
            Arc::new(StringArray::from(session_ids)),
            Arc::new(UInt32Array::from(turn_indices)),
            Arc::new(StringArray::from(roles)),
            Arc::new(StringArray::from(event_types)),
            Arc::new(StringArray::from(contents)),
            Arc::new(StringArray::from(content_hashes)),
            Arc::new(Int64Array::from(created_ats)),
            Arc::new(Int32Array::from(importances)),
            Arc::new(BooleanArray::from(pinneds)),
            Arc::new(StringArray::from(files_touched_refs)),
            Arc::new(StringArray::from(tool_calls_refs)),
            Arc::new(StringArray::from(bm25_texts)),
            Arc::new(vec_array),
        ],
    )
    .context("Failed to create memory events RecordBatch")?;

    Ok((batch, schema))
}

/// Convert a RecordBatch back into MemoryEvents (for get_events).
fn record_batch_to_events(batch: &RecordBatch) -> Result<Vec<MemoryEvent>> {
    let ids = batch
        .column_by_name("id")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .context("Missing 'id' column")?;
    let repo_ids = batch
        .column_by_name("repo_id")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .context("Missing 'repo_id' column")?;
    let sources = batch
        .column_by_name("source")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .context("Missing 'source' column")?;
    let session_ids = batch
        .column_by_name("session_id")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .context("Missing 'session_id' column")?;
    let turn_indices = batch
        .column_by_name("turn_index")
        .and_then(|c| c.as_any().downcast_ref::<UInt32Array>())
        .context("Missing 'turn_index' column")?;
    let roles = batch
        .column_by_name("role")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .context("Missing 'role' column")?;
    let event_types = batch
        .column_by_name("event_type")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .context("Missing 'event_type' column")?;
    let contents = batch
        .column_by_name("content_redacted")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .context("Missing 'content_redacted' column")?;
    let content_hashes = batch
        .column_by_name("content_hash")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .context("Missing 'content_hash' column")?;
    let created_ats = batch
        .column_by_name("created_at")
        .and_then(|c| c.as_any().downcast_ref::<Int64Array>())
        .context("Missing 'created_at' column")?;
    let importances = batch
        .column_by_name("importance")
        .and_then(|c| c.as_any().downcast_ref::<Int32Array>())
        .context("Missing 'importance' column")?;
    let pinneds = batch
        .column_by_name("pinned")
        .and_then(|c| c.as_any().downcast_ref::<BooleanArray>())
        .context("Missing 'pinned' column")?;
    let files_touched_col = batch
        .column_by_name("files_touched")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .context("Missing 'files_touched' column")?;
    let tool_calls_col = batch
        .column_by_name("tool_calls")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .context("Missing 'tool_calls' column")?;
    let bm25_texts = batch
        .column_by_name("bm25_text")
        .and_then(|c| c.as_any().downcast_ref::<StringArray>())
        .context("Missing 'bm25_text' column")?;

    let mut events = Vec::new();
    for i in 0..ids.len() {
        let files_touched: Vec<String> =
            serde_json::from_str(files_touched_col.value(i)).unwrap_or_default();
        let tool_calls: Vec<String> =
            serde_json::from_str(tool_calls_col.value(i)).unwrap_or_default();

        events.push(MemoryEvent {
            id: ids.value(i).to_string(),
            repo_id: repo_ids.value(i).to_string(),
            source: sources.value(i).to_string(),
            session_id: session_ids.value(i).to_string(),
            turn_index: turn_indices.value(i),
            role: roles.value(i).to_string(),
            event_type: event_types.value(i).to_string(),
            content_redacted: contents.value(i).to_string(),
            content_hash: content_hashes.value(i).to_string(),
            created_at: created_ats.value(i),
            importance: importances.value(i),
            pinned: pinneds.value(i),
            files_touched,
            tool_calls,
            bm25_text: bm25_texts.value(i).to_string(),
            symbol_vec: None, // Don't round-trip embeddings for now
        });
    }

    Ok(events)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_event(id: &str, content: &str, ts: i64) -> MemoryEvent {
        MemoryEvent {
            id: id.to_string(),
            repo_id: "test_repo".to_string(),
            source: "test".to_string(),
            session_id: "sess1".to_string(),
            turn_index: 0,
            role: "user".to_string(),
            event_type: "turn".to_string(),
            content_redacted: content.to_string(),
            content_hash: format!("hash_{id}"),
            created_at: ts,
            importance: 0,
            pinned: false,
            files_touched: vec!["src/main.rs".to_string()],
            tool_calls: vec!["Read".to_string()],
            bm25_text: format!("test src/main.rs Read {content}"),
            symbol_vec: None,
        }
    }

    fn make_event_for_repo(id: &str, content: &str, ts: i64, repo_id: &str) -> MemoryEvent {
        let mut event = make_event(id, content, ts);
        event.repo_id = repo_id.to_string();
        event
    }

    #[tokio::test]
    async fn test_memory_store_crud() {
        let dir = tempfile::tempdir().unwrap();
        let lance_dir = dir.path().join("lance");
        let mut store = MemoryStore::open(lance_dir.to_str().unwrap()).await.unwrap();

        // Initially empty
        assert_eq!(store.count().await.unwrap(), 0);

        // Upsert
        let events = vec![
            make_event("e1", "auth login flow", 1700000000),
            make_event("e2", "database query optimization", 1700000001),
        ];
        store.upsert_events(&events).await.unwrap();
        assert_eq!(store.count().await.unwrap(), 2);

        // Update existing event
        let updated = vec![make_event("e1", "auth login flow updated", 1700000000)];
        store.upsert_events(&updated).await.unwrap();
        assert_eq!(store.count().await.unwrap(), 2);

        // Delete
        store.delete_events(&["e1".to_string()]).await.unwrap();
        assert_eq!(store.count().await.unwrap(), 1);

        // Clear
        store.clear().await.unwrap();
        assert_eq!(store.count().await.unwrap(), 0);
    }

    #[tokio::test]
    async fn test_memory_store_fts() {
        let dir = tempfile::tempdir().unwrap();
        let lance_dir = dir.path().join("lance");
        let mut store = MemoryStore::open(lance_dir.to_str().unwrap()).await.unwrap();

        let events = vec![
            make_event("e1", "authentication login JWT tokens", 1700000000),
            make_event("e2", "database query optimization postgres", 1700000001),
            make_event("e3", "authentication OAuth2 flow", 1700000002),
        ];
        store.upsert_events(&events).await.unwrap();

        let results = store.search_fts("authentication", 10).await.unwrap();
        assert!(!results.is_empty());
        // Auth events should match
        let ids: Vec<&str> = results.iter().map(|r| r.node_id.as_str()).collect();
        assert!(ids.contains(&"e1") || ids.contains(&"e3"));
    }

    #[tokio::test]
    async fn test_memory_store_delete_expired() {
        let dir = tempfile::tempdir().unwrap();
        let lance_dir = dir.path().join("lance");
        let mut store = MemoryStore::open(lance_dir.to_str().unwrap()).await.unwrap();

        let events = vec![
            make_event("old", "old event", 1000),   // very old
            make_event("new", "new event", 1700000000), // recent
        ];
        store.upsert_events(&events).await.unwrap();
        assert_eq!(store.count().await.unwrap(), 2);

        store.delete_expired(1700000000 - 1).await.unwrap();
        assert_eq!(store.count().await.unwrap(), 1);
    }

    #[tokio::test]
    async fn test_memory_store_get_events() {
        let dir = tempfile::tempdir().unwrap();
        let lance_dir = dir.path().join("lance");
        let mut store = MemoryStore::open(lance_dir.to_str().unwrap()).await.unwrap();

        let events = vec![
            make_event("e1", "first event", 1700000000),
            make_event("e2", "second event", 1700000001),
        ];
        store.upsert_events(&events).await.unwrap();

        let retrieved = store.get_events(&["e1".to_string()]).await.unwrap();
        assert_eq!(retrieved.len(), 1);
        assert_eq!(retrieved[0].id, "e1");
        assert_eq!(retrieved[0].content_redacted, "first event");
        assert_eq!(retrieved[0].files_touched, vec!["src/main.rs"]);
    }

    #[tokio::test]
    async fn test_delete_oldest_nonpinned_ordering() {
        let dir = tempfile::tempdir().unwrap();
        let lance_dir = dir.path().join("lance");
        let mut store = MemoryStore::open(lance_dir.to_str().unwrap()).await.unwrap();

        // Insert 5 events with different timestamps
        let events = vec![
            make_event("e_old1", "oldest event", 1000),
            make_event("e_old2", "second oldest", 2000),
            make_event("e_mid", "middle event", 5000),
            make_event("e_new1", "newer event", 9000),
            make_event("e_new2", "newest event", 10000),
        ];
        store.upsert_events(&events).await.unwrap();
        assert_eq!(store.count().await.unwrap(), 5);

        // Delete the 2 oldest
        store.delete_oldest_nonpinned(2).await.unwrap();
        assert_eq!(store.count().await.unwrap(), 3);

        // The oldest two (e_old1, e_old2) should be gone; the rest should remain
        let remaining = store
            .get_events(&[
                "e_old1".into(),
                "e_old2".into(),
                "e_mid".into(),
                "e_new1".into(),
                "e_new2".into(),
            ])
            .await
            .unwrap();
        let remaining_ids: Vec<&str> = remaining.iter().map(|e| e.id.as_str()).collect();
        assert!(!remaining_ids.contains(&"e_old1"));
        assert!(!remaining_ids.contains(&"e_old2"));
        assert!(remaining_ids.contains(&"e_mid"));
        assert!(remaining_ids.contains(&"e_new1"));
        assert!(remaining_ids.contains(&"e_new2"));
    }

    #[tokio::test]
    async fn test_delete_oldest_nonpinned_skips_pinned() {
        let dir = tempfile::tempdir().unwrap();
        let lance_dir = dir.path().join("lance");
        let mut store = MemoryStore::open(lance_dir.to_str().unwrap()).await.unwrap();

        let mut pinned_event = make_event("e_pinned", "pinned old event", 500);
        pinned_event.pinned = true;

        let events = vec![
            pinned_event,
            make_event("e_old", "unpinned old event", 1000),
            make_event("e_new", "unpinned new event", 9000),
        ];
        store.upsert_events(&events).await.unwrap();
        assert_eq!(store.count().await.unwrap(), 3);

        // Delete 1 oldest non-pinned — should delete e_old, not e_pinned
        store.delete_oldest_nonpinned(1).await.unwrap();
        assert_eq!(store.count().await.unwrap(), 2);

        let remaining = store
            .get_events(&["e_pinned".into(), "e_old".into(), "e_new".into()])
            .await
            .unwrap();
        let remaining_ids: Vec<&str> = remaining.iter().map(|e| e.id.as_str()).collect();
        assert!(remaining_ids.contains(&"e_pinned"));
        assert!(!remaining_ids.contains(&"e_old"));
        assert!(remaining_ids.contains(&"e_new"));
    }

    #[tokio::test]
    async fn test_find_existing_content_hashes_global() {
        let dir = tempfile::tempdir().unwrap();
        let lance_dir = dir.path().join("lance");
        let mut store = MemoryStore::open(lance_dir.to_str().unwrap()).await.unwrap();

        // Insert events with known content_hashes
        let events = vec![
            make_event("e1", "first", 1000),  // content_hash = "hash_e1"
            make_event("e2", "second", 2000), // content_hash = "hash_e2"
            make_event("e3", "third", 3000),  // content_hash = "hash_e3"
        ];
        store.upsert_events(&events).await.unwrap();

        // Query for a mix of existing and non-existing hashes
        let found = store
            .find_existing_content_hashes(&[
                "hash_e1".into(),
                "hash_e3".into(),
                "hash_nonexistent".into(),
            ])
            .await
            .unwrap();

        assert_eq!(found.len(), 2);
        assert!(found.contains("hash_e1"));
        assert!(found.contains("hash_e3"));
        assert!(!found.contains("hash_nonexistent"));
    }

    #[tokio::test]
    async fn test_find_existing_content_hashes_empty_store() {
        let dir = tempfile::tempdir().unwrap();
        let lance_dir = dir.path().join("lance");
        let store = MemoryStore::open(lance_dir.to_str().unwrap()).await.unwrap();

        let found = store
            .find_existing_content_hashes(&["hash_any".into()])
            .await
            .unwrap();
        assert!(found.is_empty());
    }

    #[tokio::test]
    async fn test_get_events_in_range_ordering() {
        let dir = tempfile::tempdir().unwrap();
        let lance_dir = dir.path().join("lance");
        let mut store = MemoryStore::open(lance_dir.to_str().unwrap()).await.unwrap();

        let events = vec![
            make_event("e1", "early", 1000),
            make_event("e2", "middle", 5000),
            make_event("e3", "late", 9000),
            make_event("e4", "outside", 15000), // outside range
        ];
        store.upsert_events(&events).await.unwrap();

        // Query range [1000, 10000] — should include e1, e2, e3 but not e4
        let results = store.get_events_in_range(1000, 10000, 100).await.unwrap();
        assert_eq!(results.len(), 3);
        // Should be sorted by created_at descending (most recent first)
        assert_eq!(results[0].id, "e3");
        assert_eq!(results[1].id, "e2");
        assert_eq!(results[2].id, "e1");
    }

    #[tokio::test]
    async fn test_get_events_in_range_respects_limit() {
        let dir = tempfile::tempdir().unwrap();
        let lance_dir = dir.path().join("lance");
        let mut store = MemoryStore::open(lance_dir.to_str().unwrap()).await.unwrap();

        let events = vec![
            make_event("e1", "first", 1000),
            make_event("e2", "second", 2000),
            make_event("e3", "third", 3000),
        ];
        store.upsert_events(&events).await.unwrap();

        let results = store.get_events_in_range(0, 10000, 2).await.unwrap();
        assert_eq!(results.len(), 2);
        // Should get the 2 most recent
        assert_eq!(results[0].id, "e3");
        assert_eq!(results[1].id, "e2");
    }

    #[tokio::test]
    async fn test_get_events_in_range_empty() {
        let dir = tempfile::tempdir().unwrap();
        let lance_dir = dir.path().join("lance");
        let mut store = MemoryStore::open(lance_dir.to_str().unwrap()).await.unwrap();

        let events = vec![make_event("e1", "event", 5000)];
        store.upsert_events(&events).await.unwrap();

        // Query a range that excludes all events
        let results = store.get_events_in_range(6000, 10000, 100).await.unwrap();
        assert!(results.is_empty());
    }

    #[tokio::test]
    async fn test_search_fts_for_repo_scoped() {
        let dir = tempfile::tempdir().unwrap();
        let lance_dir = dir.path().join("lance");
        let mut store = MemoryStore::open(lance_dir.to_str().unwrap()).await.unwrap();

        let events = vec![
            make_event_for_repo("a1", "shared phrase marker", 1000, "repo_a"),
            make_event_for_repo("b1", "shared phrase marker", 1001, "repo_b"),
        ];
        store.upsert_events(&events).await.unwrap();

        let a_results = store
            .search_fts_for_repo("shared phrase", 10, Some("repo_a"))
            .await
            .unwrap();
        assert_eq!(a_results.len(), 1);
        assert_eq!(a_results[0].node_id, "a1");

        let b_results = store
            .search_fts_for_repo("shared phrase", 10, Some("repo_b"))
            .await
            .unwrap();
        assert_eq!(b_results.len(), 1);
        assert_eq!(b_results[0].node_id, "b1");
    }

    #[tokio::test]
    async fn test_get_events_in_range_for_repo_scoped() {
        let dir = tempfile::tempdir().unwrap();
        let lance_dir = dir.path().join("lance");
        let mut store = MemoryStore::open(lance_dir.to_str().unwrap()).await.unwrap();

        let events = vec![
            make_event_for_repo("a1", "repo a", 1000, "repo_a"),
            make_event_for_repo("b1", "repo b", 1001, "repo_b"),
        ];
        store.upsert_events(&events).await.unwrap();

        let a_results = store
            .get_events_in_range_for_repo(0, 5000, 10, Some("repo_a"))
            .await
            .unwrap();
        assert_eq!(a_results.len(), 1);
        assert_eq!(a_results[0].id, "a1");
        assert_eq!(a_results[0].repo_id, "repo_a");
    }

    #[tokio::test]
    async fn test_find_existing_content_hashes_for_repo_scoped() {
        let dir = tempfile::tempdir().unwrap();
        let lance_dir = dir.path().join("lance");
        let mut store = MemoryStore::open(lance_dir.to_str().unwrap()).await.unwrap();

        let mut e1 = make_event_for_repo("a1", "same", 1000, "repo_a");
        e1.content_hash = "same_hash".to_string();
        let mut e2 = make_event_for_repo("b1", "same", 1001, "repo_b");
        e2.content_hash = "same_hash".to_string();
        store.upsert_events(&[e1, e2]).await.unwrap();

        let a_found = store
            .find_existing_content_hashes_for_repo(&["same_hash".to_string()], Some("repo_a"))
            .await
            .unwrap();
        assert_eq!(a_found.len(), 1);
        assert!(a_found.contains("same_hash"));
    }
}
