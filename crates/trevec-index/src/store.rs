use anyhow::{Context, Result};
use arrow_array::types::Float32Type;
use arrow_array::{Array, FixedSizeListArray, Float32Array, RecordBatch, RecordBatchIterator, StringArray};
use arrow_schema::{DataType, Field, Schema};
use futures::TryStreamExt;
use lancedb::index::scalar::{FtsIndexBuilder, FullTextSearchQuery};
use lancedb::index::Index;
use lancedb::query::{ExecutableQuery, QueryBase, Select};
use lancedb::table::Table;
use lancedb::Connection;
use std::collections::HashSet;
use std::sync::Arc;

use crate::embedder::EMBEDDING_DIM;
use trevec_core::model::CodeNode;

const TABLE_NAME: &str = "code_nodes";

/// Idempotently ensure the FTS index exists on bm25_text.
/// LanceDB FTS indexes are NOT auto-updated on merge-insert — they must be
/// explicitly rebuilt after every write.
async fn ensure_fts_index(table: &Table) -> Result<()> {
    match table
        .create_index(&["bm25_text"], Index::FTS(FtsIndexBuilder::default()))
        .execute()
        .await
    {
        Ok(_) => Ok(()),
        Err(err) => {
            let err_text = err.to_string();
            // "already exists" is fine — index is present
            if err_text.contains("already exists") {
                Ok(())
            } else {
                Err(err).context("Failed to create FTS index on code_nodes")
            }
        }
    }
}

/// LanceDB-backed store for CodeNodes with FTS + vector search.
pub struct Store {
    connection: Connection,
    table: Option<Table>,
}

impl Store {
    /// Open or create a store at the given directory.
    pub async fn open(data_dir: &str) -> Result<Self> {
        let connection = lancedb::connect(data_dir)
            .execute()
            .await
            .context("Failed to connect to LanceDB")?;

        // Check if table already exists
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
                .context("Failed to open existing table")?;

            // Ensure FTS index exists (may have been lost or never created)
            if let Err(err) = ensure_fts_index(&table).await {
                tracing::warn!("Failed to ensure code_nodes FTS index on open: {err}");
            }

            Some(table)
        } else {
            None
        };

        Ok(Self { connection, table })
    }

    /// Upsert nodes into the store, creating the table and indexes if needed.
    pub async fn upsert_nodes(&mut self, nodes: &[CodeNode]) -> Result<()> {
        if nodes.is_empty() {
            return Ok(());
        }

        let (batch, schema) = nodes_to_record_batch(nodes)?;

        if let Some(table) = &self.table {
            // Use merge_insert for upsert by node ID
            let reader: Box<dyn arrow_array::RecordBatchReader + Send> =
                Box::new(RecordBatchIterator::new(vec![Ok(batch)], schema));
            let mut merge = table.merge_insert(&["id"]);
            merge.when_matched_update_all(None);
            merge.when_not_matched_insert_all();
            merge
                .execute(reader)
                .await
                .context("Failed to upsert nodes")?;

            // Rebuild FTS index after merge-insert (LanceDB doesn't auto-update it)
            ensure_fts_index(table).await?;
        } else {
            // Create the table with the first batch
            let reader: Box<dyn arrow_array::RecordBatchReader + Send> =
                Box::new(RecordBatchIterator::new(vec![Ok(batch)], schema));
            let table = self
                .connection
                .create_table(TABLE_NAME, reader)
                .execute()
                .await
                .context("Failed to create table")?;

            // Create FTS index on bm25_text
            ensure_fts_index(&table).await?;

            self.table = Some(table);
        }

        Ok(())
    }

    /// Search by BM25 full-text search.
    pub async fn search_fts(&self, query: &str, limit: usize) -> Result<Vec<SearchResult>> {
        let Some(table) = &self.table else {
            return Ok(vec![]);
        };

        let processed = preprocess_fts_query(query);
        if processed.is_empty() {
            return Ok(vec![]);
        }

        let fts_query = FullTextSearchQuery::new(processed.clone());

        let batches: Vec<RecordBatch> = match table
            .query()
            .full_text_search(fts_query)
            .select(Select::Columns(vec!["id".to_string()]))
            .limit(limit)
            .execute()
            .await
        {
            Ok(results) => results
                .try_collect()
                .await
                .context("Failed to collect FTS results")?,
            Err(first_err) => {
                // Auto-rebuild FTS index and retry once
                tracing::warn!(
                    "FTS search failed, rebuilding index and retrying: {first_err}"
                );
                if let Err(index_err) = ensure_fts_index(table).await {
                    tracing::warn!(
                        "Failed to auto-create code_nodes FTS index after search error: {index_err}"
                    );
                }

                let retry_query = FullTextSearchQuery::new(processed);
                table
                    .query()
                    .full_text_search(retry_query)
                    .select(Select::Columns(vec!["id".to_string()]))
                    .limit(limit)
                    .execute()
                    .await
                    .context("FTS search failed after index rebuild")?
                    .try_collect()
                    .await
                    .context("Failed to collect FTS results after rebuild")?
            }
        };

        let mut search_results = Vec::new();
        for batch in &batches {
            let Some(ids) = batch
                .column_by_name("id")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            else {
                tracing::warn!(
                    "FTS result batch missing 'id' column or downcast failed (columns: {:?})",
                    batch.schema().fields().iter().map(|f| f.name()).collect::<Vec<_>>()
                );
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
        let Some(table) = &self.table else {
            return Ok(vec![]);
        };

        let results = table
            .vector_search(query_vec)
            .context("Failed to build vector search")?
            .column("symbol_vec")
            .limit(limit)
            .execute()
            .await
            .context("Vector search failed")?;

        let batches: Vec<RecordBatch> = results
            .try_collect()
            .await
            .context("Failed to collect vector results")?;

        let mut search_results = Vec::new();
        for batch in &batches {
            let Some(ids) = batch
                .column_by_name("id")
                .and_then(|c| c.as_any().downcast_ref::<StringArray>())
            else {
                tracing::warn!(
                    "Vector result batch missing 'id' column or downcast failed (columns: {:?})",
                    batch.schema().fields().iter().map(|f| f.name()).collect::<Vec<_>>()
                );
                continue;
            };

            let distances = batch
                .column_by_name("_distance")
                .and_then(|c| c.as_any().downcast_ref::<Float32Array>());

            if distances.is_none() {
                tracing::warn!("Vector result batch missing '_distance' column; scores will be 0.0");
            }

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

    /// Get the total number of nodes in the store.
    pub async fn count(&self) -> Result<usize> {
        let Some(table) = &self.table else {
            return Ok(0);
        };
        table
            .count_rows(None)
            .await
            .context("Failed to count rows")
    }

    /// Delete nodes by their IDs.
    pub async fn delete_nodes(&self, node_ids: &[String]) -> Result<()> {
        let Some(table) = &self.table else {
            return Ok(());
        };
        if node_ids.is_empty() {
            return Ok(());
        }

        // Build a SQL-style IN predicate for deletion
        let escaped: Vec<String> = node_ids
            .iter()
            .map(|id| format!("'{}'", id.replace('\'', "''")))
            .collect();
        let predicate = format!("id IN ({})", escaped.join(", "));

        table
            .delete(&predicate)
            .await
            .context("Failed to delete nodes")?;

        Ok(())
    }

    /// Delete all data (drop and recreate).
    pub async fn clear(&mut self) -> Result<()> {
        if self.table.is_some() {
            self.connection
                .drop_table(TABLE_NAME, &[])
                .await
                .context("Failed to drop table")?;
            self.table = None;
        }
        Ok(())
    }
}

/// A search result with node ID and score.
#[derive(Debug, Clone)]
pub struct SearchResult {
    pub node_id: String,
    pub score: f32,
    pub rank: usize,
}

/// Stop words to strip from FTS queries.
const STOP_WORDS: &[&str] = &[
    "a", "an", "the", "is", "are", "was", "were", "be", "been", "being",
    "how", "what", "where", "when", "why", "who", "which", "that", "this",
    "do", "does", "did", "have", "has", "had", "will", "would", "could",
    "should", "can", "may", "might", "shall", "to", "of", "in", "for",
    "on", "with", "at", "by", "from", "it", "its", "and", "or", "not",
    "no", "but", "if", "then", "so", "as", "about", "into", "through",
    "just", "also", "than", "very", "too",
];

/// Preprocess a user query for FTS: strip stop words, dedupe, join with OR.
pub fn preprocess_fts_query(query: &str) -> String {
    let mut seen = HashSet::new();
    let mut terms: Vec<String> = Vec::new();

    for term in query
        .split(|c: char| !c.is_alphanumeric())
        .filter(|part| !part.is_empty())
        .map(|part| part.to_lowercase())
    {
        if STOP_WORDS.contains(&term.as_str()) {
            continue;
        }
        if seen.insert(term.clone()) {
            terms.push(term);
        }
    }

    if terms.is_empty() {
        // Fall back to non-empty normalized terms (including stop words)
        let originals: Vec<String> = query
            .split(|c: char| !c.is_alphanumeric())
            .filter(|part| !part.is_empty())
            .map(|part| part.to_lowercase())
            .collect();
        return originals.join(" OR ");
    }

    terms.join(" OR ")
}

/// Convert a slice of CodeNodes into an Arrow RecordBatch.
fn nodes_to_record_batch(nodes: &[CodeNode]) -> Result<(RecordBatch, Arc<Schema>)> {
    let dim = EMBEDDING_DIM as i32;

    let schema = Arc::new(Schema::new(vec![
        Field::new("id", DataType::Utf8, false),
        Field::new("kind", DataType::Utf8, false),
        Field::new("file_path", DataType::Utf8, false),
        Field::new("name", DataType::Utf8, false),
        Field::new("signature", DataType::Utf8, false),
        Field::new("doc_comment", DataType::Utf8, true),
        Field::new("bm25_text", DataType::Utf8, false),
        Field::new("ast_hash", DataType::Utf8, false),
        Field::new("start_line", DataType::Utf8, false),
        Field::new("end_line", DataType::Utf8, false),
        Field::new("start_byte", DataType::Utf8, false),
        Field::new("end_byte", DataType::Utf8, false),
        Field::new(
            "symbol_vec",
            DataType::FixedSizeList(
                Arc::new(Field::new("item", DataType::Float32, true)),
                dim,
            ),
            true,
        ),
    ]));

    let ids: Vec<&str> = nodes.iter().map(|n| n.id.as_str()).collect();
    let kinds: Vec<String> = nodes.iter().map(|n| n.kind.to_string()).collect();
    let file_paths: Vec<&str> = nodes.iter().map(|n| n.file_path.as_str()).collect();
    let names: Vec<&str> = nodes.iter().map(|n| n.name.as_str()).collect();
    let signatures: Vec<&str> = nodes.iter().map(|n| n.signature.as_str()).collect();
    let doc_comments: Vec<Option<&str>> = nodes.iter().map(|n| n.doc_comment.as_deref()).collect();
    let bm25_texts: Vec<&str> = nodes.iter().map(|n| n.bm25_text.as_str()).collect();
    let ast_hashes: Vec<&str> = nodes.iter().map(|n| n.ast_hash.as_str()).collect();
    let start_lines: Vec<String> = nodes.iter().map(|n| n.span.start_line.to_string()).collect();
    let end_lines: Vec<String> = nodes.iter().map(|n| n.span.end_line.to_string()).collect();
    let start_bytes: Vec<String> = nodes.iter().map(|n| n.span.start_byte.to_string()).collect();
    let end_bytes: Vec<String> = nodes.iter().map(|n| n.span.end_byte.to_string()).collect();

    // Build the embedding vectors using from_iter_primitive
    let zero_vec: Vec<Option<f32>> = vec![Some(0.0f32); EMBEDDING_DIM];
    let embeddings: Vec<Option<Vec<Option<f32>>>> = nodes
        .iter()
        .map(|n| {
            let vec_data: Vec<Option<f32>> = n
                .symbol_vec
                .as_ref()
                .map(|v| v.iter().map(|&x| Some(x)).collect())
                .unwrap_or_else(|| zero_vec.clone());
            Some(vec_data)
        })
        .collect();

    let vec_array = FixedSizeListArray::from_iter_primitive::<Float32Type, _, _>(embeddings, dim);

    let kinds_refs: Vec<&str> = kinds.iter().map(|s| s.as_str()).collect();
    let start_lines_refs: Vec<&str> = start_lines.iter().map(|s| s.as_str()).collect();
    let end_lines_refs: Vec<&str> = end_lines.iter().map(|s| s.as_str()).collect();
    let start_bytes_refs: Vec<&str> = start_bytes.iter().map(|s| s.as_str()).collect();
    let end_bytes_refs: Vec<&str> = end_bytes.iter().map(|s| s.as_str()).collect();

    let batch = RecordBatch::try_new(
        schema.clone(),
        vec![
            Arc::new(StringArray::from(ids)),
            Arc::new(StringArray::from(kinds_refs)),
            Arc::new(StringArray::from(file_paths)),
            Arc::new(StringArray::from(names)),
            Arc::new(StringArray::from(signatures)),
            Arc::new(StringArray::from(doc_comments)),
            Arc::new(StringArray::from(bm25_texts)),
            Arc::new(StringArray::from(ast_hashes)),
            Arc::new(StringArray::from(start_lines_refs)),
            Arc::new(StringArray::from(end_lines_refs)),
            Arc::new(StringArray::from(start_bytes_refs)),
            Arc::new(StringArray::from(end_bytes_refs)),
            Arc::new(vec_array),
        ],
    )
    .context("Failed to create RecordBatch")?;

    Ok((batch, schema))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_preprocess_fts_query_single_word() {
        assert_eq!(preprocess_fts_query("authentication"), "authentication");
    }

    #[test]
    fn test_preprocess_fts_query_multi_word() {
        let result = preprocess_fts_query("how does authentication work");
        assert_eq!(result, "authentication OR work");
    }

    #[test]
    fn test_preprocess_fts_query_underscore_tokens() {
        let result = preprocess_fts_query("TREVEC_TEST_MARKER_V1");
        assert_eq!(result, "trevec OR test OR marker OR v1");
    }

    #[test]
    fn test_preprocess_fts_query_truncates_long_input() {
        // Simulate a long bug report: title + noise body
        let title = "TransfoXLLMHead doesn't shift labels";
        let body = " bug ".repeat(200); // 1000 chars of noise
        let long_query = format!("{title}{body}");
        let result = preprocess_fts_query(&long_query);
        // Should contain the title terms but NOT hundreds of "bug" repetitions
        assert!(result.contains("transfoxllmhead"));
        assert!(result.contains("shift"));
        assert!(result.contains("labels"));
        // Term count should be capped
        let term_count = result.matches(" OR ").count() + 1;
        assert!(term_count <= 30, "Too many terms: {term_count}");
    }

    #[test]
    fn test_preprocess_fts_query_term_limit() {
        // Even within char limit, enforce term cap
        let many_words: String = (0..100).map(|i| format!("word{i}")).collect::<Vec<_>>().join(" ");
        let short = &many_words[..280]; // within char limit
        let result = preprocess_fts_query(short);
        let term_count = result.matches(" OR ").count() + 1;
        assert!(term_count <= 30, "Too many terms: {term_count}");
    }

    #[test]
    fn test_preprocess_fts_query_all_stop_words() {
        // When all words are stop words, fall back to OR-joining them
        let result = preprocess_fts_query("how does the");
        assert_eq!(result, "how OR does OR the");
    }

    #[test]
    fn test_preprocess_fts_query_preserves_code_terms() {
        let result = preprocess_fts_query("verify password hash");
        assert_eq!(result, "verify OR password OR hash");
    }
}
