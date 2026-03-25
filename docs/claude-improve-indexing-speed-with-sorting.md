# Improve Indexing Speed with Sorted Batching

## Problem

Embedding computation dominates indexing time. On Django (35,247 nodes), embeddings take 233 seconds out of 241 total — **96.8% of the time**. Parsing 3,004 files and building 853K edges takes under 2 seconds.

The root cause is **padding waste** in ONNX batch inference. ONNX pads every text in a batch to the longest text's token count. With unsorted batches of 256, a single long text forces 255 shorter texts to be padded, wasting compute on zeros.

## Analysis (Django, 35,247 nodes)

Embedding text length distribution:
- P10: 27 chars
- P25: 41 chars
- P50: 56 chars
- P75: 75 chars
- P90: 99 chars
- P99: 185 chars
- Max: 859 chars

Padding waste simulation:

| Strategy | Padded total chars | Overhead vs actual | Savings vs unsorted |
|----------|-------------------:|-------------------:|--------------------:|
| Unsorted batches (current) | 9,226,680 | 4.2x | — |
| Sorted batches | 2,315,573 | 1.1x | **75%** |

Sorting by text length before batching reduces wasted ONNX compute by 75%. Applied to Django's 233s embedding time, this projects to ~58 seconds — a **4x speedup**.

## Why This is Safe

The embedding model (BGE-small-en-v1.5) processes each text independently within a batch. There is no cross-attention between texts. The attention mask zeros out padded positions, so the output vector for a given text is **mathematically identical** regardless of what other texts are in the same batch.

Confirmed by reading fastembed 5.11.0 source (`text_embedding/impl.rs`):
- `encode_batch` pads all texts to the longest in the batch
- `attention_mask` is passed to ONNX, masking padded tokens
- Output vectors are extracted per-text from the batch output

**Zero impact on:**
- Retrieval quality (identical vectors)
- Query latency (same LanceDB index)
- SWE-bench benchmark scores (same retrieval output)
- Any downstream behavior

## Implementation

### Change: `crates/trevec-index/src/ingest.rs`

Current code (around line 424-436):
```rust
// 4. Compute embeddings (only for new/changed nodes)
let texts: Vec<String> = new_nodes.iter().map(|n| n.embedding_text()).collect();
let embeddings = embedder.embed_batch(&texts)?;
for (node, embedding) in new_nodes.iter_mut().zip(embeddings.into_iter()) {
    node.symbol_vec = Some(embedding);
}
```

New code:
```rust
// 4. Compute embeddings (only for new/changed nodes)
// Sort texts by length before batching to minimize ONNX padding waste.
// ONNX pads all texts in a batch to the longest text's token count.
// Unsorted batches waste ~4x compute; sorted batches reduce waste to ~1.1x.
let mut indexed_texts: Vec<(usize, String)> = new_nodes
    .iter()
    .enumerate()
    .map(|(i, n)| (i, n.embedding_text()))
    .collect();
indexed_texts.sort_by_key(|(_, text)| text.len());

let sorted_texts: Vec<String> = indexed_texts.iter().map(|(_, t)| t.clone()).collect();
let sorted_embeddings = embedder.embed_batch(&sorted_texts)?;

// Map embeddings back to original node order
for ((orig_idx, _), embedding) in indexed_texts.into_iter().zip(sorted_embeddings.into_iter()) {
    new_nodes[orig_idx].symbol_vec = Some(embedding);
}
```

### Files modified

1. `crates/trevec-index/src/ingest.rs` — sort texts by length before `embed_batch()`

No other files need changes. No config changes, no dependency changes, no API changes.

## Verification

1. `cargo test` — all existing tests pass (embeddings are only tested in `#[ignore]` tests that require model download)
2. Re-run SWE-bench benchmark on a small subset (5-10 instances) and diff the `predictions.jsonl` — should be byte-identical since vectors don't change
3. Time `trevec index` on Django before and after — expect ~4x speedup on the embedding phase

## Projected Impact

| Repo size | Current embed time | After sorting | Total index time |
|-----------|-------------------:|--------------:|-----------------:|
| 10K LoC (~3K nodes) | ~20s | ~5s | ~7s |
| 50K LoC (~15K nodes) | ~100s | ~25s | ~28s |
| 100K LoC (~35K nodes) | ~233s | ~58s | ~62s |

## Future Optimizations (not in this change)

These are separate improvements to consider later:

- **Deferred embeddings**: For large repos, compute embeddings in background while BM25 + graph are immediately usable
- **Batch size tuning**: Test 64/128/512 batch sizes on CPU vs the default 256 — smaller batches may be more cache-friendly
- **Skip trivial nodes**: Only 98/35,247 Django nodes have <15 char embedding text, so this has minimal impact
- **Throttled mode for large repos**: Quick 10s burst then continue at reduced CPU priority
