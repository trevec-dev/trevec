use std::collections::{HashMap, HashSet};
use std::path::Path;

use trevec_core::model::CodeNode;
use trevec_index::store::SearchResult;

/// Merge BM25 and vector search results using Reciprocal Rank Fusion.
/// score(d) = fts_weight * 1/(k + fts_rank) + vec_weight * 1/(k + vec_rank)
///
/// Weights are determined by query characteristics:
/// - Short identifier-like queries: FTS weight 1.2, vector weight 0.8
///   (keywords match code symbols directly)
/// - Long natural-language queries: FTS weight 0.6, vector weight 1.4
///   (semantic similarity captures intent better than keyword scatter)
/// - Default: equal weights (1.0, 1.0)
pub fn rrf_merge(
    fts_results: &[SearchResult],
    vector_results: &[SearchResult],
    k: usize,
) -> Vec<RankedResult> {
    rrf_merge_weighted(fts_results, vector_results, k, 1.0, 1.0)
}

/// RRF merge with explicit weights for FTS and vector contributions.
pub fn rrf_merge_weighted(
    fts_results: &[SearchResult],
    vector_results: &[SearchResult],
    k: usize,
    fts_weight: f64,
    vec_weight: f64,
) -> Vec<RankedResult> {
    let mut scores: HashMap<String, f64> = HashMap::new();

    // Add FTS ranks (weighted)
    for (rank, result) in fts_results.iter().enumerate() {
        let rrf_score = fts_weight / (k as f64 + (rank + 1) as f64);
        *scores.entry(result.node_id.clone()).or_default() += rrf_score;
    }

    // Add vector ranks (weighted)
    for (rank, result) in vector_results.iter().enumerate() {
        let rrf_score = vec_weight / (k as f64 + (rank + 1) as f64);
        *scores.entry(result.node_id.clone()).or_default() += rrf_score;
    }

    // Sort by combined score descending
    let mut ranked: Vec<RankedResult> = scores
        .into_iter()
        .map(|(node_id, score)| RankedResult { node_id, score, rank: 0 })
        .collect();

    ranked.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.node_id.cmp(&b.node_id))
    });

    // Assign final ranks
    for (i, result) in ranked.iter_mut().enumerate() {
        result.rank = i + 1;
    }

    ranked
}

/// Choose RRF weights based on query characteristics.
/// Returns (fts_weight, vec_weight).
///
/// Vector embeddings capture semantic intent and are more robust to vocabulary
/// mismatch (bug report talks about symptoms, gold file has the fix).
/// BM25 excels at exact identifier/symbol lookups.
pub fn query_rrf_weights(query: &str) -> (f64, f64) {
    let trimmed = query.trim();
    let word_count = trimmed.split_whitespace().count();

    if word_count <= 3 {
        // Short query — likely an identifier or symbol lookup.
        // BM25 matches exact keywords well, but still give vector a fair share.
        (1.0, 1.0)
    } else if word_count >= 15 {
        // Long query — bug report or description.
        // Semantic similarity captures intent much better than keyword scatter.
        (0.4, 1.6)
    } else {
        // Medium query — favor vector slightly.
        (0.7, 1.3)
    }
}

/// A result after RRF merge with combined score.
#[derive(Debug, Clone)]
pub struct RankedResult {
    pub node_id: String,
    pub score: f64,
    pub rank: usize,
}

/// Heuristic gate to avoid scanning all nodes for natural-language queries.
/// Literal boost is reserved for identifier-like queries where exact text
/// matching is useful (e.g., snake_case markers, paths, IDs).
fn is_literal_like_query(query: &str) -> bool {
    let trimmed = query.trim();
    if trimmed.is_empty() {
        return false;
    }

    let token_count = trimmed.split_whitespace().count();
    if token_count > 1 {
        return false;
    }

    trimmed.len() >= 8
        && (trimmed.chars().any(|c| !c.is_alphanumeric())
            || trimmed.chars().any(|c| c.is_ascii_digit())
            || (trimmed.chars().any(|c| c.is_uppercase())
                && trimmed.chars().any(|c| c.is_lowercase())))
}

/// Score a node against a literal query string.
/// Returns `Some(score)` if any field matches, `None` otherwise.
pub fn literal_match_score(node: &CodeNode, query_lower: &str) -> Option<f64> {
    if query_lower.is_empty() {
        return None;
    }

    let mut score = 0.0_f64;

    if node.name.eq_ignore_ascii_case(query_lower) {
        score = score.max(0.06);
    }
    if node.signature.to_lowercase().contains(query_lower) {
        score = score.max(0.05);
    }
    if let Some(doc) = &node.doc_comment {
        if doc.to_lowercase().contains(query_lower) {
            score = score.max(0.04);
        }
    }
    if node.bm25_text.to_lowercase().contains(query_lower) {
        score = score.max(0.035);
    }
    if node.file_path.to_lowercase().contains(query_lower) {
        score = score.max(0.025);
    }
    if node.name.to_lowercase().contains(query_lower) {
        score = score.max(0.02);
    }

    (score > 0.0).then_some(score)
}

/// Boost RRF results with literal string matching against node fields.
/// Nodes that match literally but weren't in the RRF results are injected.
/// Results are re-sorted and re-ranked after boosting.
pub fn apply_literal_boost(
    merged: &mut Vec<RankedResult>,
    nodes_map: &HashMap<String, CodeNode>,
    query_text: &str,
) {
    if !is_literal_like_query(query_text) {
        return;
    }

    let query_lower = query_text.trim().to_lowercase();
    if query_lower.is_empty() {
        return;
    }

    let mut literal_scores: HashMap<String, f64> = nodes_map
        .iter()
        .filter_map(|(id, node)| {
            literal_match_score(node, &query_lower).map(|score| (id.clone(), score))
        })
        .collect();

    if literal_scores.is_empty() {
        return;
    }

    for result in merged.iter_mut() {
        if let Some(boost) = literal_scores.remove(&result.node_id) {
            result.score += boost;
        }
    }

    merged.extend(
        literal_scores
            .into_iter()
            .map(|(node_id, score)| RankedResult {
                node_id,
                score,
                rank: 0,
            }),
    );

    merged.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.node_id.cmp(&b.node_id))
    });
    for (i, result) in merged.iter_mut().enumerate() {
        result.rank = i + 1;
    }
}

/// Check if a file path looks like a test file based on built-in patterns.
pub fn is_test_file(path: &str) -> bool {
    let normalized = path.replace('\\', "/");
    let lower = normalized.to_lowercase();

    // Directory-based: any file under a test directory
    let test_dirs = ["/tests/", "/test/", "/__tests__/", "/spec/", "/__spec__/"];
    for dir in &test_dirs {
        if lower.contains(dir) {
            return true;
        }
    }
    // Top-level tests/ or test/ (path starts with it)
    if lower.starts_with("tests/") || lower.starts_with("test/") {
        return true;
    }

    // Filename-based patterns
    let file_name = Path::new(&normalized)
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or("");
    let file_lower = file_name.to_lowercase();

    // Python: test_*.py, *_test.py, conftest.py
    if file_lower.ends_with(".py") {
        let stem = &file_lower[..file_lower.len() - 3];
        if stem.starts_with("test_") || stem.ends_with("_test") || stem == "conftest" {
            return true;
        }
    }

    // JS/TS: *.test.js, *.spec.js, *.test.ts, *.spec.ts, *.test.tsx, *.spec.tsx,
    //         *.test.jsx, *.spec.jsx
    let js_test_suffixes = [
        ".test.js", ".spec.js", ".test.ts", ".spec.ts",
        ".test.tsx", ".spec.tsx", ".test.jsx", ".spec.jsx",
        ".test.mjs", ".spec.mjs",
    ];
    for suffix in &js_test_suffixes {
        if file_lower.ends_with(suffix) {
            return true;
        }
    }

    // Rust: *_test.rs
    if file_lower.ends_with("_test.rs") {
        return true;
    }

    // Go: *_test.go
    if file_lower.ends_with("_test.go") {
        return true;
    }

    // Java/Kotlin: *Test.java, *Tests.java, *Test.kt, *Tests.kt
    if file_name.ends_with("Test.java")
        || file_name.ends_with("Tests.java")
        || file_name.ends_with("Test.kt")
        || file_name.ends_with("Tests.kt")
    {
        return true;
    }

    false
}

/// Check if a file path looks like a non-code/documentation file.
/// These files (CHANGELOG, README, etc.) match bug descriptions via keyword
/// overlap but almost never contain the actual fix.
pub fn is_noncode_file(path: &str) -> bool {
    let normalized = path.replace('\\', "/");
    let lower = normalized.to_lowercase();

    let file_name = Path::new(&normalized)
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or("");
    let file_lower = file_name.to_lowercase();

    // Filename prefix patterns (case-insensitive)
    let prefix_patterns = [
        "changelog", "readme", "contributing", "authors",
        "license", "copying", "news", "history",
    ];
    for prefix in &prefix_patterns {
        if file_lower.starts_with(prefix) {
            return true;
        }
    }

    // Extension-based patterns
    let noncode_extensions = [".md", ".rst", ".txt"];
    for ext in &noncode_extensions {
        if file_lower.ends_with(ext) {
            // Exception: docs inside source dirs that might be docstrings/specs
            // are still penalized — they rarely contain fixes.
            return true;
        }
    }

    // Top-level docs directory
    if lower.starts_with("docs/") || lower.contains("/docs/") {
        // Only penalize markdown/rst in docs, not code examples
        for ext in &[".md", ".rst", ".txt", ".html"] {
            if file_lower.ends_with(ext) {
                return true;
            }
        }
    }

    false
}

/// Check if a file path matches any user-supplied penalty path patterns (substring match).
pub fn matches_penalty_paths(path: &str, penalty_paths: &[String]) -> bool {
    if penalty_paths.is_empty() {
        return false;
    }
    let normalized = path.replace('\\', "/");
    penalty_paths.iter().any(|p| normalized.contains(p.as_str()))
}

/// Apply a score penalty to test files and custom penalty paths.
/// If penalty <= 0.0, this is a no-op.
/// Scores are multiplied by (1.0 - penalty), then re-sorted and re-ranked.
pub fn apply_test_file_penalty(
    merged: &mut Vec<RankedResult>,
    nodes_map: &HashMap<String, CodeNode>,
    penalty: f64,
    penalty_paths: &[String],
) {
    if penalty <= 0.0 {
        return;
    }
    let factor = 1.0 - penalty.min(1.0);

    let mut changed = false;
    for result in merged.iter_mut() {
        if let Some(node) = nodes_map.get(&result.node_id) {
            if is_test_file(&node.file_path) || matches_penalty_paths(&node.file_path, penalty_paths) {
                result.score *= factor;
                changed = true;
            }
        }
    }

    if changed {
        resort_and_rerank(merged);
    }
}

/// Apply a score penalty to non-code/documentation files (CHANGELOG, README, .md, etc.).
/// These files match bug descriptions via keyword overlap but almost never contain fixes.
/// If penalty <= 0.0, this is a no-op.
pub fn apply_noncode_penalty(
    merged: &mut Vec<RankedResult>,
    nodes_map: &HashMap<String, CodeNode>,
    penalty: f64,
) {
    if penalty <= 0.0 {
        return;
    }
    let factor = 1.0 - penalty.min(1.0);

    let mut changed = false;
    for result in merged.iter_mut() {
        if let Some(node) = nodes_map.get(&result.node_id) {
            if is_noncode_file(&node.file_path) {
                result.score *= factor;
                changed = true;
            }
        }
    }

    if changed {
        resort_and_rerank(merged);
    }
}

/// Check if a file should be excluded from search results entirely.
/// These are non-code files that match bug descriptions via keyword overlap
/// but almost never contain the actual fix.
pub fn should_exclude_from_search(path: &str) -> bool {
    let normalized = path.replace('\\', "/");
    let lower = normalized.to_lowercase();

    let file_name = Path::new(&normalized)
        .file_name()
        .and_then(|f| f.to_str())
        .unwrap_or("");
    let file_lower = file_name.to_lowercase();

    // Extension-based exclusions
    if file_lower.ends_with(".md") || file_lower.ends_with(".mdx") {
        return true;
    }
    if file_lower.ends_with(".lock") {
        return true;
    }
    if file_lower.ends_with(".txt") || file_lower.ends_with(".rst") {
        return true;
    }
    if file_lower.ends_with(".yaml") || file_lower.ends_with(".yml") {
        return true;
    }

    // JSON files — exclude all EXCEPT a few useful config files
    if file_lower.ends_with(".json") {
        let keep = ["package.json", "tsconfig.json", "trevec.json"];
        if !keep.iter().any(|k| file_lower == *k) {
            return true;
        }
    }

    // Documentation directories
    if lower.starts_with("site/") || lower.contains("/site/") {
        return true;
    }
    if lower.starts_with("docs/") || lower.contains("/docs/") {
        return true;
    }

    // Well-known non-code files by prefix
    let prefix_patterns = [
        "changelog", "readme", "contributing", "authors",
        "license", "copying", "news", "history",
    ];
    for prefix in &prefix_patterns {
        if file_lower.starts_with(prefix) {
            return true;
        }
    }

    false
}

/// Remove non-code files from search results, then re-rank.
/// Uses the extension/prefix-based is_noncode_file check (not the aggressive should_exclude_from_search).
pub fn filter_noncode_files(
    merged: &mut Vec<RankedResult>,
    nodes_map: &HashMap<String, CodeNode>,
) {
    let before = merged.len();
    merged.retain(|r| {
        nodes_map
            .get(&r.node_id)
            .map(|n| !is_noncode_file(&n.file_path))
            .unwrap_or(true)
    });
    if merged.len() < before {
        for (i, result) in merged.iter_mut().enumerate() {
            result.rank = i + 1;
        }
    }
}

/// Check if a file is a test fixture / expected output / benchmark file.
/// These are even less likely to be gold files than regular tests.
pub fn is_test_fixture(path: &str) -> bool {
    let lower = path.replace('\\', "/").to_lowercase();
    // Test sample/expected output directories
    if lower.contains("/samples/") && (lower.contains("expected") || lower.contains("/output")) {
        return true;
    }
    // Bundled test output files
    if lower.contains("/test") && lower.ends_with("-bundle.js") {
        return true;
    }
    // Fixture directories
    if lower.contains("/fixtures/") || lower.contains("/__fixtures__/") {
        return true;
    }
    // Snapshot files
    if lower.contains("__snapshots__") || lower.ends_with(".snap") {
        return true;
    }
    // Benchmark directories (these match keywords but never contain fixes)
    if lower.contains("/benchmarks/") || lower.contains("/benchmark/") || lower.starts_with("benchmarks/") {
        return true;
    }
    // ASV benchmarks (Python)
    if lower.starts_with("asv_bench/") || lower.contains("/asv_bench/") {
        return true;
    }
    // Test example / auto-generated test data directories
    if lower.contains("/test-examples/") || lower.contains("/test_examples/") {
        return true;
    }
    false
}

/// Apply an extra-harsh penalty to test fixture files (expected outputs, snapshots, etc.).
/// These are almost never the gold file. Keeps only 5% of score.
pub fn apply_test_fixture_penalty(
    merged: &mut Vec<RankedResult>,
    nodes_map: &HashMap<String, CodeNode>,
) {
    let factor = 0.05; // keep 5% of score
    let mut changed = false;
    for result in merged.iter_mut() {
        if let Some(node) = nodes_map.get(&result.node_id) {
            if is_test_fixture(&node.file_path) {
                result.score *= factor;
                changed = true;
            }
        }
    }
    if changed {
        resort_and_rerank(merged);
    }
}

/// Derive candidate source file paths from a test file path.
/// Strips common test patterns to find the likely production source counterpart.
pub fn derive_source_paths(test_path: &str) -> Vec<String> {
    let normalized = test_path.replace('\\', "/");
    let mut candidates = Vec::new();

    let file_name = match Path::new(&normalized).file_name().and_then(|f| f.to_str()) {
        Some(f) => f.to_string(),
        None => return candidates,
    };

    let dir = match Path::new(&normalized).parent().and_then(|p| p.to_str()) {
        Some(d) => d.to_string(),
        None => String::new(),
    };

    // JS/TS: foo.test.ts -> foo.ts, foo.spec.ts -> foo.ts
    let js_test_suffixes = [
        ".test.js", ".spec.js", ".test.ts", ".spec.ts",
        ".test.tsx", ".spec.tsx", ".test.jsx", ".spec.jsx",
        ".test.mjs", ".spec.mjs",
    ];
    for suffix in &js_test_suffixes {
        if file_name.ends_with(suffix) {
            let base = &file_name[..file_name.len() - suffix.len()];
            let ext = &suffix[suffix.find('.').unwrap() + 5..]; // skip ".test" or ".spec"
            let source_name = format!("{base}{ext}");

            // Same directory
            if dir.is_empty() {
                candidates.push(source_name.clone());
            } else {
                candidates.push(format!("{dir}/{source_name}"));
            }

            // __tests__ -> parent directory
            if dir.contains("__tests__") {
                let parent = dir.replace("__tests__", "").replace("//", "/");
                let parent = parent.trim_end_matches('/');
                if !parent.is_empty() {
                    candidates.push(format!("{parent}/{source_name}"));
                } else {
                    candidates.push(source_name.clone());
                }
            }
            break;
        }
    }

    // Python: test_foo.py -> foo.py, src/foo.py, lib/foo.py
    if file_name.ends_with(".py") {
        let stem = &file_name[..file_name.len() - 3];
        if let Some(base) = stem.strip_prefix("test_") {
            let source_name = format!("{base}.py");
            // Try various source directories
            candidates.push(source_name.clone());
            candidates.push(format!("src/{source_name}"));
            candidates.push(format!("lib/{source_name}"));
            // If test is in tests/subdir/test_foo.py, try subdir/foo.py and src/subdir/foo.py
            if dir.starts_with("tests/") || dir.starts_with("test/") {
                let sub = dir.split('/').skip(1).collect::<Vec<_>>().join("/");
                if !sub.is_empty() {
                    candidates.push(format!("{sub}/{source_name}"));
                    candidates.push(format!("src/{sub}/{source_name}"));
                }
            }
        }
        if let Some(base) = stem.strip_suffix("_test") {
            let source_name = format!("{base}.py");
            candidates.push(source_name.clone());
            candidates.push(format!("src/{source_name}"));
        }
    }

    // Go: foo_test.go -> foo.go
    if file_name.ends_with("_test.go") {
        let base = &file_name[..file_name.len() - 8];
        let source_name = format!("{base}.go");
        if dir.is_empty() {
            candidates.push(source_name);
        } else {
            candidates.push(format!("{dir}/{source_name}"));
        }
    }

    // Rust: foo_test.rs -> foo.rs
    if file_name.ends_with("_test.rs") {
        let base = &file_name[..file_name.len() - 8];
        let source_name = format!("{base}.rs");
        if dir.is_empty() {
            candidates.push(source_name);
        } else {
            candidates.push(format!("{dir}/{source_name}"));
        }
    }

    // Java/Kotlin: FooTest.java -> Foo.java, FooTests.java -> Foo.java
    for ext in [".java", ".kt"] {
        if file_name.ends_with(ext) {
            let stem = &file_name[..file_name.len() - ext.len()];
            if let Some(base) = stem.strip_suffix("Tests") {
                let source_name = format!("{base}{ext}");
                if dir.is_empty() {
                    candidates.push(source_name.clone());
                } else {
                    // Java: test dir often mirrors src dir
                    let source_dir = dir.replace("/test/", "/main/");
                    candidates.push(format!("{source_dir}/{source_name}"));
                    candidates.push(format!("{dir}/{source_name}"));
                }
            } else if let Some(base) = stem.strip_suffix("Test") {
                let source_name = format!("{base}{ext}");
                if dir.is_empty() {
                    candidates.push(source_name.clone());
                } else {
                    let source_dir = dir.replace("/test/", "/main/");
                    candidates.push(format!("{source_dir}/{source_name}"));
                    candidates.push(format!("{dir}/{source_name}"));
                }
            }
        }
    }

    candidates
}

/// Boost source files that are neighbors of test files in the results.
/// For each test file in results, derive likely source file paths
/// and boost/inject those source files.
pub fn boost_source_neighbors(
    merged: &mut Vec<RankedResult>,
    nodes_map: &HashMap<String, CodeNode>,
) {
    // Build a set of all file paths in the node map for fast lookup
    // Map: file_path -> node_ids that belong to that file
    let mut path_to_nodes: HashMap<&str, Vec<&str>> = HashMap::new();
    for (id, node) in nodes_map.iter() {
        path_to_nodes.entry(node.file_path.as_str()).or_default().push(id.as_str());
    }

    let file_paths: HashSet<&str> = path_to_nodes.keys().copied().collect();

    // For each test file in results, find source counterparts
    let mut boosts: HashMap<String, f64> = HashMap::new();
    for result in merged.iter() {
        if let Some(node) = nodes_map.get(&result.node_id) {
            if is_test_file(&node.file_path) {
                for candidate in derive_source_paths(&node.file_path) {
                    if file_paths.contains(candidate.as_str()) {
                        // Boost score = test file's original score * 1.5
                        let boost = result.score * 1.5;
                        boosts.entry(candidate).and_modify(|b| *b = b.max(boost)).or_insert(boost);
                    }
                }
            }
        }
    }

    if boosts.is_empty() {
        return;
    }

    // Track which node IDs already exist in results (owned to avoid borrow conflicts)
    let existing_ids: HashSet<String> = merged.iter().map(|r| r.node_id.clone()).collect();

    // Apply boosts to existing results
    for result in merged.iter_mut() {
        if let Some(node) = nodes_map.get(&result.node_id) {
            if let Some(boost) = boosts.get(&node.file_path) {
                result.score += boost;
            }
        }
    }

    // Inject missing source file nodes
    for (file_path, boost) in &boosts {
        if let Some(node_ids) = path_to_nodes.get(file_path.as_str()) {
            for node_id in node_ids {
                if !existing_ids.contains(*node_id) {
                    merged.push(RankedResult {
                        node_id: node_id.to_string(),
                        score: *boost,
                        rank: 0,
                    });
                }
            }
        }
    }

    resort_and_rerank(merged);
}

/// Extract file paths mentioned in the query text (from stack traces, error messages, etc.).
/// Returns a list of extracted path fragments that can be matched against node file paths.
/// Focused on Python (.py) since SWE-bench is Python-only, but also handles common extensions.
pub fn extract_file_paths_from_query(query: &str) -> Vec<String> {
    let mut paths = Vec::new();
    let mut seen = HashSet::new();

    // Pattern 1: File "path/to/file.py", line N (Python tracebacks)
    // Pattern 2: at path/to/file.py:N
    // Pattern 3: in /path/to/file.py
    // Pattern 4: bare path/to/file.ext (with at least one / and a known extension)

    // We use a simple line-by-line + token-by-token approach for robustness.
    for line in query.lines() {
        let line = line.trim();

        // Python traceback: File "django/db/models/fields/__init__.py", line 123
        if let Some(start) = line.find("File \"") {
            let after = &line[start + 6..];
            if let Some(end) = after.find('"') {
                let path = after[..end].trim();
                if looks_like_file_path(path) {
                    let normalized = normalize_extracted_path(path);
                    if !normalized.is_empty() && seen.insert(normalized.clone()) {
                        paths.push(normalized);
                    }
                }
            }
        }

        // Backtick-wrapped paths: `django/db/models/fields/__init__.py`
        let mut search_from = 0;
        while let Some(start) = line[search_from..].find('`') {
            let abs_start = search_from + start + 1;
            if abs_start >= line.len() {
                break;
            }
            if let Some(end) = line[abs_start..].find('`') {
                let token = &line[abs_start..abs_start + end];
                if looks_like_file_path(token) {
                    let normalized = normalize_extracted_path(token);
                    if !normalized.is_empty() && seen.insert(normalized.clone()) {
                        paths.push(normalized);
                    }
                }
                search_from = abs_start + end + 1;
            } else {
                break;
            }
        }

        // Scan tokens for bare paths
        for token in line.split_whitespace() {
            // Strip trailing punctuation (commas, colons, parens, quotes)
            let token = token.trim_matches(|c: char| {
                c == ',' || c == ';' || c == ')' || c == '(' || c == '\'' || c == '"' || c == '`'
            });

            // Strip line number suffix: path/to/file.py:123
            let token = if let Some(colon_pos) = token.rfind(':') {
                let after_colon = &token[colon_pos + 1..];
                if after_colon.chars().all(|c| c.is_ascii_digit()) && !after_colon.is_empty() {
                    &token[..colon_pos]
                } else {
                    token
                }
            } else {
                token
            };

            if looks_like_file_path(token) {
                let normalized = normalize_extracted_path(token);
                if !normalized.is_empty() && seen.insert(normalized.clone()) {
                    paths.push(normalized);
                }
            }
        }
    }

    paths
}

/// Check if a token looks like a file path (has a slash and a known extension).
fn looks_like_file_path(token: &str) -> bool {
    if token.len() < 4 {
        return false;
    }
    // Must contain at least one slash (path separator)
    if !token.contains('/') && !token.contains('\\') {
        return false;
    }
    // Must end with a known extension
    let lower = token.to_lowercase();
    let code_extensions = [
        ".py", ".rs", ".js", ".ts", ".jsx", ".tsx", ".go", ".java", ".kt",
        ".rb", ".c", ".cpp", ".h", ".hpp", ".cs", ".swift", ".scala",
    ];
    code_extensions.iter().any(|ext| lower.ends_with(ext))
}

/// Normalize an extracted file path: strip leading slashes, convert backslashes.
fn normalize_extracted_path(path: &str) -> String {
    let normalized = path.replace('\\', "/");
    // Strip leading / (absolute path) to get a relative path fragment
    let normalized = normalized.trim_start_matches('/');
    normalized.to_string()
}

/// Boost nodes whose file_path contains any of the extracted path fragments.
/// Applied after RRF merge to lift results that match stack traces / error paths.
pub fn apply_file_path_boost(
    merged: &mut Vec<RankedResult>,
    nodes_map: &HashMap<String, CodeNode>,
    extracted_paths: &[String],
    boost: f64,
) {
    if extracted_paths.is_empty() {
        return;
    }

    let mut changed = false;
    for result in merged.iter_mut() {
        if let Some(node) = nodes_map.get(&result.node_id) {
            let node_path = node.file_path.replace('\\', "/");
            for extracted in extracted_paths {
                if node_path.ends_with(extracted) || node_path.contains(extracted.as_str()) {
                    result.score += boost;
                    changed = true;
                    break; // Only boost once per node
                }
            }
        }
    }

    if changed {
        resort_and_rerank(merged);
    }
}

/// Boost nodes that share a directory with top-ranked results.
/// Software changes tend to be spatially local -- files in the same directory
/// as highly-ranked results are more likely to be relevant.
pub fn apply_directory_cohesion_boost(
    merged: &mut Vec<RankedResult>,
    nodes_map: &HashMap<String, CodeNode>,
    top_n: usize,
    boost: f64,
) {
    if merged.is_empty() || boost <= 0.0 {
        return;
    }

    // Collect directories from top-N results
    let mut top_dirs: HashSet<String> = HashSet::new();
    for result in merged.iter().take(top_n) {
        if let Some(node) = nodes_map.get(&result.node_id) {
            let path = node.file_path.replace('\\', "/");
            if let Some(dir) = Path::new(&path).parent().and_then(|p| p.to_str()) {
                if !dir.is_empty() {
                    top_dirs.insert(dir.to_string());
                }
            }
        }
    }

    if top_dirs.is_empty() {
        return;
    }

    // For results NOT in top-N, if their file is in one of those directories, add boost
    let mut changed = false;
    for result in merged.iter_mut().skip(top_n) {
        if let Some(node) = nodes_map.get(&result.node_id) {
            let path = node.file_path.replace('\\', "/");
            if let Some(dir) = Path::new(&path).parent().and_then(|p| p.to_str()) {
                if top_dirs.contains(dir) {
                    result.score += boost;
                    changed = true;
                }
            }
        }
    }

    if changed {
        resort_and_rerank(merged);
    }
}

/// Extract code identifiers from a query text.
/// Looks for snake_case, CamelCase, dot.notation, and backtick-wrapped tokens.
/// Returns unique identifiers suitable for a secondary BM25 search.
pub fn extract_code_identifiers(query: &str) -> Vec<String> {
    let mut identifiers = Vec::new();
    let mut seen = HashSet::new();

    for line in query.lines() {
        // Extract backtick-wrapped identifiers first
        let mut search_from = 0;
        while let Some(start) = line[search_from..].find('`') {
            let abs_start = search_from + start + 1;
            if abs_start >= line.len() {
                break;
            }
            if let Some(end) = line[abs_start..].find('`') {
                let token = line[abs_start..abs_start + end].trim();
                if !token.is_empty() && token.len() >= 2 && seen.insert(token.to_string()) {
                    identifiers.push(token.to_string());
                }
                search_from = abs_start + end + 1;
            } else {
                break;
            }
        }

        // Scan whitespace-delimited tokens
        for token in line.split_whitespace() {
            // Strip surrounding punctuation
            let token = token.trim_matches(|c: char| {
                c == ',' || c == ';' || c == ')' || c == '(' || c == '\''
                    || c == '"' || c == '`' || c == ':' || c == '.'
                    || c == '!' || c == '?'
            });

            if token.is_empty() || token.len() < 3 {
                continue;
            }

            let is_identifier = is_snake_case(token)
                || is_camel_case(token)
                || is_dot_notation(token)
                || is_path_like(token);

            if is_identifier && seen.insert(token.to_string()) {
                identifiers.push(token.to_string());
            }
        }
    }

    identifiers
}

/// Check if a token is snake_case (contains underscore with alphanumeric chars).
fn is_snake_case(token: &str) -> bool {
    token.contains('_')
        && token.chars().all(|c| c.is_alphanumeric() || c == '_')
        && token.chars().any(|c| c.is_alphabetic())
}

/// Check if a token is CamelCase (mixed case within single token, no spaces).
fn is_camel_case(token: &str) -> bool {
    if !token.chars().all(|c| c.is_alphanumeric()) {
        return false;
    }
    let has_upper = token.chars().any(|c| c.is_uppercase());
    let has_lower = token.chars().any(|c| c.is_lowercase());
    // Must start with uppercase and contain both cases
    has_upper && has_lower && token.chars().next().map_or(false, |c| c.is_uppercase())
}

/// Check if a token is dot notation (word.word.word).
fn is_dot_notation(token: &str) -> bool {
    if !token.contains('.') {
        return false;
    }
    let parts: Vec<&str> = token.split('.').collect();
    parts.len() >= 2
        && parts.iter().all(|p| {
            !p.is_empty() && p.chars().all(|c| c.is_alphanumeric() || c == '_')
        })
}

/// Check if a token looks like a path-like identifier (contains / but not a full file path).
fn is_path_like(token: &str) -> bool {
    token.contains('/')
        && !looks_like_file_path(token)
        && token.chars().all(|c| c.is_alphanumeric() || c == '/' || c == '_' || c == '-' || c == '.')
        && token.len() >= 4
}

fn resort_and_rerank(merged: &mut Vec<RankedResult>) {
    merged.sort_by(|a, b| {
        b.score
            .partial_cmp(&a.score)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| a.node_id.cmp(&b.node_id))
    });
    for (i, result) in merged.iter_mut().enumerate() {
        result.rank = i + 1;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_result(id: &str, rank: usize) -> SearchResult {
        SearchResult {
            node_id: id.to_string(),
            score: 0.0,
            rank,
        }
    }

    #[test]
    fn test_rrf_merge_basic() {
        let fts = vec![
            make_result("a", 1),
            make_result("b", 2),
            make_result("c", 3),
        ];
        let vec = vec![
            make_result("b", 1),
            make_result("a", 2),
            make_result("d", 3),
        ];

        let merged = rrf_merge(&fts, &vec, 60);

        // Both a and b appear in both lists, so they should score highest.
        // b: 1/(60+2) + 1/(60+1) = ~0.0161 + ~0.0164 = ~0.0325
        // a: 1/(60+1) + 1/(60+2) = ~0.0164 + ~0.0161 = ~0.0325
        // They should be roughly equal, both ahead of c and d
        assert!(merged.len() == 4);
        let top_ids: Vec<&str> = merged.iter().take(2).map(|r| r.node_id.as_str()).collect();
        assert!(top_ids.contains(&"a") || top_ids.contains(&"b"));
    }

    #[test]
    fn test_rrf_merge_single_list() {
        let fts = vec![
            make_result("x", 1),
            make_result("y", 2),
        ];
        let vec = vec![];

        let merged = rrf_merge(&fts, &vec, 60);
        assert_eq!(merged.len(), 2);
        assert_eq!(merged[0].node_id, "x");
        assert_eq!(merged[1].node_id, "y");
    }

    #[test]
    fn test_rrf_ranks_assigned() {
        let fts = vec![make_result("a", 1)];
        let vec = vec![make_result("a", 1)];

        let merged = rrf_merge(&fts, &vec, 60);
        assert_eq!(merged[0].rank, 1);
    }

    // --- Literal boost tests ---

    use trevec_core::model::{NodeKind, Span};

    fn make_node(id: &str, name: &str, signature: &str, file_path: &str) -> CodeNode {
        CodeNode {
            id: id.to_string(),
            kind: NodeKind::Function,
            file_path: file_path.to_string(),
            span: Span {
                start_line: 0,
                start_col: 0,
                end_line: 10,
                end_col: 0,
                start_byte: 0,
                end_byte: 100,
            },
            name: name.to_string(),
            signature: signature.to_string(),
            doc_comment: None,
            identifiers: vec![],
            bm25_text: format!("{file_path} {signature} {name}"),
            symbol_vec: None,
            ast_hash: String::new(),
        }
    }

    #[test]
    fn test_literal_match_exact_name() {
        let node = make_node("n1", "authenticate", "fn authenticate()", "src/auth.rs");
        let score = literal_match_score(&node, "authenticate");
        assert!(score.is_some());
        // Exact name match should get the highest score (0.06)
        assert_eq!(score.unwrap(), 0.06);
    }

    #[test]
    fn test_literal_match_signature() {
        let node = make_node("n1", "verify", "fn verify_password(hash: &str)", "src/auth.rs");
        // Query matches in signature but not as exact name
        let score = literal_match_score(&node, "verify_password");
        assert!(score.is_some());
        // Signature substring match = 0.05 (higher than bm25_text = 0.035)
        assert_eq!(score.unwrap(), 0.05);
    }

    #[test]
    fn test_literal_boost_injects_new_results() {
        let mut nodes_map = HashMap::new();
        nodes_map.insert(
            "n1".to_string(),
            make_node("n1", "connect_token", "fn connect_token()", "src/db.rs"),
        );
        nodes_map.insert(
            "n2".to_string(),
            make_node("n2", "shutdown", "fn shutdown()", "src/db.rs"),
        );

        // Start with RRF results that only contain n2 (no literal match for "connect_token")
        let mut merged = vec![RankedResult {
            node_id: "n2".to_string(),
            score: 0.03,
            rank: 1,
        }];

        apply_literal_boost(&mut merged, &nodes_map, "connect_token");

        // n1 should now be injected (exact name match = 0.06)
        assert!(merged.len() >= 2, "Expected at least 2 results after boost");
        // n1 with score 0.06 should be ranked first
        assert_eq!(merged[0].node_id, "n1");
        assert_eq!(merged[0].rank, 1);
    }

    #[test]
    fn test_literal_boost_no_match() {
        let mut nodes_map = HashMap::new();
        nodes_map.insert(
            "n1".to_string(),
            make_node("n1", "connect", "fn connect()", "src/db.rs"),
        );

        let mut merged = vec![RankedResult {
            node_id: "n1".to_string(),
            score: 0.03,
            rank: 1,
        }];
        let original_score = merged[0].score;

        apply_literal_boost(&mut merged, &nodes_map, "zzz_nonexistent_zzz");

        // No literal match → scores unchanged
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].score, original_score);
    }

    #[test]
    fn test_literal_boost_skips_natural_language_query() {
        let mut nodes_map = HashMap::new();
        nodes_map.insert(
            "n1".to_string(),
            make_node("n1", "connect", "fn connect()", "src/db.rs"),
        );
        nodes_map.insert(
            "n2".to_string(),
            make_node("n2", "shutdown", "fn shutdown()", "src/db.rs"),
        );

        let mut merged = vec![RankedResult {
            node_id: "n2".to_string(),
            score: 0.03,
            rank: 1,
        }];

        apply_literal_boost(&mut merged, &nodes_map, "how does connect work");

        // Natural language query should not trigger full-map literal boosting.
        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].node_id, "n2");
    }

    #[test]
    fn test_literal_boost_applies_for_identifier_like_query() {
        let mut nodes_map = HashMap::new();
        nodes_map.insert(
            "n1".to_string(),
            make_node("n1", "generate_login_token", "fn generate_login_token()", "src/auth.rs"),
        );
        nodes_map.insert(
            "n2".to_string(),
            make_node("n2", "shutdown", "fn shutdown()", "src/db.rs"),
        );

        let mut merged = vec![RankedResult {
            node_id: "n2".to_string(),
            score: 0.03,
            rank: 1,
        }];

        apply_literal_boost(&mut merged, &nodes_map, "generate_login_token");

        assert!(merged.len() >= 2);
        assert_eq!(merged[0].node_id, "n1");
    }

    // --- Test file penalty tests ---

    #[test]
    fn test_is_test_file_directory_patterns() {
        assert!(is_test_file("tests/test_auth.py"));
        assert!(is_test_file("tests/__init__.py"));
        assert!(is_test_file("src/tests/test_db.py"));
        assert!(is_test_file("src/__tests__/auth.test.js"));
        assert!(is_test_file("test/test_models.py"));
        assert!(is_test_file("django/test/utils.py"));
    }

    #[test]
    fn test_is_test_file_python() {
        assert!(is_test_file("src/test_auth.py"));
        assert!(is_test_file("src/auth_test.py"));
        assert!(is_test_file("src/conftest.py"));
    }

    #[test]
    fn test_is_test_file_js_ts() {
        assert!(is_test_file("src/auth.test.js"));
        assert!(is_test_file("src/auth.spec.ts"));
        assert!(is_test_file("src/auth.test.tsx"));
        assert!(is_test_file("src/auth.spec.jsx"));
    }

    #[test]
    fn test_is_test_file_rust_go_java() {
        assert!(is_test_file("src/auth_test.rs"));
        assert!(is_test_file("src/auth_test.go"));
        assert!(is_test_file("src/AuthTest.java"));
        assert!(is_test_file("src/AuthTests.java"));
        assert!(is_test_file("src/AuthTest.kt"));
    }

    #[test]
    fn test_is_test_file_negative() {
        assert!(!is_test_file("src/contest.py"));
        assert!(!is_test_file("src/models.py"));
        assert!(!is_test_file("src/auth.rs"));
        assert!(!is_test_file("src/main.go"));
        assert!(!is_test_file("src/utils.js"));
        assert!(!is_test_file("src/testing_utils.py"));
    }

    #[test]
    fn test_matches_penalty_paths_basic() {
        let paths = vec!["galleries/".to_string(), "plot_types/".to_string()];
        assert!(matches_penalty_paths("lib/galleries/plot1.py", &paths));
        assert!(matches_penalty_paths("plot_types/bar.py", &paths));
        assert!(!matches_penalty_paths("src/models.py", &paths));
    }

    #[test]
    fn test_matches_penalty_paths_empty() {
        assert!(!matches_penalty_paths("tests/foo.py", &[]));
    }

    #[test]
    fn test_apply_penalty_noop_at_zero() {
        let mut nodes_map = HashMap::new();
        nodes_map.insert(
            "t1".to_string(),
            make_node("t1", "test_auth", "def test_auth()", "tests/test_auth.py"),
        );
        let mut merged = vec![RankedResult {
            node_id: "t1".to_string(),
            score: 0.05,
            rank: 1,
        }];
        apply_test_file_penalty(&mut merged, &nodes_map, 0.0, &[]);
        assert_eq!(merged[0].score, 0.05);
    }

    #[test]
    fn test_apply_penalty_reorders() {
        let mut nodes_map = HashMap::new();
        nodes_map.insert(
            "t1".to_string(),
            make_node("t1", "test_auth", "def test_auth()", "tests/test_auth.py"),
        );
        nodes_map.insert(
            "s1".to_string(),
            make_node("s1", "authenticate", "fn authenticate()", "src/auth.rs"),
        );

        let mut merged = vec![
            RankedResult { node_id: "t1".to_string(), score: 0.10, rank: 1 },
            RankedResult { node_id: "s1".to_string(), score: 0.08, rank: 2 },
        ];

        apply_test_file_penalty(&mut merged, &nodes_map, 0.5, &[]);

        // t1 score: 0.10 * 0.5 = 0.05, s1 stays at 0.08
        assert_eq!(merged[0].node_id, "s1");
        assert_eq!(merged[0].rank, 1);
        assert_eq!(merged[1].node_id, "t1");
        assert_eq!(merged[1].rank, 2);
    }

    #[test]
    fn test_apply_penalty_zeroes_at_one() {
        let mut nodes_map = HashMap::new();
        nodes_map.insert(
            "t1".to_string(),
            make_node("t1", "test_auth", "def test_auth()", "tests/test_auth.py"),
        );

        let mut merged = vec![RankedResult {
            node_id: "t1".to_string(),
            score: 0.10,
            rank: 1,
        }];

        apply_test_file_penalty(&mut merged, &nodes_map, 1.0, &[]);
        assert_eq!(merged[0].score, 0.0);
    }

    #[test]
    fn test_apply_penalty_custom_paths() {
        let mut nodes_map = HashMap::new();
        nodes_map.insert(
            "g1".to_string(),
            make_node("g1", "gallery_plot", "def gallery_plot()", "galleries/plot1.py"),
        );
        nodes_map.insert(
            "s1".to_string(),
            make_node("s1", "render", "fn render()", "src/render.rs"),
        );

        let mut merged = vec![
            RankedResult { node_id: "g1".to_string(), score: 0.10, rank: 1 },
            RankedResult { node_id: "s1".to_string(), score: 0.08, rank: 2 },
        ];

        let penalty_paths = vec!["galleries/".to_string()];
        apply_test_file_penalty(&mut merged, &nodes_map, 0.9, &penalty_paths);

        // g1 penalized: 0.10 * 0.1 = 0.01, s1 stays at 0.08
        assert_eq!(merged[0].node_id, "s1");
        assert_eq!(merged[1].node_id, "g1");
    }

    #[test]
    fn test_apply_penalty_combined_builtin_and_custom() {
        let mut nodes_map = HashMap::new();
        nodes_map.insert(
            "t1".to_string(),
            make_node("t1", "test_foo", "def test_foo()", "tests/test_foo.py"),
        );
        nodes_map.insert(
            "g1".to_string(),
            make_node("g1", "gallery", "def gallery()", "galleries/ex.py"),
        );
        nodes_map.insert(
            "s1".to_string(),
            make_node("s1", "core_fn", "fn core_fn()", "src/core.rs"),
        );

        let mut merged = vec![
            RankedResult { node_id: "t1".to_string(), score: 0.12, rank: 1 },
            RankedResult { node_id: "g1".to_string(), score: 0.10, rank: 2 },
            RankedResult { node_id: "s1".to_string(), score: 0.08, rank: 3 },
        ];

        let penalty_paths = vec!["galleries/".to_string()];
        apply_test_file_penalty(&mut merged, &nodes_map, 0.9, &penalty_paths);

        // Both t1 and g1 penalized, s1 should be first
        assert_eq!(merged[0].node_id, "s1");
        assert_eq!(merged[0].rank, 1);
    }

    // --- Non-code file penalty tests ---

    #[test]
    fn test_is_noncode_file_positive() {
        assert!(is_noncode_file("CHANGELOG.md"));
        assert!(is_noncode_file("README.md"));
        assert!(is_noncode_file("README.rst"));
        assert!(is_noncode_file("CONTRIBUTING.md"));
        assert!(is_noncode_file("AUTHORS"));
        assert!(is_noncode_file("LICENSE"));
        assert!(is_noncode_file("NEWS.txt"));
        assert!(is_noncode_file("HISTORY.md"));
        assert!(is_noncode_file("docs/guide.md"));
        assert!(is_noncode_file("src/notes.txt"));
        assert!(is_noncode_file("ISSUE_TEMPLATE.md"));
    }

    #[test]
    fn test_is_noncode_file_negative() {
        assert!(!is_noncode_file("src/auth.rs"));
        assert!(!is_noncode_file("src/main.py"));
        assert!(!is_noncode_file("src/utils.js"));
        assert!(!is_noncode_file("package.json"));
        assert!(!is_noncode_file("Cargo.toml"));
        assert!(!is_noncode_file("src/config.yaml"));
        assert!(!is_noncode_file("src/index.ts"));
    }

    #[test]
    fn test_apply_noncode_penalty_reorders() {
        let mut nodes_map = HashMap::new();
        nodes_map.insert(
            "d1".to_string(),
            make_node("d1", "changelog", "", "CHANGELOG.md"),
        );
        nodes_map.insert(
            "s1".to_string(),
            make_node("s1", "authenticate", "fn authenticate()", "src/auth.rs"),
        );

        let mut merged = vec![
            RankedResult { node_id: "d1".to_string(), score: 0.10, rank: 1 },
            RankedResult { node_id: "s1".to_string(), score: 0.08, rank: 2 },
        ];

        apply_noncode_penalty(&mut merged, &nodes_map, 0.8);

        // d1 score: 0.10 * 0.2 = 0.02, s1 stays at 0.08
        assert_eq!(merged[0].node_id, "s1");
        assert_eq!(merged[0].rank, 1);
    }

    // --- should_exclude_from_search tests ---

    #[test]
    fn test_should_exclude_markdown() {
        assert!(should_exclude_from_search("CHANGELOG.md"));
        assert!(should_exclude_from_search("README.md"));
        assert!(should_exclude_from_search("docs/guide.mdx"));
        assert!(should_exclude_from_search("HISTORY.md"));
    }

    #[test]
    fn test_should_exclude_lock_yaml_txt() {
        assert!(should_exclude_from_search("package-lock.json"));
        assert!(should_exclude_from_search("yarn.lock"));
        assert!(should_exclude_from_search("ci/config.yaml"));
        assert!(should_exclude_from_search("notes.txt"));
        assert!(should_exclude_from_search("docs/api.rst"));
    }

    #[test]
    fn test_should_exclude_json_except_allowlisted() {
        assert!(should_exclude_from_search("babel.config.json"));
        assert!(should_exclude_from_search("lerna.json"));
        // Allowlisted
        assert!(!should_exclude_from_search("package.json"));
        assert!(!should_exclude_from_search("tsconfig.json"));
        assert!(!should_exclude_from_search("trevec.json"));
    }

    #[test]
    fn test_should_exclude_docs_dirs() {
        assert!(should_exclude_from_search("docs/guide.py"));
        assert!(should_exclude_from_search("site/index.html"));
    }

    #[test]
    fn test_should_not_exclude_code() {
        assert!(!should_exclude_from_search("src/auth.rs"));
        assert!(!should_exclude_from_search("src/main.py"));
        assert!(!should_exclude_from_search("src/utils.js"));
        assert!(!should_exclude_from_search("Cargo.toml"));
        assert!(!should_exclude_from_search("src/index.ts"));
        assert!(!should_exclude_from_search("package.json"));
    }

    #[test]
    fn test_filter_noncode_files_removes_and_reranks() {
        let mut nodes_map = HashMap::new();
        nodes_map.insert("d1".to_string(), make_node("d1", "changelog", "", "CHANGELOG.md"));
        nodes_map.insert("s1".to_string(), make_node("s1", "auth", "fn auth()", "src/auth.rs"));
        nodes_map.insert("d2".to_string(), make_node("d2", "readme", "", "README.md"));

        let mut merged = vec![
            RankedResult { node_id: "d1".to_string(), score: 0.12, rank: 1 },
            RankedResult { node_id: "s1".to_string(), score: 0.10, rank: 2 },
            RankedResult { node_id: "d2".to_string(), score: 0.08, rank: 3 },
        ];

        filter_noncode_files(&mut merged, &nodes_map);

        assert_eq!(merged.len(), 1);
        assert_eq!(merged[0].node_id, "s1");
        assert_eq!(merged[0].rank, 1);
    }

    // --- Test fixture tests ---

    #[test]
    fn test_is_test_fixture() {
        assert!(is_test_fixture("tests/samples/expected/output.js"));
        assert!(is_test_fixture("test/fixtures/data.json"));
        assert!(is_test_fixture("src/__fixtures__/mock.ts"));
        assert!(is_test_fixture("src/__snapshots__/app.snap"));
        assert!(is_test_fixture("src/test/cases/foo-bundle.js"));
        assert!(!is_test_fixture("src/auth.rs"));
        assert!(!is_test_fixture("tests/test_auth.py"));
    }

    // --- derive_source_paths tests ---

    #[test]
    fn test_derive_source_js_ts() {
        let paths = derive_source_paths("src/auth.test.ts");
        assert!(paths.contains(&"src/auth.ts".to_string()));

        let paths = derive_source_paths("src/utils.spec.js");
        assert!(paths.contains(&"src/utils.js".to_string()));

        let paths = derive_source_paths("src/__tests__/Button.test.tsx");
        assert!(paths.contains(&"src/Button.tsx".to_string()));
    }

    #[test]
    fn test_derive_source_python() {
        let paths = derive_source_paths("tests/test_auth.py");
        assert!(paths.contains(&"auth.py".to_string()));
        assert!(paths.contains(&"src/auth.py".to_string()));

        let paths = derive_source_paths("tests/models/test_user.py");
        assert!(paths.contains(&"src/models/user.py".to_string()));
    }

    #[test]
    fn test_derive_source_go() {
        let paths = derive_source_paths("pkg/auth_test.go");
        assert!(paths.contains(&"pkg/auth.go".to_string()));
    }

    #[test]
    fn test_derive_source_java() {
        let paths = derive_source_paths("src/test/java/com/example/AuthTest.java");
        assert!(paths.contains(&"src/main/java/com/example/Auth.java".to_string()));
    }

    // --- boost_source_neighbors tests ---

    #[test]
    fn test_boost_source_neighbors_injects() {
        let mut nodes_map = HashMap::new();
        nodes_map.insert(
            "t1".to_string(),
            make_node("t1", "test_auth", "def test_auth()", "tests/test_auth.py"),
        );
        nodes_map.insert(
            "s1".to_string(),
            make_node("s1", "authenticate", "fn authenticate()", "src/auth.py"),
        );

        // Only test file in results
        let mut merged = vec![
            RankedResult { node_id: "t1".to_string(), score: 0.10, rank: 1 },
        ];

        boost_source_neighbors(&mut merged, &nodes_map);

        // s1 should be injected with boosted score
        assert!(merged.len() >= 2, "Expected source file to be injected");
        // The source file should have a score of 0.10 * 1.5 = 0.15
        let s1 = merged.iter().find(|r| r.node_id == "s1").unwrap();
        assert!((s1.score - 0.15).abs() < 0.001);
    }

    #[test]
    fn test_boost_source_neighbors_boosts_existing() {
        let mut nodes_map = HashMap::new();
        nodes_map.insert(
            "t1".to_string(),
            make_node("t1", "test_auth", "def test_auth()", "tests/test_auth.py"),
        );
        nodes_map.insert(
            "s1".to_string(),
            make_node("s1", "authenticate", "fn authenticate()", "src/auth.py"),
        );

        let mut merged = vec![
            RankedResult { node_id: "t1".to_string(), score: 0.10, rank: 1 },
            RankedResult { node_id: "s1".to_string(), score: 0.05, rank: 2 },
        ];

        boost_source_neighbors(&mut merged, &nodes_map);

        // s1 should be boosted: 0.05 + 0.15 = 0.20
        let s1 = merged.iter().find(|r| r.node_id == "s1").unwrap();
        assert!((s1.score - 0.20).abs() < 0.001);
        // s1 should now rank first
        assert_eq!(merged[0].node_id, "s1");
    }
}
