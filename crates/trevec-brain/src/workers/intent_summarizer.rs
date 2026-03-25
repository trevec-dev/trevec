//! Worker 1: Intent Summarizer
//!
//! Generates structured summaries of code nodes using an LLM.
//! Summaries feed into bm25_text for dramatically better text search.

use trevec_core::universal::IntentSummary;

/// The structured prompt template for intent summarization.
pub const INTENT_PROMPT_TEMPLATE: &str = r#"Analyze this code and fill in ALL fields concisely:

```{language}
{source_code}
```

PURPOSE: [one sentence: what this code does]
INPUTS: [parameter names and what they represent]
OUTPUTS: [return value and what it represents]
SIDE_EFFECTS: [any mutations, I/O, or state changes, or "none"]
RELATED_CONCEPTS: [comma-separated domain concepts, e.g. "JWT authentication, token validation"]
ERROR_CASES: [what can go wrong, or "none"]"#;

/// Current prompt version (increment when changing the template).
pub const PROMPT_VERSION: &str = "intent_v1";

/// Build the prompt for a given code node.
pub fn build_prompt(language: &str, source_code: &str) -> String {
    INTENT_PROMPT_TEMPLATE
        .replace("{language}", language)
        .replace("{source_code}", source_code)
}

/// Parse an LLM response into an IntentSummary.
pub fn parse_response(response: &str, model_version: &str, source_hash: &str) -> IntentSummary {
    let mut summary = IntentSummary {
        model_version: Some(model_version.to_string()),
        source_hash: Some(source_hash.to_string()),
        ..Default::default()
    };

    for line in response.lines() {
        let line = line.trim();
        if let Some(val) = line.strip_prefix("PURPOSE:") {
            let val = val.trim();
            if !val.is_empty() {
                summary.purpose = Some(val.to_string());
            }
        } else if let Some(val) = line.strip_prefix("INPUTS:") {
            let val = val.trim();
            if !val.is_empty() {
                summary.inputs = Some(val.to_string());
            }
        } else if let Some(val) = line.strip_prefix("OUTPUTS:") {
            let val = val.trim();
            if !val.is_empty() {
                summary.outputs = Some(val.to_string());
            }
        } else if let Some(val) = line.strip_prefix("SIDE_EFFECTS:") {
            let val = val.trim();
            if !val.is_empty() && val.to_lowercase() != "none" {
                summary.side_effects = Some(val.to_string());
            }
        } else if let Some(val) = line.strip_prefix("RELATED_CONCEPTS:") {
            let val = val.trim();
            if !val.is_empty() {
                summary.related_concepts = val
                    .split(',')
                    .map(|s| s.trim().to_string())
                    .filter(|s| !s.is_empty())
                    .collect();
            }
        } else if let Some(val) = line.strip_prefix("ERROR_CASES:") {
            let val = val.trim();
            if !val.is_empty() && val.to_lowercase() != "none" {
                summary.error_cases = Some(val.to_string());
            }
        }
    }

    summary
}

/// Detect language from file extension.
pub fn detect_language(file_path: &str) -> &str {
    let path = file_path.to_lowercase();
    if path.ends_with(".rs") {
        "rust"
    } else if path.ends_with(".py") {
        "python"
    } else if path.ends_with(".js") || path.ends_with(".jsx") {
        "javascript"
    } else if path.ends_with(".ts") || path.ends_with(".tsx") {
        "typescript"
    } else if path.ends_with(".go") {
        "go"
    } else if path.ends_with(".java") {
        "java"
    } else if path.ends_with(".rb") {
        "ruby"
    } else if path.ends_with(".c") || path.ends_with(".h") {
        "c"
    } else if path.ends_with(".cpp") || path.ends_with(".hpp") {
        "cpp"
    } else if path.ends_with(".cs") {
        "csharp"
    } else if path.ends_with(".swift") {
        "swift"
    } else if path.ends_with(".zig") {
        "zig"
    } else {
        "code"
    }
}

/// Estimate the cost of summarizing a node (input + output tokens).
pub fn estimate_cost(source_len: usize, input_price_per_mtok: f64, output_price_per_mtok: f64) -> f64 {
    let input_tokens = (INTENT_PROMPT_TEMPLATE.len() + source_len) / 4;
    let output_tokens = 150; // typical summary output
    let input_cost = (input_tokens as f64 / 1_000_000.0) * input_price_per_mtok;
    let output_cost = (output_tokens as f64 / 1_000_000.0) * output_price_per_mtok;
    input_cost + output_cost
}

/// Check if a node is worth summarizing (skip trivial code).
pub fn should_summarize(source_code: &str) -> bool {
    let lines = source_code.lines().count();
    let bytes = source_code.len();
    // Skip very short functions (< 3 lines or < 50 bytes)
    lines >= 3 && bytes >= 50
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_build_prompt() {
        let prompt = build_prompt("rust", "fn authenticate() { }");
        assert!(prompt.contains("```rust"));
        assert!(prompt.contains("fn authenticate()"));
        assert!(prompt.contains("PURPOSE:"));
        assert!(prompt.contains("RELATED_CONCEPTS:"));
    }

    #[test]
    fn test_parse_response() {
        let response = r#"PURPOSE: Authenticates a user via JWT token validation.
INPUTS: req - the HTTP request containing a Bearer token
OUTPUTS: Result<User> - the authenticated user or an error
SIDE_EFFECTS: Updates the user's last_login timestamp in the database
RELATED_CONCEPTS: JWT authentication, token validation, session management
ERROR_CASES: Expired token, invalid signature, user not found"#;

        let summary = parse_response(response, "qwen2.5-coder-1.5b", "abc123");

        assert_eq!(
            summary.purpose.as_deref(),
            Some("Authenticates a user via JWT token validation.")
        );
        assert!(summary.inputs.is_some());
        assert!(summary.outputs.is_some());
        assert!(summary.side_effects.is_some());
        assert_eq!(summary.related_concepts.len(), 3);
        assert!(summary.related_concepts.contains(&"JWT authentication".to_string()));
        assert!(summary.error_cases.is_some());
        assert_eq!(summary.model_version.as_deref(), Some("qwen2.5-coder-1.5b"));
        assert_eq!(summary.source_hash.as_deref(), Some("abc123"));
    }

    #[test]
    fn test_parse_response_with_none_fields() {
        let response = r#"PURPOSE: Simple helper function.
INPUTS: none
OUTPUTS: unit
SIDE_EFFECTS: none
RELATED_CONCEPTS: utility
ERROR_CASES: none"#;

        let summary = parse_response(response, "test", "hash");
        assert!(summary.purpose.is_some());
        assert!(summary.side_effects.is_none()); // "none" should be filtered
        assert!(summary.error_cases.is_none()); // "none" should be filtered
    }

    #[test]
    fn test_parse_response_partial() {
        let response = "PURPOSE: Does something useful.";
        let summary = parse_response(response, "test", "hash");
        assert!(summary.purpose.is_some());
        assert!(summary.inputs.is_none());
        assert!(!summary.is_empty());
    }

    #[test]
    fn test_detect_language() {
        assert_eq!(detect_language("src/main.rs"), "rust");
        assert_eq!(detect_language("app.py"), "python");
        assert_eq!(detect_language("index.ts"), "typescript");
        assert_eq!(detect_language("Main.java"), "java");
        assert_eq!(detect_language("unknown.xyz"), "code");
    }

    #[test]
    fn test_should_summarize() {
        assert!(should_summarize("fn main() {\n    let x = 1;\n    println!(\"{}\", x);\n}"));
        assert!(!should_summarize("fn f() {}"));
        assert!(!should_summarize("x"));
    }

    #[test]
    fn test_estimate_cost() {
        let cost = estimate_cost(500, 3.0, 15.0);
        assert!(cost > 0.0);
        assert!(cost < 0.01); // Should be very cheap per node
    }

    #[test]
    fn test_bm25_enrichment_flow() {
        // Simulate the full flow: code → prompt → response → bm25_text
        let source = "fn authenticate(req: &Request) -> Result<User> {\n    verify_jwt(req)\n}";
        let prompt = build_prompt("rust", source);
        assert!(prompt.contains("rust"));

        let response = "PURPOSE: Authenticates user via JWT.\nRELATED_CONCEPTS: JWT, auth";
        let summary = parse_response(response, "test", "hash");

        let bm25 = summary.to_bm25_text();
        assert!(bm25.contains("Authenticates user via JWT"));
        assert!(bm25.contains("JWT"));
        assert!(bm25.contains("auth"));
    }
}
