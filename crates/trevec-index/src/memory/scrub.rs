use regex::Regex;
use std::sync::OnceLock;

/// Secret-scrubbing patterns and their replacements.
const PATTERNS: &[(&str, &str)] = &[
    // OpenAI-style API keys (sk-proj must come before sk- to match longer prefix first)
    (r"sk-proj-[a-zA-Z0-9\-_]{40,}", "<REDACTED_API_KEY>"),
    (r"sk-[a-zA-Z0-9]{20,}", "<REDACTED_API_KEY>"),
    // GitHub tokens
    (r"github_pat_[a-zA-Z0-9_]{22,}", "<REDACTED_GITHUB_TOKEN>"),
    (r"ghp_[a-zA-Z0-9]{36,}", "<REDACTED_GITHUB_TOKEN>"),
    // AWS access keys
    (r"AKIA[0-9A-Z]{16}", "<REDACTED_AWS_KEY>"),
    // Private keys (PEM)
    (
        r"-----BEGIN[A-Z ]*PRIVATE KEY-----[\s\S]*?-----END[A-Z ]*PRIVATE KEY-----",
        "<REDACTED_PRIVATE_KEY>",
    ),
    // Bearer tokens
    (r"Bearer\s+[a-zA-Z0-9\-._~+/]+=*", "<REDACTED_BEARER>"),
    // Slack tokens
    (r"xox[bsrp]-[a-zA-Z0-9\-]{10,}", "<REDACTED_SLACK_TOKEN>"),
    // Email addresses
    (
        r"[a-zA-Z0-9._%+\-]+@[a-zA-Z0-9.\-]+\.[a-zA-Z]{2,}",
        "<REDACTED_EMAIL>",
    ),
];

static COMPILED: OnceLock<Vec<(Regex, &'static str)>> = OnceLock::new();

fn compiled_patterns() -> &'static [(Regex, &'static str)] {
    COMPILED.get_or_init(|| {
        PATTERNS
            .iter()
            .map(|(pattern, replacement)| (Regex::new(pattern).unwrap(), *replacement))
            .collect()
    })
}

/// Scrub secrets from text. Returns (scrubbed_text, redaction_count).
pub fn scrub(text: &str) -> (String, usize) {
    let mut result = text.to_string();
    let mut count = 0;

    for (regex, replacement) in compiled_patterns() {
        let before_len = result.len();
        let replaced = regex.replace_all(&result, *replacement);
        if replaced.len() != before_len || replaced != result {
            // Count individual matches
            count += regex.find_iter(&result).count();
            result = replaced.into_owned();
        }
    }

    (result, count)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_scrub_openai_key() {
        let input = "My key is sk-abcdefghijklmnopqrstuvwxyz and more text";
        let (scrubbed, count) = scrub(input);
        assert!(!scrubbed.contains("sk-abcdefghijklmnopqrstuvwxyz"));
        assert!(scrubbed.contains("<REDACTED_API_KEY>"));
        assert_eq!(count, 1);
    }

    #[test]
    fn test_scrub_openai_project_key() {
        let input = "key: sk-proj-abcdefghijklmnopqrstuvwxyz1234567890ABCDEF";
        let (scrubbed, count) = scrub(input);
        assert!(scrubbed.contains("<REDACTED_API_KEY>"));
        assert_eq!(count, 1);
    }

    #[test]
    fn test_scrub_github_token() {
        let input = "token: ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij";
        let (scrubbed, count) = scrub(input);
        assert!(scrubbed.contains("<REDACTED_GITHUB_TOKEN>"));
        assert_eq!(count, 1);
    }

    #[test]
    fn test_scrub_github_pat() {
        let input = "pat: github_pat_ABCDEFGHIJKLMNOPQRSTUV";
        let (scrubbed, count) = scrub(input);
        assert!(scrubbed.contains("<REDACTED_GITHUB_TOKEN>"));
        assert_eq!(count, 1);
    }

    #[test]
    fn test_scrub_aws_key() {
        let input = "aws_access_key=AKIAIOSFODNN7EXAMPLE";
        let (scrubbed, count) = scrub(input);
        assert!(scrubbed.contains("<REDACTED_AWS_KEY>"));
        assert_eq!(count, 1);
    }

    #[test]
    fn test_scrub_bearer_token() {
        let input = "Authorization: Bearer eyJhbGciOiJIUzI1NiJ9.payload.sig";
        let (scrubbed, count) = scrub(input);
        assert!(scrubbed.contains("<REDACTED_BEARER>"));
        assert_eq!(count, 1);
    }

    #[test]
    fn test_scrub_slack_token() {
        let input = "SLACK_TOKEN=xoxb-1234567890-abcdefghij";
        let (scrubbed, count) = scrub(input);
        assert!(scrubbed.contains("<REDACTED_SLACK_TOKEN>"));
        assert_eq!(count, 1);
    }

    #[test]
    fn test_scrub_email() {
        let input = "Contact alice@example.com for details";
        let (scrubbed, count) = scrub(input);
        assert!(scrubbed.contains("<REDACTED_EMAIL>"));
        assert!(!scrubbed.contains("alice@example.com"));
        assert_eq!(count, 1);
    }

    #[test]
    fn test_scrub_private_key() {
        let input = "-----BEGIN RSA PRIVATE KEY-----\nMIIEpAIBAAK...\n-----END RSA PRIVATE KEY-----";
        let (scrubbed, count) = scrub(input);
        assert!(scrubbed.contains("<REDACTED_PRIVATE_KEY>"));
        assert_eq!(count, 1);
    }

    #[test]
    fn test_scrub_multiple_secrets() {
        let input = "key=sk-abcdefghijklmnopqrstuvwxyz token=ghp_ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghij email=bob@corp.io";
        let (scrubbed, count) = scrub(input);
        assert!(scrubbed.contains("<REDACTED_API_KEY>"));
        assert!(scrubbed.contains("<REDACTED_GITHUB_TOKEN>"));
        assert!(scrubbed.contains("<REDACTED_EMAIL>"));
        assert!(count >= 3);
    }

    #[test]
    fn test_scrub_no_secrets() {
        let input = "This is normal text with no secrets at all.";
        let (scrubbed, count) = scrub(input);
        assert_eq!(scrubbed, input);
        assert_eq!(count, 0);
    }

    #[test]
    fn test_scrub_empty() {
        let (scrubbed, count) = scrub("");
        assert_eq!(scrubbed, "");
        assert_eq!(count, 0);
    }
}
