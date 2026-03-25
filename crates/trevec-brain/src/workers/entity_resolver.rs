//! Worker 2: Entity Resolver
//!
//! Deduplicates and links entities across files/domains without LLM calls
//! for 95%+ of cases. Only ambiguous cases (0.75-0.92 score) go to LLM.
//!
//! Uses: Jaro-Winkler + cosine similarity + co-occurrence scoring.

/// Compute Jaro-Winkler similarity between two strings.
/// Returns a value in [0.0, 1.0] where 1.0 = identical.
pub fn jaro_winkler(a: &str, b: &str) -> f64 {
    if a.is_empty() && b.is_empty() {
        return 1.0;
    }
    if a == b {
        return 1.0;
    }
    if a.is_empty() || b.is_empty() {
        return 0.0;
    }

    let jaro = jaro_similarity(a, b);

    // Winkler prefix bonus
    let prefix_len = a
        .chars()
        .zip(b.chars())
        .take(4)
        .take_while(|(a, b)| a == b)
        .count();

    jaro + (prefix_len as f64 * 0.1 * (1.0 - jaro))
}

/// Compute Jaro similarity between two strings.
fn jaro_similarity(a: &str, b: &str) -> f64 {
    let a_chars: Vec<char> = a.chars().collect();
    let b_chars: Vec<char> = b.chars().collect();
    let a_len = a_chars.len();
    let b_len = b_chars.len();

    if a_len == 0 && b_len == 0 {
        return 1.0;
    }

    let match_distance = (a_len.max(b_len) / 2).saturating_sub(1);

    let mut a_matches = vec![false; a_len];
    let mut b_matches = vec![false; b_len];

    let mut matches = 0.0;
    let mut transpositions = 0.0;

    for i in 0..a_len {
        let start = i.saturating_sub(match_distance);
        let end = (i + match_distance + 1).min(b_len);

        for j in start..end {
            if b_matches[j] || a_chars[i] != b_chars[j] {
                continue;
            }
            a_matches[i] = true;
            b_matches[j] = true;
            matches += 1.0;
            break;
        }
    }

    if matches == 0.0 {
        return 0.0;
    }

    let mut k = 0;
    for i in 0..a_len {
        if !a_matches[i] {
            continue;
        }
        while !b_matches[k] {
            k += 1;
        }
        if a_chars[i] != b_chars[k] {
            transpositions += 1.0;
        }
        k += 1;
    }

    (matches / a_len as f64
        + matches / b_len as f64
        + (matches - transpositions / 2.0) / matches)
        / 3.0
}

/// Normalize a code identifier for comparison.
/// Converts camelCase/PascalCase to lowercase with separators.
/// Consecutive uppercase letters (acronyms like HTTP) are kept together.
pub fn normalize_identifier(name: &str) -> String {
    let mut result = String::new();
    let chars: Vec<char> = name.chars().collect();

    for (i, &ch) in chars.iter().enumerate() {
        if ch == '_' || ch == '-' {
            if !result.ends_with(' ') {
                result.push(' ');
            }
        } else if ch.is_uppercase() {
            let prev_upper = i > 0 && chars[i - 1].is_uppercase();
            let prev_sep = i > 0 && (chars[i - 1] == '_' || chars[i - 1] == '-');
            let next_lower = i + 1 < chars.len() && chars[i + 1].is_lowercase();

            // Insert space before uppercase if:
            // - Previous char was lowercase (camelCase boundary)
            // - Previous char was uppercase AND next is lowercase (end of acronym: HTTPResponse -> HTTP Response)
            if i > 0 && !prev_sep {
                let prev_lower = chars[i - 1].is_lowercase();
                if prev_lower || (prev_upper && next_lower) {
                    if !result.ends_with(' ') {
                        result.push(' ');
                    }
                }
            }
            result.push(ch.to_lowercase().next().unwrap_or(ch));
        } else {
            result.push(ch);
        }
    }

    result.trim().to_string()
}

/// Compute a composite entity resolution score.
///
/// score = alpha * jaro_winkler(name_a, name_b)
///       + beta  * jaro_winkler(normalized_a, normalized_b)
///       + gamma * same_module_bonus
///       + delta * signature_overlap
pub fn entity_score(
    name_a: &str,
    name_b: &str,
    same_module: bool,
    sig_a: Option<&str>,
    sig_b: Option<&str>,
) -> f64 {
    let alpha = 0.35;
    let beta = 0.35;
    let gamma = 0.15;
    let delta = 0.15;

    let name_sim = jaro_winkler(name_a, name_b);
    let norm_a = normalize_identifier(name_a);
    let norm_b = normalize_identifier(name_b);
    let norm_sim = jaro_winkler(&norm_a, &norm_b);
    let module_bonus = if same_module { 1.0 } else { 0.0 };

    let sig_overlap = match (sig_a, sig_b) {
        (Some(a), Some(b)) => {
            let a_words: std::collections::HashSet<&str> = a.split_whitespace().collect();
            let b_words: std::collections::HashSet<&str> = b.split_whitespace().collect();
            let intersection = a_words.intersection(&b_words).count();
            let union = a_words.union(&b_words).count();
            if union > 0 {
                intersection as f64 / union as f64
            } else {
                0.0
            }
        }
        _ => 0.0,
    };

    alpha * name_sim + beta * norm_sim + gamma * module_bonus + delta * sig_overlap
}

/// Resolution decision based on score.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ResolutionDecision {
    /// Score > 0.92: auto-merge
    AutoMerge,
    /// Score 0.75-0.92: needs LLM verification
    NeedsVerification,
    /// Score < 0.75: separate entities
    Separate,
}

pub fn resolve(score: f64) -> ResolutionDecision {
    if score > 0.92 {
        ResolutionDecision::AutoMerge
    } else if score >= 0.75 {
        ResolutionDecision::NeedsVerification
    } else {
        ResolutionDecision::Separate
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_jaro_winkler_identical() {
        assert!((jaro_winkler("hello", "hello") - 1.0).abs() < 0.001);
    }

    #[test]
    fn test_jaro_winkler_similar() {
        let score = jaro_winkler("authenticate", "authentication");
        assert!(score > 0.9);
    }

    #[test]
    fn test_jaro_winkler_different() {
        let score = jaro_winkler("hello", "world");
        assert!(score < 0.5);
    }

    #[test]
    fn test_jaro_winkler_empty() {
        assert_eq!(jaro_winkler("", "hello"), 0.0);
        assert_eq!(jaro_winkler("hello", ""), 0.0);
        assert!((jaro_winkler("", "") - 1.0).abs() < 0.001); // both empty = identical
    }

    #[test]
    fn test_jaro_winkler_prefix_bonus() {
        // Strings with shared prefix should score higher than without
        let with_prefix = jaro_winkler("getUserById", "getUserByName");
        let without_prefix = jaro_winkler("getUser", "fetchUser");
        assert!(with_prefix > without_prefix);
    }

    #[test]
    fn test_normalize_identifier() {
        assert_eq!(normalize_identifier("getUserById"), "get user by id");
        assert_eq!(normalize_identifier("get_user_by_id"), "get user by id");
        assert_eq!(normalize_identifier("HTTPResponse"), "http response");
        assert_eq!(normalize_identifier("simple"), "simple");
    }

    #[test]
    fn test_entity_score_same_name() {
        let score = entity_score("authenticate", "authenticate", true, None, None);
        // alpha*1.0 + beta*1.0 + gamma*1.0 + delta*0.0 = 0.35 + 0.35 + 0.15 = 0.85
        assert!(score > 0.8, "Expected > 0.8, got {}", score);
        assert_eq!(resolve(score), ResolutionDecision::NeedsVerification);
        // With same name + same module, this would auto-merge with sigs
        let score2 = entity_score(
            "authenticate",
            "authenticate",
            true,
            Some("fn authenticate() -> bool"),
            Some("fn authenticate() -> bool"),
        );
        assert!(score2 > 0.92, "Expected > 0.92, got {}", score2);
        assert_eq!(resolve(score2), ResolutionDecision::AutoMerge);
    }

    #[test]
    fn test_entity_score_naming_convention() {
        let score = entity_score(
            "getUserById",
            "get_user_by_id",
            false,
            Some("fn getUserById(id: &str) -> User"),
            Some("def get_user_by_id(id: str) -> User"),
        );
        // Name sim is moderate, normalized sim should be high
        assert!(score > 0.6, "Expected > 0.6, got {}", score);
    }

    #[test]
    fn test_entity_score_different() {
        let score = entity_score("processPayment", "validateEmail", false, None, None);
        assert!(score < 0.75);
        assert_eq!(resolve(score), ResolutionDecision::Separate);
    }

    #[test]
    fn test_entity_score_same_module_bonus() {
        let without = entity_score("handler", "handle", false, None, None);
        let with = entity_score("handler", "handle", true, None, None);
        assert!(with > without);
    }

    #[test]
    fn test_resolution_thresholds() {
        assert_eq!(resolve(0.95), ResolutionDecision::AutoMerge);
        assert_eq!(resolve(0.85), ResolutionDecision::NeedsVerification);
        assert_eq!(resolve(0.50), ResolutionDecision::Separate);
    }
}
