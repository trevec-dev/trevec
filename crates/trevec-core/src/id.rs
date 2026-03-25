use crate::model::NodeKind;

/// Generate a deterministic node ID from its defining properties.
/// ID = blake3(file_path|kind|signature|start_byte), truncated to 32 hex chars.
pub fn generate_node_id(file_path: &str, kind: NodeKind, signature: &str, start_byte: usize) -> String {
    let input = format!("{}|{}|{}|{}", file_path, kind, signature, start_byte);
    let hash = blake3::hash(input.as_bytes());
    hash.to_hex()[..32].to_string()
}

/// Generate a hash of source content for incremental update detection.
pub fn compute_ast_hash(source: &[u8]) -> String {
    let hash = blake3::hash(source);
    hash.to_hex()[..32].to_string()
}

/// Compute a blake3 hash of a file's entire contents.
pub fn compute_file_hash(content: &[u8]) -> String {
    let hash = blake3::hash(content);
    hash.to_hex()[..32].to_string()
}

/// Generate a bundle ID from the query and timestamp.
pub fn generate_bundle_id(query: &str) -> String {
    let input = format!(
        "{}|{}",
        query,
        std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .as_nanos()
    );
    let hash = blake3::hash(input.as_bytes());
    hash.to_hex()[..16].to_string()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_generate_node_id_deterministic() {
        let id1 = generate_node_id("src/main.rs", NodeKind::Function, "fn main()", 0);
        let id2 = generate_node_id("src/main.rs", NodeKind::Function, "fn main()", 0);
        assert_eq!(id1, id2);
        assert_eq!(id1.len(), 32);
    }

    #[test]
    fn test_generate_node_id_different_inputs() {
        let id1 = generate_node_id("src/main.rs", NodeKind::Function, "fn main()", 0);
        let id2 = generate_node_id("src/lib.rs", NodeKind::Function, "fn main()", 0);
        assert_ne!(id1, id2);
    }

    #[test]
    fn test_compute_ast_hash_deterministic() {
        let hash1 = compute_ast_hash(b"fn main() { println!(\"hello\"); }");
        let hash2 = compute_ast_hash(b"fn main() { println!(\"hello\"); }");
        assert_eq!(hash1, hash2);
        assert_eq!(hash1.len(), 32);
    }

    #[test]
    fn test_compute_ast_hash_different_content() {
        let hash1 = compute_ast_hash(b"fn main() {}");
        let hash2 = compute_ast_hash(b"fn main() { return; }");
        assert_ne!(hash1, hash2);
    }

    #[test]
    fn test_compute_file_hash_deterministic() {
        let h1 = compute_file_hash(b"fn main() { println!(\"hello\"); }");
        let h2 = compute_file_hash(b"fn main() { println!(\"hello\"); }");
        assert_eq!(h1, h2);
        assert_eq!(h1.len(), 32);
    }

    #[test]
    fn test_compute_file_hash_different_content() {
        let h1 = compute_file_hash(b"version 1");
        let h2 = compute_file_hash(b"version 2");
        assert_ne!(h1, h2);
    }

    #[test]
    fn test_generate_bundle_id_unique() {
        let id1 = generate_bundle_id("test query");
        let id2 = generate_bundle_id("test query");
        // Different because of timestamp
        assert_ne!(id1, id2);
        assert_eq!(id1.len(), 16);
    }
}
