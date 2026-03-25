//! Conversation Parser: extracts entities, preferences, decisions, and topics
//! from chat session data. Uses rule-based extraction (no LLM).
//!
//! Designed to compete on LongMemEval-s benchmark.

use anyhow::Result;
use regex::Regex;
use serde::Deserialize;
use std::collections::HashMap;
use std::sync::OnceLock;

use trevec_core::universal::*;
use trevec_core::{Confidence, TrevecConfig};

use crate::registry::{DomainParser, ParseResult};

// ── Input Format ─────────────────────────────────────────────────────────────

#[derive(Debug, Deserialize)]
pub struct ConversationInput {
    #[serde(default)]
    pub sessions: Vec<SessionInput>,
}

#[derive(Debug, Deserialize)]
pub struct SessionInput {
    pub id: String,
    #[serde(default)]
    pub timestamp: Option<String>,
    pub messages: Vec<MessageInput>,
}

#[derive(Debug, Deserialize)]
pub struct MessageInput {
    pub role: String,
    pub content: String,
    #[serde(default)]
    pub timestamp: Option<String>,
}

// ── Parser ───────────────────────────────────────────────────────────────────

pub struct ConversationParser;

impl DomainParser for ConversationParser {
    fn domain_id(&self) -> &'static str {
        "conversation"
    }

    fn supported_extensions(&self) -> &[&'static str] {
        &[".conversation.json", ".chat.json", ".longmemeval.json"]
    }

    fn parse(
        &self,
        file_path: &str,
        source: &[u8],
        _config: &TrevecConfig,
    ) -> Result<ParseResult> {
        let input: ConversationInput = serde_json::from_slice(source)?;
        parse_conversation(file_path, &input)
    }

    fn can_parse(&self, file_path: &str, first_bytes: &[u8]) -> bool {
        let path_lower = file_path.to_lowercase();
        if self
            .supported_extensions()
            .iter()
            .any(|ext| path_lower.ends_with(ext))
        {
            return true;
        }
        // Also try JSON files that look like conversations
        if path_lower.ends_with(".json") && first_bytes.len() >= 20 {
            let start = String::from_utf8_lossy(&first_bytes[..first_bytes.len().min(200)]);
            return start.contains("\"sessions\"") || start.contains("\"messages\"");
        }
        false
    }
}

/// Parse a conversation input into universal nodes and edges.
pub fn parse_conversation(file_path: &str, input: &ConversationInput) -> Result<ParseResult> {
    let mut nodes = Vec::new();
    let mut edges = Vec::new();
    let mut prev_session_id: Option<String> = None;

    for session in &input.sessions {
        let session_node_id = make_id(&format!("session:{}", session.id));
        let session_ts = session
            .timestamp
            .as_deref()
            .and_then(parse_timestamp)
            .unwrap_or(0);

        // Create session node
        nodes.push(UniversalNode {
            id: session_node_id.clone(),
            kind: UniversalKind::Topic,
            domain: DomainTag::Conversation,
            label: format!("Session {}", session.id),
            file_path: file_path.to_string(),
            span: None,
            signature: None,
            doc_comment: None,
            identifiers: vec![],
            bm25_text: format!("session {} conversation", session.id),
            symbol_vec: None,
            ast_hash: None,
            temporal: Some(TemporalMeta::at(session_ts)),
            attributes: HashMap::new(),
            intent_summary: None,
        });

        // Link sequential sessions
        if let Some(ref prev_id) = prev_session_id {
            edges.push(UniversalEdge::new(
                prev_id.clone(),
                session_node_id.clone(),
                UniversalEdgeType::FollowedBy,
                Confidence::Certain,
            ));
        }
        prev_session_id = Some(session_node_id.clone());

        // Process messages
        for (msg_idx, msg) in session.messages.iter().enumerate() {
            let msg_node_id = make_id(&format!("msg:{}:{}", session.id, msg_idx));
            let msg_ts = msg
                .timestamp
                .as_deref()
                .and_then(parse_timestamp)
                .unwrap_or(session_ts);

            // Message node
            let msg_preview = if msg.content.len() > 100 {
                format!("{}...", &msg.content[..100])
            } else {
                msg.content.clone()
            };

            nodes.push(UniversalNode {
                id: msg_node_id.clone(),
                kind: UniversalKind::Message,
                domain: DomainTag::Conversation,
                label: msg_preview,
                file_path: file_path.to_string(),
                span: None,
                signature: Some(format!("[{}] {}", msg.role, session.id)),
                doc_comment: None,
                identifiers: extract_keywords(&msg.content),
                bm25_text: format!("{} {} {}", msg.role, session.id, msg.content),
                symbol_vec: None,
                ast_hash: None,
                temporal: Some(TemporalMeta::at(msg_ts)),
                attributes: {
                    let mut m = HashMap::new();
                    m.insert(
                        "role".into(),
                        AttributeValue::String(msg.role.clone()),
                    );
                    m.insert(
                        "turn_index".into(),
                        AttributeValue::Int(msg_idx as i64),
                    );
                    m
                },
                intent_summary: None,
            });

            // Session contains message
            edges.push(UniversalEdge::new(
                session_node_id.clone(),
                msg_node_id.clone(),
                UniversalEdgeType::Contain,
                Confidence::Certain,
            ));

            // Extract entities from user messages
            if msg.role == "user" {
                // Extract preferences
                for pref in extract_preferences(&msg.content) {
                    let pref_id = make_id(&format!(
                        "pref:{}:{}:{}",
                        session.id, msg_idx, pref.subject
                    ));
                    nodes.push(UniversalNode {
                        id: pref_id.clone(),
                        kind: UniversalKind::Preference,
                        domain: DomainTag::Conversation,
                        label: format!("{}: {}", pref.subject, pref.sentiment),
                        file_path: file_path.to_string(),
                        span: None,
                        signature: None,
                        doc_comment: Some(pref.context.clone()),
                        identifiers: vec![pref.subject.clone()],
                        bm25_text: format!(
                            "preference {} {} {}",
                            pref.subject, pref.sentiment, pref.context
                        ),
                        symbol_vec: None,
                        ast_hash: None,
                        temporal: Some(TemporalMeta::at(msg_ts)),
                        attributes: {
                            let mut m = HashMap::new();
                            m.insert(
                                "sentiment".into(),
                                AttributeValue::String(pref.sentiment.clone()),
                            );
                            m
                        },
                        intent_summary: None,
                    });

                    edges.push(UniversalEdge::new(
                        msg_node_id.clone(),
                        pref_id,
                        UniversalEdgeType::ExpressedPreference,
                        Confidence::Likely,
                    ));
                }

                // Extract decisions
                for decision in extract_decisions(&msg.content) {
                    let dec_id = make_id(&format!(
                        "dec:{}:{}:{}",
                        session.id, msg_idx, decision
                    ));
                    nodes.push(UniversalNode {
                        id: dec_id.clone(),
                        kind: UniversalKind::Decision,
                        domain: DomainTag::Conversation,
                        label: decision.clone(),
                        file_path: file_path.to_string(),
                        span: None,
                        signature: None,
                        doc_comment: None,
                        identifiers: extract_keywords(&decision),
                        bm25_text: format!("decision {}", decision),
                        symbol_vec: None,
                        ast_hash: None,
                        temporal: Some(TemporalMeta::at(msg_ts)),
                        attributes: HashMap::new(),
                        intent_summary: None,
                    });

                    edges.push(UniversalEdge::new(
                        msg_node_id.clone(),
                        dec_id,
                        UniversalEdgeType::DecidedOn,
                        Confidence::Likely,
                    ));
                }

                // Extract named entities and create entity nodes
                for entity in extract_named_entities(&msg.content) {
                    let entity_id =
                        make_id(&format!("entity:{}:{}", file_path, entity.name));
                    // Only create entity node if not already created (deduplicate by checking label)
                    if !nodes.iter().any(|n| {
                        n.kind == UniversalKind::Entity && n.label == entity.name
                    }) {
                        nodes.push(UniversalNode {
                            id: entity_id.clone(),
                            kind: UniversalKind::Entity,
                            domain: DomainTag::Conversation,
                            label: entity.name.clone(),
                            file_path: file_path.to_string(),
                            span: None,
                            signature: None,
                            doc_comment: None,
                            identifiers: vec![entity.name.clone()],
                            bm25_text: format!(
                                "entity {} {}",
                                entity.name, entity.entity_type
                            ),
                            symbol_vec: None,
                            ast_hash: None,
                            temporal: Some(TemporalMeta::at(msg_ts)),
                            attributes: {
                                let mut m = HashMap::new();
                                m.insert(
                                    "entity_type".into(),
                                    AttributeValue::String(entity.entity_type.clone()),
                                );
                                m
                            },
                            intent_summary: None,
                        });
                    }

                    edges.push(UniversalEdge::new(
                        msg_node_id.clone(),
                        entity_id,
                        UniversalEdgeType::Mentions,
                        Confidence::Likely,
                    ));
                }
            }
        }
    }

    Ok(ParseResult { nodes, edges })
}

// ── Entity Extraction (Rule-Based) ──────────────────────────────────────────

#[derive(Debug, Clone)]
struct ExtractedPreference {
    subject: String,
    sentiment: String,
    context: String,
}

#[derive(Debug, Clone)]
struct ExtractedEntity {
    name: String,
    entity_type: String,
}

/// Extract preferences from text using pattern matching.
fn extract_preferences(text: &str) -> Vec<ExtractedPreference> {
    static PATTERNS: OnceLock<Vec<(Regex, &str)>> = OnceLock::new();
    let patterns = PATTERNS.get_or_init(|| {
        vec![
            (
                Regex::new(r"(?i)I (?:really )?(?:love|enjoy|like|prefer|adore)\s+(.+?)(?:\.|,|!|$)")
                    .unwrap(),
                "positive",
            ),
            (
                Regex::new(r"(?i)I (?:really )?(?:hate|dislike|can't stand|don't like|detest)\s+(.+?)(?:\.|,|!|$)")
                    .unwrap(),
                "negative",
            ),
            (
                Regex::new(r"(?i)I prefer\s+(.+?)\s+(?:over|to|rather than)\s+(.+?)(?:\.|,|!|$)")
                    .unwrap(),
                "comparative",
            ),
            (
                Regex::new(r"(?i)my (?:favorite|favourite)\s+(?:\w+\s+)?(?:is|are)\s+(.+?)(?:\.|,|!|$)")
                    .unwrap(),
                "positive",
            ),
        ]
    });

    let mut prefs = Vec::new();
    for (re, sentiment) in patterns.iter() {
        for cap in re.captures_iter(text) {
            if let Some(m) = cap.get(1) {
                let subject = m.as_str().trim().to_string();
                if !subject.is_empty() && subject.len() < 100 {
                    prefs.push(ExtractedPreference {
                        subject,
                        sentiment: sentiment.to_string(),
                        context: text.to_string(),
                    });
                }
            }
        }
    }
    prefs
}

/// Extract decisions from text using pattern matching.
fn extract_decisions(text: &str) -> Vec<String> {
    static PATTERNS: OnceLock<Vec<Regex>> = OnceLock::new();
    let patterns = PATTERNS.get_or_init(|| {
        vec![
            Regex::new(r"(?i)(?:let's|let us) (?:go with|use|try|do|pick)\s+(.+?)(?:\.|,|!|$)")
                .unwrap(),
            Regex::new(
                r"(?i)I (?:decided|choose|chose|picked|selected|went with)\s+(.+?)(?:\.|,|!|$)",
            )
            .unwrap(),
            Regex::new(r"(?i)(?:we should|I'll|I will|we'll)\s+(.+?)(?:\.|,|!|$)").unwrap(),
        ]
    });

    let mut decisions = Vec::new();
    for re in patterns.iter() {
        for cap in re.captures_iter(text) {
            if let Some(m) = cap.get(1) {
                let decision = m.as_str().trim().to_string();
                if !decision.is_empty() && decision.len() < 150 {
                    decisions.push(decision);
                }
            }
        }
    }
    decisions
}

/// Extract named entities from text using simple heuristics.
fn extract_named_entities(text: &str) -> Vec<ExtractedEntity> {
    static LOCATION_RE: OnceLock<Regex> = OnceLock::new();
    static PERSON_RE: OnceLock<Regex> = OnceLock::new();

    let location_re = LOCATION_RE.get_or_init(|| {
        Regex::new(r"(?i)(?:moved to|live in|living in|based in|from|visited|traveling to|went to|in)\s+([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)").unwrap()
    });

    let person_re = PERSON_RE.get_or_init(|| {
        Regex::new(
            r"(?:my (?:friend|sister|brother|wife|husband|partner|colleague|boss|mom|dad|mother|father)\s+)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)",
        )
        .unwrap()
    });

    let mut entities = Vec::new();
    let mut seen = std::collections::HashSet::new();

    // Locations
    for cap in location_re.captures_iter(text) {
        if let Some(m) = cap.get(1) {
            let name = m.as_str().to_string();
            if !seen.contains(&name) && !is_common_word(&name) {
                seen.insert(name.clone());
                entities.push(ExtractedEntity {
                    name,
                    entity_type: "location".to_string(),
                });
            }
        }
    }

    // People (look for capitalized words after relationship indicators)
    for cap in person_re.captures_iter(text) {
        if let Some(m) = cap.get(1) {
            let name = m.as_str().to_string();
            if !seen.contains(&name)
                && !is_common_word(&name)
                && name.len() > 1
                && cap.get(0).map_or(false, |full| {
                    full.as_str().starts_with("my ") || name.contains(' ')
                })
            {
                seen.insert(name.clone());
                entities.push(ExtractedEntity {
                    name,
                    entity_type: "person".to_string(),
                });
            }
        }
    }

    entities
}

/// Extract significant keywords from text for identifiers field.
fn extract_keywords(text: &str) -> Vec<String> {
    static STOP_WORDS: OnceLock<std::collections::HashSet<&str>> = OnceLock::new();
    let stop_words = STOP_WORDS.get_or_init(|| {
        [
            "the", "a", "an", "is", "are", "was", "were", "be", "been", "being", "have", "has",
            "had", "do", "does", "did", "will", "would", "shall", "should", "may", "might",
            "must", "can", "could", "i", "you", "he", "she", "it", "we", "they", "me", "him",
            "her", "us", "them", "my", "your", "his", "its", "our", "their", "this", "that",
            "these", "those", "what", "which", "who", "whom", "when", "where", "why", "how",
            "not", "no", "nor", "but", "or", "and", "if", "then", "else", "so", "just", "too",
            "very", "really", "also", "of", "in", "on", "at", "to", "for", "with", "by",
            "from", "about", "into", "through", "during", "before", "after", "above", "below",
            "up", "down", "out", "off", "over", "under", "again", "further", "once", "here",
            "there", "all", "both", "each", "more", "most", "other", "some", "such", "than",
            "as",
        ]
        .into_iter()
        .collect()
    });

    text.split_whitespace()
        .map(|w| w.trim_matches(|c: char| !c.is_alphanumeric()).to_lowercase())
        .filter(|w| w.len() > 2 && !stop_words.contains(w.as_str()))
        .take(20)
        .collect()
}

/// Check if a word is too common to be a named entity.
fn is_common_word(word: &str) -> bool {
    static COMMON: OnceLock<std::collections::HashSet<&str>> = OnceLock::new();
    let common = COMMON.get_or_init(|| {
        [
            "The", "This", "That", "These", "Those", "Here", "There", "When", "Where", "What",
            "Which", "Who", "How", "Some", "Any", "All", "Most", "Much", "Many", "Few",
            "Other", "Another", "Each", "Every", "Both", "Such", "Rather", "Just", "Only",
            "Also", "Then", "Now", "Today", "Tomorrow", "Yesterday", "Monday", "Tuesday",
            "Wednesday", "Thursday", "Friday", "Saturday", "Sunday", "January", "February",
            "March", "April", "May", "June", "July", "August", "September", "October",
            "November", "December", "But", "And", "However", "Although", "Since", "Because",
            "So", "Yet", "Still",
        ]
        .into_iter()
        .collect()
    });
    common.contains(word)
}

/// Generate a deterministic ID from a string.
fn make_id(input: &str) -> String {
    let hash = blake3::hash(input.as_bytes());
    hash.to_hex()[..32].to_string()
}

/// Parse an ISO timestamp to Unix epoch seconds.
fn parse_timestamp(ts: &str) -> Option<i64> {
    // Simple parser for ISO 8601 timestamps like "2026-01-15T10:30:00Z"
    // Just extract the date and approximate
    let parts: Vec<&str> = ts.split('T').collect();
    if parts.is_empty() {
        return None;
    }
    let date_parts: Vec<&str> = parts[0].split('-').collect();
    if date_parts.len() != 3 {
        return None;
    }
    let year: i64 = date_parts[0].parse().ok()?;
    let month: i64 = date_parts[1].parse().ok()?;
    let day: i64 = date_parts[2].parse().ok()?;

    // Approximate Unix timestamp (good enough for ordering)
    let days = (year - 1970) * 365 + (month - 1) * 30 + day;
    Some(days * 86400)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_test_input() -> ConversationInput {
        ConversationInput {
            sessions: vec![
                SessionInput {
                    id: "session_001".to_string(),
                    timestamp: Some("2026-01-15T10:30:00Z".to_string()),
                    messages: vec![
                        MessageInput {
                            role: "user".to_string(),
                            content: "I just moved to Denver from Seattle. I really love hiking."
                                .to_string(),
                            timestamp: Some("2026-01-15T10:30:00Z".to_string()),
                        },
                        MessageInput {
                            role: "assistant".to_string(),
                            content: "Welcome to Denver! There are great trails nearby."
                                .to_string(),
                            timestamp: Some("2026-01-15T10:30:05Z".to_string()),
                        },
                    ],
                },
                SessionInput {
                    id: "session_002".to_string(),
                    timestamp: Some("2026-01-16T14:00:00Z".to_string()),
                    messages: vec![MessageInput {
                        role: "user".to_string(),
                        content: "I hate cold weather. Let's go with the indoor gym plan."
                            .to_string(),
                        timestamp: Some("2026-01-16T14:00:00Z".to_string()),
                    }],
                },
            ],
        }
    }

    #[test]
    fn test_parse_conversation_basic() {
        let input = make_test_input();
        let result = parse_conversation("test.conversation.json", &input).unwrap();

        // Should have session nodes + message nodes + entity/preference nodes
        assert!(!result.nodes.is_empty());
        assert!(!result.edges.is_empty());

        // Check session nodes exist
        let sessions: Vec<_> = result
            .nodes
            .iter()
            .filter(|n| n.kind == UniversalKind::Topic)
            .collect();
        assert_eq!(sessions.len(), 2);

        // Check message nodes
        let messages: Vec<_> = result
            .nodes
            .iter()
            .filter(|n| n.kind == UniversalKind::Message)
            .collect();
        assert_eq!(messages.len(), 3);

        // All nodes should be in Conversation domain
        assert!(result
            .nodes
            .iter()
            .all(|n| n.domain == DomainTag::Conversation));
    }

    #[test]
    fn test_extract_preferences() {
        let prefs = extract_preferences("I really love hiking and I hate cold weather.");
        assert!(!prefs.is_empty());

        let positive: Vec<_> = prefs.iter().filter(|p| p.sentiment == "positive").collect();
        let negative: Vec<_> = prefs.iter().filter(|p| p.sentiment == "negative").collect();

        assert!(!positive.is_empty());
        assert!(positive[0].subject.contains("hiking"));
        assert!(!negative.is_empty());
        assert!(negative[0].subject.contains("cold weather"));
    }

    #[test]
    fn test_extract_preferences_favorite() {
        let prefs = extract_preferences("My favorite color is blue.");
        assert!(!prefs.is_empty());
        assert!(prefs[0].subject.contains("blue"));
    }

    #[test]
    fn test_extract_decisions() {
        let decisions = extract_decisions("Let's go with the indoor gym plan.");
        assert!(!decisions.is_empty());
        assert!(decisions[0].contains("indoor gym plan"));
    }

    #[test]
    fn test_extract_decisions_chose() {
        let decisions = extract_decisions("I decided to use Python for the project.");
        assert!(!decisions.is_empty());
        assert!(decisions[0].contains("Python"));
    }

    #[test]
    fn test_extract_named_entities_location() {
        let entities = extract_named_entities("I moved to Denver from Seattle.");
        let locations: Vec<_> = entities
            .iter()
            .filter(|e| e.entity_type == "location")
            .collect();
        assert!(locations.iter().any(|e| e.name == "Denver"));
        assert!(locations.iter().any(|e| e.name == "Seattle"));
    }

    #[test]
    fn test_extract_named_entities_person() {
        let entities = extract_named_entities("my friend Alice Smith told me about it.");
        let people: Vec<_> = entities
            .iter()
            .filter(|e| e.entity_type == "person")
            .collect();
        assert!(!people.is_empty());
        assert!(people.iter().any(|e| e.name.contains("Alice")));
    }

    #[test]
    fn test_extract_keywords() {
        let kw = extract_keywords("I really love hiking in the mountains near Denver");
        assert!(kw.contains(&"love".to_string()));
        assert!(kw.contains(&"hiking".to_string()));
        assert!(kw.contains(&"mountains".to_string()));
        assert!(kw.contains(&"denver".to_string()));
        // Stop words should be excluded
        assert!(!kw.contains(&"the".to_string()));
        assert!(!kw.contains(&"in".to_string()));
    }

    #[test]
    fn test_session_linking() {
        let input = make_test_input();
        let result = parse_conversation("test.json", &input).unwrap();

        // Should have a FollowedBy edge between sessions
        let followed_by: Vec<_> = result
            .edges
            .iter()
            .filter(|e| e.edge_type == UniversalEdgeType::FollowedBy)
            .collect();
        assert_eq!(followed_by.len(), 1);
    }

    #[test]
    fn test_conversation_parser_trait() {
        let parser = ConversationParser;
        assert_eq!(parser.domain_id(), "conversation");
        assert!(parser.can_parse("session.conversation.json", &[]));
        assert!(parser.can_parse("data.chat.json", &[]));
        assert!(!parser.can_parse("code.rs", &[]));
    }

    #[test]
    fn test_json_sniffing() {
        let parser = ConversationParser;
        let json_start = b"{\"sessions\": [{\"id\": \"s1\"";
        assert!(parser.can_parse("data.json", json_start));
    }

    #[test]
    fn test_parse_from_json_bytes() {
        let json = r#"{
            "sessions": [{
                "id": "s1",
                "timestamp": "2026-01-15T10:00:00Z",
                "messages": [
                    {"role": "user", "content": "I love pizza"},
                    {"role": "assistant", "content": "Great choice!"}
                ]
            }]
        }"#;

        let parser = ConversationParser;
        let config = TrevecConfig::default();
        let result = parser
            .parse("test.conversation.json", json.as_bytes(), &config)
            .unwrap();

        assert!(!result.nodes.is_empty());
        // Should have session + 2 messages + preference for pizza
        let prefs: Vec<_> = result
            .nodes
            .iter()
            .filter(|n| n.kind == UniversalKind::Preference)
            .collect();
        assert!(!prefs.is_empty());
    }

    #[test]
    fn test_parse_timestamp() {
        let ts = parse_timestamp("2026-01-15T10:30:00Z");
        assert!(ts.is_some());
        assert!(ts.unwrap() > 0);

        assert!(parse_timestamp("invalid").is_none());
        assert!(parse_timestamp("").is_none());
    }

    #[test]
    fn test_empty_conversation() {
        let input = ConversationInput {
            sessions: vec![],
        };
        let result = parse_conversation("empty.json", &input).unwrap();
        assert!(result.nodes.is_empty());
        assert!(result.edges.is_empty());
    }

    #[test]
    fn test_entity_deduplication() {
        let input = ConversationInput {
            sessions: vec![SessionInput {
                id: "s1".to_string(),
                timestamp: None,
                messages: vec![
                    MessageInput {
                        role: "user".to_string(),
                        content: "I moved to Denver. Denver is great.".to_string(),
                        timestamp: None,
                    },
                ],
            }],
        };
        let result = parse_conversation("test.json", &input).unwrap();
        // Denver entity should only appear once
        let denver_entities: Vec<_> = result
            .nodes
            .iter()
            .filter(|n| n.kind == UniversalKind::Entity && n.label == "Denver")
            .collect();
        assert!(denver_entities.len() <= 1);
    }
}
