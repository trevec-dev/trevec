//! Universal Context Graph data model.
//!
//! Extends the code-specific `CodeNode` with domain-agnostic types that support
//! conversations, documents, structured data, and cross-domain relationships.
//! All extensions are additive — existing `CodeNode` behavior is unchanged.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::fmt;

use crate::model::{CodeNode, Confidence, EdgeType, NodeId, NodeKind, Span};

// ── Domain Tag ───────────────────────────────────────────────────────────────

/// Tags which domain a node belongs to, enabling cross-domain queries.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DomainTag {
    /// Source code (existing CodeNode behavior)
    Code,
    /// Chat sessions, messages, preferences, decisions
    Conversation,
    /// PDF, Markdown, HTML — sections, headings, facts
    Document,
    /// JSON, CSV, API responses — entities, records, transactions
    Structured,
    /// Brain-generated observations about code changes
    Observation,
}

impl fmt::Display for DomainTag {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            DomainTag::Code => write!(f, "code"),
            DomainTag::Conversation => write!(f, "conversation"),
            DomainTag::Document => write!(f, "document"),
            DomainTag::Structured => write!(f, "structured"),
            DomainTag::Observation => write!(f, "observation"),
        }
    }
}

impl DomainTag {
    pub fn from_str_loose(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "code" => Some(DomainTag::Code),
            "conversation" | "conv" | "chat" => Some(DomainTag::Conversation),
            "document" | "doc" => Some(DomainTag::Document),
            "structured" | "data" => Some(DomainTag::Structured),
            "observation" | "obs" => Some(DomainTag::Observation),
            _ => None,
        }
    }
}

// ── Universal Node Kind ──────────────────────────────────────────────────────

/// Superset of `NodeKind` that covers all domains.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum UniversalKind {
    // ── Code domain (mirrors NodeKind exactly) ───────────────────────────
    Function,
    Method,
    Class,
    Interface,
    Module,
    Struct,
    Enum,
    Trait,
    Macro,
    Type,
    DocSection,

    // ── Conversation domain ──────────────────────────────────────────────
    Message,
    Topic,
    Preference,
    Decision,
    Summary,

    // ── Document domain ──────────────────────────────────────────────────
    Section,
    Heading,
    Fact,
    Citation,

    // ── Structured data domain ───────────────────────────────────────────
    Entity,
    Record,
    Transaction,
    Event,
    Rule,

    // ── Observation domain (Brain-generated) ─────────────────────────────
    ObservationEntry,
    Reflection,
}

impl fmt::Display for UniversalKind {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Function => "function",
            Self::Method => "method",
            Self::Class => "class",
            Self::Interface => "interface",
            Self::Module => "module",
            Self::Struct => "struct",
            Self::Enum => "enum",
            Self::Trait => "trait",
            Self::Macro => "macro",
            Self::Type => "type",
            Self::DocSection => "doc_section",
            Self::Message => "message",
            Self::Topic => "topic",
            Self::Preference => "preference",
            Self::Decision => "decision",
            Self::Summary => "summary",
            Self::Section => "section",
            Self::Heading => "heading",
            Self::Fact => "fact",
            Self::Citation => "citation",
            Self::Entity => "entity",
            Self::Record => "record",
            Self::Transaction => "transaction",
            Self::Event => "event",
            Self::Rule => "rule",
            Self::ObservationEntry => "observation",
            Self::Reflection => "reflection",
        };
        write!(f, "{s}")
    }
}

impl From<NodeKind> for UniversalKind {
    fn from(kind: NodeKind) -> Self {
        match kind {
            NodeKind::Function => Self::Function,
            NodeKind::Method => Self::Method,
            NodeKind::Class => Self::Class,
            NodeKind::Interface => Self::Interface,
            NodeKind::Module => Self::Module,
            NodeKind::Struct => Self::Struct,
            NodeKind::Enum => Self::Enum,
            NodeKind::Trait => Self::Trait,
            NodeKind::Macro => Self::Macro,
            NodeKind::Type => Self::Type,
            NodeKind::DocSection => Self::DocSection,
        }
    }
}

impl UniversalKind {
    /// Try to convert back to the code-specific `NodeKind`.
    pub fn to_node_kind(self) -> Option<NodeKind> {
        match self {
            Self::Function => Some(NodeKind::Function),
            Self::Method => Some(NodeKind::Method),
            Self::Class => Some(NodeKind::Class),
            Self::Interface => Some(NodeKind::Interface),
            Self::Module => Some(NodeKind::Module),
            Self::Struct => Some(NodeKind::Struct),
            Self::Enum => Some(NodeKind::Enum),
            Self::Trait => Some(NodeKind::Trait),
            Self::Macro => Some(NodeKind::Macro),
            Self::Type => Some(NodeKind::Type),
            Self::DocSection => Some(NodeKind::DocSection),
            _ => None,
        }
    }

    /// Whether this kind belongs to the code domain.
    pub fn is_code(&self) -> bool {
        self.to_node_kind().is_some()
    }
}

// ── Temporal Metadata ────────────────────────────────────────────────────────

/// Temporal metadata for versioned nodes and edges.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TemporalMeta {
    /// Unix epoch seconds — when this node/edge was created.
    pub t_created: i64,
    /// Validity window: (start, optional end). Open-ended if end is None.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub t_valid: Option<(i64, Option<i64>)>,
    /// When this was superseded by newer information.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub t_superseded: Option<i64>,
    /// Source context: git commit hash, amendment ID, transaction ref, etc.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_context: Option<String>,
}

impl TemporalMeta {
    /// Create a new temporal meta with the current timestamp.
    pub fn now() -> Self {
        Self {
            t_created: std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap_or_default()
                .as_secs() as i64,
            t_valid: None,
            t_superseded: None,
            source_context: None,
        }
    }

    /// Create with a specific creation timestamp.
    pub fn at(t_created: i64) -> Self {
        Self {
            t_created,
            t_valid: None,
            t_superseded: None,
            source_context: None,
        }
    }

    /// Whether this node/edge is currently valid (not superseded, validity window open).
    pub fn is_active(&self, now: i64) -> bool {
        if self.t_superseded.is_some() {
            return false;
        }
        match self.t_valid {
            Some((start, Some(end))) => now >= start && now <= end,
            Some((start, None)) => now >= start,
            None => true,
        }
    }
}

// ── Attribute Values ─────────────────────────────────────────────────────────

/// Extensible attribute value for domain-specific metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(untagged)]
pub enum AttributeValue {
    String(String),
    Int(i64),
    Float(f64),
    Bool(bool),
    List(Vec<String>),
}

impl fmt::Display for AttributeValue {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::String(s) => write!(f, "{s}"),
            Self::Int(i) => write!(f, "{i}"),
            Self::Float(v) => write!(f, "{v}"),
            Self::Bool(b) => write!(f, "{b}"),
            Self::List(l) => write!(f, "[{}]", l.join(", ")),
        }
    }
}

// ── Brain Enrichment Data ────────────────────────────────────────────────────

/// Structured intent summary generated by the Brain's intent summarizer.
#[derive(Debug, Clone, Default, PartialEq, Serialize, Deserialize)]
pub struct IntentSummary {
    /// What this code does (one sentence).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub purpose: Option<String>,
    /// Parameter names and what they represent.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub inputs: Option<String>,
    /// Return value and what it represents.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub outputs: Option<String>,
    /// Mutations, I/O, or state changes.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub side_effects: Option<String>,
    /// Domain concepts this implements (e.g., "JWT authentication").
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub related_concepts: Vec<String>,
    /// What can go wrong.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub error_cases: Option<String>,
    /// The model and prompt version that generated this summary.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub model_version: Option<String>,
    /// The ast_hash at generation time (for cache invalidation).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub source_hash: Option<String>,
}

impl IntentSummary {
    /// Build a single-line text summary for bm25_text enrichment.
    pub fn to_bm25_text(&self) -> String {
        let mut parts = Vec::new();
        if let Some(ref p) = self.purpose {
            parts.push(p.clone());
        }
        if !self.related_concepts.is_empty() {
            parts.push(self.related_concepts.join(" "));
        }
        if let Some(ref s) = self.side_effects {
            parts.push(s.clone());
        }
        parts.join(" ")
    }

    /// Whether this summary has any content.
    pub fn is_empty(&self) -> bool {
        self.purpose.is_none()
            && self.inputs.is_none()
            && self.outputs.is_none()
            && self.related_concepts.is_empty()
    }
}

// ── Universal Node ───────────────────────────────────────────────────────────

/// A domain-agnostic node in the Universal Context Graph.
/// Superset of `CodeNode` — all code-specific fields are preserved.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct UniversalNode {
    pub id: NodeId,
    pub kind: UniversalKind,
    pub domain: DomainTag,
    pub label: String,
    pub file_path: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub span: Option<Span>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub signature: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub doc_comment: Option<String>,
    #[serde(default, skip_serializing_if = "Vec::is_empty")]
    pub identifiers: Vec<String>,
    pub bm25_text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub symbol_vec: Option<Vec<f32>>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub ast_hash: Option<String>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temporal: Option<TemporalMeta>,
    #[serde(default, skip_serializing_if = "HashMap::is_empty")]
    pub attributes: HashMap<String, AttributeValue>,
    /// Brain-generated intent summary (async enrichment, may be None).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub intent_summary: Option<IntentSummary>,
}

impl From<CodeNode> for UniversalNode {
    fn from(node: CodeNode) -> Self {
        Self {
            id: node.id,
            kind: UniversalKind::from(node.kind),
            domain: DomainTag::Code,
            label: node.name,
            file_path: node.file_path,
            span: Some(node.span),
            signature: Some(node.signature),
            doc_comment: node.doc_comment,
            identifiers: node.identifiers,
            bm25_text: node.bm25_text,
            symbol_vec: node.symbol_vec,
            ast_hash: Some(node.ast_hash),
            temporal: None,
            attributes: HashMap::new(),
            intent_summary: None,
        }
    }
}

impl UniversalNode {
    /// Try to convert back to a `CodeNode`. Only succeeds for Code-domain nodes.
    pub fn to_code_node(&self) -> Option<CodeNode> {
        if self.domain != DomainTag::Code {
            return None;
        }
        let kind = self.kind.to_node_kind()?;
        Some(CodeNode {
            id: self.id.clone(),
            kind,
            file_path: self.file_path.clone(),
            span: self.span.clone()?,
            name: self.label.clone(),
            signature: self.signature.clone().unwrap_or_default(),
            doc_comment: self.doc_comment.clone(),
            identifiers: self.identifiers.clone(),
            bm25_text: self.bm25_text.clone(),
            symbol_vec: self.symbol_vec.clone(),
            ast_hash: self.ast_hash.clone().unwrap_or_default(),
        })
    }

    /// Build enriched bm25_text including Brain intent summary if available.
    pub fn enriched_bm25_text(&self) -> String {
        if let Some(ref summary) = self.intent_summary {
            let extra = summary.to_bm25_text();
            if extra.is_empty() {
                self.bm25_text.clone()
            } else {
                format!("{} {}", self.bm25_text, extra)
            }
        } else {
            self.bm25_text.clone()
        }
    }
}

// ── Extended Edge Types ──────────────────────────────────────────────────────

/// Extended edge types for the Universal Context Graph.
/// Includes all existing `EdgeType` variants plus new cross-domain types.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub enum UniversalEdgeType {
    // ── Existing (mirrors EdgeType) ──────────────────────────────────────
    Import,
    Call,
    Contain,
    Implement,
    Inherit,
    Discussed,
    Triggered,
    Reference,

    // ── Conversation domain ──────────────────────────────────────────────
    ExpressedPreference,
    DecidedOn,
    Mentions,
    FollowedBy,

    // ── Temporal ─────────────────────────────────────────────────────────
    Supersedes,
    ContradictedBy,

    // ── Cross-domain ─────────────────────────────────────────────────────
    CrossDomainLink,
    SameAs,

    // ── Structured data ──────────────────────────────────────────────────
    BelongsTo,
    TransfersTo,
    Prerequisite,

    // ── Brain-generated ──────────────────────────────────────────────────
    PredictedCall,
    PredictedImport,
    ObservedIn,
    SummarizesObservations,
}

impl fmt::Display for UniversalEdgeType {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        let s = match self {
            Self::Import => "import",
            Self::Call => "call",
            Self::Contain => "contain",
            Self::Implement => "implement",
            Self::Inherit => "inherit",
            Self::Discussed => "discussed",
            Self::Triggered => "triggered",
            Self::Reference => "reference",
            Self::ExpressedPreference => "expressed_preference",
            Self::DecidedOn => "decided_on",
            Self::Mentions => "mentions",
            Self::FollowedBy => "followed_by",
            Self::Supersedes => "supersedes",
            Self::ContradictedBy => "contradicted_by",
            Self::CrossDomainLink => "cross_domain_link",
            Self::SameAs => "same_as",
            Self::BelongsTo => "belongs_to",
            Self::TransfersTo => "transfers_to",
            Self::Prerequisite => "prerequisite",
            Self::PredictedCall => "predicted_call",
            Self::PredictedImport => "predicted_import",
            Self::ObservedIn => "observed_in",
            Self::SummarizesObservations => "summarizes_observations",
        };
        write!(f, "{s}")
    }
}

impl From<EdgeType> for UniversalEdgeType {
    fn from(et: EdgeType) -> Self {
        match et {
            EdgeType::Import => Self::Import,
            EdgeType::Call => Self::Call,
            EdgeType::Contain => Self::Contain,
            EdgeType::Implement => Self::Implement,
            EdgeType::Inherit => Self::Inherit,
            EdgeType::Discussed => Self::Discussed,
            EdgeType::Triggered => Self::Triggered,
            EdgeType::Reference => Self::Reference,
        }
    }
}

impl UniversalEdgeType {
    /// Convert back to the code-specific `EdgeType` if applicable.
    pub fn to_edge_type(self) -> Option<EdgeType> {
        match self {
            Self::Import => Some(EdgeType::Import),
            Self::Call => Some(EdgeType::Call),
            Self::Contain => Some(EdgeType::Contain),
            Self::Implement => Some(EdgeType::Implement),
            Self::Inherit => Some(EdgeType::Inherit),
            Self::Discussed => Some(EdgeType::Discussed),
            Self::Triggered => Some(EdgeType::Triggered),
            Self::Reference => Some(EdgeType::Reference),
            _ => None,
        }
    }

    /// Whether this is a Brain-predicted edge (lower confidence by default).
    pub fn is_predicted(&self) -> bool {
        matches!(self, Self::PredictedCall | Self::PredictedImport)
    }
}

/// An edge in the Universal Context Graph with optional temporal metadata.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct UniversalEdge {
    pub src_id: NodeId,
    pub dst_id: NodeId,
    pub edge_type: UniversalEdgeType,
    pub confidence: Confidence,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub temporal: Option<TemporalMeta>,
}

impl UniversalEdge {
    pub fn new(
        src_id: NodeId,
        dst_id: NodeId,
        edge_type: UniversalEdgeType,
        confidence: Confidence,
    ) -> Self {
        Self {
            src_id,
            dst_id,
            edge_type,
            confidence,
            temporal: None,
        }
    }

    pub fn with_temporal(mut self, temporal: TemporalMeta) -> Self {
        self.temporal = Some(temporal);
        self
    }
}

// ── Retention Scoring ────────────────────────────────────────────────────────

/// Ebbinghaus-inspired retention score for Brain-enriched data.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct RetentionScore {
    /// The computed score (0.0 to ~2.0). Below threshold → eligible for pruning.
    pub score: f64,
    /// Number of times this node appeared in retrieval results.
    pub access_count: u32,
    /// Unix epoch of last access.
    pub last_access: i64,
    /// Whether this is promoted to permanent (no decay).
    pub permanent: bool,
}

impl RetentionScore {
    /// Compute retention score using Ebbinghaus decay formula.
    ///
    /// `score = importance * exp(-lambda * delta_t) * (1 + 0.2 * access_count)`
    ///
    /// Where lambda = ln(2) / half_life_days
    pub fn compute(
        importance: f64,
        access_count: u32,
        last_access: i64,
        now: i64,
        half_life_days: f64,
    ) -> Self {
        let lambda = (2.0_f64).ln() / (half_life_days * 86400.0);
        let delta_t = (now - last_access).max(0) as f64;
        let score = importance * (-lambda * delta_t).exp() * (1.0 + 0.2 * access_count as f64);

        Self {
            score,
            access_count,
            last_access,
            permanent: false,
        }
    }

    /// Whether this should be pruned (score below threshold).
    pub fn should_prune(&self, threshold: f64) -> bool {
        !self.permanent && self.score < threshold
    }

    /// Whether this should be refreshed (Brain should re-enrich).
    pub fn should_refresh(&self, low: f64, high: f64) -> bool {
        !self.permanent && self.score >= low && self.score < high
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_domain_tag_display() {
        assert_eq!(DomainTag::Code.to_string(), "code");
        assert_eq!(DomainTag::Conversation.to_string(), "conversation");
        assert_eq!(DomainTag::Document.to_string(), "document");
        assert_eq!(DomainTag::Structured.to_string(), "structured");
        assert_eq!(DomainTag::Observation.to_string(), "observation");
    }

    #[test]
    fn test_domain_tag_from_str() {
        assert_eq!(DomainTag::from_str_loose("code"), Some(DomainTag::Code));
        assert_eq!(
            DomainTag::from_str_loose("chat"),
            Some(DomainTag::Conversation)
        );
        assert_eq!(
            DomainTag::from_str_loose("doc"),
            Some(DomainTag::Document)
        );
        assert_eq!(DomainTag::from_str_loose("xyz"), None);
    }

    #[test]
    fn test_universal_kind_from_node_kind() {
        assert_eq!(
            UniversalKind::from(NodeKind::Function),
            UniversalKind::Function
        );
        assert_eq!(UniversalKind::from(NodeKind::Class), UniversalKind::Class);
        assert!(UniversalKind::Function.is_code());
        assert!(!UniversalKind::Message.is_code());
    }

    #[test]
    fn test_universal_kind_roundtrip() {
        for kind in [
            NodeKind::Function,
            NodeKind::Method,
            NodeKind::Class,
            NodeKind::Trait,
            NodeKind::Module,
        ] {
            let uk = UniversalKind::from(kind);
            assert_eq!(uk.to_node_kind(), Some(kind));
        }
    }

    #[test]
    fn test_universal_kind_non_code() {
        assert_eq!(UniversalKind::Message.to_node_kind(), None);
        assert_eq!(UniversalKind::Entity.to_node_kind(), None);
        assert_eq!(UniversalKind::Section.to_node_kind(), None);
    }

    #[test]
    fn test_temporal_meta_now() {
        let tm = TemporalMeta::now();
        assert!(tm.t_created > 0);
        assert!(tm.t_valid.is_none());
        assert!(tm.t_superseded.is_none());
        assert!(tm.is_active(tm.t_created));
    }

    #[test]
    fn test_temporal_meta_active() {
        let tm = TemporalMeta {
            t_created: 1000,
            t_valid: Some((1000, Some(2000))),
            t_superseded: None,
            source_context: None,
        };
        assert!(tm.is_active(1500));
        assert!(!tm.is_active(2500)); // after end
        assert!(!tm.is_active(500)); // before start
    }

    #[test]
    fn test_temporal_meta_superseded() {
        let tm = TemporalMeta {
            t_created: 1000,
            t_valid: None,
            t_superseded: Some(2000),
            source_context: None,
        };
        assert!(!tm.is_active(2500));
    }

    #[test]
    fn test_code_node_to_universal_roundtrip() {
        let code_node = CodeNode {
            id: "abc123".to_string(),
            kind: NodeKind::Function,
            file_path: "src/main.rs".to_string(),
            span: Span {
                start_line: 0,
                start_col: 0,
                end_line: 10,
                end_col: 0,
                start_byte: 0,
                end_byte: 200,
            },
            name: "main".to_string(),
            signature: "fn main()".to_string(),
            doc_comment: Some("Entry point".to_string()),
            identifiers: vec!["println".to_string()],
            bm25_text: "src/main.rs fn main() println Entry point".to_string(),
            symbol_vec: None,
            ast_hash: "hash123".to_string(),
        };

        let universal: UniversalNode = code_node.clone().into();
        assert_eq!(universal.domain, DomainTag::Code);
        assert_eq!(universal.kind, UniversalKind::Function);
        assert_eq!(universal.label, "main");
        assert!(universal.intent_summary.is_none());

        let back = universal.to_code_node().unwrap();
        assert_eq!(back.id, code_node.id);
        assert_eq!(back.kind, code_node.kind);
        assert_eq!(back.name, code_node.name);
        assert_eq!(back.signature, code_node.signature);
    }

    #[test]
    fn test_universal_node_non_code_no_roundtrip() {
        let node = UniversalNode {
            id: "msg001".to_string(),
            kind: UniversalKind::Message,
            domain: DomainTag::Conversation,
            label: "Hello world".to_string(),
            file_path: "session_001.json".to_string(),
            span: None,
            signature: None,
            doc_comment: None,
            identifiers: vec![],
            bm25_text: "Hello world".to_string(),
            symbol_vec: None,
            ast_hash: None,
            temporal: Some(TemporalMeta::at(1000)),
            attributes: HashMap::new(),
            intent_summary: None,
        };

        assert!(node.to_code_node().is_none());
    }

    #[test]
    fn test_intent_summary_bm25() {
        let summary = IntentSummary {
            purpose: Some("Authenticates user via JWT".to_string()),
            inputs: Some("request with Bearer token".to_string()),
            outputs: Some("authenticated User".to_string()),
            side_effects: Some("updates last_login timestamp".to_string()),
            related_concepts: vec!["JWT".to_string(), "authentication".to_string()],
            error_cases: Some("expired token".to_string()),
            model_version: Some("qwen2.5-coder-1.5b".to_string()),
            source_hash: Some("abc123".to_string()),
        };

        let text = summary.to_bm25_text();
        assert!(text.contains("Authenticates user via JWT"));
        assert!(text.contains("JWT authentication"));
        assert!(text.contains("updates last_login"));
        assert!(!summary.is_empty());
    }

    #[test]
    fn test_intent_summary_empty() {
        let summary = IntentSummary::default();
        assert!(summary.is_empty());
        assert!(summary.to_bm25_text().is_empty());
    }

    #[test]
    fn test_enriched_bm25_text() {
        let mut node = UniversalNode {
            id: "n1".to_string(),
            kind: UniversalKind::Function,
            domain: DomainTag::Code,
            label: "authenticate".to_string(),
            file_path: "src/auth.rs".to_string(),
            span: None,
            signature: Some("fn authenticate(req: &Request)".to_string()),
            doc_comment: None,
            identifiers: vec![],
            bm25_text: "src/auth.rs fn authenticate".to_string(),
            symbol_vec: None,
            ast_hash: None,
            temporal: None,
            attributes: HashMap::new(),
            intent_summary: None,
        };

        // Without enrichment: just base text
        assert_eq!(node.enriched_bm25_text(), "src/auth.rs fn authenticate");

        // With enrichment: base + intent
        node.intent_summary = Some(IntentSummary {
            purpose: Some("validates JWT tokens".to_string()),
            related_concepts: vec!["JWT".to_string(), "authentication".to_string()],
            ..Default::default()
        });
        let enriched = node.enriched_bm25_text();
        assert!(enriched.contains("src/auth.rs fn authenticate"));
        assert!(enriched.contains("validates JWT tokens"));
        assert!(enriched.contains("JWT authentication"));
    }

    #[test]
    fn test_universal_edge_type_from_edge_type() {
        assert_eq!(
            UniversalEdgeType::from(EdgeType::Call),
            UniversalEdgeType::Call
        );
        assert_eq!(
            UniversalEdgeType::from(EdgeType::Import),
            UniversalEdgeType::Import
        );
    }

    #[test]
    fn test_universal_edge_type_roundtrip() {
        for et in [
            EdgeType::Import,
            EdgeType::Call,
            EdgeType::Contain,
            EdgeType::Implement,
            EdgeType::Inherit,
        ] {
            let uet = UniversalEdgeType::from(et);
            assert_eq!(uet.to_edge_type(), Some(et));
        }
    }

    #[test]
    fn test_predicted_edge_types() {
        assert!(UniversalEdgeType::PredictedCall.is_predicted());
        assert!(UniversalEdgeType::PredictedImport.is_predicted());
        assert!(!UniversalEdgeType::Call.is_predicted());
    }

    #[test]
    fn test_universal_edge_construction() {
        let edge = UniversalEdge::new(
            "src".to_string(),
            "dst".to_string(),
            UniversalEdgeType::CrossDomainLink,
            Confidence::Likely,
        )
        .with_temporal(TemporalMeta::at(1000));

        assert_eq!(edge.edge_type, UniversalEdgeType::CrossDomainLink);
        assert_eq!(edge.confidence, Confidence::Likely);
        assert!(edge.temporal.is_some());
    }

    #[test]
    fn test_retention_score_compute() {
        let now = 1_000_000;
        let last_access = now - 86400; // 1 day ago

        let score = RetentionScore::compute(1.0, 5, last_access, now, 3.0);
        // With half_life=3 days, 1 day should retain ~79% base
        // * (1 + 0.2 * 5) = 2.0 multiplier = ~1.58
        assert!(score.score > 1.0);
        assert!(!score.permanent);
        assert!(!score.should_prune(0.15));
    }

    #[test]
    fn test_retention_score_decay() {
        let now = 1_000_000;
        let old_access = now - 86400 * 30; // 30 days ago

        let score = RetentionScore::compute(0.5, 0, old_access, now, 3.0);
        // 30 days with 3-day half-life → very small score
        assert!(score.score < 0.01);
        assert!(score.should_prune(0.15));
    }

    #[test]
    fn test_retention_score_permanent() {
        let mut score = RetentionScore::compute(0.1, 0, 0, 1_000_000, 3.0);
        score.permanent = true;
        assert!(!score.should_prune(0.15)); // permanent never pruned
    }

    #[test]
    fn test_retention_score_refresh_zone() {
        let now = 1_000_000;
        let access = now - 86400 * 2; // 2 days ago

        let score = RetentionScore::compute(0.5, 1, access, now, 3.0);
        // Should be in the refresh zone (0.15-0.65)
        assert!(score.should_refresh(0.15, 0.65) || score.score >= 0.65);
    }

    #[test]
    fn test_attribute_value_display() {
        assert_eq!(AttributeValue::String("hello".into()).to_string(), "hello");
        assert_eq!(AttributeValue::Int(42).to_string(), "42");
        assert_eq!(AttributeValue::Bool(true).to_string(), "true");
    }

    #[test]
    fn test_attribute_value_serde() {
        let val = AttributeValue::String("test".into());
        let json = serde_json::to_string(&val).unwrap();
        let back: AttributeValue = serde_json::from_str(&json).unwrap();
        assert_eq!(val, back);
    }

    #[test]
    fn test_universal_node_serde_roundtrip() {
        let node = UniversalNode {
            id: "test123".to_string(),
            kind: UniversalKind::Entity,
            domain: DomainTag::Structured,
            label: "Alice Smith".to_string(),
            file_path: "contacts.json".to_string(),
            span: None,
            signature: None,
            doc_comment: None,
            identifiers: vec!["alice".to_string(), "smith".to_string()],
            bm25_text: "Alice Smith contacts".to_string(),
            symbol_vec: Some(vec![0.1, 0.2, 0.3]),
            ast_hash: None,
            temporal: Some(TemporalMeta::at(1000)),
            attributes: {
                let mut m = HashMap::new();
                m.insert("role".into(), AttributeValue::String("VP Sales".into()));
                m.insert("active".into(), AttributeValue::Bool(true));
                m
            },
            intent_summary: None,
        };

        let json = serde_json::to_string(&node).unwrap();
        let back: UniversalNode = serde_json::from_str(&json).unwrap();
        assert_eq!(back.id, "test123");
        assert_eq!(back.domain, DomainTag::Structured);
        assert_eq!(back.kind, UniversalKind::Entity);
        assert_eq!(
            back.attributes.get("role"),
            Some(&AttributeValue::String("VP Sales".into()))
        );
    }
}
