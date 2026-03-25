use serde::Deserialize;
use std::path::Path;

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct TrevecConfig {
    pub index: IndexConfig,
    pub retrieval: RetrievalConfig,
    pub embeddings: EmbeddingsConfig,
    pub memory: MemoryConfig,
    pub brain: BrainConfig,
}

#[derive(Debug, Clone, Deserialize, Default)]
#[serde(default)]
pub struct IndexConfig {
    pub exclude: Vec<String>,
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct RetrievalConfig {
    pub anchors: usize,
    pub budget: usize,
    /// Penalty multiplier for test files: score *= (1.0 - penalty). 0.0 = disabled, 1.0 = full suppression.
    pub test_file_penalty: f64,
    /// Penalty multiplier for non-code files (CHANGELOG, README, .md, etc.).
    pub noncode_penalty: f64,
    /// Additional path patterns to penalize (substring match against file path).
    pub penalty_paths: Vec<String>,
}

impl Default for RetrievalConfig {
    fn default() -> Self {
        Self {
            anchors: 5,
            budget: 4096,
            test_file_penalty: 0.85,
            noncode_penalty: 0.8,
            penalty_paths: Vec::new(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct EmbeddingsConfig {
    pub model: String,
    /// Device for inference: "cpu" (default), "cuda", "directml"
    pub device: String,
}

impl Default for EmbeddingsConfig {
    fn default() -> Self {
        Self {
            model: "BAAI/bge-small-en-v1.5".to_string(),
            device: "cpu".to_string(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct MemoryConfig {
    /// Enable episodic memory (default: true)
    pub enabled: bool,
    /// Enabled source types (default: all)
    pub sources: Vec<String>,
    /// Days to retain events (default: 30)
    pub retention_days: u32,
    /// Days to retain raw turn events (default: 30)
    pub raw_retention_days: u32,
    /// Max disk usage in MB (default: 512)
    pub max_disk_mb: u32,
    /// Max total events (default: 200_000)
    pub max_events: u32,
    /// Max characters per event content (default: 8000)
    pub max_event_chars: usize,
    /// GC interval in minutes (default: 60)
    pub gc_interval_minutes: u32,
    /// Enable semantic embeddings for memory events (default: true)
    pub semantic: bool,
    /// Redaction mode: "strict" or "relaxed" (default: "strict")
    pub redaction_mode: String,
    pub cursor: CursorConfig,
    pub claude_code: ClaudeCodeConfig,
    pub codex: CodexConfig,
}

impl Default for MemoryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            sources: vec![
                "cursor".into(),
                "claude_code".into(),
                "codex".into(),
                "trevec_tool_calls".into(),
            ],
            retention_days: 30,
            raw_retention_days: 30,
            max_disk_mb: 512,
            max_events: 200_000,
            max_event_chars: 8000,
            gc_interval_minutes: 60,
            semantic: true,
            redaction_mode: "strict".to_string(),
            cursor: CursorConfig::default(),
            claude_code: ClaudeCodeConfig::default(),
            codex: CodexConfig::default(),
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct CursorConfig {
    pub enabled: bool,
    /// Custom path to state.vscdb (auto-detected if None)
    pub db_path: Option<String>,
}

impl Default for CursorConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            db_path: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct ClaudeCodeConfig {
    pub enabled: bool,
    /// Custom path to ~/.claude/projects/ (auto-detected if None)
    pub projects_dir: Option<String>,
}

impl Default for ClaudeCodeConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            projects_dir: None,
        }
    }
}

#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct CodexConfig {
    pub enabled: bool,
    /// Custom path to ~/.codex/sessions/ (auto-detected if None)
    pub sessions_dir: Option<String>,
}

impl Default for CodexConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            sessions_dir: None,
        }
    }
}

/// Configuration for the Brain async intelligence engine.
#[derive(Debug, Clone, Deserialize)]
#[serde(default)]
pub struct BrainConfig {
    /// Master switch: enable/disable all Brain workers.
    pub enabled: bool,
    /// Mode: "local" (run local LLM), "cloud" (use API), "hybrid" (prefer local, fall back to cloud).
    pub mode: String,
    /// Max CPU threads for Brain workers (default: 1).
    pub max_threads: usize,
    /// Seconds of idle before Brain activates (default: 30).
    pub idle_delay_secs: u64,
    /// Cloud LLM provider: "anthropic", "openai", "openrouter".
    pub cloud_provider: String,
    /// Cloud model name (e.g., "claude-haiku-4-5").
    pub cloud_model: String,
    /// Environment variable name for the API key.
    pub api_key_env: String,
    /// Daily spending cap for cloud LLM calls.
    pub max_cost_per_day: f64,
    /// Worker toggles.
    pub intent_summarizer: bool,
    pub entity_resolver: bool,
    pub link_predictor: bool,
    pub cross_domain_linker: bool,
    pub observation_agent: bool,
    /// Retention scoring config.
    pub retention_enabled: bool,
    /// Half-life in days for intent summaries (default: 30).
    pub intent_half_life_days: f64,
    /// Half-life in days for observations (default: 14).
    pub observation_half_life_days: f64,
    /// Score below which enrichments are pruned (default: 0.15).
    pub prune_threshold: f64,
}

impl Default for BrainConfig {
    fn default() -> Self {
        Self {
            enabled: false,
            mode: "cloud".to_string(),
            max_threads: 1,
            idle_delay_secs: 30,
            cloud_provider: "openrouter".to_string(),
            cloud_model: "qwen/qwen2.5-coder-1.5b-instruct".to_string(),
            api_key_env: "TREVEC_BRAIN_API_KEY".to_string(),
            max_cost_per_day: 1.0,
            intent_summarizer: true,
            entity_resolver: true,
            link_predictor: true,
            cross_domain_linker: false,
            observation_agent: true,
            retention_enabled: true,
            intent_half_life_days: 30.0,
            observation_half_life_days: 14.0,
            prune_threshold: 0.15,
        }
    }
}

impl TrevecConfig {
    /// Load config from `<data_dir>/config.toml`, falling back to defaults on
    /// missing or malformed file.
    pub fn load(data_dir: &Path) -> Self {
        let config_path = data_dir.join("config.toml");
        match std::fs::read_to_string(&config_path) {
            Ok(content) => match toml::from_str(&content) {
                Ok(config) => config,
                Err(e) => {
                    tracing::warn!("Failed to parse config.toml, using defaults: {e}");
                    Self::default()
                }
            },
            Err(_) => Self::default(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_defaults() {
        let config = TrevecConfig::default();
        assert!(config.index.exclude.is_empty());
        assert_eq!(config.retrieval.anchors, 5);
        assert_eq!(config.retrieval.budget, 4096);
        assert_eq!(config.embeddings.model, "BAAI/bge-small-en-v1.5");
    }

    #[test]
    fn test_parse_partial() {
        let toml_str = r#"
[retrieval]
budget = 8192
"#;
        let config: TrevecConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.retrieval.budget, 8192);
        assert_eq!(config.retrieval.anchors, 5); // default preserved
        assert!(config.index.exclude.is_empty());
    }

    #[test]
    fn test_parse_excludes() {
        let toml_str = r#"
[index]
exclude = ["vendor/**", "*.generated.*"]
"#;
        let config: TrevecConfig = toml::from_str(toml_str).unwrap();
        assert_eq!(config.index.exclude.len(), 2);
        assert_eq!(config.index.exclude[0], "vendor/**");
    }

    #[test]
    fn test_memory_defaults_extractors_enabled() {
        let config = TrevecConfig::default();
        assert!(config.memory.enabled);
        assert!(config.memory.cursor.enabled);
        assert!(config.memory.claude_code.enabled);
        assert!(config.memory.codex.enabled);
        assert_eq!(config.memory.sources.len(), 4);
        assert!(config.memory.sources.contains(&"cursor".to_string()));
        assert!(config.memory.sources.contains(&"claude_code".to_string()));
        assert!(config.memory.sources.contains(&"codex".to_string()));
        assert!(config.memory.sources.contains(&"trevec_tool_calls".to_string()));
    }

    #[test]
    fn test_load_missing_file() {
        let tmp = tempfile::tempdir().unwrap();
        let config = TrevecConfig::load(tmp.path());
        assert_eq!(config.retrieval.budget, 4096);
    }

    #[test]
    fn test_load_malformed_file() {
        let tmp = tempfile::tempdir().unwrap();
        std::fs::write(tmp.path().join("config.toml"), "not valid toml {{{").unwrap();
        let config = TrevecConfig::load(tmp.path());
        assert_eq!(config.retrieval.budget, 4096);
    }
}
