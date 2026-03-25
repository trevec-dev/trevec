//! Cloud LLM inference backend.
//!
//! Calls OpenRouter, Anthropic, or OpenAI APIs for Brain enrichment.

use anyhow::{bail, Result};
use serde::{Deserialize, Serialize};

/// Cloud LLM provider configuration.
#[derive(Debug, Clone)]
pub struct CloudConfig {
    pub provider: String,
    pub model: String,
    pub api_key: String,
    pub max_tokens: u32,
}

impl CloudConfig {
    /// Create a config from environment variables and Brain config.
    pub fn from_brain_config(brain_config: &trevec_core::config::BrainConfig) -> Result<Self> {
        let api_key = std::env::var(&brain_config.api_key_env).unwrap_or_default();
        if api_key.is_empty() {
            bail!(
                "Brain API key not set. Set {} environment variable.",
                brain_config.api_key_env
            );
        }
        Ok(Self {
            provider: brain_config.cloud_provider.clone(),
            model: brain_config.cloud_model.clone(),
            api_key,
            max_tokens: 300,
        })
    }
}

#[derive(Debug, Serialize)]
struct ChatMessage {
    role: String,
    content: String,
}

#[derive(Debug, Serialize)]
struct ChatRequest {
    model: String,
    messages: Vec<ChatMessage>,
    max_tokens: u32,
    temperature: f32,
}

#[derive(Debug, Deserialize)]
struct ChatResponse {
    choices: Vec<ChatChoice>,
}

#[derive(Debug, Deserialize)]
struct ChatChoice {
    message: ChatMessageResponse,
}

#[derive(Debug, Deserialize)]
struct ChatMessageResponse {
    content: String,
}

/// Call the cloud LLM API with a prompt.
pub async fn call_llm(config: &CloudConfig, system: &str, user: &str) -> Result<String> {
    let url = match config.provider.as_str() {
        "openrouter" => "https://openrouter.ai/api/v1/chat/completions",
        "anthropic" => "https://api.anthropic.com/v1/messages",
        "openai" => "https://api.openai.com/v1/chat/completions",
        _ => bail!("Unknown provider: {}", config.provider),
    };

    let client = reqwest::Client::new();

    // For Anthropic, use their specific format
    if config.provider == "anthropic" {
        return call_anthropic(config, system, user).await;
    }

    // OpenAI-compatible format (OpenRouter, OpenAI)
    let request = ChatRequest {
        model: config.model.clone(),
        messages: vec![
            ChatMessage {
                role: "system".to_string(),
                content: system.to_string(),
            },
            ChatMessage {
                role: "user".to_string(),
                content: user.to_string(),
            },
        ],
        max_tokens: config.max_tokens,
        temperature: 0.0,
    };

    let response = client
        .post(url)
        .header("Authorization", format!("Bearer {}", config.api_key))
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        bail!("LLM API error ({}): {}", status, body);
    }

    let resp: ChatResponse = response.json().await?;
    resp.choices
        .first()
        .map(|c| c.message.content.clone())
        .ok_or_else(|| anyhow::anyhow!("Empty response from LLM"))
}

/// Call Anthropic API (different format).
async fn call_anthropic(config: &CloudConfig, system: &str, user: &str) -> Result<String> {
    #[derive(Serialize)]
    struct AnthropicRequest {
        model: String,
        max_tokens: u32,
        system: String,
        messages: Vec<ChatMessage>,
    }

    #[derive(Deserialize)]
    struct AnthropicResponse {
        content: Vec<AnthropicContent>,
    }

    #[derive(Deserialize)]
    struct AnthropicContent {
        text: String,
    }

    let client = reqwest::Client::new();
    let request = AnthropicRequest {
        model: config.model.clone(),
        max_tokens: config.max_tokens,
        system: system.to_string(),
        messages: vec![ChatMessage {
            role: "user".to_string(),
            content: user.to_string(),
        }],
    };

    let response = client
        .post("https://api.anthropic.com/v1/messages")
        .header("x-api-key", &config.api_key)
        .header("anthropic-version", "2023-06-01")
        .header("Content-Type", "application/json")
        .json(&request)
        .send()
        .await?;

    if !response.status().is_success() {
        let status = response.status();
        let body = response.text().await.unwrap_or_default();
        bail!("Anthropic API error ({}): {}", status, body);
    }

    let resp: AnthropicResponse = response.json().await?;
    resp.content
        .first()
        .map(|c| c.text.clone())
        .ok_or_else(|| anyhow::anyhow!("Empty response from Anthropic"))
}

/// Estimate cost for a single LLM call.
pub fn estimate_call_cost(
    input_tokens: usize,
    output_tokens: usize,
    input_price_per_mtok: f64,
    output_price_per_mtok: f64,
) -> f64 {
    let input_cost = (input_tokens as f64 / 1_000_000.0) * input_price_per_mtok;
    let output_cost = (output_tokens as f64 / 1_000_000.0) * output_price_per_mtok;
    input_cost + output_cost
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_estimate_call_cost() {
        // Claude Haiku pricing: $0.25/MTok input, $1.25/MTok output
        let cost = estimate_call_cost(1000, 200, 0.25, 1.25);
        assert!(cost > 0.0);
        assert!(cost < 0.001); // Should be very cheap
    }

    #[test]
    fn test_cloud_config_missing_key() {
        let brain_config = trevec_core::config::BrainConfig {
            api_key_env: "NONEXISTENT_KEY_12345".to_string(),
            ..Default::default()
        };
        assert!(CloudConfig::from_brain_config(&brain_config).is_err());
    }
}
