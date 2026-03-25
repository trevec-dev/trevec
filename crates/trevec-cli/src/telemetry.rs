use serde::{Deserialize, Serialize};
use std::path::PathBuf;
use std::sync::OnceLock;

/// Write-only PostHog project API key. Override at build time with TREVEC_POSTHOG_KEY env var.
const POSTHOG_API_KEY: &str = match option_env!("TREVEC_POSTHOG_KEY") {
    Some(key) => key,
    None => "phc_sga8hpfz4zuCXU2a9DbvrNuOytg3PhvUtG03ocS2Ng7",
};
const POSTHOG_URL: &str = "https://us.i.posthog.com/capture/";

static CLIENT: OnceLock<PostHogClient> = OnceLock::new();

/// Telemetry opt-in/out state persisted at `~/.trevec/telemetry.json`.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TelemetryConfig {
    pub enabled: bool,
    #[serde(default)]
    pub noticed: bool,
}

impl Default for TelemetryConfig {
    fn default() -> Self {
        Self {
            enabled: true,
            noticed: false,
        }
    }
}

/// Path to the telemetry config file.
pub fn config_path() -> Option<PathBuf> {
    std::env::var("HOME")
        .ok()
        .map(|h| PathBuf::from(h).join(".trevec/telemetry.json"))
}

/// Load the telemetry config from disk. Returns `None` if the file doesn't exist.
pub fn load_config() -> Option<TelemetryConfig> {
    let path = config_path()?;
    let content = std::fs::read_to_string(path).ok()?;
    serde_json::from_str(&content).ok()
}

/// Save the telemetry config to disk.
pub fn save_config(config: &TelemetryConfig) -> anyhow::Result<()> {
    let path = config_path().ok_or_else(|| anyhow::anyhow!("Cannot determine home directory"))?;
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    let json = serde_json::to_string_pretty(config)?;
    std::fs::write(path, json)?;
    Ok(())
}

/// Check if telemetry is disabled via env vars or config file.
pub fn is_disabled() -> bool {
    // Environment variable overrides
    if std::env::var("TREVEC_TELEMETRY_DISABLED")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        return true;
    }
    if std::env::var("DO_NOT_TRACK")
        .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
        .unwrap_or(false)
    {
        return true;
    }

    // Config file
    if let Some(config) = load_config() {
        return !config.enabled;
    }

    false
}

/// Print first-run telemetry notice and create config file.
/// Call this on first `trevec init` or `trevec serve`.
pub fn maybe_show_first_run_notice() {
    if load_config().is_some() {
        return; // Already seen the notice
    }
    eprintln!();
    eprintln!("Trevec collects anonymous usage data to improve the product.");
    eprintln!("Run `trevec telemetry disable` to opt out. Learn more: https://trevec.dev/telemetry");
    eprintln!();

    let config = TelemetryConfig {
        enabled: true,
        noticed: true,
    };
    let _ = save_config(&config);
}

pub struct PostHogClient {
    http: reqwest::Client,
    device_id: String,
}

impl PostHogClient {
    fn new() -> Self {
        let http = reqwest::Client::builder()
            .timeout(std::time::Duration::from_secs(3))
            .build()
            .unwrap_or_default();

        let hostname = gethostname();
        let username = std::env::var("USER")
            .or_else(|_| std::env::var("USERNAME"))
            .unwrap_or_default();
        let device_id = blake3::hash(format!("{hostname}{username}").as_bytes())
            .to_hex()[..32]
            .to_string();

        Self { http, device_id }
    }
}

fn gethostname() -> String {
    std::process::Command::new("hostname")
        .output()
        .ok()
        .and_then(|o| String::from_utf8(o.stdout).ok())
        .map(|s| s.trim().to_string())
        .unwrap_or_default()
}

/// Get or initialize the global PostHog client.
fn client() -> &'static PostHogClient {
    CLIENT.get_or_init(PostHogClient::new)
}

/// Fire-and-forget telemetry event. Does nothing if telemetry is disabled.
/// Spawns a tokio task — never blocks the caller.
pub fn capture(event: &str, properties: serde_json::Value) {
    if is_disabled() {
        return;
    }

    let client = client();
    let debug = std::env::var("TREVEC_TELEMETRY_DEBUG").is_ok();

    let mut props = properties;
    if let Some(obj) = props.as_object_mut() {
        obj.insert("version".into(), env!("CARGO_PKG_VERSION").into());
        obj.insert("os".into(), std::env::consts::OS.into());
        obj.insert("arch".into(), std::env::consts::ARCH.into());
    }

    let payload = serde_json::json!({
        "api_key": POSTHOG_API_KEY,
        "event": event,
        "distinct_id": client.device_id,
        "properties": props,
    });

    if debug {
        eprintln!("[telemetry:debug] {}", serde_json::to_string_pretty(&payload).unwrap_or_default());
        return;
    }

    let http = client.http.clone();
    tokio::spawn(async move {
        let _ = http
            .post(POSTHOG_URL)
            .json(&payload)
            .send()
            .await;
    });
}
