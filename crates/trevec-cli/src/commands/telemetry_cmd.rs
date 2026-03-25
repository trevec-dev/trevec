use anyhow::Result;
use clap::Subcommand;

use crate::telemetry;

#[derive(Subcommand)]
pub enum TelemetryAction {
    /// Disable anonymous telemetry
    Disable,
    /// Enable anonymous telemetry
    Enable,
    /// Show current telemetry status
    Status,
}

pub async fn run(action: TelemetryAction) -> Result<()> {
    match action {
        TelemetryAction::Disable => {
            let config = telemetry::TelemetryConfig {
                enabled: false,
                noticed: true,
            };
            telemetry::save_config(&config)?;
            eprintln!("Telemetry disabled.");
        }
        TelemetryAction::Enable => {
            let config = telemetry::TelemetryConfig {
                enabled: true,
                noticed: true,
            };
            telemetry::save_config(&config)?;
            eprintln!("Telemetry enabled.");
        }
        TelemetryAction::Status => {
            let disabled = telemetry::is_disabled();
            if disabled {
                eprintln!("Telemetry: disabled");
            } else {
                eprintln!("Telemetry: enabled");
            }

            // Show why it's disabled
            if std::env::var("TREVEC_TELEMETRY_DISABLED")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false)
            {
                eprintln!("  (via TREVEC_TELEMETRY_DISABLED env var)");
            }
            if std::env::var("DO_NOT_TRACK")
                .map(|v| v == "1" || v.eq_ignore_ascii_case("true"))
                .unwrap_or(false)
            {
                eprintln!("  (via DO_NOT_TRACK env var)");
            }
            if let Some(path) = telemetry::config_path() {
                eprintln!("  Config: {}", path.display());
            }
        }
    }
    Ok(())
}
