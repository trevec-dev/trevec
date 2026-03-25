#![warn(clippy::all)]

use anyhow::Result;
use clap::{Parser, Subcommand};
use std::path::PathBuf;

#[derive(Parser)]
#[command(name = "trevec", version, about = "Context infrastructure for AI-assisted development")]
struct Cli {
    #[command(subcommand)]
    command: Commands,
}

#[derive(Subcommand)]
enum McpAction {
    /// Configure MCP integration for IDE clients (global, one-time)
    Setup {
        /// Target client: claude, claude-code, cursor, codex, or all (default: all)
        #[arg(long, default_value = "all")]
        client: String,

        /// Skip all confirmation prompts
        #[arg(long, short = 'y')]
        yes: bool,
    },

    /// Remove MCP configuration for IDE clients
    Remove {
        /// Target client: claude, claude-code, cursor, codex, or all (default: all)
        #[arg(long, default_value = "all")]
        client: String,

        /// Skip all confirmation prompts
        #[arg(long, short = 'y')]
        yes: bool,
    },

    /// Check MCP setup health
    Doctor {
        /// Path to the repository (default: current directory)
        #[arg(default_value = ".")]
        path: PathBuf,
    },
}

#[derive(Subcommand)]
enum Commands {
    /// Initialize trevec for a repository (creates .trevec/, indexes, writes rules)
    Init {
        /// Path to the repository (default: current directory)
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Skip indexing (just create .trevec/ config files)
        #[arg(long)]
        no_index: bool,

        /// Skip writing IDE rule files
        #[arg(long)]
        no_rules: bool,
    },

    /// Index a repository
    Index {
        /// Path to the repository (default: current directory)
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Where to store index data (default: .trevec)
        #[arg(long, default_value = ".trevec")]
        data_dir: PathBuf,

        /// Show per-phase timing
        #[arg(short, long)]
        verbose: bool,

        /// Use GPU for embeddings (requires CUDA + onnxruntime-gpu)
        #[arg(long)]
        gpu: bool,
    },

    /// Query the index for relevant context
    Ask {
        /// The search query (ignored in --batch mode)
        #[arg(default_value = "")]
        query: String,

        /// Path to the repository (default: current directory)
        #[arg(long, default_value = ".")]
        path: PathBuf,

        /// Where index data is stored (default: .trevec)
        #[arg(long, default_value = ".trevec")]
        data_dir: PathBuf,

        /// Token budget for context assembly
        #[arg(long)]
        budget: Option<usize>,

        /// Number of anchor nodes
        #[arg(long)]
        anchors: Option<usize>,

        /// Output as JSON
        #[arg(long)]
        json: bool,

        /// Show retrieval debug info
        #[arg(short, long)]
        verbose: bool,

        /// Batch mode: read queries from stdin (one per line), output JSON per line
        #[arg(long)]
        batch: bool,

        /// Use GPU for embeddings (requires CUDA + onnxruntime-gpu)
        #[arg(long)]
        gpu: bool,
    },

    /// Query the index (alias for ask)
    #[command(hide = true)]
    Query {
        /// The search query (ignored in --batch mode)
        #[arg(default_value = "")]
        query: String,

        /// Path to the repository (default: current directory)
        #[arg(long, default_value = ".")]
        path: PathBuf,

        /// Where index data is stored (default: .trevec)
        #[arg(long, default_value = ".trevec")]
        data_dir: PathBuf,

        /// Token budget for context assembly
        #[arg(long)]
        budget: Option<usize>,

        /// Number of anchor nodes
        #[arg(long)]
        anchors: Option<usize>,

        /// Output as JSON
        #[arg(long)]
        json: bool,

        /// Show retrieval debug info
        #[arg(short, long)]
        verbose: bool,

        /// Batch mode: read queries from stdin (one per line), output JSON per line
        #[arg(long)]
        batch: bool,

        /// Use GPU for embeddings (requires CUDA + onnxruntime-gpu)
        #[arg(long)]
        gpu: bool,
    },

    /// Inspect the index (debug)
    Inspect {
        /// Where index data is stored (default: .trevec)
        #[arg(long, default_value = ".trevec")]
        data_dir: PathBuf,

        /// Show node/edge/file counts
        #[arg(long)]
        stats: bool,

        /// Show node details by ID or name
        #[arg(long)]
        node: Option<String>,
    },

    /// Start the MCP server over stdio
    Serve {
        /// Path to the repository (default: current directory)
        #[arg(long, default_value = ".")]
        path: PathBuf,

        /// Where index data is stored (default: .trevec)
        #[arg(long, default_value = ".trevec")]
        data_dir: PathBuf,
    },

    /// Watch for file changes and re-index automatically
    Watch {
        /// Path to the repository (default: current directory)
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Where to store index data (default: .trevec)
        #[arg(long, default_value = ".trevec")]
        data_dir: PathBuf,

        /// Show per-phase timing
        #[arg(short, long)]
        verbose: bool,
    },

    /// MCP server management (setup, remove)
    Mcp {
        #[command(subcommand)]
        action: McpAction,
    },

    /// Manage tracked projects
    Projects {
        #[command(subcommand)]
        action: crate::commands::projects::ProjectsAction,
    },

    /// Show token savings and usage stats
    Stats {
        /// Path to the repository (default: current directory)
        #[arg(default_value = ".")]
        path: PathBuf,

        /// Show stats for all tracked projects
        #[arg(long)]
        all: bool,

        /// Output as JSON
        #[arg(long)]
        json: bool,
    },

    /// Set up MCP integration (use `trevec mcp setup` instead)
    #[command(hide = true)]
    Setup {
        /// Remove trevec MCP configuration instead of adding it
        #[arg(long)]
        remove: bool,
    },

    /// Update trevec to the latest version
    Update {
        /// Only check for updates, don't download
        #[arg(long)]
        check: bool,
    },

    /// Manage anonymous telemetry (enable/disable/status)
    Telemetry {
        #[command(subcommand)]
        action: crate::commands::telemetry_cmd::TelemetryAction,
    },

    /// Manage episodic memory (AI chat history)
    Memory {
        /// Path to the repository (default: current directory)
        #[arg(long, default_value = ".")]
        path: PathBuf,

        /// Where index data is stored (default: .trevec)
        #[arg(long, default_value = ".trevec")]
        data_dir: PathBuf,

        #[command(subcommand)]
        action: crate::commands::memory::MemoryAction,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    // Initialize logging to stderr
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::from_default_env()
                .add_directive("trevec=info".parse().unwrap()),
        )
        .with_writer(std::io::stderr)
        .init();

    let cli = Cli::parse();

    match cli.command {
        Commands::Init { path, no_index, no_rules } => {
            crate::commands::init::run(path, no_index, no_rules).await?;
        }
        Commands::Index {
            path,
            data_dir,
            verbose,
            gpu,
        } => {
            crate::commands::index::run(path, data_dir, verbose, gpu).await?;
        }
        Commands::Ask {
            query,
            path,
            data_dir,
            budget,
            anchors,
            json,
            verbose,
            batch,
            gpu,
        }
        | Commands::Query {
            query,
            path,
            data_dir,
            budget,
            anchors,
            json,
            verbose,
            batch,
            gpu,
        } => {
            if batch {
                crate::commands::query::run_batch(path, data_dir, budget, anchors, verbose, gpu)
                    .await?;
            } else {
                crate::commands::query::run(query, path, data_dir, budget, anchors, json, verbose, gpu)
                    .await?;
            }
        }
        Commands::Inspect {
            data_dir,
            stats,
            node,
        } => {
            crate::commands::inspect::run(data_dir, stats, node).await?;
        }
        Commands::Serve { path, data_dir } => {
            crate::commands::serve::run(path, data_dir).await?;
        }
        Commands::Watch {
            path,
            data_dir,
            verbose,
        } => {
            crate::commands::watch::run(path, data_dir, verbose).await?;
        }
        Commands::Mcp { action } => match action {
            McpAction::Setup {
                client,
                yes,
            } => {
                crate::commands::setup::run(&client, yes).await?;
            }
            McpAction::Remove {
                client,
                yes,
            } => {
                crate::commands::setup::run_remove(&client, yes).await?;
            }
            McpAction::Doctor { path } => {
                crate::commands::doctor::run(path).await?;
            }
        },
        Commands::Projects { action } => {
            crate::commands::projects::run(action).await?;
        }
        Commands::Stats { path, all, json } => {
            crate::commands::stats::run(path, all, json).await?;
        }
        Commands::Update { check } => {
            crate::commands::update::run(check).await?;
        }
        Commands::Setup { remove } => {
            if remove {
                crate::commands::setup::run_remove("all", false).await?;
            } else {
                crate::commands::setup::run("all", false).await?;
            }
        }
        Commands::Telemetry { action } => {
            crate::commands::telemetry_cmd::run(action).await?;
        }
        Commands::Memory {
            path,
            data_dir,
            action,
        } => {
            crate::commands::memory::run(action, path, data_dir).await?;
        }
    }

    Ok(())
}

mod commands;
mod telemetry;
