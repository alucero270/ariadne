mod app;
mod config;
mod http;
mod mcp;
mod upstream;

use anyhow::{Context, Result};
use app::SearchRequest;
use ariadne_core::EvalSet;
use clap::{Parser, Subcommand};
use config::{check_serve_auth, Config, Settings};
use std::net::SocketAddr;
use std::path::PathBuf;

#[derive(Parser)]
#[command(
    name = "ariadne",
    version,
    about = "Hybrid search over a folder of Markdown, served over HTTP and MCP"
)]
struct Cli {
    #[command(flatten)]
    settings: Settings,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Build or incrementally update the index from --root.
    Index,
    /// Update the index, then search it with the same defaults the server uses.
    Search {
        query: String,
        #[arg(short)]
        k: Option<u64>,
        /// hybrid, bm25, or dense.
        #[arg(long)]
        mode: Option<String>,
    },
    /// Update the index, then measure Recall@k and MRR with the served defaults against an eval
    /// set ({"questions":[{"q","expect":[paths]}]}).
    Eval {
        file: PathBuf,
        #[arg(short)]
        k: Option<u64>,
    },
    /// Update the index, then serve the HTTP API and MCP endpoint.
    Serve {
        #[arg(long, env = "ARIADNE_BIND", default_value = "127.0.0.1:8090")]
        bind: SocketAddr,
        /// Bearer token clients must send. Required unless bound to loopback.
        #[arg(long, env = "ARIADNE_TOKEN", hide_env_values = true)]
        token: Option<String>,
    },
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let config = Config::new(cli.settings)?;
    match cli.command {
        Command::Index => {
            let stats = config.open_index()?.build()?;
            println!("{}", serde_json::to_string(&stats)?);
        }
        Command::Search { query, k, mode } => {
            let (app, _) = config.open_app()?;
            let request = SearchRequest {
                query,
                k,
                mode,
                rerank: None,
                expand: None,
            };
            for h in app.search(request).await? {
                println!("{:.4}  [[{}]]  {:?}", h.score, h.reference, h.ranks);
            }
        }
        Command::Eval { file, k } => {
            let set: EvalSet = serde_json::from_str(&std::fs::read_to_string(&file)?)
                .with_context(|| format!("reading eval set {}", file.display()))?;
            let (app, _) = config.open_app()?;
            let report = app.evaluate(set, k).await?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::Serve { bind, token } => {
            check_serve_auth(&bind, token.as_deref())?;
            let (app, stats) = config.open_app()?;
            eprintln!("indexed: {}", serde_json::to_string(&stats)?);
            let listener = tokio::net::TcpListener::bind(bind).await?;
            eprintln!("ariadne listening on http://{bind}");
            axum::serve(listener, http::router(app, token))
                .with_graceful_shutdown(async {
                    let _ = tokio::signal::ctrl_c().await;
                })
                .await?;
        }
    }
    Ok(())
}
