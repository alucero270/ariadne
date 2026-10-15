mod server;

use anyhow::{bail, Context, Result};
use ariadne_core::eval::{evaluate, EvalSet};
use ariadne_core::models::{Embedder, HttpEmbedder, HttpReranker, Reranker};
use ariadne_core::source::DEFAULT_EXCLUDES;
use ariadne_core::{Index, Mode, SearchOptions};
use clap::{Parser, Subcommand};
use std::net::SocketAddr;
use std::path::{Path, PathBuf};
use std::sync::{Arc, Mutex};

const MIN_TOKEN_CHARS: usize = 16;

#[derive(Parser)]
#[command(
    name = "ariadne",
    version,
    about = "Hybrid search over a folder of Markdown, served over HTTP and MCP"
)]
struct Cli {
    /// Folder to index. Read-only; nothing outside it is read and symlinks are skipped.
    #[arg(long, env = "ARIADNE_ROOT", global = true)]
    root: Option<PathBuf>,
    /// SQLite index file. Derived data: delete it and it rebuilds from the root.
    #[arg(long, env = "ARIADNE_DB", default_value = "ariadne.db", global = true)]
    db: PathBuf,
    /// OpenAI-compatible base URL for embeddings (e.g. llama-swap). Unset means keyword search only.
    #[arg(long, env = "ARIADNE_EMBED_URL", global = true)]
    embed_url: Option<String>,
    #[arg(long, env = "ARIADNE_EMBED_MODEL", global = true)]
    embed_model: Option<String>,
    /// Base URL of a llama.cpp-style /v1/rerank endpoint. Unset disables reranking.
    #[arg(long, env = "ARIADNE_RERANK_URL", global = true)]
    rerank_url: Option<String>,
    #[arg(long, env = "ARIADNE_RERANK_MODEL", global = true)]
    rerank_model: Option<String>,
    /// Bearer token sent to the embedding and rerank endpoints.
    #[arg(
        long,
        env = "ARIADNE_UPSTREAM_KEY",
        hide_env_values = true,
        global = true,
        default_value = ""
    )]
    upstream_key: String,
    /// Largest chunk, in characters, before a section is split at paragraph boundaries.
    #[arg(long, default_value_t = 1200, global = true)]
    max_chars: usize,
    #[command(subcommand)]
    command: Command,
}

#[derive(Subcommand)]
enum Command {
    /// Build or incrementally update the index from --root.
    Index,
    /// Search the index.
    Search {
        query: String,
        #[arg(short, default_value_t = 5)]
        k: usize,
        #[arg(long, default_value = "hybrid")]
        mode: Mode,
    },
    /// Measure Recall@k and MRR against an eval set ({"questions":[{"q","expect":[paths]}]}).
    Eval {
        file: PathBuf,
        #[arg(short, default_value_t = 5)]
        k: usize,
        /// Apply the configured reranker during evaluation.
        #[arg(long)]
        rerank: bool,
    },
    /// Serve the HTTP API and MCP endpoint. Indexes --root at startup.
    Serve {
        #[arg(long, env = "ARIADNE_BIND", default_value = "127.0.0.1:8090")]
        bind: SocketAddr,
        /// Bearer token clients must send. Required unless bound to loopback.
        #[arg(long, env = "ARIADNE_TOKEN", hide_env_values = true)]
        token: Option<String>,
    },
}

fn excludes() -> Vec<String> {
    DEFAULT_EXCLUDES.iter().map(|s| s.to_string()).collect()
}

fn open_index(cli: &Cli) -> Result<Index> {
    let embedder: Option<Box<dyn Embedder>> = match (&cli.embed_url, &cli.embed_model) {
        (Some(url), Some(model)) => {
            Some(Box::new(HttpEmbedder::new(url, model, &cli.upstream_key)))
        }
        (None, None) => None,
        _ => bail!("set both ARIADNE_EMBED_URL and ARIADNE_EMBED_MODEL, or neither"),
    };
    let reranker: Option<Box<dyn Reranker>> = match (&cli.rerank_url, &cli.rerank_model) {
        (Some(url), Some(model)) => {
            Some(Box::new(HttpReranker::new(url, model, &cli.upstream_key)))
        }
        (None, None) => None,
        _ => bail!("set both ARIADNE_RERANK_URL and ARIADNE_RERANK_MODEL, or neither"),
    };
    Index::open(&cli.db, embedder, reranker)
        .with_context(|| format!("opening index {}", cli.db.display()))
}

fn require_root(cli: &Cli) -> Result<&Path> {
    cli.root
        .as_deref()
        .context("--root (or ARIADNE_ROOT) is required for this command")
}

fn check_serve_auth(bind: &SocketAddr, token: Option<&str>) -> Result<()> {
    match token {
        Some(t) if t.chars().count() < MIN_TOKEN_CHARS => {
            bail!("ARIADNE_TOKEN must be at least {MIN_TOKEN_CHARS} characters")
        }
        None if !bind.ip().is_loopback() => {
            bail!("refusing to listen on {bind} without ARIADNE_TOKEN; set a token or bind to 127.0.0.1")
        }
        _ => Ok(()),
    }
}

#[tokio::main]
async fn main() -> Result<()> {
    let cli = Cli::parse();
    let mut index = open_index(&cli)?;
    match &cli.command {
        Command::Index => {
            let stats = index.build(require_root(&cli)?, &excludes(), cli.max_chars)?;
            println!("{}", serde_json::to_string(&stats)?);
        }
        Command::Search { query, k, mode } => {
            let opts = SearchOptions {
                k: *k,
                mode: *mode,
                ..Default::default()
            };
            for h in index.search(query, &opts)? {
                println!("{:.4}  [[{}]]  {:?}", h.score, h.reference, h.ranks);
            }
        }
        Command::Eval { file, k, rerank } => {
            if let Some(root) = &cli.root {
                index.build(root, &excludes(), cli.max_chars)?;
            }
            let set: EvalSet = serde_json::from_str(&std::fs::read_to_string(file)?)
                .with_context(|| format!("reading eval set {}", file.display()))?;
            let report = evaluate(&mut index, &set, *k, *rerank, cli.embed_url.is_some())?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::Serve { bind, token } => {
            check_serve_auth(bind, token.as_deref())?;
            if let Some(root) = &cli.root {
                let stats = index.build(root, &excludes(), cli.max_chars)?;
                eprintln!("indexed: {}", serde_json::to_string(&stats)?);
            }
            let state = server::AppState {
                index: Arc::new(Mutex::new(index)),
                root: cli.root.clone(),
                excludes: excludes(),
                max_chars: cli.max_chars,
                token: token.clone(),
            };
            let listener = tokio::net::TcpListener::bind(bind).await?;
            eprintln!("ariadne listening on http://{bind}");
            axum::serve(listener, server::router(state))
                .with_graceful_shutdown(async {
                    let _ = tokio::signal::ctrl_c().await;
                })
                .await?;
        }
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn serve_auth_rules() {
        let local: SocketAddr = "127.0.0.1:8090".parse().unwrap();
        let open: SocketAddr = "0.0.0.0:8090".parse().unwrap();
        assert!(check_serve_auth(&local, None).is_ok());
        assert!(check_serve_auth(&open, None).is_err());
        assert!(check_serve_auth(&open, Some("short")).is_err());
        assert!(check_serve_auth(&open, Some("0123456789abcdef")).is_ok());
    }
}
