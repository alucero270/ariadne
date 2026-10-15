mod app;
mod http;
mod mcp;
mod upstream;

use anyhow::{bail, Context, Result};
use ariadne_core::eval::{evaluate, EvalSet};
use ariadne_core::{Index, Mode, Models, SearchOptions, Source};
use clap::{Parser, Subcommand};
use std::net::SocketAddr;
use std::path::PathBuf;
use upstream::{HttpEmbedder, HttpReranker};

const MIN_TOKEN_CHARS: usize = 16;
/// Well under the HTTP layer's 4 KiB cap on the Authorization header, so a valid token can
/// always be sent.
const MAX_TOKEN_CHARS: usize = 1024;

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
    /// Update the index, then measure Recall@k and MRR against an eval set
    /// ({"questions":[{"q","expect":[paths]}]}).
    Eval {
        file: PathBuf,
        #[arg(short, default_value_t = 5)]
        k: usize,
        /// Apply the configured reranker during evaluation.
        #[arg(long)]
        rerank: bool,
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

impl Cli {
    fn source(&self) -> Result<Source> {
        let root = self
            .root
            .clone()
            .context("--root (or ARIADNE_ROOT) is required")?;
        Ok(Source {
            max_chars: self.max_chars,
            ..Source::new(root)
        })
    }

    fn models(&self) -> Result<Models> {
        let embedder = match (&self.embed_url, &self.embed_model) {
            (Some(url), Some(model)) => {
                Some(Box::new(HttpEmbedder::new(url, model, &self.upstream_key)) as _)
            }
            (None, None) => None,
            _ => bail!("set both ARIADNE_EMBED_URL and ARIADNE_EMBED_MODEL, or neither"),
        };
        let reranker = match (&self.rerank_url, &self.rerank_model) {
            (Some(url), Some(model)) => {
                Some(Box::new(HttpReranker::new(url, model, &self.upstream_key)) as _)
            }
            (None, None) => None,
            _ => bail!("set both ARIADNE_RERANK_URL and ARIADNE_RERANK_MODEL, or neither"),
        };
        Ok(Models { embedder, reranker })
    }

    fn open_index(&self) -> Result<Index> {
        Index::open(&self.db, self.source()?, self.models()?)
            .with_context(|| format!("opening index {}", self.db.display()))
    }
}

fn check_serve_auth(bind: &SocketAddr, token: Option<&str>) -> Result<()> {
    match token {
        Some(t) if !(MIN_TOKEN_CHARS..=MAX_TOKEN_CHARS).contains(&t.chars().count()) => {
            bail!("ARIADNE_TOKEN must be {MIN_TOKEN_CHARS} to {MAX_TOKEN_CHARS} characters")
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
    match &cli.command {
        Command::Index => {
            let stats = cli.open_index()?.build()?;
            println!("{}", serde_json::to_string(&stats)?);
        }
        Command::Search { query, k, mode } => {
            let opts = SearchOptions {
                k: *k,
                mode: *mode,
                ..Default::default()
            };
            for h in cli.open_index()?.search(query, &opts)? {
                println!("{:.4}  [[{}]]  {:?}", h.score, h.reference, h.ranks);
            }
        }
        Command::Eval { file, k, rerank } => {
            let set: EvalSet = serde_json::from_str(&std::fs::read_to_string(file)?)
                .with_context(|| format!("reading eval set {}", file.display()))?;
            let mut index = cli.open_index()?;
            index.build()?;
            let report = evaluate(&mut index, &set, *k, *rerank)?;
            println!("{}", serde_json::to_string_pretty(&report)?);
        }
        Command::Serve { bind, token } => {
            check_serve_auth(bind, token.as_deref())?;
            let mut writer = cli.open_index()?;
            eprintln!("indexed: {}", serde_json::to_string(&writer.build()?)?);
            let app = app::App::new(cli.open_index()?, writer);
            let listener = tokio::net::TcpListener::bind(bind).await?;
            eprintln!("ariadne listening on http://{bind}");
            axum::serve(listener, http::router(app, token.clone()))
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
        assert!(check_serve_auth(&open, Some(&"x".repeat(MAX_TOKEN_CHARS + 1))).is_err());
    }
}
